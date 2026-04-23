import json
import os
from copy import copy

import torch
import ultralytics
import ultralytics.models
import ultralytics.utils.loss
import ultralytics.utils.ops as ops
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER

try:
    from ultralytics.utils.ops import non_max_suppression
except ImportError:
    from ultralytics.utils.nms import non_max_suppression

from hierarchical_loss.hierarchical_loss import (
    hierarchical_bce,
    hierarchical_conditional_bce,
    hierarchical_conditional_bce_soft_root,
    hierarchical_probabilistic_bce,
)
from hierarchical_loss.hierarchy import Hierarchy
from hierarchical_loss.hierarchy_tensor_utils import accumulate_hierarchy
from hierarchical_loss.path_utils import (
    batch_filter_empty_paths,
    batch_truncate_paths_marginals,
    optimal_hierarchical_path,
)
from hierarchical_yolo.yolo_utils import conditionals_to_marginals

def load_hierarchy_from_env(yolo_names: dict) -> Hierarchy:
    """
    Safely loads and constructs the Hierarchy object from the environment.
    """
    # Strict indexing allows Python to naturally throw a clean KeyError 
    # if the environment variable is missing, failing fast and obviously.
    hierarchy_path = os.environ['HIERARCHY_PATH']

    with open(hierarchy_path, 'r') as f:
        raw_tree = json.load(f)
        
    name_to_id = {v: k for k, v in yolo_names.items()}
    return Hierarchy(raw_tree, name_to_id)

def _hierarchical_postprocess(preds, hierarchy, args, eval_subset_ids=None):
    """
    Shared post-processing engine for both Validation and Prediction.
    Converts to marginals, runs Root-Anchored NMS, traverses the hierarchy,
    and cleanly snaps to the active vocabulary tier (if provided).
    """
    preds_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds

    # 1. Convert to true Marginals natively so NMS and greedy traversal use exact math
    marginals_tensor = conditionals_to_marginals(
        preds_tensor, 
        hierarchy.index_tensor, 
        eval_subset_ids=None # Do not mask yet; allow uninhibited traversal
    )

    # 2. Raw Output & Root-Anchored NMS
    _, nms_idxs = non_max_suppression(
        marginals_tensor,
        args.conf,
        args.iou,
        agnostic=args.agnostic_nms,
        max_det=args.max_det,
        classes=hierarchy.roots.tolist(),
        multi_label=True,
        return_idxs=True
    )

    # 3. Extract Full Score Vectors & Compute Optimal Paths
    bscores = []
    all_boxes = []
    for i, idx in enumerate(nms_idxs):
        flat_idx = idx.flatten().long()
        # nms_output shape: (4+C, N_filtered)
        nms_output = marginals_tensor[i].index_select(1, flat_idx)
        
        all_boxes.append(nms_output[:4, :])
        bscores.append(nms_output[4:, :])
    
    # Greedy Path Traversal
    raw_paths, raw_path_scores = optimal_hierarchical_path(
        bscores, 
        hierarchy.parent_child_tensor_tree, 
        hierarchy.roots
    )
    
    # 4. Truncate using Marginal Probabilities
    trunc_results = batch_truncate_paths_marginals(
        raw_paths, raw_path_scores, threshold=args.conf
    )
    # Safely transpose the list of tuples into two separated lists 
    # (Safe against 0-length batches, unlike zip(*...))
    trunc_paths = [res[0] for res in trunc_results]
    trunc_scores = [res[1] for res in trunc_results]

    # 5. Snap-to-Vocabulary ("Casting Up")
    if eval_subset_ids is not None:
        if isinstance(eval_subset_ids, torch.Tensor):
            eval_subset_ids = eval_subset_ids.tolist()
        eval_subset_set = set(eval_subset_ids)

        snapped_paths = []
        snapped_scores = []
        for p_list, s_list in zip(trunc_paths, trunc_scores):
            new_p_list = []
            new_s_list = []
            for path, score_path in zip(p_list, s_list):
                valid_idx = -1
                # Walk backward from the deepest node to find an allowed vocabulary hit
                for i in range(len(path) - 1, -1, -1):
                    if path[i] in eval_subset_set:
                        valid_idx = i
                        break
                if valid_idx != -1:
                    new_p_list.append(path[:valid_idx+1])
                    new_s_list.append(score_path[:valid_idx+1])
                else:
                    new_p_list.append([]) # Completely invalid path
                    new_s_list.append([])
            snapped_paths.append(new_p_list)
            snapped_scores.append(new_s_list)
        trunc_paths, trunc_scores = snapped_paths, snapped_scores
    
    # Filter out paths that became completely empty
    filtered_batch = batch_filter_empty_paths(
        all_boxes, trunc_paths, trunc_scores
    )

    # 6. Translate back to standard Ultralytics (N, 6) tensor arrays
    results_tensors = []
    for i, (boxes, paths, scores) in enumerate(filtered_batch):
        boxes_t = boxes.T  # (N_filtered, 4)
        
        final_pred = torch.zeros((len(paths), 6), device=preds_tensor.device)
        if len(paths) > 0:
            final_pred[:, :4] = boxes_t
            # Conf: True Marginal score of the deepest surviving node
            final_pred[:, 4] = torch.tensor([s[-1].item() for s in scores], device=preds_tensor.device, dtype=torch.float)
            # Cls: ID of the deepest surviving node
            final_pred[:, 5] = torch.tensor([p[-1] for p in paths], device=preds_tensor.device, dtype=torch.float)
        
        results_tensors.append(final_pred)
        
    return results_tensors

class v8HierarchicalDetectionLoss(ultralytics.utils.loss.v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 object detection."""
    
    def __init__(self, model, tal_topk=10, hierarchy=None):  # model must be de-paralleled
        super().__init__(model, tal_topk=tal_topk)
        device = next(model.parameters()).device  # get model device

        self.hierarchy = hierarchy.to(device)
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = ultralytics.utils.tal.make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        # assign on marginals vs
        '''
        logsigmoid_pred_scores = torch.nn.LogSigmoid()(pred_scores)
        hierarchical_pred_scores = accumulate_hierarchy(
            logsigmoid_pred_scores, 
            self.hierarchy.index_tensor, 
            reduce_op=torch.sum,
            identity_value=0.
        )
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            #torch.exp(hierarchical_pred_scores)
            root_pred_scores,
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        '''
  
        # assign on roots
        root_pred_scores = pred_scores[..., self.hierarchy.node_to_root].detach().sigmoid()

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            #torch.exp(hierarchical_pred_scores)
            root_pred_scores,
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        target_weights, target_indices = target_scores.max(dim=-1)

        hierarchical_class_loss = hierarchical_probabilistic_bce(
            pred_scores,
            target_weights,
            target_indices,
            self.hierarchy.ancestor_mask,
            self.hierarchy.ancestor_sibling_mask,
            self.hierarchy.root_mask
        )
        loss[1] = hierarchical_class_loss.sum() / target_scores_sum
         

        # 2. Compute Structural Loss (Normalized by Hierarchy Depth/Width)
        # Returns: (B, N_anchors)
        '''
        loss_per_anchor = hierarchical_conditional_bce(
            pred_scores,
            target_indices,
            self.hierarchy.ancestor_mask,
            self.hierarchy.ancestor_sibling_mask
        )
        loss[1] = (target_weights * loss_per_anchor).sum() / target_scores_sum
        '''
        '''
        # 3. Apply Target Weights (Quality Normalization)
        # We value high-IoU matches more than low-IoU matches
        weighted_loss = loss_per_anchor * target_weights

        # 4. Final Normalization
        # Divide by the sum of weights (the "effective" batch size)
        # Use max(..., 1) to prevent NaN on empty batches
        loss_norm = target_weights.sum().clamp(min=1.0)
        loss[1] = weighted_loss.sum() / loss_norm
        '''

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class HierarchicalDetectionTrainer(ultralytics.models.yolo.detect.DetectionTrainer):
    """
    Trainer class for YOLOv8 hierarchical object detection.

    This class overrides the default YOLO trainer to inject the phylogenetic 
    hierarchy into the model architecture and to utilize the custom hierarchical 
    loss and validation logic in a Multi-GPU (DDP) safe manner.
    """
    
    # We completely removed the __init__ and the _hierarchy class variable!
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLO hierarchical detection model.

        Parameters
        ----------
        cfg : str, optional
            Path to model configuration file. By default None.
        weights : str, optional
            Path to model weights. By default None.
        verbose : bool, optional
            Whether to display model information. By default True.

        Returns
        -------
        HierarchicalDetectionModel
            YOLO hierarchical detection model.
        """
        yolo_names = self.data['names']
        hierarchy_obj = load_hierarchy_from_env(yolo_names)

        # Pass the newly built hierarchy object to the model
        model = HierarchicalDetectionModel(
            cfg, 
            nc=self.data["nc"], 
            verbose=verbose and ultralytics.utils.RANK == -1, 
            hierarchy=hierarchy_obj
        )
        
        if weights:
            model.load(weights)
            
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        
        # Safely extract the hierarchy from the trainer's model 
        model = self.model.module if hasattr(self.model, 'module') else self.model
        hierarchy = getattr(model, 'hierarchy', None)
        
        return HierarchicalDetectionValidator(
            self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args), 
            _callbacks=self.callbacks,
            hierarchy=hierarchy
        )


class HierarchicalDetectionModel(ultralytics.nn.tasks.DetectionModel):
    """
    A YOLO detection model augmented with hierarchical taxonomic awareness.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the base `DetectionModel`.
    hierarchy : Hierarchy | None, optional
        The parsed `Hierarchy` object containing the taxonomic tree structure, 
        masks, and node relationships.
    **kwargs : dict
        Keyword arguments passed to the base `DetectionModel`.
    """
    
    def __init__(self, *args, hierarchy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchy = hierarchy
    
    def init_criterion(self):
        """Initialize the loss criterion for the HierarchicalDetectionModel."""
        return v8HierarchicalDetectionLoss(self, hierarchy=self.hierarchy)


class HierarchicalDetectionValidator(ultralytics.models.yolo.detect.DetectionValidator):
    """
    Validator class for YOLOv8 hierarchical object detection.

    This validator intercepts the model's conditional probability predictions 
    and converts them into marginal probabilities before running standard Non-Max 
    Suppression (NMS). This allows for accurate mAP calculation and optional 
    apples-to-apples comparisons with flat baseline models by restricting evaluation 
    to a specific subset of the taxonomy.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the base `DetectionValidator`.
    eval_subset_ids : list[int] | set[int] | torch.Tensor | None, optional
        Specific category IDs to evaluate against. If provided, probabilities for 
        classes outside this subset are zeroed out before NMS.
    **kwargs : dict
        Keyword arguments passed to the base `DetectionValidator`.
    """
    
    def __init__(self, *args, eval_subset_ids=None, hierarchy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_subset_ids = eval_subset_ids
        self.hierarchy = hierarchy

    def _repack(self, results_tensors):
        """
        Converts the (N, 6) tensor results from the hierarchical postprocessor
        into the dictionary format expected by the Ultralytics v8.3+ validation engine.
        """
        repacked_results = []
        for pred in results_tensors:
            if len(pred) == 0:
                repacked_results.append({
                    "bboxes": torch.zeros((0, 4), device=pred.device),
                    "conf": torch.zeros((0,), device=pred.device),
                    "cls": torch.zeros((0,), device=pred.device)
                })
            else:
                repacked_results.append({
                    "bboxes": pred[:, :4],
                    "conf": pred[:, 4],
                    "cls": pred[:, 5]
                })
        return repacked_results

    def postprocess(self, preds):
        """
        Intercepts the model predictions, routes them through the shared 
        hierarchical postprocessing engine, and optionally casts predictions 
        up to the allowed vocabulary tier.
        """
        # 1. Safely extract dynamic overrides (standalone eval context)
        if hasattr(self, 'model') and self.model is not None:
            subset_ids = getattr(self.model, 'eval_subset_ids', self.eval_subset_ids)
            active_hierarchy = getattr(self.model, 'hierarchy', self.hierarchy)
        else:
            # Training context: model is detached. Rely on organically constrained BCE logic.
            subset_ids = self.eval_subset_ids
            active_hierarchy = self.hierarchy

        # 2. Defensive fallback: If hierarchy is somehow lost in DDP context, rebuild it
        if active_hierarchy is None:
            yolo_names = getattr(self, 'data', {}).get('names', {})
            active_hierarchy = load_hierarchy_from_env(yolo_names)
            self.hierarchy = active_hierarchy  # Cache it locally

        results_tensors = _hierarchical_postprocess(
            preds, 
            active_hierarchy, 
            self.args, 
            eval_subset_ids=subset_ids
        )
        
        return self._repack(results_tensors)


class HierarchicalDetectionPredictor(ultralytics.models.yolo.detect.DetectionPredictor):
    """
    Predictor class for YOLOv8 hierarchical object detection.
    
    Applies greedy path search and marginal truncation to output the deepest 
    confident node in the taxonomy tree for each detection.
    """
    def postprocess(self, preds, img, orig_imgs):
        """
        Intercepts raw model predictions, runs the shared hierarchical postprocessor,
        and packages the results back into Ultralytics Results objects.
        """
        hierarchy = getattr(self.model.model, 'hierarchy', None)

        # 1. Run the unified processing engine (Predictor natively uses the full taxonomy)
        results_tensors = _hierarchical_postprocess(preds, hierarchy, self.args)

        # 2. Box Scaling & Translation back to Ultralytics `Results` UI Format
        results = []
        for i, final_pred in enumerate(results_tensors):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            
            if not isinstance(orig_imgs, torch.Tensor) and final_pred.shape[0] > 0:
                final_pred[:, :4] = ops.scale_boxes(img.shape[2:], final_pred[:, :4], orig_img.shape)

            img_path = self.batch[0][i] if hasattr(self, 'batch') and self.batch is not None else f"image_{i}"
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=final_pred))
            
        return results


class HierarchicalYOLO(ultralytics.YOLO):
    """
    A native YOLO wrapper that automatically routes to Hierarchical Trainers, 
    Validators, and Models without requiring framework monkey-patching.
    """
    
    def __init__(self, model="yolov8n.pt", task=None, hierarchy=None, **kwargs):
        super().__init__(model=model, task=task, **kwargs)
        if hierarchy is not None:
            self.model.hierarchy = hierarchy

    @property
    def task_map(self):
        """Overrides the internal registry to use our custom hierarchical classes."""
        base_map = super().task_map
        base_map["detect"]["trainer"] = HierarchicalDetectionTrainer
        base_map["detect"]["validator"] = HierarchicalDetectionValidator
        base_map["detect"]["model"] = HierarchicalDetectionModel
        base_map["detect"]["predictor"] = HierarchicalDetectionPredictor
        return base_map
        
    def val(self, **kwargs):
        """Intercepts validation call to pass dynamic attributes to the inner PyTorch model."""
        eval_subset_ids = kwargs.pop('eval_subset_ids', None)
        if eval_subset_ids is not None:
            self.model.eval_subset_ids = eval_subset_ids
        
        return super().val(**kwargs)
