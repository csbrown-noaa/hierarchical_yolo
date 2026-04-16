from copy import copy
import ultralytics.utils.loss
import ultralytics.models
import ultralytics
import json
import torch
from ultralytics.utils import LOGGER

import ultralytics.utils.ops as ops
try:
    from ultralytics.utils.ops import non_max_suppression
except ImportError:
    from ultralytics.utils.nms import non_max_suppression

from ultralytics.engine.results import Results
from hierarchical_loss.hierarchy import Hierarchy
from hierarchical_loss.hierarchy_tensor_utils import (
    accumulate_hierarchy
)
from hierarchical_loss.hierarchical_loss import hierarchical_bce, hierarchical_conditional_bce, hierarchical_conditional_bce_soft_root, hierarchical_probabilistic_bce
from hierarchical_yolo.yolo_utils import conditionals_to_marginals
from hierarchical_loss.path_utils import (
    optimal_hierarchical_path,
    batch_truncate_paths_marginals,
    batch_filter_empty_paths
)

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
        import os
        # 1. Safely extract the path passed during DDP distribution via environment variable
        hierarchy_path = os.environ.get('HIERARCHY_PATH')
        hierarchy_obj = None
        
        if hierarchy_path:
            # 2. Load the JSON (happens independently and safely on each GPU)
            with open(hierarchy_path, 'r') as f:
                raw_hierarchy = json.load(f)
                
            # 3. Grab the YOLO class ID mapping directly from the Trainer's parsed YAML.
            # self.data['names'] looks like {0: 'fish', 1: 'shark'}
            # We invert it to {'fish': 0, 'shark': 1} exactly like your old code did!
            yolo_names = self.data['names']
            name_to_id = {v: k for k, v in yolo_names.items()}
            
            # 4. Initialize the Hierarchy object locally
            hierarchy_obj = Hierarchy(raw_hierarchy, name_to_id)
        else:
            print("WARNING: 'hierarchy_path' not provided in training arguments! Hierarchical loss may fail.")

        # 5. Pass the newly built hierarchy object to the model
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

    def postprocess(self, preds):
        """
        Intercepts the model predictions, calculates marginal probabilities 
        down the phylogenetic tree, optionally masks specific categories, 
        and passes them to the standard NMS.

        Parameters
        ----------
        preds : tuple | torch.Tensor
            The raw predictions from the YOLO `Detect` head.

        Returns
        -------
        tuple | torch.Tensor
            The modified prediction object where the conditional probabilities 
            have been replaced by the computed marginals.
        """
        # Unpack the payload (handling Ultralytics framework tuples/lists vs raw tensors)
        is_tuple = isinstance(preds, tuple)
        is_list = isinstance(preds, list)

        if is_tuple or is_list:
            inference_out = preds[0]
            remainder = preds[1:]
        else:
            inference_out = preds
            remainder = ()

        # 1. Look for custom attributes dynamically attached to the PyTorch model 
        # (for standalone eval), and fallback to self attributes (for training loop)
        subset_ids = getattr(self.model, 'eval_subset_ids', self.eval_subset_ids)
        active_hierarchy = getattr(self.model, 'hierarchy', self.hierarchy)

        # Delegate pure tensor math to our utility function
        inference_out = conditionals_to_marginals(
            inference_out, 
            active_hierarchy.index_tensor, 
            eval_subset_ids=subset_ids
        )
            
        # Repackage the payload exactly as we received it to maintain framework compatibility
        if is_tuple:
            preds = (inference_out, *remainder)
        elif is_list:
            preds = [inference_out] + list(remainder)
        else:
            preds = inference_out

        return super().postprocess(preds)


class HierarchicalDetectionPredictor(ultralytics.models.yolo.detect.DetectionPredictor):
    """
    Predictor class for YOLOv8 hierarchical object detection.
    
    Applies greedy path search and marginal truncation to output the deepest 
    confident node in the taxonomy tree for each detection.
    """
    def postprocess(self, preds, img, orig_imgs):
        """
        Intercepts raw model predictions, performs root-anchored NMS, computes 
        the optimal hierarchical path, and truncates at the confidence threshold.
        """
        hierarchy = getattr(self.model.model, 'hierarchy', None)
        if hierarchy is None:
            return super().postprocess(preds, img, orig_imgs)

        preds_tensor = preds[0] if isinstance(preds, tuple) else preds

        # 1. Raw Output & Root-Anchored NMS
        # return_idxs=True is critical to extract the full score vectors
        _, nms_idxs = non_max_suppression(
            preds_tensor,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=hierarchy.roots.tolist(),
            multi_label=True,
            return_idxs=True
        )

        # 2. Extract Full Score Vectors & Compute Optimal Paths
        bscores = []
        all_boxes = []
        for i, idx in enumerate(nms_idxs):
            flat_idx = idx.flatten().long()
            # nms_output shape: (4+C, N_filtered)
            nms_output = preds_tensor[i].index_select(1, flat_idx)
            
            boxes = nms_output[:4, :]  # (4, N_filtered)
            scores = nms_output[4:, :] # (C, N_filtered)
            
            all_boxes.append(boxes)
            bscores.append(scores)
        
        # Greedy Path Traversal
        raw_paths, raw_path_scores = optimal_hierarchical_path(
            bscores, 
            hierarchy.parent_child_tensor_tree, 
            hierarchy.roots
        )
        
        # 3. Truncate using Marginal Probabilities (Recycle NMS Conf)
        trunc_paths, trunc_scores = batch_truncate_paths_marginals(
            raw_paths, raw_path_scores, threshold=self.args.conf
        )
        
        # Filter out paths that became completely empty
        filtered_batch = batch_filter_empty_paths(
            all_boxes, trunc_paths, trunc_scores
        )

        # 4. Translate back to Ultralytics `Results` Format (Deepest Confident Node)
        results = []
        for i, (boxes, paths, scores) in enumerate(filtered_batch):
            boxes_t = boxes.T  # (N_filtered, 4)
            
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor) and boxes_t.numel() > 0:
                boxes_t[:, :4] = ops.scale_boxes(img.shape[2:], boxes_t[:, :4], orig_img.shape)

            # Standard YOLO format: (N, 6) -> [x1, y1, x2, y2, conf, cls]
            final_pred = torch.zeros((len(paths), 6), device=preds_tensor.device)
            if len(paths) > 0:
                final_pred[:, :4] = boxes_t
                # Conf: Marginal score of that deepest node
                final_pred[:, 4] = torch.tensor([s[-1].item() for s in scores], device=preds_tensor.device, dtype=torch.float)
                # Cls: The deepest node is the last element
                final_pred[:, 5] = torch.tensor([p[-1] for p in paths], device=preds_tensor.device, dtype=torch.float)

            img_path = self.batch[0][i] if hasattr(self, 'batch') and self.batch is not None else f"image_{i}"
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=final_pred))
            
        return results


class HierarchicalYOLO(ultralytics.YOLO):
    """
    A native YOLO wrapper that automatically routes to Hierarchical Trainers, 
    Validators, and Models without requiring framework monkey-patching.
    """
    
    @property
    def task_map(self):
        """Overrides the internal registry to use our custom hierarchical classes."""
        base_map = super().task_map
        base_map["detect"]["trainer"] = HierarchicalDetectionTrainer
        base_map["detect"]["validator"] = HierarchicalDetectionValidator
        base_map["detect"]["model"] = HierarchicalDetectionModel
        base_map["detect"]["predictor"] = HierarchicalDetectionPredictor
        return base_map
