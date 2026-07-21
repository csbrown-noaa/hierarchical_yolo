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
    snap_to_vocabulary,
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


def _hierarchical_spatial_filter(preds, hierarchy, args):
    """
    Phase 1: Converts conditional predictions to marginals and applies 
    root-anchored Non-Max Suppression.
    """
    preds_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds

    # 1. Convert conditionals to true marginal probabilities natively
    marginals_tensor = conditionals_to_marginals(
        preds_tensor, 
        hierarchy.index_tensor, 
        eval_subset_ids=None 
    )

    # 2. Root-Anchored NMS: Filter bounding boxes based on objectness (root probability)
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

    # 3. Extract final bounded boxes and their corresponding full soft-score vectors
    all_boxes = []
    bscores = []
    for i, idx in enumerate(nms_idxs):
        flat_idx = idx.flatten().long()
        nms_output = marginals_tensor[i].index_select(1, flat_idx)
        
        all_boxes.append(nms_output[:4, :])
        bscores.append(nms_output[4:, :])
        
    return all_boxes, bscores


def _hierarchical_taxonomic_resolve(all_boxes, bscores, hierarchy, args, eval_subset_ids=None):
    """
    Phase 2: Traverses the taxonomy graph, extracts optimal paths, 
    truncates by marginal confidence, and filters empty predictions.
    
    Returns parallel lists of tensors: (boxes, conf, cls, soft_scores)
    """
    # 4. Greedy Path Traversal: Compute optimal taxonomy paths based on soft scores
    raw_paths, raw_path_scores = optimal_hierarchical_path(
        bscores, 
        hierarchy.parent_child_tensor_tree, 
        hierarchy.roots
    )
    
    # 5. Marginal Truncation: Cut paths where confidence drops below the threshold
    trunc_results = batch_truncate_paths_marginals(
        raw_paths, raw_path_scores, threshold=args.conf
    )

    # 6. Snap-to-Vocabulary ("Casting Up"): Optionally force predictions to a specific tier
    if eval_subset_ids is not None:
        trunc_results = snap_to_vocabulary(trunc_results, eval_subset_ids)

    trunc_paths = [res[0] for res in trunc_results]
    trunc_scores = [res[1] for res in trunc_results]

    # 7. Empty Path Filtering (Box-Blind Index Resolution)
    filtered_results = batch_filter_empty_paths(trunc_paths, return_indices=True)

    # 8. Apply Math-Engine indices back onto Spatial Boxes and format as PyTorch Tensors
    resolved_batch = []
    for i, (f_paths, keep_idx) in enumerate(filtered_results):
        device = all_boxes[i].device
        
        if len(keep_idx) == 0:
            f_boxes = torch.zeros((0, 4), device=device)
            f_conf = torch.zeros((0,), device=device)
            f_cls = torch.zeros((0,), device=device)
            f_soft_scores = torch.zeros((0, bscores[i].shape[0]), device=device)
        else:
            f_boxes = all_boxes[i][:, keep_idx].T  # (N, 4)
            f_conf = torch.tensor([trunc_scores[i][k][-1].item() for k in keep_idx], device=device, dtype=torch.float)
            f_cls = torch.tensor([trunc_paths[i][k][-1] for k in keep_idx], device=device, dtype=torch.float)
            f_soft_scores = bscores[i][:, keep_idx].T  # (N, C)
            
        resolved_batch.append((f_boxes, f_conf, f_cls, f_soft_scores))

    return resolved_batch


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
  
        # assign on roots
        root_pred_scores = pred_scores[..., self.hierarchy.node_to_root].detach().sigmoid()

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
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
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a YOLO hierarchical detection model.
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
    """
    
    def __init__(self, *args, eval_subset_ids=None, hierarchy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_subset_ids = eval_subset_ids
        self.hierarchy = hierarchy

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

        preds_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds
        self.hierarchy = active_hierarchy.to(preds_tensor.device)

        all_boxes, bscores = _hierarchical_spatial_filter(preds, active_hierarchy, self.args)
        resolved_batch = _hierarchical_taxonomic_resolve(
            all_boxes, bscores, active_hierarchy, self.args, eval_subset_ids=subset_ids
        )
        
        repacked_results = []
        for boxes, conf, cls, _ in resolved_batch:
            repacked_results.append({
                "bboxes": boxes,
                "conf": conf,
                "cls": cls
            })
            
        return repacked_results


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
        if hierarchy is None:
            return super().postprocess(preds, img, orig_imgs)

        preds_tensor = preds[0] if isinstance(preds, (tuple, list)) else preds
        device = preds_tensor.device

        all_boxes, bscores = _hierarchical_spatial_filter(preds, hierarchy, self.args)
        resolved_batch = _hierarchical_taxonomic_resolve(all_boxes, bscores, hierarchy, self.args)

        results = []
        for i, (boxes, conf, cls, soft_scores) in enumerate(resolved_batch):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            
            if len(boxes) > 0:
                if not isinstance(orig_imgs, torch.Tensor):
                    boxes = ops.scale_boxes(img.shape[2:], boxes, orig_img.shape)
                final_pred = torch.cat([boxes, conf.unsqueeze(1), cls.unsqueeze(1)], dim=1)
            else:
                final_pred = torch.zeros((0, 6), device=device)

            img_path = self.batch[0][i] if hasattr(self, 'batch') and self.batch is not None else f"image_{i}"
            
            res = Results(orig_img, path=img_path, names=self.model.names, boxes=final_pred)
            res.hierarchical_soft_scores = soft_scores 
            results.append(res)
            
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
