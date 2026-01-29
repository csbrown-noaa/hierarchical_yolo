from copy import copy
import ultralytics.utils.loss
import ultralytics.models
import torch
from hierarchical_loss.hierarchy import Hierarchy
from hierarchical_loss.hierarchy_tensor_utils import (
    accumulate_hierarchy
)
from hierarchical_loss.hierarchical_loss import hierarchical_bce, hierarchical_conditional_bce

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

        # TODO! When figuring out the target scores here, do we need to do anything to the pred_scores??
        logsigmoid_pred_scores = torch.nn.LogSigmoid()(pred_scores)
        hierarchical_pred_scores = accumulate_hierarchy(
            logsigmoid_pred_scores, 
            self.hierarchy.index_tensor, 
            reduce_op=torch.sum,
            identity_value=0.
        )
        #####

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            torch.exp(hierarchical_pred_scores),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        '''
        unnormalized_loss = hierarchical_bce(
            pred_scores, 
            target_scores, 
            self.hierarchy.index_tensor
        )
        '''
        
        # 1. Prepare Inputs
        # Get the class ID (index) and the Quality Score (weight) for each anchor
        # target_scores: (B, N_anchors, N_classes)
        target_weights, target_indices = target_scores.max(dim=-1)

        # 2. Compute Structural Loss (Normalized by Hierarchy Depth/Width)
        # Returns: (B, N_anchors)
        loss_per_anchor = hierarchical_conditional_bce(
            pred_scores,
            target_indices,
            self.hierarchy.ancestor_mask,
            self.hierarchy.ancestor_sibling_mask
        )
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
        loss[1] = (target_weights * loss_per_anchor).sum() / target_scores_sum

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

    '''
        hierarchy = { child: parent }
    '''
    _hierarchy = None #unfortunately, we have to define the hierarchy here as a class variable
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchy = self.__class__._hierarchy
        
    def get_model(self, cfg=None, weights=None, verbose=True):
        #TODO: override this
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (HierarchicalDetectionModel): YOLO hierarchical detection model.
        """
        model = HierarchicalDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and ultralytics.utils.RANK == -1, hierarchy=self.hierarchy)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        #TODO override this
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return HierarchicalDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

class HierarchicalDetectionModel(ultralytics.nn.tasks.DetectionModel):
    
    def __init__(self, *args, hierarchy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hierarchy = hierarchy
    
    def init_criterion(self):
        """Initialize the loss criterion for the HierarchicalDetectionModel."""
        return v8HierarchicalDetectionLoss(self, hierarchy=self.hierarchy)

class HierarchicalDetectionValidator(ultralytics.models.yolo.detect.DetectionValidator):
    #TODO: this needs to basically be completely overhauled.
    pass
