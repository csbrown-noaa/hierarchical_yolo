from copy import copy
import ultralytics
import torch

'''
hierarchy = { parent: set([children]) }
'''

class v8HierarchicalDetectionLoss(ultralytics.utils.loss.v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk=10, hierarchy=None):  # model must be de-paralleled
        # hiearchy should be {child_id: parent_id}
        self.hierarchy = hierarchy
        self.bce2 = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.blah = 0
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        super().__init__(model, tal_topk=tal_topk)

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

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way        

        #TODO: fix this to be hierarchical loss
        ultralytics.utils.LOGGER.info('pred shape')
        ultralytics.utils.LOGGER.info(pred_scores.shape)
        ultralytics.utils.LOGGER.info('target shape')
        ultralytics.utils.LOGGER.info(target_scores.shape)
        ultralytics.utils.LOGGER.info('bboxes shape')
        ultralytics.utils.LOGGER.info(target_bboxes.shape)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        ultralytics.utils.LOGGER.info('class loss')
        ultralytics.utils.LOGGER.info(loss[1])
        ultralytics.utils.LOGGER.info(self.bce2(pred_scores, target_scores.to(dtype)) / target_scores_sum)
        ultralytics.utils.LOGGER.info('target onehots check')
        ultralytics.utils.LOGGER.info((torch.count_nonzero(target_scores, dim=1) - 1).abs().sum())
        ultralytics.utils.LOGGER.info((torch.count_nonzero(target_scores, dim=1) - 1).sum())
        if not self.blah:
            formatted_lines = []
            for i in range(target_scores.shape[1]):
                vec = target_scores[0, i]
                line = f"{i:04d}: " + ", ".join(f"{x:.4f}" for x in vec.tolist())
                formatted_lines.append(line)
            ultralytics.utils.LOGGER.info("\n".join(formatted_lines))
            self.blah += 1
        
        frontier = set(hierarchy.keys())
        while frontier:
            parent = frontier.pop()
            if parent in hierarchy:
                children = hierarchy[parent]
                for child in children:
                    pred_scores[0, :, child] *= pred_scores[0, :, parent]
                frontier |= hierarchy[parent]

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
