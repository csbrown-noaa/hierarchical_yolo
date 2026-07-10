import os
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

class AsymmetricalConfusionMatrix:
    """
    Computes an asymmetrical confusion matrix for misaligned datasets, 
    comparing predictions from V_model against ground truths from V_data.

    Attributes
    ----------
    data_names : dict
        Mapping of integer ID to string name for the dataset (Target).
    model_names : dict
        Mapping of integer ID to string name for the model (Prediction).
    matrix : np.ndarray
        The confusion matrix of shape (len(data_names) + 1, len(model_names) + 1).
    """

    def __init__(self, data_names, model_names, conf=0.25, iou_thres=0.45):
        self.data_names = data_names
        self.model_names = model_names
        self.num_targets = len(data_names)
        self.num_preds = len(model_names)
        self.matrix = np.zeros((self.num_targets + 1, self.num_preds + 1), dtype=np.int32)
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Process a batch of predictions and ground truths to update the matrix.
        """
        if detections is None or len(detections) == 0:
            if labels is not None and len(labels) > 0:
                for label in labels:
                    tgt_cls = int(label[0])
                    self.matrix[tgt_cls, self.num_preds] += 1
            return

        if labels is None or len(labels) == 0:
            for det in detections:
                if det[4] > self.conf:
                    pred_cls = int(det[5])
                    self.matrix[self.num_targets, pred_cls] += 1
            return

        # Filter detections by confidence
        detections = detections[detections[:, 4] > self.conf]
        if len(detections) == 0:
            for label in labels:
                tgt_cls = int(label[0])
                self.matrix[tgt_cls, self.num_preds] += 1
            return

        # Calculate IoU
        ious = box_iou(labels[:, 1:], detections[:, :4])
        
        # Greedy matching
        matches = []
        if ious.max() > self.iou_thres:
            ious_clone = ious.clone()
            while True:
                max_iou = ious_clone.max()
                if max_iou < self.iou_thres:
                    break
                tgt_idx, pred_idx = (ious_clone == max_iou).nonzero(as_tuple=False)[0]
                matches.append((tgt_idx.item(), pred_idx.item()))
                ious_clone[tgt_idx, :] = -1
                ious_clone[:, pred_idx] = -1

        matched_tgt_indices = {m[0] for m in matches}
        matched_pred_indices = {m[1] for m in matches}

        # 1. Matches
        for tgt_idx, pred_idx in matches:
            tgt_cls = int(labels[tgt_idx, 0])
            pred_cls = int(detections[pred_idx, 5])
            self.matrix[tgt_cls, pred_cls] += 1

        # 2. False Negatives
        for i in range(len(labels)):
            if i not in matched_tgt_indices:
                tgt_cls = int(labels[i, 0])
                self.matrix[tgt_cls, self.num_preds] += 1

        # 3. False Positives
        for i in range(len(detections)):
            if i not in matched_pred_indices:
                pred_cls = int(detections[i, 5])
                self.matrix[self.num_targets, pred_cls] += 1

    def save(self, save_dir):
        """Saves matrix to CSV and generates a plot."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        target_labels = [self.data_names.get(i, f"tgt_{i}") for i in range(self.num_targets)] + ["Background"]
        pred_labels = [self.model_names.get(i, f"pred_{i}") for i in range(self.num_preds)] + ["Background"]

        df = pd.DataFrame(self.matrix, index=target_labels, columns=pred_labels)
        df.to_csv(save_dir / "asymmetrical_confusion_matrix.csv")
        print(f"✅ Saved asymmetrical confusion matrix to: {save_dir}/asymmetrical_confusion_matrix.csv")

        # Plotting (Pure Matplotlib, mimicking Ultralytics native behavior)
        try:
            fig_size = max(10, min(40, max(self.num_targets, self.num_preds) // 3))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            
            # Use imshow instead of seaborn heatmap
            cax = ax.imshow(self.matrix, cmap='Blues', aspect='auto')
            fig.colorbar(cax)

            # Set axis labels and ticks
            ax.set_xticks(np.arange(len(pred_labels)))
            ax.set_yticks(np.arange(len(target_labels)))
            
            # Rotate the tick labels and set their alignment
            ax.set_xticklabels(pred_labels, rotation=90, ha="right", fontsize=8)
            ax.set_yticklabels(target_labels, fontsize=8)

            plt.ylabel('Ground Truth (Data)')
            plt.xlabel('Prediction (Model)')
            plt.title('Asymmetrical Confusion Matrix')
            
            plt.tight_layout()
            plt.savefig(save_dir / "asymmetrical_confusion_matrix.png", dpi=150)
            plt.close()
            print(f"✅ Saved heatmap to: {save_dir}/asymmetrical_confusion_matrix.png")
        except Exception as e:
            print(f"⚠️ Could not plot matrix: {e}")

class MisalignedValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asym_matrix = None
        self.data_to_model_map = {}
        self.num_shared_classes = 0
        self.num_data_classes = 0

    def init_metrics(self, model):
        super().init_metrics(model)
        model_names = model.names
        data_names = self.data.get('names', {})
        self.asym_matrix = AsymmetricalConfusionMatrix(data_names, model_names, conf=self.args.conf)
        
        model_name_to_id = {v: k for k, v in model_names.items()}
        for data_id, name in data_names.items():
            self.data_to_model_map[data_id] = model_name_to_id.get(name, -1)
            
        self.num_shared_classes = len([v for v in self.data_to_model_map.values() if v != -1])
        self.num_data_classes = len(data_names)

    def update_metrics(self, preds, batch):
        # 1. Custom Matrix Processing (uses exact padded tensor dimensions)
        _, _, h, w = batch['img'].shape
        
        for i, pred in enumerate(preds):
            mask = batch['batch_idx'] == i
            
            boxes = batch['bboxes'][mask]
            if len(boxes) > 0:
                boxes_xyxy = xywh2xyxy(boxes)
                boxes_xyxy[:, [0, 2]] *= w
                boxes_xyxy[:, [1, 3]] *= h
            else:
                boxes_xyxy = torch.empty((0, 4), device=boxes.device)
            
            cls_labels = batch['cls'][mask]
            formatted_labels = torch.cat([cls_labels.view(-1, 1), boxes_xyxy], dim=1)
            self.asym_matrix.process_batch(pred, formatted_labels)

        # 2. Native Engine Preprocessing: Remap classes and strip OOV distractors
        safe_batch = {k: v for k, v in batch.items()}
        safe_cls = []
        safe_bboxes = []
        safe_batch_idx = []
        
        for idx in range(len(batch['cls'])):
            orig_cls = int(batch['cls'][idx].item())
            mapped_cls = self.data_to_model_map.get(orig_cls, -1)
            
            if mapped_cls != -1:
                safe_cls.append(mapped_cls)
                safe_bboxes.append(batch['bboxes'][idx])
                safe_batch_idx.append(batch['batch_idx'][idx])

        if len(safe_cls) > 0:
            if batch['cls'].ndim == 2:
                safe_batch['cls'] = torch.tensor(safe_cls, device=batch['cls'].device).reshape(-1, 1).float()
            else:
                safe_batch['cls'] = torch.tensor(safe_cls, device=batch['cls'].device).float()
                
            safe_batch['bboxes'] = torch.stack(safe_bboxes)
            safe_batch['batch_idx'] = torch.tensor(safe_batch_idx, device=batch['batch_idx'].device).float()
        else:
            safe_batch['cls'] = torch.empty((0, 1) if batch['cls'].ndim == 2 else (0,), device=batch['cls'].device)
            safe_batch['bboxes'] = torch.empty((0, 4), device=batch['bboxes'].device)
            safe_batch['batch_idx'] = torch.empty((0,), device=batch['batch_idx'].device)

        super().update_metrics(preds, safe_batch)

    def finalize_metrics(self, *args, **kwargs):
        super().finalize_metrics(*args, **kwargs)
        
        penalty_factor = self.num_shared_classes / max(1, self.num_data_classes)
        native_map50_95 = self.metrics.box.map
        strict_map50_95 = native_map50_95 * penalty_factor
        
        print("\n" + "="*50)
        print("📊 MISALIGNED EVALUATION RESULTS")
        print("="*50)
        print(f"Native mAP50-95 (Shared Classes Only): {native_map50_95:.4f}")
        print(f"Strict mAP50-95 (Penalized for OOV):   {strict_map50_95:.4f}")
        print(f"Penalty Factor Applied:                {penalty_factor:.4f} ({self.num_shared_classes}/{self.num_data_classes} classes)")
        print("="*50 + "\n")

        # Guard against DDP parallel overwriting
        if self.asym_matrix and getattr(self.args, 'rank', -1) in (-1, 0):
            self.asym_matrix.save(self.save_dir)

class MisalignedYOLO(YOLO):
    @property
    def task_map(self):
        base_map = super().task_map
        base_map["detect"]["validator"] = MisalignedValidator
        return base_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Misaligned Datasets Validator")
    parser.add_argument('--weights', required=True, type=str, help="Path to model weights")
    parser.add_argument('--data_yaml', required=True, type=str, help="Target dataset YAML")
    parser.add_argument('--split', type=str, default='val', help="Dataset split")
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--project', type=str, default='runs/misaligned_eval')
    parser.add_argument('--name', type=str, default='distractor_test')
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    # Instantiate our custom YOLO wrapper
    model = MisalignedYOLO(args.weights)
    
    # Run validation
    model.val(
        data=args.data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        conf=args.conf,
        plots=True,
        save_json=True
    )
