import os
import argparse
import json
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from hierarchical_yolo.hierarchical_detection import HierarchicalYOLO, HierarchicalDetectionValidator
from hierarchical_yolo.yolo_utils import get_yolo_class_names
from hierarchical_loss.hierarchy import Hierarchy

# ==========================================
# Helpers
# ==========================================

def build_hierarchy(hierarchy_json: str, master_yaml: str) -> Hierarchy:
    """
    Constructs the Hierarchy object in memory from the dataset configurations.
    """
    with open(hierarchy_json, 'r') as f:
        raw_tree = json.load(f)
    
    with open(master_yaml, 'r') as f:
        master_names = get_yolo_class_names(f)
        
    name_to_id = {v: k for k, v in master_names.items()}
    return Hierarchy(raw_tree, name_to_id)

# ==========================================
# Custom Validators for Objectness (Test A)
# ==========================================

class FlatObjectnessValidator(DetectionValidator):
    """
    A custom validator for deep Flat YOLO models that evaluates pure objectness.
    
    This intercepts the validation pipeline, dynamically forcing all ground truth 
    labels to Class 0, and collapsing all predicted class probabilities into a 
    single max-probability score.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nc = 1
        self.names = {0: 'object'}

    def init_metrics(self, model):
        super().init_metrics(model)
        self.nc = 1
        self.names = {0: 'object'}
        if hasattr(self, 'metrics'):
            self.metrics.names = self.names

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch['cls'] = torch.zeros_like(batch['cls'])
        return batch

    def postprocess(self, preds):
        preds_tensor = preds[0]
        
        boxes = preds_tensor[:, :4, :]
        # p(Object) ≈ max(p(class_i)) for mutually exclusive flat models
        scores, _ = preds_tensor[:, 4:, :].max(dim=1, keepdim=True)
        
        collapsed_preds = torch.cat([boxes, scores], dim=1)
        
        # Delegate the actual NMS and output formatting back to the base validator
        return super().postprocess((collapsed_preds, *preds[1:]))

class HierarchicalObjectnessValidator(DetectionValidator):
    """
    A custom validator for Hierarchical YOLO models that evaluates pure objectness.
    
    Isolates the raw conditional probability of the Root node(s), treating them 
    as Class 0 for evaluation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nc = 1
        self.names = {0: 'object'}

    def init_metrics(self, model):
        super().init_metrics(model)
        self.nc = 1
        self.names = {0: 'object'}
        if hasattr(self, 'metrics'):
            self.metrics.names = self.names

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        batch['cls'] = torch.zeros_like(batch['cls'])
        return batch

    def postprocess(self, preds):
        preds_tensor = preds[0]
        
        hierarchy = getattr(self.model, 'hierarchy', None)
        if hierarchy is None:
            raise ValueError("Hierarchy object not found on model.")
            
        cls_probs = preds_tensor[:, 4:, :]  # [B, C, Detections]
        root_indices = hierarchy.roots.to(cls_probs.device)
        
        # Root conditional probabilities ARE their marginal probabilities 
        # (they have no ancestors to multiply with)
        root_scores, _ = cls_probs[:, root_indices, :].max(dim=1, keepdim=True)
        
        boxes = preds_tensor[:, :4, :]
        collapsed_preds = torch.cat([boxes, root_scores], dim=1)
        
        # Delegate the actual NMS and output formatting back to the base validator
        return super().postprocess((collapsed_preds, *preds[1:]))

# ==========================================
# YOLO Wrappers to Inject Validators
# ==========================================

class FlatObjectnessYOLO(YOLO):
    @property
    def task_map(self):
        base_map = super().task_map
        base_map["detect"]["validator"] = FlatObjectnessValidator
        return base_map

class HierarchicalObjectnessYOLO(HierarchicalYOLO):
    @property
    def task_map(self):
        base_map = super().task_map
        base_map["detect"]["validator"] = HierarchicalObjectnessValidator
        return base_map

# ==========================================
# Benchmark Orchestration Functions
# ==========================================

def run_objectness(
    weights: str,
    model_type: str,
    data_yaml: str,
    hierarchy_json: str = None,
    split: str = 'val',
    imgsz: int = 640,
    batch: int = 16,
    device: str = ''
):
    """
    Executes the Objectness Benchmark for a single model.
    Collapses deep class predictions to a singular root/objectness score.
    """
    print("\n" + "="*50)
    print(f"🚀 RUNNING OBJECTNESS EVALUATION ({model_type.upper()})")
    print("="*50)

    run_device = None if not device else device

    if model_type == 'flat':
        model = FlatObjectnessYOLO(weights)
        res = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, device=run_device, plots=False)
    
    elif model_type == 'hierarchical':
        if not hierarchy_json:
            raise ValueError("hierarchy_json must be provided for hierarchical evaluation.")
        hierarchy_obj = build_hierarchy(hierarchy_json, data_yaml)
        model = HierarchicalObjectnessYOLO(weights, hierarchy=hierarchy_obj)
        res = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, device=run_device, plots=False)
    
    else:
        raise ValueError("model_type must be either 'flat' or 'hierarchical'")

    print("\n" + "="*50)
    print(f"📊 OBJECTNESS RESULTS (mAP50-95): {res.box.map:.4f}")
    print("="*50)
    return res


def run_specificity(
    weights: str,
    hierarchical_eval_yaml: str,
    flat_data_yaml: str,
    hierarchy_json: str,
    split: str = 'val',
    imgsz: int = 640,
    batch: int = 16,
    device: str = ''
):
    """
    Executes the Specificity Benchmark for a Hierarchical model.
    Dynamically masks predictions to exactly match the target vocabulary 
    defined by a baseline flat dataset YAML.
    """
    print("\n" + "="*50)
    print("🚀 RUNNING SPECIFICITY EVALUATION (HIERARCHICAL MASKED)")
    print("="*50)

    hierarchy_obj = build_hierarchy(hierarchy_json, hierarchical_eval_yaml)
    run_device = None if not device else device

    # 1. Map Flat IDs to Master IDs
    with open(flat_data_yaml, 'r') as f:
        flat_names_map = get_yolo_class_names(f)
    subset_names = list(flat_names_map.values())

    eval_subset_ids = [hierarchy_obj.node_to_idx[name] for name in subset_names if name in hierarchy_obj.node_to_idx]
    print(f"Mapped {len(eval_subset_ids)} active baseline classes to Master Taxonomy IDs.")

    # 2. Evaluate Masked Hierarchical Model
    model = HierarchicalYOLO(weights, hierarchy=hierarchy_obj)
    res = model.val(
        data=hierarchical_eval_yaml,
        split=split,
        eval_subset_ids=eval_subset_ids, 
        imgsz=imgsz, 
        batch=batch, 
        device=run_device, 
        plots=False
    )

    print("\n" + "="*50)
    print(f"📊 SPECIFICITY RESULTS (mAP50-95): {res.box.map:.4f}")
    print("="*50)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apples-to-Apples Evaluation Toolbox")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Objectness Subparser
    p_obj = subparsers.add_parser("objectness", help="Evaluate a single model's pure objectness.")
    p_obj.add_argument('--weights', required=True, type=str, help="Path to model weights")
    p_obj.add_argument('--model_type', required=True, choices=['flat', 'hierarchical'], help="Type of model to squash")
    p_obj.add_argument('--data_yaml', required=True, type=str, help="Dataset YAML to evaluate against")
    p_obj.add_argument('--hierarchy_json', type=str, help="Master hierarchy.json (required for hierarchical models)")
    p_obj.add_argument('--split', type=str, default='val', help="Dataset split (e.g., 'val' or 'test')")
    p_obj.add_argument('--imgsz', type=int, default=640)
    p_obj.add_argument('--batch', type=int, default=16)
    p_obj.add_argument('--device', type=str, default='')

    # Specificity Subparser
    p_spec = subparsers.add_parser("specificity", help="Evaluate a hierarchical model masked to a flat baseline's vocabulary.")
    p_spec.add_argument('--weights', required=True, type=str, help="Path to hierarchical model weights")
    p_spec.add_argument('--hierarchical_eval_yaml', required=True, type=str, help="Curriculum YAML with snapped ground truths")
    p_spec.add_argument('--flat_data_yaml', required=True, type=str, help="Dataset YAML defining the target flat vocabulary")
    p_spec.add_argument('--hierarchy_json', required=True, type=str, help="Master hierarchy.json file")
    p_spec.add_argument('--split', type=str, default='val', help="Dataset split (e.g., 'val' or 'test')")
    p_spec.add_argument('--imgsz', type=int, default=640)
    p_spec.add_argument('--batch', type=int, default=16)
    p_spec.add_argument('--device', type=str, default='')

    args = parser.parse_args()

    if args.command == "objectness":
        run_objectness(
            weights=args.weights,
            model_type=args.model_type,
            data_yaml=args.data_yaml,
            hierarchy_json=args.hierarchy_json,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
    elif args.command == "specificity":
        run_specificity(
            weights=args.weights,
            hierarchical_eval_yaml=args.hierarchical_eval_yaml,
            flat_data_yaml=args.flat_data_yaml,
            hierarchy_json=args.hierarchy_json,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
