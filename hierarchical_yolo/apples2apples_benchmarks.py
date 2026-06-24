import os
import argparse
import json
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

try:
    from ultralytics.utils.ops import non_max_suppression
except ImportError:
    from ultralytics.utils.nms import non_max_suppression

from hierarchical_yolo.hierarchical_detection import HierarchicalYOLO, HierarchicalDetectionValidator
from hierarchical_yolo.yolo_utils import get_yolo_class_names
from hierarchical_loss.hierarchy_tensor_utils import conditional_to_marginal
from hierarchical_loss.hierarchy import Hierarchy

# ==========================================
# Helpers
# ==========================================

def build_hierarchy(hierarchy_json: str, master_yaml: str) -> Hierarchy:
    """
    Constructs the Hierarchy object in memory from the dataset configurations.

    Parameters
    ----------
    hierarchy_json : str
        Path to the global hierarchy.json file.
    master_yaml : str
        Path to the dataset YAML file containing the full master class list.

    Returns
    -------
    Hierarchy
        The instantiated Hierarchy tree mapping.
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
    A custom validator for standard Flat YOLO models that evaluates pure objectness.
    
    This intercepts the validation pipeline, dynamically forcing all ground truth 
    labels to Class 0, and collapsing all predicted class probabilities into a 
    single max-probability score.
    """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nc = 1
        self.names = {0: 'object'}

    def init_metrics(self, model):
        super().init_metrics(model)
        self.nc = 1
        self.names = {0: 'object'}
        if hasattr(self, 'metrics'):
            self.metrics.names = self.names

    def preprocess(self, batch):
        """Intercepts ground truth and forces all boxes to belong to Class 0."""
        batch = super().preprocess(batch)
        batch['cls'] = torch.zeros_like(batch['cls'])
        return batch

    def postprocess(self, preds):
        """Collapses multi-class predictions to a single max-probability class."""
        preds_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
        
        boxes = preds_tensor[:, :4, :]
        # p(Object) ≈ max(p(class_i)) for independence-assumption flat models
        scores, _ = preds_tensor[:, 4:, :].max(dim=1, keepdim=True)
        
        collapsed_preds = torch.cat([boxes, scores], dim=1)
        
        return non_max_suppression(
            collapsed_preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            multi_label=False,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det
        )

class HierarchicalObjectnessValidator(DetectionValidator):
    """
    A custom validator for Hierarchical YOLO models that evaluates pure objectness.
    
    Converts conditional predictions to marginals and isolates only the marginal 
    probability of the Root node(s), treating them as Class 0 for evaluation.
    """
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
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
        preds_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
        
        hierarchy = getattr(self.model, 'hierarchy', None)
        if hierarchy is None:
            raise ValueError("Hierarchy object not found on model.")
            
        # 1. Convert Conditionals to Marginals: [B, Detections, C]
        cls_probs = preds_tensor[:, 4:, :].transpose(1, 2)
        marginal_probs = conditional_to_marginal(cls_probs, hierarchy.index_tensor)
        
        # 2. Isolate the Root Node Score(s)
        marginal_probs_t = marginal_probs.transpose(1, 2)  # [B, C, Detections]
        root_indices = hierarchy.roots.to(marginal_probs_t.device)
        
        # Max across all roots (matches independence assumption amongst siblings)
        root_scores, _ = marginal_probs_t[:, root_indices, :].max(dim=1, keepdim=True)
        
        boxes = preds_tensor[:, :4, :]
        collapsed_preds = torch.cat([boxes, root_scores], dim=1)
        
        return non_max_suppression(
            collapsed_preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            multi_label=False,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det
        )

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

def run_objectness_benchmark(
    hierarchical_weights: str,
    flat_original_weights: str,
    flat_root_weights: str,
    hierarchical_yaml: str,
    flat_original_yaml: str,
    flat_root_yaml: str,
    hierarchy_json: str,
    split: str = 'val',
    imgsz: int = 640,
    batch: int = 16,
    device: str = ''
) -> tuple:
    """
    Executes Test A: The Objectness Benchmark.

    Evaluates how well each model architecture acts as a pure binary object 
    detector, independent of its taxonomic capabilities.
    """
    print("\n" + "="*50)
    print("🚀 RUNNING TEST A: OBJECTNESS BENCHMARK")
    print("="*50)

    # Initialize hierarchy securely in memory using the master dataset YAML
    hierarchy_obj = build_hierarchy(hierarchy_json, hierarchical_yaml)
    run_device = None if not device else device

    # 1. Flat-Root Model (Native Objectness)
    print("\n--- Evaluating: Flat-Root Model ---")
    model_root = YOLO(flat_root_weights)
    # Pure native run against its own snapped YAML
    res_root = model_root.val(data=flat_root_yaml, split=split, imgsz=imgsz, batch=batch, device=run_device, plots=False)

    # 2. Flat-Original Model (Collapsed Objectness)
    print("\n--- Evaluating: Flat-Original Model (Collapsed) ---")
    model_flat = FlatObjectnessYOLO(flat_original_weights)
    # Interceptor uses native YAML to satisfy loader, then collapses GTs to 0 in memory
    res_flat = model_flat.val(data=flat_original_yaml, split=split, imgsz=imgsz, batch=batch, device=run_device, plots=False)

    # 3. Hierarchical Model (Marginal Root Objectness)
    print("\n--- Evaluating: Hierarchical Model (Marginal Root) ---")
    model_hier = HierarchicalObjectnessYOLO(hierarchical_weights, hierarchy=hierarchy_obj)
    # Interceptor uses native YAML to satisfy loader, then collapses GTs to 0 in memory
    res_hier = model_hier.val(data=hierarchical_yaml, split=split, imgsz=imgsz, batch=batch, device=run_device, plots=False)

    print("\n" + "="*50)
    print("📊 OBJECTNESS RESULTS SUMMARY (mAP50-95)")
    print("="*50)
    print(f"Flat-Root Model       : {res_root.box.map:.4f}")
    print(f"Flat-Original Model   : {res_flat.box.map:.4f}")
    print(f"Hierarchical Model    : {res_hier.box.map:.4f}")
    
    return res_root, res_flat, res_hier


def run_specificity_benchmark(
    hierarchical_weights: str,
    flat_original_weights: str,
    hierarchical_yaml: str,
    flat_original_yaml: str,
    hierarchy_json: str,
    split: str = 'val',
    imgsz: int = 640,
    batch: int = 16,
    device: str = ''
) -> tuple:
    """
    Executes Test B: The Specificity Benchmark.

    Evaluates fine-grained discrimination on an identical hypothesis space. The 
    Hierarchical model's predictions are dynamically masked to only predict the 
    exact subset of classes present in the Flat-Original model.
    """
    print("\n" + "="*50)
    print("🚀 RUNNING TEST B: SPECIFICITY BENCHMARK")
    print("="*50)

    hierarchy_obj = build_hierarchy(hierarchy_json, hierarchical_yaml)
    run_device = None if not device else device

    # 1. Map Flat IDs to Master IDs
    with open(flat_original_yaml, 'r') as f:
        flat_names_map = get_yolo_class_names(f)
    subset_names = list(flat_names_map.values())

    eval_subset_ids = [hierarchy_obj.node_to_idx[name] for name in subset_names if name in hierarchy_obj.node_to_idx]
    print(f"Mapped {len(eval_subset_ids)} active baseline classes to Master Taxonomy IDs.")

    # 2. Evaluate Flat-Original
    print("\n--- Evaluating: Flat-Original Model ---")
    model_flat = YOLO(flat_original_weights)
    # Native evaluation on its own dataset
    res_flat = model_flat.val(data=flat_original_yaml, split=split, imgsz=imgsz, batch=batch, device=run_device, plots=False)

    # 3. Evaluate Hierarchical
    print("\n--- Evaluating: Hierarchical Model (Masked Specificity) ---")
    model_hier = HierarchicalYOLO(hierarchical_weights, hierarchy=hierarchy_obj)
    # The `eval_subset_ids` kwarg triggers prediction masking natively in HierarchicalYOLO.
    # Note: Evaluated on the full unsnapped GT dataset, enforcing strict comparisons.
    res_hier = model_hier.val(
        data=hierarchical_yaml,
        split=split,
        eval_subset_ids=eval_subset_ids, 
        imgsz=imgsz, 
        batch=batch, 
        device=run_device, 
        plots=False
    )

    print("\n" + "="*50)
    print("📊 SPECIFICITY RESULTS SUMMARY (mAP50-95)")
    print("="*50)
    print(f"Flat-Original Model   : {res_flat.box.map:.4f}")
    print(f"Hierarchical Model    : {res_hier.box.map:.4f}")
    
    return res_flat, res_hier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apples-to-Apples Evaluation Suite for Hierarchical Models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Objectness Subparser
    p_obj = subparsers.add_parser("objectness", help="Run Test A: Objectness Benchmark")
    p_obj.add_argument('--hierarchical_weights', required=True, type=str)
    p_obj.add_argument('--flat_original_weights', required=True, type=str)
    p_obj.add_argument('--flat_root_weights', required=True, type=str)
    p_obj.add_argument('--hierarchical_yaml', required=True, type=str, help="Dataset YAML for the hierarchical model")
    p_obj.add_argument('--flat_original_yaml', required=True, type=str, help="Dataset YAML for the flat original model")
    p_obj.add_argument('--flat_root_yaml', required=True, type=str, help="Dataset YAML for the flat root model")
    p_obj.add_argument('--hierarchy_json', required=True, type=str)
    p_obj.add_argument('--split', type=str, default='val', help="Dataset split to evaluate (e.g., 'val' or 'test')")
    p_obj.add_argument('--imgsz', type=int, default=640)
    p_obj.add_argument('--batch', type=int, default=16)
    p_obj.add_argument('--device', type=str, default='')

    # Specificity Subparser
    p_spec = subparsers.add_parser("specificity", help="Run Test B: Specificity Benchmark")
    p_spec.add_argument('--hierarchical_weights', required=True, type=str)
    p_spec.add_argument('--flat_original_weights', required=True, type=str)
    p_spec.add_argument('--hierarchical_yaml', required=True, type=str, help="Dataset YAML for the hierarchical model")
    p_spec.add_argument('--flat_original_yaml', required=True, type=str, help="Dataset YAML for the flat original model")
    p_spec.add_argument('--hierarchy_json', required=True, type=str)
    p_spec.add_argument('--split', type=str, default='val', help="Dataset split to evaluate (e.g., 'val' or 'test')")
    p_spec.add_argument('--imgsz', type=int, default=640)
    p_spec.add_argument('--batch', type=int, default=16)
    p_spec.add_argument('--device', type=str, default='')

    args = parser.parse_args()

    if args.command == "objectness":
        run_objectness_benchmark(
            hierarchical_weights=args.hierarchical_weights,
            flat_original_weights=args.flat_original_weights,
            flat_root_weights=args.flat_root_weights,
            hierarchical_yaml=args.hierarchical_yaml,
            flat_original_yaml=args.flat_original_yaml,
            flat_root_yaml=args.flat_root_yaml,
            hierarchy_json=args.hierarchy_json,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
    elif args.command == "specificity":
        run_specificity_benchmark(
            hierarchical_weights=args.hierarchical_weights,
            flat_original_weights=args.flat_original_weights,
            hierarchical_yaml=args.hierarchical_yaml,
            flat_original_yaml=args.flat_original_yaml,
            hierarchy_json=args.hierarchy_json,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
