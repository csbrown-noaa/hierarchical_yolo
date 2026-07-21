import os
import json
import argparse
import yaml
import glob
import torch

from hierarchical_yolo.hierarchical_detection import HierarchicalYOLO, load_hierarchy_from_env
from yolo_kwcoco_serializer.yolo_kwcoco_serializer import Yolo2KwcocoSerializer

def resolve_latest_weights(model_dir: str, project_name: str) -> str:
    """
    Discovers the most recently modified 'best.pt' weights for a given project.
    """
    if not model_dir or not project_name:
        raise ValueError("Must provide both model_dir and project_name to infer weights.")
        
    project_path = os.path.join(model_dir, project_name)
    print(f"No weights provided. Searching for the most recent 'best.pt' in {project_path}...")
    
    search_pattern = os.path.join(project_path, "**", "weights", "best.pt")
    weight_files = glob.glob(search_pattern, recursive=True)
    
    if not weight_files:
        raise FileNotFoundError(f"Could not automatically find any 'best.pt' files in '{project_path}'.")
        
    latest_weights = max(weight_files, key=os.path.getmtime)
    print(f"-> Inferred latest weights: {latest_weights}\n")
    return latest_weights

def resolve_image_directory(data_yaml_path: str, split: str) -> str:
    """
    Parses a YOLO dataset YAML file to determine the absolute path to the physical 
    image directory for a specific split.
    """
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    if split not in data_cfg:
        raise ValueError(f"Split '{split}' not found in {data_yaml_path}")
        
    base_path = data_cfg.get('path', os.path.dirname(data_yaml_path))
    img_dir_rel = data_cfg[split]
    
    if isinstance(img_dir_rel, list):
        img_dir_rel = img_dir_rel[0]
        
    img_dir = os.path.join(base_path, img_dir_rel)
    return os.path.abspath(img_dir)

def main():
    parser = argparse.ArgumentParser(description="Export hierarchical predictions to a viewer-compatible COCO JSON.")
    
    # Workspace arguments (Required for standard dataset eval)
    parser.add_argument('--workspace_dir', type=str, default=None, help="Path to the compiled hierarchical workspace.")
    parser.add_argument('--model_dir', type=str, default=None, help="Path to the root directory where models and runs were saved.")
    parser.add_argument('--project_name', type=str, default=None, help="The specific namespace/experiment run to evaluate.")
    
    # Direct overrides (Required for wild video/image inference)
    parser.add_argument('--weights', type=str, default=None, help="Path to trained best.pt. Inferred automatically if omitted.")
    parser.add_argument('--source', type=str, default=None, help="Direct path to images or video. Overrides workspace split.")
    parser.add_argument('--hierarchy_json', type=str, default=None, help="Direct path to hierarchy.json. Overrides workspace.")
    
    parser.add_argument('--split', type=str, default='val', help="Dataset split to evaluate (must match pycocowriter YAML key).")
    parser.add_argument('--output', type=str, default='hierarchical_preds.json', help="Output JSON path")
    parser.add_argument('--nms_conf_thres', type=float, default=0.01, help="Very permissive NMS confidence threshold")
    parser.add_argument('--nms_iou_thres', type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument('--batch_size', type=int, default=32, help="Inference batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Inference image size")
    parser.add_argument('--device', type=str, default='', help="Device to use for inference (e.g., '0' or 'cpu').")
    args = parser.parse_args()
    
    # 1. Resolve Weights
    weights_path = args.weights if args.weights else resolve_latest_weights(args.model_dir, args.project_name)
        
    # 2. Resolve Hierarchy and Source Paths
    if args.workspace_dir:
        hierarchy_path = args.hierarchy_json or os.path.join(args.workspace_dir, "hierarchy.json")
        data_yaml = os.path.join(args.workspace_dir, "master_yolo", "train.yaml")
        source_path = args.source or resolve_image_directory(data_yaml, args.split)
    else:
        hierarchy_path = args.hierarchy_json
        source_path = args.source
        if not hierarchy_path or not source_path:
            raise ValueError("If --workspace_dir is omitted, you must provide --source and --hierarchy_json.")

    if not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f"Missing hierarchy.json at {hierarchy_path}")
        
    # 3. Load Model and Hierarchy
    print(f"Loading model from {weights_path}...")
    
    # Inject hierarchy path into the environment for safe loading
    os.environ['HIERARCHY_PATH'] = hierarchy_path
    
    # Load base model to extract class names
    temp_model = HierarchicalYOLO(weights_path)
    hierarchy_obj = load_hierarchy_from_env(temp_model.names)
    
    # Reload model correctly bound to the hierarchy object
    model = HierarchicalYOLO(weights_path, hierarchy=hierarchy_obj)
    
    run_device = args.device if args.device else None
    
    # 4. Execute Streaming Inference
    print(f"Starting batched inference on source: {source_path}...")
    
    # Initialize the stateful serializer with our class dictionary
    serializer = Yolo2KwcocoSerializer(categories=model.names)
    
    results_stream = model.predict(
        source=source_path,
        stream=True,
        conf=args.nms_conf_thres,
        iou=args.nms_iou_thres,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=run_device,
        save=False,  # Headless mode prevents UI crash from custom attributes
        plots=False,
        verbose=False
    )
    
    for i, res in enumerate(results_stream):
        # The serializer automatically handles routing, spatial mapping, 
        # and checking for our 'soft_scores' attribute injection.
        serializer.add_result(res)
        
        if (i + 1) % 100 == 0:
            print(f"  -> Processed {i + 1} frames/images...")
            
    # 5. Serialization
    print("Saving predictions to KWCOCO JSON...")
    serializer.save(args.output)

if __name__ == "__main__":
    main()
