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

def predict(
    weights_path: str,
    hierarchy_path: str,
    source_path: str,
    output_path: str = 'hierarchical_preds.json',
    nms_conf_thres: float = 0.01,
    nms_iou_thres: float = 0.7,
    batch_size: int = 32,
    imgsz: int = 640,
    device: str = '',
    tracker: str = None,
    persist: bool = False
):
    """
    Core prediction function containing strictly defined keyword arguments.
    Executes streaming hierarchical inference (with optional tracking) 
    and serializes results to KWCOCO.
    """
    if not os.path.exists(hierarchy_path):
        raise FileNotFoundError(f"Missing hierarchy.json at {hierarchy_path}")

    # 1. Load Model and Hierarchy
    print(f"Loading model from {weights_path}...")
    
    # Inject hierarchy path into the environment for safe loading
    os.environ['HIERARCHY_PATH'] = hierarchy_path
    
    # Load base model to extract class names
    temp_model = HierarchicalYOLO(weights_path)
    hierarchy_obj = load_hierarchy_from_env(temp_model.names)
    
    # Reload model correctly bound to the hierarchy object
    model = HierarchicalYOLO(weights_path, hierarchy=hierarchy_obj)
    
    run_device = device if device else None
    
    # 2. Configure Inference Engine Arguments
    inference_args = dict(
        source=source_path,
        stream=True,
        conf=nms_conf_thres,
        iou=nms_iou_thres,
        imgsz=imgsz,
        batch=batch_size,
        device=run_device,
        save=False,  # Headless mode prevents UI crash from custom attributes
        plots=False,
        verbose=False
    )
    
    # Initialize the stateful serializer with our class dictionary
    serializer = Yolo2KwcocoSerializer(categories=model.names)
    
    # 3. Execute Streaming Inference (Route to track or predict)
    if tracker:
        print(f"Starting batched tracking (Tracker: {tracker}) on source: {source_path}...")
        inference_args['tracker'] = tracker
        inference_args['persist'] = persist
        results_stream = model.track(**inference_args)
    else:
        print(f"Starting batched inference on source: {source_path}...")
        results_stream = model.predict(**inference_args)
    
    for i, res in enumerate(results_stream):
        # The serializer automatically handles routing, spatial mapping, 
        # tracking IDs, and checking for our 'soft_scores' attribute injection.
        serializer.add_result(res)
        
        if (i + 1) % 100 == 0:
            print(f"  -> Processed {i + 1} frames/images...")
            
    # 4. Serialization
    print("Saving predictions to KWCOCO JSON...")
    serializer.save(output_path)

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
    
    # Inference arguments
    parser.add_argument('--split', type=str, default='val', help="Dataset split to evaluate (must match pycocowriter YAML key).")
    parser.add_argument('--output', type=str, default='hierarchical_preds.json', help="Output JSON path")
    parser.add_argument('--nms_conf_thres', type=float, default=0.01, help="Very permissive NMS confidence threshold")
    parser.add_argument('--nms_iou_thres', type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument('--batch_size', type=int, default=32, help="Inference batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Inference image size")
    parser.add_argument('--device', type=str, default='', help="Device to use for inference (e.g., '0' or 'cpu').")
    
    # Tracking arguments
    parser.add_argument('--tracker', type=str, default=None, help="Tracker config (e.g., 'botsort.yaml' or 'bytetrack.yaml'). Enables tracking.")
    parser.add_argument('--persist', action='store_true', help="Persist tracks between frames/streams (required for video tracking).")
    
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

    # 3. Delegate to the core prediction function
    predict(
        weights_path=weights_path,
        hierarchy_path=hierarchy_path,
        source_path=source_path,
        output_path=args.output,
        nms_conf_thres=args.nms_conf_thres,
        nms_iou_thres=args.nms_iou_thres,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        persist=args.persist
    )

if __name__ == "__main__":
    main()
