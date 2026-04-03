import os
import json
import argparse
import yaml
import glob
from PIL import Image
from ultralytics import YOLO

from hierarchical_loss.tree_utils import get_roots
from hierarchical_yolo.yolo_utils import (
    soft_hierarchical_predictions, 
    serialize_soft_hierarchical_predictions
)

def main():
    parser = argparse.ArgumentParser(description="Export hierarchical predictions to a viewer-compatible COCO JSON.")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, 
        help="Path to the dataset directory containing train.yaml and hierarchy_data/"
    )
    parser.add_argument(
        '--model_dir', 
        type=str, 
        required=True,
        help="Path to the root directory where models and runs were saved."
    )
    parser.add_argument(
        '--project_name', 
        type=str, 
        required=True,
        help="The specific namespace/experiment run to evaluate."
    )
    parser.add_argument(
        '--weights', 
        type=str, 
        default=None, 
        help="Path to the trained best.pt model weights. If omitted, infers the most recently trained best.pt within the provided model_dir/project_name."
    )
    parser.add_argument(
        '--split', 
        type=str, 
        default='val', 
        help="Dataset split to evaluate (e.g., 'val' or 'test')"
    )
    parser.add_argument(
        '--url_prefix', 
        type=str, 
        required=True, 
        help="Base URL for the images (e.g., 'https://storage.googleapis.com/...')"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='hierarchical_preds.json', 
        help="Output JSON path"
    )
    
    # Coarse NMS settings for wide-net downstream filtering
    parser.add_argument('--nms_conf_thres', type=float, default=0.01, help="Very permissive NMS confidence threshold")
    parser.add_argument('--nms_iou_thres', type=float, default=0.7, help="NMS IoU threshold")
    
    # Hardware & Inference parameters
    parser.add_argument('--batch_size', type=int, default=32, help="Inference batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Inference image size")
    parser.add_argument('--device', type=str, default='', help="Device to use for inference (e.g., '0' or 'cpu').")
    
    args = parser.parse_args()
    
    # 0. Infer weights if not provided
    if not args.weights:
        project_path = os.path.join(args.model_dir, args.project_name)
        print(f"No weights provided. Searching for the most recent 'best.pt' in {project_path}...")
        search_pattern = os.path.join(project_path, "**", "weights", "best.pt")
        weight_files = glob.glob(search_pattern, recursive=True)
        if not weight_files:
            raise FileNotFoundError(f"Could not automatically find any 'best.pt' files in '{project_path}'. Please specify --weights manually.")
        args.weights = max(weight_files, key=os.path.getmtime)
        print(f"-> Inferred latest weights: {args.weights}\n")
        
    # 1. Resolve Config Paths based on data_dir
    data_yaml = os.path.join(args.data_dir, "train.yaml")
    hierarchy_json = os.path.join(args.data_dir, "hierarchy_data", "hierarchy.json")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Missing train.yaml in {args.data_dir}")
    if not os.path.exists(hierarchy_json):
        raise FileNotFoundError(f"Missing hierarchy.json at {hierarchy_json}")
    
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    if args.split not in data_cfg:
        raise ValueError(f"Split '{args.split}' not found in {data_yaml}")
        
    # Resolve image directory 
    base_path = data_cfg.get('path', args.data_dir)
    img_dir_rel = data_cfg[args.split]
    if isinstance(img_dir_rel, list):
        img_dir_rel = img_dir_rel[0]
        
    img_dir = os.path.join(base_path, img_dir_rel)
    if not os.path.isabs(img_dir):
        img_dir = os.path.abspath(img_dir)
        
    image_filenames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Discovered {len(image_filenames)} images in {img_dir}")
    
    # 2. Load Model & Reconstruct Hierarchy
    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)
    if args.device:
        # Handle comma-separated lists (e.g., "0,1,2") by grabbing the primary device
        primary_device = args.device.split(',')[0].strip()
        
        # Translate raw digits (e.g., '0') to strict PyTorch format ('cuda:0')
        device_str = f"cuda:{primary_device}" if primary_device.isdigit() else primary_device
        print(f"Pushing model to device: {device_str}")
        model.to(device_str)
        
    idx_to_node = model.names
    name_to_id = {v: k for k, v in idx_to_node.items()}
    
    print(f"Loading hierarchy from {hierarchy_json}...")
    with open(hierarchy_json, 'r') as f:
        raw_tree = json.load(f) # Expected {child_name: parent_name}
        
    # Convert name-based tree to ID-based tree for PyTorch NMS
    hierarchy_id_map = {}
    for child_name, parent_name in raw_tree.items():
        if child_name in name_to_id and parent_name in name_to_id:
            hierarchy_id_map[name_to_id[child_name]] = name_to_id[parent_name]
            
    roots = get_roots(hierarchy_id_map)
    
    # 3. Inference Loop
    all_boxes = []
    all_scores = []
    image_metadata = []
    
    print(f"Starting batched inference (Batch size: {args.batch_size})...")
    for i in range(0, len(image_filenames), args.batch_size):
        batch_files = image_filenames[i:i+args.batch_size]
        batch_images = []
        
        for file_name in batch_files:
            img_path = os.path.join(img_dir, file_name)
            img = Image.open(img_path).convert('RGB')
            batch_images.append(img)
            image_metadata.append({
                'id': len(image_metadata),
                'file_name': file_name,
                'width': img.width,
                'height': img.height
            })
            
        boxes, scores = soft_hierarchical_predictions(
            model=model, 
            hierarchy=hierarchy_id_map, 
            images=batch_images, 
            shape=(args.imgsz, args.imgsz),
            nms_iou_thres=args.nms_iou_thres,
            nms_conf_thres=args.nms_conf_thres
        )
        
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        print(f"  -> Processed {min(i+args.batch_size, len(image_filenames))} / {len(image_filenames)}")
        
    # 4. Serialization
    print("Serializing predictions into viewer-compatible JSON...")
    out_json = serialize_soft_hierarchical_predictions(
        boxes_batch=all_boxes,
        scores_batch=all_scores,
        image_metadata=image_metadata,
        idx_to_node=idx_to_node,
        hierarchy_roots=roots,
        url_prefix=args.url_prefix,
        input_shape=(args.imgsz, args.imgsz)
    )
    
    with open(args.output, 'w') as f:
        json.dump(out_json, f)
        
    print(f"✅ Export complete! Saved to {args.output}")

if __name__ == "__main__":
    main()
