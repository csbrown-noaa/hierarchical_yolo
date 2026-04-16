import os
import json
import argparse
import yaml
import glob
from PIL import Image
from ultralytics import YOLO
import torch

from hierarchical_loss.tree_utils import get_roots
from hierarchical_yolo.yolo_utils import (
    soft_hierarchical_predictions, 
    serialize_soft_hierarchical_predictions
)


def resolve_latest_weights(model_dir: str, project_name: str) -> str:
    """
    Discovers the most recently modified 'best.pt' weights for a given project.

    Parameters
    ----------
    model_dir : str
        The root directory where models and runs are saved.
    project_name : str
        The specific namespace or experiment run to evaluate.

    Returns
    -------
    str
        The absolute path to the most recently modified 'best.pt' file.

    Raises
    ------
    FileNotFoundError
        If no 'best.pt' files are found in the specified project directory.
    """
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

    Parameters
    ----------
    data_yaml_path : str
        Path to the dataset configuration file (e.g., train.yaml).
    split : str
        The dataset split to evaluate (e.g., 'val' or 'test').

    Returns
    -------
    str
        The absolute path to the directory containing the physical images.

    Raises
    ------
    ValueError
        If the specified split does not exist in the YAML file.
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


def build_or_load_coco_skeleton(coco_source: str | None, img_dir: str) -> list[dict]:
    """
    Constructs the base COCO images list by either inheriting metadata from an 
    existing JSON or dynamically scanning a directory for 'wild inference'.

    Parameters
    ----------
    coco_source : str | None
        Optional path to a base COCO JSON to inherit metadata from.
    img_dir : str
        Path to the physical image directory (used as fallback for wild inference).

    Returns
    -------
    list[dict]
        A list of image metadata dictionaries, guaranteed to have 'id' and 'file_name'.
    """
    images_list = []
    
    if coco_source and os.path.exists(coco_source):
        print(f"Inheriting image metadata from {coco_source}...")
        with open(coco_source, 'r') as f:
            coco_data = json.load(f)
            images_list = coco_data.get('images', [])
    else:
        print(f"No COCO source provided. Scanning directory {img_dir} for wild inference...")
        image_filenames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for file_name in image_filenames:
            images_list.append({
                'file_name': file_name
            })
            
    # Enforce IDs just in case the skeleton JSON was missing them
    for i, img_meta in enumerate(images_list):
        if 'id' not in img_meta:
            img_meta['id'] = i
            
    return images_list


def build_id_hierarchy(raw_tree: dict[str, str], name_to_id: dict[str, int]) -> dict[int, int]:
    """
    Converts a taxonomy tree mapping strings to a tree mapping integer IDs.

    Parameters
    ----------
    raw_tree : dict[str, str]
        A dictionary mapping child class names to parent class names.
    name_to_id : dict[str, int]
        A mapping of class names to their corresponding integer IDs in the model.

    Returns
    -------
    dict[int, int]
        A dictionary mapping child integer IDs to parent integer IDs.

    Examples
    --------
    >>> raw = {"dog": "animal", "cat": "animal"}
    >>> n2i = {"animal": 0, "dog": 1, "cat": 2, "car": 3}
    >>> build_id_hierarchy(raw, n2i)
    {1: 0, 2: 0}
    """
    hierarchy_id_map = {}
    for child_name, parent_name in raw_tree.items():
        if child_name in name_to_id and parent_name in name_to_id:
            hierarchy_id_map[name_to_id[child_name]] = name_to_id[parent_name]
    return hierarchy_id_map


def run_batched_inference(
    model: YOLO, 
    hierarchy_id_map: dict[int, int], 
    images_list: list[dict], 
    img_dir: str, 
    batch_size: int, 
    imgsz: int, 
    nms_iou_thres: float, 
    nms_conf_thres: float
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Executes hierarchical soft-prediction over a batched dataset, lazy-patching 
    image dimensions into the metadata list on the fly.

    Parameters
    ----------
    model : YOLO
        The loaded and device-mapped Ultralytics YOLO model.
    hierarchy_id_map : dict[int, int]
        The converted integer-based hierarchical taxonomy tree.
    images_list : list[dict]
        The active list of image metadata dictionaries. Mutated in-place to add dimensions.
    img_dir : str
        The path to the physical directory where images reside.
    batch_size : int
        The number of images to push through the GPU per batch.
    imgsz : int
        The squared input resolution for the model (e.g., 640).
    nms_iou_thres : float
        The Intersection over Union threshold for Non-Max Suppression.
    nms_conf_thres : float
        The minimal confidence threshold for Non-Max Suppression.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        A tuple containing lists of bounding box tensors and soft-score tensors.
    """
    all_boxes = []
    all_scores = []
    
    print(f"Starting batched inference (Batch size: {batch_size})...")
    for i in range(0, len(images_list), batch_size):
        batch_meta = images_list[i:i+batch_size]
        batch_images = []
        
        for img_meta in batch_meta:
            img_path = os.path.join(img_dir, img_meta['file_name'])
            img = Image.open(img_path).convert('RGB')
            batch_images.append(img)
            
            # Lazy-patch dimensions if missing from the skeleton JSON
            if 'width' not in img_meta or 'height' not in img_meta:
                img_meta['width'] = img.width
                img_meta['height'] = img.height
            
        boxes, scores = soft_hierarchical_predictions(
            model=model, 
            hierarchy=hierarchy_id_map, 
            images=batch_images, 
            shape=(imgsz, imgsz),
            nms_iou_thres=nms_iou_thres,
            nms_conf_thres=nms_conf_thres
        )
        
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        print(f"  -> Processed {min(i+batch_size, len(images_list))} / {len(images_list)}")
        
    return all_boxes, all_scores


def main():
    parser = argparse.ArgumentParser(description="Export hierarchical predictions to a viewer-compatible COCO JSON.")
    parser.add_argument('--workspace_dir', type=str, required=True, help="Path to the compiled hierarchical workspace.")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the root directory where models and runs were saved.")
    parser.add_argument('--project_name', type=str, required=True, help="The specific namespace/experiment run to evaluate.")
    parser.add_argument('--weights', type=str, default=None, help="Path to trained best.pt. Inferred automatically if omitted.")
    parser.add_argument('--split', type=str, default='val', help="Dataset split to evaluate (must match pycocowriter YAML key).")
    parser.add_argument('--coco_source', type=str, default=None, help="Optional path to a base COCO JSON to inherit 'images' metadata from.")
    parser.add_argument('--output', type=str, default='hierarchical_preds.json', help="Output JSON path")
    parser.add_argument('--nms_conf_thres', type=float, default=0.01, help="Very permissive NMS confidence threshold")
    parser.add_argument('--nms_iou_thres', type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument('--batch_size', type=int, default=32, help="Inference batch size")
    parser.add_argument('--imgsz', type=int, default=640, help="Inference image size")
    parser.add_argument('--device', type=str, default='', help="Device to use for inference (e.g., '0' or 'cpu').")
    args = parser.parse_args()
    
    # 1. Resolve Weights
    weights_path = args.weights if args.weights else resolve_latest_weights(args.model_dir, args.project_name)
        
    # 2. Resolve Paths internally via the workspace
    data_yaml = os.path.join(args.workspace_dir, "master_yolo", "train.yaml")
    hierarchy_json = os.path.join(args.workspace_dir, "hierarchy.json")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Missing leaf-node dataset YAML in {args.workspace_dir}/master_yolo/")
    if not os.path.exists(hierarchy_json):
        raise FileNotFoundError(f"Missing hierarchy.json at {hierarchy_json}")
        
    img_dir = resolve_image_directory(data_yaml, args.split)
        
    # 3. Build the Image Metadata List (Tiered Strategy)
    images_list = build_or_load_coco_skeleton(args.coco_source, img_dir)
    print(f"Discovered {len(images_list)} images to process.")
    
    # 4. Load Model
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    if args.device:
        primary_device = args.device.split(',')[0].strip()
        device_str = f"cuda:{primary_device}" if primary_device.isdigit() else primary_device
        print(f"Pushing model to device: {device_str}")
        model.to(device_str)
        
    idx_to_node = model.names
    name_to_id = {v: k for k, v in idx_to_node.items()}
    
    # 5. Load and Reconstruct Hierarchy
    print(f"Loading hierarchy from {hierarchy_json}...")
    with open(hierarchy_json, 'r') as f:
        raw_tree = json.load(f)
        
    hierarchy_id_map = build_id_hierarchy(raw_tree, name_to_id)
    roots = get_roots(hierarchy_id_map)
    
    # 6. Execute Inference
    all_boxes, all_scores = run_batched_inference(
        model=model, 
        hierarchy_id_map=hierarchy_id_map, 
        images_list=images_list, 
        img_dir=img_dir, 
        batch_size=args.batch_size, 
        imgsz=args.imgsz, 
        nms_iou_thres=args.nms_iou_thres, 
        nms_conf_thres=args.nms_conf_thres
    )
        
    # 7. Serialization
    print("Serializing predictions into viewer-compatible JSON...")
    out_json = serialize_soft_hierarchical_predictions(
        boxes_batch=all_boxes,
        scores_batch=all_scores,
        images_list=images_list,
        idx_to_node=idx_to_node,
        hierarchy_roots=roots,
        input_shape=(args.imgsz, args.imgsz)
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out_json, f)
        
    print(f"✅ Export complete! Saved to {args.output}")

if __name__ == "__main__":
    main()
