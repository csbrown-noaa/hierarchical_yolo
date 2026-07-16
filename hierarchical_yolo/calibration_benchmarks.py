import os
import json
import argparse
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Pycocotools Imports
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# YOLO Imports (Only required for the 'pipeline' command)
from ultralytics import YOLO
from hierarchical_yolo.hierarchical_detection import HierarchicalYOLO, load_hierarchy_from_env
from hierarchical_yolo.yolo_utils import get_yolo_class_names

# ==========================================
# ECOSYSTEM-INDEPENDENT CALIBRATION MATH
# ==========================================

def populate_coco_categories(preds_list: list, yolo_idx_to_name: dict) -> list:
    """
    Injects the string class name into each headless YOLO prediction 
    using the mapping from the YOLO dataset YAML.
    """
    for pred in preds_list:
        yolo_id = int(pred['category_id'])
        if yolo_id in yolo_idx_to_name:
            pred['category_name'] = yolo_idx_to_name[yolo_id]
        else:
            pred['category_name'] = "UNKNOWN"
    return preds_list

def align_coco_jsons(pred_data: list, gt_data: dict) -> tuple[list, dict]:
    """
    In-memory alignment tool. Resolves image ID discrepancies and 
    maps prediction categories to official Ground Truth IDs by name.
    """
    # 1. Map Prediction Categories to GT Categories by Name
    gt_name_to_id = {cat['name']: cat['id'] for cat in gt_data.get('categories', [])}
    
    patched_cat_count = 0
    unmatched_cats = set()

    for pred in pred_data:
        name = pred.get('category_name')
        if name in gt_name_to_id:
            new_id = gt_name_to_id[name]
            if pred['category_id'] != new_id:
                pred['category_id'] = new_id
                patched_cat_count += 1
        else:
            unmatched_cats.add(name)
            
    # 2. Align Image IDs
    id_map = {}
    for img in gt_data.get('images', []):
        official_id = img['id']
        file_stem = Path(img['file_name']).stem
        
        id_map[file_stem] = official_id
        id_map[img['file_name']] = official_id
        id_map[str(official_id)] = official_id # Fallback if already an int string

    patched_img_count = 0
    unmatched_imgs = set()
    
    for pred in pred_data:
        raw_img_id = str(pred['image_id'])
        if raw_img_id in id_map:
            new_img_id = id_map[raw_img_id]
            if pred['image_id'] != new_img_id:
                pred['image_id'] = new_img_id
                patched_img_count += 1
        else:
            unmatched_imgs.add(raw_img_id)
            
    # 3. Console Logging
    print("\n" + "-"*50)
    print("🔄 IN-MEMORY JSON ALIGNMENT (BY NAME)")
    print("-"*50)
    print(f"✅ Mapped {patched_cat_count} prediction categories to official GT IDs via string matching.")
    if unmatched_cats:
        print(f"❌ WARNING: Found {len(unmatched_cats)} predicted category names missing from Ground Truth: {unmatched_cats}")

    if patched_img_count > 0:
        print(f"⚠️ Mapped {patched_img_count} prediction 'image_id's to official GT integers.")
    else:
        print(f"✅ All prediction 'image_id's were already perfectly aligned with GT.")
        
    if unmatched_imgs:
        print(f"❌ WARNING: Found {len(unmatched_imgs)} predicted image IDs that did not match any Ground Truth image!")
    print("-"*50)
        
    return pred_data, gt_data

def _verify_alignment(pred_data: list, gt_data: dict):
    """
    Standalone diagnostic function to verify that the aligned predictions 
    perfectly map to the ground truth structures. Fails hard if mismatches exist.
    """
    print("🔍 Verifying alignment integrity...")
    gt_image_ids = {img['id'] for img in gt_data.get('images', [])}
    gt_category_ids = {cat['id'] for cat in gt_data.get('categories', [])}
    
    invalid_images = set()
    invalid_categories = set()
    
    for pred in pred_data:
        if pred['image_id'] not in gt_image_ids:
            invalid_images.add(pred['image_id'])
        if pred['category_id'] not in gt_category_ids:
            invalid_categories.add(pred['category_id'])
            
    if invalid_images or invalid_categories:
        error_msg = "\n❌ Alignment Verification Failed!\n"
        if invalid_images:
            error_msg += f"  - Found {len(invalid_images)} predicted image_ids not in Ground Truth.\n"
            error_msg += f"    Examples: {list(invalid_images)[:5]}\n"
        if invalid_categories:
            error_msg += f"  - Found {len(invalid_categories)} predicted category_ids not in Ground Truth.\n"
            error_msg += f"    Examples: {list(invalid_categories)[:5]}\n"
        raise AssertionError(error_msg)
        
    print("✅ Alignment verification passed! All prediction IDs exist in Ground Truth.")

def evaluate_coco_calibration(pred_json_path: str, gt_json_path: str, iou_threshold: float = 0.5) -> list:
    """
    End-to-end pipeline to load COCO datasets and perform spatial bipartite 
    matching using the official pycocotools API. Extracts the raw matching 
    status for Expected Calibration Error (ECE) calculation.

    Assumes pred_json_path and gt_json_path are already perfectly aligned.

    Parameters
    ----------
    pred_json_path : str
        The absolute or relative path to the aligned predictions JSON file.
    gt_json_path : str
        The absolute or relative path to the aligned ground truth JSON file.
    iou_threshold : float, optional
        The permissive Intersection over Union required to consider a prediction 
        a True Positive. Defaults to 0.5.

    Returns
    -------
    list of tuple
        A list of tuples, where each tuple contains (confidence_score, is_correct)
        for a single prediction. is_correct is 1 for True Positive, 0 for False Positive.
    """
    print(f"\nLoading Ground Truth: {gt_json_path}")
    cocoGt = COCO(gt_json_path)
        
    print(f"Loading Predictions: {pred_json_path}")
    with open(pred_json_path, 'r') as f:
        preds = json.load(f)
    if not preds:
        print("Predictions JSON is empty. Returning empty results.")
        return []
        
    cocoDt = cocoGt.loadRes(pred_json_path)

    # 1. Initialize COCOeval and override parameters for Calibration
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    
    # We only care about a single threshold to evaluate probabilistic calibration
    cocoEval.params.iouThrs = np.array([iou_threshold])
    # Constrain area ranges and max detections to avoid duplicate matrix building
    cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2]]  
    cocoEval.params.areaRngLbl = ['all']
    cocoEval.params.maxDets = [300]

    # 2. Run the C-optimized spatial matching (populates cocoEval.evalImgs)
    cocoEval.evaluate()

    # 3. Intercept and Extract instance-level data before aggregation
    results = []
    
    for e in cocoEval.evalImgs:
        # e is None if an image had neither ground truths nor predictions
        if e is None:
            continue
            
        scores = e['dtScores']
        matches = e['dtMatches'][0]  # Index 0 accesses our single IoU threshold
        ignores = e['dtIgnore'][0]   

        for score, match, ignore in zip(scores, matches, ignores):
            # Skip detections that COCO rules dictate should be ignored 
            # (e.g., matched an 'iscrowd' region or fell outside area limits)
            if ignore:
                continue
            
            is_correct = 1 if match > 0 else 0
            results.append((score, is_correct))
            
    print(f"Extracted {len(results)} valid predictions for calibration analysis.")
    return results


def calculate_and_plot_ece(results: list, num_bins: int, output_dir: str, title: str = "Reliability Diagram"):
    """
    Calculates Expected Calibration Error (ECE) and plots a Reliability Diagram.

    Parameters
    ----------
    results : list of tuple
        A list of (confidence_score, is_correct) tuples generated by bipartite matching.
    num_bins : int
        The number of evenly spaced confidence bins to divide the [0, 1] range into.
    output_dir : str
        The directory path where the resulting reliability diagram plot will be saved.
    title : str, optional
        The title for the generated plot and console output. Defaults to "Reliability Diagram".

    Returns
    -------
    None
    """
    if not results:
        print("No predictions to evaluate calibration on.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    confidences = np.array([r[0] for r in results])
    accuracies = np.array([r[1] for r in results])
    
    bins = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    ece = 0.0
    total_preds = len(results)

    # Calculate statistics per bin
    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
        else:
            bin_acc = 0.0
            bin_conf = 0.0
            
        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        
        # ECE Math
        if bin_count > 0:
            weight = bin_count / total_preds
            ece += weight * np.abs(bin_acc - bin_conf)

    # Console Output
    print("\n" + "="*50)
    print(f"📊 CALIBRATION METRICS: {title}")
    print("="*50)
    print(f"Expected Calibration Error (ECE): {ece:.4f}\n")
    print(f"{'Bin Range':<15} | {'Count':<8} | {'Accuracy':<10} | {'Avg Conf':<10} | {'Gap'}")
    print("-" * 65)
    for i in range(num_bins):
        gap = bin_accuracies[i] - bin_confidences[i]
        rng = f"{bins[i]:.1f} - {bins[i+1]:.1f}"
        print(f"{rng:<15} | {bin_counts[i]:<8} | {bin_accuracies[i]:.4f}   | {bin_confidences[i]:.4f}   | {gap:+.4f}")

    # Plotting Reliability Diagram
    fig, ax1 = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Accuracy Bars
    bin_centers = (bins[:-1] + bins[1:]) / 2
    widths = 1.0 / num_bins
    
    # Draw bars (blue for accuracy)
    ax1.bar(bin_centers, bin_accuracies, width=widths, color='#0072B2', edgecolor='white', alpha=0.8, label='Empirical Accuracy')
    
    # Draw Overconfidence/Underconfidence Gap (red outline)
    ax1.bar(bin_centers, bin_confidences, width=widths, fill=False, edgecolor='#E69F00', linewidth=2, linestyle=':', label='Avg Confidence (Gap)')

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{title}\nECE: {ece:.4f}', pad=20)
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot density histogram as an inset or secondary visual
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, bin_counts, width=widths, color='lightgray', alpha=0.3, zorder=-1)
    ax2.set_ylabel('Number of Predictions (Density)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    # Adjust ax2 scale so it doesn't overlap visually too much with the main chart
    max_count = max(bin_counts) if max(bin_counts) > 0 else 1
    ax2.set_ylim([0, max_count * 4]) 

    plot_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}_reliability.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n✅ Reliability Diagram saved to: {plot_path}")
    print("="*50)


# ==========================================
# ORCHESTRATION PIPELINES
# ==========================================

def _analyze_predictions_json(project: str, name: str, gt_json: str, data_yaml: str, iou_thres: float, bins: int, title: str):
    """
    Shared helper to locate a previously generated YOLO predictions JSON, align it with the 
    Ground Truth in memory, and run the calibration math.

    Parameters
    ----------
    project : str
        The root project directory where the YOLO run is saved.
    name : str
        The specific run name containing the predictions (e.g., 'calibration_eval').
    gt_json : str
        The absolute path to the ground truth COCO JSON file.
    iou_thres : float
        The Intersection over Union required to consider a prediction a True Positive.
    bins : int
        The number of confidence bins used to calculate the Expected Calibration Error.
    title : str
        The title for the generated Reliability Diagram plot.

    Raises
    ------
    FileNotFoundError
        If the `predictions.json` is not found in the run directory.
    """
    val_dir = os.path.join(project, name)
    pred_json_path = os.path.join(val_dir, "predictions.json")
    
    if not os.path.exists(pred_json_path):
        raise FileNotFoundError(f"Expected predictions JSON not found at {pred_json_path}. Did Ultralytics validation fail?")

    # 1. Load everything into memory
    with open(gt_json, 'r') as f:
        gt_data = json.load(f)
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    with open(data_yaml, 'r') as f:
        yolo_idx_to_name = get_yolo_class_names(f)
        # Ensure indices are integers for mapping
        yolo_idx_to_name = {int(k): v for k, v in yolo_idx_to_name.items()}

    # 2. Process in-memory dicts
    pred_data = populate_coco_categories(pred_data, yolo_idx_to_name)
    pred_data, gt_data = align_coco_jsons(pred_data, gt_data)
    
    # 3. Verify alignment before feeding into pycocotools
    _verify_alignment(pred_data, gt_data)

    # 4. Utilize tempfile context manager to handle automatic OS cleanup 
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_gt = os.path.join(temp_dir, "aligned_gt.json")
        temp_pred = os.path.join(temp_dir, "aligned_preds.json")
        
        with open(temp_gt, 'w') as f:
            json.dump(gt_data, f)
        with open(temp_pred, 'w') as f:
            json.dump(pred_data, f)
            
        results = evaluate_coco_calibration(temp_pred, temp_gt, iou_threshold=iou_thres)
        
    calculate_and_plot_ece(results, num_bins=bins, output_dir=val_dir, title=title)


def evaluate_hierarchical_model_calibration(
    weights: str,
    data_yaml: str,
    gt_json: str,
    hierarchy_json: str,
    flat_baseline_yaml: str,
    split: str = 'val',
    imgsz: int = 640,
    batch: int = 16,
    device: str = '',
    iou_thres: float = 0.5,
    bins: int = 10,
    project: str = 'runs/apples2apples',
    name: str = 'calibration_eval'
):
    """
    Runs inference using a Hierarchical YOLO model clamped to a target tier 
    (e.g., leaves) to generate COCO-format predictions, then calculates calibration ECE.
    """
    print("\n" + "="*50)
    print("🚀 RUNNING CALIBRATION EVALUATION (HIERARCHICAL)")
    print("="*50)

    os.environ['HIERARCHY_PATH'] = hierarchy_json
    
    # Load the master taxonomy to resolve IDs
    with open(data_yaml, 'r') as f:
        yolo_names = get_yolo_class_names(f)
    hierarchy_obj = load_hierarchy_from_env(yolo_names)
    
    # Determine the leaf-node IDs to snap predictions to
    with open(flat_baseline_yaml, 'r') as f:
        flat_names_map = get_yolo_class_names(f)
        
    subset_names = list(flat_names_map.values())
    eval_subset_ids = [hierarchy_obj.node_to_idx[name] for name in subset_names if name in hierarchy_obj.node_to_idx]
    
    model = HierarchicalYOLO(weights, hierarchy=hierarchy_obj)
    run_device = None if not device else device
    
    model.val(
        data=data_yaml,
        split=split,
        eval_subset_ids=eval_subset_ids,  # Clamps hierarchical predictions to leaf nodes
        imgsz=imgsz, 
        batch=batch, 
        device=run_device, 
        plots=False,         # Save time, we build our own plots
        save_json=True,      # MANDATORY: Triggers COCO JSON output
        project=project,
        name=name,
        exist_ok=True
    )
    
    _analyze_predictions_json(project, name, gt_json, data_yaml, iou_thres, bins, "Hierarchical Model Calibration")


def evaluate_flat_model_calibration(
    weights: str,
    data_yaml: str,
    gt_json: str,
    split: str = 'val',
    imgsz: int = 640,
    batch: int = 16,
    device: str = '',
    iou_thres: float = 0.5,
    bins: int = 10,
    project: str = 'runs/apples2apples',
    name: str = 'calibration_eval'
):
    """
    Runs standard inference using a Flat YOLO model to generate COCO-format 
    predictions, then calculates calibration ECE.
    """
    print("\n" + "="*50)
    print("🚀 RUNNING CALIBRATION EVALUATION (FLAT)")
    print("="*50)

    model = YOLO(weights)
    run_device = None if not device else device
    
    model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz, 
        batch=batch, 
        device=run_device, 
        plots=False,         
        save_json=True,      
        project=project,
        name=name,
        exist_ok=True
    )

    _analyze_predictions_json(project, name, gt_json, data_yaml, iou_thres, bins, "Flat Model Calibration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration Evaluation Toolbox")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Pipeline Subparser (End-to-End)
    p_pipe = subparsers.add_parser("pipeline", help="Run YOLO validation and compute calibration automatically.")
    p_pipe.add_argument('--weights', required=True, type=str)
    p_pipe.add_argument('--model_type', required=True, choices=['flat', 'hierarchical'])
    p_pipe.add_argument('--data_yaml', required=True, type=str, help="YAML file to trigger validation")
    p_pipe.add_argument('--gt_json', required=True, type=str, help="Absolute path to ground truth COCO JSON")
    p_pipe.add_argument('--hierarchy_json', type=str, help="Master hierarchy.json (required for hierarchical)")
    p_pipe.add_argument('--flat_baseline_yaml', type=str, help="YAML defining leaf-node vocabulary (required for hierarchical)")
    p_pipe.add_argument('--split', type=str, default='val')
    p_pipe.add_argument('--imgsz', type=int, default=640)
    p_pipe.add_argument('--batch', type=int, default=16)
    p_pipe.add_argument('--device', type=str, default='')
    p_pipe.add_argument('--iou_thres', type=float, default=0.5, help="IoU threshold for True Positive assignment")
    p_pipe.add_argument('--bins', type=int, default=10, help="Number of confidence bins for ECE")
    p_pipe.add_argument('--project', type=str, default='runs/calibration')
    p_pipe.add_argument('--name', type=str, default='eval')

    # Standalone JSON Math Subparser (Ecosystem Independent)
    p_json = subparsers.add_parser("eval_json", help="Evaluate ECE using preexisting COCO predictions and GT JSONs.")
    p_json.add_argument('--pred_json', required=True, type=str, help="Path to predictions JSON")
    p_json.add_argument('--gt_json', required=True, type=str, help="Path to ground truth JSON")
    p_json.add_argument('--data_yaml', required=True, type=str, help="Path to YOLO dataset YAML to resolve class names")
    p_json.add_argument('--iou_thres', type=float, default=0.5, help="IoU threshold")
    p_json.add_argument('--bins', type=int, default=10, help="Number of confidence bins")
    p_json.add_argument('--output_dir', type=str, default='runs/calibration/json_eval')
    p_json.add_argument('--title', type=str, default='Reliability Diagram')

    args = parser.parse_args()

    if args.command == "pipeline":
        if args.model_type == 'hierarchical':
            if not args.hierarchy_json or not args.flat_baseline_yaml:
                parser.error("Hierarchical evaluation requires --hierarchy_json and --flat_baseline_yaml.")
                
            evaluate_hierarchical_model_calibration(
                weights=args.weights,
                data_yaml=args.data_yaml,
                gt_json=args.gt_json,
                hierarchy_json=args.hierarchy_json,
                flat_baseline_yaml=args.flat_baseline_yaml,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                iou_thres=args.iou_thres,
                bins=args.bins,
                project=args.project,
                name=args.name
            )
        elif args.model_type == 'flat':
            evaluate_flat_model_calibration(
                weights=args.weights,
                data_yaml=args.data_yaml,
                gt_json=args.gt_json,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                iou_thres=args.iou_thres,
                bins=args.bins,
                project=args.project,
                name=args.name
            )
    elif args.command == "eval_json":
        with open(args.gt_json, 'r') as f:
            gt_data = json.load(f)
        with open(args.pred_json, 'r') as f:
            pred_data = json.load(f)
        with open(args.data_yaml, 'r') as f:
            yolo_idx_to_name = get_yolo_class_names(f)
            yolo_idx_to_name = {int(k): v for k, v in yolo_idx_to_name.items()}

        pred_data = populate_coco_categories(pred_data, yolo_idx_to_name)
        pred_data, gt_data = align_coco_jsons(pred_data, gt_data)
        
        # Verify alignment before feeding into pycocotools
        _verify_alignment(pred_data, gt_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_gt = os.path.join(temp_dir, "aligned_gt.json")
            temp_pred = os.path.join(temp_dir, "aligned_preds.json")
            
            with open(temp_gt, 'w') as f:
                json.dump(gt_data, f)
            with open(temp_pred, 'w') as f:
                json.dump(pred_data, f)

            res = evaluate_coco_calibration(temp_pred, temp_gt, args.iou_thres)
            
        calculate_and_plot_ece(res, args.bins, args.output_dir, args.title)
