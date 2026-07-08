import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# YOLO Imports (Only required for the 'pipeline' command)
from ultralytics import YOLO
from hierarchical_yolo.hierarchical_detection import HierarchicalYOLO, load_hierarchy_from_env
from hierarchical_yolo.yolo_utils import get_yolo_class_names

# ==========================================
# ECOSYSTEM-INDEPENDENT CALIBRATION MATH
# ==========================================

def calculate_iou(boxA: list, boxB: list) -> float:
    """
    Calculates Intersection over Union (IoU) for two COCO-format boxes.

    Parameters
    ----------
    boxA : list or tuple of float
        Bounding box A in [x, y, width, height] format.
    boxB : list or tuple of float
        Bounding box B in [x, y, width, height] format.

    Returns
    -------
    float
        The Intersection over Union score, bounded between 0.0 and 1.0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)


def load_and_group_coco_data(pred_json_path: str, gt_json_path: str) -> tuple:
    """
    Loads COCO predictions and ground truth JSONs and groups them by image ID.

    Parameters
    ----------
    pred_json_path : str
        The absolute or relative path to the predictions JSON file.
    gt_json_path : str
        The absolute or relative path to the ground truth JSON file.

    Returns
    -------
    tuple of (dict, dict)
        A tuple containing two dictionaries:
        - preds_by_image: Predictions grouped by image_id.
        - gt_by_image: Ground truth annotations grouped by image_id.
    """
    print(f"\nLoading Ground Truth: {gt_json_path}")
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
        
    print(f"Loading Predictions: {pred_json_path}")
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)

    # Group Ground Truths by image_id
    gt_by_image = {}
    for ann in gt_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)

    # Group Predictions by image_id
    preds_by_image = {}
    for p in pred_data:
        img_id = p['image_id']
        if img_id not in preds_by_image:
            preds_by_image[img_id] = []
        preds_by_image[img_id].append(p)

    return preds_by_image, gt_by_image


def compute_bipartite_matches(preds_by_image: dict, gt_by_image: dict, iou_threshold: float = 0.5) -> list:
    """
    Evaluates grouped predictions against ground truths using bipartite matching.

    Parameters
    ----------
    preds_by_image : dict
        Predictions dictionary keyed by image_id.
    gt_by_image : dict
        Ground truth dictionary keyed by image_id.
    iou_threshold : float, optional
        The minimum Intersection over Union required to consider a prediction 
        a True Positive. Defaults to 0.5.

    Returns
    -------
    list of tuple
        A list of tuples, where each tuple contains (confidence_score, is_correct)
        for a single prediction. is_correct is 1 for True Positive, 0 for False Positive.
    """
    results = []
    print(f"Matching predictions to ground truth (IoU >= {iou_threshold})...")
    
    for img_id, preds in preds_by_image.items():
        gts = gt_by_image.get(img_id, [])
        
        # COCO Eval standard: evaluate highest confidence predictions first
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        matched_gt_ids = set()
        
        for p in preds:
            best_iou = 0.0
            best_gt_id = None
            
            for gt in gts:
                if gt['id'] in matched_gt_ids:
                    continue
                # Calibration requires correct class specification
                if gt['category_id'] != p['category_id']:
                    continue
                    
                iou = calculate_iou(p['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_id = gt['id']
            
            if best_iou >= iou_threshold:
                matched_gt_ids.add(best_gt_id)
                results.append((p['score'], 1))  # True Positive
            else:
                results.append((p['score'], 0))  # False Positive

    print(f"Evaluated {len(results)} total predictions.")
    return results


def evaluate_coco_calibration(pred_json_path: str, gt_json_path: str, iou_threshold: float = 0.5) -> list:
    """
    End-to-end pipeline to load COCO datasets and perform bipartite matching.

    Parameters
    ----------
    pred_json_path : str
        The absolute or relative path to the predictions JSON file.
    gt_json_path : str
        The absolute or relative path to the ground truth JSON file.
    iou_threshold : float, optional
        The minimum Intersection over Union required to consider a prediction 
        a True Positive. Defaults to 0.5.

    Returns
    -------
    list of tuple
        A list of tuples, where each tuple contains (confidence_score, is_correct)
        for a single prediction. is_correct is 1 for True Positive, 0 for False Positive.
    """
    preds_by_image, gt_by_image = load_and_group_coco_data(pred_json_path, gt_json_path)
    results = compute_bipartite_matches(preds_by_image, gt_by_image, iou_threshold)
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
# ORCHESTRATION PIPELINE
# ==========================================

def run_calibration_pipeline(
    weights: str,
    model_type: str,
    data_yaml: str,
    gt_json: str,
    hierarchy_json: str = None,
    flat_baseline_yaml: str = None,
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
    Wraps YOLO validation to generate predictions.json, then evaluates calibration.

    Parameters
    ----------
    weights : str
        Path to the trained model weights file (e.g., 'best.pt').
    model_type : str
        The type of model to evaluate. Must be either 'flat' or 'hierarchical'.
    data_yaml : str
        Path to the YOLO dataset YAML file to trigger validation.
    gt_json : str
        Absolute path to the ground truth COCO JSON file.
    hierarchy_json : str, optional
        Master hierarchy.json file path (required if model_type is 'hierarchical').
    flat_baseline_yaml : str, optional
        YAML file defining the leaf-node vocabulary (required if model_type is 'hierarchical').
    split : str, optional
        The dataset split to run validation on. Defaults to 'val'.
    imgsz : int, optional
        The input image size for the model. Defaults to 640.
    batch : int, optional
        The batch size for validation inference. Defaults to 16.
    device : str, optional
        The device to run inference on (e.g., 'cuda:0', 'cpu'). Defaults to ''.
    iou_thres : float, optional
        The Intersection over Union threshold for matching predictions to ground truth. Defaults to 0.5.
    bins : int, optional
        The number of confidence bins used to calculate the Expected Calibration Error. Defaults to 10.
    project : str, optional
        The root directory to save YOLO runs and calibration outputs. Defaults to 'runs/apples2apples'.
    name : str, optional
        The specific experiment name/folder for the saved outputs. Defaults to 'calibration_eval'.

    Returns
    -------
    None
    """
    print("\n" + "="*50)
    print(f"🚀 RUNNING PIPELINE: CALIBRATION EVALUATION ({model_type.upper()})")
    print("="*50)

    run_device = None if not device else device
    eval_subset_ids = None
    
    if model_type == 'hierarchical':
        if not hierarchy_json or not flat_baseline_yaml:
            raise ValueError("hierarchical evaluation requires --hierarchy_json and --flat_baseline_yaml.")
            
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
    
    elif model_type == 'flat':
        model = YOLO(weights)
    else:
        raise ValueError("model_type must be either 'flat' or 'hierarchical'")

    # 1. Run YOLO validation to dump predictions.json
    res = model.val(
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

    # 2. Locate the dumped JSON (Ultralytics standard output naming)
    val_dir = os.path.join(project, name)
    pred_json_path = os.path.join(val_dir, "predictions.json")
    
    if not os.path.exists(pred_json_path):
        raise FileNotFoundError(f"Expected predictions JSON not found at {pred_json_path}. Did Ultralytics validation fail?")

    # 3. Run the independent Calibration Math
    title = f"{model_type.capitalize()} Model Calibration"
    results = evaluate_coco_calibration(pred_json_path, gt_json, iou_threshold=iou_thres)
    calculate_and_plot_ece(results, num_bins=bins, output_dir=val_dir, title=title)


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
    p_json.add_argument('--iou_thres', type=float, default=0.5, help="IoU threshold")
    p_json.add_argument('--bins', type=int, default=10, help="Number of confidence bins")
    p_json.add_argument('--output_dir', type=str, default='runs/calibration/json_eval')
    p_json.add_argument('--title', type=str, default='Reliability Diagram')

    args = parser.parse_args()

    if args.command == "pipeline":
        run_calibration_pipeline(
            weights=args.weights,
            model_type=args.model_type,
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
    elif args.command == "eval_json":
        res = evaluate_coco_calibration(args.pred_json, args.gt_json, args.iou_thres)
        calculate_and_plot_ece(res, args.bins, args.output_dir, args.title)
