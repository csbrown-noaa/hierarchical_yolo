import os
import json
import argparse
from pathlib import Path

from hierarchical_loss import hierarchy_coco_utils
from hierarchical_yolo.hierarchical_detection import HierarchicalDetectionTrainer

def get_max_depth(hierarchy_json_path: str) -> int:
    """
    Parses the global hierarchy tree to determine its maximum phylogenetic depth.

    Parameters
    ----------
    hierarchy_json_path : str
        The path to the 'hierarchy.json' artifact generated during data prep.

    Returns
    -------
    int
        The maximum integer depth of the tree (root = 0).
    """
    with open(hierarchy_json_path, 'r') as f:
        hierarchy_tree = json.load(f)
        
    lineages = hierarchy_coco_utils.build_all_lineages(hierarchy_tree)
    max_depth = max(len(lin) for lin in lineages.values()) - 1
    return max_depth


def train_curriculum(
    data_dir: str, 
    model_dir: str,
    project_name: str,
    base_model: str = "yolov8n.pt", 
    shallow_epochs: int = 2, 
    final_epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    val: bool = False
) -> None:
    """
    Orchestrates the staged curriculum training of a hierarchical YOLO model.

    This passes the weights sequentially from shallow levels of the taxonomy
    (e.g., broad categories like 'animal') down to the final deep classes
    (e.g., 'dog', 'cat'), locking in generalized hierarchical features before
    attempting fine-grained classification.

    Parameters
    ----------
    data_dir : str
        The root directory containing the prepared dataset and 'alternate_depth' curriculum folders.
    model_dir : str
        The root directory where models and logs will be saved.
    project_name : str
        The specific namespace for this experiment run.
    base_model : str, optional
        The starting weights for the Depth 0 model (default is "yolov8n.pt").
    shallow_epochs : int, optional
        The number of epochs to train at each intermediate depth (default is 2).
    final_epochs : int, optional
        The number of epochs to train at the maximum phylogenetic depth (default is 20).
    imgsz : int, optional
        Image size for training (default is 640).
    batch : int, optional
        Batch size for training (default is 16).
    device : str, optional
        Device to run training on (e.g., '', '0', '0,1,2,3'). Default is empty string (auto).
    val : bool, optional
        Whether to run validation during training (default is False, as hierarchical validation is WIP).
    """
    hierarchy_json_path = os.path.join(data_dir, 'hierarchy_data', 'hierarchy.json')
    
    if not os.path.exists(hierarchy_json_path):
        raise FileNotFoundError(f"Hierarchy JSON not found: {hierarchy_json_path}. Did you run data prep?")
        
    max_depth = get_max_depth(hierarchy_json_path)
    print(f"\n{'='*60}")
    print(f"🎓 Initiating Hierarchical Curriculum Training")
    print(f"Graph parsed. Maximum phylogenetic depth is {max_depth}.")
    print(f"{'='*60}\n")
    
    # We maintain a pointer to the weights, starting with the base model, 
    # and updating it to the best.pt of the previous stage after each loop.
    current_weights = base_model
    project_path = os.path.join(model_dir, project_name)
    
    for depth in range(max_depth + 1):
        is_final_stage = (depth == max_depth)
        epochs = final_epochs if is_final_stage else shallow_epochs
        
        print(f"\n--- Staging Curriculum: Depth {depth:03d} / {max_depth:03d} ---")
        data_yaml = os.path.join(data_dir, "alternate_depth", f"{depth:03d}", "train.yaml")
        run_name = f"curriculum_depth_{depth:03d}"
        
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Missing curriculum YAML: {data_yaml}")
            
        overrides = {
            "model": current_weights,
            "data": data_yaml,
            "epochs": epochs,
            "project": project_path,
            "name": run_name,
            "exist_ok": True,  # Allows clean resuming if the script is interrupted and rerun
            "imgsz": imgsz,
            "batch": batch,
            "device": device if device else None,
            "val": val,
        }
        
        # 1. Initialize our custom DDP-ready hierarchical trainer
        trainer = HierarchicalDetectionTrainer(overrides=overrides)
        
        # 2. Inject the hierarchy path globally via environment variable so DDP workers can inherit it
        os.environ["HIERARCHY_PATH"] = hierarchy_json_path
            
        # 3. Execute Training
        print(f"Training Stage {depth} for {epochs} epochs using weights: {current_weights}")
        trainer.train()
        
        # 4. Advance the weights pointer to the output of this stage for the next loop
        best_weights_path = os.path.join(trainer.save_dir, "weights", "best.pt")
        if not os.path.exists(best_weights_path):
            # Fallback to last.pt if validation didn't save a best.pt 
            best_weights_path = os.path.join(trainer.save_dir, "weights", "last.pt")
            
        current_weights = best_weights_path
        print(f"Stage {depth} complete. Weights saved to: {current_weights}")

    print("\n" + "="*60)
    print(f"🎉 Curriculum Training Complete! Final Model: {current_weights}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staged Curriculum Training for Hierarchical YOLO.")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=os.path.expanduser('~/datasets/coco_hierarchical'),
        help="Path to the target root dataset directory containing the curriculum data."
    )
    parser.add_argument(
        '--model_dir', 
        type=str, 
        required=True,
        help="Path to the directory where models, runs, and artifacts will be saved."
    )
    parser.add_argument(
        '--project_name', 
        type=str, 
        required=True,
        help="The specific namespace for this experiment run (e.g., 'baseline_v1' or 'heavy_augs')."
    )
    parser.add_argument(
        '--base_model', 
        type=str, 
        default='yolov8n.pt',
        help="The foundational weights to begin Depth 0 training with (e.g., yolov8n.pt, yolov8s.pt)."
    )
    parser.add_argument(
        '--shallow_epochs', 
        type=int, 
        default=2,
        help="Number of epochs to train each intermediate curriculum depth."
    )
    parser.add_argument(
        '--final_epochs', 
        type=int, 
        default=20,
        help="Number of epochs to train the final maximum-depth hierarchy."
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=640,
        help="Image size for training."
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=16,
        help="Batch size for training."
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='',
        help="Device(s) to use for training (e.g., '0', '0,1,2', or 'cpu')."
    )
    parser.add_argument(
        '--val', 
        action='store_true',
        help="Enable validation during training (default is False, omit flag to disable)."
    )
    
    args = parser.parse_args()
    
    train_curriculum(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        project_name=args.project_name,
        base_model=args.base_model,
        shallow_epochs=args.shallow_epochs,
        final_epochs=args.final_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        val=args.val
    )
