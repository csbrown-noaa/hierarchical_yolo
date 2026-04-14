import os
import shutil
import argparse

import pycocowriter.coco2yolo
from hierarchical_yolo.hierarchical_curriculum_builder import build_hierarchical_curriculum
from hierarchical_yolo.flat_baseline_builder import build_flat_baselines
from hierarchical_yolo import yolo_fs_utils

def build_hierarchical_workspace(source_dir: str, workspace_dir: str, hierarchy_source: str = None) -> None:
    """
    Compiles a complete hierarchical training workspace from a directory of COCO JSONs.
    
    This acts as a generic orchestrator that takes a "clean" COCO staging directory 
    (containing any number of JSON splits and a hierarchy map) and compiles the 
    various YOLO datasets required for hierarchical and baseline training/evaluation.
    """
    print("=" * 60)
    print(f"🚀 Initializing Hierarchical Training Workspace")
    print(f"Source Directory (Images & Raw Data): {source_dir}")
    print(f"Workspace Directory (Compiled YOLO Data): {workspace_dir}")
    print("=" * 60)

    # 1. Define Paths
    master_coco_dir = os.path.join(workspace_dir, "master_coco")
    master_yolo_dir = os.path.join(workspace_dir, "master_yolo")
    tier_full_head_dir = os.path.join(workspace_dir, "tier_yolo_full_head")
    tier_flat_dir = os.path.join(workspace_dir, "tier_yolo_flat_specialists")
    
    # Locate hierarchy.json
    if hierarchy_source is None:
        hierarchy_source = os.path.join(source_dir, "hierarchy.json")
        # Legacy fallback
        if not os.path.exists(hierarchy_source):
            legacy_source = os.path.join(source_dir, "hierarchy_data", "hierarchy.json")
            if os.path.exists(legacy_source):
                hierarchy_source = legacy_source

    if not os.path.exists(hierarchy_source):
        raise FileNotFoundError(f"Missing hierarchy.json at {hierarchy_source}. Ensure your staging orchestrator generated it or provide the path explicitly.")
            
    hierarchy_dest = os.path.join(workspace_dir, "hierarchy.json")

    # 2. Populate master_coco (The Clean Room)
    print("\n--- Phase 1: Populating Master COCO Clean Room ---")
    os.makedirs(master_coco_dir, exist_ok=True)
            
    shutil.copyfile(hierarchy_source, hierarchy_dest)
    print(f"Copied hierarchy.json to workspace root.")

    # Safely discover only valid COCO split files (ignores metadata/hierarchy JSONs)
    split_files = pycocowriter.coco2yolo.discover_coco_files(source_dir)
    coco_files = split_files.get('train', []) + split_files.get('val', []) + split_files.get('test', [])
    
    if not coco_files:
        raise FileNotFoundError(f"No valid COCO JSON splits (train/val/test) found in {source_dir}. Ensure your staging orchestrator saved them here.")

    for file in coco_files:
        basename = os.path.basename(file)
        dest_path = os.path.join(master_coco_dir, basename)
        shutil.copyfile(file, dest_path)
        print(f"Copied {basename} -> master_coco/")

    # 3. Master YOLO Conversion
    print("\n--- Phase 2: Master YOLO Conversion ---")
    os.makedirs(master_yolo_dir, exist_ok=True)
    
    workspace_split_files = pycocowriter.coco2yolo.discover_coco_files(master_coco_dir)
    workspace_coco_paths = workspace_split_files.get('train', []) + workspace_split_files.get('val', []) + workspace_split_files.get('test', [])
    
    # Setup symlinks first to intercept image downloads from pycocowriter
    yolo_fs_utils.enforce_symlinks(workspace_coco_paths, source_dir, master_yolo_dir)
    pycocowriter.coco2yolo.coco2yolo(master_coco_dir, master_yolo_dir)

    # 4. Tier YOLO Full Head (Generalist Curriculum)
    print("\n--- Phase 3: Building Tier YOLO Full Head Curriculum ---")
    build_hierarchical_curriculum(
        coco_dir=master_coco_dir,
        hierarchy_path=hierarchy_dest,
        curriculum_dir=tier_full_head_dir,
        image_source_dir=source_dir  # Symlinks point safely back to the raw source directory
    )

    # 5. Tier YOLO Flat Specialists (Baseline Models)
    print("\n--- Phase 4: Building Tier YOLO Flat Specialists ---")
    build_flat_baselines(
        coco_dir=master_coco_dir,
        hierarchy_path=hierarchy_dest,
        flat_models_dir=tier_flat_dir,
        image_source_dir=source_dir  # Symlinks point safely back to the raw source directory
    )

    print("\n" + "=" * 60)
    print(f"✅ Workspace Generation Complete! Ready for training in: {workspace_dir}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Hierarchical YOLO Workspace Compiler")
    parser.add_argument(
        '--source_dir', 
        type=str, 
        required=True,
        help="Path to the source staging directory containing the raw COCO JSONs, original images, and hierarchy.json"
    )
    parser.add_argument(
        '--workspace_dir', 
        type=str, 
        default=os.path.expanduser('~/hierarchical_training_workspace'),
        help="Target directory for the generated YOLO datasets and configuration"
    )
    parser.add_argument(
        '--hierarchy_path',
        type=str,
        default=None,
        help="Optional explicitly defined path to hierarchy.json. Defaults to hierarchy.json within source_dir."
    )
    args = parser.parse_args()
    
    build_hierarchical_workspace(args.source_dir, args.workspace_dir, args.hierarchy_path)
