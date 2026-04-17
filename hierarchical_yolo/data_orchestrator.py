import os
import shutil
import argparse
import json

import pycocowriter.coco2yolo
from hierarchical_loss.hierarchy_expander import (
    StaticTaxonomyProvider, 
    align_coco_dictionaries
)
from hierarchical_yolo.hierarchical_curriculum_builder import build_hierarchical_curriculum
from hierarchical_yolo.flat_baseline_builder import build_flat_baselines
from hierarchical_yolo import yolo_fs_utils

def build_hierarchical_workspace(
    source_dir: str, 
    workspace_dir: str
) -> None:
    """
    Compiles a complete hierarchical training workspace from a directory of COCO JSONs.
    
    This acts as a generic orchestrator that takes a "clean" COCO staging directory 
    (containing any number of JSON splits and a hierarchy.json) and compiles the 
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
    hierarchy_dest = os.path.join(workspace_dir, "hierarchy.json")

    # 2. Phase 1: Master Taxonomic Alignment & Clean Room Population
    print("\n--- Phase 1: Taxonomic Alignment & Populating Master COCO Clean Room ---")
    os.makedirs(master_coco_dir, exist_ok=True)

    # Safely discover only valid COCO split files (ignores metadata JSONs)
    split_files = pycocowriter.coco2yolo.discover_coco_files(source_dir)
    coco_files = split_files.get('train', []) + split_files.get('val', []) + split_files.get('test', [])
    
    if not coco_files:
        raise FileNotFoundError(f"No valid COCO JSON splits (train/val/test) found in {source_dir}. Ensure your staging orchestrator saved them here.")

    # Load all target dictionaries into memory
    print("Loading raw COCO JSON splits into memory...")
    coco_dicts = []
    for file in coco_files:
        with open(file, 'r') as f:
            coco_dicts.append(json.load(f))

    # Setup the Static Provider from the source directory's hierarchy.json
    hierarchy_source = os.path.join(source_dir, "hierarchy.json")

    if not os.path.exists(hierarchy_source):
        raise FileNotFoundError(f"Missing hierarchy.json at {hierarchy_source}. Ensure your staging script generated it or provide a manually created one.")
    
    with open(hierarchy_source, 'r') as f:
        raw_tree = json.load(f)
    taxonomy_provider = StaticTaxonomyProvider(raw_tree)

    # Run Universal Alignment across all dataset splits
    aligned_dicts, master_tree = align_coco_dictionaries(coco_dicts, taxonomy_provider)

    # Dynamically write the resulting Master Hierarchy to the workspace root
    with open(hierarchy_dest, 'w') as f:
        json.dump(master_tree, f, indent=4)
    print(f"\nExported master taxonomy tree -> {hierarchy_dest}")

    # Write Aligned Dicts safely to the Clean Room
    for file, aligned_dict in zip(coco_files, aligned_dicts):
        basename = os.path.basename(file)
        dest_path = os.path.join(master_coco_dir, basename)
        
        with open(dest_path, 'w') as f:
            json.dump(aligned_dict, f)
        print(f"Saved aligned split {basename} -> master_coco/")

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
    args = parser.parse_args()
    
    build_hierarchical_workspace(args.source_dir, args.workspace_dir)
