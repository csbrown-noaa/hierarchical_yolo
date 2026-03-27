import os
import json
import shutil
import argparse

import pycocowriter.coco2yolo
from hierarchical_loss import hierarchy_coco_utils
from . import yolo_fs_utils

def build_curriculum_depth(
    current_depth: int, 
    all_json_paths: list[str], 
    lineages: dict, 
    name_to_id: dict, 
    max_depth: int,
    data_dir: str,
    curriculum_dir: str
) -> None:
    """
    Orchestrates the staging and conversion of a single curriculum depth layer.

    This function isolates the dataset for a specific phylogenetic depth, mutates 
    the annotations in memory to reflect their ancestral categories, and passes 
    the modified data to the Pycocowriter engine to generate a YOLO-compliant 
    dataset with proper image symlinks.

    Parameters
    ----------
    current_depth : int
        The targeted level of phylogenetic depth (0 = root) to map the dataset to.
    all_json_paths : list[str]
        A list of absolute file paths to the source COCO JSON files.
    lineages : dict
        A dictionary mapping each taxon to its full phylogenetic lineage from root to leaf.
    name_to_id : dict
        A dictionary mapping the master scientific names to their original COCO category IDs.
    max_depth : int
        The maximum phylogenetic depth found in the entire master tree (used for logging).
    data_dir : str
        The root directory containing the original dataset, used as the source for image symlinks.
    curriculum_dir : str
        The destination directory where the resulting YOLO curriculum datasets will be saved.

    Returns
    -------
    None
    """
    print(f"\n{'='*50}\nBuilding Curriculum Set: Depth {current_depth} / {max_depth}\n{'='*50}")
    
    depth_dest_dir = os.path.join(curriculum_dir, f"{current_depth:03d}")
    staging_dir = os.path.join(depth_dest_dir, "staging")
    os.makedirs(staging_dir, exist_ok=True)
    
    depth_map = hierarchy_coco_utils.build_depth_map(lineages, current_depth, name_to_id)
    
    # 1. Cast all discovered datasets to the current depth into the staging folder
    for json_path in all_json_paths:
        basename = os.path.basename(json_path)
        with open(json_path, 'r') as f:
            coco_dict = json.load(f)
            
        casted_coco = hierarchy_coco_utils.cast_coco_to_depth(coco_dict, depth_map)
        
        out_path = os.path.join(staging_dir, basename)
        with open(out_path, 'w') as f:
            json.dump(casted_coco, f)
            
    # 2. Setup Symlinks FIRST to intercept image downloads from pycocowriter
    yolo_fs_utils.enforce_symlinks(all_json_paths, data_dir, depth_dest_dir)
    
    # 3. Run YOLO conversion passing the staging directory
    print("  -> Running Pycocowriter Conversion...")
    pycocowriter.coco2yolo.coco2yolo(staging_dir, depth_dest_dir)
    
    # 4. Clean up staging artifacts
    shutil.rmtree(staging_dir)
    print(f"Depth {current_depth} completed successfully.")


def build_hierarchical_curriculum(data_dir: str) -> None:
    """
    Orchestrates the generation of hierarchical curriculum datasets.
    
    This acts as the main entry point to parse the master hierarchy tree, discover 
    all relevant COCO split files (train/val/test), and iteratively generate 
    independent YOLO models mapped to every available phylogenetic depth while 
    preserving the full 1-to-N network head shape.

    Parameters
    ----------
    data_dir : str
        The base directory containing the source COCO JSONs, original imagery, 
        and the 'hierarchy_data/hierarchy.json' file.

    Returns
    -------
    None
    """
    print(f"Initializing Hierarchical Curriculum Generation for: {data_dir}")
    
    hierarchy_json = os.path.join(data_dir, 'hierarchy_data', 'hierarchy.json')
    curriculum_dir = os.path.join(data_dir, 'alternate_depth')
    
    if not os.path.exists(hierarchy_json):
        print(f"Error: Hierarchy JSON not found at {hierarchy_json}")
        return

    # Load and parse taxonomy
    with open(hierarchy_json, 'r') as f:
        hierarchy_tree = json.load(f)

    lineages = hierarchy_coco_utils.build_all_lineages(hierarchy_tree)
    max_depth = max(len(lin) for lin in lineages.values()) - 1
    print(f"Graph parsed. Maximum phylogenetic depth is {max_depth}.")

    # Discover and map active datasets
    split_files = pycocowriter.coco2yolo.discover_coco_files(data_dir)
    all_json_paths = split_files['train'] + split_files['val'] + split_files['test']
    
    if not all_json_paths:
        print(f"Error: No COCO JSON files found in {data_dir}.")
        return

    # Extract global category map from the first file (strictly unified)
    with open(all_json_paths[0], 'r') as f:
        reference_coco = json.load(f)
    name_to_id = {cat['name']: cat['id'] for cat in reference_coco.get('categories', [])}

    # Iteratively build curriculum sets
    for current_depth in range(max_depth + 1):
        build_curriculum_depth(
            current_depth=current_depth, 
            all_json_paths=all_json_paths, 
            lineages=lineages, 
            name_to_id=name_to_id, 
            max_depth=max_depth,
            data_dir=data_dir,
            curriculum_dir=curriculum_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hierarchical curriculum datasets at varying taxonomic depths.")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=os.path.expanduser('~/datasets/gfisher'),
        help="Path to the root dataset directory containing the COCO JSONs and hierarchy_data/"
    )
    args = parser.parse_args()
    
    build_hierarchical_curriculum(args.data_dir)
