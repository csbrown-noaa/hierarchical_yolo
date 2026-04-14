import os
import json
import shutil
import argparse

import pycocowriter.coco2yolo
from hierarchical_loss import hierarchy_coco_utils
from . import yolo_fs_utils

def generate_flat_baseline(
    current_depth: int, 
    all_json_paths: list[str], 
    lineages: dict, 
    name_to_id: dict, 
    master_categories: list,
    image_source_dir: str,
    flat_models_dir: str
) -> None:
    """
    Builds a fully self-contained, densely indexed YOLO dataset for a specific 
    phylogenetic depth.

    This function extracts the taxonomic hierarchy at the given depth, maps 
    all annotations to their ancestor categories at that depth, strips away 
    unused categories, and dynamically re-indexes the remaining categories to 
    a contiguous 1-to-N range. It then runs the YOLO conversion to finalize 
    the dataset.

    Parameters
    ----------
    current_depth : int
        The targeted level of phylogenetic depth (0 = root) to restrict the model to.
    all_json_paths : list[str]
        A list of absolute file paths to the source COCO JSON files.
    lineages : dict
        A dictionary mapping each taxon to its full phylogenetic lineage from 
        root to leaf.
    name_to_id : dict
        A dictionary mapping the master scientific names to their original 
        COCO category IDs.
    master_categories : list
        The full, original list of category dictionaries from the master COCO file, 
        used to preserve metadata during the rebuild.
    image_source_dir : str
        The root directory containing the original dataset images, used as the source 
        for image symlinks.
    flat_models_dir : str
        The destination directory where the resulting flat baseline dataset 
        will be saved.

    Returns
    -------
    None
    """
    print(f"\n{'='*50}\nBuilding Flat Baseline: Depth {current_depth}\n{'='*50}")
    
    depth_dest_dir = os.path.join(flat_models_dir, f"{current_depth:03d}")
    staging_dir = os.path.join(depth_dest_dir, "staging")
    os.makedirs(staging_dir, exist_ok=True)

    # 1. Map and Cast all splits to the current depth
    depth_map = hierarchy_coco_utils.build_depth_map(lineages, current_depth, name_to_id)
    
    casted_cocos = {}
    for path in all_json_paths:
        with open(path, 'r') as f:
            coco_dict = json.load(f)
        casted_cocos[path] = hierarchy_coco_utils.cast_coco_to_depth(coco_dict, depth_map)

    # 2. Gather active IDs globally so Train/Val/Test share the exact same ID mapping
    active_ids = hierarchy_coco_utils.get_active_category_ids(*casted_cocos.values())
    old_to_new, new_to_old = hierarchy_coco_utils.build_dense_category_map(active_ids)

    # 3. Restrict, Re-index, and Stage the JSONs
    for path, casted_coco in casted_cocos.items():
        final_coco = hierarchy_coco_utils.restrict_and_reindex_coco(casted_coco, old_to_new, master_categories)
        with open(os.path.join(staging_dir, os.path.basename(path)), 'w') as f:
            json.dump(final_coco, f)

    # 4. Convert to YOLO Format (with intercepted symlinks)
    yolo_fs_utils.enforce_symlinks(all_json_paths, image_source_dir, depth_dest_dir)
    print("  -> Running Pycocowriter Conversion...")
    pycocowriter.coco2yolo.coco2yolo(staging_dir, depth_dest_dir)
    
    shutil.rmtree(staging_dir)
    print(f"Depth {current_depth} completed successfully.")


def build_flat_baselines(
    coco_dir: str, 
    hierarchy_path: str, 
    flat_models_dir: str, 
    image_source_dir: str
) -> None:
    """
    Orchestrates the generation of flat YOLO baselines from a hierarchical dataset.
    
    This acts as the main entry point to parse the master hierarchy tree, discover 
    all relevant COCO split files (train/val/test), and iteratively generate 
    independent flat YOLO models for every available phylogenetic depth.

    Parameters
    ----------
    coco_dir : str
        The directory containing the staging master COCO JSONs.
    hierarchy_path : str
        The file path to the master hierarchy.json.
    flat_models_dir : str
        The destination directory where the flat baseline YOLO models will be generated.
    image_source_dir : str
        The root directory containing the original dataset images, used to enforce symlinks.

    Returns
    -------
    None
    """
    print(f"Initializing Flat Baseline Generation from: {coco_dir}")
    
    if not os.path.exists(hierarchy_path):
        print(f"Error: Hierarchy JSON not found at {hierarchy_path}")
        return
        
    with open(hierarchy_path, 'r') as f:
        hierarchy_tree = json.load(f)

    lineages = hierarchy_coco_utils.build_all_lineages(hierarchy_tree)
    max_depth = max(len(lin) for lin in lineages.values()) - 1
    
    split_files = pycocowriter.coco2yolo.discover_coco_files(coco_dir)
    all_json_paths = split_files['train'] + split_files['val'] + split_files['test']
    
    if not all_json_paths:
        print(f"Error: No COCO JSON files found in {coco_dir}.")
        return

    with open(all_json_paths[0], 'r') as f:
        reference_coco = json.load(f)
        
    master_categories = reference_coco.get('categories', [])
    name_to_id = {cat['name']: cat['id'] for cat in master_categories}

    # Generate baselines iteratively
    for current_depth in range(max_depth + 1):
        generate_flat_baseline(
            current_depth=current_depth, 
            all_json_paths=all_json_paths, 
            lineages=lineages, 
            name_to_id=name_to_id, 
            master_categories=master_categories,
            image_source_dir=image_source_dir,
            flat_models_dir=flat_models_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flat YOLO baselines at varying taxonomic depths.")
    parser.add_argument('--coco_dir', type=str, required=True, help="Path to the master COCO JSONs")
    parser.add_argument('--hierarchy_path', type=str, required=True, help="Path to hierarchy.json")
    parser.add_argument('--flat_models_dir', type=str, required=True, help="Output destination for YOLO datasets")
    parser.add_argument('--image_source_dir', type=str, required=True, help="Path back to the raw image vault for symlinks")
    args = parser.parse_args()
    
    build_flat_baselines(args.coco_dir, args.hierarchy_path, args.flat_models_dir, args.image_source_dir)
