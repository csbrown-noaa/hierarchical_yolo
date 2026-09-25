import os
import json
import argparse
from pycocowriter import cocomerge
import pycocowriter.coco2yolo

def merge_hierarchies(base, update):
    """
    Merges two flat hierarchy dictionaries structured as {child: parent}.
    Logs a warning if a child category has conflicting parents across datasets.
    """
    for child, parent in update.items():
        if child in base and base[child] != parent:
            print(f"Warning: Conflicting hierarchy for '{child}'. "
                  f"Existing parent '{base[child]}' is being overwritten by '{parent}'.")
        base[child] = parent
    return base

def build_mega_dataset(source_dirs: list[str], target_dir: str) -> None:
    """
    Merges multiple orchestrated COCO datasets into a single mega-dataset.
    Combines JSON splits, merges hierarchies, and hard-links images to conserve disk space.
    """
    print("=" * 60)
    print("🧬 Initiating Mega-Dataset Merging Process")
    print(f"Target Directory: {target_dir}")
    print(f"Source Directories: {source_dirs}")
    print("=" * 60)

    os.makedirs(target_dir, exist_ok=True)
    splits = ['train', 'val', 'test']
    master_hierarchy = {}

    for split in splits:
        dicts_to_merge = []
        image_sources = [] 
        
        print(f"\n--- Processing Split: {split.upper()} ---")
        
        for sdir in source_dirs:
            # Dynamically discover all COCO JSON files for this dataset split
            split_files = pycocowriter.coco2yolo.discover_coco_files(sdir)
            found_jsons = split_files.get(split, [])
            
            if not found_jsons:
                print(f"No '{split}' split found in {sdir}. Skipping.")
                continue
                
            for json_path in found_jsons:
                print(f"Loading {os.path.basename(json_path)} from {sdir}...")
                with open(json_path, 'r') as f:
                    coco_dict = json.load(f)
                    dicts_to_merge.append(coco_dict)
                    
                    # Deduce the physical image directory corresponding to this JSON
                    # e.g., if json_path is 'sdir/mytrain.json', img_dir is 'sdir/mytrain/images'
                    img_dir_name = os.path.splitext(os.path.basename(json_path))[0]
                    source_img_dir = os.path.join(sdir, img_dir_name, "images")
                    image_sources.append((source_img_dir, coco_dict))

        if dicts_to_merge:
            print(f"Merging {len(dicts_to_merge)} datasets for {split} split...")
            merged_dict = cocomerge.coco_merge(*dicts_to_merge)
            
            target_split_path = os.path.join(target_dir, f"{split}.json")
            with open(target_split_path, 'w') as f:
                json.dump(merged_dict, f)
            print(f"Saved merged {split}.json -> {target_split_path}")

            print("Hard-linking associated images...")
            link_count = 0
            
            # Canonical target directory for this split (e.g., target_dir/train/images/)
            target_img_dir = os.path.join(target_dir, split, "images")
            
            for source_img_dir, coco_data in image_sources:
                for img in coco_data.get('images', []):
                    file_name = img.get('file_name')
                    if not file_name:
                        continue
                        
                    src_path = os.path.join(source_img_dir, file_name)
                    dst_path = os.path.join(target_img_dir, file_name)
                    
                    if os.path.exists(src_path):
                        if not os.path.exists(dst_path):
                            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                            # Removed try/except to fail hard on any OS or linking errors
                            os.link(src_path, dst_path)
                            link_count += 1
                    else:
                        # Fail hard if an image referenced in the JSON is missing on disk
                        raise FileNotFoundError(f"Source image missing: {src_path}")
            
            print(f"Successfully hard-linked {link_count} new images for {split} split.")

    print("\n--- Processing Taxonomic Hierarchies ---")
    for sdir in source_dirs:
        hierarchy_path = os.path.join(sdir, "hierarchy.json")
        if os.path.exists(hierarchy_path):
            print(f"Merging hierarchy from {sdir}...")
            with open(hierarchy_path, 'r') as f:
                hierarchy_data = json.load(f)
                master_hierarchy = merge_hierarchies(master_hierarchy, hierarchy_data)

    if master_hierarchy:
        target_hierarchy_path = os.path.join(target_dir, "hierarchy.json")
        with open(target_hierarchy_path, 'w') as f:
            json.dump(master_hierarchy, f, indent=4)
        print(f"Saved master hierarchy tree -> {target_hierarchy_path}")
    else:
        print("No hierarchy.json files found across source directories.")

    print("\n" + "=" * 60)
    print(f"✅ Mega-Dataset compilation complete at: {target_dir}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Multi-Dataset COCO Merger")
    parser.add_argument(
        '--source_dirs', 
        nargs='+', 
        required=True,
        help="List of space-separated paths to staged COCO dataset directories."
    )
    parser.add_argument(
        '--target_dir', 
        type=str, 
        required=True,
        help="Path where the merged mega-dataset will be created."
    )
    
    args = parser.parse_args()
    build_mega_dataset(args.source_dirs, args.target_dir)
