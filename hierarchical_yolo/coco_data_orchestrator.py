import os
import argparse
import urllib.request
import zipfile

import pycocowriter.coco2yolo
from hierarchical_loss.hierarchy_expander import StaticTaxonomyProvider, process_hierarchical_dataset
from hierarchical_yolo.hierarchical_curriculum_builder import build_hierarchical_curriculum
from hierarchical_yolo.flat_baseline_builder import build_flat_baselines

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def get_hierarchy_path() -> str:
    """
    Locates the static coco_hierarchy.json file bundled within the package resources.
    Uses modern importlib with a graceful fallback for older Python versions.
    """
    try:
        from importlib.resources import files
        path = files('hierarchical_yolo.models').joinpath('coco_hierarchy.json')
        return str(path)
    except ImportError:
        import pkg_resources
        return pkg_resources.resource_filename('hierarchical_yolo.models', 'coco_hierarchy.json')


def download_and_extract_coco(data_dir: str) -> list[str]:
    """
    Downloads the official COCO 2017 annotations zip file and extracts the train/val JSONs.
    
    Parameters
    ----------
    data_dir : str
        The root directory where the downloads should be cached and extracted.
        
    Returns
    -------
    list[str]
        A list of absolute file paths to the extracted train and val JSON files.
    """
    zip_path = os.path.join(data_dir, "annotations_trainval2017.zip")
    extract_dir = os.path.join(data_dir, "annotations_extracted")
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        
        if not os.path.exists(zip_path):
            print(f"  -> Downloading COCO 2017 annotations from {COCO_ANNOTATIONS_URL}...")
            urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path)
        
        print("  -> Extracting COCO annotations zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
    # The zip creates a nested 'annotations' folder internally
    train_json = os.path.join(extract_dir, "annotations", "instances_train2017.json")
    val_json = os.path.join(extract_dir, "annotations", "instances_val2017.json")
    
    if not os.path.exists(train_json) or not os.path.exists(val_json):
        raise FileNotFoundError("CRITICAL: Failed to locate expected COCO JSONs after extraction.")
        
    return [train_json, val_json]


def main() -> None:
    """
    Executes the end-to-end data preparation pipeline for the COCO Hierarchical dataset.
    """
    parser = argparse.ArgumentParser(description="End-to-end data preparation for Hierarchical COCO.")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=os.path.expanduser('~/datasets/coco_hierarchical'),
        help="Path to the target root dataset directory."
    )
    args = parser.parse_args()
    
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Initiating End-to-End Hierarchical COCO Data Pipeline")
    print("=" * 60)

    # Phase 0: Download COCO annotations
    print("\n--- Phase 0: Fetching Raw COCO 2017 Annotations ---")
    coco_json_paths = download_and_extract_coco(data_dir)

    # Phase 1: Ingestion & Alignment using Static Taxonomy
    print("\n--- Phase 1: Ingestion & Static Taxonomy Alignment ---")
    hierarchy_path = get_hierarchy_path()
    provider = StaticTaxonomyProvider(hierarchy_path)
    
    process_hierarchical_dataset(
        data_dir=data_dir, 
        coco_sources=coco_json_paths,
        taxonomy_provider=provider
    )

    # Phase 2: Convert to Master YOLO format
    print("\n--- Phase 2: Master YOLO Conversion & Image Fetching ---")
    # Note: This will trigger pycocowriter to fetch the actual COCO imagery
    # using the `coco_url` keys present inside the aligned JSON dictionaries.
    pycocowriter.coco2yolo.coco2yolo(data_dir, data_dir)

    # Phase 3: Build the Hierarchical Curriculum 
    print("\n--- Phase 3: Hierarchical Curriculum Generation ---")
    build_hierarchical_curriculum(data_dir=data_dir)

    # Phase 4: Build the Flat Baselines 
    print("\n--- Phase 4: Flat Baseline Generation ---")
    build_flat_baselines(data_dir=data_dir)

    print("\n" + "=" * 60)
    print("✅ COCO Hierarchical Pipeline Complete! All datasets and configs are ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
