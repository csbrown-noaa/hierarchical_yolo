from .hierarchical_detection import HierarchicalDetectionTrainer
from .yolo_utils import get_yolo_class_names
import yaml
import json
from importlib import resources
from hierarchical_loss.hierarchy import Hierarchy


YOLO_DATASET_YAML = resources.files('hierarchical_yolo.models.coco128').joinpath('hierarchicalcoco128.yaml')
with open(YOLO_DATASET_YAML, 'r') as f:
    COCO_YOLO_ID_MAP = get_yolo_class_names(f)
COCO_HIERARCHY_JSON = resources.files('hierarchical_yolo.models').joinpath('coco_hierarchy.json')
with open(COCO_HIERARCHY_JSON, 'r') as f:
    COCO_HIERARCHY = json.load(f)

class MSCOCOHierarchicalDetectionTrainer(HierarchicalDetectionTrainer):
    # Hierarchy requires the index -> name map in the other direction
    _hierarchy = Hierarchy(COCO_HIERARCHY, {v: k for k,v in COCO_YOLO_ID_MAP.items()})

def prepare_coco_data(destination_directory: str) -> None:
    import pycocowriter.cocomerge
    import pycocowriter.coco2yolo
    import os
    import urllib.request
    import zipfile
    DATA = os.path.join(destination_directory, 'hierarchical_coco')
    os.makedirs(DATA, exist_ok = True)
    COCO_ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    COCO_TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2017.zip'
    COCO_TEST_IMAGES_URL = 'http://images.cocodataset.org/zips/test2017.zip'
    COCO_VAL_IMAGES_URL = 'http://images.cocodataset.org/zips/val2017.zip'
    # paths to data
    ANNOTATIONS = os.path.join(DATA, 'annotations', 'instances')
    TRAIN_ANNOTATIONS = os.path.join(ANNOTATIONS, 'instances_train2017.json')
    TEST_ANNOTATIONS = os.path.join(ANNOTATIONS, 'instances_test2017.json')

    # download and unzip everything
    for url in [
        COCO_ANNOTATIONS_URL,
        COCO_TRAIN_IMAGES_URL,
        COCO_TEST_IMAGES_URL,
        COCO_VAL_IMAGES_URL
    ]:
        filename = url.split('/')[-1]
        destination = os.path.join(DATA, filename)
        urllib.request.urlretrieve(url, destination)
        with zipfile.ZipFile(destination, "r") as f:
            f.extractall(DATA)
        os.remove(destination)

    # load the hierarchy
    COCO_HIERARCHY_JSON = resources.files('hierarchical_yolo.models').joinpath('coco_hierarchy.json')
    with open(COCO_HIERARCHY_JSON, 'r') as f:
        COCO_HIERARCHY = json.load(f)
    all_category_names = sorted(COCO_HIERARCHY.keys() | COCO_HIERARCHY.values())
    all_categories = [
        { 
            'id': i,
            'name': cat 
        } for i, cat in enumerate(all_category_names)
    ]

    # update the files to have the new categories
    for coco_file in [TEST_ANNOTATIONS, TRAIN_ANNOTATIONS]:
        with open(coco_file, 'r') as f:
            coco = json.load(f)
        expanded_coco = pycocowriter.cocomerge.coco_merge(coco, all_categories)
        expanded_coco = pycocowriter.cocomerge.coco_collapse_categories(expanded_coco)
        expanded_coco = pycocowriter.cocomerge.coco_reindex_categories(expanded_coco)
        with open(coco_file, 'w') as f:
            json.dump(expanded_coco, f)

    # create yolo-compatible annotations
    pycocowriter.coco2yolo.coco2yolo(ANNOTATIONS, DATA) 
