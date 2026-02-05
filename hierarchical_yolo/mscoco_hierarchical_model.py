from .hierarchical_detection import HierarchicalDetectionTrainer
from .yolo_utils import get_yolo_class_names
import yaml
import json
from importlib import resources
from hierarchical_loss.hierarchy import Hierarchy
import contextlib
import os
import urllib.request
import zipfile


YOLO_DATASET_YAML = resources.files('hierarchical_yolo.models.coco128').joinpath('hierarchicalcoco.yaml')
with open(YOLO_DATASET_YAML, 'r') as f:
    COCO_YOLO_ID_MAP = get_yolo_class_names(f)
COCO_HIERARCHY_JSON = resources.files('hierarchical_yolo.models').joinpath('coco_hierarchy.json')
with open(COCO_HIERARCHY_JSON, 'r') as f:
    COCO_HIERARCHY = json.load(f)

class MSCOCOHierarchicalDetectionTrainer(HierarchicalDetectionTrainer):
    # Hierarchy requires the index -> name map in the other direction
    _hierarchy = Hierarchy(COCO_HIERARCHY, {v: k for k,v in COCO_YOLO_ID_MAP.items()})


def download_coco_data(destination_directory: str) -> None:
    data = os.path.join(destination_directory, 'coco')
    os.makedirs(data, exist_ok = True)
    COCO_ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    #COCO_TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2017.zip'
    #COCO_TEST_IMAGES_URL = 'http://images.cocodataset.org/zips/test2017.zip'
    #COCO_VAL_IMAGES_URL = 'http://images.cocodataset.org/zips/val2017.zip'

    # download and unzip everything
    for url in [
        COCO_ANNOTATIONS_URL
        #COCO_TRAIN_IMAGES_URL,
        #COCO_TEST_IMAGES_URL,
        #COCO_VAL_IMAGES_URL
    ]:
        filename = url.split('/')[-1]
        print(f"downloading {filename}")
        destination = os.path.join(data, filename)
        urllib.request.urlretrieve(url, destination)
        print(f"extracting {filename}")
        with zipfile.ZipFile(destination, "r") as f:
            f.extractall(data)
        os.remove(destination)

def prepare_coco_data(destination_directory: str) -> None:
    import pycocowriter.cocomerge
    import pycocowriter.coco2yolo
    import os
    import urllib.request
    import zipfile
    data = os.path.join(destination_directory, 'coco')
    # paths to data
    ANNOTATIONS = os.path.join(data, 'annotations')
    HIERARCHICAL_ANNOTATIONS = os.path.join(ANNOTATIONS, 'hierarchical_annotations')
    os.makedirs(HIERARCHICAL_ANNOTATIONS, exist_ok = True)
    # load the hierarchy
    print(f"loading hierarchy")
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
    expanded_category_coco = {
        'info': {},
        'images': [],
        'annotations': [],
        'licenses': [],
        'categories': all_categories
    }
    # update the files to have the new categories
    for coco_file in ['instances_train2017.json', 'instances_val2017.json']:
        path = os.path.join(ANNOTATIONS, coco_file)
        print(f"loading {coco_file}")
        with open(path, 'r') as f:
            coco = json.load(f)
        print(f"expanding {coco_file}")
        expanded_coco = pycocowriter.cocomerge.coco_merge(coco, expanded_category_coco)
        expanded_coco = pycocowriter.cocomerge.coco_collapse_categories(expanded_coco)
        expanded_coco = pycocowriter.cocomerge.coco_reindex_categories(expanded_coco)
        print(f"writing {coco_file}")
        with open(os.path.join(HIERARCHICAL_ANNOTATIONS, coco_file.replace('instances_','')), 'w') as f:
            json.dump(expanded_coco, f)

    # create yolo-compatible annotations
    print(f"converting to yolo")
    pycocowriter.coco2yolo.coco2yolo(HIERARCHICAL_ANNOTATIONS, data) 
