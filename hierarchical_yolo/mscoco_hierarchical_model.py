import ultralytics.utils.loss
import ultralytics.models
from hierarchical_yolo.hierarchical_detection import HierarchicalDetectionTrainer


COCO_CLASSES = {
    0: u'__background__',
    1: u'person',
    2: u'bicycle',
    3: u'car',
    4: u'motorcycle',
    5: u'airplane',
    6: u'bus',
    7: u'train',
    8: u'truck',
    9: u'boat',
    10: u'traffic light',
    11: u'fire hydrant',
    12: u'stop sign',
    13: u'parking meter',
    14: u'bench',
    15: u'bird',
    16: u'cat',
    17: u'dog',
    18: u'horse',
    19: u'sheep',
    20: u'cow',
    21: u'elephant',
    22: u'bear',
    23: u'zebra',
    24: u'giraffe',
    25: u'backpack',
    26: u'umbrella',
    27: u'handbag',
    28: u'tie',
    29: u'suitcase',
    30: u'frisbee',
    31: u'skis',
    32: u'snowboard',
    33: u'sports ball',
    34: u'kite',
    35: u'baseball bat',
    36: u'baseball glove',
    37: u'skateboard',
    38: u'surfboard',
    39: u'tennis racket',
    40: u'bottle',
    41: u'wine glass',
    42: u'cup',
    43: u'fork',
    44: u'knife',
    45: u'spoon',
    46: u'bowl',
    47: u'banana',
    48: u'apple',
    49: u'sandwich',
    50: u'orange',
    51: u'broccoli',
    52: u'carrot',
    53: u'hot dog',
    54: u'pizza',
    55: u'donut',
    56: u'cake',
    57: u'chair',
    58: u'couch',
    59: u'potted plant',
    60: u'bed',
    61: u'dining table',
    62: u'toilet',
    63: u'tv',
    64: u'laptop',
    65: u'mouse',
    66: u'remote',
    67: u'keyboard',
    68: u'cell phone',
    69: u'microwave',
    70: u'oven',
    71: u'toaster',
    72: u'sink',
    73: u'refrigerator',
    74: u'book',
    75: u'clock',
    76: u'vase',
    77: u'scissors',
    78: u'teddy bear',
    79: u'hair dryer',
    80: u'toothbrush'
}


# A hierarchical representation of the COCO classes.
# Intermediate nodes are new categories, and leaf nodes are the original COCO classes.
COCO_HIERARCHY = {
    'object': {
        'living_thing': {
            'person': [],
            'animal': {
                'domestic_animal': {
                    'cat': [], 'dog': [], 'bird': []
                },
                'livestock_animal': {
                    'horse': [], 'sheep': [], 'cow': []
                },
                'wild_animal': {
                    'elephant': [], 'bear': [], 'zebra': [], 'giraffe': []
                }
            }
        },
        'vehicle': {
            'land_vehicle': {
                'bicycle': [], 'car': [], 'motorcycle': [], 'bus': [], 'train': [], 'truck': []
            },
            'airplane': [],
            'boat': []
        },
        'outdoor_and_sports_object': {
            'street_fixture': {
                'traffic light': [], 'fire hydrant': [], 'stop sign': [], 'parking meter': [], 'bench': []
            },
            'sports_equipment': {
                'ball_sport_equipment': {
                    'sports ball': [], 'baseball bat': [], 'baseball glove': [], 'tennis racket': []
                },
                'board_sport_equipment': {
                    'skateboard': [], 'surfboard': [], 'snowboard': []
                },
                'other_sports_equipment': {
                    'frisbee': [], 'skis': [], 'kite': []
                }
            }
        },
        'indoor_object': {
            'furniture': {
                'chair': [], 'couch': [], 'potted plant': [], 'bed': [], 'dining table': []
            },
            'appliance_and_electronics': {
                'appliance': {
                    'microwave': [], 'oven': [], 'toaster': [], 'sink': [], 'refrigerator': [], 'toilet': []
                },
                'electronics': {
                    'tv': [], 'laptop': [], 'mouse': [], 'remote': [], 'keyboard': [], 'cell phone': []
                }
            },
            'kitchenware': {
                'bottle': [], 'wine glass': [], 'cup': [], 'fork': [], 'knife': [], 'spoon': [], 'bowl': []
            },
            'personal_care_item': {
                'hair dryer': [], 'toothbrush': []
            }
        },
        'accessory_and_item': {
            'personal_accessory': {
                'backpack': [], 'umbrella': [], 'handbag': [], 'tie': [], 'suitcase': []
            },
            'home_and_office_item': {
                'book': [], 'clock': [], 'vase': [], 'scissors': [], 'teddy bear': []
            }
        },
        'food': {
            'fruit': {
                'banana': [], 'apple': [], 'orange': []
            },
            'vegetable': {
                'broccoli': [], 'carrot': []
            },
            'prepared_meal': {
                'sandwich': [], 'hot dog': [], 'pizza': []
            },
            'dessert': {
                'donut': [], 'cake': []
            }
        }
    }
}

# --- NEW DATA STRUCTURES ---

# 1. Extended Hierarchy with unique IDs for all categories (original and new)
COCO_EXTENDED_HIERARCHY = {
    0: '__background__', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 
    12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 
    18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 
    24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 
    30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 
    35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 
    39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 
    45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 
    51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 
    57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table', 62: 'toilet', 
    63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 
    69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 
    75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair dryer', 
    80: 'toothbrush', 81: 'object', 82: 'living_thing', 83: 'animal', 
    84: 'domestic_animal', 85: 'livestock_animal', 86: 'wild_animal', 87: 'vehicle', 
    88: 'land_vehicle', 89: 'outdoor_and_sports_object', 90: 'street_fixture', 
    91: 'sports_equipment', 92: 'ball_sport_equipment', 93: 'board_sport_equipment', 
    94: 'other_sports_equipment', 95: 'indoor_object', 96: 'furniture', 
    97: 'appliance_and_electronics', 98: 'appliance', 99: 'electronics', 
    100: 'kitchenware', 101: 'personal_care_item', 102: 'accessory_and_item', 
    103: 'personal_accessory', 104: 'home_and_office_item', 105: 'food', 106: 'fruit', 
    107: 'vegetable', 108: 'prepared_meal', 109: 'dessert'
}

# 2. Child-to-Parent mapping for the extended hierarchy
COCO_EXTENDED_HIERARCHY_CHILD_PARENT_MAP = {
    82: 81, 87: 81, 89: 81, 95: 81, 102: 81, 105: 81, 1: 82, 83: 82, 84: 83, 85: 83, 
    86: 83, 16: 84, 17: 84, 15: 84, 18: 85, 19: 85, 20: 85, 21: 86, 22: 86, 23: 86, 
    24: 86, 88: 87, 5: 87, 9: 87, 2: 88, 3: 88, 4: 88, 6: 88, 7: 88, 8: 88, 90: 89, 
    91: 89, 10: 90, 11: 90, 12: 90, 13: 90, 14: 90, 92: 91, 93: 91, 94: 91, 33: 92, 
    35: 92, 36: 92, 39: 92, 37: 93, 38: 93, 32: 93, 30: 94, 31: 94, 34: 94, 96: 95, 
    97: 95, 100: 95, 101: 95, 57: 96, 58: 96, 59: 96, 60: 96, 61: 96, 98: 97, 99: 97, 
    69: 98, 70: 98, 71: 98, 72: 98, 73: 98, 62: 98, 63: 99, 64: 99, 65: 99, 66: 99, 
    67: 99, 68: 99, 40: 100, 41: 100, 42: 100, 43: 100, 44: 100, 45: 100, 46: 100, 
    79: 101, 80: 101, 103: 102, 104: 102, 25: 103, 26: 103, 27: 103, 28: 103, 29: 103, 
    74: 104, 75: 104, 76: 104, 77: 104, 78: 104, 106: 105, 107: 105, 108: 105, 109: 105, 
    47: 106, 48: 106, 50: 106, 51: 107, 52: 107, 49: 108, 53: 108, 54: 108, 55: 109, 56: 109
}


def preorder_apply(current_node_name: str | None, tree: dict, f: callable, *args):
    """Recursively applies a function to each node of a tree in pre-order.

    Parameters
    ----------
    current_node_name : str or None
        The name of the parent node for the current `tree`. Should be None for
        the initial call on the root.
    tree : dict
        The tree structure, represented as a nested dictionary.
    f : callable
        The function to apply to each child node. It receives the child's name,
        the parent's name, and any additional arguments.
    *args
        Variable length argument list passed on to the callable `f`.
    
    Notes
    -----
    The tree is expected in the format:
    {'root': {'child1': {}, 'child2': {'grandchild1': {}}}}
    Leaf nodes are represented by an empty dictionary or another falsey value.
    The callable `f` should have the signature `f(child_name, parent_name, *args)`.
    """
    if tree:
        for child_name, subtree in tree.items():
            if current_node_name:
                f(child_name, current_node_name, *args)
            preorder_apply(child_name, subtree, f, *args)


def flatten_tree(tree: dict, flattened_tree: dict[str, str] | None = None) -> dict[str, str]:
    """Flattens a nested dictionary tree into a child-parent mapping.

    Parameters
    ----------
    tree : dict
        The nested dictionary to flatten.
    flattened_tree : dict[str, str] or None, optional
        An accumulator for the flattened structure. If None, a new dictionary
        is created. Defaults to None.

    Returns
    -------
    dict[str, str]
        A dictionary where keys are child names and values are their
        corresponding parent names.
    """
    if flattened_tree is None:
        flattened_tree = {}
    preorder_apply(None, tree, flattened_tree.__setitem__)
    return flattened_tree


def replace_flat_dict_strs(flat_dict: dict[str, str], replacement_map: dict[str, int]) -> dict[int, int]:
    """Replaces the string keys and values of a flat dictionary with values from a map.

    Parameters
    ----------
    flat_dict : dict[str, str]
        The flat child-parent dictionary with string names.
    replacement_map : dict[str, int]
        A dictionary used to look up the new values (e.g., integer IDs)
        for the keys and values of `flat_dict`.

    Returns
    -------
    dict[int, int]
        A new dictionary with keys and values replaced according to `replacement_map`.
    """
    return {replacement_map[k]: replacement_map[v] for k, v in flat_dict.items()}


def test_hierarchy():
    """
    Verifies the child-parent map by generating it directly from a depth-first
    traversal of the nested COCO_HIERARCHY.
    """
    # 1. Reverse the ID -> name map for easy name lookup.
    name_to_id = {v: k for k, v in COCO_EXTENDED_HIERARCHY.items()}

    flat_hierarchy = flatten_tree(COCO_HIERARCHY)
    id_hierarchy = replace_flat_dict_strs(flat_hierarchy, name_to_id)

    # --- Verify the reconstruction ---
    # Sort items for a consistent comparison.
    generated_map_items = sorted(id_hierarchy.items())
    existing_map_items = sorted(COCO_EXTENDED_HIERARCHY_CHILD_PARENT_MAP.items())

    assert generated_map_items == existing_map_items, "Generated child-parent map does not match!"
    
    print("Hierarchy test passed successfully!")


class MSCOCOHierarchicalDetectionTrainer(HierarchicalDetectionTrainer):
    # Fix the 0: __background__ misalignment with YOLO categories
    id_to_name = {k-1: v for k, v in COCO_EXTENDED_HIERARCHY.items() if k > 0}
    # Reverse the ID -> name map for easy name lookup
    name_to_id = {v: k for k, v in id_to_name.items()}

    flat_hierarchy = flatten_tree(COCO_HIERARCHY)
    _hierarchy = replace_flat_dict_strs(flat_hierarchy, name_to_id)
     






if __name__ == '__main__':
    # You can run this file to test the hierarchy reconstruction
    test_hierarchy()

