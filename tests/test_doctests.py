import utils
import functools
import unittest

import hierarchical_yolo.utils
import hierarchical_yolo.coco_utils
import hierarchical_yolo.hierarchical_detection
import hierarchical_yolo.hierarchical_loss
import hierarchical_yolo.hierarchy_tensor_utils
import hierarchical_yolo.path_utils
import hierarchical_yolo.tree_utils
import hierarchical_yolo.viz_utils
import hierarchical_yolo.worms_utils
import hierarchical_yolo.yolo_utils

MODULES = [
    hierarchical_yolo.utils,
    hierarchical_yolo.coco_utils,
    hierarchical_yolo.hierarchical_detection,
    hierarchical_yolo.hierarchical_loss,
    hierarchical_yolo.hierarchy_tensor_utils,
    hierarchical_yolo.path_utils,
    hierarchical_yolo.tree_utils,
    hierarchical_yolo.viz_utils,
    hierarchical_yolo.worms_utils,
    hierarchical_yolo.yolo_utils
]

class TestEmpty(unittest.TestCase):
    def test_empty(self):
        pass

def load_tests(loader, tests, ignore):
    return utils.doctests(hierarchical_yolo.utils, tests)
    #return functools.reduce(lambda tests_so_far, module: utils.doctests(module, tests_so_far), MODULES, tests)

if __name__ == '__main__':
    unittest.main()
