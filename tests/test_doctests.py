import utils
import functools
import unittest

import hierarchical_yolo.hierarchical_detection
import hierarchical_yolo.yolo_utils

MODULES = [
    hierarchical_yolo.hierarchical_detection,
    hierarchical_yolo.yolo_utils
]

# this is just here so that unittest discover will run this file
class TestEmpty(unittest.TestCase):
    def test_empty(self):
        pass

def load_tests(loader, tests, ignore):
    return functools.reduce(lambda tests_so_far, module: utils.doctests(module, tests_so_far), MODULES, tests)

if __name__ == '__main__':
    unittest.main()
