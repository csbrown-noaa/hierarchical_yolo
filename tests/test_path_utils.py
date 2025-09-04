import unittest
import utils
import hierarchical_yolo.path_utils

def load_tests(loader, tests, ignore):
    return utils.doctests(hierarchical_yolo.path_utils, tests)

if __name__ == '__main__':
    unittest.main()
