import unittest
import utils
import hierarchical_yolo.utils
import torch

class TestIndexTensor(unittest.TestCase):

    '''
    deep7 extended categories:

      0: Kalekale
      1: Opakapaka
      2: "Hapu\u02BBupu\u02BBu"
      3: Gindai
      4: Other or Can't Tell
      5: Ehu
      6: Lehi
      7: Onaga
      8: Snapper
      9: Grouper
    '''

    deep7_extended_hierarchy = {
        0: 8,
        1: 8,
        2: 9,
        3: 8,
        5: 8,
        6: 8,
        7: 8,
        8: 4,
        9: 4
    }

    deep7_extended_hierarchy_as_index_tensor = torch.tensor([
        [ 0,  8,  4],
        [ 1,  8,  4],
        [ 2,  9,  4],
        [ 3,  8,  4],
        [ 4, -1, -1],
        [ 5,  8,  4],
        [ 6,  8,  4],
        [ 7,  8,  4],
        [ 8,  4, -1],
        [ 9,  4, -1]], dtype=torch.int32)

    def test_index_tensor(self):
        index_tensor = hierarchical_yolo.utils.build_hierarchy_index_tensor(TestIndexTensor.deep7_extended_hierarchy)
        torch.testing.assert_close(index_tensor, TestIndexTensor.deep7_extended_hierarchy_as_index_tensor)

class TestHierarchicalIndex(unittest.TestCase):

    flat_scores = [
        [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10],
        [0.91,0.90,0.88,0.85,0.81,0.76,0.70,0.63,0.55,0.46],
        [0.00,0.99,0.11,0.88,0.22,0.77,0.33,0.66,0.55,0.44],
        [0.50,0.51,0.49,0.52,0.48,0.53,0.47,0.54,0.46,0.55]
    ]

    target_scores = [
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0.5],
        [0,0,0,0,0.3,0,0,0,0,0],
        [0,0.2,0,0,0,0,0,0,0,0]
    ]

    masked_hierarchical_scores = [
        [0.0100, 0.0900, 0.0500],
        [0.4600, 0.8100, 1.0000],
        [0.2200, 1.0000, 1.0000],
        [0.5100, 0.4600, 0.4800]
    ]


    def test_hierarchical_index_flat_scores(self):
        hierarchy_index_tensor = TestIndexTensor.deep7_extended_hierarchy_as_index_tensor
        hierarchy_mask = hierarchy_index_tensor == -1
        target_scores = torch.tensor(TestHierarchicalIndex.target_scores).unsqueeze(0)
        pred_scores = torch.tensor(TestHierarchicalIndex.flat_scores).unsqueeze(0)
        target_indices = torch.argmax(target_scores, dim=2)
        masked_hierarchical_scores = hierarchical_yolo.utils.hierarchically_index_flat_scores(pred_scores, target_indices, hierarchy_index_tensor, hierarchy_mask)
        torch.testing.assert_close(masked_hierarchical_scores, torch.tensor(TestHierarchicalIndex.masked_hierarchical_scores).unsqueeze(0))
        

def load_tests(loader, tests, ignore):
    return utils.doctests(hierarchical_yolo.utils, tests)

if __name__ == '__main__':
    unittest.main()
