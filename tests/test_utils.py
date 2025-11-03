import unittest
import torch

def mock_batchify(tensor):
    return tensor.unsqueeze(0).expand(2,-1,-1)
        

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

    deep7_extended_hierarchy_as_parent_tensor = torch.tensor(
        [ 8, 8, 9, 8,-1, 8, 8, 8, 4, 4], 
        dtype = torch.long
    )

    deep7_extended_hierarchy_as_sibling_mask = torch.tensor([
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0]])


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
        [0.4600, 0.8100, 0.0000],
        [0.2200, 0.0000, 0.0000],
        [0.5100, 0.4600, 0.4800]
    ]


    mock_raw_yolo_output = [
        [[1.3e+02, 5.1e+02, 4.2e+02, 9.9e+01],
         [3.2e+02, 2.7e+02, 5.4e+02, 2.8e+02],
         [1.1e+02, 1.1e+02, 6.2e+01, 1.9e+02],
         [1.3e+02, 8.6e+01, 8.3e+00, 1.9e+02],
         [4.7e-04, 1.3e-03, 6.9e-05, 5.3e-04],
         [2.6e-01, 5.5e-02, 1.7e-04, 4.3e-03],
         [4.6e-01, 1.4e-01, 1.5e-04, 3.5e-03],
         [2.7e-02, 4.6e-02, 1.2e-04, 2.8e-03],
         [1.0e+00, 1.0e+00, 9.7e-04, 5.0e-01],
         [7.9e-01, 5.8e-01, 7.6e-05, 3.4e-04],
         [3.4e-01, 1.3e-01, 1.8e-04, 5.5e-03],
         [5.4e-06, 1.8e-05, 3.6e-05, 9.3e-04]],
        [[1.2e+02, 4.9e+02, 4.1e+02, 1.3e+02],
         [2.9e+02, 2.7e+02, 5.4e+02, 2.7e+02],
         [1.5e+02, 7.0e+01, 4.0e+01, 1.2e+02],
         [1.4e+02, 1.8e+02, 1.7e+01, 9.1e+01],
         [2.1e-03, 6.0e-04, 1.2e-04, 1.3e-03],
         [7.6e-03, 4.8e-03, 3.0e-02, 1.6e-01],
         [3.3e-03, 7.6e-03, 5.1e-02, 1.3e-01],
         [2.2e-03, 5.4e-03, 9.1e-05, 1.8e-02],
         [6.7e-02, 6.8e-01, 1.0e+00, 1.0e+00],
         [9.6e-04, 6.5e-03, 9.8e-01, 4.4e-01],
         [1.5e-03, 1.0e-02, 1.2e-03, 1.4e-01],
         [9.4e-04, 3.0e-04, 1.5e-05, 1.6e-05]]]

    mock_optimal_paths = [[[4, 5], [4, 5], [4, 6]], [[4, 5], [4, 5], [4, 6]]]
    mock_optimal_path_scores = [
        [[1.0000, 0.7900],
         [1.0000, 0.5800],
         [0.5000, 0.0055]],
        [[1.0000, 0.9800],
         [1.0000, 0.4400],
         [0.6800, 0.0100]]
    ]

    sibling_logsumexp = torch.tensor([
        [1.838753, 1.838753, 0.030000, 1.838753, 0.050000, 1.838753, 1.838753, 1.838753, 0.788160, 0.788160],
        [2.588757, 2.588757, 0.880000, 2.588757, 0.810000, 2.588757, 2.588757, 2.588757, 1.199159, 1.199159],
        [2.449941, 2.449941, 0.110000, 2.449941, 0.220000, 2.449941, 2.449941, 2.449941, 1.189659, 1.189659],
        [2.303682, 2.303682, 0.490000, 2.303682, 0.480000, 2.303682, 2.303682, 2.303682, 1.199159, 1.199159]])

   
    ''' TODO! This requires postprocess_raw_output to take in nms parameters.  We generally need to hoist these out to be user-variables 
    def test_hierarchical_paths(self):
        raw_output = torch.tensor(TestHierarchicalIndex.mock_raw_yolo_output)
        hierarchy = hierarchical_yolo.deep7_model.Deep7HierarchicalDetectionTrainer._hierarchy
        boxes, class_scores = hierarchical_yolo.utils.postprocess_raw_output(raw_output, hierarchy)
        optimal_paths, optimal_path_scores = hierarchical_yolo.utils.optimal_hierarchical_paths(class_scores, hierarchy)
        for expected_path, actual_path in zip(TestHierarchicalIndex.mock_optimal_paths, optimal_paths):
            torch.testing.assert_close(expected_path, actual_path)
        for expected_path_scores, actual_path_scores in zip(TestHierarchicalIndex.mock_optimal_path_scores, optimal_path_scores):
            for expected_path_score, actual_path_score in zip(expected_path_scores, actual_path_scores):
                expected_path_score = torch.tensor(expected_path_score, device=raw_output.device)
                torch.testing.assert_close(expected_path_score, actual_path_score)
    '''

if __name__ == '__main__':
    unittest.main()
