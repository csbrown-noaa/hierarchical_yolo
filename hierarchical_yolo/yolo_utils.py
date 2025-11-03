from hierarchical_loss.tree_utils import get_roots
from hierarchical_loss.path_utils import optimal_hierarchical_paths
import ultralytics
import torch

def yolo_raw_predict(model, images, shape, cuda=False):
    import torchvision.transforms as T
    model.eval()

    transform = T.Compose([
        T.Resize(shape),
        T.ToTensor(),
    ])
    input_tensor = torch.stack([transform(img) for img in images])
    if cuda:
        input_tensor = input_tensor.to('cuda')

    with torch.no_grad():
        raw_output = model.model(input_tensor)[0]

    return raw_output

def postprocess_raw_output(raw_yolo_output, hierarchy, nms_iou_thres=None, nms_conf_thres=None):
    nms_iou_thres = nms_iou_thres or 0.7
    nms_conf_thres = nms_conf_thres or 0.25
    all_boxes = []
    all_class_scores = []
    # TODO: cache this
    roots = get_roots(hierarchy)
    # TODO: hoist iou_thres etc to a higher call
    _, nms_idxs = ultralytics.utils.ops.non_max_suppression(raw_yolo_output, classes=roots, return_idxs=True, iou_thres=nms_iou_thres, conf_thres=nms_conf_thres)
    for i, idx in enumerate(nms_idxs):
        flat_idx = idx.flatten()
        nms_output = raw_yolo_output[i].index_select(1, flat_idx)
        boxes = nms_output[:4, :]
        class_scores = nms_output[4:, :]
        all_boxes.append(boxes)
        all_class_scores.append(class_scores)
    return all_boxes, all_class_scores

def hierarchical_predict(model, hierarchy, images, shape=(640,640), cuda=False, nms_iou_thres=None, nms_conf_thres=None):
    """
    Performs hierarchical prediction on a batch of images using a trained YOLO model.

    This function takes a standard YOLO model trained on a flat set of classes
    and applies a hierarchical structure to its raw output. It processes the
    model's predictions to find the optimal path through the provided hierarchy
    for each detected bounding box.

    Parameters
    ----------
    model : ultralytics.YOLO
        An evaluated YOLO model instance trained on a flat class structure.
    hierarchy : dict
        A dictionary representing the class hierarchy, with keys as child class
        IDs and values as parent class IDs (e.g., {child_id: parent_id}).
    images : list or torch.Tensor
        A batch of images to perform prediction on. Images should be in a
        format compatible with the YOLO model, such as PIL.Image objects
        converted to RGB.
    shape : tuple, optional
        The input image shape (height, width) for the model, by default (640, 640).
    cuda : bool, optional
        If True, the model will be run on a CUDA-enabled GPU, by default False.

    Returns
    -------
    tuple
        A tuple containing three elements:
        - boxes (list): A list of tensors, where each tensor contains the
          bounding boxes detected in the corresponding input image.
        - optimal_paths (list): A list where each element corresponds to an
          image and contains the most likely hierarchical class path for each
          detected bounding box.
        - optimal_path_scores (list): A list containing the scores associated
          with each optimal path.
    """

    raw_output = yolo_raw_predict(model, images, shape, cuda)

    boxes, pred_scoreses = postprocess_raw_output(raw_output, hierarchy, nms_iou_thres=nms_iou_thres, nms_conf_thres=nms_conf_thres)

    optimal_paths, optimal_path_scores = optimal_hierarchical_paths(pred_scoreses, hierarchy)

    return boxes, optimal_paths, optimal_path_scores