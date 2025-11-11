from hierarchical_loss.tree_utils import get_roots
from hierarchical_loss.path_utils import optimal_hierarchical_paths
import ultralytics
try:
    from ultralytics.utils.ops import non_max_suppression # old version
except:
    from ultralytics.utils.nms import non_max_suppression # new version
import torch
from PIL import Image

def yolo_raw_predict(
    model: ultralytics.YOLO,
    images: list[Image.Image],
    shape: tuple[int, int],
    cuda: bool = False,
) -> torch.Tensor:
    """Runs a batch of images through a YOLO model's raw backbone.

    This function bypasses the standard `.predict()` method to get the
    raw output tensor from the model's backbone. It handles preprocessing
    (resize, to-tensor) and moves the input tensor to the GPU if specified.

    Parameters
    ----------
    model : ultralytics.YOLO
        An evaluated YOLO model instance.
    images : List[Image.Image]
        A list of PIL.Image.Image objects to be processed.
    shape : Tuple[int, int]
        The target (height, width) to resize images to for the model input.
    cuda : bool, optional
        If True, moves the input tensor to 'cuda'. By default False.

    Returns
    -------
    torch.Tensor
        The raw output tensor from the model's backbone.
    """
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


def postprocess_raw_output(
    raw_yolo_output: torch.Tensor,
    hierarchy: dict[int, int],
    nms_iou_thres: float | None = None,
    nms_conf_thres: float | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Applies Non-Max Suppression (NMS) to raw YOLO output.

    This function passes the `classes=roots` argument to the Ultralytics
    `non_max_suppression` function. This internally filters detections
    by finding the single highest-scoring class for each box, and then
    keeping only those boxes where that single highest-scoring class
    is one of the `roots`.

    It then separates the boxes and the full class score vectors for the
    surviving detections.

    .. todo::
       The current logic is flawed for hierarchical models. `non_max_suppression`
       finds the `argmax` over *all* classes, and only THEN does the `classes`
       argument do anything. But, for hierarchical conditional
       probabilities, the single max score is not meaningful. The intended
       logic (to be fixed in a future branch) is to run NMS *only* on the
       subset of root class scores, not to filter by the global argmax.

    Parameters
    ----------
    raw_yolo_output : torch.Tensor
        The raw output tensor from the YOLO model, expected in
        `(B, C+4, N)` format, where B=batch, C=classes, N=proposals.
    hierarchy : dict[int, int]
        The class hierarchy in `{child_id: parent_id}` format. This is
        used to find the root classes.
    nms_iou_thres : float, optional
        IoU threshold for NMS. By default 0.7.
    nms_conf_thres : float, optional
        Confidence threshold for NMS. By default 0.25.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor]]
        A tuple containing two lists, one for boxes and one for scores,
        with one entry per batch item:
        1. `all_boxes`: List[torch.Tensor], where each tensor is shape
           `(4, N_filtered)` containing bounding boxes.
        2. `all_class_scores`: List[torch.Tensor], where each tensor is
           shape `(C, N_filtered)` containing all class scores for the
           surviving boxes.
    """
    nms_iou_thres = nms_iou_thres or 0.7
    nms_conf_thres = nms_conf_thres or 0.25
    all_boxes = []
    all_class_scores = []
    # TODO: cache this
    roots = get_roots(hierarchy)
    # TODO: this logic is wrong.  We need to subset raw_yolo_output to root classes instead of using the `classes` arg
    _, nms_idxs = non_max_suppression(raw_yolo_output, classes=roots, return_idxs=True, iou_thres=nms_iou_thres, conf_thres=nms_conf_thres, multi_label=True)
    for i, idx in enumerate(nms_idxs):
        flat_idx = idx.flatten().long()
        nms_output = raw_yolo_output[i].index_select(1, flat_idx)
        boxes = nms_output[:4, :]
        class_scores = nms_output[4:, :]
        all_boxes.append(boxes)
        all_class_scores.append(class_scores)
    return all_boxes, all_class_scores

def hierarchical_predict(
    model: ultralytics.YOLO,
    hierarchy: dict[int, int],
    images: list[Image.Image],
    shape: tuple[int, int] = (640, 640),
    cuda: bool = False,
    nms_iou_thres: float | None = None,
    nms_conf_thres: float | None = None,
) -> tuple[list[torch.Tensor], list[list[list[int]]], list[list[torch.Tensor]]]:
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
    hierarchy : dict[int, int]
        A dictionary representing the class hierarchy, with keys as child class
        IDs and values as parent class IDs (e.g., {child_id: parent_id}).
    images : list[Image.Image]
        A batch of images to perform prediction on, as PIL.Image objects.
    shape : tuple[int, int], optional
        The input image shape (height, width) for the model, by default (640, 640).
    cuda : bool, optional
        If True, the model will be run on a CUDA-enabled GPU, by default False.
    nms_iou_thres : float, optional
        IoU threshold for NMS. Passed to `postprocess_raw_output`.
    nms_conf_thres : float, optional
        Confidence threshold for NMS. Passed to `postprocess_raw_output`.

    Returns
    -------
    tuple[list[torch.Tensor], list[list[list[int]]], list[list[torch.Tensor]]]
        A tuple containing three elements:
        1. `boxes` (list[torch.Tensor]): A list of tensors (one per image),
           where each tensor is shape `(4, N_filtered)` and contains the
           bounding boxes detected in that image.
        2. `optimal_paths` (list[list[list[int]]]): A nested list
           `[batch][detection][path_node]` containing the most likely
           hierarchical class path for each detected bounding box.
        3. `optimal_path_scores` (list[list[torch.Tensor]]): A nested list
           `[batch][detection]` containing 1D tensors of scores
           associated with each optimal path.
    """
    raw_output = yolo_raw_predict(model, images, shape, cuda)

    boxes, pred_scoreses = postprocess_raw_output(raw_output, hierarchy, nms_iou_thres=nms_iou_thres, nms_conf_thres=nms_conf_thres)

    optimal_paths, optimal_path_scores = optimal_hierarchical_paths(pred_scoreses, hierarchy)

    return boxes, optimal_paths, optimal_path_scores