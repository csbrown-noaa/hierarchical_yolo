from hierarchical_loss.tree_utils import get_roots
from hierarchical_loss.path_utils import optimal_hierarchical_paths
import ultralytics
try:
    from ultralytics.utils.ops import non_max_suppression # old version
except:
    from ultralytics.utils.nms import non_max_suppression # new version
import torch
from PIL import Image
import yaml

from torchvision.ops import batched_nms
from hierarchical_loss.hierarchy_tensor_utils import conditional_to_marginal


def yolo_raw_predict(
    model: ultralytics.YOLO,
    images: list[Image.Image],
    shape: tuple[int, int]
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
    input_tensor = torch.stack([transform(img) for img in images]).to(model.device)

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
    raw_output = yolo_raw_predict(model, images, shape)

    boxes, pred_scoreses = postprocess_raw_output(raw_output, hierarchy, nms_iou_thres=nms_iou_thres, nms_conf_thres=nms_conf_thres)

    optimal_paths, optimal_path_scores = optimal_hierarchical_paths(pred_scoreses, hierarchy)

    return boxes, optimal_paths, optimal_path_scores


def get_yolo_class_names(yaml_file) -> dict[int, str]:
    """Reads the class names from an Ultralytics YAML file.

    This function opens and parses a YAML file, expecting to find a
    'names' key. This key can be a dictionary mapping integer class IDs
    to string class names, or a list of class names where the index
    is the class ID.

    Parameters
    ----------
    yaml_file : file-like
        The opened YAML file (e.g., from `open(path)` or `io.StringIO`).

    Returns
    -------
    dict[int, str]
        A dictionary where keys are integer class IDs and values are the
        corresponding class name strings.

    Raises
    ------
    ValueError
        If the 'names' key is not found in the YAML file.
    TypeError
        If the 'names' key is not a list or a dictionary.

    Examples
    --------
    >>> import yaml
    >>> from io import StringIO
    >>>
    >>> # Mock a YAML file with 'names' as a dictionary
    >>> mock_yaml_content_dict = '''
    ... path: ../datasets/coco128
    ... names:
    ...   0: person
    ...   1: bicycle
    ...   2: car
    ... '''
    >>> mock_file_dict = StringIO(mock_yaml_content_dict)
    >>> class_dict = get_yolo_class_names(mock_file_dict)
    >>> print(class_dict)
    {0: 'person', 1: 'bicycle', 2: 'car'}
    """
    data = yaml.safe_load(yaml_file)
    if 'names' in data and isinstance(data['names'], dict):
        # Ensure keys are integers, as YAML might parse them as strings
        # in some cases, although typically not for this format.
        class_names = {int(k): v for k, v in data['names'].items()}
        return class_names

def soft_hierarchical_predictions(
    model: ultralytics.YOLO,
    hierarchy: dict[int, int],
    images: list[Image.Image],
    shape: tuple[int, int] = (640, 640),
    nms_iou_thres: float | None = None,
    nms_conf_thres: float | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Runs a batch of images through the model and applies class-agnostic NMS
    anchored at the hierarchical roots. 
    
    Returns spatial bounding boxes and the full unadulterated C-length soft 
    score vectors (conditional probabilities) for every surviving box.
    """
    raw_output = yolo_raw_predict(model, images, shape)
    boxes, pred_scores = postprocess_raw_output(
        raw_output, hierarchy, nms_iou_thres=nms_iou_thres, nms_conf_thres=nms_conf_thres
    )
    return boxes, pred_scores


def serialize_soft_hierarchical_predictions(
    boxes_batch: list[torch.Tensor],
    scores_batch: list[torch.Tensor],
    images_list: list[dict],
    idx_to_node: dict[int, str],
    hierarchy_roots: list[int],
    input_shape: tuple[int, int] = (640, 640)
) -> dict:
    """
    Serializes soft hierarchical predictions into an extended COCO JSON dictionary.
    
    Scales the YOLO (e.g., 640x640) coordinates back to native image dimensions 
    and attaches the full soft score vector to each annotation for downstream 
    dynamic filtering in a web viewer.
    """
    categories = [
        {'name': idx_to_node[i], 'id': i}
        for i in sorted(idx_to_node.keys())
    ]
    
    annotations = []
    ann_idx = 0
    
    # Determine the fallback root ID for annotations (e.g., 'Biota' ID)
    root_category_id = hierarchy_roots[0] if hierarchy_roots else 0
    
    for boxes, scores, img_meta in zip(boxes_batch, scores_batch, images_list):
        scale_w = img_meta['width'] / input_shape[1]
        scale_h = img_meta['height'] / input_shape[0]
        
        if boxes.numel() == 0:
            continue
            
        # Iterate over the N filtered boxes and scores for this image
        for box, score_vec in zip(boxes.T, scores.T):
            x_min = int(box[0].item() * scale_w)
            y_min = int(box[1].item() * scale_h)
            w = int((box[2].item() - box[0].item()) * scale_w)
            h = int((box[3].item() - box[1].item()) * scale_h)
            
            annotations.append({
                'id': ann_idx,
                'image_id': img_meta['id'],
                'bbox': [x_min, y_min, w, h],
                'area': w * h,
                'category_id': root_category_id,
                'scores': [f'{x.item():.5f}' for x in score_vec]
            })
            ann_idx += 1
            
    return {
        'images': images_list, # Directly passes through the pre-patched unified dictionary
        'annotations': annotations,
        'categories': categories
    }

def conditionals_to_marginals(
    preds: torch.Tensor,
    hierarchy_index_tensor: torch.Tensor,
    eval_subset_ids: list[int] | set[int] | torch.Tensor | None = None
) -> torch.Tensor:
    """
    YOLO-specific wrapper to convert conditional probabilities to marginal probabilities.

    This function intercepts the Ultralytics tensor shapes (typically `[B, 4+C, Detections]`),
    extracts the class probabilities, computes the hierarchical marginals down the 
    phylogenetic tree, and optionally masks out specific categories for subset evaluation.

    Parameters
    ----------
    preds : torch.Tensor
        The raw predictions from the YOLO `Detect` head. Shape `[B, 4+C, Detections]`.
    hierarchy_index_tensor : torch.Tensor
        An int tensor of shape `[N, M]` encoding the hierarchy, where `N` is the
        number of classes and `M` is the maximum hierarchy depth.
    eval_subset_ids : list[int] | set[int] | torch.Tensor | None, optional
        A collection of category IDs to evaluate. If provided, all categories *not* in this subset will have their marginal probabilities zeroed out before NMS.
        By default None (evaluates all classes).

    Returns
    -------
    torch.Tensor
        The modified prediction tensor where the conditional 
        probabilities have been replaced by the computed (and optionally masked) marginals.
    """
    # Extract Class Probabilities: [B, 4 + C, Detections] -> [B, Detections, C]
    cls_probs = preds[:, 4:, :].transpose(1, 2)
    
    # Apply Hierarchical Math (Conditional -> Marginal)
    marginal_probs = conditional_to_marginal(cls_probs, hierarchy_index_tensor)
    
    # Optional Subsetting
    if eval_subset_ids is not None:
        if not isinstance(eval_subset_ids, torch.Tensor):
            eval_subset_ids = torch.tensor(list(eval_subset_ids), device=marginal_probs.device)
            
        mask = torch.ones(marginal_probs.shape[-1], dtype=torch.bool, device=marginal_probs.device)
        mask[eval_subset_ids] = False
        marginal_probs[..., mask] = 0.0
        
    # Pack it back up for Ultralytics NMS
    preds[:, 4:, :] = marginal_probs.transpose(1, 2)
    
    return preds
