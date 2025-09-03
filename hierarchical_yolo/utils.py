import ultralytics
import torch
from typing import Callable

def log_matrix(m):
    formatted_lines = []
    for i in range(m.shape[0]):
        vec = m[i]
        line = f"{i:04d}: " + ", ".join(f"{x:.4f}" for x in vec.tolist())
        formatted_lines.append(line)
    ultralytics.utils.LOGGER.info("\n".join(formatted_lines))

def tree_walk(tree, node):
    yield node
    while node in tree:
        node = tree[node]
        yield node

def preorder_apply(tree, f, *args):
    visited = {}
    for node in tree:
        path = [node]
        while (node in tree) and (node not in visited):
            node = tree[node]
            path.append(node)
        if node not in visited:
            visited[node] = f(node, None, *args)
        for i in range(-2, -len(path) - 1, -1):
            visited[path[i]] = f(path[i], visited[path[i+1]], *args)
    return visited

def truncate_path(path, score, threshold = 0.3):
    truncated_path, truncated_score = [], []
    for category, p in zip(path, score):
        if p < threshold:
            break
        truncated_path.append(category)
    return truncated_path, score[:len(truncated_path)]

def increment_chain_len(_, parent_chain_len):
    if not parent_chain_len: return 1
    return parent_chain_len + 1

def get_ancestor_chain_lens(tree: dict) -> dict:
    '''
    Get lengths of ancestor chains in a { child: parent } dictionary tree

    Examples
    --------
    >>> get_ancestor_chain_lens({ 0:1, 1:2, 2:3, 4:5, 5:6, 7:8 })
    {3: 1, 2: 2, 1: 3, 0: 4, 6: 1, 5: 2, 4: 3, 8: 1, 7: 2}

    Parameters
    ----------
    tree: dict
        A tree in { child: parent } format.

    Returns
    -------
    lengths: dict
        The lengths of the path to the root from each node { node: length }

    '''
    return preorder_apply(tree, increment_chain_len)

def set_indices(index, parent_index, tensor):
    tensor[index, 0] = index
    if parent_index is not None:
        tensor[index, 1:] = tensor[parent_index, :-1]
    return index

def build_parent_tensor(tree, device=None):
    '''
    Converts a tree in dict format to a 1D tensor

    Parameters
    ----------
    tree: dict
        A tree in { child: parent } format.  Must use ids starting at 0 and running to C-1 where C is the number of categories.

    Returns
    -------
    parent_tensor: tensor (C)
        A tensor where each index has the index of its parent, or -1 if it is a root
    '''
    nodes = set(tree.keys()) | set(tree.values())
    C = max(nodes) + 1

    parent_tensor = torch.full((C,), -1, dtype=torch.long, device=device)

    for child, parent in tree.items():
        parent_tensor[child] = parent

    return parent_tensor

def build_hierarchy_sibling_mask(parent_tensor, device=None):
    '''
    Build a sibling mask (CxG) for use with logsumexp_over_siblings.
    
    Parameters
    ----------
    parent_tensor: tensor (C)
        Each node's parent index, -1 for root nodes.

    Returns
    -------
    sibling_mask: tensor (CxG)
        Boolean mask where row i has True in columns corresponding to sibling groups it belongs to.
        G = number of sibling groups (including a root group)
    '''
    C = parent_tensor.shape[0]

    # Identify all unique parents (groups), including -1 for roots
    unique_parents, inverse_indices = torch.unique(parent_tensor, return_inverse=True)
    G = len(unique_parents)

    # Map parent index -> column in sibling_mask
    parent_to_group = {p.item(): g for g, p in enumerate(unique_parents)}

    sibling_mask = torch.zeros(C, G, dtype=torch.bool, device=device)

    # Assign each node to the column of its parent group
    sibling_mask[torch.arange(C), inverse_indices] = True

    return sibling_mask


def logsumexp_over_siblings(flat_scores, sibling_mask):
    '''
    Parameters
    ----------
    flat_scores: tensor (BxC)
        raw scores for each category, batch-wise
    sibling_mask: tensor (CxG)
        a mask where sibling_mask[i,j] == sibling_mask[k,j] == 1 iff i and k are siblings

    Returns
    -------
    logsumexp: tensor (BxC)
        the logsumexp over all of the siblings of each category.  logsumexp[i,j] == logsumexp[i,k] if j,k are siblings.
    '''

    B, C = flat_scores.shape
    G = sibling_mask.shape[1]
    scores_expanded = flat_scores.unsqueeze(1)  # (B, 1, C)
    masked_scores = scores_expanded + torch.log(sibling_mask.T.unsqueeze(0))  # (B, G, C)
    logsumexp_by_group = torch.logsumexp(masked_scores, dim=-1)  # (B, G)
    zerod_logsumexp = logsumexp_by_group.masked_fill(torch.isinf(logsumexp_by_group), 9)
    logsumexp = (sibling_mask * zerod_logsumexp.unsqueeze(1)).sum(dim=-1)  # (B, C)

    return logsumexp


def build_hierarchy_index_tensor(hierarchy, device=None):
    '''
    Translate a hierarchy into a tensor representation.  Parent node ids are to the right of a node.  -1 is always to the right of roots or -1s.

    Example:
    >>> hierarchy = {0:1, 1:2, 3:4}
    >>> build_hierarchy_index_tensor(hierarchy)
    tensor([[ 0,  1,  2],
            [ 1,  2, -1],
            [ 2, -1, -1],
            [ 3,  4, -1],
            [ 4, -1, -1]], dtype=torch.int32)
    '''
    lens = get_ancestor_chain_lens(hierarchy)
    index_tensor = torch.full((len(lens), max(lens.values())), -1, dtype=torch.int32, device=device)
    preorder_apply(hierarchy, set_indices, index_tensor)
    return index_tensor


def accumulate_hierarchy(
    predictions: torch.Tensor,
    hierarchy_index: torch.Tensor,
    cumulative_op: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Performs a cumulative operation along a hierarchical structure.

    This function applies a cumulative operation (e.g., `torch.cumsum`) along
    each ancestral path in a hierarchy. The implementation is fully vectorized,
    avoiding Python loops for performance. It first gathers the initial values
    for all nodes along each path, applies the cumulative operation, and then
    selects the final accumulated value for each node.

    For associative operations like `torch.cumsum` and `torch.cumprod`, this
    produces the same result as a level-by-level iterative approach.

    Parameters
    ----------
    predictions : torch.Tensor
        A tensor of shape `[B, D, N]`, where `B` is the batch size, `D` is the
        number of detections, and `N` is the number of classes.
    hierarchy_index : torch.Tensor
        An int tensor of shape `[N, M]` encoding the hierarchy, where `N` is the
        number of classes and `M` is the maximum hierarchy depth. Each row `i`
        contains the path from node `i` to its root. Parent node IDs are to
        the right of child node IDs. A value of -1 is used for padding.
    cumulative_op : callable
        A function that performs a cumulative operation along a dimension,
        such as `torch.cumsum` or `torch.cumprod`. It must accept a tensor
        and a `dim` argument.

    Returns
    -------
    torch.Tensor
        A new tensor with the same shape as `predictions` containing the
        accumulated values.

    Examples
    --------
    >>> import torch
    >>> hierarchy_index = torch.tensor([
    ...     [ 0,  1,  2],
    ...     [ 1,  2, -1],
    ...     [ 2, -1, -1],
    ...     [ 3,  4, -1],
    ...     [ 4, -1, -1]
    ... ], dtype=torch.int64)
    >>> # Predictions for 5 classes: [0., 10., 20., 30., 40.]
    >>> predictions = torch.arange(0, 50, 10, dtype=torch.float32).view(1, 1, 5)
    >>> # Perform a cumulative sum
    >>> cumsum_preds = accumulate_hierarchy(predictions, hierarchy_index, torch.cumsum)
    >>> print(cumsum_preds.squeeze())
    tensor([30., 30., 20., 70., 40.])
    """
    B, D, N = predictions.shape
    M = hierarchy_index.shape[1]

    # 1. GATHER: Collect prediction values for each node in each path.
    # Create a mask for valid indices (non -1)
    valid_mask = hierarchy_index != -1

    # Create a "safe" index tensor to prevent out-of-bounds errors from -1.
    # We replace -1 with a valid index (e.g., 0) and will zero out its
    # contribution later using the mask.
    safe_index = hierarchy_index.masked_fill(~valid_mask, 0)

    # Use advanced indexing to gather values. `predictions[:, :, safe_index]`
    # creates a tensor of shape [B, D, N, M].
    path_values = predictions[:, :, safe_index]

    # Zero out the values from padded (-1) indices.
    path_values = path_values * valid_mask.to(path_values.dtype)

    # 2. ACCUMULATE: Apply the cumulative operation along the path dimension.
    accumulated_paths = cumulative_op(path_values, -1)

    # 3. SELECT: The final value for each node is the last valid accumulated
    # value in its path.
    # Find the length of each path to get the index of the last valid element.
    path_lengths = valid_mask.sum(dim=1)
    end_indices = path_lengths - 1

    # Reshape indices to be compatible with torch.gather.
    end_indices = end_indices.view(1, 1, N, 1).expand(B, D, -1, -1)

    # Gather the final accumulated value for each node from its path.
    final_values = torch.gather(accumulated_paths, -1, end_indices).squeeze(-1)

    return final_values





def hierarchically_index_flat_scores(flat_scores, target_indices, hierarchy_index_tensor, hierarchy_mask, device=None):
    '''
    Takes a vector of "flat" scores over the entire category space, and extracts a vector of scores over only the branch indicated by the target index.
    Viz. If the target_index is 4, and 4's hierarchy goes 4,8,-1,-1, then this will extract flat_score[4] and flat_score[8].  The rest are padded out.
    '''

    batch_size, n_proposals, n_categories = flat_scores.shape
    hierarchy_size = hierarchy_index_tensor.shape[1]

    # Expand hierarchy_index_tensor and mask to shape (B, N, H)
    hierarchy_indices = hierarchy_index_tensor[target_indices]
    flat_mask = hierarchy_mask[target_indices]

    # Construct batch indices
    batch_indices = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand(batch_size, n_proposals, hierarchy_size) # (B, N, H)
    proposal_indices = torch.arange(n_proposals, device=device).view(1, n_proposals, 1).expand(batch_size, n_proposals, hierarchy_size) # (B, N, H)

    # Now index into flat_scores[b, n, c] with b, n, c from the tensors
    gathered_scores = flat_scores[batch_indices, proposal_indices, hierarchy_indices]  # (B, N, H)

    # Mask out invalid entries
    masked_scores = gathered_scores.masked_fill(flat_mask, 0.)

    return masked_scores


def hierarchical_loss(hierarchical_predictions, targets, mask):
    ''' TODO: This is broken!  We have to softmax over SIBLINGS, not the hierarchy! '''
    logsigmoids = torch.nn.functional.logsigmoid(hierarchical_predictions) * mask
    #ultralytics.utils.LOGGER.info(logsigmoids.shape)
    #ultralytics.utils.LOGGER.info(logsigmoids)
    summed_logsigmoids = torch.sum(logsigmoids, dim=2)
    #ultralytics.utils.LOGGER.info(summed_logsigmoids.shape)
    #ultralytics.utils.LOGGER.info(summed_logsigmoids)
    exp_summed_logsigmoids = torch.exp(summed_logsigmoids)
    #ultralytics.utils.LOGGER.info(exp_summed_logsigmoids.shape)
    #ultralytics.utils.LOGGER.info(exp_summed_logsigmoids)
    log1sigmoids = torch.log1p(-exp_summed_logsigmoids)
    #ultralytics.utils.LOGGER.info(log1sigmoids.shape)
    #ultralytics.utils.LOGGER.info(log1sigmoids)
    return -(targets * summed_logsigmoids + (1 - targets) * log1sigmoids)

def get_roots(tree):
    ancestor_chain_lens = get_ancestor_chain_lens(tree)
    return [node for node in ancestor_chain_lens if ancestor_chain_lens[node] == 1]

def postprocess_raw_output(raw_yolo_output, hierarchy):
    all_boxes = []
    all_class_scores = []
    _, nms_idxs = ultralytics.utils.ops.non_max_suppression(raw_yolo_output, classes=get_roots(hierarchy), return_idxs=True, iou_thres=0.8)
    for i, idx in enumerate(nms_idxs):
        #print(idx)
        try:
            flat_idx = idx.flatten()
            nms_output = raw_yolo_output[i].index_select(1, flat_idx)
        except Exception as e:
            print(idx)
            print(nms_idxs)
            print(raw_yolo_output)
            raise e
        #print('pickle')
        #print(nms_output.shape)
        boxes = nms_output[:4, :]
        class_scores = nms_output[4:, :]
        all_boxes.append(boxes)
        all_class_scores.append(class_scores)
    return all_boxes, all_class_scores

def mul_by_index(index, val, mat):
    if val is not None:
        mat[index, :] *= val
    return mat[index, :]

def get_marginal_confidences(confidences, hierarchy):
    new_confidences = []
    for confidence in confidences:
        confidence = torch.clone(confidence)
        preorder_apply(hierarchy, mul_by_index, confidence)
        new_confidences.append(confidence)
    return new_confidences

def append_to_parentchild_tree(node, ancestral_chain, parentchild_tree):
    ancestral_chain = ancestral_chain or []
    for parent in ancestral_chain:
        parentchild_tree = parentchild_tree[parent]
    if node not in parentchild_tree:
        parentchild_tree[node] = {}
    return ancestral_chain + [node]

def invert_childparent_tree(tree):
    '''
    inverts a {child: parent} tree into a {root: tree} tree
    '''
    parentchild_tree = {}
    preorder_apply(tree, append_to_parentchild_tree, parentchild_tree)
    return parentchild_tree

def get_optimal_ancestral_chain(confidences, hierarchy):
    '''
    TODO: This method is very loopy.  It wasn't immediately clear how to tensorify these operations.
    '''
    inverted_tree = invert_childparent_tree(hierarchy)
    bpaths = []
    for b, confidence in enumerate(confidences):
        paths = []
        for i in range(confidence.shape[1]):
            confidence_row = confidence[..., i]
            path = []
            path_tree = inverted_tree
            while path_tree:
                parents = list(path_tree.keys())
                best = confidence_row.index_select(0, torch.tensor(parents, device=confidence.device)).argmax()
                path.append(parents[best])
                path_tree = path_tree[parents[best]]
            paths.append(path)
        bpaths.append(paths)
    return bpaths

def optimal_hierarchical_paths(class_scores, hierarchy):
    optimal_paths = get_optimal_ancestral_chain(class_scores, hierarchy)

    optimal_path_scores = []
    for scores, paths in zip(class_scores, optimal_paths):
        optimal_path_score = []
        for score, path in zip(scores.T, paths):
            optimal_path_score.append(torch.gather(score, 0, torch.tensor(path, device=scores.device)))
        optimal_path_scores.append(optimal_path_score)
    return optimal_paths, optimal_path_scores

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

def hierarchical_predict(model, hierarchy, images, shape=(640,640), cuda=False):

    #TODO cache this
    parent_tensor = build_parent_tensor(hierarchy)
    sibling_mask = build_hierarchy_sibling_mask(parent_tensor)
    raw_output = yolo_raw_predict(model, images, shape, cuda)

    #print(raw_output.shape)
    #print(raw_output[:,:,20])
    boxes, pred_scoreses = postprocess_raw_output(raw_output, hierarchy)
    #print(boxes[0].shape)
    #print(pred_scoreses[0].shape)
    '''
    scores = []
    for flat_pred in pred_scoreses:
        #print(flat_pred.shape) 
        #flat_pred = pred_scores.view(pred_scores.shape[0] * pred_scores.shape[1], pred_scores.shape[2])
        sibling_normalized_flat_pred = flat_pred - logsumexp_over_siblings(flat_pred, sibling_mask)
        #sibling_normalized_pred_scores = sibling_normalized_flat_pred.view(*pred_scores.shape)
        scores.append(sibling_normalized_flat_pred)
    '''    

    optimal_paths, optimal_path_scores = optimal_hierarchical_paths(pred_scoreses, hierarchy)

    return boxes, optimal_paths, optimal_path_scores




