import ultralytics
import torch

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

def build_hierarchy_index_tensor(hierarchy, device=None):
    lens = get_ancestor_chain_lens(hierarchy)
    index_tensor = torch.full((len(lens), max(lens.values())), -1, dtype=torch.int32, device=device)
    preorder_apply(hierarchy, set_indices, index_tensor)
    return index_tensor

def hierarchically_index_flat_scores(flat_scores, target_indices, hierarchy_index_tensor, hierarchy_mask, device=None):
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
    logsigmoids = torch.nn.functional.logsigmoid(hierarchical_predictions) * mask
    summed_logsigmoids = torch.sum(logsigmoids, dim=2)
    log1sigmoids = torch.log1p(-torch.exp(summed_logsigmoids))
    return -(targets * summed_logsigmoids + (1 - targets) * log1sigmoids)

def get_roots(tree):
    ancestor_chain_lens = get_ancestor_chain_lens(tree)
    return [node for node in ancestor_chain_lens if ancestor_chain_lens[node] == 1]

def postprocess_raw_output(raw_yolo_output, hierarchy):
    all_boxes = []
    all_class_scores = []
    _, nms_idxs = ultralytics.utils.ops.non_max_suppression(raw_yolo_output, classes=get_roots(hierarchy), return_idxs=True)
    for i, idx in enumerate(nms_idxs):
        nms_output = raw_yolo_output[i].index_select(1, idx)
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
        optimal_path_scores.append(torch.gather(scores, 0, torch.tensor(paths, device=scores.device).T).T)

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

    raw_output = yolo_raw_predict(model, images, shape, cuda)

    boxes, class_scores = postprocess_raw_output(raw_output, hierarchy)
    
    optimal_paths, optimal_path_scores = optimal_hierarchical_paths(class_scores, hierarchy)

    return boxes, optimal_paths, optimal_path_scores




