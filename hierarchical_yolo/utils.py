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
    batch_size = target_indices.shape[1]
    hierarchy_size = hierarchy_index_tensor.shape[1]
    category_size = flat_scores.shape[2]
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, hierarchy_size)
    flat_indices = batch_indices * category_size + hierarchy_index_tensor[target_indices]
    flat_mask = hierarchy_mask[target_indices]
    unraveled_indices = torch.unravel_index(flat_indices, (1, batch_size, category_size))
    raveled_scores = flat_scores[unraveled_indices]
    masked_raveled_scores = raveled_scores.masked_fill(flat_mask, 1)
    return masked_raveled_scores

