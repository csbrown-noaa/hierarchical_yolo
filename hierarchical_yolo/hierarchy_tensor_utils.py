import torch
from .tree_utils import *
from typing import Callable

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

def build_hierarchy_index_tensor(hierarchy, device=None):
    '''
    Translate a hierarchy into a tensor representation.  Parent node ids are to the right of a node.  -1 is always to the right of roots or -1s.

    Examples
    ----------
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


# TODO! The `cumsum` implementation here doesn't make sense since we only take the last value... just have this do a sum.
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


def expand_target_hierarchy(
    target: torch.Tensor, hierarchy_index: torch.Tensor
) -> torch.Tensor:
    """Expands a one-hot target tensor up the hierarchy.

    This function takes a target tensor that is "one-hot" along the class
    dimension (i.e., contains a single non-zero value) and propagates that
    value to all ancestors of the target class. The implementation is fully
    vectorized.

    Parameters
    ----------
    target : torch.Tensor
        A tensor of shape `[B, D, N]`, where `B` is the batch size, `D` is the
        number of detections, and `N` is the number of classes. It is assumed
        to be one-hot along the last dimension.
    hierarchy_index : torch.Tensor
        An int tensor of shape `[N, M]` encoding the hierarchy, where `N` is the
        number of classes and `M` is the maximum hierarchy depth. Each row `i`
        contains the path from node `i` to its root.

    Returns
    -------
    torch.Tensor
        A new tensor with the same shape as `target` where the target value has
        been propagated up the hierarchy.

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
    >>> # Target is one-hot at index 0
    >>> target = torch.tensor([0.4, 0., 0., 0., 0.]).view(1, 1, 5)
    >>> expanded_target = expand_target_hierarchy(target, hierarchy_index)
    >>> print(expanded_target.squeeze())
    tensor([0.4000, 0.4000, 0.4000, 0.0000, 0.0000])
    >>> target = torch.tensor([0., 0., 0., 0.3, 0.]).view(1, 1, 5)
    >>> expanded_target = expand_target_hierarchy(target, hierarchy_index)
    >>> print(expanded_target.squeeze())
    tensor([0.0000, 0.0000, 0.0000, 0.3000, 0.3000])
    """
    M = hierarchy_index.shape[1]

    # Find the single non-zero value and its index in the target tensor.
    hot_value, hot_index = torch.max(target, dim=-1)

    # Gather the ancestral paths corresponding to the hot indices.
    # The shape will be [B, D, M].
    paths = hierarchy_index[hot_index]

    # Create a mask for valid indices (non -1) to handle padded paths.
    valid_mask = paths != -1

    # Create a "safe" index tensor to prevent out-of-bounds errors from -1.
    # We replace -1 with a valid index (e.g., 0) and will zero out its
    # contribution later using a masked source.
    safe_paths = paths.masked_fill(~valid_mask, 0)
    safe_paths_ints = safe_paths.to(torch.int64)

    # Prepare the source tensor for the scatter operation.
    # It should have the same value (`hot_value`) for all valid path members.
    src_values = hot_value.unsqueeze(-1).expand(-1, -1, M)
    masked_src = src_values * valid_mask.to(src_values.dtype)

    # Create an output tensor and scatter the hot value into all ancestral positions.
    expanded_target = torch.zeros_like(target)
    expanded_target.scatter_(dim=-1, index=safe_paths_ints, src=masked_src)

    return expanded_target
