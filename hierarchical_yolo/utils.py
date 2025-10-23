import ultralytics
import torch

def log_matrix(m):
    formatted_lines = []
    for i in range(m.shape[0]):
        vec = m[i]
        line = f"{i:04d}: " + ", ".join(f"{x:.4f}" for x in vec.tolist())
        formatted_lines.append(line)
    ultralytics.utils.LOGGER.info("\n".join(formatted_lines))


def argmax_from_subset(scores, indices):
    """
    Finds the argmax from a subset of indices

    The core operation is performed on the last dimension of the tensor

    Parameters
    ----------
        scores (torch.Tensor): Tensor of scores with shape (*D, N).
        indices (torch.Tensor): A 1D Tensor of viable indices with shape (K,).

    Returns
    ----------
        torch.Tensor: A tensor of shape (*D) containing the argmax index.

    Examples
    ----------
        >>> scores = torch.tensor([
        ...         [10, 20, 30, 5, 40],
        ...         [99, 88, 77, 66, 55]
        ... ])
        >>> indices = torch.tensor([0, 2, 4])
        >>> argmax_from_subset(scores, indices)
        tensor([4, 0])
    """
    # 1. Use advanced indexing to select the subset of scores.
    # The ellipsis (...) selects all leading dimensions.
    # The result `subset_scores` will have shape (*D, K).
    subset_scores = scores[..., indices]

    # 2. Find the argmax within this subset along the last dimension.
    # The result `local_argmax_indices` will have shape (*D).
    local_argmax_indices = torch.argmax(subset_scores, dim=-1)

    # 3. Use the local argmaxes to index directly into the original 1D `indices`.
    # This is the simplest and most efficient step.
    return indices[local_argmax_indices]


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

    # B, C = flat_scores.shape
    # G = sibling_mask.shape[1]
    scores_expanded = flat_scores.unsqueeze(1)  # (B, 1, C)
    masked_scores = scores_expanded + torch.log(sibling_mask.T.unsqueeze(0))  # (B, G, C)
    logsumexp_by_group = torch.logsumexp(masked_scores, dim=-1)  # (B, G)
    zerod_logsumexp = logsumexp_by_group.masked_fill(torch.isinf(logsumexp_by_group), 9)
    logsumexp = (sibling_mask * zerod_logsumexp.unsqueeze(1)).sum(dim=-1)  # (B, C)

    return logsumexp


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - exp(x)) in a numerically stable way.

    This function is designed to prevent the loss of precision that occurs
    when `x` is very close to zero (i.e., a small negative number).
    Directly computing `log(1 - exp(x))` can lead to catastrophic
    cancellation and result in `-inf`.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor containing negative values (log-probabilities).
        The function is not designed for `x >= 0`, as `1 - exp(x)` would be
        zero or negative, making the logarithm undefined.

    Returns
    -------
    torch.Tensor
        The computed `log(1 - exp(x))` values, with the same shape as `x`.

    Notes
    -----
    The function uses two different mathematical identities based on the
    value of `x` to ensure stability:
    
    1. For `x > -ln(2)` (i.e., `x` is close to 0), it computes
       `log(-expm1(x))`. The `torch.expm1(x)` function computes `exp(x) - 1`
       with high precision, avoiding cancellation.
    2. For `x <= -ln(2)`, `exp(x)` is small, so the expression `1 - exp(x)`
       is not problematic. For better precision, `log1p(-exp(x))` is used,
       where `torch.log1p(y)` computes `log(1 + y)`.

    Examples
    --------
    >>> import torch
    >>> log_p = torch.tensor([-1e-9, -2.0, -20.0])
    >>> log1mexp(log_p)
    tensor([-2.0723e+01, -1.4541e-01, -2.0612e-09])


    """
    # The threshold is -ln(2) approx -0.7
    threshold = -0.7
    # For x > threshold, exp(x) is close to 1
    result_close_to_zero = torch.log(-torch.expm1(x))
    # For x <= threshold, exp(x) is small
    result_far_from_zero = torch.log1p(-torch.exp(x))

    return torch.where(x > threshold, result_close_to_zero, result_far_from_zero)

