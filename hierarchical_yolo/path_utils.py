import torch
import itertools

def truncate_path_conditionals(path: list[int], score: torch.Tensor, threshold: float = 0.25) -> tuple[list[int], torch.Tensor]:
    """Truncates a path based on a conditional probability threshold.

    This function iterates through a path and its corresponding conditional
    probabilities, stopping at the first element where the probability
    is below the given threshold.

    Parameters
    ----------
    path : list[int]
        A list of category indices representing the path.
    score : torch.Tensor
        A 1D tensor where each element is the conditional probability
        of the corresponding category in the path.
    threshold : float, optional
        The probability threshold below which to truncate, by default 0.25.

    Returns
    -------
    tuple[list[int], torch.Tensor]
        A tuple containing the truncated path and its corresponding scores.

    Examples
    --------
    >>> import torch
    >>> path = [4, 7]
    >>> score = torch.tensor([0.5412, 0.4371])
    >>> truncate_path_conditionals(path, score, threshold=0.589)
    ([], tensor([]))
    >>> path = [4, 2]
    >>> score = torch.tensor([0.9896, 0.5891])
    >>> truncate_path_conditionals(path, score, threshold=0.589)
    ([4, 2], tensor([0.9896, 0.5891]))
    """
    truncated_path, truncated_score = [], []
    for category, p in zip(path, score):
        if p < threshold:
            break
        truncated_path.append(category)
    return truncated_path, score[:len(truncated_path)]

def truncate_path_marginals(path: list[int], score: torch.Tensor, threshold: float = 0.25) -> tuple[list[int], torch.Tensor]:
    """Truncates a path based on a marginal probability threshold.

    This function iterates through a path, calculating the cumulative
    product (marginal probability) of the scores. It stops at the first
    element where this cumulative product falls below the given threshold.

    Parameters
    ----------
    path : list[int]
        A list of category indices representing the path.
    score : torch.Tensor
        A 1D tensor where each element is the conditional probability
        of the corresponding category in the path.
    threshold : float, optional
        The probability threshold below which to truncate, by default 0.25.

    Returns
    -------
    tuple[list[int], torch.Tensor]
        A tuple containing the truncated path and its corresponding scores.

    Examples
    --------
    >>> import torch
    >>> path = [4, 2]
    >>> score = torch.tensor([0.9896, 0.5891])
    >>> truncate_path_marginals(path, score, threshold=0.589)
    ([4], tensor([0.9896]))
    >>> path = [4, 6]
    >>> score = torch.tensor([0.9246, 0.7684])
    >>> truncate_path_marginals(path, score, threshold=0.589)
    ([4, 6], tensor([0.9246, 0.7684]))
    """
    truncated_path, truncated_score = [], []
    marginal_p = 1
    for category, p in zip(path, score):
        marginal_p *= p
        if marginal_p < threshold:
            break
        truncated_path.append(category)
    return truncated_path, score[:len(truncated_path)]

def truncate_paths_marginals(predicted_paths: list[list[int]], predicted_path_scores: list[torch.Tensor], threshold: float = 0.25) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Applies marginal probability truncation to a list of paths.

    This function iterates through lists of paths and scores, applying
    the `truncate_path_marginals` function to each path-score pair.

    Parameters
    ----------
    predicted_paths : list[list[int]]
        A list of paths, where each path is a list of category indices.
    predicted_path_scores : list[torch.Tensor]
        A list of 1D tensors, each corresponding to a path in `predicted_paths`.
    threshold : float, optional
        The probability threshold to pass to the truncation function,
        by default 0.25.

    Returns
    -------
    tuple[list[list[int]], list[torch.Tensor]]
        A tuple containing the list of truncated paths and the list of
        their corresponding truncated scores.

    Examples
    --------
    >>> import torch
    >>> paths = [[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]]
    >>> scores = [torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])]
    >>> tpaths, tscores = truncate_paths_marginals(paths, scores, threshold=0.589)
    >>> tpaths
    [[4], [4, 6], [4, 5], [], []]
    >>> tscores
    [tensor([0.9896]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])]
    """
    tpaths, tscores = [], []
    for paths, scores in zip(predicted_paths, predicted_path_scores):
        tpath, tscore = truncate_path_marginals(paths, scores, threshold=threshold)
        tpaths.append(tpath), tscores.append(tscore)
    return tpaths, tscores

def truncate_paths_conditionals(predicted_paths: list[list[int]], predicted_path_scores: list[torch.Tensor], threshold: float = 0.25) -> tuple[list[list[int]], list[torch.Tensor]]:
    """Applies conditional probability truncation to a list of paths.

    This function iterates through lists of paths and scores, applying
    the `truncate_path_conditionals` function to each path-score pair.

    Parameters
    ----------
    predicted_paths : list[list[int]]
        A list of paths, where each path is a list of category indices.
    predicted_path_scores : list[torch.Tensor]
        A list of 1D tensors, each corresponding to a path in `predicted_paths`.
    threshold : float, optional
        The probability threshold to pass to the truncation function,
        by default 0.25.

    Returns
    -------
    tuple[list[list[int]], list[torch.Tensor]]
        A tuple containing the list of truncated paths and the list of
        their corresponding truncated scores.

    Examples
    --------
    >>> import torch
    >>> paths = [[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]]
    >>> scores = [torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])]
    >>> tpaths, tscores = truncate_paths_conditionals(paths, scores, threshold=0.589)
    >>> tpaths
    [[4, 2], [4, 6], [4, 5], [], []]
    >>> tscores
    [tensor([0.9896, 0.5891]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])]
    """
    tpaths, tscores = [], []
    for paths, scores in zip(predicted_paths, predicted_path_scores):
        tpath, tscore = truncate_path_conditionals(paths, scores, threshold=threshold)
        tpaths.append(tpath), tscores.append(tscore)
    return tpaths, tscores


def batch_truncate_paths_marginals(predicted_paths: list[list[list[int]]], predicted_path_scores: list[list[torch.Tensor]], threshold: float = 0.25) -> list[tuple[list[list[int]], list[torch.Tensor]]]:
    """Applies marginal probability truncation to a batch of path lists.

    This function maps the `truncate_paths_marginals` function over a
    batch of predicted paths and scores.

    Parameters
    ----------
    predicted_paths : list[list[list[int]]]
        A batch of path lists. Each item in the outer list corresponds to
        an item in the batch.
    predicted_path_scores : list[list[torch.Tensor]]
        A batch of score lists, corresponding to `predicted_paths`.
    threshold : float, optional
        The probability threshold to use for truncation, by default 0.25.

    Returns
    -------
    list[tuple[list[list[int]], list[torch.Tensor]]]
        A list of tuples, where each tuple contains the truncated paths and
        scores for an item in the batch.

    Examples
    --------
    >>> import torch
    >>> paths_batch = [[[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]], [[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]]]
    >>> scores_batch = [[torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])], [torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])]]
    >>> batch_truncate_paths_marginals(paths_batch, scores_batch, 0.589)
    [([[4], [4, 6], [4, 5], [], []], [tensor([0.9896]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])]), ([[4], [4, 6], [4, 5], [], []], [tensor([0.9896]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])])]
    """
    B = len(predicted_paths)
    return list(itertools.starmap(truncate_paths_marginals, zip(predicted_paths, predicted_path_scores, itertools.repeat(threshold, B))))


def batch_truncate_paths_conditionals(predicted_paths: list[list[list[int]]], predicted_path_scores: list[list[torch.Tensor]], threshold: float = 0.25) -> list[tuple[list[list[int]], list[torch.Tensor]]]:
    """Applies conditional probability truncation to a batch of path lists.

    This function maps the `truncate_paths_conditionals` function over a
    batch of predicted paths and scores.

    Parameters
    ----------
    predicted_paths : list[list[list[int]]]
        A batch of path lists. Each item in the outer list corresponds to
        an item in the batch.
    predicted_path_scores : list[list[torch.Tensor]]
        A batch of score lists, corresponding to `predicted_paths`.
    threshold : float, optional
        The probability threshold to use for truncation, by default 0.25.

    Returns
    -------
    list[tuple[list[list[int]], list[torch.Tensor]]]
        A list of tuples, where each tuple contains the truncated paths and
        scores for an item in the batch.

    Examples
    --------
    >>> import torch
    >>> paths_batch = [[[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]], [[4, 2], [4, 6], [4, 5], [4, 7], [4, 2]]]
    >>> scores_batch = [[torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])], [torch.tensor([0.9896, 0.5891]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([0.5412, 0.4371]), torch.tensor([0.5001, 0.0830])]]
    >>> batch_truncate_paths_conditionals(paths_batch, scores_batch, 0.589)
    [([[4, 2], [4, 6], [4, 5], [], []], [tensor([0.9896, 0.5891]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])]), ([[4, 2], [4, 6], [4, 5], [], []], [tensor([0.9896, 0.5891]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765]), tensor([]), tensor([])])]
    """
    B = len(predicted_paths)
    return list(itertools.starmap(truncate_paths_conditionals, zip(predicted_paths, predicted_path_scores, itertools.repeat(threshold, B))))


def filter_empty_paths(predicted_boxes: torch.Tensor, predicted_paths: list[list[int]], predicted_path_scores: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]], list[torch.Tensor]]:
    """Filters out predictions with empty paths.

    After truncation, some paths may become empty. This function removes
    those empty paths along with their corresponding scores and bounding
    boxes.

    Parameters
    ----------
    predicted_boxes : torch.Tensor
        A 2D tensor of bounding box predictions, where columns correspond
        to individual predictions (e.g., shape [4, N]).
    predicted_paths : list[list[int]]
        A list of predicted paths.
    predicted_path_scores : list[torch.Tensor]
        A list of predicted path scores.

    Returns
    -------
    tuple[torch.Tensor, list[list[int]], list[torch.Tensor]]
        A tuple containing the filtered boxes, paths, and scores,
        with empty path predictions removed.

    Examples
    --------
    >>> import torch
    >>> boxes = torch.tensor([[482.27, 395.77, 241.98, 359.60, 258.38], [8.11, 156.87, 152.91, 335.40, 24.81], [610.42, 429.38, 307.70, 382.68, 413.79], [103.86, 200.93, 197.57, 352.40, 197.61]])
    >>> paths = [[4], [4, 6], [4, 5], [], []]
    >>> scores = [torch.tensor([0.9896]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([]), torch.tensor([])]
    >>> f_boxes, f_paths, f_scores = filter_empty_paths(boxes, paths, scores)
    >>> f_boxes
    tensor([[482.27, 395.77, 241.98],
            [  8.11, 156.87, 152.91],
            [610.42, 429.38, 307.70],
            [103.86, 200.93, 197.57]])
    >>> f_paths
    [[4], [4, 6], [4, 5]]
    >>> f_scores
    [tensor([0.9896]), tensor([0.9246, 0.7684]), tensor([0.8949, 0.8765])]
    """
    keep_idx = [i for i, path in enumerate(predicted_paths) if len(path) > 0]
    keep_idx_tensor = torch.tensor(keep_idx)
    return (
        predicted_boxes[:,keep_idx],
        [predicted_paths[k] for k in keep_idx],
        [predicted_path_scores[k] for k in keep_idx]
    )



def batch_filter_empty_paths(predicted_boxes: list[torch.Tensor], predicted_paths: list[list[list[int]]], predicted_path_scores: list[list[torch.Tensor]]) -> list[tuple[torch.Tensor, list[list[int]], list[torch.Tensor]]]:
    """Applies empty path filtering to a batch of predictions.

    This function maps the `filter_empty_paths` function over a batch of
    predicted boxes, paths, and scores.

    Parameters
    ----------
    predicted_boxes : list[torch.Tensor]
        A batch of bounding box tensors.
    predicted_paths : list[list[list[int]]]
        A batch of predicted path lists.
    predicted_path_scores : list[list[torch.Tensor]]
        A batch of predicted path score lists.

    Returns
    -------
    list[tuple[torch.Tensor, list[list[int]], list[torch.Tensor]]]
        A list of tuples, where each tuple contains the filtered boxes,
        paths, and scores for an item in the batch.

    Examples
    --------
    >>> import torch
    >>> boxes_batch = [torch.tensor([[482.27, 395.77, 241.98, 359.60, 258.38], [8.11, 156.87, 152.91, 335.40, 24.81], [610.42, 429.38, 307.70, 382.68, 413.79], [103.86, 200.93, 197.57, 352.40, 197.61]]), torch.tensor([[482.27, 395.77, 241.98, 359.60, 258.38], [8.11, 156.87, 152.91, 335.40, 24.81], [610.42, 429.38, 307.70, 382.68, 413.79], [103.86, 200.93, 197.57, 352.40, 197.61]])]
    >>> paths_batch = [[[4], [4, 6], [4, 5], [], []], [[4], [4, 6], [4, 5], [], []]]
    >>> scores_batch = [[torch.tensor([0.9896]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([]), torch.tensor([])], [torch.tensor([0.9896]), torch.tensor([0.9246, 0.7684]), torch.tensor([0.8949, 0.8765]), torch.tensor([]), torch.tensor([])]]
    >>> result = batch_filter_empty_paths(boxes_batch, paths_batch, scores_batch)
    >>> len(result)
    2
    >>> result[0][0] # boxes for first batch item
    tensor([[482.27, 395.77, 241.98],
            [  8.11, 156.87, 152.91],
            [610.42, 429.38, 307.70],
            [103.86, 200.93, 197.57]])
    >>> result[0][1] # paths for first batch item
    [[4], [4, 6], [4, 5]]
    """
    B = len(predicted_paths)
    return list(itertools.starmap(filter_empty_paths, zip(predicted_boxes, predicted_paths, predicted_path_scores)))
