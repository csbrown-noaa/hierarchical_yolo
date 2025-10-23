import torch
from .hierarchy_tensor_utils import accumulate_hierarchy, expand_target_hierarchy
from .utils import log1mexp

def hierarchical_loss(pred, targets, hierarchy_index):
    logsigmoids = torch.nn.functional.logsigmoid(pred)
    hierarchical_summed_logsigmoids = accumulate_hierarchy(logsigmoids, hierarchy_index, torch.cumsum)
    hierarchical_expanded_targets = expand_target_hierarchy(targets, hierarchy_index)
    hierarchical_summed_log1sigmoids = log1mexp(hierarchical_summed_logsigmoids)
    return -(
      (hierarchical_expanded_targets * hierarchical_summed_logsigmoids) 
      + (1 - hierarchical_expanded_targets) * hierarchical_summed_log1sigmoids
    )