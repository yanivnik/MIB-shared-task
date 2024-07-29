import torch
from typing import Tuple
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint


def get_mean_ablation(model: HookedTransformer, 
                      dataloader: torch.utils.data.DataLoader,
                      node_names: Tuple[str], 
                      per_position=True) -> ActivationCache:
    """
    Compute the mean activations of a set of nodes over a dataset.

    Args:
        model: The model to compute the activations for.
        dataloader: The dataloader to compute the activations over.
        node_names: The names of the nodes to compute the activations from.
        per_position: If True, compute the mean activations per position in the sequence (Assumes same sequence length for all batches).
                      Otherwise, compute the mean activations over the entire sequence (No assumption on sequence length across batches).
    """

    model.reset_hooks()
    mean_activations = {}
    total_seq_len = 0

    def forward_cache_hook(act: torch.tensor, # batch, seq, dim
                           hook: HookPoint):
        if hook.name not in mean_activations:
            if per_position:
                mean_activations[hook.name] = torch.zeros(1, *act.shape[1:]) # 1, seq, dim
            else:
                mean_activations[hook.name] = torch.zeros(1, *act.shape[2:]) # 1, dim
        sum_dims = (0) if per_position else (0, 1)
        mean_activations[hook.name] += act.detach().clone().sum(dim=sum_dims)

    def nodes_filter(name): 
        return name.endswith(node_names)

    # Sum activations over all batches
    model.reset_hooks()
    model.add_hook(name=nodes_filter, hook=forward_cache_hook, dir="fwd")
    for batch in dataloader:
        if not per_position:
            total_seq_len += batch.shape[0] * batch.shape[1] # Count total sequence length for mean
        model(batch)
    model.reset_hooks()

    # Mean over all batches
    for name, acts in mean_activations.items():
        mean_div = len(dataloader) if per_position else total_seq_len
        mean_activations[name] = acts / mean_div

    mean_activations = ActivationCache(mean_activations, model)
    return mean_activations