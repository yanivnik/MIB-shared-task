from transformer_lens import HookedTransformer, ActivationCache
import pandas as pd
import torch
def get_mean_ablation(model: HookedTransformer, dataloader, nodes_names, per_position=False) -> ActivationCache:

    model.reset_hooks()
    mean_activations = {}

    def forward_cache_hook(act, hook):
        if per_position:
            mean_activations[hook.name] = torch.unsqueeze(act.detach().clone().mean(0), 0)  # batch seq d
        else:
            mean_activations[hook.name] = torch.unsqueeze(act.detach().clone().mean((0, 1)), 0)  # batch seq d


    def nodes(name): return name.endswith(nodes_names)

    model.add_hook(nodes, forward_cache_hook, "fwd")
    for x in dataloader:
        model(x)
    model.reset_hooks()
    mean_activations = ActivationCache(mean_activations, model)

    return mean_activations