from typing import Callable, List, Union, Optional
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode

def get_npos_input_lengths(model: HookedTransformer, inputs: List[str]):
    """Gets the length of the position dimension, and the length of each input sequence

    Args:
        model (HookedTransformer): The model to use
        inputs (List[str]): A list of string inputs

    Returns:
        Tuple[int, Tensor]: The size of the position dimension and the length of each input sequence
    """
    tokenized = model.tokenizer(inputs, padding='longest', return_tensors='pt', add_special_tokens=True)
    n_pos = 1 + tokenized.attention_mask.size(1)
    input_lengths = 1 + tokenized.attention_mask.sum(1)
    return n_pos, input_lengths

def make_hooks_and_matrices(model: HookedTransformer, graph: Graph, batch_size:int , n_pos:int, scores: Tensor):
    """Makes a matrix, and hooks to fill it and the score matrix up

    Args:
        model (HookedTransformer): model to attribute
        graph (Graph): graph to attribute
        batch_size (int): size of the particular batch you're attributing
        n_pos (int): size of the position dimension
        scores (Tensor): The scores tensor you intend to fill

    Returns:
        Tuple[Tuple[List, List, List], Tensor]: The final tensor ([batch, pos, n_src_nodes, d_model]) stores activation differences, i.e. corrupted - clean activations. The first set of hooks will add in the activations they are run on (run these on corrupted input), while the second set will subtract out the activations they are run on (run these on clean input). The third set of hooks will compute the gradients and update the scores matrix that you passed in. 
    """
    activation_difference = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []
    
    def activation_hook(index, activations, hook, add:bool=True):
        acts = activations.detach()
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e
    
    def gradient_hook(fwd_index: Union[slice, int], bwd_index: Union[slice, int], gradients:torch.Tensor, hook):
        """Takes in a gradient and uses it and activation_difference 
        to compute an update to the score matrix

        Args:
            fwd_index (Union[slice, int]): The forward index of the (src) node
            bwd_index (Union[slice, int]): The backward index of the (dst) node
            gradients (torch.Tensor): The gradients of this backward pass 
            hook (_type_): (unused)

        """
        grads = gradients.detach()
        try:
            if isinstance(fwd_index, slice):
                fwd_index = fwd_index.start
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            s = einsum(activation_difference[:, :, :fwd_index], grads,'batch pos forward hidden, batch pos backward hidden -> forward backward')
            s = s.squeeze(1)
            scores[:fwd_index, bwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), grads.size())
            raise e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        fwd_index =  graph.forward_index(node)
        if not isinstance(node, LogitNode):
            fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, fwd_index, bwd_index)))
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append((node.in_hook, partial(gradient_hook, fwd_index, bwd_index)))
            
    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

def get_scores(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], quiet=False):
    scores = torch.zeros((graph.n_forward, graph.n_backward), dtype=model.cfg.dtype)    
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        n_pos, input_lengths = get_npos_input_lengths(model, clean)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            corrupted_logits = model(corrupted)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean)
            metric_value = metric(logits, corrupted_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        n_pos, input_lengths = get_npos_input_lengths(model, clean)

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations 
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) 
                new_input.requires_grad = True 
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(1, steps+1):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores

allowed_aggregations = {'sum', 'mean'}#, 'l2'}        
def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], aggregation='sum', integrated_gradients: Optional[int]=None, quiet=False):
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if integrated_gradients is None:
        scores = get_scores(model, graph, dataloader, metric, quiet=quiet)
    else:
        assert integrated_gradients > 0, f"integrated_gradients gives positive # steps (m), but got {integrated_gradients}"
        scores = get_scores_ig(model, graph, dataloader, metric, steps=integrated_gradients, quiet=quiet)

        if aggregation == 'mean':
            scores /= model.cfg.d_model
        #elif aggregation == 'l2':
        #    scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)
        
    scores = scores.cpu()
    #graph.edge_scores = scores
    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)]