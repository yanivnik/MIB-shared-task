from typing import Callable, List, Union
import math
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import rearrange, einsum

from attribute import tokenize_plus, make_hooks_and_matrices
from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node


def evaluate_graph(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metrics: List[Callable[[Tensor], Tensor]], prune:bool=True, quiet=False, zero_ablate=False, neuron_level=False):
    """
    Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.
    """
    if prune:
        graph.prune_dead_nodes()

    empty_circuit = not graph.nodes['logits'].in_graph

    # Construct a matrix that indicates which edges are in the graph
    in_graph_matrix = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)
    for edge in graph.edges.values():
        if edge.in_graph:
            in_graph_matrix[graph.forward_index(edge.parent, attn_slice=False), graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False)] = 1
    
    # same thing but for neurons
    if neuron_level:
        neuron_matrix = torch.ones((graph.n_forward, model.cfg.d_model), device='cuda', dtype=model.cfg.dtype)
        for node in graph.nodes.values():
            if node.neurons is not None:
                neuron_matrix[graph.forward_index(node, attn_slice=False)] = node.neurons

        # If an edge is in the graph, but not all its neurons are, we need to update that edge anyway
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph, 'forward backward, forward -> forward backward')
    else:
        neuron_matrix = None

    # We take the opposite matrix, because we'll use at as a mask to specify 
    # which edges we want to corrupt
    in_graph_matrix = 1 - in_graph_matrix
    if neuron_level:
        neuron_matrix = 1 - neuron_matrix


    # For each node in the graph, corrupt its inputs, if the corresponding edge isn't in the graph 
    # We corrupt it by adding in the activation difference (b/w clean and corrupted acts)
    def make_input_construction_hook(act_index, activation_differences, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            if neuron_matrix is not None:
                update = einsum(activation_differences[:, :, :len(in_graph_vector)], neuron_matrix[:len(in_graph_vector)], in_graph_vector,'batch pos previous hidden, previous hidden, previous -> batch pos hidden')
            else:
                update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,'batch pos previous hidden, previous -> batch pos hidden')
            activations[act_index] += update
            return activations
        return input_construction_hook

    def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
        input_construction_hooks = []
        for node in graph.nodes.values():
            if isinstance(node, InputNode) or not node.in_graph:
                pass
            elif isinstance(node, LogitNode) or isinstance(node, MLPNode):
                fwd_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(node.index, activation_differences, in_graph_matrix[:fwd_index, bwd_index], neuron_matrix)
                input_construction_hooks.append((node.in_hook, input_cons_hook))
            elif isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    fwd_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=False)
                    input_cons_hook = make_input_construction_hook(node.index, activation_differences, in_graph_matrix[:fwd_index, bwd_index], neuron_matrix)
                    input_construction_hooks.append((node.qkv_inputs[i], input_cons_hook))
            else:
                raise ValueError(f"Invalid node: {node} of type {type(node)}")
        return input_construction_hooks
    
    # and here we actually run / evaluate the model
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]
    
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)
        
        # fwd_hooks_corrupted adds in corrupted acts to activation_difference
        # fwd_hooks_clean subtracts out clean acts from activation_difference
        # activation difference is of size (batch, pos, src_nodes, hidden)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(model, graph, len(clean), n_pos, None)
        
        input_construction_hooks = make_input_construction_hooks(activation_difference, in_graph_matrix, neuron_matrix)
        with torch.inference_mode():
            
            if not zero_ablate:
                # We intervene by subtracting out clean and adding in corrupted activations
                # In the case of zero ablation, we skip the adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            else:
                corrupted_logits = model(corrupted_tokens)
                
            with model.hooks(fwd_hooks_clean + input_construction_hooks):
                if empty_circuit:
                    # if the circuit is totally empty, so is nodes_in_graph
                    # so we just corrupt everything manually like this
                    logits = model(corrupted_tokens, attention_mask=attention_mask)
                else:
                    logits = model(clean_tokens, attention_mask=attention_mask)

        for i, metric in enumerate(metrics):
            r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results


def evaluate_area_under_curve(model: HookedTransformer, graph: Graph, dataloader, metrics, prune=True, quiet=False,
                              node_eval=True, neuron_level=False,
                              run_corrupted=False, above_curve=False,
                              log_scale=True, inverse=False,
                              absolute=False, zero_ablate=False):
    baseline_score = evaluate_baseline(model, dataloader, metrics, run_corrupted=run_corrupted).mean().item()
    
    if node_eval:
        filtered_nodes = [node for node in graph.nodes.values() if isinstance(node, MLPNode)]
        if neuron_level:
            scores = torch.cat([node.neuron_scores for node in filtered_nodes], dim=-1)
        else:
            scores = torch.tensor([node.score for node in filtered_nodes])
        if absolute:
                scores = scores.abs()
        sorted_scores = scores.sort(descending=True).values
    
    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

    faithfulnesses = []
    for pct in percentages:
        this_graph = graph
        if node_eval:
            curr_num_items = int(pct * len(sorted_scores))
            threshold = sorted_scores[curr_num_items - 1].item()

            if neuron_level:
                print(f"Computing results for {pct*100}% of neurons (N={curr_num_items})")
                for node in filtered_nodes:
                    node.neurons = (node.neuron_scores >= threshold) if not inverse else (node.neuron_scores < threshold)
                    node.in_graph = torch.any(node.neurons)
            else:
                print(f"Computing results for {pct*100}% of nodes (N={curr_num_items})")
                for node in filtered_nodes:
                    node.in_graph = (node.score >= threshold) if not inverse else (node.score < threshold)

        else:
            curr_num_items = int(pct * len(graph.edges))
            print(f"Computing results for {pct*100}% of edges (N={curr_num_items})")
            this_graph.apply_topn(curr_num_items, absolute=absolute)

        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics,
                                       prune=prune, quiet=quiet, zero_ablate=zero_ablate,
                                       neuron_level=neuron_level).mean().item()
        faithfulness = ablated_score / baseline_score
        print(faithfulness)
        faithfulnesses.append(faithfulness)
    
    area_under = 0.
    area_from_100 = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i+1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (percentages[i_2] - percentages[i_1]) * \
                        (((abs(1. - faithfulnesses[i_1])) + (abs(1. - faithfulnesses[i_2]))) / 2)
        area_from_100 += trapezoidal 
        
        trapezoidal = (percentages[i_2] - percentages[i_1]) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return area_under, area_from_100, average, faithfulnesses


def evaluate_baseline(model: HookedTransformer, dataloader:DataLoader, metrics: List[Callable[[Tensor], Tensor]], run_corrupted=False):
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    
    results = [[] for _ in metrics]
    for clean, corrupted, label in tqdm(dataloader):
        tokenized = model.tokenizer(clean, padding='longest', return_tensors='pt', add_special_tokens=True)
        input_lengths = 1 + tokenized.attention_mask.sum(1)
        with torch.inference_mode():
            corrupted_logits = model(corrupted)
            logits = model(clean)
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, input_lengths, label).cpu()
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results

def evaluate_kl(model: HookedTransformer, inputs, target_inputs):
    results = []
    for inp, target in tqdm(zip(inputs, target_inputs), total=len(inputs)):
        
        batch_size = len(inp)
        tokenized = model.tokenizer(inp, padding='longest', return_tensors='pt', add_special_tokens=True)
        input_length = 1 + tokenized.attention_mask.sum(1)
        
        with torch.inference_mode():
            target_logits = model(target)
            logits = model(inp)

        idx = torch.arange(batch_size, device=logits.device)

        logits = logits[idx, input_length - 1]
        target_logits = target_logits[idx, input_length - 1]

        logprobs = torch.log_softmax(logits, dim=-1)
        target_logprobs = torch.log_softmax(target_logits, dim=-1)

        r = torch.nn.functional.kl_div(logprobs, target_logprobs, log_target=True, reduction='mean')
        if len(r.size()) == 0:
            r = r.unsqueeze(0)
        results.append(r)

    return torch.cat(results)