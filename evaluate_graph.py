from typing import Callable, List, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import rearrange, einsum
from copy import deepcopy

from graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode, Node

def evaluate_graph(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metrics: List[Callable[[Tensor], Tensor]], prune:bool=True, quiet=False,
                   node_eval=False, edge_eval=True):
    """
    Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune beforehand to make sure your circuit is valid.
    """
    if prune:
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)

    empty_circuit = not graph.nodes['logits'].in_graph

    fwd_names = {edge.parent.out_hook for edge in graph.edges.values()}
    fwd_filter = lambda x: x in fwd_names
    
    corrupted_fwd_cache, corrupted_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)
    mixed_fwd_cache, mixed_fwd_hooks, _ = model.get_caching_hooks(fwd_filter)

    nodes_in_graph = [node for node in graph.nodes.values() if node.in_graph if not isinstance(node, InputNode)]
    nodes_not_in_graph = [node for node in graph.nodes.values() if node.in_graph is False if not isinstance(node, InputNode)]

    # For each node in the graph, construct its input (in the case of attention heads, multiple inputs) by corrupting the incoming edges that are not in the circuit.
    # We assume that the corrupted cache is filled with corresponding corrupted activations, and that the mixed cache contains the computed activations from preceding nodes in this forward pass.
    def make_input_construction_hook(node: Node, qkv=None):
        def input_construction_hook(activations, hook):
            for edge in node.parent_edges:
                if edge.qkv != qkv:
                    continue

                parent:Node = edge.parent
                if not edge.in_graph:
                    activations[edge.index] -= mixed_fwd_cache[parent.out_hook][parent.index]
                    activations[edge.index] += corrupted_fwd_cache[parent.out_hook][parent.index]
            return activations
        return input_construction_hook
    
    def make_node_hook(node: Node):
        def node_hook(activations, hook):
            if not node.in_graph:
                activations[node.index] = 0.
            return activations
        return node_hook

    input_construction_hooks = []
    node_hooks = []
    if edge_eval:
        for node in nodes_in_graph:
            if isinstance(node, InputNode):
                pass
            elif isinstance(node, LogitNode) or isinstance(node, MLPNode):
                input_construction_hooks.append((node.in_hook, make_input_construction_hook(node)))
            elif isinstance(node, AttentionNode):
                for i, letter in enumerate('qkv'):
                    if node.qkv_inputs is None:
                        continue
                    input_construction_hooks.append((node.qkv_inputs[i], make_input_construction_hook(node, qkv=letter)))
            else:
                raise ValueError(f"Invalid node: {node} of type {type(node)}")
    if node_eval:
        for node in nodes_not_in_graph:
            if isinstance(node, InputNode):
                pass
            elif isinstance(node, MLPNode):
                node_hooks.append((node.out_hook, make_node_hook(node)))
        # elif isinstance(node, AttentionNode):
        #     for i, letter in enumerate('qkv'):
        #         node_hooks.append((node.qkv_inputs[i], make_node_hook(node, qkv=letter)))
            
    # and here we actually run / evaluate the model
    metrics_list = True
    if not isinstance(metrics, list):
        metrics = [metrics]
        metrics_list = False
    results = [[] for _ in metrics]
    
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        tokenized = model.tokenizer(clean, padding='longest', return_tensors='pt', add_special_tokens=True)
        input_lengths = 1 + tokenized.attention_mask.sum(1)
        with torch.inference_mode():
            with model.hooks(corrupted_fwd_hooks):
                corrupted_logits = model(corrupted)

            with model.hooks(mixed_fwd_hooks + input_construction_hooks + node_hooks):
                if empty_circuit:
                    # if the circuit is totally empty, so is nodes_in_graph
                    # so we just corrupt everything manually like this
                    logits = model(corrupted)
                else:
                    logits = model(clean)

        for i, metric in enumerate(metrics):
            r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if not metrics_list:
        results = results[0]
    return results


def evaluate_area_under_curve(model, graph, dataloader, metrics, prune=True, quiet=False,
                              node_eval=True, run_corrupted=False):
    baseline_score = evaluate_baseline(model, dataloader, metrics, run_corrupted=run_corrupted).mean().item()
    
    if node_eval:
        filtered_nodes = [(node, graph.nodes[node]) for node in graph.nodes if isinstance(graph.nodes[node], MLPNode)]
        sorted_itemlist = sorted(filtered_nodes, key=lambda x: x[1].score, reverse=True)
        num_nodes = len(sorted_itemlist)
    else:
        filtered_edges = [(name, graph.edges[name]) for name in graph.edges]
        sorted_itemlist = sorted(filtered_edges, key=lambda x: x[1].score, reverse=True)
        num_edges = len(sorted_itemlist)
    order = 10
    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)
    # percentages = (.001, .002, .005)
    faithfulnesses = []
    for pct in percentages:
        this_graph = graph
        if node_eval:
            curr_num_items = int(pct * num_nodes)
            print(f"Computing results for {pct*100}% of nodes (N={curr_num_items})")
            for idx, node in enumerate(sorted_itemlist):
                if idx < curr_num_items:
                    this_graph.nodes[node[0]].in_graph = True
                else:
                    this_graph.nodes[node[0]].in_graph = False
        else:
            curr_num_items = int(pct * num_edges)
            print(f"Computing results for {pct*100}% of edges (N={curr_num_items})")
            for idx, edge in enumerate(sorted_itemlist):
                if idx < curr_num_items:
                    this_graph.edges[edge[0]].in_graph = True
                else:
                    this_graph.edges[edge[0]].in_graph = False
        edge_eval = not node_eval
        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics,
                                       prune=prune, quiet=quiet, node_eval=node_eval,
                                       edge_eval=edge_eval).mean().item()
        faithfulness = ablated_score / baseline_score
        faithfulnesses.append(faithfulness)
    
    area = 0.
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i+1
        trapezoidal = (percentages[i_2] - percentages[i_1]) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area += trapezoidal
    average = sum(faithfulnesses) / len(faithfulnesses)
    return area, average


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