import math 
from typing import Literal, Optional
from functools import partial
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from dataset import HFEAPDataset
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from metrics import get_metric

def evaluate_area_under_curve(model: HookedTransformer, graph: Graph, dataloader, metrics, quiet:bool=False, 
                              level:Literal['edge', 'node','neuron']='edge', log_scale:bool=True, absolute:bool=True, intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', intervention_dataloader:DataLoader=None,
                              no_normalize:Optional[bool]=False, apply_greedy:bool=False):
    baseline_score = evaluate_baseline(model, dataloader, metrics).mean().item()
    graph.apply_topn(0, True)
    corrupted_score = evaluate_graph(model, graph, dataloader, metrics, quiet=quiet, intervention=intervention, intervention_dataloader=intervention_dataloader).mean().item()
    
    if level == 'neuron':
        assert graph.neurons_scores is not None, "Neuron scores must be present for neuron-level evaluation"
        n_scored_items = (~torch.isnan(graph.neurons_scores)).sum().item()
    elif level == 'node':
        assert graph.nodes_scores is not None, "Node scores must be present for node-level evaluation"
        n_scored_items = (~torch.isnan(graph.nodes_scores)).sum().item()
    else:
        n_scored_items = len(graph.edges)
    
    percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

    faithfulnesses = []
    weighted_edge_counts = []
    for pct in percentages:
        this_graph = graph
        curr_num_items = int(pct * n_scored_items)
        print(f"Computing results for {pct*100}% of {level}s (N={curr_num_items})")
        if apply_greedy:
            assert level == 'edge', "Greedy application only supported for edge-level evaluation"
            this_graph.apply_greedy(curr_num_items, absolute=absolute, prune=True)
        else:
            this_graph.apply_topn(curr_num_items, absolute, level=level, prune=True)
        
        weighted_edge_count = this_graph.weighted_edge_count()
        weighted_edge_counts.append(weighted_edge_count)

        ablated_score = evaluate_graph(model, this_graph, dataloader, metrics,
                                       quiet=quiet, intervention=intervention,
                                       intervention_dataloader=intervention_dataloader).mean().item()
        if no_normalize:
            faithfulness = ablated_score
        else:
            faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
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
    return weighted_edge_counts, area_under, area_from_100, average, faithfulnesses


def compare_graphs(reference: Graph, hypothesis: Graph, by_node: bool = False):
    # Track {true, false} {positives, negatives}
    TP, FP, TN, FN = 0, 0, 0, 0
    total = 0

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges

    for obj in ref_objs.values():
        total += 1
        if obj.name not in hyp_objs:
            if obj.in_graph:
                TP += 1
            else:
                FP += 1
            continue
            
        if obj.in_graph and hyp_objs[obj.name].in_graph:
            TP += 1
        elif obj.in_graph and not hyp_objs[obj.name].in_graph:
            FN += 1
        elif not obj.in_graph and hyp_objs[obj.name].in_graph:
            FP += 1
        elif not obj.in_graph and not hyp_objs[obj.name].in_graph:
            TN += 1
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    # f1 = (2 * precision * recall) / (precision + recall)
    TP_rate = recall
    FP_rate = FP / (FP + TN)

    return {"precision": precision,
            "recall": recall,
            "TP_rate": TP_rate,
            "FP_rate": FP_rate}

def area_under_roc(reference: Graph, hypothesis: Graph, by_node: bool = False):
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    if by_node:
        ref_objs = reference.nodes
        hyp_objs = hypothesis.nodes
    else:
        ref_objs = reference.edges
        hyp_objs = hypothesis.edges
    
    num_objs = len(ref_objs.values())
    for pct in (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1):
        this_num_objs = pct * num_objs
        if by_node:
            raise NotImplementedError("")
        else:
            hypothesis.apply_greedy(this_num_objs)
        scores = compare_graphs(reference, hypothesis)
        tpr_list.append(scores["TP_rate"])
        fpr_list.append(scores["FP_rate"])
        precision_list.append(scores["precision"])
        recall_list.append(scores["recall"])
    
    return {"TPR": tpr_list, "FPR": fpr_list,
            "precision": precision_list, "recall": recall_list}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type='str', nargs='+', required=True)
    parser.add_argument("--tasks", type='str', nargs='+', required=True)
    parser.add_argument("--circuit-path", type='str', required=True)
    parser.add_argument("--ablation", type='str', choices=['patching', 'zero', 'mean'], default='patching')
    parser.add_argument("--split", type='str', choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument("--level", type='str', choices=['edge', 'node', 'neuron'], default='edge')
    parser.add_argument("--absolute", type='bool', default=True)
    parser.add_argument("--batch-size", type='int', default=20)
    parser.add_argument("--circuit-dir", type='str', default='circuits')
    parser.add_argument("--circuit-files", type='str', nargs='+', default=None)
    parser.add_argument("--output-dir", type='str', default='results')
    args = parser.parse_args()

    i = 0
    for model_name in args.models:
        model = HookedTransformer.from_pretrained(model_name)
        model_name_saveable = model_name.split('/')[-1]
        for task in args.tasks:

            p = f'{args.circuit_dir}/{model_name_saveable}_{task}_{args.method}_{args.ablation}_{args.level}.json'

            if args.circuit_files is not None:
                p = args.circuit_files[i]
                i += 1

            if p.endswith('.json'):
                graph = Graph.from_json(p)
            elif p.endswith('.pt'):
                graph = Graph.from_pt(p)
            else:
                raise ValueError(f"Invalid file extension: {p.suffix}")
            
            dataset = HFEAPDataset(task, model.tokenizer, split=args.split, task=task, model_name=model_name)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            metric = get_metric('logit_diff', args.task, model.tokenizer, model)
            attribution_metric = partial(metric, mean=False, loss=False)
            
            weighted_edge_counts, area_under, area_from_100, average, faithfulnesses = evaluate_area_under_curve(model, graph, dataloader, attribution_metric, level=args.level, log_scale=True, absolute=args.absolute, intervention=args.ablation)

            d = {
                "weighted_edge_counts": weighted_edge_counts,
                "area_under": area_under,
                "area_from_100": area_from_100,
                "average": average,
                "faithfulnesses": faithfulnesses
            }
            Path(args.output_dir).mkdir(exist_ok=True)
            with open(args.output_dir / f"{model_name}_{task}_{args.ablation}.pkl", 'wb') as f:
                pickle.dump(d, f)

            