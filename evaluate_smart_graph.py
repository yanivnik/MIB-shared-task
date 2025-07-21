"""
Script to evaluate smart graph building methods for circuit discovery.

This script converts the notebook functionality into a command-line tool that:
1. Loads model and data
2. Performs attribution on edges and nodes
3. Builds graphs using different methods
4. Evaluates faithfulness and calculates AUC scores

Example usage:
    python evaluate_smart_graph.py --model qwen2.5 --task mcqa --method greedy --n-edges 250 --n-nodes 45
"""

import argparse
from datetime import datetime
import logging
import math
import copy
import random
import sys
import os
from typing import Callable, List, Tuple, Dict, Any, Optional
from functools import partial


import torch
import numpy as np
from einops import einsum
import networkx as nx
import json
import pulp

# Add EAP-IG to path
sys.path.append("EAP-IG/src")

from torch.utils.data import DataLoader, Subset
from transformer_lens import HookedTransformer
from datasets import load_dataset


from eap.graph import Graph, LogitNode
from eap.attribute import attribute
from eap.attribute_node import attribute_node
from eap.evaluate import evaluate_graph, evaluate_baseline
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from plotting import plot_auc_results


def graph_to_dict(graph: Graph) -> Dict[str, Any]:
    """
    Creates a serializable dictionary from a Graph object, containing only tensors and config.
    This avoids recursion errors when pickling the graph object's circular references.
    """
    if graph is None:
        return None

    d = {
        "cfg": dict(graph.cfg),
        "scores": graph.scores,
        "in_graph": graph.in_graph,
        "nodes_in_graph": graph.nodes_in_graph,
    }
    if graph.nodes_scores is not None:
        d["nodes_scores"] = graph.nodes_scores
    if graph.neurons_in_graph is not None:
        d["neurons_in_graph"] = graph.neurons_in_graph
    if graph.neurons_scores is not None:
        d["neurons_scores"] = graph.neurons_scores
    return d


def dict_to_graph(d: Dict[str, Any]) -> Graph:
    """
    Restores a Graph object from a dictionary that was created by graph_to_dict.
    """
    if d is None:
        return None

    # Determine if neuron_level and node_scores were enabled based on the dictionary keys
    neuron_level = "neurons_in_graph" in d and d["neurons_in_graph"] is not None
    node_scores_enabled = "nodes_scores" in d and d["nodes_scores"] is not None

    # Recreate the graph structure from the config
    graph = Graph.from_model(
        d["cfg"], neuron_level=neuron_level, node_scores=node_scores_enabled
    )

    # Restore the saved tensor data
    graph.scores = d["scores"]
    graph.in_graph = d["in_graph"]
    graph.nodes_in_graph = d["nodes_in_graph"]

    if node_scores_enabled:
        graph.nodes_scores = d["nodes_scores"]
    if neuron_level:
        graph.neurons_in_graph = d["neurons_in_graph"]
        if "neurons_scores" in d and d["neurons_scores"] is not None:
            graph.neurons_scores = d["neurons_scores"]

    return graph


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the script.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("evaluate_smart_graph.log"),
        ],
    )


def set_deterministic(seed=1337):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name: str, cache_dir: str) -> HookedTransformer:
    """
    Load and configure the transformer model.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        HookedTransformer: Configured transformer model.

    Raises:
        ValueError: If model_name is not supported.
    """
    logging.info(f"Loading model: {model_name}")

    if model_name not in MODEL_NAME_TO_FULLNAME:
        raise ValueError(
            f"Unsupported model: {model_name}. Available models: {list(MODEL_NAME_TO_FULLNAME.keys())}"
        )

    full_model_name = MODEL_NAME_TO_FULLNAME[model_name]

    # Load model with appropriate configuration
    if model_name in ("qwen2.5", "gemma2", "llama3"):
        model = HookedTransformer.from_pretrained(
            full_model_name,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device="cuda",
            cache_dir=cache_dir,
        )
    else:
        model = HookedTransformer.from_pretrained(full_model_name, cache_dir=cache_dir)

    # Configure model settings
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    logging.info(f"Model loaded successfully: {full_model_name}")
    return model


def load_dataset_and_metric(
    task_name: str,
    model_tokenizer,
    model: HookedTransformer,
    num_examples: int = 500,
    batch_size: int = 64,
    cache_dir: str = "/mnt/nlp",
) -> Tuple[DataLoader, callable, float]:
    """
    Load and configure the dataset with metric function.

    Args:
        task_name (str): Name of the task.
        model_tokenizer: Tokenizer from the model.
        model (HookedTransformer): The transformer model.
        num_examples (int): Number of examples to use from dataset.
        batch_size (int): Batch size for dataloader.

    Returns:
        Tuple[DataLoader, callable, float]: Dataloader, metric function, and baseline score.

    Raises:
        ValueError: If task_name is not supported.
    """
    logging.info(f"Loading dataset for task: {task_name}")

    if task_name not in TASKS_TO_HF_NAMES:
        raise ValueError(
            f"Unsupported task: {task_name}. Available tasks: {list(TASKS_TO_HF_NAMES.keys())}"
        )

    dataset_name = f"mib-bench/{TASKS_TO_HF_NAMES[task_name]}"

    # Load dataset
    dataset = HFEAPDataset(dataset_name, model_tokenizer, task=task_name)
    if num_examples is not None:
        dataset.head(num_examples)
        logging.info(
            f"Using {num_examples} examples out of {len(dataset)} from dataset"
        )
    else:
        logging.info(f"Using all {len(dataset)} examples from dataset")

    dataloader = dataset.to_dataloader(batch_size=batch_size)

    # Get metric function
    metric_fn = get_metric(
        metric_name="logit_diff",
        task=["ignore this"],
        tokenizer=model_tokenizer,
        model=model,
    )

    # Calculate baseline
    logging.info("Calculating baseline performance...")
    baseline = (
        evaluate_baseline(model, dataloader, partial(metric_fn, loss=False, mean=False))
        .mean()
        .item()
    )

    logging.info(f"Dataset loaded. Baseline performance: {baseline:.4f}")
    return dataloader, metric_fn, baseline


def perform_attribution(
    model: HookedTransformer,
    dataloader: DataLoader,
    metric_fn: callable,
    ig_steps: int = 5,
) -> Tuple[Graph, Graph]:
    """
    Perform attribution on both edges and nodes.

    Args:
        model (HookedTransformer): The transformer model.
        dataloader (DataLoader): Data loader for the dataset.
        metric_fn (callable): Metric function for attribution.
        ig_steps (int): Number of integrated gradient steps.

    Returns:
        Tuple[Graph, Graph]: Edge graph and node graph with attribution scores.
    """
    logging.info("Starting attribution process...")

    # Create graphs for edge and node attribution
    g_edges = Graph.from_model(model)
    g_nodes = Graph.from_model(model)

    logging.info(
        f"Graph created with {len(g_edges.nodes)} nodes and {len(g_edges.edges)} edges"
    )

    # Perform attribution
    logging.info("Performing edge attribution...")
    attribute(
        model,
        g_edges,
        dataloader,
        partial(metric_fn, loss=True, mean=True),
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
    )

    logging.info("Performing node attribution...")
    attribute_node(
        model,
        g_nodes,
        dataloader,
        partial(metric_fn, loss=True, mean=True),
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
    )

    logging.info("Attribution completed successfully")
    return g_edges, g_nodes


def bootstrap_sample_dataloader(
    dataloader: DataLoader, n_samples: int = None, replace: bool = True
) -> DataLoader:
    """
    Create a bootstrap sample from the original dataloader.

    Args:
        dataloader (DataLoader): Original dataloader to sample from.
        n_samples (int, optional): Number of samples to draw. If None, uses same size as original.
        replace (bool): Whether to sample with replacement.

    Returns:
        DataLoader: Bootstrap sampled dataloader.
    """
    dataset = dataloader.dataset
    original_size = len(dataset)

    if n_samples is None:
        n_samples = original_size

    if replace:
        indices = np.random.choice(original_size, n_samples, replace=True)
    else:
        indices = np.random.choice(
            original_size, min(n_samples, original_size), replace=False
        )

    # Convert numpy indices to Python ints to avoid indexing issues
    indices = indices.tolist()

    bootstrap_dataset = Subset(dataset, indices)

    return DataLoader(
        bootstrap_dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=getattr(dataloader, "num_workers", 0),
        pin_memory=getattr(dataloader, "pin_memory", False),
        collate_fn=getattr(dataloader, "collate_fn", None),
    )


def get_bootstrapped_edge_scores(
    model: Any,
    dataloader: DataLoader,
    metric_fn: Callable,
    n_bootstraps: int = 20,
    ig_steps: int = 5,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run bootstrap analysis and return edge/node score histories.

    Args:
        model: The transformer model.
        dataloader (DataLoader): Dataloader for attribution.
        metric_fn (Callable): Metric function.
        n_bootstraps (int): Number of bootstrap samples.
        ig_steps (int): Number of IG steps.

    Returns:
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]: Edge and node score histories.
    """
    logging.info(
        f"Calculating confidence interval stats with {n_bootstraps} bootstraps..."
    )
    edge_scores_history = {}

    g_edges_boot = Graph.from_model(model)

    for bootstrap_idx in range(n_bootstraps):
        logging.info(f"Processing bootstrap {bootstrap_idx + 1}/{n_bootstraps}")
        bootstrap_dataloader = bootstrap_sample_dataloader(dataloader)

        g_edges_boot.reset(empty=True)

        attribute(
            model,
            g_edges_boot,
            bootstrap_dataloader,
            partial(metric_fn, loss=True, mean=True),
            method="EAP-IG-inputs",
            ig_steps=ig_steps,
        )
        for edge_name, edge in g_edges_boot.edges.items():
            if edge_name not in edge_scores_history:
                edge_scores_history[edge_name] = []
            edge_scores_history[edge_name].append(abs(float(edge.score)))

    return edge_scores_history


def select_best_edges_with_ilp(G: nx.MultiDiGraph, n: int):
    sources = [v for v, d in G.in_degree() if d == 0]
    sinks = [v for v, d in G.out_degree() if d == 0]
    source, sink = sources[0], sinks[0]

    prob = pulp.LpProblem(
        "MaxScoreSubgraph", pulp.LpMaximize
    )  # LP: maximize sum(score_e * x_e)

    # binary var x_e for each edge, y_v for each node
    x = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        x[(u, v, k, data["name"])] = pulp.LpVariable(f"x_{u}_{v}_{k}", cat="Binary")
    y = {}
    for v in G.nodes():
        y[v] = pulp.LpVariable(f"y_{v}", cat="Binary")

    # Objective
    prob += pulp.lpSum(
        data["score"] * x[(u, v, k, data["name"])]
        for u, v, k, data in G.edges(keys=True, data=True)
    )

    # 1) Edge count constraint
    prob += pulp.lpSum(x.values()) <= n

    # 2) Node-edge consistency: if edge selected, its endpoints are used
    for u, v, k, edge_name in x:
        prob += x[(u, v, k, edge_name)] <= y[u]
        prob += x[(u, v, k, edge_name)] <= y[v]

    # 3) Connectivity constraints source and sink must be used
    prob += y[source] == 1
    prob += y[sink] == 1

    # every used non-source node has ≥1 incoming selected edge
    for v in G.nodes():
        if v == source:
            continue
        in_edges = [x[e] for e in x if e[1] == v]
        prob += pulp.lpSum(in_edges) >= y[v]

    # every used non-sink node has ≥1 outgoing selected edge
    for v in G.nodes():
        if v == sink:
            continue
        out_edges = [x[e] for e in x if e[0] == v]
        prob += pulp.lpSum(out_edges) >= y[v]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    # prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=300, threads=4, gapRel=0.01))

    # extract selected edges
    selected_edges = [
        (u, v, k, edge_name)
        for (u, v, k, edge_name), var in x.items()
        if var.value() > 0.5
    ]
    return selected_edges


def build_greedy_graph(g_edges: Graph, edge_percent: float, abs=True) -> Graph:
    """
    Build graph using greedy edge selection.

    Args:
        g_edges (Graph): Graph with edge attribution scores.
        edge_percent (float): Percentage of edges to keep.

    Returns:
        Graph: Pruned graph with selected edges.
    """
    n_edges = int(len(g_edges.edges) * edge_percent)
    g_edges.apply_greedy(n_edges=n_edges, absolute=abs, reset=True, prune=True)
    return g_edges


def build_ilp_graph(g_edges: Graph, edge_percent: float, abs=True) -> Graph:
    n_edges = int(len(g_edges.edges) * edge_percent)
    if abs:
        g_edges.scores = g_edges.scores.abs()
    g_nx = g_edges.to_networkx()
    selected_ilp_edges = select_best_edges_with_ilp(g_nx, n=n_edges)

    g_edges.reset()
    for u, v, _, edge_name in selected_ilp_edges:
        g_edges.edges[edge_name].in_graph = True

    return g_edges


def calculate_auc(
    faithfulnesses: List[float], percentages: List[float], log_scale: bool = False
) -> Tuple[float, float, float]:
    """
    Calculate area under curve metrics.

    Args:
        faithfulnesses (List[float]): List of faithfulness scores.
        percentages (List[float]): List of corresponding percentages.
        log_scale (bool): Whether to use log scale for x-axis.

    Returns:
        Tuple[float, float, float]: Area under curve, area from 1, and average faithfulness.
    """
    if not faithfulnesses or not percentages:
        return 0.0, 0.0, 0.0

    area_under = 0.0
    area_from_1 = 0.0

    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i + 1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]

        if log_scale:
            x_1 = math.log(x_1) if x_1 > 0 else float("-inf")
            x_2 = math.log(x_2) if x_2 > 0 else float("-inf")

        # Calculate trapezoidal area
        trapezoidal_from_1 = (x_2 - x_1) * (
            (abs(1.0 - faithfulnesses[i_1]) + abs(1.0 - faithfulnesses[i_2])) / 2
        )
        area_from_1 += trapezoidal_from_1

        trapezoidal = (x_2 - x_1) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal

    average = sum(faithfulnesses) / len(faithfulnesses) if faithfulnesses else 0.0
    return area_under, area_from_1, average


def evaluate_faithfulness_across_percentages(
    model: HookedTransformer,
    g_edges: Graph,
    dataloader: DataLoader,
    metric_fn: callable,
    baseline: float,
    args: argparse.Namespace,
) -> Tuple[List[float], float]:
    """
    Evaluate faithfulness across different percentages of edges.

    Args:
        model (HookedTransformer): The transformer model.
        g_edges (Graph): Graph with edge attribution scores.
        dataloader (DataLoader): Data loader for the dataset.
        metric_fn (callable): Metric function for evaluation.
        baseline (float): Baseline performance score.
        args (argparse.Namespace): Command-line arguments for caching and parameters.

    Returns:
        Tuple[List[float], float]: List of faithfulness scores and AUC score.
    """
    logging.info(
        f"Evaluating faithfulness across {len(args.percentages)} percentages using {args.method} method"
    )

    faithfulnesses = []
    sums_of_edge_scores = []

    # If using bootstrapping, calculate edge candidates and replace the graph scores
    if "bootstrapping" in args.method:
        attribution_cache_filename = f"{args.model}_{args.task}_bootstrapping={args.n_bootstraps}_{args.num_examples}_{args.ig_steps}_seed={args.seed}_attribution.pt"
        attribution_cache_path = os.path.join(
            args.output_dir, "cache", attribution_cache_filename
        )
        if os.path.exists(attribution_cache_path):
            edge_scores_history = torch.load(attribution_cache_path)[
                "edge_scores_history"
            ]
        else:
            edge_scores_history = get_bootstrapped_edge_scores(
                model,
                dataloader,
                metric_fn,
                n_bootstraps=args.n_bootstraps,
                ig_steps=args.ig_steps,
            )
            torch.save(
                {"edge_scores_history": edge_scores_history},
                attribution_cache_path,
            )

        for edge_name, scores in edge_scores_history.items():
            if not args.no_abs:
                scores = [abs(score) for score in scores]
            mean_score = sum(scores) / len(scores)
            g_edges.edges[edge_name].score = mean_score

    for i, edge_percent in enumerate(args.percentages):
        logging.info(
            f"Evaluating percentage {i+1}/{len(args.percentages)}: {edge_percent}"
        )

        # Build graph for this percentage
        if "greedy" in args.method:
            graph = build_greedy_graph(g_edges, edge_percent, abs=not args.no_abs)
        elif "ilp" in args.method:
            graph = build_ilp_graph(g_edges, edge_percent, abs=not args.no_abs)
        else:
            raise ValueError(f"Unsupported method: {args.method}")

        print(
            f"Percentage: {edge_percent}, Edges (pruned / selected): {graph.in_graph.sum().item()}/{int(len(graph.edges) * edge_percent)}"
        )

        # Evaluate faithfulness
        results = evaluate_graph(
            model,
            graph,
            dataloader,
            partial(metric_fn, loss=False, mean=False),
            quiet=True,
        )
        faith = results.mean().item() / baseline
        faithfulnesses.append(faith)
        logging.info(f"Percentage {edge_percent}: Faithfulness = {faith:.4f}")

        sum_of_edge_scores = sum(
            [e.score.item() for e in graph.edges.values() if e.in_graph]
        )
        sums_of_edge_scores.append(sum_of_edge_scores)

    sum_of_all_edges = sum([e.score for e in graph.edges.values()])

    # Calculate AUC
    auc_score, _, _ = calculate_auc(
        faithfulnesses + [1.0], args.percentages + [1.0], log_scale=False
    )
    logging.info(f"AUC Score: {auc_score:.4f}")

    return (
        args.percentages + [1.0],
        faithfulnesses + [1.0],
        sums_of_edge_scores + [sum_of_all_edges.item()],
        auc_score,
    )


def main():
    """
    Main function to run the smart graph evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate smart graph building methods for circuit discovery"
    )

    # Model and task arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="gpt2-small",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "ioi",
            "mcqa",
            "arithmetic_addition",
            "arithmetic_subtraction",
            "arc_easy",
            "arc_challenge",
        ],
        help="Task to evaluate on",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Graph building arguments
    parser.add_argument(
        "--method",
        type=str,
        default="greedy",
        choices=["greedy", "ilp", "greedy+bootstrapping", "ilp+bootstrapping"],
        help="Method to use for graph building",
    )

    # Smart method arguments
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=10,
        help="Number of bootstrap samples for smart methods",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for confidence intervals (0.0-1.0)",
    )
    parser.add_argument(
        "--significance-threshold",
        type=float,
        default=0.0,
        help="Significance threshold for confidence intervals",
    )

    # Data and training arguments
    parser.add_argument(
        "--num-examples",
        type=int,
        default=500,
        help="Number of examples to use from dataset",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for data loading"
    )
    parser.add_argument(
        "--ig-steps", type=int, default=5, help="Number of integrated gradient steps"
    )
    parser.add_argument(
        "--no-abs",
        action="store_true",
        help="Create graph based on real scores (not absolute values of scores).",
    )
    # parser.add_argument(
    #     "--abs-scores",
    #     type=bool,
    #     default=True,
    #     help="Use absolute scores for edge selection (Use False for CPR eval, True for CMD eval)",
    # )

    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        help="Percentages to evaluate for AUC calculation",
    )

    # Logging and output arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results"
    )

    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Do not save detailed results to a JSON file (saved by default).",
    )

    parser.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Do not generate and save a plot for AUC evaluations (saved by default).",
    )

    parser.add_argument(
        "--baseline-method-for-plot",
        type=str,
        default="greedy",
        help="Method to use as a baseline in the plot (e.g., 'greedy').",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/mnt/nlp",
        help="Directory to save cached results",
    )

    args = parser.parse_args()

    set_deterministic(args.seed)

    # Setup logging
    setup_logging(args.log_level)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Load model and dataset
        model = load_model(args.model, args.cache_dir)
        dataloader, metric_fn, baseline = load_dataset_and_metric(
            args.task,
            model.tokenizer,
            model,
            args.num_examples,
            args.batch_size,
            args.cache_dir,
        )

        # === Attribution Caching ===
        attribution_cache_filename = f"{args.model}_{args.task}_{args.num_examples}_{args.ig_steps}_attribution.pt"
        attribution_cache_path = os.path.join(cache_dir, attribution_cache_filename)

        if os.path.exists(attribution_cache_path):
            logging.info(
                f"Loading cached attribution scores from {attribution_cache_path}"
            )
            cached_data = torch.load(attribution_cache_path)
            g_edges = dict_to_graph(cached_data["g_edges"])
            g_nodes = dict_to_graph(cached_data["g_nodes"])
        else:
            # Perform attribution if no cache file exists
            g_edges, g_nodes = perform_attribution(
                model, dataloader, metric_fn, args.ig_steps
            )
            logging.info(f"Saving attribution scores to {attribution_cache_path}")

            # Save the serializable dictionary representation of the graphs
            torch.save(
                {"g_edges": graph_to_dict(g_edges), "g_nodes": graph_to_dict(g_nodes)},
                attribution_cache_path,
            )

        # Evaluate across percentages
        percentages, faithfulnesses, sums_of_edge_scores, auc_score = (
            evaluate_faithfulness_across_percentages(
                model,
                g_edges,
                dataloader,
                metric_fn,
                baseline,
                args=args,
            )
        )

        # Print results
        print(f"\nResults for {args.model} on {args.task} using {args.method} method:")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nFaithfulness scores:")
        for i, (percent, faith) in enumerate(zip(percentages, faithfulnesses)):
            print(f"  {percent}: {faith:.4f}")
        print("\nSum of edge scores:")
        for i, (percent, sum_score) in enumerate(zip(percentages, sums_of_edge_scores)):
            print(f"  {percent}: {sum_score:.4f}")

        # --- Save results to JSON and Plot ---
        relevant_metric = "CPR" if args.no_abs else "CMD"
        if not args.no_save_json or not args.no_save_plot:
            date_hour = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            results_data = {
                "evaluation_info": {
                    "model": args.model,
                    "task": args.task,
                    "method": args.method,
                    "relevant_metric": relevant_metric,
                    "num_examples": args.num_examples,
                    "ig_steps": args.ig_steps,
                    "timestamp": date_hour,
                },
                "method_parameters": {
                    k: v
                    for k, v in vars(args).items()
                    if k
                    in [
                        "n_bootstraps",
                        "confidence_level",
                        "significance_threshold",
                    ]
                },
                "results": {
                    "baseline_performance": baseline,
                    "auc_score": auc_score,
                    "percentages": percentages,
                    "faithfulnesses": faithfulnesses,
                    "sums_of_edge_scores": sums_of_edge_scores,
                },
            }

            if not args.no_save_json:
                file_name = f"{args.model}_{args.task}_{args.method}_{relevant_metric}_auc_{auc_score:.4f}_{date_hour}.json"
                file_path = os.path.join(args.output_dir, file_name)
                logging.info(f"Saving JSON results to {file_path}")
                with open(file_path, "w") as f:
                    json.dump(results_data, f, indent=4)

            if not args.no_save_plot:
                baseline_results_path = None
                if args.baseline_method_for_plot:
                    pattern = f"{args.model}_{args.task}_{args.baseline_method_for_plot}_{relevant_metric}"
                    files = [
                        f
                        for f in os.listdir(args.output_dir)
                        if f.startswith(pattern) and f.endswith(".json")
                    ]
                    if files:
                        files.sort(reverse=True)
                        baseline_results_path = os.path.join(args.output_dir, files[0])
                        logging.info(
                            f"Found baseline results for plotting: {baseline_results_path}"
                        )
                    else:
                        logging.warning(
                            f"No baseline JSON file found for pattern '{pattern}'."
                        )

                plot_auc_results(results_data, args.output_dir, baseline_results_path)

        logging.info("Evaluation completed successfully")

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
