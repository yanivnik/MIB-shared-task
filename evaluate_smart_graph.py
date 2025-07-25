import argparse
from datetime import datetime
import logging
import math
import random
import sys
import os
import torch
import numpy as np
import networkx as nx
import json
import pulp
from typing import Callable, List, Tuple, Dict, Any, Optional
from functools import partial
from scipy.stats import norm


# Add EAP-IG to path
sys.path.append("EAP-IG/src")

from torch.utils.data import DataLoader, Subset
from transformer_lens import HookedTransformer


from eap.graph import Graph, Edge
from eap.attribute import attribute
from eap.evaluate import evaluate_graph, evaluate_baseline
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES
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


def load_model(model_name: str, model_data_dir: str = "") -> HookedTransformer:
    """
    Load and configure the transformer model.

    Args:
        model_name (str): Name of the model to load.
        model_data_dir (str): Directory to load the model from.

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
            cache_dir=os.path.join(model_data_dir, "models"),
        )
    else:
        model = HookedTransformer.from_pretrained(
            full_model_name, cache_dir=os.path.join(model_data_dir, "models")
        )

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
    model_data_dir: str = "",
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
    dataset = HFEAPDataset(dataset_name, model_tokenizer, task=task_name, split="train")
    val_dataset = HFEAPDataset(
        dataset_name, model_tokenizer, task=task_name, split="validation"
    )

    if num_examples is not None:
        dataset.head(num_examples)
        val_dataset.head(num_examples)  # To avoid evaluating too many entries
        logging.info(
            f"Using {num_examples} examples out of {len(dataset)} from dataset"
        )
    else:
        logging.info(f"Using all {len(dataset)} examples from dataset")

    dataloader = dataset.to_dataloader(batch_size=batch_size)
    val_dataloader = val_dataset.to_dataloader(batch_size=batch_size)

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
        evaluate_baseline(
            model, val_dataloader, partial(metric_fn, loss=False, mean=False)
        )
        .mean()
        .item()
    )

    logging.info(
        f"Dataset loaded. Baseline performance (on validation dataset): {baseline:.4f}"
    )
    return dataloader, val_dataloader, metric_fn, baseline


def perform_attribution(
    model: HookedTransformer,
    dataloader: DataLoader,
    metric_fn: callable,
    ig_steps: int = 5,
) -> Graph:
    """
    Perform attribution on both edges and nodes.

    Args:
        model (HookedTransformer): The transformer model.
        dataloader (DataLoader): Data loader for the dataset.
        metric_fn (callable): Metric function for attribution.
        ig_steps (int): Number of integrated gradient steps.

    Returns:
        Graph: Edge graph and node graph with attribution scores.
    """
    logging.info("Starting attribution process...")

    # Create graphs for edge and node attribution
    g_edges = Graph.from_model(model)

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

    logging.info("Attribution completed successfully")
    return g_edges


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
            edge_scores_history[edge_name].append(float(edge.score))

    return edge_scores_history


def select_best_edges_with_ilp(
    G: nx.MultiDiGraph,
    n: int,
    pnr: Optional[float] = None,
    positive_edge_names: Optional[List[str]] = None,
) -> List[Tuple[str, str, int, str]]:
    """ """
    assert (pnr and positive_edge_names) or (not pnr and not positive_edge_names)

    sources = [v for v, d in G.in_degree() if d == 0]
    sinks = [v for v, d in G.out_degree() if d == 0]
    input_node, output_node = sources[0], sinks[0]

    prob = pulp.LpProblem("GetHighestScoringSubgraphWithRatio", pulp.LpMaximize)

    # binary var x_e for each edge, y_v for each node
    x = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        x[(u, v, k, data["name"])] = pulp.LpVariable(f"x_{u}_{v}_{k}", cat="Binary")
    y = {}
    for v in G.nodes():
        y[v] = pulp.LpVariable(f"y_{v}", cat="Binary")

    # Objective - maximize the sum of edge scores
    prob += pulp.lpSum(
        data["score"] * x[(u, v, k, data["name"])]
        for u, v, k, data in G.edges(keys=True, data=True)
    )

    # Edge budgest constraint
    prob += pulp.lpSum(x.values()) <= n

    # Node-edge consistency: if edge selected, its endpoints are used
    for u, v, k, edge_name in x:
        prob += x[(u, v, k, edge_name)] <= y[u]
        prob += x[(u, v, k, edge_name)] <= y[v]

    # Connectivity constraints: input and output must be used
    prob += y[input_node] == 1
    prob += y[output_node] == 1

    # Every used non-source node has ≥1 incoming selected edge
    for v in G.nodes():
        if v == input_node:
            continue
        in_edges = [x[e] for e in x if e[1] == v]
        prob += pulp.lpSum(in_edges) >= y[v]

    # Every used non-sink node has ≥1 outgoing selected edge
    for v in G.nodes():
        if v == output_node:
            continue
        out_edges = [x[e] for e in x if e[0] == v]
        prob += pulp.lpSum(out_edges) >= y[v]

    # Positive-to-Negative ratio constraint
    if pnr is not None:
        assert positive_edge_names is not None
        # Make sure at least pnr positive edges are included (if possible, otherwise make sure all positive edges are used)
        required_pos = min(int(pnr * n), len(positive_edge_names))
        positive_edges = [e for e in x if G.edges[e[:3]]["name"] in positive_edge_names]
        if len(positive_edges) < required_pos:
            # If not enough positive edges, use all positive edges
            required_pos = len(positive_edges)
        prob += pulp.lpSum(x[e] for e in positive_edges) >= required_pos

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    # prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=300, threads=4, gapRel=0.01))

    # extract selected edges
    selected_edges = [
        (u, v, k, edge_name)
        for (u, v, k, edge_name), var in x.items()
        if var.value() > 0.5
    ]
    return selected_edges


def build_greedy_graph(
    g_edges: Graph,
    edge_percent: float,
    forced_abs=False,
    preferred_edges: Optional[List[Edge]] = None,
) -> Graph:
    """
    Build graph using greedy edge selection.

    Args:
        g_edges (Graph): Graph with edge attribution scores.
        edge_percent (float): Percentage of edges to keep.

    Returns:
        Graph: Pruned graph with selected edges.
    """
    # if preferred_edges is None:
    n_edges = int(len(g_edges.edges) * edge_percent)
    g_edges.apply_greedy(n_edges=n_edges, absolute=forced_abs, reset=True, prune=True)
    return g_edges
    # else:
    #     # First build a graph from the preferred edges only
    #     n_edges = int(len(g_edges.edges) * edge_percent)
    #     n_preffered_edges = min(n_edges, len(preferred_edges))
    #     g_edges.apply_greedy(n_edges=n_preffered_edges, absolute=)

    #     # Now add edges from remaining list


def build_ilp_graph(
    g_edges: Graph,
    edge_percent: float,
    pnr: Optional[float] = None,
    forced_abs: bool = False,
) -> Graph:
    n_edges = int(len(g_edges.edges) * edge_percent)
    positive_edge_names = (
        None if pnr is None else [e.name for e in g_edges.edges.values() if e.score > 0]
    )
    if forced_abs:
        g_edges.scores = g_edges.scores.abs()
    g_nx = g_edges.to_networkx()

    selected_ilp_edges = select_best_edges_with_ilp(
        g_nx, n_edges, pnr, positive_edge_names
    )

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
        dataloader (DataLoader): Data loader for the evaluation dataset.
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

    pnr = args.pnr if "ratio" in args.method else None

    # If using bootstrapping, calculate edge candidates and replace the graph scores
    preferred_edges = {}
    if "bootstrapping" in args.method:
        set_deterministic(args.seed)
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

        alpha = 1 - args.confidence_level
        z_score = norm.ppf(1 - alpha / 2)  # Two-tailed test

        # Filter out edges based on bootstrapping lower bound (i.e. confidence interval)
        for edge_name, scores in edge_scores_history.items():
            assert len(scores) >= 3  # Need minimum samples for filtering
            mean_score = np.mean(scores)
            g_edges.edges[edge_name].score = mean_score
            std_score = np.std(scores, ddof=1)
            margin_of_error = z_score * (std_score / np.sqrt(len(scores)))
            lower_bound = mean_score - margin_of_error
            upper_bound = mean_score + margin_of_error
            if (mean_score > 0 and lower_bound > args.significance_threshold) or (
                mean_score < 0 and upper_bound < args.significance_threshold
            ):
                preferred_edges[edge_name] = mean_score

    def abs_id(x):
        if args.metric == "CPR":
            return x
        else:
            return abs(x)

    for i, edge_percent in enumerate(args.percentages):
        logging.info(
            f"Evaluating percentage {i+1}/{len(args.percentages)}: {edge_percent}"
        )

        if "bootstrapping" in args.method and "ilp" in args.method:
            # Set the score of all non-preferred edges to 0 (so they won't be chosen as positive and will get chosen last by abs score)
            for edge_name, edge in g_edges.edges.items():
                if edge_name not in preferred_edges:
                    edge.score = 0.0

            # TODO Refactor to one line
            if "ratio" in args.method:
                graph = build_ilp_graph(g_edges, edge_percent, pnr=pnr, forced_abs=True)
            else:
                graph = build_ilp_graph(
                    g_edges, edge_percent, forced_abs=(args.metric == "CMD")
                )

        elif "bootstrapping" in args.method and "ratio" in args.method:
            # Top-N assumption
            n_edges_to_select = int(len(g_edges.edges) * edge_percent)
            sorted_edge_candidates = sorted(
                preferred_edges, key=lambda x: abs(preferred_edges[x]), reverse=True
            )
            signed_sorted_edge_candidates = sorted(
                preferred_edges, key=lambda x: preferred_edges[x], reverse=True
            )

            num_positive_edges = int(n_edges_to_select * pnr)

            # Select top X% edges
            significant_edges = signed_sorted_edge_candidates[:num_positive_edges]
            remaining_n_edges = n_edges_to_select - num_positive_edges

            # Add remaining edges based on absolute score
            remaining_edges = [
                x for x in sorted_edge_candidates if x not in significant_edges
            ][:remaining_n_edges]
            significant_edges.extend(remaining_edges)

            if len(significant_edges) < n_edges_to_select:
                remaining_n_edges = n_edges_to_select - len(significant_edges)
                remaining_edges = sorted(
                    [x for x in g_edges.edges.items() if x[0] not in significant_edges],
                    key=lambda x: abs(x[1].score),
                    reverse=True,
                )
                significant_edges.extend(remaining_edges[:remaining_n_edges])

            g_edges.reset(empty=True)
            for edge_name in significant_edges:
                if edge_name in g_edges.edges:
                    g_edges.edges[edge_name].in_graph = True

            g_edges.prune()
            graph = g_edges

        elif "ratio" in args.method and "ilp" in args.method:
            graph = build_ilp_graph(g_edges, edge_percent, pnr=pnr, forced_abs=True)

        elif "bootstrapping" in args.method:
            # Top-N assumption
            n_edges_to_select = int(len(g_edges.edges) * edge_percent)
            sorted_edge_candidates = sorted(
                preferred_edges.keys(),
                key=lambda x: abs_id(preferred_edges[x]),
                reverse=True,
            )
            significant_edges = sorted_edge_candidates[:n_edges_to_select]

            if len(significant_edges) < n_edges_to_select:
                remaining_n_edges = n_edges_to_select - len(significant_edges)
                remaining_edges = sorted(
                    [x for x in g_edges.edges.items() if x[0] not in significant_edges],
                    key=lambda x: abs_id(x[1].score),
                    reverse=True,
                )
                significant_edges.extend(remaining_edges[:remaining_n_edges])

            g_edges.reset(empty=True)
            for edge_name in significant_edges:
                if edge_name in g_edges.edges:
                    g_edges.edges[edge_name].in_graph = True

            g_edges.prune()
            graph = g_edges

        elif "ratio" in args.method:
            # Assume top-N
            edges = [e for e in g_edges.edges.values()]

            n_edges_to_select = int(len(g_edges.edges) * edge_percent)
            sorted_edges = sorted(edges, key=lambda x: abs(x.score), reverse=True)
            signed_sorted_edges = sorted(edges, key=lambda x: x.score, reverse=True)
            num_positive_edges = int(n_edges_to_select * pnr)

            significant_edges = signed_sorted_edges[:num_positive_edges]
            remaining_n_edges = n_edges_to_select - num_positive_edges
            remaining_edges = [x for x in sorted_edges if x not in significant_edges][
                :remaining_n_edges
            ]
            significant_edges.extend(remaining_edges)

            g_edges.reset(empty=True)
            for edge in significant_edges:
                if edge.name in g_edges.edges:
                    g_edges.edges[edge.name].in_graph = True

            g_edges.prune()
            graph = g_edges

        elif "ilp" in args.method:
            logging.info("Building graph with ILP")
            graph = build_ilp_graph(
                g_edges, edge_percent, forced_abs=args.metric == "CMD"
            )

        elif "greedy" in args.method:
            logging.info("Building graph with Greedy mechanism")
            graph = build_greedy_graph(
                g_edges, edge_percent, forced_abs=args.metric == "CMD"
            )

        else:
            raise ValueError(f"Unknown methods: {args.method}")

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
    auc_score, dist_from_one, _ = calculate_auc(
        faithfulnesses + [1.0], args.percentages + [1.0], log_scale=False
    )

    return (
        args.percentages + [1.0],
        faithfulnesses + [1.0],
        sums_of_edge_scores + [sum_of_all_edges.item()],
        auc_score,
        dist_from_one,
    )


def parse_args():
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
        nargs="+",
        default=["greedy"],
        choices=["greedy", "ilp", "bootstrapping", "ratio"],
        help="Methods to use for graph building (can specify multiple methods)",
    )

    # Smart method arguments
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=10,
        help="Number of bootstrap samples for smart methods",
    )

    parser.add_argument(
        "--pnr",
        type=float,
        default=-1.0,
        help='Positive-to-Negative ratios to be used when using "ratio" method. None means "ratio" method isn\'t used.',
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
        "--percentages",
        type=float,
        nargs="+",
        default=[0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        help="Percentages to evaluate for AUC calculation",
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=["CPR", "CMD"],
        help="Metric to use for evaluation (CPR - Maximize Faithfulness or CMD - Maximize being near y=1.0)",
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
        "--output-dir",
        type=str,
        default="/mnt/nlp/shared_projects/MIB-shared-task/results",
        help="Directory to save results",
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
        "--model-data-dir",
        type=str,
        default="/mnt/nlp",
        help="Directory to save cached results",
    )

    args = parser.parse_args()

    # Validate arguments
    if "ratio" in args.method:
        if args.pnr is None or args.pnr < 0:
            raise ValueError(
                "Positive-to-Negative ratio (pnr) must be specified when using 'ratio' method."
            )
        if args.metric == "CPR":
            raise ValueError(
                "CPR metric is not compatible with 'ratio' method. When using CPR you want to use only positive scores."
            )

    return args


def main():
    """
    Main function to run the smart graph evaluation.
    """
    args = parse_args()

    set_deterministic(args.seed)

    # Setup logging
    setup_logging(args.log_level)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    attribution_cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(attribution_cache_dir, exist_ok=True)

    try:
        # Load model and dataset
        model = load_model(args.model, args.model_data_dir)
        dataloader, val_dataloader, metric_fn, baseline = load_dataset_and_metric(
            args.task,
            model.tokenizer,
            model,
            args.num_examples,
            args.batch_size,
            args.model_data_dir,
        )

        # === Attribution Caching ===
        attribution_cache_filename = f"{args.model}_{args.task}_{args.num_examples}_{args.ig_steps}_attribution.pt"
        attribution_cache_path = os.path.join(
            attribution_cache_dir, attribution_cache_filename
        )

        if os.path.exists(attribution_cache_path):
            logging.info(
                f"Loading cached attribution scores from {attribution_cache_path}"
            )
            cached_data = torch.load(attribution_cache_path)
            g_edges = dict_to_graph(cached_data["g_edges"])
        else:
            # Perform attribution if no cache file exists
            g_edges = perform_attribution(model, dataloader, metric_fn, args.ig_steps)
            logging.info(f"Saving attribution scores to {attribution_cache_path}")

            # Save the serializable dictionary representation of the graphs
            torch.save(
                {"g_edges": graph_to_dict(g_edges)},
                attribution_cache_path,
            )

        # Evaluate across percentages
        percentages, faithfulnesses, sums_of_edge_scores, auc_score, dist_from_one = (
            evaluate_faithfulness_across_percentages(
                model,
                g_edges,
                val_dataloader,
                metric_fn,
                baseline,
                args=args,
            )
        )

        # Print results
        print(
            f"\nResults for {args.model} on {args.task} using {args.method} method for metric {args.metric}{f' with pnr {args.pnr}' if args.pnr >= 0 else ''}:"
        )
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Dist from Y=1.0 Score: {dist_from_one:.4f}")

        print("\nFaithfulness scores:")
        for i, (percent, faith) in enumerate(zip(percentages, faithfulnesses)):
            print(f"  {percent}: {faith:.4f}")
        print("\nSum of edge scores:")
        for i, (percent, sum_score) in enumerate(zip(percentages, sums_of_edge_scores)):
            print(f"  {percent}: {sum_score:.4f}")

        # --- Save results to JSON and Plot ---
        if not args.no_save_json or not args.no_save_plot:
            date_hour = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            results_data = {
                "evaluation_info": {
                    "model": args.model,
                    "task": args.task,
                    "method": args.method,
                    "relevant_metric": args.metric,
                    "pnr": args.pnr,
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
                    "dist_from_one": dist_from_one,
                    "percentages": percentages,
                    "faithfulnesses": faithfulnesses,
                    "sums_of_edge_scores": sums_of_edge_scores,
                },
            }

            if not args.no_save_json:
                file_name = f"{args.model}_{args.task}_{'+'.join(sorted(args.method))}_{args.metric}{f'_pnr{args.pnr}' if args.pnr >= 0 else ''}_auc_{auc_score:.4f}_dist_{dist_from_one:.4f}_{date_hour}.json"
                file_path = os.path.join(args.output_dir, file_name)
                logging.info(f"Saving JSON results to {file_path}")
                with open(file_path, "w") as f:
                    json.dump(results_data, f, indent=4)

            if not args.no_save_plot:
                baseline_results_path = None
                if args.baseline_method_for_plot:
                    pattern = f"{args.model}_{args.task}_{args.baseline_method_for_plot}_{args.metric}"
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
