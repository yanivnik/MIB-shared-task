from typing import Callable, List, Optional, Literal, Tuple
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer

from tqdm import tqdm

from .utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations
from .evaluate import evaluate_graph, evaluate_baseline
from .graph import Graph


def get_scores_exact(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    quiet=False,
):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """

    graph.in_graph |= (
        graph.real_edge_mask
    )  # All edges that are real are now in the graph
    baseline = evaluate_baseline(model, dataloader, metric).mean().item()
    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
        edge.in_graph = False
        intervened_performance = (
            evaluate_graph(
                model,
                graph,
                dataloader,
                metric,
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
                quiet=True,
                skip_clean=True,
            )
            .mean()
            .item()
        )
        edge.score = intervened_performance - baseline
        edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores


def get_scores_nap(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    quiet=False,
):
    """Gets NODE attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device="cuda", dtype=model.cfg.dtype
    )

    if "mean" in intervention:
        assert (
            intervention_dataloader is not None
        ), "Intervention dataloader must be provided for mean interventions"
        per_position = "positional" in intervention
        means = compute_mean_activations(
            model, graph, intervention_dataloader, per_position=per_position
        )
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        )

        with torch.inference_mode():
            if intervention == "patching":
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            elif "mean" in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores


def get_scores_nap_ig(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps=30,
    quiet=False,
):
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
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device="cuda", dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, n_pos_corrupted = tokenize_plus(model, corrupted)

        if n_pos != n_pos_corrupted:
            print(
                f"Number of positions must match, but do not: {n_pos} (clean) != {n_pos_corrupted} (corrupted)"
            )
            print(clean)
            print(corrupted)
            raise ValueError("Number of positions must match")

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[
                :, :, graph.forward_index(graph.nodes["input"])
            ].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = (
                input_activations_corrupted
                - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]
            )

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (
                    input_activations_clean - input_activations_corrupted
                )
                new_input.requires_grad = True
                return new_input

            return hook_fn

        total_steps = 0
        for step in range(0, steps):
            total_steps += 1
            with model.hooks(
                fwd_hooks=[
                    (graph.nodes["input"].out_hook, input_interpolation_hook(step))
                ],
                bwd_hooks=bwd_hooks,
            ):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                if torch.isnan(metric_value).any().item():
                    print("Metric value is NaN")
                    print(f"Clean: {clean}")
                    print(f"Corrupted: {corrupted}")
                    print(f"Label: {label}")
                    print(f"Metric: {metric}")
                    raise ValueError("Metric value is NaN")
                metric_value.backward()

            if torch.isnan(scores).any().item():
                print("Metric value is NaN")
                print(f"Clean: {clean}")
                print(f"Corrupted: {corrupted}")
                print(f"Label: {label}")
                print(f"Metric: {metric}")
                print(f"Step: {step}")
                raise ValueError("Metric value is NaN")

    scores /= total_items
    scores /= total_steps

    return scores


allowed_aggregations = {"sum", "mean"}


def attribute_node(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    method: Literal["NAP", "NAP-IG", "exact"],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    aggregation="sum",
    ig_steps: Optional[int] = None,
    intervention_dataloader: Optional[DataLoader] = None,
    quiet=False,
):

    # assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    # assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    # assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"

    if model.cfg.n_key_value_heads is not None:
        assert (
            model.cfg.ungroup_grouped_query_attention
        ), "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"

    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"aggregation must be in {allowed_aggregations}, but got {aggregation}"
        )

    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == "NAP":
        scores = get_scores_nap(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
        )
    elif method == "NAP-IG":
        scores = get_scores_nap_ig(
            model,
            graph,
            dataloader,
            metric,
            steps=ig_steps,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
        )
    # TODO IMPLEMENT
    # elif method == 'exact':
    #     scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention, intervention_dataloader=intervention_dataloader,
    #                               quiet=quiet)
    else:
        raise ValueError(
            f"method must be in ['NAP', 'NAP-IG', 'exact'], but got {method}"
        )

    if aggregation == "mean":
        scores /= model.cfg.d_model

    graph.scores[:] = scores.to(graph.scores.device)
