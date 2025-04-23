import argparse
import os
from functools import partial

from transformer_lens import HookedTransformer

from dataset import HFEAPDataset
from eap.graph import Graph
from eap.attribute import attribute
from eap.attribute_node import attribute_node
from metrics import get_metric
from print_results import COL_MAPPING

TASKS_TO_HF_NAMES = {
    'ioi': 'ioi',
    'mcqa': 'copycolors_mcqa',
    'arithmetic_addition': 'arithmetic_addition',
    'arithmetic_subtraction': 'arithmetic_subtraction',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
}

MODEL_NAME_TO_FULLNAME = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', required=True)
    parser.add_argument("--tasks", type=str, nargs='+', required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--ablation", type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument("--optimal_ablation_path", type=str, default=None)
    parser.add_argument("--level", type=str, choices=['node', 'neuron', 'edge'], default='edge')
    parser.add_argument("--split", type=str, choices=['train', 'validation', 'test'], default='train')
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--circuit-dir", type=str, default='circuits')
    args = parser.parse_args()

    for model_name in args.models:
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name])
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True
        for task in args.tasks:
            if f"{task.replace('_', '-')}_{model_name}" not in COL_MAPPING:
                continue
            graph = Graph.from_model(model)
            hf_task_name = f'mib-bench/{TASKS_TO_HF_NAMES[task]}'
            dataset = HFEAPDataset(hf_task_name, model.tokenizer, split=args.split, task=task, model_name=model_name, num_examples=args.num_examples)
            if args.head is not None:
                head = args.head
                if len(dataset) < head:
                    print(f"Warning: dataset has only {len(dataset)} examples, but head is set to {head}; using all examples.")
                    head = len(dataset)
                dataset.head(head)
            dataloader = dataset.to_dataloader(batch_size=args.batch_size)
            metric = get_metric('logit_diff', task, model.tokenizer, model)
            attribution_metric = partial(metric, mean=True, loss=True)
            if args.level == 'edge':
                attribute(model, graph, dataloader, attribution_metric, args.method, args.ablation, 
                          ig_steps=args.ig_steps, optimal_ablation_path=args.optimal_ablation_path)
            else:
                attribute_node(model, graph, dataloader, attribution_metric, args.method, 
                               args.ablation, neuron=args.level == 'neuron', ig_steps=args.ig_steps,
                               optimal_ablation_path=args.optimal_ablation_path)

            # Save the graph
            method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
            circuit_path = os.path.join(args.circuit_dir, method_name_saveable, f"{task.replace('_', '-')}_{model_name}")
            os.makedirs(circuit_path, exist_ok=True)
            
            graph.to_pt(f'{circuit_path}/importances.pt')
