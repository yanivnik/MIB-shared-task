# %%
from functools import partial

import torch
from transformer_lens import HookedTransformer
import json

from circuit_loading import load_graph_from_json
import plotly.express as px

from dataset import HFEAPDataset
from metrics import get_metric
from evaluate_graph import evaluate_area_under_curve
# %%
model_name = "qwen"

if model_name == "gpt2-small":
    model_url = "gpt2-small"
elif model_name == "qwen":
    model_url = "Qwen/Qwen2.5-0.5B-Instruct"
else:
    raise Exception('Model name not defined')


model = HookedTransformer.from_pretrained(model_url, device="mps")
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

if model_name == "qwen":
    model.set_ungroup_grouped_query_attention(True)
# %%
path = '/Users/alestolfo/workspace/optimal-ablations/results/pruning'
task = 'ioi'
ablation = 'cf'
name = f'ugs_mib_{model_name}'
# lamb = 0.002

lambs = [.01, 0.002, .001, .0005, .0002, .0001, 1e-5, 5e-6, 2e-6, 1e-6]
lambs = [0.0001]
# lambs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

with open('./hf_token.txt', 'r') as f:
    hf_token = f.read().strip()

if task == "ioi":
    dataset_url = "mech-interp-bench/ioi"
elif task == "arithmetic":
    dataset_url = "mech-interp-bench/arithmetic_addition"
elif task == "arc":
    dataset_url = "mech-interp-bench/arc_easy"
elif task == "mcqa":
    dataset_url = "mech-interp-bench/copycolors_mcqa"
elif task == "greater-than":
    dataset_url = "mech-interp-bench/greater_than"
else:
    raise ValueError(f"Task {task} not recognized")

split = "test"
if split == "train":
    num_examples = None
elif split == "test" and task == "greater-than":
    split = 'train'
    num_examples = None
elif split == "test":
    num_examples = 100
dataset = HFEAPDataset(dataset_url, model.tokenizer, task=task, num_examples=num_examples, model_name=model_name, hf_token=hf_token, split=split)
if split == "train":
    dataset.dataset = dataset.tail(100)
elif split == "train" and task == "greater-than":
    dataset.dataset = dataset.tail(200)
    dataset.dataset = dataset.head(100)
dataloader = dataset.to_dataloader(20)
    
metric_fn = get_metric("logit_diff", task, model.tokenizer, model, model_name=model_name)

results = {}
override = False

for lamb in lambs:

    res_folder = f'{path}/{task}/{ablation}/{name}/{lamb}'

    if not override:
        # check if results already exist
        try:
            with open(f'{res_folder}/results.json', 'r') as f:
                results[lamb] = json.load(f)
            continue
        except FileNotFoundError:
            pass

    graph_path = f'{res_folder}/graph.json'
    g = load_graph_from_json(graph_path)

    res = evaluate_area_under_curve(model, g, dataloader, partial(metric_fn, loss=False, mean=False, prob=True), absolute=True)

    results[lamb] = res

    # store results in a json file
    with open(f'{path}/{task}/{ablation}/{name}/{lamb}/results_{split}{num_examples}.json', 'w') as f:
        json.dump(results, f)
# %%
import plotly.express as px
import pandas as pd
# make line plot with results
percentages = (.1, .2, .5, 1, 2, 5, 10, 20, 50, 100)

# Prepare data for plotting
data = []
for lamb, res in results.items():
    faithfulness_scores = res[-1]
    for percentage, score in zip(percentages, faithfulness_scores):
        data.append({'Percentage': percentage, 'Faithfulness': score, 'Lambda': lamb})

# Create DataFrame
df = pd.DataFrame(data)

# Make line plot with results
fig = px.line(df, x='Percentage', y='Faithfulness', color='Lambda', log_x=True)

# Title and labels
fig.update_layout(title=f'UGS - GPT-2 - {task} - Faithfulness vs. Percentage of Edges Kept',
                  xaxis_title='Percentage of Edges Kept',
                  yaxis_title='Faithfulness')

fig.show()
# %%
