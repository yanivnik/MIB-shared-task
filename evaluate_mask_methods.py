# %%
from functools import partial

import torch
from transformer_lens import HookedTransformer

from circuit_loading import load_graph_from_json
import plotly.express as px

from dataset import HFEAPDataset
from metrics import get_metric
from evaluate_graph import evaluate_graph, evaluate_baseline, evaluate_area_under_curve
# %%
model = HookedTransformer.from_pretrained("gpt2-small", device="mps")
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
# %%
path = '/Users/alestolfo/workspace/optimal-ablations/results/pruning'
task = 'ioi'
ablation = 'cf'
name = 'ugs'
# lamb = 0.002

lambs = [.01, 0.002, .001, .0005, .0002, .0001, 1e-5, 5e-6, 2e-6, 1e-6]
# lambs = [0.001]

dataset = HFEAPDataset("danaarad/ioi_dataset", model.tokenizer, task="ioi", num_examples=100)
dataloader = dataset.to_dataloader(20)
metric_fn = get_metric("logit_diff", "ioi", model.tokenizer, model)

results = {}

for lamb in lambs:

    res_folder = f'{path}/{task}/{ablation}/{name}/{lamb}'
    graph_path = f'{res_folder}/graph.json'
    g = load_graph_from_json(graph_path)

    res = evaluate_area_under_curve(model, g, dataloader, partial(metric_fn, loss=False, mean=False))

    results[lamb] = res

# %%
results
# %%
scores = [e.score for e in g.edges.values()]

fig = px.histogram(x=scores, nbins=500)
fig.show()
# %%
import plotly.express as px
import pandas as pd
# make line plot with results
percentages = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

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
fig.update_layout(title='UGS - GPT-2 - IOI - Faithfulness vs. Percentage of Edges Kept',
                  xaxis_title='Percentage of Edges Kept',
                  yaxis_title='Faithfulness')

fig.show()
# %%
