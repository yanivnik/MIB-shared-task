# %%
import json
import torch
from transformer_lens import HookedTransformer

from graph import Graph
# %%
path = '/Users/alestolfo/workspace/optimal-ablations/results/pruning'
task = 'ioi'
ablation = 'cf'
name = 'ugs_mib'
# lamb = 0.001
lambs = [0.01, 0.002, .001, .0005, .0002, .0001, 1e-5, 5e-6, 2e-6, 1e-6]
lambs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

# load an empty graph to use as reference
model_name = "gpt2-small"
model = HookedTransformer.from_pretrained(model_name, device="mps")
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True
g = Graph.from_model(model)
print(f'Loaded graph with {len(g.nodes)} nodes and {len(g.edges)} edges')
print(f'Graph has {g.real_edge_mask.sum()} real edges')


for lamb in lambs:
    res_folder = f'{path}/{task}/{ablation}/{name}/{lamb}'
    snapshot_path = f'{res_folder}/snapshot.pth'

    # load snapshot
    snapshot = torch.load(snapshot_path, map_location=torch.device('cpu'))

    sampling_params = {}
    thetas = {}
    total_param_count = 0
    for key, params in snapshot['pruner_dict'].items():
        layer_idx = int(key.split('.')[-1])
        edge_type = key.split('.')[-2]
        if edge_type not in sampling_params:
            sampling_params[edge_type] = {}
            thetas[edge_type] = {}
        sampling_params[edge_type][layer_idx] = params
        thetas[edge_type][layer_idx] = 1 / (1 + torch.exp(-params))
        total_param_count += params.numel()
    print(f'Loaded {total_param_count} parameters')

    
    # convert thetas to edge scores
    edges = {}
    for layer_idx in range(g.cfg['n_layers']):
        for i, letter in enumerate('qkv'):
            # curr_thetas_attn: n_heads x num_prev_layers (= layer_idx) x n_heads
            curr_thetas_attn = thetas['attn-attn'][layer_idx][i]
            # curr_thetas_mlp: n_heads x num_prev_layers (= layer_idx) + 1 
            curr_thetas_mlp = thetas['mlp-attn'][layer_idx][i]

            # add attn-attn edges
            for src_layer_idx in range(layer_idx):
                for src_head_idx in range(curr_thetas_attn.shape[2]):
                    for dest_head_idx in range(curr_thetas_attn.shape[0]):
                        edges[f'a{src_layer_idx}.h{src_head_idx}->a{layer_idx}.h{dest_head_idx}<{letter}>'] = curr_thetas_attn[dest_head_idx, src_layer_idx, src_head_idx].item()

            # add mlp-attn edges
            for src_layer_idx in range(layer_idx+1):
                for src_head_idx in range(curr_thetas_mlp.shape[0]): 
                    if src_layer_idx == 0:
                        src_str = 'input'
                    else:
                        src_str = f'm{src_layer_idx-1}'
                    edges[f'{src_str}->a{layer_idx}.h{src_head_idx}<{letter}>'] = curr_thetas_mlp[src_head_idx, src_layer_idx].item() # not sure about this, why dim 1 is num_prev_layers + 1? beacuse mlp 0 is the input?

        # curr_thetas_attn: num_prev_layers (= layer_idx) + 1 x n_heads
        curr_thetas_attn = thetas['attn-mlp'][layer_idx]
        # curr_thetas_mlp: num_prev_layers (= layer_idx) 
        curr_thetas_mlp = thetas['mlp-mlp'][layer_idx]

        # add attn-mlp edges
        for src_layer_idx in range(layer_idx+1):
            for src_head_idx in range(curr_thetas_attn.shape[1]):
                edges[f'a{src_layer_idx}.h{src_head_idx}->m{layer_idx}'] = curr_thetas_attn[src_layer_idx, src_head_idx].item()

        # add mlp-mlp edges
            if src_layer_idx == 0:
                src_str = 'input'
            else:
                src_str = f'm{src_layer_idx-1}'

            edges[f'{src_str}->m{layer_idx}'] = curr_thetas_mlp[src_layer_idx].item()


    # last mlp are the logits? check this
    curr_thetas_attn = thetas['attn-mlp'][layer_idx+1]
    curr_thetas_mlp = thetas['mlp-mlp'][layer_idx+1]

    for src_layer_idx in range(model.cfg.n_layers):
        for src_head_idx in range(curr_thetas_attn.shape[0]):
            edges[f'a{src_layer_idx}.h{src_head_idx}->logits'] = curr_thetas_attn[src_head_idx, src_layer_idx].item()

    for src_layer_idx in range(model.cfg.n_layers+1):
        if src_layer_idx == 0:
            src_str = 'input'
        else:
            src_str = f'm{src_layer_idx-1}'

        edges[f'{src_str}->logits'] = curr_thetas_mlp[src_layer_idx].item()

    print(f'Converted thetas to {len(edges)} edges')


    # check if there are missing edges
    missing = []
    excess = []
    for edge_name in g.edges.keys():
        if edge_name not in edges.keys():
            missing.append(edge_name)

    for edge_name in edges.keys():
        if edge_name not in g.edges.keys():
            excess.append(edge_name)

    print(f'Missing edges: {len(missing)}')
    print(f'Excess edges: {len(excess)}')

    
    # format and save graph
    edges_formatted = {k : { "score": v, "in_graph": False } for k, v in edges.items()}
    nodes_dict = { name: {"in_graph": False} for name in g.nodes.keys() }
    dict_to_store = {}
    cfg_dict = { "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads, "parallel_attn_mlp": False, "d_model": model.cfg.d_model } 
    dict_to_store["cfg"] = cfg_dict
    dict_to_store["edges"] = edges_formatted
    dict_to_store["nodes"] = nodes_dict
    
    dest_path = f'{res_folder}/graph.json'
    with open(dest_path, 'w') as f:
        json.dump(dict_to_store, f)


# %%
# extra: look into scores
import plotly.express as px

# make histogram of scores
fig = px.histogram(x=list(edges.values()), nbins=500)

fig.show()
# %%
{k:v for k,v in edges.items() if '->logits' in k}
# %%
