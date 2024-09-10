import pickle
import os
import json
import torch
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--num_layers", "-l", type=int, default=12)
    parser.add_argument("--n_heads", "-a", type=int, default=12)
    parser.add_argument("--d_model", "-d", type=int, default=768)
    parser.add_argument("--threshold", "-t", type=float, default=0.02)
    args = parser.parse_args()

    filename = args.filename
    threshold = args.threshold
    num_layers = args.num_layers

    outname = filename.split(".pt")[0] + ".json"
    config = {"n_layers": num_layers, "n_heads": args.n_heads, "parallel_attn_mlp": False, "d_model": args.d_model}
    nodes = {"input": True, "logits": True}
    neurons = {}
    neuron_scores = {}

    neuron_matrix = torch.load(open(filename, 'rb'))
    has_attn = False
    has_inputs = False
    if len(neuron_matrix) > 1 and neuron_matrix[1].shape[0] > 1:
        has_attn = True
    if len(neuron_matrix) > 1 and neuron_matrix[-1].shape[0] == 1:
        has_inputs = True
 
    # neuron_matrix = neuron_matrix.squeeze(dim=1)

    for layer in range(num_layers):
        nodes[f"m{layer}"] = True
        if f"m{layer}" not in neurons:
            neurons[f"m{layer}"] = []
            neuron_scores[f"m{layer}"] = []
            if has_attn:
                for i in range(config["n_heads"]):
                    neurons[f"a{layer}.h{i}"] = []
                    neuron_scores[f"a{layer}.h{i}"] = []
        for neuron_idx in range(config["d_model"]):
            neurons[f"m{layer}"].append(neuron_matrix[0][layer, 0][neuron_idx].item() > threshold)
            neuron_scores[f"m{layer}"].append(neuron_matrix[0][layer, 0][neuron_idx].item())
        
        if has_attn:
            head_scores = neuron_matrix[1][layer]
            for head_idx in range(config["n_heads"]):
                for neuron_idx in range(config["d_model"]):
                    neurons[f"a{layer}.h{head_idx}"].append(head_scores[head_idx, neuron_idx].item() > threshold)
                    neuron_scores[f"a{layer}.h{head_idx}"].append(head_scores[head_idx, neuron_idx].item())

        if has_inputs:
            neurons["input"] = []
            neuron_scores["input"] = []
            for neuron_idx in range(config["d_model"]):
                neurons["input"].append(neuron_matrix[-1][0][neuron_idx].item() > threshold)
                neuron_scores["input"].append(neuron_matrix[-1][0][neuron_idx].item())

    edges = {}
    for layer1 in range(num_layers):
        for layer2 in range(num_layers):
            if layer1 == layer2:
                continue
            if layer1 > layer2:
                continue
            edges[f"m{layer1}->m{layer2}"] = {"score": 1., "in_graph": True}

    with open(outname, 'w') as outdata:
        all_data = {"cfg": config, "nodes": nodes, "edges": edges,
                    "neurons": neurons, "neuron_scores": neuron_scores}
        outdata.write(json.dumps(all_data))