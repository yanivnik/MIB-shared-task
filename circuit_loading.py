import json
import torch

from graph import Graph


def load_graph_from_json(json_path: str):
    """
    Load a Graph object from a JSON file.
    The JSON should have the following keys:
        1. 'cfg': Configuration dictionary, containing similar values to a TLens configuration object.
        2. 'nodes': Dict[str, bool] which maps a node name (i.e. 'm11' or 'a0.h11') to a boolean value, indicating if the node is part of the circuit.
        3. 'edges': Dict[str, Dict] which maps an edge name ('node->node') to a dictionary containins values 

    NOTE: This method isn't disk-space efficient, and shouldn't be used when the circuits contains edges between neuron-resolution nodes.
    """

    with open(json_path, 'r') as f:
        d = json.load(f)

    g = Graph.from_model(d['cfg'])
    for name in d['nodes']:
        if type(d['nodes'][name]) == dict:
            in_graph = d['nodes'][name]["in_graph"]
            score = d['nodes'][name]["score"]
        else:
            in_graph = d['nodes'][name]
            score = None

        if name.startswith("m") and "." not in name:    # include all neurons
            if "d_model" in d["cfg"]:
                for neuron in range(d['cfg']['d_model']):
                    g.nodes[f'{name}.{neuron}'].in_graph = in_graph
                    if score is not None:
                        g.nodes[f'{name}.{neuron}'].score = score
            else:
                g.nodes[name].in_graph = in_graph
                if score is not None:
                    g.nodes[name].score = score
        else:
            g.nodes[name].in_graph = in_graph
            g.nodes[name].score = score
        
    for name, info in d['edges'].items():
        g.edges[name].score = info['score']
        g.edges[name].in_graph = info['in_graph']
    
    return g


# TODO: MAYBE CHANGE THE WAY THAT THE EDGES ARE STORED IN THE GRAPH OBJECT TO THE TENSOR BASED ONE, TO SUPPORT EFFICIENT MASKING
def load_graph_from_pt(pt_path):
    """
    Load a graph object from a pytorch-serialized file.
    The file should contain a dict with the following items -
        1. 'cfg': Configuration dictionary, containing similar values to a TLens configuration object.
        2. 'src_nodes': Dict[str, bool] which maps a node name (i.e. 'm11' or 'a0.h11') to a boolean value, indicating if the node is part of the circuit.
        3. 'dst_nodes': List[str] containing the names of the possible destination nodes, in the same order as the edges tensor.
        4. 'edges': torch.tensor[n_src_nodes, n_dst_nodes], where each value in (src, dst) represents the edge score between the src node and dst node.
    """
    d = torch.load(pt_path)
    assert all([k in d.keys() for k in ['cfg', 'src_nodes', 'dst_nodes', 'edges']]), "Bad torch circuit file format - Missing keys"
    assert d['edges'].shape == (len(d['src_nodes']), len(d['dst_nodes'])), "Bad edges array shape"

    g = Graph.from_model(d['cfg'])

    for name, in_graph in d['src_nodes'].items():
        if name.startswith("m") and "." not in name:
            # MLP Node without specific neurons found - include all neurons
            for neuron in range(d['cfg']['d_model']):
                g.nodes[f'{name}.{neuron}'].in_graph = in_graph
        else:
            g.nodes[name].in_graph = in_graph

    # Enumerate over the tensor and fill the edge values in the graph
    for src_idx, src_name in enumerate(d['src_nodes']):
        for dst_idx, dst_name in enumerate(d['dst_nodes']):
            edge_name = f'{src_name}->{dst_name}'
            if edge_name in g.edges.keys():
                g.edges[edge_name].score = d['edges'][src_idx, dst_idx]
                g.edges[edge_name].in_graph = (d['edges'][src_idx, dst_idx] != -torch.inf)

    return g


def _json_to_pt(json_path, output_pt_path=None):
    """
    Utility function to convert a JSON file formatted for load_graph_from_json to a Pytorch file formatted for load_graph_from_pt.

    Args:
        json_path (str): Path to the input JSON file.
        output_pt_path (str, optional): Path to the output PT file. If None, will be saved in the same directory as the input JSON file.
    """
    g = load_graph_from_json(json_path)
    src_nodes = {name: node.in_graph for name, node in g.nodes.items()}
    dst_nodes = set([e.split('->')[1] for e in g.edges.keys()])
    edges = torch.full((len(src_nodes), len(dst_nodes)), -torch.inf)
    
    for src_idx, src_name in enumerate(src_nodes):
        for dst_idx, dst_name in enumerate(dst_nodes): # TODO MAKE MORE EFFICIENT - ONLY SCAN WHEN src < dst in ORDER                
            edge_name = f'{src_name}->{dst_name}'
            if edge_name in g.edges:
                edges[src_idx, dst_idx] = g.edges[edge_name].score if g.edges[edge_name].in_graph else -torch.inf

    torch.save({
        'cfg': g.cfg,
        'src_nodes': src_nodes,
        'dst_nodes': dst_nodes,
        'edges': edges
    }, (output_pt_path or json_path.replace('.json', '.pt')))