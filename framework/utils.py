import numpy as np
import torch
import networkx as nx


def get_node_edge(graph):
    degree_sorted_ascend = sorted(graph.degree, key=lambda x: x[1])

    return degree_sorted_ascend[-1][0]

def h_hop_neighbor(G, node, h):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items() if length == h]
                    
def get_enclosing_subgraph(graph, edge_to_delete):
    subgraph = {0: [edge_to_delete]}
    s, t = edge_to_delete
    
    neighbor_s = []
    neighbor_t = []
    for h in range(1, 2+1):
        neighbor_s += h_hop_neighbor(graph, s, h)
        neighbor_t += h_hop_neighbor(graph, t, h)
        
        nodes = neighbor_s + neighbor_t + [s, t]
        
        subgraph[h] = list(graph.subgraph(nodes).edges())
        
    return subgraph

@torch.no_grad()
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels
