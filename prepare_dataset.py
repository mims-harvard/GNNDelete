import os
import pickle
import torch
import networkx as nx
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, to_networkx
from framework.utils import *


data_dir = './data'
df_size = 100
seeds = [42, 21, 13, 87, 100]
pyg_datasets = ['Cora', 'PubMed']
other_datasets = ['FreeBase', 'WikiData']
os.makedirs(data_dir, exist_ok=True)


def main():
    for d in pyg_datasets:
        for s in seeds:
            print('Processing:', d, s)
            seed_everything(s)

            dataset = Planetoid(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
            data = dataset[0]
            print(dataset.data)
            data.train_mask = data.val_mask = data.test_mask = data.y = None

            # D
            data = train_test_split_edges(data)
            print(data)

            with open(os.path.join(data_dir, d, f'd_{s}.pkl'), 'wb') as f:
                pickle.dump((dataset, data), f)

            # Two ways to sample Df from the training set
            ## 1. Df is within 2 hop local enclosing subgraph of Dtest
            ## 2. Df is outside of 2 hop local enclosing subgraph of Dtest
            
            # All the candidate edges (train edges)
            graph = to_networkx(Data(edge_index=data.train_pos_edge_index, x=data.x))

            # Get the 2 hop local enclosing subgraph for all test edges
            local_edges = set()
            for i in range(data.test_pos_edge_index.shape[1]):
                edge = data.test_pos_edge_index[:, i].tolist()
                subgraph = get_enclosing_subgraph(graph, edge)
                local_edges = local_edges | set(subgraph[2])
            
            distant_edges = graph.edges() - local_edges

            local_edges = torch.tensor(sorted(list([i for i in local_edges if i[0] < i[1]])))
            distant_edges = torch.tensor(sorted(list([i for i in distant_edges if i[0] < i[1]])))

            df_in = torch.randperm(local_edges.shape[0])[:df_size]
            df_out = torch.randperm(distant_edges.shape[0])[:df_size]

            df_in = local_edges[df_in]
            df_out = distant_edges[df_out]

            df_in_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
            df_out_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)

            for row in df_in:
                i = (data.train_pos_edge_index.T == row).all(axis=1).nonzero()
                df_in_mask[i] = True

            for row in df_out:
                i = (data.train_pos_edge_index.T == row).all(axis=1).nonzero()
                df_out_mask[i] = True

            torch.save(
                {'out': df_out_mask, 'in': df_in_mask},
                os.path.join(data_dir, d, f'df_{s}.pt')
            )


if __name__ == "__main__":
    main()
