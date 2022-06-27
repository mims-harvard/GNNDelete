import os
import pickle
import torch
import networkx as nx
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull, Coauthor, Flickr
from torch_geometric.utils import train_test_split_edges, k_hop_subgraph
from framework.utils import *


data_dir = './data'
df_size = [i / 100 for i in range(10)] + [i / 10 for i in range(10)] + [i for i in range(10)]       # Df_size in percentage
seeds = [42, 21, 13, 87, 100]
pyg_datasets = ['Cora', 'PubMed', 'DBLP', 'CS', 'Physics', 'Flickr'][:-1]
other_datasets = ['FreeBase', 'WikiData']
os.makedirs(data_dir, exist_ok=True)


def main():
    for d in pyg_datasets:
        for s in seeds:
            print('Processing:', d, s)
            seed_everything(s)

            if d in ['Cora', 'PUbMed', 'DBLP']:
                dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
            if d in ['CS', 'Physics']:
                dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
            if d in ['Flickr']:
                dataset = Flickr(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
            
            data = dataset[0]
            print(dataset.data)
            data.train_mask = data.val_mask = data.test_mask = data.y = None

            # D
            data = train_test_split_edges(data, test_ratio=0.2)
            print(data)

            with open(os.path.join(data_dir, d, f'd_{s}.pkl'), 'wb') as f:
                pickle.dump((dataset, data), f)

            # Two ways to sample Df from the training set
            ## 1. Df is within 2 hop local enclosing subgraph of Dtest
            ## 2. Df is outside of 2 hop local enclosing subgraph of Dtest
            
            # All the candidate edges (train edges)
            # graph = to_networkx(Data(edge_index=data.train_pos_edge_index, x=data.x))

            # Get the 2 hop local enclosing subgraph for all test edges
            _, local_edges, _, mask = k_hop_subgraph(
                data.test_pos_edge_index.flatten().unique(), 
                2, 
                data.train_pos_edge_index, 
                num_nodes=dataset[0].num_nodes)
            distant_edges = data.train_pos_edge_index[:, ~mask]
            print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

            in_mask = mask
            out_mask = ~mask

            # df_in_mask = torch.zeros_like(mask)
            # df_out_mask = torch.zeros_like(mask)

            # df_in_all_idx = in_mask.nonzero().squeeze()
            # df_out_all_idx = out_mask.nonzero().squeeze()
            # df_in_selected_idx = df_in_all_idx[torch.randperm(df_in_all_idx.shape[0])[:df_size]]
            # df_out_selected_idx = df_out_all_idx[torch.randperm(df_out_all_idx.shape[0])[:df_size]]
            
            # df_in_mask[df_in_selected_idx] = True
            # df_out_mask[df_out_selected_idx] = True

            # assert (in_mask & out_mask).sum() == 0
            # assert (df_in_mask & df_out_mask).sum() == 0


            # local_edges = set()
            # for i in range(data.test_pos_edge_index.shape[1]):
            #     edge = data.test_pos_edge_index[:, i].tolist()
            #     subgraph = get_enclosing_subgraph(graph, edge)
            #     local_edges = local_edges | set(subgraph[2])
            
            # distant_edges = graph.edges() - local_edges

            # print('aaaaaaa', len(local_edges), len(distant_edges))
            # local_edges = torch.tensor(sorted(list([i for i in local_edges if i[0] < i[1]])))
            # distant_edges = torch.tensor(sorted(list([i for i in distant_edges if i[0] < i[1]])))


            # df_in = torch.randperm(local_edges.shape[1])[:df_size]
            # df_out = torch.randperm(distant_edges.shape[1])[:df_size]

            # df_in = local_edges[:, df_in]
            # df_out = distant_edges[:, df_out]

            # df_in_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
            # df_out_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)

            # for row in df_in:
            #     i = (data.train_pos_edge_index.T == row).all(axis=1).nonzero()
            #     df_in_mask[i] = True

            # for row in df_out:
            #     i = (data.train_pos_edge_index.T == row).all(axis=1).nonzero()
            #     df_out_mask[i] = True

            torch.save(
                {'out': out_mask, 'in': in_mask},
                os.path.join(data_dir, d, f'df_{s}.pt')
            )


if __name__ == "__main__":
    main()
