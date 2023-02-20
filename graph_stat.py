import os
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull, Coauthor, Flickr, RelLinkPredDataset, WordNet18RR
from ogb.linkproppred import PygLinkPropPredDataset


data_dir = './data'
datasets = ['Cora', 'PubMed', 'DBLP', 'CS', 'Physics', 'ogbl-citation2', 'ogbl-collab', 'FB15k-237', 'WordNet18RR', 'ogbl-biokg', 'ogbl-wikikg2'][-2:]

def get_stat(d):
    if d in ['Cora', 'PubMed', 'DBLP']:
        dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    if d in ['CS', 'Physics']:
        dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    if d in ['Flickr']:
        dataset = Flickr(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    if 'ogbl' in d:
        dataset = PygLinkPropPredDataset(root=os.path.join(data_dir, d), name=d)

    data = dataset[0]
    print(d)
    print('Number of nodes:', data.num_nodes)
    print('Number of edges:', data.num_edges)
    print('Number of max deleted edges:', int(0.05 * data.num_edges))
    if hasattr(data, 'edge_type'):
        print('Number of nodes:', data.edge_type.unique().shape)

def main():
    for d in datasets:
        get_stat(d)   

if __name__ == "__main__":
    main()
