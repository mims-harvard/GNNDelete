import os
import wandb
import pickle
import torch
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_undirected, is_undirected
import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull, Coauthor, Flickr, RelLinkPredDataset, WordNet18, WordNet18RR
from torch_geometric.seed import seed_everything

from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.trainer.base import NodeClassificationTrainer
from framework.utils import negative_sampling_kg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = parse_args()
    args.checkpoint_dir = 'checkpoint_node'
    args.dataset = 'DBLP'
    args.unlearning_model = 'original'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Dataset
    dataset = CitationFull(os.path.join(args.data_dir, args.dataset), args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    print('Original data', data)

    split = T.RandomNodeSplit()
    data = split(data)
    assert is_undirected(data.edge_index)

    print('Split data', data)
    args.in_dim = data.x.shape[1]
    args.out_dim = dataset.num_classes

    wandb.init(config=args)

    # Model
    model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type).to(device)
    wandb.watch(model, log_freq=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)

    # Train
    trainer = NodeClassificationTrainer(args)
    trainer.train(model, data, optimizer, args)

    # Test
    trainer.test(model, data)
    trainer.save_log()


if __name__ == "__main__":
    main()
