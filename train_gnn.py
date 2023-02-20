import os
import wandb
import pickle
import torch
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.datasets import RelLinkPredDataset, WordNet18
from torch_geometric.seed import seed_everything

from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.trainer.base import Trainer
from framework.utils import negative_sampling_kg



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    args.unlearning_model = 'original'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Directed dataset:', dataset, data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = dataset.num_features

    wandb.init(config=args)

    # Use proper training data for original and Dr
    if args.gnn in ['rgcn', 'rgat']:
        if not hasattr(data, 'train_mask'):
            data.train_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        # data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
        # data.edge_index_mask = data.dtrain_mask.repeat(2)
        
    else:
        data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)

    # To undirected
    if args.gnn in ['rgcn', 'rgat']:
        r, c = data.train_pos_edge_index
        rev_edge_index = torch.stack([c, r], dim=0)
        rev_edge_type = data.train_edge_type + args.num_edge_type

        data.edge_index = torch.cat((data.train_pos_edge_index, rev_edge_index), dim=1)
        data.edge_type = torch.cat([data.train_edge_type, rev_edge_type], dim=0)
        # data.train_mask = data.train_mask.repeat(2)

        data.dr_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
        assert is_undirected(data.edge_index)
    
    else:
        train_pos_edge_index = to_undirected(data.train_pos_edge_index)
        data.train_pos_edge_index = train_pos_edge_index
        data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
        assert is_undirected(data.train_pos_edge_index)


    print('Undirected dataset:', data)
    
    # Model
    model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type).to(device)
    wandb.watch(model, log_freq=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)

    # Train
    trainer = get_trainer(args)
    trainer.train(model, data, optimizer, args)

    # Test
    trainer.test(model, data)
    trainer.save_log()


if __name__ == "__main__":
    main()
