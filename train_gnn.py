import os
import wandb
import pickle
import torch

from framework import get_model, get_trainer
from framework.models.gcn import GCN
from framework.training_args import parse_args
from framework.trainer.base import Trainer


df_size = 800
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args = parse_args()
    if args.unlearning_model == 'original':
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))
    elif args.unlearning_model == 'retrain':
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, f'{args.df}-{args.random_seed}')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Dataset:', dataset, data)
    args.in_dim = dataset.num_features

    wandb.init(config=args)

    # Use proper training data for original and Dr
    if args.unlearning_model == 'original':
        data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    
    elif args.unlearning_model == 'retrain':
        df_mask = torch.load(os.path.join(args.data_dir, args.dataset, f'df_{args.random_seed}.pt'))
        df_mask = df_mask[args.df]
        data.dtrain_mask = ~df_mask

        assert data.dtrain_mask.sum().item() == data.train_pos_edge_index.shape[1] - df_size

    else:
        raise NotImplementedError

    # Model
    model = get_model(args).to(device)
    wandb.watch(model, log_freq=100)
    # if args.dataset in ['Cora']:
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)

    # Train
    trainer = Trainer(args)
    trainer.train(model, data, optimizer, args)

    # Test
    trainer.test(model, data)
    trainer.save_log()


if __name__ == "__main__":
    main()
