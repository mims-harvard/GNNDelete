import os
import wandb
import pickle
import torch

from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.trainer.base import Trainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_edge_type_mapping = {
    'FB15k-237': 237,
    'WordNet18RR': 11
}

def main():
    args = parse_args()
    args.unlearning_model = 'original'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Dataset:', dataset, data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = dataset.num_features

    wandb.init(config=args)

    # Use proper training data for original and Dr
    data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    if args.gnn in ['rgcn', 'rgat']:
        data.edge_index_mask = data.dtrain_mask.repeat(2)

    # Model
    if args.gnn in ['rgcn', 'rgat']:
        num_edge_type = num_edge_type_mapping[args.dataset]
    else:
        num_edge_type = None
    
    model = get_model(args, num_edge_type=num_edge_type).to(device)
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
