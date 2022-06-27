import os
import wandb
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.utils import *


df_size = 800
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLPAttacker(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.fc1 = nn.Linear(2 * args.out_dim, args.out_dim)
        self.fc2 = nn.Linear(args.out_dim, args.out_dim//2)
        self.fc3 = nn.Linear(args.out_dim//2, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def main():
    args = parse_args()
    # args.unlearning_model = 'member_infer'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.unlearning_model, str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Dataset:', dataset, data)
    args.in_dim = dataset.num_features

    data = data.to(device)

    wandb.init(config=args)

    # Model
    shadow_model = get_model(args).to(device)
    shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=args.lr)
    
    attack_model = MLPAttacker(args).to(device)
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), args.lr)
    
    wandb.watch(attack_model, log_freq=100)

    # Train shadow model
    trainer = get_trainer(args)
    all_neg = trainer.train_shadow(shadow_model, data, shadow_optimizer, args)
    ckpt = torch.load(os.path.join(args.checkpoint_dir, 'shadow_model_best.pt'))
    shadow_model.load_state_dict(ckpt['model_state'], strict=False)

    # Train attack model
    feature, label = trainer.prepare_attack_training_data(shadow_model, data, all_neg)
    train_feature, valid_feature, train_label, valid_label = train_test_split(feature, label, test_size=0.2, stratify=label)
    
    train_loader = DataLoader(
        TensorDataset(train_feature, train_label),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)
    valid_loader = DataLoader(
        TensorDataset(valid_feature, valid_label),
        batch_size=args.batch_size*8,
        shuffle=False,
        num_workers=8)
        
    trainer.train_attack(attack_model, train_loader, valid_loader, attack_optimizer, args)


    # Test
    # trainer.test_attack(model, data)
    trainer.save_log()


if __name__ == "__main__":
    main()
