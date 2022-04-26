import os
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from framework.models.gcn import GCN
from framework.training_args import parse_args
from framework.trainer.retrain import RetrainTrainer
from framework.trainer.gradient_ascent import GradientAscentTrainer
from framework.trainer.gradient_ascent_with_mp import GradientAscentWithMessagePassingTrainer
from framework.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_args(path):
    with open(path, 'r') as f:
        args = json.load(f)
    parser = argparse.ArgumentParser()
    for k, v in args.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args()

    return args

@torch.no_grad()
def get_node_embedding(model, data):
    model.eval()
    node_embedding = model(data.x.to(device), data.edge_index.to(device))

    return node_embedding

@torch.no_grad()
def get_output(model, node_embedding, data):
    model.eval()
    node_embedding = node_embedding.to(device)
    edge = data.edge_index.to(device)
    output = model.decode(node_embedding, edge, edge_type)

    return output

def main():
    args = parse_args()
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original')

    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
        '-'.join([str(i) for i in [args.df, args.df_idx, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Dataset:', dataset, data)
    args.in_dim = dataset.num_features

    # Df and Dr
    assert args.df != 'none'
    df_mask = torch.load(os.path.join(args.data_dir, args.dataset, f'df_{args.random_seed}.pt'))
    df_mask = df_mask[args.df]
    df_idx = [int(i) for i in args.df_idx.split(',')]
    df_idx_global = df_mask.nonzero()[df_idx]
    
    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_idx_global] = False

    ga_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    ga_mask[df_idx_global] = True

    data.dr_mask = dr_mask
    data.ga_mask = ga_mask
    data.edge_index = data.train_pos_edge_index[:, dr_mask]

    assert ga_mask.sum() == len(df_idx)
    assert dr_mask.shape[0] - len(df_idx) == data.train_pos_edge_index[:, dr_mask].shape[1]


    # Unlearning training data
    # graph = to_networkx(Data(edge_index=data.train_pos_edge_index, x=data.x))
    # edge_to_delete = data['train_pos_edge_index'][:, df_mask][:, df_idx[0]].tolist()
    # subgraph = get_enclosing_subgraph(graph, edge_to_delete)
    # unlearn_train_data = Data(train_pos_edge_index=torch.tensor(edge_to_delete).unsqueeze(0).T, x=data.x)

    # Model
    retrain_ckpt = torch.load(os.path.join(args.checkpoint_dir, 'model_best.pt'))
    retrain_args = load_args(os.path.join(args.checkpoint_dir, 'training_args.json'))
    retrain = GCN(retrain_args)
    retrain.load_state_dict(retrain_ckpt['model_state'])
    retrain = retrain.to(device)
    retrain.eval()

    model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'))
    model_args = load_args(os.path.join(original_path, 'training_args.json'))
    model = GCN(model_args)
    model.load_state_dict(model_ckpt['model_state'])
    model = model.to(device)
    
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)

    # Train
    if args.unlearning_model == 'retrain':
        trainer = RetrainTrainer(args)
    else:
        trainer = GradientAscentTrainer(args)

    trainer.train(retrain, model, data, optimizer, args)


if __name__ == "__main__":
    main()
