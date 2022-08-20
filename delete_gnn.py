import os
import json
import wandb
import pickle
import argparse
import torch
import torch.nn as nn
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from framework import get_model, get_trainer
from framework.models.gcn import GCN
from framework.training_args import parse_args
from framework.utils import *
from train_mi import MLPAttacker


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_args(path):
    with open(path, 'r') as f:
        d = json.load(f)
    parser = argparse.ArgumentParser()
    for k, v in d.items():
        parser.add_argument('--' + k, default=v)
    try:
        parser.add_argument('--df_size', default=0.5)
    except:
        pass
    args = parser.parse_args()

    for k, v in d.items():
        setattr(args, k, v)

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
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    attack_path_all = os.path.join(args.checkpoint_dir, args.dataset, 'member_infer_all', str(args.random_seed))
    attack_path_sub = os.path.join(args.checkpoint_dir, args.dataset, 'member_infer_sub', str(args.random_seed))

    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, 
        '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Dataset:', dataset, data)
    args.in_dim = dataset.num_features

    wandb.init(config=args)

    # Df and Dr
    assert args.df != 'none'
    df_size = int(args.df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')

    df_mask_all = torch.load(os.path.join(args.data_dir, args.dataset, f'df_{args.random_seed}.pt'))[args.df]
    df_nonzero = df_mask_all.nonzero().squeeze()

    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    # df_idx = [int(i) for i in args.df_idx.split(',')]
    # df_idx_global = df_mask.nonzero()[df_idx]
    
    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False

    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    data.dr_mask = dr_mask
    data.df_mask = df_mask
    data.edge_index = data.train_pos_edge_index[:, dr_mask]

    assert df_mask.sum() == len(df_global_idx)
    assert dr_mask.shape[0] - len(df_global_idx) == data.train_pos_edge_index[:, dr_mask].shape[1]
    data.dtrain_mask = dr_mask


    # Edges in S_Df
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        2, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    data.sdf_mask = two_hop_mask

    # Nodes in S_Df
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        1, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop

    # Model
    model = get_model(args, sdf_node_1hop, sdf_node_2hop)

    if args.unlearning_model != 'retrain':  # Start from trained GNN model
        if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
            logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt')).to(device)
        else:
            logits_ori = None

        model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'))
        model.load_state_dict(model_ckpt['model_state'], strict=False)
   
    else:       # Initialize a new GNN model
        retrain = None
        logits_ori = None

    model = model.to(device)

    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
    
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters()])
    
    optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)#, weight_decay=args.weight_decay)
    
    wandb.watch(model, log_freq=100)

    # MI attack model
    attack_model_all = None
    # attack_model_all = MLPAttacker(args)
    # attack_ckpt = torch.load(os.path.join(attack_path_all, 'attack_model_best.pt'))
    # attack_model_all.load_state_dict(attack_ckpt['model_state'])
    # attack_model_all = attack_model_all.to(device)

    attack_model_sub = None
    # attack_model_sub = MLPAttacker(args)
    # attack_ckpt = torch.load(os.path.join(attack_path_sub, 'attack_model_best.pt'))
    # attack_model_sub.load_state_dict(attack_ckpt['model_state'])
    # attack_model_sub = attack_model_sub.to(device)

    # Train
    trainer = get_trainer(args)
    trainer.train(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    # Test
    if args.unlearning_model != 'retrain':
        retrain_path = args.checkpoint_dir.replace(args.unlearning_model, 'retrain')
        retrain_ckpt = torch.load(os.path.join(retrain_path, 'model_best.pt'))
        retrain_args = load_args(os.path.join(retrain_path, 'training_args.json'))
        retrain = get_model(retrain_args)
        retrain.load_state_dict(retrain_ckpt['model_state'])
        retrain = retrain.to(device)
        retrain.eval()

    else:
        retrain = None
    
    trainer.test(model, data, model_retrain=retrain, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    trainer.save_log()


if __name__ == "__main__":
    main()
