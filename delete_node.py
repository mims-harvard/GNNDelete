import os
import copy
import json
import wandb
import pickle
import argparse
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, to_networkx, k_hop_subgraph, is_undirected
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull, Coauthor, Flickr, RelLinkPredDataset, WordNet18, WordNet18RR
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.seed import seed_everything

from framework import get_model, get_trainer
from framework.models.gcn import GCN
from framework.models.deletion import GCNDelete
from framework.training_args import parse_args
from framework.utils import *
from framework.trainer.gnndelete_nodeemb import GNNDeleteNodeClassificationTrainer
from train_mi import MLPAttacker


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    return torch.cat([row[mask], col[mask]], dim=0)

def main():
    args = parse_args()
    args.checkpoint_dir = 'checkpoint_node'
    args.dataset = 'DBLP'
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original', str(args.random_seed))
    attack_path_all = os.path.join(args.checkpoint_dir, args.dataset, 'member_infer_all', str(args.random_seed))
    attack_path_sub = os.path.join(args.checkpoint_dir, args.dataset, 'member_infer_sub', str(args.random_seed))
    seed_everything(args.random_seed)

    if 'gnndelete' in args.unlearning_model:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, f'{args.unlearning_model}-node_deletion', 
            '-'.join([str(i) for i in [args.loss_fct, args.loss_type, args.alpha, args.neg_sample_random]]),
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    else:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, f'{args.unlearning_model}-node_deletion', 
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

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

    # Df and Dr
    if args.df_size >= 100:     # df_size is number of nodes/edges to be deleted
        df_size = int(args.df_size)
    else:                       # df_size is the ratio
        df_size = int(args.df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.num_nodes:,}')
    print(f'Df size: {df_size:,}')

    # Delete nodes
    df_nodes = torch.randperm(data.num_nodes)[:df_size]
    global_node_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    global_node_mask[df_nodes] = False

    dr_mask_node = global_node_mask
    df_mask_node = ~global_node_mask
    assert df_mask_node.sum() == df_size

    # Delete edges associated with deleted nodes from training set
    res = [torch.eq(data.edge_index, aelem).logical_or_(torch.eq(data.edge_index, aelem)) for aelem in df_nodes]
    df_mask_edge = torch.any(torch.stack(res, dim=0), dim = 0)
    df_mask_edge = df_mask_edge.sum(0).bool()
    dr_mask_edge = ~df_mask_edge

    df_edge = data.edge_index[:, df_mask_edge]
    data.directed_df_edge_index = to_directed(df_edge)
    # print(df_edge.shape, directed_df_edge_index.shape)
    # raise

    print('Deleting the following nodes:', df_nodes)

    # # Delete edges associated with deleted nodes from valid and test set
    # res = [torch.eq(data.val_pos_edge_index, aelem).logical_or_(torch.eq(data.val_pos_edge_index, aelem)) for aelem in df_nodes]
    # mask = torch.any(torch.stack(res, dim=0), dim = 0)
    # mask = mask.sum(0).bool()
    # mask = ~mask
    # data.val_pos_edge_index = data.val_pos_edge_index[:, mask]
    # data.val_neg_edge_index = data.val_neg_edge_index[:, :data.val_pos_edge_index.shape[1]]

    # res = [torch.eq(data.test_pos_edge_index, aelem).logical_or_(torch.eq(data.test_pos_edge_index, aelem)) for aelem in df_nodes]
    # mask = torch.any(torch.stack(res, dim=0), dim = 0)
    # mask = mask.sum(0).bool()
    # mask = ~mask
    # data.test_pos_edge_index = data.test_pos_edge_index[:, mask]
    # data.test_neg_edge_index = data.test_neg_edge_index[:, :data.test_pos_edge_index.shape[1]]


    # For testing
    # data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask_edge]
    # if args.gnn in ['rgcn', 'rgat']:
    #     data.directed_df_edge_type = data.train_edge_type[df_mask]
        
    # Edges in S_Df
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.edge_index[:, df_mask_edge].flatten().unique(), 
        2, 
        data.edge_index,
        num_nodes=data.num_nodes)

    # Nodes in S_Df
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.edge_index[:, df_mask_edge].flatten().unique(), 
        1, 
        data.edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop


    # To undirected for message passing
    # print(is_undir0.0175ected(data.train_pos_edge_index), data.train_pos_edge_index.shape, two_hop_mask.shape, df_mask.shape, two_hop_mask.shape)
    # assert not is_undirected(data.edge_index)
    print(is_undirected(data.edge_index))

    if args.gnn in ['rgcn', 'rgat']:
        r, c = data.train_pos_edge_index
        rev_edge_index = torch.stack([c, r], dim=0)
        rev_edge_type = data.train_edge_type + args.num_edge_type

        data.edge_index = torch.cat((data.train_pos_edge_index, rev_edge_index), dim=1)
        data.edge_type = torch.cat([data.train_edge_type, rev_edge_type], dim=0)
        # data.train_mask = data.train_mask.repeat(2)

        two_hop_mask = two_hop_mask.repeat(2).view(-1)
        df_mask = df_mask.repeat(2).view(-1)
        dr_mask = dr_mask.repeat(2).view(-1)
        assert is_undirected(data.edge_index)
    
    else:
        # train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
        two_hop_mask = two_hop_mask.bool()
        df_mask_edge = df_mask_edge.bool()
        dr_mask_edge = ~df_mask_edge
        
        # data.train_pos_edge_index = train_pos_edge_index
        # assert is_undirected(data.train_pos_edge_index)


    print('Undirected dataset:', data)
    # print(is_undirected(train_pos_edge_index), train_pos_edge_index.shape, two_hop_mask.shape, df_mask.shape, two_hop_mask.shape)

    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask_edge
    data.dr_mask = dr_mask_edge
    data.dtrain_mask = dr_mask_edge
    # print(is_undirected(data.train_pos_edge_index), data.train_pos_edge_index.shape, data.two_hop_mask.shape, data.df_mask.shape, data.two_hop_mask.shape)
    # raise

    # Model
    model = GCNDelete(args)
    # model = get_model(args, sdf_node_1hop, sdf_node_2hop, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)

    if args.unlearning_model != 'retrain':  # Start from trained GNN model
        if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
            logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))
            if logits_ori is not None:
                logits_ori = logits_ori.to(device)
        else:
            logits_ori = None

        model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
        model.load_state_dict(model_ckpt['model_state'], strict=False)
   
    else:       # Initialize a new GNN model
        retrain = None
        logits_ori = None

    model = model.to(device)

    if 'gnndelete' in args.unlearning_model and 'nodeemb' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': 0.0}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])

        if 'layerwise' in args.loss_type:
            optimizer1 = torch.optim.Adam(model.deletion1.parameters(), lr=args.lr)
            optimizer2 = torch.optim.Adam(model.deletion2.parameters(), lr=args.lr)
            optimizer = [optimizer1, optimizer2]
        else:
            optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)

    else:
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
    trainer = GNNDeleteNodeClassificationTrainer(args)
    trainer.train(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    # Test
    if args.unlearning_model != 'retrain':
        retrain_path = os.path.join(
            'checkpoint', args.dataset, args.gnn, 'retrain', 
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
        retrain_ckpt = torch.load(os.path.join(retrain_path, 'model_best.pt'), map_location=device)
        retrain_args = copy.deepcopy(args)
        retrain_args.unlearning_model = 'retrain'
        retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
        retrain.load_state_dict(retrain_ckpt['model_state'])
        retrain = retrain.to(device)
        retrain.eval()

    else:
        retrain = None
    
    trainer.test(model, data, model_retrain=retrain, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
    trainer.save_log()


if __name__ == "__main__":
    main()
