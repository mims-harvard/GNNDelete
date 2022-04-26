import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from framework.models.gcn import GCN
from framework.training_args import parse_args
from framework.trainer.base import Trainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def do_train(model, data, optimizer, args, config):
    best_loss = 100000
    for epoch in trange(args.epochs, desc='Epoch'):
        model.train()
        total_step = 0
        total_loss = 0

        # Positive and negative sample
        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
        
        z = model(data.x, data.train_pos_edge_index)
        # edge = torch.cat([train_pos_edge_index, neg_edge_index], dim=-1)
        # logits = model.decode(z, edge[0], edge[1])
        logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
        label = get_link_labels(data.train_pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(logits, label)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        total_step += 1
        total_loss += loss.item()

        if (epoch+1) % args.valid_freq == 0:
            msg = [
                f'Epoch: {epoch:>4d}', 
                f'train loss: {total_loss / total_step:.6f}'
            ]
            tqdm.write(' | '.join(msg))

            valid_loss, auc, aup = do_eval(model, data, 'val')
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch

                print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                ckpt = {
                    'config': args,
                    'model_state': model.state_dict(),
                    'node_emb': z,
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': total_loss / total_step,
                    'valid_loss': valid_loss,
                    'auc': auc,
                    'aup': aup,
                }
                torch.save(ckpt, os.path.join(args.checkpoint_dir, config + '_best.pt'))
    
    # Save models and node embeddings
    print('Saving final checkpoint')
    ckpt = {
        'config': args,
        'model_state': model.state_dict(),
        'node_emb': z,
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': total_loss / total_step,
        'valid_loss': valid_loss,
        'auc': auc,
        'aup': aup,
    }
    torch.save(ckpt, os.path.join(args.checkpoint_dir, config + '_final.pt'))

    print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_loss:.4f}')


@torch.no_grad()
def do_eval(model, data, stage='val'):
    model.eval()
    pos_edge_index = data[f'{stage}_pos_edge_index']
    neg_edge_index = data[f'{stage}_neg_edge_index']

    z = model(data.x, data.train_pos_edge_index)
    # edge = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    # logits = model.decode(z, edge[0], edge[1]).sigmoid()
    logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
    label = get_link_labels(pos_edge_index, neg_edge_index)

    loss = F.binary_cross_entropy_with_logits(logits, label)
    auc = roc_auc_score(label.cpu(), logits.cpu())
    aup = average_precision_score(label.cpu(), logits.cpu())

    msg = [
        f'{stage}',
        f'loss: {loss:.6f}',
        f'auc: {auc:.4f}',
        f'aup: {aup:.4f}',
    ]
    tqdm.write(' | '.join(msg))

    return loss, auc, aup

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

    if args.retrain:
        args.checkpoint_dir = os.path.join(
            args.checkpoint_dir, args.dataset, args.gnn, 'retrain', 
            '-'.join([str(i) for i in [args.df, args.df_idx, args.random_seed]])) 
    else:
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original') 
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Dataset:', dataset, data)
    args.in_dim = dataset.num_features

    if args.retrain:
        df_mask = torch.load(os.path.join(args.data_dir, args.dataset, f'df_{args.random_seed}.pt'))
        df_mask = df_mask[args.df]
        df_idx = [int(i) for i in args.df_idx.split(',')]
        df_idx_global = df_mask.nonzero()[df_idx]
        
        dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
        dr_mask[df_idx_global] = False

    else:
        dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)

    data.dr_mask = dr_mask

    # Model
    model = GCN(args).to(device)
    # if args.dataset in ['Cora']:
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)

    # Train
    trainer = Trainer(args)
    # trainer.train(model, data, optimizer, args)

    # Test
    trainer.test(model, data)

    trainer.save_log()


if __name__ == "__main__":
    main()
