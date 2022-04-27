import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from ..evaluation import *
from ..models.gcn import GCN
from ..training_args import parse_args
from ..utils import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.trainer_log = {'log': []}

        with open(os.path.join(self.args.checkpoint_dir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f)

    
    @torch.no_grad()
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def train(self, model, data, optimizer, args):
        best_valid_loss = 1000000
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()
            total_step = 0
            total_loss = 0

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.dtrain_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.dtrain_mask.sum())
            
            z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
            # edge = torch.cat([train_pos_edge_index[:, data.dtrain_mask], neg_edge_index], dim=-1)
            # logits = model.decode(z, edge[0], edge[1])
            logits = model.decode(z, data.train_pos_edge_index[:, data.dtrain_mask], neg_edge_index)
            label = get_link_labels(data.train_pos_edge_index[:, data.dtrain_mask], neg_edge_index)
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

                valid_loss, auc, aup, df_logit = self.eval(model, data, 'val')

                self.trainer_log['log'].append({
                    'train_loss': total_loss / total_step,
                    'valid_loss': valid_loss,
                    'valid_auc': auc,
                    'valid_aup': aup, 
                    'df_logit': df_logit
                })

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        # Save predicted probability of all node pairs
        logit_all_pair = model.decode(z, data.train_pos_edge_index).sigmoid().detach().cpu()
        torch.save(logit_all_pair, os.path.join(args.checkpoint_dir, 'pred_proba.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss


    @torch.no_grad()
    def eval(self, model, data, stage='val'):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        auc = roc_auc_score(label.cpu(), logits.cpu())
        aup = average_precision_score(label.cpu(), logits.cpu())

        if self.args.retrain:
            df_logit = model.decode(z, data.train_pos_edge_index[:, ~data.dtrain_mask]).sigmoid().detach().cpu().item()
        else:
            df_logit = float('nan')

        msg = [
            f'{stage}',
            f'loss: {loss:.6f}',
            f'auc: {auc:.4f}',
            f'aup: {aup:.4f}',
            f'df_logit: {df_logit:.4f}',
        ]
        tqdm.write(' | '.join(msg))

        return loss, auc, aup, df_logit

    @torch.no_grad()
    def test(self, model, data, ckpt=None):
        
        if ckpt is None:    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))

        loss, auc, aup, df_logit = self.eval(model, data, 'test')

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = auc
        self.trainer_log['dt_aup'] = aup
        self.trainer_log['df_logit'] = df_logit

        return loss, auc, aup, df_logit

    @torch.no_grad()
    def get_output(self, model, node_embedding, data):
        model.eval()
        node_embedding = node_embedding.to(device)
        edge = data.edge_index.to(device)
        output = model.decode(node_embedding, edge, edge_type)

        return output

    def save_log(self):
        print(self.trainer_log)
        with open(os.path.join(self.args.checkpoint_dir, 'trainer_log.json'), 'w') as f:
            json.dump(self.trainer_log, f)
