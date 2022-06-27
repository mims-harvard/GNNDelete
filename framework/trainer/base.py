import os
import time
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from ..evaluation import *
from ..training_args import parse_args
from ..utils import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.trainer_log = {
            'unlearning_model': args.unlearning_model, 
            'dataset': args.dataset, 
            'log': []}
        self.logit_all_pair = None

        with open(os.path.join(self.args.checkpoint_dir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f)

    def freeze_unused_weights(self, model, mask):
        grad_mask = torch.zeros_like(mask)
        grad_mask[mask] = 1

        model.deletion1.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
        model.deletion2.deletion_weight.register_hook(lambda grad: grad.mul_(grad_mask))
    
    @torch.no_grad()
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels

    def train(self, model, data, optimizer, args):
        if self.args.dataset in ['Cora', 'PubMed', 'DBLP', 'CS']:
            return self.train_fullbatch(model, data, optimizer, args)

        if self.args.dataset in ['Physics']:
            return self.train_minibatch(model, data, optimizer, args)        

    def train_fullbatch(self, model, data, optimizer, args):
        start_time = time.time()
        best_valid_loss = 1000000

        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

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

            if (epoch+1) % args.valid_freq == 0:
                log = {
                    'Epoch': epoch,
                    'train_loss': loss.item(),
                }
                wandb.log(log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                tqdm.write(' | '.join(msg))

                valid_loss, auc, aup, df_logit, logit_all_pair = self.eval(model, data, 'val')

                self.trainer_log['log'].append({
                    'train_loss': loss.item(),
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

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss

    def train_minibatch(self, model, data, optimizer, args):
        start_time = time.time()
        best_valid_loss = 1000000

        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

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

            if (epoch+1) % args.valid_freq == 0:
                log = {
                    'Epoch': epoch,
                    'train_loss': loss.item(),
                }
                wandb.log(log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                tqdm.write(' | '.join(msg))

                valid_loss, auc, aup, df_logit, logit_all_pair = self.eval(model, data, 'val')

                self.trainer_log['log'].append({
                    'train_loss': loss.item(),
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

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        auc = roc_auc_score(label.cpu(), logits.cpu())
        aup = average_precision_score(label.cpu(), logits.cpu())

        if self.args.unlearning_model in ['original']:
            df_logit = []
        else:
            df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()

        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_auc': auc,
            f'{stage}_aup': aup,
        }
        wandb.log(log)
        msg = [f'{i}: {j:.4f}' if isinstance(j, (np.floating, float)) else f'{i}: {j:>4d}' for i, j in log.items()]
        tqdm.write(' | '.join(msg))

        return loss, auc, aup, df_logit, logit_all_pair

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        loss, auc, aup, df_logit, logit_all_pair = self.eval(model, data, 'test', True)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = auc
        self.trainer_log['dt_aup'] = aup
        self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair

        # AUC AUP on Df
        if len(df_logit) > 0:
            auc = []
            aup = []
            z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
            for i in range(500):
                mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
                idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
                mask[idx] = True
                pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

                logit = df_logit + pos_logit
                label = [0] * len(df_logit) +  [1] * len(df_logit)
                auc.append(roc_auc_score(label, logit))
                aup.append(average_precision_score(label, logit))

            self.trainer_log['df_auc'] = np.mean(auc)
            self.trainer_log['df_aup'] = np.mean(aup)


        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

        # MI Attack after unlearning
        if attack_model_all is not None:
            mi_logit_all_after, mi_sucrate_all_after = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_after'] = mi_logit_all_after
            self.trainer_log['mi_sucrate_all_after'] = mi_sucrate_all_after
        if attack_model_sub is not None:
            mi_logit_sub_after, mi_sucrate_sub_after = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_after'] = mi_logit_sub_after
            self.trainer_log['mi_sucrate_sub_after'] = mi_sucrate_sub_after
            
            self.trainer_log['mi_ratio_all'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_all_after'], self.trainer_log['mi_logit_all_before'])])
            self.trainer_log['mi_ratio_sub'] = np.mean([i[1] / j[1] for i, j in zip(self.trainer_log['mi_logit_sub_after'], self.trainer_log['mi_logit_sub_before'])])
            print(self.trainer_log['mi_ratio_all'], self.trainer_log['mi_ratio_sub'], self.trainer_log['mi_sucrate_all_after'], self.trainer_log['mi_sucrate_sub_after'])
            print(self.trainer_log['df_auc'], self.trainer_log['df_aup'])

        return loss, auc, aup, df_logit

    @torch.no_grad()
    def get_output(self, model, node_embedding, data):
        model.eval()
        node_embedding = node_embedding.to(device)
        edge = data.edge_index.to(device)
        output = model.decode(node_embedding, edge, edge_type)

        return output

    def save_log(self):
        # print(self.trainer_log)
        with open(os.path.join(self.args.checkpoint_dir, 'trainer_log.json'), 'w') as f:
            json.dump(self.trainer_log, f)
        
        torch.save(self.logit_all_pair, os.path.join(self.args.checkpoint_dir, 'pred_proba.pt'))


class KGTrainer(Trainer):
    def negative_sampling_kg(self, edge_index, edge_type):
        '''Generate negative samples but keep the node type the same'''

        edge_index_copy = edge_index.clone()
        for et in edge_type.unique():
            mask = (edge_type == et)
            old_source = edge_index_copy[0, mask]
            new_index = torch.randperm(old_source.shape[0])
            new_source = old_source[new_index]
            edge_index_copy[0, mask] = new_source
        
        return edge_index_copy
    
    def train(self, model, loader, optimizer, args):
        best_valid_loss = 1000000
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            for batch in loader:
                # Positive and negative sample
                pos_edge_index = batch.edge_index[:, mask]
                pos_edge_type = batch.edge_type[mask]
                neg_edge_index = self.negative_sampling_kg(pos_edge_index, pos_edge_type)
                edge_index = torch.cat([pos_edge_index, pos_edge_type], dim=-1)

                neg_edge_index = negative_sampling_kg(
                    edge_index=data.train_pos_edge_index[:, data.dtrain_mask],
                    num_nodes=data.num_nodes,
                    num_neg_samples=data.dtrain_mask.sum())
            

            # Edge label
            label_pos = torch.ones(edge_pos.size(1)).to(edge_pos.device)
            label_neg = torch.zeros(edge_neg.size(1)).to(edge_pos.device)
            label = torch.cat([label_pos, label_neg], dim=-1)
            
            # Link prediction
            edge_type = torch.cat([edge_type, edge_type], dim=-1)
            proba = self.decode(embedding, edge, edge_type)

            # Calculate loss
            loss = F.binary_cross_entropy(proba, label)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            if (epoch+1) % args.valid_freq == 0:
                log = {
                    'Epoch': epoch,
                    'train_loss': loss.item(),
                }
                wandb.log(log)
                msg = [print(type(j)) for i, j in log.items()]
                tqdm.write(' | '.join(msg))

                valid_loss, auc, aup, df_logit, logit_all_pair = self.eval(model, data, 'val')

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

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid loss = {best_valid_loss:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_loss'] = best_valid_loss


    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        auc = roc_auc_score(label.cpu(), logits.cpu())
        aup = average_precision_score(label.cpu(), logits.cpu())

        if data.dtrain_mask.sum() != data.train_pos_edge_index.shape[1]:    # Deletion
            df_logit = model.decode(z, data.train_pos_edge_index[:, ~data.dtrain_mask]).sigmoid().detach().cpu().item()
        else:                                                               # Original
            df_logit = float('nan')

        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_auc': auc,
            f'{stage}_aup': aup,
            f'{stage}_df_logit': df_logit,
        }
        wandb.log(log)
        msg = [f'{i}: {j:.4f}' if type(j) is float else f'{i}: {j:>4d}' for i, j in log.items()]
        tqdm.write(' | '.join(msg))

        return loss, auc, aup, df_logit, logit_all_pair

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, ckpt=None):
        
        if ckpt is None:    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        loss, auc, aup, df_logit, logit_all_pair = self.eval(model, data, 'test', True)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = auc
        self.trainer_log['dt_aup'] = aup
        self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair

        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

        return loss, auc, aup, df_logit
