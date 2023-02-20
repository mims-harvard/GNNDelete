import os
import time
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from ogb.graphproppred import Evaluator
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from ..evaluation import *
from ..training_args import parse_args
from ..utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = 'cpu'

class Trainer:
    def __init__(self, args):
        self.args = args
        self.trainer_log = {
            'unlearning_model': args.unlearning_model, 
            'dataset': args.dataset, 
            'log': []}
        self.logit_all_pair = None
        self.df_pos_edge = []

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

    @torch.no_grad()
    def get_embedding(self, model, data, on_cpu=False):
        original_device = next(model.parameters()).device

        if on_cpu:
            model = model.cpu()
            data = data.cpu()
        
        z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])

        model = model.to(original_device)

        return z

    def train(self, model, data, optimizer, args):
        if self.args.dataset in ['Cora', 'PubMed', 'DBLP', 'CS']:
            return self.train_fullbatch(model, data, optimizer, args)

        if self.args.dataset in ['Physics']:
            return self.train_minibatch(model, data, optimizer, args)

        if 'ogbl' in self.args.dataset:
            return self.train_minibatch(model, data, optimizer, args)

    def train_fullbatch(self, model, data, optimizer, args):
        start_time = time.time()
        best_valid_loss = 1000000

        data = data.to(device)
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.dtrain_mask.sum())
            
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

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item()
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

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

        data.edge_index = data.train_pos_edge_index
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            epoch_loss = 0
            for step, batch in enumerate(tqdm(loader, desc='Step', leave=False)):
                # Positive and negative sample
                train_pos_edge_index = batch.edge_index.to(device)
                z = model(batch.x.to(device), train_pos_edge_index)

                neg_edge_index = negative_sampling(
                    edge_index=train_pos_edge_index,
                    num_nodes=z.size(0))
                
                logits = model.decode(z, train_pos_edge_index, neg_edge_index)
                label = get_link_labels(train_pos_edge_index, neg_edge_index)
                loss = F.binary_cross_entropy_with_logits(logits, label)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                log = {
                    'epoch': epoch,
                    'step': step,
                    'train_loss': loss.item(),
                }
                wandb.log(log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                tqdm.write(' | '.join(msg))

                epoch_loss += loss.item()

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

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
        self.trainer_log['training_time'] = np.mean([i['epoch_time'] for i in self.trainer_log['log'] if 'epoch_time' in i])

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if self.args.eval_on_cpu:
            model = model.to('cpu')
        
        if hasattr(data, 'dtrain_mask'):
            mask = data.dtrain_mask
        else:
            mask = data.dr_mask
        z = model(data.x, data.train_pos_edge_index[:, mask])
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        dt_auc = roc_auc_score(label.cpu(), logits.cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        if self.args.unlearning_model in ['original']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask]).sigmoid().tolist()
            df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        if len(df_logit) > 0:
            df_auc = []
            df_aup = []
        
            # Sample pos samples
            if len(self.df_pos_edge) == 0:
                for i in range(500):
                    mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
                    idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
                    mask[idx] = True
                    self.df_pos_edge.append(mask)
            
            # Use cached pos samples
            for mask in self.df_pos_edge:
                pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()
                
                logit = df_logit + pos_logit
                label = [0] * len(df_logit) +  [1] * len(df_logit)
                df_auc.append(roc_auc_score(label, logit))
                df_aup.append(average_precision_score(label, logit))
        
            df_auc = np.mean(df_auc)
            df_aup = np.mean(df_aup)

        else:
            df_auc = np.nan
            df_aup = np.nan

        # Logits for all node pairs
        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        }

        if self.args.eval_on_cpu:
            model = model.to(device)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args.dataset:
            pred_all = False
        else:
            pred_all = True
        loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = dt_auc
        self.trainer_log['dt_aup'] = dt_aup
        self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup
        self.trainer_log['auc_sum'] = dt_auc + df_auc
        self.trainer_log['aup_sum'] = dt_aup + df_aup
        self.trainer_log['auc_gap'] = abs(dt_auc - df_auc)
        self.trainer_log['aup_gap'] = abs(dt_aup - df_aup)

        # # AUC AUP on Df
        # if len(df_logit) > 0:
        #     auc = []
        #     aup = []

        #     if self.args.eval_on_cpu:
        #         model = model.to('cpu')
            
        #     z = model(data.x, data.train_pos_edge_index[:, data.dtrain_mask])
        #     for i in range(500):
        #         mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
        #         idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
        #         mask[idx] = True
        #         pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()

        #         logit = df_logit + pos_logit
        #         label = [0] * len(df_logit) +  [1] * len(df_logit)
        #         auc.append(roc_auc_score(label, logit))
        #         aup.append(average_precision_score(label, logit))

        #     self.trainer_log['df_auc'] = np.mean(auc)
        #     self.trainer_log['df_aup'] = np.mean(aup)


        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

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

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log

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
    def train(self, model, data, optimizer, args):
        model = model.to(device)
        start_time = time.time()
        best_metric = 0

        print('Num workers:', len(os.sched_getaffinity(0)))
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=128, walk_length=2, num_steps=args.num_steps, num_workers=len(os.sched_getaffinity(0))
        )
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            epoch_loss = 0
            epoch_time = 0
            for step, batch in enumerate(tqdm(loader, desc='Step', leave=False)):
                start_time = time.time()
                batch = batch.to(device)

                # Message passing
                edge_index = batch.edge_index#[:, batch.train_mask]
                edge_type = batch.edge_type#[batch.train_mask]
                z = model(batch.x, edge_index, edge_type)

                # Positive and negative sample
                decoding_mask = (edge_type < args.num_edge_type)       # Only select directed edges for link prediction
                decoding_edge_index = edge_index[:, decoding_mask]
                decoding_edge_type = edge_type[decoding_mask]

                neg_edge_index = negative_sampling_kg(
                    edge_index=decoding_edge_index,
                    edge_type=decoding_edge_type)

                pos_logits = model.decode(z, decoding_edge_index, decoding_edge_type)
                neg_logits = model.decode(z, neg_edge_index, decoding_edge_type)
                logits = torch.cat([pos_logits, neg_logits], dim=-1)
                label = get_link_labels(decoding_edge_index, neg_edge_index)
                # reg_loss = z.pow(2).mean() + model.W.pow(2).mean()
                loss = F.binary_cross_entropy_with_logits(logits, label)# + 1e-2 * reg_loss

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                log = {
                    'epoch': epoch,
                    'step': step,
                    'train_loss': loss.item(),
                }
                wandb.log(log)
                # msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                # tqdm.write(' | '.join(msg))

                epoch_loss += loss.item()
                epoch_time += time.time() - start_time

            if (epoch + 1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step,
                    'epoch_time': epoch_time
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if dt_aup > best_metric:
                    best_metric = dt_aup
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid aup = {best_metric:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_metric'] = best_metric
        self.trainer_log['training_time'] = np.mean([i['epoch_time'] for i in self.trainer_log['log'] if 'epoch_time' in i])

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']
        pos_edge_type = data[f'{stage}_edge_type']
        neg_edge_type = data[f'{stage}_edge_type']

        if self.args.eval_on_cpu:
            model = model.to('cpu')
        
        z = model(data.x, data.edge_index[:, data.dr_mask], data.edge_type[data.dr_mask])

        decoding_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        decoding_edge_type = torch.cat([pos_edge_type, neg_edge_type], dim=-1)
        logits = model.decode(z, decoding_edge_index, decoding_edge_type)
        label = get_link_labels(pos_edge_index, neg_edge_index)

        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        dt_auc = roc_auc_score(label.cpu(), logits.cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        if self.args.unlearning_model in ['original']:
            df_logit = []
        else:
            # df_logit = model.decode(z, data.train_pos_edge_index[:, data.df_mask], data.train_edge_type[data.df_mask]).sigmoid().tolist()
            df_logit = model.decode(z, data.directed_df_edge_index, data.directed_df_edge_type).sigmoid().tolist()

        dr_mask = data.dr_mask[:data.dr_mask.shape[0] // 2]
        if len(df_logit) > 0:
            df_auc = []
            df_aup = []

            for i in range(500):
                mask = torch.zeros(data.train_pos_edge_index[:, dr_mask].shape[1], dtype=torch.bool)
                idx = torch.randperm(data.train_pos_edge_index[:, dr_mask].shape[1])[:len(df_logit)]
                mask[idx] = True
                pos_logit = model.decode(z, data.train_pos_edge_index[:, dr_mask][:, mask], data.train_edge_type[dr_mask][mask]).sigmoid().tolist()

                logit = df_logit + pos_logit
                label = [0] * len(df_logit) +  [1] * len(df_logit)
                df_auc.append(roc_auc_score(label, logit))
                df_aup.append(average_precision_score(label, logit))
        
            df_auc = np.mean(df_auc)
            df_aup = np.mean(df_aup)

        else:
            df_auc = np.nan
            df_aup = np.nan

        # Logits for all node pairs
        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_logit_mean': np.mean(df_logit) if len(df_logit) > 0 else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if len(df_logit) > 0 else np.nan
        }

        if self.args.eval_on_cpu:
            model = model.to(device)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='ckpt'):
        
        if ckpt is 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args.dataset:
            pred_all = False
        else:
            pred_all = True
        loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = dt_auc
        self.trainer_log['dt_aup'] = dt_aup
        self.trainer_log['df_logit'] = df_logit
        self.logit_all_pair = logit_all_pair
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup

        # if model_retrain is not None:    # Deletion
            # self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

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

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log

    def _train(self, model, data, optimizer, args):
        model = model.to(device)
        data = data.to(device)
        start_time = time.time()
        best_valid_loss = 1000000

        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            # Message passing
            z = model(data.x, data.edge_index, data.edge_type)

            # Positive and negative sample
            mask = (data.edge_type < args.num_edge_type)       # Only select directed edges for link prediction

            neg_edge_index = negative_sampling_kg(
                edge_index=data.train_pos_edge_index,
                edge_type=data.train_edge_type)

            pos_logits = model.decode(z, data.train_pos_edge_index, data.train_edge_type)
            neg_logits = model.decode(z, neg_edge_index, data.train_edge_type)
            logits = torch.cat([pos_logits, neg_logits], dim=-1)
            label = get_link_labels(data.train_pos_edge_index, neg_edge_index)
            reg_loss = z.pow(2).mean() + model.W.pow(2).mean()
            loss = F.binary_cross_entropy_with_logits(logits, label) + 1e-2 * reg_loss

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            log = {
                'epoch': epoch,
                'train_loss': loss.item(),
            }
            wandb.log(log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if valid_loss < best_valid_loss:
                    best_valid_loss = dt_auc + df_auc
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
        self.trainer_log['training_time'] = np.mean([i['epoch_time'] for i in self.trainer_log['log'] if 'epoch_time' in i])
        
class NodeClassificationTrainer(Trainer):
    def train(self, model, data, optimizer, args):
        start_time = time.time()
        best_epoch = 0
        best_valid_acc = 0

        data = data.to(device)
        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            z = F.log_softmax(model(data.x, data.edge_index), dim=1)
            loss = F.nll_loss(z[data.train_mask], data.y[data.train_mask])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_acc, dt_f1, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item()
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if dt_acc > best_valid_acc:
                    best_valid_acc = dt_acc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid Acc = {dt_acc:.4f}')
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

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid acc = {best_valid_acc:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_acc'] = best_valid_acc

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()

        if self.args.eval_on_cpu:
            model = model.to('cpu')
        
        # if hasattr(data, 'dtrain_mask'):
        #     mask = data.dtrain_mask
        # else:
        #     mask = data.dr_mask
        z = F.log_softmax(model(data.x, data.edge_index), dim=1)

        # DT AUC AUP
        loss = F.nll_loss(z[data.val_mask], data.y[data.val_mask]).cpu().item()
        pred = torch.argmax(z[data.val_mask], dim=1).cpu()
        dt_acc = accuracy_score(data.y[data.val_mask].cpu(), pred)
        dt_f1 = f1_score(data.y[data.val_mask].cpu(), pred, average='micro')

        # DF AUC AUP
        # if self.args.unlearning_model in ['original', 'original_node']:
        #     df_logit = []
        # else:
        #     df_logit = model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        # if len(df_logit) > 0:
        #     df_auc = []
        #     df_aup = []
        
        #     # Sample pos samples
        #     if len(self.df_pos_edge) == 0:
        #         for i in range(500):
        #             mask = torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
        #             idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
        #             mask[idx] = True
        #             self.df_pos_edge.append(mask)
            
        #     # Use cached pos samples
        #     for mask in self.df_pos_edge:
        #         pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()
                
        #         logit = df_logit + pos_logit
        #         label = [0] * len(df_logit) +  [1] * len(df_logit)
        #         df_auc.append(roc_auc_score(label, logit))
        #         df_aup.append(average_precision_score(label, logit))
        
        #     df_auc = np.mean(df_auc)
        #     df_aup = np.mean(df_aup)

        # else:
        #     df_auc = np.nan
        #     df_aup = np.nan

        # Logits for all node pairs
        if pred_all:
            logit_all_pair = (z @ z.t()).cpu()
        else:
            logit_all_pair = None

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_acc': dt_acc,
            f'{stage}_dt_f1': dt_f1,
        }

        if self.args.eval_on_cpu:
            model = model.to(device)

        return loss, dt_acc, dt_f1, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        if 'ogbl' in self.args.dataset:
            pred_all = False
        else:
            pred_all = True
        loss, dt_acc, dt_f1, test_log = self.eval(model, data, 'test', pred_all)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_acc'] = dt_acc
        self.trainer_log['dt_f1'] = dt_f1
        # self.trainer_log['df_logit'] = df_logit
        # self.logit_all_pair = logit_all_pair
        # self.trainer_log['df_auc'] = df_auc
        # self.trainer_log['df_aup'] = df_aup

        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

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

        return loss, dt_acc, dt_f1, test_log

class GraphTrainer(Trainer):
    def train(self, model, dataset, split_idx, optimizer, args):
        self.train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
        self.valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
        self.test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

        start_time = time.time()
        best_epoch = 0
        best_valid_auc = 0

        for epoch in trange(args.epochs, desc='Epoch'):
            model.train()

            for batch in tqdm(self.train_loader, desc="Iteration", leave=False):
                batch = batch.to(device)
                pred = model(batch)
                optimizer.zero_grad()
                ## ignore nan targets (unlabeled) when computing training loss.
                is_labeled = batch.y == batch.y
                loss = F.binary_cross_entropy_with_logits(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss.backward()
                optimizer.step()

            if (epoch+1) % args.valid_freq == 0:
                valid_auc, valid_log = self.eval(model, dataset, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item()
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(train_log)
                self.trainer_log['log'].append(valid_log)

                if valid_auc > best_valid_auc:
                    best_valid_auc = valid_auc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid auc = {valid_auc:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        self.trainer_log['training_time'] = time.time() - start_time

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best valid auc = {best_valid_auc:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_valid_auc'] = best_valid_auc

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        y_true = []
        y_pred = []

        if stage == 'val':
            loader = self.valid_loader
        else:
            loader = self.test_loader

        if self.args.eval_on_cpu:
            model = model.to('cpu')

        for batch in tqdm(loader):
            batch = batch.to(device)
            pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        evaluator = Evaluator('ogbg-molhiv')
        auc = evaluator.eval({"y_true": y_true, "y_pred": y_pred})['rocauc']
        log = {
            f'val_auc': auc,
        }

        if self.args.eval_on_cpu:
            model = model.to(device)

        return auc, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt='best'):
        
        if ckpt == 'best':    # Load best ckpt
            ckpt = torch.load(os.path.join(self.args.checkpoint_dir, 'model_best.pt'))
            model.load_state_dict(ckpt['model_state'])

        dt_auc, test_log = self.eval(model, data, 'test')
        self.trainer_log['dt_auc'] = dt_auc

        if model_retrain is not None:    # Deletion
            self.trainer_log['ve'] = verification_error(model, model_retrain).cpu().item()
            # self.trainer_log['dr_kld'] = output_kldiv(model, model_retrain, data=data).cpu().item()

        return dt_auc, test_log
