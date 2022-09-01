import os
import time
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer
from ..evaluation import *
from ..utils import *


def weight(model):
    t = 0
    for p in model.parameters():
        t += torch.norm(p)
    
    return t

class GradientAscentTrainer(Trainer):

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            return self.train_minibatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_loss = 100000

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before


        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.df_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.df_mask.sum())

            z = model(data.x, data.train_pos_edge_index)
            logits = model.decode(z, data.train_pos_edge_index[:, data.df_mask])
            label = torch.ones_like(logits, dtype=torch.float, device='cuda')
            loss = -F.binary_cross_entropy_with_logits(logits, label)

            print('aaaaaaaaaaaaaa', data.df_mask.sum(), weight(model))

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
            }
            wandb.log(log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
            tqdm.write(' | '.join(msg))

            valid_loss, auc, aup, df_logt, logit_all_pair = self.eval(model, data, 'val')

            self.trainer_log['log'].append({
                'epoch': epoch, 
                'dt_loss': valid_loss,
                'dt_auc': auc,
                'dt_aup': aup,
            })

        self.trainer_log['training_time'] = time.time() - start_time

        # Save
        ckpt = {
            'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        best_metric = 0

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        data.edge_index = data.train_pos_edge_index
        data.node_id = torch.arange(data.x.shape[0])
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()

            epoch_loss = 0
            epoch_time = 0
            for step, batch in enumerate(tqdm(loader, leave=False)):
                start_time = time.time()
                batch = batch.to('cuda')

                z = model(batch.x, batch.edge_index[:, batch.dtrain_mask])
                
                # Positive and negative sample
                neg_edge_index = negative_sampling(
                    edge_index=batch.edge_index[:, batch.df_mask],
                    num_nodes=z.size(0))

                logits = model.decode(z, batch.edge_index[:, batch.df_mask])
                label = torch.ones_like(logits, dtype=torch.float, device='cuda')
                loss = -F.binary_cross_entropy_with_logits(logits, label)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                end_time = time.time()
                epoch_loss += loss.item()
                epoch_time += end_time - start_time

            epoch_loss /= step
            epoch_time /= step

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step,
                    'train_time': epoch_time / step,
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if dt_auc + df_auc > best_metric:
                    best_metric = dt_auc + df_auc
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))
