import os
import json
import wandb
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
from torch_geometric.utils import negative_sampling
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

from .base import Trainer
from ..evaluation import *
from ..utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MIAttackTrainer(Trainer):
    '''This code is adapted from https://github.com/iyempissy/rebMIGraph'''

    def __init__(self, args):
        self.args = args
        self.trainer_log = {
            'unlearning_model': 'member_infer', 
            'dataset': args.dataset, 
            'seed': args.random_seed,
            'shadow_log': [],
            'attack_log': []}
        self.logit_all_pair = None

        with open(os.path.join(self.args.checkpoint_dir, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f)

    def train_shadow(self, model, data, optimizer, args):
        best_valid_loss = 1000000

        all_neg = []
        # Train shadow model using the test data
        for epoch in trange(args.epochs, desc='Train shadow model'):
            model.train()

            # Positive and negative sample
            neg_edge_index = negative_sampling(
                edge_index=data.test_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.test_pos_edge_index.shape[1])
            
            z = model(data.x, data.test_pos_edge_index)
            logits = model.decode(z, data.test_pos_edge_index, neg_edge_index)
            label = get_link_labels(data.test_pos_edge_index, neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            all_neg.append(neg_edge_index.cpu())

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, auc, aup, df_logit, logit_all_pair = self.eval_shadow(model, data, 'val')
                
                log = {
                    'shadow_epoch': epoch,
                    'shadow_train_loss': loss.item(),
                    'shadow_valid_loss': valid_loss,
                    'shadow_valid_auc': auc,
                    'shadow_valid_aup': aup, 
                    'shadow_df_logit': df_logit
                }
                wandb.log(log)
                self.trainer_log['shadow_log'].append(log)
                
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                tqdm.write(' | '.join(msg))

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch

                    self.trainer_log['shadow_best_epoch'] = best_epoch
                    self.trainer_log['shadow_best_valid_loss'] = best_valid_loss

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'shadow_model_best.pt'))

        return torch.cat(all_neg, dim=-1)

    @torch.no_grad()
    def eval_shadow(self, model, data, stage='val'):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        z = model(data.x, data.val_pos_edge_index)
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        auc = roc_auc_score(label.cpu(), logits.cpu())
        aup = average_precision_score(label.cpu(), logits.cpu())
        df_logit = float('nan')

        logit_all_pair = (z @ z.t()).cpu()

        log = {
            f'{stage}_loss': loss,
            f'{stage}_auc': auc,
            f'{stage}_aup': aup,
            f'{stage}_df_logit': df_logit,
        }
        wandb.log(log)
        msg = [f'{i}: {j:.4f}' if isinstance(j, (np.floating, float)) else f'{i}: {j:>4d}' for i, j in log.items()]
        tqdm.write(' | '.join(msg))

        return loss, auc, aup, df_logit, logit_all_pair
    
    def train_attack(self, model, train_loader, valid_loader, optimizer, args):
        loss_fct = nn.CrossEntropyLoss()
        best_auc = 0
        best_epoch = 0
        for epoch in trange(50, desc='Train attack model'):
            model.train()

            train_loss = 0
            for x, y in train_loader:
                logits = model(x.to(device))
                loss = loss_fct(logits, y.to(device))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()

            valid_loss, valid_acc, valid_auc, valid_f1 = self.eval_attack(model, valid_loader)

            log = {
                'attack_train_loss': train_loss / len(train_loader), 
                'attack_valid_loss': valid_loss,
                'attack_valid_acc': valid_acc, 
                'attack_valid_auc': valid_auc, 
                'attack_valid_f1': valid_f1}
            wandb.log(log)
            self.trainer_log['attack_log'].append(log)

            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
            tqdm.write(' | '.join(msg))


            if valid_auc > best_auc:
                best_auc = valid_auc
                best_epoch = epoch
                self.trainer_log['attack_best_auc'] = valid_auc
                self.trainer_log['attack_best_epoch'] = epoch
                
                ckpt = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                }
                torch.save(ckpt, os.path.join(args.checkpoint_dir, 'attack_model_best.pt'))
        
    @torch.no_grad()
    def eval_attack(self, model, eval_loader):
        loss_fct = nn.CrossEntropyLoss()
        pred = []
        label = []
        for x, y in eval_loader:
            logits = model(x.to(device))
            loss = loss_fct(logits, y.to(device))
            _, p = torch.max(logits, 1)
            
            pred.extend(p.cpu())
            label.extend(y)
        
        pred = torch.stack(pred)
        label = torch.stack(label)

        return loss.item(), accuracy_score(label.numpy(), pred.numpy()), roc_auc_score(label.numpy(), pred.numpy()), f1_score(label.numpy(), pred.numpy(), average='macro')

    @torch.no_grad()
    def prepare_attack_training_data(self, model, data, all_neg=None):
        '''Prepare the training data of attack model (Present vs. Absent)
            Present edges (label = 1): training data of shadow model (Test pos and neg edges)
            Absent edges (label = 0): validation data of shadow model (Valid pos and neg edges)
        '''

        z = model(data.x, data.test_pos_edge_index)

        # Sample same size of neg as pos
        sample_idx = torch.randperm(all_neg.shape[1])[:data.test_pos_edge_index.shape[1]]
        neg_subset = all_neg[:, sample_idx]

        present_edge_index = torch.cat([data.test_pos_edge_index, data.test_neg_edge_index], dim=-1)
        
        if 'sub' in self.args.unlearning_model:
            absent_edge_index = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index], dim=-1)
        else:   #if 'all' in self.args.unlearning_model:
            absent_edge_index = torch.cat([data.val_pos_edge_index, data.val_neg_edge_index, data.train_pos_edge_index, neg_subset.to(device)], dim=-1)
        
        edge_index = torch.cat([present_edge_index, absent_edge_index], dim=-1)

        feature = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1).cpu()
        label = get_link_labels(present_edge_index, absent_edge_index).long().cpu()

        return feature, label
