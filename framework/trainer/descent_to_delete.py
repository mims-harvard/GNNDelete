import os
import time
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from .base import Trainer
from ..evaluation import *
from ..utils import *


class DtdTrainer(Trainer):
    '''This code is adapte from https://github.com/ChrisWaites/descent-to-delete'''
    
    def compute_sigma(self, num_examples, iterations, lipshitz, smooth, strong, epsilon, delta):
        """Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf"""

        print('delta', delta)
        gamma = (smooth - strong) / (smooth + strong)
        numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
        denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
        # print('sigma', numerator, denominator, numerator / denominator)
    
        return numerator / denominator

    def publish(self, model, sigma):
        """Publishing function which adds Gaussian noise with scale sigma."""

        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(p + torch.empty_like(p).normal_(0, sigma))

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_valid_loss = 100000

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
                edge_index=data.train_pos_edge_index[:, data.dr_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.dr_mask.sum())

            z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
            logits = model.decode(z, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            label = get_link_labels(data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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
                'dt_loss': valid_loss,
                'dt_auc': auc,
                'dt_aup': aup
            })
        
        train_size = data.dr_mask.sum().cpu().item()
        sigma = self.compute_sigma(
            train_size, 
            args.epochs, 
            1 + args.weight_decay, 
            4 - args.weight_decay, 
            args.weight_decay, 
            5, 
            1 / train_size / train_size)
        
        self.publish(model, sigma)

        self.trainer_log['sigma'] = sigma
        self.trainer_log['training_time'] = time.time() - start_time

        # Save
        ckpt = {
            'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
