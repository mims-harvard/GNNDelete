import os
import json
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from .base import Trainer
from ..evaluation import *
from ..utils import *


class RetrainTrainer(Trainer):
    def __init__(self, args):
        self.trainer_log = {'unlearning_model': 'gradient_ascent_with_mp', 'log': []}

    def freeze_unused_mask(self, model, edge_to_delete, subgraph, h):
        gradient_mask = torch.zeros_like(delete_model.operator)
        
        edges = subgraph[h]
        for s, t in edges:
            if s < t:
                gradient_mask[s, t] = 1
        gradient_mask = gradient_mask.to(device)
        model.operator.register_hook(lambda grad: grad.mul_(gradient_mask))
        
    def train(self, retrain, model, data, optimizer, args):
        best_loss = 100000
        for epoch in trange(args.epochs, desc='Unlearning'):
            model.train()
            total_step = 0
            total_loss = 0

            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.dr_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.dr_mask.sum())

            z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
            logits = model.decode(z, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            label = self.get_link_labels(data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            total_step += 1
            total_loss += loss.item()

            msg = [
                f'Epoch: {epoch:>4d}', 
                f'train loss: {total_loss / total_step:.6f}'
            ]
            tqdm.write(' | '.join(msg))

            valid_loss, auc, aup = self.eval(model, data, 'val')

            self.trainer_log['log'].append({
                'dt_loss': valid_loss,
                'dt_auc': auc,
                'dt_aup': aup
            })

        # Eval unlearn
        loss, auc, aup = self.test(model, data)
        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = auc
        self.trainer_log['dt_aup'] = aup

        self.trainer_log['ve'] = 0
        self.trainer_log['dr_kld'] = 0

        embedding = get_node_embedding_data(model, data)
        logits = model.decode(embedding, data.train_pos_edge_index[:, ~data.ga_mask]).sigmoid().detach().cpu()
        self.trainer_log['df_score'] = logits[:1].cpu().item()


        # Save
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model.pt'))
        print(self.trainer_log)
        with open(os.path.join(args.checkpoint_dir, 'trainer_log.json'), 'w') as f:
            json.dump(self.trainer_log, f)
