import os
import time
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer
from ..evaluation import *
from ..utils import *


def BoundedKLD(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))

class GNNDeleteEmbeddingDistanceTrainer(Trainer):

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            return self.train_minibatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to('cuda')
        data = data.to('cuda')

        best_metric = 0
        if 'kld' in args.unlearning_model:
            loss_fct = BoundedKLD
        else:
            loss_fct = nn.MSELoss()
        # neg_size = 10

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        # All node paris in S_Df without Df. For Local Causality
        ## S_Df all pair mask
        sdf_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_2hop_mask], with_replacement=True).t()
        sdf_all_pair_mask[idx[0], idx[1]] = True
        sdf_all_pair_mask[idx[1], idx[0]] = True

        # print(data.sdf_node_2hop_mask.sum())
        # print(sdf_all_pair_mask.nonzero())
        # print(data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1])

        assert sdf_all_pair_mask.sum().cpu() == data.sdf_node_2hop_mask.sum().cpu() * data.sdf_node_2hop_mask.sum().cpu()

        ## Remove Df itself
        sdf_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        sdf_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## Lower triangular mask
        idx = torch.tril_indices(data.num_nodes, data.num_nodes, -1)
        lower_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        lower_mask[idx[0], idx[1]] = True

        ## The final mask is the intersection
        sdf_all_pair_without_df_mask = sdf_all_pair_mask & lower_mask

        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum(), a, sdf_all_pair_mask.sum())
        # print('aaaaaaaaaaaa', lower_mask.sum())
        # print('aaaaaaaaaaaa', sdf_all_pair_without_df_mask.sum())
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum())
        # assert sdf_all_pair_without_df_mask.sum() == \
        #         data.sdf_node_2hop_mask.sum().cpu() * (data.sdf_node_2hop_mask.sum().cpu() - 1) // 2 - data.df_mask.sum().cpu()


        # Node representation for local causality
        with torch.no_grad():
            z1_ori, z2_ori = model.get_original_embeddings(data.x, data.train_pos_edge_index[:, data.dtrain_mask], return_all_emb=True)

        total_time = 0
        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()
            start_time = time.time()

            z1, z2 = model(data.x, data.train_pos_edge_index[:, data.sdf_mask], return_all_emb=True)
            print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())

            # Effectiveness and Randomness
            neg_size = data.df_mask.sum()
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=neg_size)

            df_logits = model.decode(z2, data.train_pos_edge_index[:, data.df_mask], neg_edge_index)
            loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
            # df_logits = model.decode(
            #     z, 
            #     data.train_pos_edge_index[:, data.df_mask].repeat(1, neg_size), 
            #     neg_edge_index).sigmoid()
            
            # loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])
            # print('df_logits', df_logits)
            # raise

            # Local causality
            if sdf_all_pair_without_df_mask.sum() != 0:
                loss_l = loss_fct(z1_ori[data.sdf_node_1hop_mask], z1[data.sdf_node_1hop_mask]) + \
                         loss_fct(z2_ori[data.sdf_node_2hop_mask], z2[data.sdf_node_2hop_mask])
                print('local proba', loss_l.item())
            
            else:
                loss_l = torch.tensor(0)
                print('local proba', 0)


            alpha = 0.5
            if 'ablation_random' in self.args.unlearning_model:
                loss_l = torch.tensor(0)
                loss = loss_e
            elif 'ablation_locality' in self.args.unlearning_model:
                loss_e = torch.tensor(0)
                loss = loss_l
            else:
                loss = alpha * loss_e + (1 - alpha) * loss_l

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            end_time = time.time()

            log = {
                'epoch': epoch,
                'train_loss': loss.item(),
                'train_loss_l': loss_l.item(),
                'train_loss_e': loss_e.item(),
                'train_time': end_time - start_time,
            }
            # wandb.log(log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_loss_l': loss_e.item(),
                    'train_loss_e': loss_l.item(),
                    'train_time': end_time - start_time,
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

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_loss = 100000
        if 'kld' in args.unlearning_model:
            loss_fct = BoundedKLD
        else:
            loss_fct = nn.MSELoss()
        # neg_size = 10

        # MI Attack before unlearning
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        z_ori = self.get_embedding(model, data, on_cpu=True)
        z_ori_two_hop = z_ori[data.sdf_node_2hop_mask]

        data.edge_index = data.train_pos_edge_index
        data.node_id = torch.arange(data.x.shape[0])
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()

            print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())

            epoch_loss_e = 0
            epoch_loss_l = 0
            epoch_loss = 0
            for step, batch in enumerate(tqdm(loader, leave=False)):
                # print('data', batch)
                # print('two hop nodes', batch.sdf_node_2hop_mask.sum())
                batch = batch.to('cuda')

                train_pos_edge_index = batch.edge_index
                z = model(batch.x, train_pos_edge_index[:, batch.sdf_mask], batch.sdf_node_1hop_mask, batch.sdf_node_2hop_mask)
                z_two_hop = z[batch.sdf_node_2hop_mask]

                # Effectiveness and Randomness
                neg_size = batch.df_mask.sum()
                neg_edge_index = negative_sampling(
                    edge_index=train_pos_edge_index,
                    num_nodes=z.size(0),
                    num_neg_samples=neg_size)

                df_logits = model.decode(z, train_pos_edge_index[:, batch.df_mask], neg_edge_index)
                loss_e = loss_fct(df_logits[:neg_size], df_logits[neg_size:])

                # Local causality
                mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
                mask[batch.node_id[batch.sdf_node_2hop_mask]] = True
                z_ori_subset = z_ori[mask].to('cuda')

                # Only take the lower triangular part
                num_nodes = z_ori_subset.shape[0]
                idx = torch.tril_indices(num_nodes, num_nodes, -1)
                local_lower_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
                local_lower_mask[idx[0], idx[1]] = True

                logits_ori = (z_ori_subset @ z_ori_subset.t())[local_lower_mask].sigmoid()
                logits = (z_two_hop @ z_two_hop.t())[local_lower_mask].sigmoid()

                loss_l = loss_fct(logits, logits_ori)


                alpha = 0.5
                if 'ablation_random' in self.args.unlearning_model:
                    loss_l = torch.tensor(0)
                    loss = loss_e
                elif 'ablation_locality' in self.args.unlearning_model:
                    loss_e = torch.tensor(0)
                    loss = loss_l
                else:
                    loss = alpha * loss_e + (1 - alpha) * loss_l

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss_e += loss_e.item()
                epoch_loss_l += loss_l.item()
                epoch_loss += loss.item()

            epoch_loss_e /= step
            epoch_loss_l /= step
            epoch_loss /= step


            if (epoch+1) % args.valid_freq == 0:
                valid_loss, auc, aup, df_logt, logit_all_pair = self.eval(model, data, 'val')

                log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss,
                    'train_loss_e': epoch_loss_e,
                    'train_loss_l': epoch_loss_l,
                    'valid_dt_loss': valid_loss,
                    'valid_dt_auc': auc,
                    'valid_dt_aup': aup,
                }
                wandb.log(log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                tqdm.write(' | '.join(msg))

                self.trainer_log['log'].append(log)

        self.trainer_log['training_time'] = time.time() - start_time

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
