import os
import copy
import time
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer, KGTrainer, NodeClassificationTrainer
from ..evaluation import *
from ..utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def BoundedKLDMean(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'batchmean'))

def BoundedKLDSum(logits, truth):
    return 1 - torch.exp(-F.kl_div(F.log_softmax(logits, -1), truth.softmax(-1), None, None, 'sum'))

def CosineDistanceMean(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).mean()

def CosineDistanceSum(logits, truth):
    return (1 - F.cosine_similarity(logits, truth)).sum()

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n
    return torch.matmul(torch.matmul(H, K), H)  

def rbf(X, sigma=None):
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX

def kernel_HSIC(X, Y, sigma=None):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))
        
def linear_HSIC(X, Y):
    L_X = torch.matmul(X, X.T)
    L_Y = torch.matmul(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))

def LinearCKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def RBFCKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))
    return hsic / (var1 * var2)


def get_loss_fct(name):
    # if name == 'mse':
    #     loss_fct = nn.MSELoss(reduction='mean')
    # elif name == 'kld':
    #     loss_fct = BoundedKLDMean
    # elif name == 'cosine':
    #     loss_fct = CosineDistanceMean
    
    if name == 'kld_mean':
        loss_fct = BoundedKLDMean
    elif name == 'kld_sum':
        loss_fct = BoundedKLDSum
    elif name == 'mse_mean':
        loss_fct = nn.MSELoss(reduction='mean')
    elif name == 'mse_sum':
        loss_fct = nn.MSELoss(reduction='sum')
    elif name == 'cosine_mean':
        loss_fct = CosineDistanceMean
    elif name == 'cosine_sum':
        loss_fct = CosineDistanceSum
    elif name == 'linear_cka':
        loss_fct = LinearCKA
    elif name == 'rbf_cka':
        loss_fct = RBFCKA
    else:
        raise NotImplementedError

    return loss_fct
    
class GNNDeleteNodeembTrainer(Trainer):

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            return self.train_minibatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to('cuda')
        data = data.to('cuda')

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

        # All node paris in S_Df without Df
        ## S_Df 1 hop all pair mask
        sdf1_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_1hop_mask], with_replacement=True).t()
        sdf1_all_pair_mask[idx[0], idx[1]] = True
        sdf1_all_pair_mask[idx[1], idx[0]] = True

        assert sdf1_all_pair_mask.sum().cpu() == data.sdf_node_1hop_mask.sum().cpu() * data.sdf_node_1hop_mask.sum().cpu()

        ## Remove Df itself
        sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## S_Df 2 hop all pair mask
        sdf2_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_2hop_mask], with_replacement=True).t()
        sdf2_all_pair_mask[idx[0], idx[1]] = True
        sdf2_all_pair_mask[idx[1], idx[0]] = True

        assert sdf2_all_pair_mask.sum().cpu() == data.sdf_node_2hop_mask.sum().cpu() * data.sdf_node_2hop_mask.sum().cpu()

        ## Remove Df itself
        sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        ## Lower triangular mask
        idx = torch.tril_indices(data.num_nodes, data.num_nodes, -1)
        lower_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        lower_mask[idx[0], idx[1]] = True

        ## The final mask is the intersection
        sdf1_all_pair_without_df_mask = sdf1_all_pair_mask & lower_mask
        sdf2_all_pair_without_df_mask = sdf2_all_pair_mask & lower_mask

        # print(data.sdf_node_2hop_mask.sum())
        # print(sdf_all_pair_mask.nonzero())
        # print(data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1])
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum(), a, sdf_all_pair_mask.sum())
        # print('aaaaaaaaaaaa', lower_mask.sum())
        # print('aaaaaaaaaaaa', sdf_all_pair_without_df_mask.sum())
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum())
        # assert sdf_all_pair_without_df_mask.sum() == \
        #         data.sdf_node_2hop_mask.sum().cpu() * (data.sdf_node_2hop_mask.sum().cpu() - 1) // 2 - data.df_mask.sum().cpu()

        # 
        non_df_node_mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=data.x.device)
        non_df_node_mask[data.directed_df_edge_index.flatten().unique()] = False

        data.sdf_node_1hop_mask_non_df_mask = data.sdf_node_1hop_mask & non_df_node_mask
        data.sdf_node_2hop_mask_non_df_mask = data.sdf_node_2hop_mask & non_df_node_mask

        
        # Original node embeddings
        with torch.no_grad():
            z1_ori, z2_ori = model.get_original_embeddings(data.x, data.train_pos_edge_index[:, data.dr_mask], return_all_emb=True)

        loss_fct = get_loss_fct(self.args.loss_fct)

        neg_edge = neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.df_mask.sum())

        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()

            start_time = time.time()
            z1, z2 = model(data.x, data.train_pos_edge_index[:, data.sdf_mask], return_all_emb=True)
            # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())
            # print('aaaaaa', z[data.sdf_node_2hop_mask].sum())

            # Randomness
            pos_edge = data.train_pos_edge_index[:, data.df_mask]
            # neg_edge = torch.randperm(data.num_nodes)[:pos_edge.view(-1).shape[0]].view(2, -1)

            embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
            embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)

            embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
            embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)

            loss_r1 = loss_fct(embed1, embed1_ori)
            loss_r2 = loss_fct(embed2, embed2_ori)

            # Local causality
            loss_l1 = loss_fct(z1[data.sdf_node_1hop_mask_non_df_mask], z1_ori[data.sdf_node_1hop_mask_non_df_mask])
            loss_l2 = loss_fct(z2[data.sdf_node_2hop_mask_non_df_mask], z2_ori[data.sdf_node_2hop_mask_non_df_mask])


            # Total loss
            '''both_all, both_layerwise, only2_layerwise, only2_all, only1'''
            if self.args.loss_type == 'both_all':
                loss_l = loss_l1 + loss_l2
                loss_r = loss_r1 + loss_r2

                #### alpha * loss_r + (1 - alpha) * loss_l
                loss = self.args.alpha * loss_r + (1 - self.args.alpha) * loss_l

                #### loss_r + lambda * loss_l
                # loss = loss_l + self.args.alpha * loss_r
                loss.backward()
                optimizer.step()
            
            elif self.args.loss_type == 'both_layerwise':
                #### alpha * loss_r + (1 - alpha) * loss_l
                loss_l = loss_l1 + loss_l2
                loss_r = loss_r1 + loss_r2

                loss1 = self.args.alpha * loss_r1 + (1 - self.args.alpha) * loss_l1
                loss1.backward(retain_graph=True)
                optimizer[0].step()
                optimizer[0].zero_grad()

                loss2 = self.args.alpha * loss_r2 + (1 - self.args.alpha) * loss_l2
                loss2.backward(retain_graph=True)
                optimizer[1].step()
                optimizer[1].zero_grad()

                loss = loss1 + loss2

                #### loss_r + lambda * loss_l
                # loss_l = loss_l1 + loss_l2
                # loss_r = loss_r1 + loss_r2

                # loss1 = loss_r1 + self.args.alpha * loss_l1
                # loss1.backward(retain_graph=True)
                # optimizer[0].step()
                # optimizer[0].zero_grad()

                # loss2 = loss_r2 + self.args.alpha * loss_l2
                # loss2.backward()
                # optimizer[1].step()
                # optimizer[1].zero_grad()

                # loss = loss1 + loss2
                
            elif self.args.loss_type == 'only2_layerwise':
                loss_l = loss_l1 + loss_l2
                loss_r = loss_r1 + loss_r2

                optimizer[0].zero_grad()

                #### alpha * loss_r + (1 - alpha) * loss_l
                loss2 = self.args.alpha * loss_r2 + (1 - self.args.alpha) * loss_l2

                #### loss_r + lambda * loss_l
                # loss2 = loss_r2 + self.args.alpha * loss_l2

                loss2.backward()
                optimizer[1].step()
                optimizer[1].zero_grad()

                loss = loss2

            elif self.args.loss_type == 'only2_all':
                loss_l = loss_l2
                loss_r = loss_r2

                loss = loss_l + self.args.alpha * loss_r

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            elif self.args.loss_type == 'only1':
                loss_l = loss_l1
                loss_r = loss_r1

                loss = loss_l + self.args.alpha * loss_r
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            else:
                raise NotImplementedError

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss_r': loss_r.item(),
                'loss_l': loss_l.item(),
                'train_time': epoch_time
            }
            wandb.log(step_log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'loss_r': loss_r.item(),
                    'loss_l': loss_l.item(),
                    'train_time': epoch_time,
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
                        # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))


    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
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

        non_df_node_mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=data.x.device)
        non_df_node_mask[data.directed_df_edge_index.flatten().unique()] = False

        data.sdf_node_1hop_mask_non_df_mask = data.sdf_node_1hop_mask & non_df_node_mask
        data.sdf_node_2hop_mask_non_df_mask = data.sdf_node_2hop_mask & non_df_node_mask


        data.edge_index = data.train_pos_edge_index
        data.node_id = torch.arange(data.x.shape[0])
        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        # all_neg_edge = negative_sampling(
        #     edge_index=data.train_pos_edge_index,
        #     num_nodes=data.num_nodes,
        #     num_neg_samples=100000
        # )
        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()

            epoch_loss_l = 0
            epoch_loss_r = 0
            epoch_loss = 0
            epoch_time = 0
            for step, batch in enumerate(tqdm(loader, leave=False)):
                batch = batch.to(device)
                start_time = time.time()

                # Original embedding
                with torch.no_grad():
                    z1_ori, z2_ori = model.get_original_embeddings(batch.x, batch.edge_index, return_all_emb=True)
                    # z1_ori = z1_ori[batch.sdf_node_2hop_mask]
                    # z2_ori = z2_ori[batch.sdf_node_2hop_mask]

                z1, z2 = model(batch.x, batch.edge_index[:, batch.sdf_mask], batch.sdf_node_1hop_mask, batch.sdf_node_2hop_mask, return_all_emb=True)

                # Randomness
                pos_edge = batch.edge_index[:, batch.df_mask]
                neg_edge = negative_sampling(
                    edge_index=batch.edge_index,
                    num_nodes=batch.x.shape[0],
                    num_neg_samples=pos_edge.shape[1]
                )
                # neg_edge = all_neg_edge[:, :pos_edge.shape[1]]

                embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
                embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)

                embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
                embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)

                loss_r1 = loss_fct(embed1, embed1_ori)
                loss_r2 = loss_fct(embed2, embed2_ori)

                # Local causality
                loss_l1 = loss_fct(z1[batch.sdf_node_1hop_mask_non_df_mask], z1_ori[batch.sdf_node_1hop_mask_non_df_mask])
                loss_l2 = loss_fct(z2[batch.sdf_node_2hop_mask_non_df_mask], z2_ori[batch.sdf_node_2hop_mask_non_df_mask])


                # Total loss
                loss_l = loss_l1 + loss_l2
                loss_r = loss_r1 + loss_r2

                loss1 = self.args.alpha * loss_r1 + (1 - self.args.alpha) * loss_l1
                loss1.backward(retain_graph=True)
                optimizer[0].step()
                optimizer[0].zero_grad()

                loss2 = self.args.alpha * loss_r2 + (1 - self.args.alpha) * loss_l2
                loss2.backward(retain_graph=True)
                optimizer[1].step()
                optimizer[1].zero_grad()

                loss = loss1 + loss2

                end_time = time.time()
                epoch_loss_l += loss_l.item()
                epoch_loss_r += loss_r.item()
                epoch_loss += loss.item()
                epoch_time += end_time - start_time

                step_log = {
                    'Epoch': epoch,
                    'train_loss': loss.item(),
                    'train_loss_l': loss_l.item(),
                    'train_loss_r': loss_r.item(),
                    'train_time': end_time - start_time
                }
                wandb.log(step_log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
                tqdm.write(' | '.join(msg))

            if (epoch+1) % args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')

                train_log = {
                    'epoch': epoch,
                    'train_loss': epoch_loss / step,
                    'train_loss_l': epoch_loss_l / step,
                    'train_loss_r': epoch_loss_r / step,
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
                        # 'optimizer_state': optimizer.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            # 'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))


class GNNDeleteNodeClassificationTrainer(NodeClassificationTrainer):
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to('cuda')
        data = data.to('cuda')

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

        # All node paris in S_Df without Df
        ## S_Df 1 hop all pair mask
        # sdf1_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        # idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_1hop_mask], with_replacement=True).t()
        # sdf1_all_pair_mask[idx[0], idx[1]] = True
        # sdf1_all_pair_mask[idx[1], idx[0]] = True

        # assert sdf1_all_pair_mask.sum().cpu() == data.sdf_node_1hop_mask.sum().cpu() * data.sdf_node_1hop_mask.sum().cpu()

        # ## Remove Df itself
        # sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        # sdf1_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        # ## S_Df 2 hop all pair mask
        # sdf2_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        # idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_2hop_mask], with_replacement=True).t()
        # sdf2_all_pair_mask[idx[0], idx[1]] = True
        # sdf2_all_pair_mask[idx[1], idx[0]] = True

        # assert sdf2_all_pair_mask.sum().cpu() == data.sdf_node_2hop_mask.sum().cpu() * data.sdf_node_2hop_mask.sum().cpu()

        # ## Remove Df itself
        # sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1]] = False
        # sdf2_all_pair_mask[data.train_pos_edge_index[:, data.df_mask][1], data.train_pos_edge_index[:, data.df_mask][0]] = False

        # ## Lower triangular mask
        # idx = torch.tril_indices(data.num_nodes, data.num_nodes, -1)
        # lower_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        # lower_mask[idx[0], idx[1]] = True

        # ## The final mask is the intersection
        # sdf1_all_pair_without_df_mask = sdf1_all_pair_mask & lower_mask
        # sdf2_all_pair_without_df_mask = sdf2_all_pair_mask & lower_mask

        non_df_node_mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=data.x.device)
        non_df_node_mask[data.directed_df_edge_index.flatten().unique()] = False

        data.sdf_node_1hop_mask_non_df_mask = data.sdf_node_1hop_mask & non_df_node_mask
        data.sdf_node_2hop_mask_non_df_mask = data.sdf_node_2hop_mask & non_df_node_mask
        
        # Original node embeddings
        with torch.no_grad():
            z1_ori, z2_ori = model.get_original_embeddings(data.x, data.edge_index[:, data.dr_mask], return_all_emb=True)

        loss_fct = get_loss_fct(self.args.loss_fct)

        neg_edge = neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.df_mask.sum())

        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()

            start_time = time.time()
            z1, z2 = model(data.x, data.edge_index[:, data.sdf_mask], return_all_emb=True)
            # print('current deletion weight', model.deletion1.deletion_weight.sum(), model.deletion2.deletion_weight.sum())
            # print('aaaaaa', z[data.sdf_node_2hop_mask].sum())

            # Randomness
            pos_edge = data.edge_index[:, data.df_mask]
            # neg_edge = torch.randperm(data.num_nodes)[:pos_edge.view(-1).shape[0]].view(2, -1)

            embed1 = torch.cat([z1[pos_edge[0]], z1[pos_edge[1]]], dim=0)
            embed1_ori = torch.cat([z1_ori[neg_edge[0]], z1_ori[neg_edge[1]]], dim=0)

            embed2 = torch.cat([z2[pos_edge[0]], z2[pos_edge[1]]], dim=0)
            embed2_ori = torch.cat([z2_ori[neg_edge[0]], z2_ori[neg_edge[1]]], dim=0)

            loss_r1 = loss_fct(embed1, embed1_ori)
            loss_r2 = loss_fct(embed2, embed2_ori)

            # Local causality
            loss_l1 = loss_fct(z1[data.sdf_node_1hop_mask_non_df_mask], z1_ori[data.sdf_node_1hop_mask_non_df_mask])
            loss_l2 = loss_fct(z2[data.sdf_node_2hop_mask_non_df_mask], z2_ori[data.sdf_node_2hop_mask_non_df_mask])


            # Total loss
            '''both_all, both_layerwise, only2_layerwise, only2_all, only1'''
            loss_l = loss_l1 + loss_l2
            loss_r = loss_r1 + loss_r2

            loss1 = self.args.alpha * loss_r1 + (1 - self.args.alpha) * loss_l1
            loss1.backward(retain_graph=True)
            optimizer[0].step()
            optimizer[0].zero_grad()

            loss2 = self.args.alpha * loss_r2 + (1 - self.args.alpha) * loss_l2
            loss2.backward(retain_graph=True)
            optimizer[1].step()
            optimizer[1].zero_grad()

            loss = loss1 + loss2

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'loss_r': loss_r.item(),
                'loss_l': loss_l.item(),
                'train_time': epoch_time
            }
            wandb.log(step_log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_acc, dt_f1, valid_log = self.eval(model, data, 'val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'loss_r': loss_r.item(),
                    'loss_l': loss_l.item(),
                    'train_time': epoch_time,
                }
                
                for log in [train_log, valid_log]:
                    wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                if dt_acc + dt_f1 > best_metric:
                    best_metric = dt_acc + dt_f1
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': model.state_dict(),
                        # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

class KGGNNDeleteNodeembTrainer(KGTrainer):
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
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

        # All node paris in S_Df without Df
        ## S_Df 1 hop all pair mask
        # sdf1_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        # idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_1hop_mask], with_replacement=True).t()
        # sdf1_all_pair_mask[idx[0], idx[1]] = True
        # sdf1_all_pair_mask[idx[1], idx[0]] = True

        # assert sdf1_all_pair_mask.sum().cpu() == data.sdf_node_1hop_mask.sum().cpu() * data.sdf_node_1hop_mask.sum().cpu()

        # ## Remove Df itself
        # sdf1_all_pair_mask[data.edge_index[:, data.df_mask][0], data.edge_index[:, data.df_mask][1]] = False
        # sdf1_all_pair_mask[data.edge_index[:, data.df_mask][1], data.edge_index[:, data.df_mask][0]] = False

        # ## S_Df 2 hop all pair mask
        # sdf2_all_pair_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        # idx = torch.combinations(torch.arange(data.num_nodes)[data.sdf_node_2hop_mask], with_replacement=True).t()
        # sdf2_all_pair_mask[idx[0], idx[1]] = True
        # sdf2_all_pair_mask[idx[1], idx[0]] = True

        # assert sdf2_all_pair_mask.sum().cpu() == data.sdf_node_2hop_mask.sum().cpu() * data.sdf_node_2hop_mask.sum().cpu()

        # ## Remove Df itself
        # sdf2_all_pair_mask[data.edge_index[:, data.df_mask][0], data.edge_index[:, data.df_mask][1]] = False
        # sdf2_all_pair_mask[data.edge_index[:, data.df_mask][1], data.edge_index[:, data.df_mask][0]] = False

        # ## Lower triangular mask
        # idx = torch.tril_indices(data.num_nodes, data.num_nodes, -1)
        # lower_mask = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.bool)
        # lower_mask[idx[0], idx[1]] = True

        # ## The final mask is the intersection
        # sdf1_all_pair_without_df_mask = sdf1_all_pair_mask & lower_mask
        # sdf2_all_pair_without_df_mask = sdf2_all_pair_mask & lower_mask

        # print(data.sdf_node_2hop_mask.sum())
        # print(sdf_all_pair_mask.nonzero())
        # print(data.train_pos_edge_index[:, data.df_mask][0], data.train_pos_edge_index[:, data.df_mask][1])
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum(), a, sdf_all_pair_mask.sum())
        # print('aaaaaaaaaaaa', lower_mask.sum())
        # print('aaaaaaaaaaaa', sdf_all_pair_without_df_mask.sum())
        # print('aaaaaaaaaaaa', data.sdf_node_2hop_mask.sum())
        # assert sdf_all_pair_without_df_mask.sum() == \
        #         data.sdf_node_2hop_mask.sum().cpu() * (data.sdf_node_2hop_mask.sum().cpu() - 1) // 2 - data.df_mask.sum().cpu()

        # 
        non_df_node_mask = torch.ones(data.x.shape[0], dtype=torch.bool, device=data.x.device)
        non_df_node_mask[data.directed_df_edge_index.flatten().unique()] = False

        data.sdf_node_1hop_mask_non_df_mask = data.sdf_node_1hop_mask & non_df_node_mask
        data.sdf_node_2hop_mask_non_df_mask = data.sdf_node_2hop_mask & non_df_node_mask

        model_ori = copy.deepcopy(model)

        loss_fct = get_loss_fct(self.args.loss_fct)


        loader = GraphSAINTRandomWalkSampler(
            data, batch_size=args.batch_size, walk_length=2, num_steps=args.num_steps,
        )
        for epoch in trange(args.epochs, desc='Unlerning'):
            model.train()

            epoch_loss_e = 0
            epoch_loss_l = 0
            epoch_loss = 0
            epoch_time = 0
            for step, batch in enumerate(tqdm(loader, leave=False)):
                start_time = time.time()
                batch = batch.to(device)

                # Message passing
                edge_index = batch.edge_index[:, batch.dr_mask]
                edge_type = batch.edge_type[batch.dr_mask]
                z1, z2 = model(batch.x, edge_index, edge_type, batch.sdf_node_1hop_mask_non_df_mask, batch.sdf_node_2hop_mask_non_df_mask, return_all_emb=True)

                # Original node embeddings
                with torch.no_grad():
                    z1_ori, z2_ori = model.get_original_embeddings(batch.x, edge_index, edge_type, return_all_emb=True)

                # Randomness
                pos_edge_index = batch.edge_index[:, batch.df_mask]
                pos_edge_type = batch.edge_type[batch.df_mask]

                decoding_mask = pos_edge_type < self.args.num_edge_type
                decoding_edge_index = pos_edge_index[:, decoding_mask]
                decoding_edge_type = pos_edge_type[decoding_mask]
                # print(pos_edge_type.max(), decoding_edge_type.max(), self.args.num_edge_type)
                # raise
                neg_edge_index = negative_sampling_kg(
                    edge_index=decoding_edge_index,
                    edge_type=decoding_edge_type)

                embed1 = torch.cat([z1[decoding_edge_index[0]], z1[decoding_edge_index[1]]], dim=0)
                embed1_ori = torch.cat([z1_ori[neg_edge_index[0]], z1_ori[neg_edge_index[1]]], dim=0)

                embed2 = torch.cat([z2[decoding_edge_index[0]], z2[decoding_edge_index[1]]], dim=0)
                embed2_ori = torch.cat([z2_ori[neg_edge_index[0]], z2_ori[neg_edge_index[1]]], dim=0)

                loss_r1 = loss_fct(embed1, embed1_ori)
                loss_r2 = loss_fct(embed2, embed2_ori)

                # Local causality
                loss_l1 = loss_fct(z1[batch.sdf_node_1hop_mask_non_df_mask], z1_ori[batch.sdf_node_1hop_mask_non_df_mask])
                loss_l2 = loss_fct(z2[batch.sdf_node_2hop_mask_non_df_mask], z2_ori[batch.sdf_node_2hop_mask_non_df_mask])


                # Loss
                loss_l = loss_l1 + loss_l2
                loss_r = loss_r1 + loss_r2

                loss1 = self.args.alpha * loss_r1 + (1 - self.args.alpha) * loss_l1
                loss1.backward(retain_graph=True)
                optimizer[0].step()
                optimizer[0].zero_grad()

                loss2 = self.args.alpha * loss_r2 + (1 - self.args.alpha) * loss_l2
                loss2.backward(retain_graph=True)
                optimizer[1].step()
                optimizer[1].zero_grad()

                loss = loss1 + loss2


                end_time = time.time()
                epoch_time = end_time - start_time

                step_log = {
                    'Epoch': epoch,
                    'train_loss': loss.item(),
                    'loss_r': loss_r.item(),
                    'loss_l': loss_l.item(),
                    'train_time': epoch_time
                }
                wandb.log(step_log)
                msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
                tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, valid_log = self.eval(model, data, 'val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'loss_r': loss_r.item(),
                    'loss_l': loss_l.item(),
                    'train_time': epoch_time,
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
                    }
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

        # Save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))