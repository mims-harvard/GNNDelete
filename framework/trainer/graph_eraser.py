import os
import json
import copy
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, subgraph

from .base import Trainer
from ..evaluation import *
from ..utils import *


class ConstrainedKmeans:
    '''This code is from https://github.com/MinChen00/Graph-Unlearning'''
    
    def __init__(self, args, data_feat, num_clusters, node_threshold, terminate_delta, max_iteration=20):
        self.args = args
        self.data_feat = data_feat
        self.num_clusters = num_clusters
        self.node_threshold = node_threshold
        self.terminate_delta = terminate_delta
        self.max_iteration = max_iteration

    def initialization(self):
        centroids = np.random.choice(np.arange(self.data_feat.shape[0]), self.num_clusters, replace=False)
        self.centroid = {}
        for i in range(self.num_clusters):
            self.centroid[i] = self.data_feat[centroids[i]]

    def clustering(self):
        centroid = copy.deepcopy(self.centroid)
        km_delta = []

        # pbar = tqdm(total=self.max_iteration)
        # pbar.set_description('Clustering')

        for i in trange(self.max_iteration, desc='Graph partition'):
            # self.logger.info('iteration %s' % (i,))

            self._node_reassignment()
            self._centroid_updating()

            # record the average change of centroids, if the change is smaller than a very small value, then terminate
            delta = self._centroid_delta(centroid, self.centroid)
            km_delta.append(delta)
            centroid = copy.deepcopy(self.centroid)

            if delta <= self.terminate_delta:
                break
            print("delta: %s" % delta)
        # pbar.close()
        return self.clusters, km_delta

    def _node_reassignment(self):
        self.clusters = {}
        for i in range(self.num_clusters):
            self.clusters[i] = np.zeros(0, dtype=np.uint64)

        distance = np.zeros([self.num_clusters, self.data_feat.shape[0]])

        for i in range(self.num_clusters):
            distance[i] = np.sum(np.power((self.data_feat - self.centroid[i]), 2), axis=1)

        sort_indices = np.unravel_index(np.argsort(distance, axis=None), distance.shape)
        clusters = sort_indices[0]
        users = sort_indices[1]
        selected_nodes = np.zeros(0, dtype=np.int64)
        counter = 0

        while len(selected_nodes) < self.data_feat.shape[0]:
            cluster = int(clusters[counter])
            user = users[counter]
            if self.clusters[cluster].size < self.node_threshold:
                self.clusters[cluster] = np.append(self.clusters[cluster], np.array(int(user)))
                selected_nodes = np.append(selected_nodes, np.array(int(user)))

                # delete all the following pairs for the selected user
                user_indices = np.where(users == user)[0]
                a = np.arange(users.size)
                b = user_indices[user_indices > counter]
                remain_indices = a[np.where(np.logical_not(np.isin(a, b)))[0]]
                clusters = clusters[remain_indices]
                users = users[remain_indices]

            counter += 1

    def _centroid_updating(self):
        for i in range(self.num_clusters):
            self.centroid[i] = np.mean(self.data_feat[self.clusters[i].astype(int)], axis=0)

    def _centroid_delta(self, centroid_pre, centroid_cur):
        delta = 0.0
        for i in range(len(centroid_cur)):
            delta += np.sum(np.abs(centroid_cur[i] - centroid_pre[i]))

        return delta

    def generate_shard_data(self, data):
        shard_data = {}
        for shard in trange(self.args['num_shards'], desc='Generate shard data'):
            train_shard_indices = list(self.community_to_node[shard])
            shard_indices = np.union1d(train_shard_indices, self.test_indices)

            x = data.x[shard_indices]
            y = data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.test_indices))

            shard_data[shard] = data

        self.data_store.save_shard_data(self.shard_data)

class OptimalAggregator:
    def __init__(self, run, target_model, data, args):
        self.args = args

        self.run = run
        self.target_model = target_model
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_shards = args.num_clusters

    def generate_train_data(self):
        data_store = DataStore(self.args)
        train_indices, _ = data_store.load_train_test_split()

        # sample a set of nodes from train_indices
        if self.args["num_opt_samples"] == 1000:
            train_indices = np.random.choice(train_indices, size=1000, replace=False)
        elif self.args["num_opt_samples"] == 10000:
            train_indices = np.random.choice(train_indices, size=int(train_indices.shape[0] * 0.1), replace=False)
        elif self.args["num_opt_samples"] == 1:
            train_indices = np.random.choice(train_indices, size=int(train_indices.shape[0]), replace=False)

        train_indices = np.sort(train_indices)
        self.logger.info("Using %s samples for optimization" % (int(train_indices.shape[0])))

        x = self.data.x[train_indices]
        y = self.data.y[train_indices]
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices)

        train_data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
        train_data.train_mask = torch.zeros(train_indices.shape[0], dtype=torch.bool)
        train_data.test_mask = torch.ones(train_indices.shape[0], dtype=torch.bool)
        self.true_labels = y

        self.posteriors = {}
        for shard in range(self.num_shards):
            self.target_model.data = train_data
            data_store.load_target_model(self.run, self.target_model, shard)
            self.posteriors[shard] = self.target_model.posterior().to(self.device)

    def optimization(self):
        weight_para = nn.Parameter(torch.full((self.num_shards,), fill_value=1.0 / self.num_shards), requires_grad=True)
        optimizer = optim.Adam([weight_para], lr=self.args['opt_lr'])
        scheduler = MultiStepLR(optimizer, milestones=[500, 1000], gamma=self.args['opt_lr'])

        train_dset = OptDataset(self.posteriors, self.true_labels)
        train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, num_workers=0)

        min_loss = 1000.0
        for epoch in range(self.args.epochs):
            loss_all = 0.0

            for posteriors, labels in train_loader:
                labels = labels.to(self.device)

                optimizer.zero_grad()
                loss = self._loss_fn(posteriors, labels, weight_para)
                loss.backward()
                loss_all += loss

                optimizer.step()
                with torch.no_grad():
                    weight_para[:] = torch.clamp(weight_para, min=0.0)

            scheduler.step()

            if loss_all < min_loss:
                ret_weight_para = copy.deepcopy(weight_para)
                min_loss = loss_all

            self.logger.info('epoch: %s, loss: %s' % (epoch, loss_all))

        return ret_weight_para / torch.sum(ret_weight_para)

    def _loss_fn(self, posteriors, labels, weight_para):
        aggregate_posteriors = torch.zeros_like(posteriors[0])
        for shard in range(self.num_shards):
            aggregate_posteriors += weight_para[shard] * posteriors[shard]

        aggregate_posteriors = F.softmax(aggregate_posteriors, dim=1)
        loss_1 = F.cross_entropy(aggregate_posteriors, labels)
        loss_2 = torch.sqrt(torch.sum(weight_para ** 2))

        return loss_1 + loss_2

class Aggregator:
    def __init__(self, run, target_model, data, shard_data, args):
        self.args = args

        self.run = run
        self.target_model = target_model
        self.data = data
        self.shard_data = shard_data

        self.num_shards = args.num_clusters

    def generate_posterior(self, suffix=""):
        self.true_label = self.shard_data[0].y[self.shard_data[0]['test_mask']].detach().cpu().numpy()
        self.posteriors = {}

        for shard in range(self.args.num_clusters):
            self.target_model.data = self.shard_data[shard]
            self.data_store.load_target_model(self.run, self.target_model, shard, suffix)
            self.posteriors[shard] = self.target_model.posterior()

    def _optimal_aggregator(self):
        optimal = OptimalAggregator(self.run, self.target_model, self.data, self.args)
        optimal.generate_train_data()
        weight_para = optimal.optimization()
        self.data_store.save_optimal_weight(weight_para, run=self.run)

        posterior = self.posteriors[0] * weight_para[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard] * weight_para[shard]

        return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro")

class GraphEraserTrainer(Trainer):

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):

        with torch.no_grad():
            z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])

        # Retrain the model
        for c in model.children():
            print('before', torch.norm(c.lin.weight), torch.norm(c.bias))
        for c in model.children():
            c.reset_parameters()
        for c in model.children():
            print('after', torch.norm(c.lin.weight), torch.norm(c.bias))
        model = model.cpu()

        num_nodes = data.num_nodes
        node_threshold = math.ceil(
            num_nodes / args.num_clusters + args.shard_size_delta * (num_nodes - num_nodes / args.num_clusters))
        print(f'Number of nodes: {num_nodes}. Shard threshold: {node_threshold}')

        cluster = ConstrainedKmeans(
            args,
            z.cpu().numpy(), 
            args.num_clusters, 
            node_threshold, 
            args.terminate_delta, 
            args.kmeans_max_iters)
        cluster.initialization()

        community, km_deltas = cluster.clustering()
        # with open(os.path.join(args.checkpoint_dir, 'kmeans_delta.pkl'), 'wb') as f:
        #     pickle.dump(km_deltas, f)

        community_to_node = {}
        for i in range(args.num_clusters):
            community_to_node[i] = np.array(community[i].astype(int))

        models = {}
        test_result = []
        for shard_id in trange(args.num_clusters, desc='Sharded retraining'):
            model_shard_id = copy.deepcopy(model).to('cuda')
            optimizer = torch.optim.Adam(model_shard_id.parameters(), lr=args.lr)

            subset_train, _ = subgraph(
                torch.tensor(community[shard_id], dtype=torch.long, device=device),
                data.train_pos_edge_index, 
                num_nodes=data.num_nodes)

            self.train_model(model_shard_id, data, subset_train, optimizer, args, shard_id)

            with torch.no_grad():
                z = model_shard_id(data.x, subset_train)
                logits = model_shard_id.decode(data.test_pos_edge_index, data.test_neg_edge_index)

        weight_para = nn.Parameter(torch.full((self.num_shards,), fill_value=1.0 / self.num_shards), requires_grad=True)
        optimizer = optim.Adam([weight_para], lr=self.args.lr)
                

        aggregator.generate_posterior()
        self.aggregate_f1_score = aggregator.aggregate()
        aggregate_time = time.time() - start_time
        self.logger.info("Partition cost %s seconds." % aggregate_time)

        self.logger.info("Final Test F1: %s" % (self.aggregate_f1_score,))

        

    def train_model(self, model, data, subset_train, optimizer, args, shard_id):

        best_loss = 100000
        for epoch in range(args.epochs):
            model.train()

            neg_edge_index = negative_sampling(
                edge_index=subset_train,
                num_nodes=data.num_nodes,
                num_neg_samples=subset_train.shape[1])

            z = model(data.x, subset_train)
            logits = model.decode(z, subset_train, neg_edge_index)
            label = self.get_link_labels(subset_train, neg_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

        valid_loss, auc, aup, _, _,  = self.eval_model(model, data, subset_train, 'val')
        log = {
            'train_loss': loss.item(),
            'valid_loss': valid_loss,
            'valid_auc': auc,
            'valid_aup': aup,
        }
        msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
        tqdm.write(' | '.join(msg))
        self.trainer_log[f'shard_{shard_id}'] = log

        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'model_{shard_id}.pt'))

    @torch.no_grad()
    def eval_model(self, model, data, subset_train, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        z = model(data.x, subset_train)
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = self.get_link_labels(pos_edge_index, neg_edge_index)

        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        auc = roc_auc_score(label.cpu(), logits.cpu())
        aup = average_precision_score(label.cpu(), logits.cpu())

        if self.args.unlearning_model in ['original', 'retrain']:
            df_logit = float('nan')
        else:
            # df_logit = float('nan')
            df_logit = model.decode(z, subset_train).sigmoid().detach().cpu().item()

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
        msg = [f'{i}: {j:.4f}' if isinstance(j, (np.floating, float)) else f'{i}: {j:>4d}' for i, j in log.items()]
        tqdm.write(' | '.join(msg))

        return loss, auc, aup, df_logit, logit_all_pair
