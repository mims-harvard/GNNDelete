import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, average_precision_score


class RGCN(nn.Module):
    def __init__(self, args, num_nodes, num_edges, num_edge_type):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_edge_type = num_edge_type

        self.in_dim = args.in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim

        self.node_embedding = np.zeros((num_nodes, args.out_dim))

        # Encoder: RGCN
        self.rgcn1 = FastRGCNConv(
            args.in_dim, args.hidden_dim, num_edge_type, num_bases=args.num_bases, num_blocks=args.num_blocks)
        self.rgcn2 = FastRGCNConv(
            args.hidden_dim, args.out_dim, num_edge_type, num_bases=args.num_bases, num_blocks=args.num_blocks)
        self.relu = nn.ReLU()
        
        # Decoder: DistMult
        self.W = nn.Parameter(torch.Tensor(num_edge_type, args.out_dim))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))

        # self.save_hyperparameters()
    
    def forward(self, x, edge, edge_type):
        x = self.rgcn1(x, edge, edge_type)
        x = self.relu(x)
        x = self.rgcn2(x, edge, edge_type)
        out = F.log_softmax(x, dim=1)
        
        return out

    def decode(self, embedding, edge, edge_type):
        h = embedding[edge[0, :]]
        t = embedding[edge[1, :]]
        r = self.W[edge_type]
        score = torch.sum(h * r * t, dim=1)
        
        return torch.sigmoid(score)        

    def negative_sampling(self, edge_pos):
        '''Generate negative samples'''

        edge_neg = edge_pos.clone()
        shuffle = torch.randperm(edge_neg.shape[0])
        edge_neg = edge_neg[shuffle]
        
        return edge_neg

    def shared_step(self, embedding, batch, stage):
        # Get data
        if stage == 'train':
            mask = batch.train_mask
        elif stage == 'valid':
            mask = batch.valid_mask
        elif stage == 'test':
            mask = batch.test_mask
        elif stage == 'all-test':
            mask = batch.valid_mask + batch.test_mask
        else:
            raise ValueError(f'Mask {stage} not supported')
        
        # Positive and negative sample
        edge_pos = batch.edge_index[:, mask]
        edge_type = batch.edge_type[mask]
        edge_neg = self.negative_sampling(edge_pos)
        edge = torch.cat([edge_pos, edge_neg], dim=-1)

        # Edge label
        label_pos = torch.ones(edge_pos.size(1)).to(edge_pos.device)
        label_neg = torch.zeros(edge_neg.size(1)).to(edge_pos.device)
        label = torch.cat([label_pos, label_neg], dim=-1)
        
        # Link prediction
        edge_type = torch.cat([edge_type, edge_type], dim=-1)
        proba = self.decode(embedding, edge, edge_type)

        # Calculate loss
        loss = F.binary_cross_entropy(proba, label)

        return loss, label.detach().cpu(), proba.detach().cpu()

    def hits_at_k(self, k=1):
        return 0

    def get_eval_metrics(self, label, proba):
        auc = roc_auc_score(label.numpy(), proba.numpy())
        aup = average_precision_score(label.numpy(), proba.numpy())

        return auc, aup
    
    def get_ranking_metrics(self, eval_triplets, true_triplets):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        ranks = []
        for head in [True, False]:  # head or tail prediction

            for fr in range(0, len(test_set), batch_size):
                to = min(fr + batch_size, len(test_set))

                batch = test_set[fr:to, :].to(device=device)
                bn, _ = batch.size()

                # compute the full score matrix (filter later)
                bases   = batch[:, 1:] if head else batch[:, :2]
                targets = batch[:, 0]  if head else batch[:, 2]

                # collect the triples for which to compute scores
                bexp = bases.view(bn, 1, 2).expand(bn, num_nodes, 2)
                ar   = torch.arange(num_nodes, device=device).view(1, num_nodes, 1).expand(bn, num_nodes, 1)
                toscore = torch.cat([ar, bexp] if head else [bexp, ar], dim=2)
                assert toscore.size() == (bn, num_nodes, 3)

                scores, _ = model(graph, toscore)
                assert scores.size() == (bn, num_nodes)

                # filter out the true triples that aren't the target
                if filter_candidates:
                    filter_scores(scores, batch, true_triples, head=head)

                # Select the true scores, and count the number of values larger than than
                true_scores = scores[torch.arange(bn, device=device), targets]
                raw_ranks = torch.sum(scores > true_scores.view(bn, 1), dim=1, dtype=torch.long)
                # -- This is the "optimistic" rank (assuming it's sorted to the front of the ties)
                num_ties = torch.sum(scores == true_scores.view(bn, 1), dim=1, dtype=torch.long)

                # Account for ties (put the true example halfway down the ties)
                branks = raw_ranks + (num_ties - 1) // 2

                ranks.extend((branks + 1).tolist())

            mr = np.mean(ranks)
            mrr = np.mean([1 / rank for rank in ranks])

            hits = []
            for k in [1, 3, 5, 10]:
                hits.append(np.mean([1.0 if rank <= k else 0.0 for rank in ranks]))

        return mr, mrr, hits, ranks

        return mr, mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10

    def training_step(self, embedding, batch):
        loss, label, proba = self.shared_step(embedding, batch, 'train')

        return loss, label, proba

    def eval_step(self, embedding, batch, stage, eval_triplets, true_triplets):
        loss, label, proba = self.shared_step(embedding, batch, stage)
        auc, aup = self.get_eval_metrics(label, proba)
        # mr, mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10 = self.get_ranking_metrics(eval_triplets, true_triplets)

        # log = {
        #     'loss': loss.item(), 
        #     'AUC': auc,
        #     'AUP': aup,
        #     'MR': mr,
        #     'MRR': mrr,
        #     'Hits@1': hits_at_1,
        #     'Hits@3': hits_at_3,
        #     'Hits@5': hits_at_5,
        #     'Hits@10': hits_at_10,
        # }

        return loss, auc, aup#, mr, mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10
