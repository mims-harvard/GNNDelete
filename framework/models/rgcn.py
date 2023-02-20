import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv
from sklearn.metrics import roc_auc_score, average_precision_score


class RGCN(nn.Module):
    def __init__(self, args, num_nodes, num_edge_type, **kwargs):
        super().__init__()
        self.args = args
        self.num_edge_type = num_edge_type

        # Encoder: RGCN
        self.node_emb = nn.Embedding(num_nodes, args.in_dim)
        if num_edge_type > 20:
            self.conv1 = RGCNConv(args.in_dim, args.hidden_dim, num_edge_type * 2, num_blocks=4)
            self.conv2 = RGCNConv(args.hidden_dim, args.out_dim, num_edge_type * 2, num_blocks=4)
        else:
            self.conv1 = RGCNConv(args.in_dim, args.hidden_dim, num_edge_type * 2)
            self.conv2 = RGCNConv(args.hidden_dim, args.out_dim, num_edge_type * 2)
        self.relu = nn.ReLU()
        
        # Decoder: DistMult
        self.W = nn.Parameter(torch.Tensor(num_edge_type, args.out_dim))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x, edge, edge_type, return_all_emb=False):
        x = self.node_emb(x)
        x1 = self.conv1(x, edge, edge_type)
        x = self.relu(x1)
        x2 = self.conv2(x, edge, edge_type)
        
        if return_all_emb:
            return x1, x2

        return x2

    def decode(self, z, edge_index, edge_type):
        h = z[edge_index[0]]
        t = z[edge_index[1]]
        r = self.W[edge_type]
        
        logits = torch.sum(h * r * t, dim=1)
        
        return logits        

class RGCNDelete(RGCN):
    def __init__(self):
        pass
