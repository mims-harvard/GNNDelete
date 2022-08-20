import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv
from sklearn.metrics import roc_auc_score, average_precision_score


class RGAT(nn.Module):
    def __init__(self, args, num_edge_type, **kwargs):
        super().__init__()
        self.args = args
        self.num_edge_type = num_edge_type

        # Encoder: RGAT
        self.conv1 = FastRGATConv(args.in_dim, args.hidden_dim, num_edge_type)
        self.conv2 = FastRGATConv(args.hidden_dim, args.out_dim, num_edge_type)
        self.relu = nn.ReLU()
        
        # Decoder: DistMult
        self.W = nn.Parameter(torch.Tensor(num_edge_type, args.out_dim))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x, edge, edge_type):
        x = self.conv1(x, edge, edge_type)
        x = self.relu(x)
        x = self.conv2(x, edge, edge_type)
        out = F.log_softmax(x, dim=1)
        
        return out

    def decode(self, embedding, edge, edge_type):
        h = embedding[edge[0, :]]
        t = embedding[edge[1, :]]
        r = self.W[edge_type]
        score = torch.sum(h * r * t, dim=1)
        
        return torch.sigmoid(score)
