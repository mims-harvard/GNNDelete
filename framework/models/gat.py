import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.conv1 = GATConv(args.in_dim, args.hidden_dim)
        self.conv2 = GATConv(args.hidden_dim, args.out_dim)
        # self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, edge_index, return_all_emb=False):
        x1 = self.conv1(x, edge_index)
        x = F.relu(x1)
        # x = self.dropout(x)
        x2 = self.conv2(x, edge_index)

        if return_all_emb:
            return x1, x2
        
        return x2

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits
