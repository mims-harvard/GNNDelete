import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

        
class GIN(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.transition = nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(p=args.dropout)
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(args.in_dim, args.hidden_dim), 
            nn.ReLU(), 
        )
        self.conv1 = GINConv(self.mlp1)
        self.mlp2 = nn.Sequential(
            nn.Linear(args.hidden_dim, args.out_dim), 
            nn.ReLU(),
        )
        self.conv2= GINConv(self.mlp2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        x = self.conv2(x, edge_index)

        return x

    def decode(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is not None:
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        else:
            edge_index = pos_edge_index
            logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

        return logits
