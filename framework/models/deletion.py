import torch
import torch.nn as nn
import torch.nn.functional as F
from . import GCN, GAT, GIN


class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            x[mask] = torch.matmul(x[mask], self.deletion_weight)
        
        return x

class DeletionLayerKG(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if self.mask is not None:
            x[self.mask] = torch.matmul(x[self.mask], self.deletion_weight)
        
        return x

class GCNDelete(GCN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None):
        with torch.no_grad():
            x = self.conv1(x, edge_index)
        
        x = self.deletion1(x, mask_1hop)

        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.deletion2(x, mask_2hop)

        return x

class GATDelete(GAT):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None):
        with torch.no_grad():
            x = self.conv1(x, edge_index)
        
        x = self.deletion1(x, mask_1hop)

        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.deletion2(x, mask_2hop)

        return x

class GINDelete(GIN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, mask_1hop=None, mask_2hop=None):
        with torch.no_grad():
            x = self.conv1(x, edge_index)
        
        x = self.deletion1(x, mask_1hop)

        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.deletion2(x, mask_2hop)

        return x

class RGCNDelete(GIN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index):
        with torch.no_grad():
            x = self.conv1(x, edge_index)
        
        x = self.deletion1(x)

        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.deletion2(x)

        return x

class RGATDelete(GIN):
    def __init__(self, args, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args)
        self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
        self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index):
        with torch.no_grad():
            x = self.conv1(x, edge_index)
        
        x = self.deletion1(x)

        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.deletion2(x)

        return x
