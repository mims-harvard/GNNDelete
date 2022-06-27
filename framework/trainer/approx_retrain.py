import os
import wandb
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch.utils.data import DataLoader, TensorDataset

from .base import Trainer
from ..evaluation import *
from ..utils import *


DTYPE = np.float16

class ApproxTrainer(Trainer):
    '''This code is adapted from https://github.com/zleizzo/datadeletion'''

    def gram_schmidt(self, X):
        """
        Uses numpy's qr factorization method to perform Gram-Schmidt.
        Args:
            X: (k x d matrix) X[i] = i-th vector
        Returns:
            U: (k x d matrix) U[i] = i-th orthonormal vector
            C: (k x k matrix) Coefficient matrix, C[i] = coeffs for X[i], X = CU
        """
        (k, d) = X.shape
        if k <= d:
            q, r = np.linalg.qr(np.transpose(X))
        else:
            q, r = np.linalg.qr(np.transpose(X), mode='complete')
        U = np.transpose(q)
        C = np.transpose(r)
        return U, C
    
    def LKO_pred(self, X, Y, ind, H=None, reg=1e-4):
        """
        Computes the LKO model's prediction values on the left-out points.
        Args:
            X: (n x d matrix) Covariate matrix
            Y: (n x 1 vector) Response vector
            ind: (k x 1 list) List of indices to be removed
            H: (n x n matrix, optional) Hat matrix X (X^T X)^{-1} X^T
        Returns:
            LKO: (k x 1 vector) Retrained model's predictions on X[i], i in ind
        """
        n = len(Y)
        k = len(ind)
        d = len(X[0, :])
        if H is None:
            H = np.matmul(X, np.linalg.solve(np.matmul(X.T, X) + reg * np.eye(d), X.T))

        LOO = np.zeros(k)
        for i in range(k):
            idx = ind[i]
            # This is the LOO residual y_i - \hat{y}^{LOO}_i
            LOO[i] = (Y[idx] - np.matmul(H[idx, :], Y)) / (1 - H[idx, idx])

        # S = I - T from the paper
        S = np.eye(k)
        for i in range(k):
            for j in range(k):
                if j != i:
                    idx_i = ind[i]
                    idx_j = ind[j]
                    S[i, j] = -H[idx_i, idx_j] / (1 - H[idx_i, idx_i])

        LKO = np.linalg.solve(S, LOO)

        return Y[ind] - LKO


    def lin_res(self, X, Y, theta, ind, H=None, reg=1e-4):
        """
        Approximate retraining via the projective residual update.
        Args:
            X: (n x d matrix) Covariate matrix
            Y: (n x 1 vector) Response vector
            theta: (d x 1 vector) Current value of parameters to be updated
            ind: (k x 1 list) List of indices to be removed
            H: (n x n matrix, optional) Hat matrix X (X^T X)^{-1} X^T
        Returns:
            updated: (d x 1 vector) Updated parameters
        """
        d = len(X[0])
        k = len(ind)

        # Step 1: Compute LKO predictions
        LKO = self.LKO_pred(X, Y, ind, H, reg)

        # Step 2: Eigendecompose B
        # 2.I
        U, C = self.gram_schmidt(X[ind, :])
        # 2.II
        Cmatrix = np.matmul(C.T, C)
        eigenval, a = np.linalg.eigh(Cmatrix)
        V = np.matmul(a.T, U)

        # Step 3: Perform the update
        # 3.I
        grad = np.zeros_like(theta) # 2D grad
        for i in range(k):
            grad += (X[ind[i], :] * theta - LKO[i]) * X[ind[i], :]
        # 3.II
        step = np.zeros_like(theta) # 2D grad
        for i in range(k):
            factor = 1 / eigenval[i] if eigenval[i] > 1e-10 else 0
            step += factor * V[i, :] * grad * V[i, :]
        # 3.III
        return step
        # update = theta - step
        # return update

    @torch.no_grad()
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model=None):
        model.eval()
        best_loss = 100000

        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index[:, data.dr_mask],
            num_nodes=data.num_nodes,
            num_neg_samples=data.dr_mask.sum())

        z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
        edge_index_all = torch.cat([data.train_pos_edge_index[:, data.dr_mask], neg_edge_index], dim=1)

        X = z[edge_index_all[0]] * z[edge_index_all[1]]
        Y = self.get_link_labels(data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
        X = X.cpu()
        Y = Y.cpu()

        # According to the code, theta should be of (d, d). So only update the weights of the last layer
        theta = model.conv2.lin.weight.cpu().numpy()
        ind = [int(i) for i in self.args.df_idx.split(',')]

        # Not enough RAM for solving matrix inverse. So break into multiple batches
        update = []
        loader = DataLoader(TensorDataset(X, Y), batch_size=4096, num_workers=8)
        for x, y in tqdm(loader, desc='Unlearning'):
            
            x = x.numpy()
            y = y.numpy()

            update_step = self.lin_res(x, y, theta.T, ind)
            update.append(torch.tensor(update_step))

        update = torch.stack(update).mean(0)
        model.conv2.lin.weight = torch.nn.Parameter(model.conv2.lin.weight - update.t().cuda())

        print(f'Update model weights from {torch.norm(torch.tensor(theta))} to {torch.norm(model.conv2.lin.weight)}')

        valid_loss, auc, aup, df_logt, logit_all_pair = self.eval(model, data, 'val')

        self.trainer_log['log'].append({
            'dt_loss': valid_loss,
            'dt_auc': auc,
            'dt_aup': aup
        })

        # Save
        ckpt = {
            'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'node_emb': None,
            'optimizer_state': None,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
