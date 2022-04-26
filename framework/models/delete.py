import torch
import torch.nn as nn
import pytorch_lightning as pl

from .rgcn import RGCN


class Delete(pl.LightningModule):
    def __init__(self, args, num_rel):
        super().__init__(args, num_rel)
   
    def training_step(self, batch, batch_idx):
        loss, label, proba = self.shared_step(batch, 'train')
        
        log = {'loss': loss.item()}
        self.log_dict(log)

        return log

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 'valid')
        auc, aup, mr, mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10 = self.get_eval_metrics(label, proba)
        
        log = {
            'loss': loss.item(), 
            'AUC': auc,
            'AUP': aup,
            'MR': mr,
            'MRR': mrr,
            'Hits@1': hits_at_1,
            'Hits@3': hits_at_3,
            'Hits@5': hits_at_5,
            'Hits@10': hits_at_10,
        }
        self.log_dict(log)

        return log

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 'test')
        auc, aup, mr, mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10 = self.get_eval_metrics(label, proba)
        
        log = {
            'loss': loss.item(), 
            'AUC': auc,
            'AUP': aup,
            'MR': mr,
            'MRR': mrr,
            'Hits@1': hits_at_1,
            'Hits@3': hits_at_3,
            'Hits@5': hits_at_5,
            'Hits@10': hits_at_10,
        }
        self.log_dict(log)

        return log

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return [optimizer], [None]
