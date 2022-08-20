from .models import GCN, GAT, GIN, RGCN, RGAT, GCNDelete, GATDelete, GINDelete, RGCNDelete, RGATDelete
from .trainer.base import Trainer, KGTrainer
from .trainer.retrain import RetrainTrainer
from .trainer.gnndelete import GNNDeleteTrainer
from .trainer.gradient_ascent import GradientAscentTrainer
from .trainer.descent_to_delete import DtdTrainer
from .trainer.approx_retrain import ApproxTrainer
from .trainer.graph_eraser import GraphEraserTrainer
from .trainer.member_infer import MIAttackTrainer


trainer_mapping = {
    'original': Trainer,
    'retrain': RetrainTrainer,
    'gnndelete': GNNDeleteTrainer,
    'gradient_ascent': GradientAscentTrainer,
    'descent_to_delete': DtdTrainer,
    'approx_retrain': ApproxTrainer,
    'gnndelete_mse': GNNDeleteTrainer,
    'gnndelete_kld': GNNDeleteTrainer,
    'gnndelete_cosine': GNNDeleteTrainer,
    'graph_eraser': GraphEraserTrainer,
    'member_infer_all': MIAttackTrainer,
    'member_infer_sub': MIAttackTrainer,
}

kg_trainer_mapping = {
    'original': KGTrainer,
    'retrain': RetrainTrainer,
    'gnndelete': GNNDeleteTrainer,
    'gradient_ascent': GradientAscentTrainer,
    'descent_to_delete': DtdTrainer,
    'approx_retrain': ApproxTrainer,
    'gnndelete_mse': GNNDeleteTrainer,
    'gnndelete_kld': GNNDeleteTrainer,
    'gnndelete_cosine': GNNDeleteTrainer,
    'graph_eraser': GraphEraserTrainer,
    'member_infer_all': MIAttackTrainer,
    'member_infer_sub': MIAttackTrainer,
}


def get_model(args, mask_1hop=None, mask_2hop=None, num_edge_type=None):

    if 'gnndelete' in args.unlearning_model:
        model_mapping = {'gcn': GCNDelete, 'gat': GATDelete, 'gin': GINDelete, 'rgcn': RGCNDelete, 'rgat': RGATDelete}

    else:
        model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN, 'rgcn': RGCN, 'rgat': RGAT}

    return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_edge_type=num_edge_type)


def get_trainer(args):
    if args.gnn in ['rgcn', 'rgat']:
        return kg_trainer_mapping[args.unlearning_model](args)

    else:
        return trainer_mapping[args.unlearning_model](args)
