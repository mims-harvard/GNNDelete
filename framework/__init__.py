from .models import GCN, GAT, GIN, RGCN, RGAT, GCNDelete, GATDelete, GINDelete, RGCNDelete, RGATDelete
from .trainer.base import Trainer, KGTrainer, NodeClassificationTrainer
from .trainer.retrain import RetrainTrainer, KGRetrainTrainer
from .trainer.gnndelete import GNNDeleteTrainer
from .trainer.gnndelete_nodeemb import GNNDeleteNodeembTrainer, KGGNNDeleteNodeembTrainer
from .trainer.gradient_ascent import GradientAscentTrainer, KGGradientAscentTrainer
from .trainer.descent_to_delete import DtdTrainer
from .trainer.approx_retrain import ApproxTrainer
from .trainer.graph_eraser import GraphEraserTrainer
from .trainer.graph_editor import GraphEditorTrainer
from .trainer.member_infer import MIAttackTrainer, MIAttackTrainerNode


trainer_mapping = {
    'original': Trainer,
    'original_node': NodeClassificationTrainer,
    'retrain': RetrainTrainer,
    'gnndelete': GNNDeleteTrainer,
    'gradient_ascent': GradientAscentTrainer,
    'descent_to_delete': DtdTrainer,
    'approx_retrain': ApproxTrainer,
    'gnndelete_mse': GNNDeleteTrainer,
    'gnndelete_kld': GNNDeleteTrainer,
    'gnndelete_nodeemb': GNNDeleteNodeembTrainer,
    'gnndelete_cosine': GNNDeleteTrainer,
    'graph_eraser': GraphEraserTrainer,
    'graph_editor': GraphEditorTrainer,
    'member_infer_all': MIAttackTrainer,
    'member_infer_sub': MIAttackTrainer,
    'member_infer_all_node': MIAttackTrainerNode,
    'member_infer_sub_node': MIAttackTrainerNode,
}

kg_trainer_mapping = {
    'original': KGTrainer,
    'retrain': KGRetrainTrainer,
    'gnndelete': KGGNNDeleteNodeembTrainer,
    'gradient_ascent': KGGradientAscentTrainer,
    'descent_to_delete': DtdTrainer,
    'approx_retrain': ApproxTrainer,
    'gnndelete_mse': GNNDeleteTrainer,
    'gnndelete_kld': GNNDeleteTrainer,
    'gnndelete_cosine': GNNDeleteTrainer,
    'gnndelete_nodeemb': KGGNNDeleteNodeembTrainer,
    'graph_eraser': GraphEraserTrainer,
    'member_infer_all': MIAttackTrainer,
    'member_infer_sub': MIAttackTrainer,
}


def get_model(args, mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None):

    if 'gnndelete' in args.unlearning_model:
        model_mapping = {'gcn': GCNDelete, 'gat': GATDelete, 'gin': GINDelete, 'rgcn': RGCNDelete, 'rgat': RGATDelete}

    else:
        model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN, 'rgcn': RGCN, 'rgat': RGAT}

    return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)


def get_trainer(args):
    if args.gnn in ['rgcn', 'rgat']:
        return kg_trainer_mapping[args.unlearning_model](args)

    else:
        return trainer_mapping[args.unlearning_model](args)
