import argparse


num_edge_type_mapping = {
    'FB15k-237': 237,
    'WordNet18': 18,
    'WordNet18RR': 11,
    'ogbl-biokg': 51
}

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--unlearning_model', type=str, default='retrain',
                        help='unlearning method')
    parser.add_argument('--gnn', type=str, default='gcn', 
                        help='GNN architecture')
    parser.add_argument('--in_dim', type=int, default=128, 
                        help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default=64, 
                        help='output dimension')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='data dir')
    parser.add_argument('--df', type=str, default='none',
                        help='Df set to use')
    parser.add_argument('--df_idx', type=str, default='none',
                        help='indices of data to be deleted')
    parser.add_argument('--df_size', type=float, default=0.5,
                        help='Df size')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='dataset')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=8192, 
                        help='batch size for GraphSAINTRandomWalk sampler')
    parser.add_argument('--walk_length', type=int, default=2,
                        help='random walk length for GraphSAINTRandomWalk sampler')
    parser.add_argument('--num_steps', type=int, default=32,
                        help='number of steps for GraphSAINTRandomWalk sampler')

    # Training
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, 
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        help='optimizer to use')
    parser.add_argument('--epochs', type=int, default=3000, 
                        help='number of epochs to train')
    parser.add_argument('--valid_freq', type=int, default=100,
                        help='# of epochs to do validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='checkpoint folder')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha in loss function')
    parser.add_argument('--neg_sample_random', type=str, default='non_connected',
                        help='type of negative samples for randomness')
    parser.add_argument('--loss_fct', type=str, default='mse_mean',
                        help='loss function. one of {mse, kld, cosine}')
    parser.add_argument('--loss_type', type=str, default='both_layerwise',
                        help='type of loss. one of {both_all, both_layerwise, only2_layerwise, only2_all, only1}')

    # GraphEraser
    parser.add_argument('--num_clusters', type=int, default=10, 
                        help='top k for evaluation')
    parser.add_argument('--kmeans_max_iters', type=int, default=1, 
                        help='top k for evaluation')
    parser.add_argument('--shard_size_delta', type=float, default=0.005)
    parser.add_argument('--terminate_delta', type=int, default=0)

    # GraphEditor
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--num_remove_links', type=int, default=11)
    parser.add_argument('--parallel_unlearning', type=int, default=4)

    parser.add_argument('--lam', type=float, default=0)
    parser.add_argument('--regen_feats', action='store_true')
    parser.add_argument('--regen_neighbors', action='store_true')
    parser.add_argument('--regen_links', action='store_true')
    parser.add_argument('--regen_subgraphs', action='store_true')
    parser.add_argument('--hop_neighbors', type=int, default=20)


    # Evaluation
    parser.add_argument('--topk', type=int, default=500, 
                        help='top k for evaluation')
    parser.add_argument('--eval_on_cpu', type=bool, default=False, 
                        help='whether to evaluate on CPU')

    # KG
    parser.add_argument('--num_edge_type', type=int, default=None, 
                        help='number of edges types')

    args = parser.parse_args()

    if 'ogbl' in args.dataset:
        args.eval_on_cpu = True

    # For KG
    if args.gnn in ['rgcn', 'rgat']:
        args.lr = 1e-3
        args.epochs = 3000
        args.valid_freq = 500
        args.batch_size //= 2
        args.num_edge_type = num_edge_type_mapping[args.dataset]
        args.eval_on_cpu = True
        # args.in_dim = 512
        # args.hidden_dim = 256
        # args.out_dim = 128

    if args.unlearning_model in ['original', 'retrain']:
        args.epochs = 2000
        args.valid_freq = 500
        
        # For large graphs
        if args.gnn not in ['rgcn', 'rgat'] and 'ogbl' in args.dataset:
            args.epochs = 600
            args.valid_freq = 200
        if args.gnn in ['rgcn', 'rgat'] and 'ogbl' in args.dataset:
            args.batch_size = 1024

    if 'gnndelete' in args.unlearning_model:
        if args.gnn not in ['rgcn', 'rgat'] and 'ogbl' in args.dataset:
            args.epochs = 600
            args.valid_freq = 100
        if args.gnn in ['rgcn', 'rgat']:
            if args.dataset == 'WordNet18':
                args.epochs = 50
                args.valid_freq = 2
                args.batch_size = 1024
            if args.dataset == 'ogbl-biokg':
                args.epochs = 50
                args.valid_freq = 10
                args.batch_size = 64

    elif args.unlearning_model == 'gradient_ascent':
        args.epochs = 10
        args.valid_freq = 1
    
    elif args.unlearning_model == 'descent_to_delete':
        args.epochs = 1

    elif args.unlearning_model == 'graph_editor':
        args.epochs = 400
        args.valid_freq = 200


    if args.dataset == 'ogbg-molhiv':
        args.epochs = 100
        args.valid_freq = 5

    return args
