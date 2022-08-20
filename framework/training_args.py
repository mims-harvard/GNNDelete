import argparse


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
    parser.add_argument('--batch_size', type=int, default=128, 
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
    parser.add_argument('--valid_freq', type=int, default=10,
                        help='# of epochs to do validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='checkpoint folder')

    # GraphEraser
    parser.add_argument('--num_clusters', type=int, default=10, 
                        help='top k for evaluation')
    parser.add_argument('--kmeans_max_iters', type=int, default=1, 
                        help='top k for evaluation')
    parser.add_argument('--shard_size_delta', type=float, default=0.005)
    parser.add_argument('--terminate_delta', type=int, default=0)

    # Evaluation
    parser.add_argument('--topk', type=int, default=500, 
                        help='top k for evaluation')
    parser.add_argument('--eval_on_cpu', type=bool, default=False, 
                        help='whether to evaluate on CPU')

    args = parser.parse_args()

    if 'ogbl' in args.dataset:
        args.valid_freq = 20
        args.epochs = 200
        args.batch_size = args.batch_size * 16
        args.eval_on_cpu = True

    return args
