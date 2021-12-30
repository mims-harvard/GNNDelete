import os
import pickle
import argparse
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn

from rgcn.models.rgcn import RGCN
from rgcn.data_loader import get_loader
from rgcn.utils import log_metrics
from train import do_eval, get_node_embedding, get_score

torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument('--in_dim', type=int, default=128, 
                        help='frequency of showing training results on screen')
    parser.add_argument('--out_dim', type=int, default=64, 
                        help='frequency of showing training results on console')
    parser.add_argument('--hidden_dim', type=int, default=64, 
                        help='frequency of saving the latest results')
    parser.add_argument('--num_bases', type=int, default=None, 
                        help='frequency of saving the latest results')
    parser.add_argument('--num_blocks', type=int, default=None, 
                        help='frequency of saving the latest results')

    # Data
    parser.add_argument('--dataset', type=str, default='WIKIDATA12k',
                        help='dataset')
    parser.add_argument('--node_feature', type=str, default='random',
                        help='input node feature.')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='for re-produce purpose (split, etc.)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size for GraphSAINTRandomWalk sampler')
    parser.add_argument('--walk_length', type=int, default=16,
                        help='random walk length for GraphSAINTRandomWalk sampler')
    parser.add_argument('--num_steps', type=int, default=32,
                        help='number of steps for GraphSAINTRandomWalk sampler')

    # Training
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, 
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        help='optimizer to use')
    parser.add_argument('--epochs', type=int, default=10000, 
                        help='number of epochs to train')
    parser.add_argument('--init_checkpoint', action='store_true', 
                        help='whether to start from checkpoint')
    parser.add_argument('--valid_freq', type=int, default=50,
                        help='# of epochs to do validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='checkpoint folder')

    args = parser.parse_args()

    return args

def train_step(model, loader, valid_triplets, true_triplets, optimizer, num_nodes, args):
    model.train()
    # node_embedding = torch.zeros((num_nodes, args.out_dim))

    for batch in loader:

        # Message passing
        batch = batch.to('cuda')
        embedding = model(batch.x, batch.edge_index, batch.edge_type)
        loss, label, proba = model.training_step(embedding, batch)
        # node_embedding[batch.node_id.cpu()] = embedding.detach().cpu()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

    node_embedding = get_node_embedding(model, loader.data)
    _ = do_eval(node_embedding, model, loader, valid_triplets, true_triplets, args, 'valid')
    
    return node_embedding

def compute_sigma(num_examples, iterations, lipshitz, smooth, strong, epsilon, delta):
    """Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf"""
    gamma = (smooth - strong) / (smooth + strong)
    numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
    denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
    # print('sigma', numerator, denominator, numerator / denominator)
    
    return numerator / denominator

def publish(model, sigma):
    """Publishing function which adds Gaussian noise with scale sigma."""

    model.W = nn.Parameter(model.W + torch.empty_like(model.W).normal_(0, sigma))
    model.rgcn1.weight = nn.Parameter(model.rgcn1.weight + torch.empty_like(model.rgcn1.weight).normal_(0, sigma))
    model.rgcn2.weight = nn.Parameter(model.rgcn2.weight + torch.empty_like(model.rgcn2.weight).normal_(0, sigma))

def main():
    args = parse_args()
    config = '-'.join([str(i) for i in 
        [args.dataset, args.in_dim, args.hidden_dim, args.out_dim, args.epochs, args.lr]])

    loader, valid_triplets, test_triplets, true_triplets, num_nodes, num_edges, num_edge_type = get_loader(args)

    # Model
    model = RGCN(args, num_nodes, num_edges, num_edge_type)
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Delete
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, f'{config}.pt'))
    node_embedding = checkpoint['ent_emb']
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']

    # Delete
    score_list = []
    num_steps = len(loader)
    num_train = loader.data.train_mask.sum()//2
    print('num_train', num_train, optimizer.param_groups[0]['lr'])
    for i in trange(20, desc='Delete data'): 

        # Dataset
        loader, valid_triplets, test_triplets, true_triplets, num_nodes, num_edges, num_edge_type = get_loader(args, delete=list(range(i)))

        epoch_score = []
        for epoch in trange(50, desc='Iteration', leave=False):
            node_embedding = train_step(model, loader, valid_triplets, true_triplets, optimizer, num_nodes, args)
            sigma = compute_sigma(num_train, epoch+1, 1 + 0.05, 4 - 0.05, 0.05, 5, 1 / num_train / num_train)
            publish(model, sigma)
            # print('norm', get_norm(model))

            node_embedding = get_node_embedding(model, loader.data)
            score = get_score(model, node_embedding, loader.data)
            epoch_score.append(score)
        score_list.append(epoch_score)

    out_config = 'delete-' + config
    with open(os.path.join(args.checkpoint_dir, f'{out_config}.pkl'), 'wb') as f:
        pickle.dump(score_list, f)

    # node_embedding, best_model_state, best_valid_loss = do_train(model, loader, valid_triplets, true_triplets, optimizer, num_nodes, args)
    
    # Save models and node embeddings
    # ckpt = {
    #     'hparam': vars(args),
    #     'model_state': best_model_state,
    #     'ent_emb': node_embedding,
    #     'rel_emb': model.W,
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'epoch': args.epochs,
    #     'loss': best_valid_loss
    # }
    # torch.save(ckpt, os.path.join(args.checkpoint_dir, config + '.pt'))


if __name__ == "__main__":
    main()
