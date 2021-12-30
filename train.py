import os
import argparse
from tqdm import tqdm, trange
import torch
import torch.nn as nn

from rgcn.models.rgcn import RGCN
from rgcn.data_loader import get_loader
from rgcn.utils import log_metrics


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
    parser.add_argument('--dataset', type=str, default='YAGO11k',
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
    parser.add_argument('--epochs', type=int, default=2000, 
                        help='number of epochs to train')
    parser.add_argument('--init_checkpoint', action='store_true', 
                        help='whether to start from checkpoint')
    parser.add_argument('--valid_freq', type=int, default=50,
                        help='# of epochs to do validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='checkpoint folder')

    # Evaluation
    parser.add_argument('--topk', type=int, default=500, 
                        help='top k for evaluation')

    args = parser.parse_args()

    return args

def get_eval_metric(node_embedding, model, dataloader, num_edge_type, num_edge_meta_type, id2rel):
    """Get performance on valid / test set"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_log = []
    embedding = torch.FloatTensor(node_embedding).to(device)
    for i, batch in enumerate(dataloader):
        eval_log_step, auc_edge, aup_edge, f1_edge = test_step(
            embedding[batch.nid], model, batch.to(device), num_edge_type, num_edge_meta_type, 'test')
        eval_log.append(eval_log_step)
    
    metric = log_metrics('Test', 0, eval_log)
    
    print('Overall performance:', metric)
    
    if len(auc_edge) <= 50:
        print(f'{"Edge type":<24}  |  AUC per edge type  |  AUP per edge type  |  Macro-F1 per edge type')
        for index in auc_edge.keys():
            print(f"{id2rel[str(index)]+':':<24}  |  {auc_edge[index]:>20.4f}  |  {aup_edge[index]:>20.4f}  |  {f1_edge[index]:>20.4f}")
    else:
        print('Performance per edge type is loo long to be shown in terminal.')


def do_train(model, loader, valid_triplets, true_triplets, optimizer, num_nodes, args):
    # node_embedding = torch.zeros((num_nodes, args.out_dim))

    best_loss = 100000
    for epoch in trange(args.epochs, desc='Epoch'):
        model.train()
        total_step = 0
        total_loss = 0
        for batch in loader: #, desc='Interation', leave=False):

            # Message passing
            batch = batch.to('cuda')
            embedding = model(batch.x, batch.edge_index, batch.edge_type)
            loss, label, proba = model.training_step(embedding, batch)
            # node_embedding[batch.node_id.cpu()] = embedding.detach().cpu()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            total_step += 1
            total_loss += loss.item()

        if epoch % args.valid_freq == 0:
            msg = [
                f'Epoch: {epoch:>4d}', 
                f'train loss: {total_loss / total_step:.6f}'
            ]
            tqdm.write(' | '.join(msg))

            node_embedding = get_node_embedding(model, loader.data)
            valid_loss = do_eval(node_embedding, model, loader, valid_triplets, true_triplets, args, 'valid')
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model_state = model.state_dict()
    
    return best_model_state, best_loss


@torch.no_grad()
def do_eval(node_embedding, model, loader, valid_triplets, true_triplets, args, stage='valid'):
    model.eval()

    total_step = 0
    total_loss = 0
    total_auc = 0
    total_aup = 0
    # total_mr = 0
    # total_mrr = 0
    # total_hits_at_1 = 0
    # total_hits_at_3 = 0
    # total_hits_at_5 = 0
    # total_hits_at_10 = 0

    for batch in tqdm(loader, desc='Interation', leave=False):
        batch = batch.to('cuda')
        embedding = node_embedding[batch.node_id].to('cuda')
        loss, auc, aup = model.eval_step(embedding, batch, stage, valid_triplets, true_triplets)
        # loss, auc, aup, mr, mrr, hits_at_1, hits_at_3, hits_at_5, hits_at_10 = model.eval_step(embedding, batch, stage)

        total_step += 1
        total_loss += loss.item()
        total_auc += auc
        total_aup += aup
        # total_mr += mr
        # total_mrr += mrr
        # total_hits_at_1 += hits_at_1
        # total_hits_at_3 += hits_at_3
        # total_hits_at_5 += hits_at_5
        # total_hits_at_10 += hits_at_10

    msg = [
        f'{stage}',
        f'loss: {total_loss / total_step:.6f}',
        f'auc: {total_auc / total_step:.4f}',
        f'aup: {total_aup / total_step:.4f}',
        # f'MR: {total_mr / total_step:.2f}',
        # f'MRR: {total_mrr / total_step:.2f}',
        # f'Hits@1: {total_hits_at_1 / total_step:.2%}',
        # f'Hits@3: {total_hits_at_3 / total_step:.2%}',
        # f'Hits@5: {total_hits_at_5 / total_step:.2%}',
        # f'Hits@10: {total_hits_at_10 / total_step:.2%}',
    ]
    tqdm.write(' | '.join(msg))
    
    return total_loss / total_step

@torch.no_grad()
def get_node_embedding(model, data):
    model.eval()
    node_embedding = model(data.x.to('cuda'), data.edge_index.to('cuda'), data.edge_type.to('cuda'))

    return node_embedding

@torch.no_grad()
def get_score(model, node_embedding, data):
    model.eval()
    node_embedding = node_embedding.to('cuda')
    edge = data.edge_index.to('cuda')
    edge_type = data.edge_type.to('cuda')

    train_score = model.decode(node_embedding, edge[:, data.train_mask], edge_type[data.train_mask])
    valid_score = model.decode(node_embedding, edge[:, data.valid_mask], edge_type[data.valid_mask])
    test_score = model.decode(node_embedding, edge[:, data.test_mask], edge_type[data.test_mask])
    all_score = model.decode(node_embedding, edge, edge_type)

    return train_score, valid_score, test_score, all_score

def get_norm(model):
    return torch.norm(model.rgcn1.weight.cpy()) + torch.norm(model.rgcn2.weight.cpy()) + torch.norm(model.W.cpu())

def main():
    args = parse_args()
    config = '-'.join([str(i) for i in 
        [args.dataset, args.in_dim, args.hidden_dim, args.out_dim, args.epochs, args.lr]])

    # Dataset
    loader, valid_triplets, test_triplets, true_triplets, num_nodes, num_edges, num_edge_type = get_loader(args)

    # Model
    model = RGCN(args, num_nodes, num_edges, num_edge_type)
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train
    if args.init_checkpoint:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'test.pt'))
        node_embedding = checkpoint['ent_emb']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
    else:
        start_epoch = 0
        loss = 0.0

    best_model_state, best_valid_loss = do_train(model, loader, valid_triplets, true_triplets, optimizer, num_nodes, args)
    node_embedding = get_node_embedding(model, loader.data)
    score = get_score(model, node_embedding, loader.data)

    # Save models and node embeddings
    ckpt = {
        'hparam': vars(args),
        'model_state': best_model_state,
        'ent_emb': node_embedding,
        'rel_emb': model.W,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs,
        'loss': best_valid_loss,
        'score': score
    }
    torch.save(ckpt, os.path.join(args.checkpoint_dir, config + '.pt'))


if __name__ == "__main__":
    main()
