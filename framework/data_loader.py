import os
import torch
from torch_geometric.data import Data, GraphSAINTRandomWalkSampler


def load_dict(filename):
    '''Load entity and relation to id mapping'''

    mapping = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            mapping[l[0]] = l[1]

    return mapping

def load_edges(filename):
    with open(filename, 'r') as f:
        r = f.readlines()
    r = [i.strip().split('\t') for i in r]

    return r

def generate_true_dict(all_triples):
    heads = {(r, t) : [] for _, r, t in all_triples}
    tails = {(h, r) : [] for h, r, _ in all_triples}

    for h, r, t in all_triples:
        heads[r, t].append(h)
        tails[h, r].append(t)

    return heads, tails

def get_loader(args, delete=[]):
    prefix = os.path.join('./data', args.dataset)

    # Edges
    train = load_edges(os.path.join(prefix, 'train.txt'))
    valid = load_edges(os.path.join(prefix, 'valid.txt'))
    test = load_edges(os.path.join(prefix, 'test.txt'))
    train = [(int(i[0]), int(i[1]), int(i[2])) for i in train]
    valid = [(int(i[0]), int(i[1]), int(i[2])) for i in valid]
    test = [(int(i[0]), int(i[1]), int(i[2])) for i in test]
    train_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in train]
    valid_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in valid]
    test_rev = [(int(i[2]), int(i[1]), int(i[0])) for i in test]
    train = train + train_rev
    valid = valid + valid_rev
    test = test + test_rev
    all_edge = train + valid + test

    true_triples = generate_true_dict(all_edge)

    edge = torch.tensor([(int(i[0]), int(i[2])) for i in all_edge], dtype=torch.long).t()
    edge_type = torch.tensor([int(i[1]) for i in all_edge], dtype=torch.long)#.view(-1, 1)

    # Masks
    train_size = len(train)
    valid_size = len(valid)
    test_size = len(test)
    total_size = train_size + valid_size + test_size

    train_mask = torch.zeros((total_size,)).bool()
    train_mask[:train_size] = True

    valid_mask = torch.zeros((total_size,)).bool()
    valid_mask[train_size:train_size + valid_size] = True
    
    test_mask = torch.zeros((total_size,)).bool()
    test_mask[-test_size:] = True

    # Graph size
    num_nodes = edge.flatten().unique().shape[0]
    num_edges = edge.shape[1]
    num_edge_type = edge_type.unique().shape[0]

    # Node feature
    x = torch.rand((num_nodes, args.in_dim))

    # Delete edges
    if len(delete) > 0:
        delete_idx = torch.tensor(delete, dtype=torch.long)
        num_train_edges = train_size // 2
        train_mask[delete_idx] = False
        train_mask[delete_idx + num_train_edges] = False
        train_size -= 2 * len(delete)
    
    node_id = torch.arange(num_nodes)
    dataset = Data(
        edge_index=edge, edge_type=edge_type, x=x, node_id=node_id, 
        train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)

    dataloader = GraphSAINTRandomWalkSampler(
        dataset, batch_size=args.batch_size, walk_length=args.walk_length, num_steps=args.num_steps)

    print(f'Dataset: {args.dataset}, Num nodes: {num_nodes}, Num edges: {num_edges//2}, Num relation types: {num_edge_type}')
    print(f'Train edges: {train_size//2}, Valid edges: {valid_size//2}, Test edges: {test_size//2}')
    
    return dataloader, valid, test, true_triples, num_nodes, num_edges, num_edge_type
