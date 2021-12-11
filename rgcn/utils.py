import numpy as np
import torch


def force_reproducibility(seed):
    ''' Set random seed for numpy and torch '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_metrics(mode, epoch, log):
    metric = {}
    for m in log[0].keys():
        metric[m] = sum([l[m] for l in log])/len(log)
        
#     print(f'Epoch {epoch:03d}, {mode}')
#     for m in metric:
#         print(f'{m:8}: {metric[m]:7.4f}', end=' | ')
#     print(f'\n{"":{"-"}<{64}}')

    return metric
