import boto3
import awswrangler as wr
from tqdm.auto import tqdm
from itertools import product

import numpy as np
import pandas as pd
import torch

from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay

from .utils import *


def get_true_chem_link(disease, df, score_col, cd_train, cd_test, all_chem_set, 
                       num_with_ground_truth_data=None):
    r"""Get chemical links in test set. This can be used to format the results 
        for CRank and calculate performance of the prediction (per disease).

    Parameters:
    -----------
        disease: name of the disease
        df: all prediction
        score_col: column name of the prediction output
        cd_train: dict {disease: set(chemicals)}, adjlist of training data
        cd_test: dict {disease: set(chemicals)}, adjlist of valid + test data
        all_chem_set: set, all chemicals in the graph
        num_with_ground_truth_data: int, number of pairs with ground truth data
    """

    df = df[df.disease == disease]
    df = df.rename(columns={score_col: 'prediction'}).sort_values(by='cid')
    df = df[['cid', 'disease', 'prediction']]
    
    train_chem = cd_train.get(disease)    # chemical-disease links in training set
    if not train_chem:
        train_chem = set()
    test_chem = cd_test.get(disease)      # chemical-disease links in valid + test set
    if not test_chem:
        test_chem = set()
    chem_subset = sorted(list(all_chem_set - train_chem)) # Links to appear in final output
    
    df['label'] = 0
    df.loc[df.cid.isin(test_chem), 'label'] = 1
    df = df[~df.cid.isin(train_chem)]
    
    if len(df['label'].value_counts()) > 1:
        if num_with_ground_truth_data:
            num_with_ground_truth_data += 1
    
    return df


def get_eval_metrics(disease_name, pred, cd_train, cd_test, cids, do_random_expectation=1000, ax=None, show=None, top_k=500):
    mean_fpr = np.linspace(0, 1, 101)
    precisions = []; recalls = []; aucs = []; aups = []; random_precisions_mean = []; random_precisions_std = []

    for i, d in enumerate(disease_name):
        df = get_true_chem_link(d, pred, 'therapeutic_score', cd_train, cd_test, cids)
        logit = df.prediction
        label = df.label

        if len(df.label.value_counts()) > 1:        
            predicted = list(df.sort_values(by='prediction', ascending=False).cid)
            true = cd_test[d]

            p_at_k = precision_at_k(true, predicted, top_k)
            r_at_k = recall_at_k(true, predicted, top_k)
            auc, tpr = plot_roc(label, logit, mean_fpr, ax, show)
            ap = plot_pr(label, logit, ax, show)

            if do_random_expectation:
                mean, std = random_precision_at_k(true, predicted, top_k, do_random_expectation)

        else:
            p_at_k = np.nan; r_at_k = np.nan; auc = np.nan; ap = np.nan; mean = np.nan; std = np.nan

        precisions.append(p_at_k)
        recalls.append(r_at_k)
        aucs.append(auc)
        aups.append(ap)
        random_precisions_mean.append(mean)
        random_precisions_std.append(std)
        
        if ax is not None:
            ax.set(title=d, xlabel='', ylabel='')
            ax.legend(loc='lower right')
    
    return precisions, recalls, aucs, aups, random_precisions_mean, random_precisions_std


umap_params = {
    'A1': {'n_neighbors': 10, 'min_dist': 0.25}, 
    'A2': {'n_neighbors': 10, 'min_dist': 0.8}, 
    'A3': {'n_neighbors': 5, 'min_dist': 0.5}, 
    'A4': {'n_neighbors': 10, 'min_dist': 1}
}

def do_inference(source_node_ids, target_node_ids, source_node_names, target_node_names, 
                 relations, node_embedding, model, device='cuda'):
    
    pred_score_list = [np.zeros((len(source_node_ids), len(target_node_ids))) for r in relations]
    for idx, source in enumerate(source_node_ids):
        # Batch the prediction to run
        edge_list = [torch.LongTensor([[source] * len(target_node_ids), target_node_ids]).to(device) 
                     for r in relations]
        edge_type_list = [torch.LongTensor([r] * len(target_node_ids)).to(device) 
                          for r in relations]
        edge = torch.cat(edge_list, axis=1)
        edge_type = torch.cat(edge_type_list)

        score = model.decode_distmult(node_embedding, edge, edge_type)
        score = score.reshape((-1, len(relations))).cpu().detach().numpy()
        
        # Score for each edge type
        for i in range(len(relations)):
            pred_score_list[i][idx, :] = score[:, i]
    
    # Plain cosine similarity as score
    cosine_score = cosine_similarity(
        node_embedding[source_node_ids].cpu().numpy(), 
        node_embedding[target_node_ids].cpu().numpy())
    
    # A1-A4 pipelines in the Covid paper
    # Build an \epsilon similarity graph in 2D
    # Find the nearest neighbors
    embeddings = node_embedding[np.concatenate([source_node_ids, target_node_ids])].cpu().numpy()
    umap_ranking = {}
    for i in range(1, 5):
        pipeline_name = f'A{i}'
        umap_args = umap_params[pipeline_name]
        
        coords = UMAP(n_components=2, metric='cosine', random_state=0, **umap_args).fit_transform(embeddings)

        distances = euclidean_distances(X=coords[:len(source_node_ids)], Y=coords[len(source_node_ids):])
        umap_ranking[pipeline_name] = -1 * distances.flatten()
    
    umap_ranking = pd.DataFrame(umap_ranking)

    pred = pd.DataFrame({
        'cid': [int(i[0]) for i in product(source_node_names, target_node_names)],
        'disease': [i[1] for i in product(source_node_names, target_node_names)],
        'source_id': [i[0] for i in product(source_node_ids, target_node_ids)],
        'target_id': [i[1] for i in product(source_node_ids, target_node_ids)],
        'therapeutic_score': pred_score_list[0].flatten(),
        'markermechanism_score': pred_score_list[1].flatten(),
        'cosine_score': cosine_score.flatten()
    })

    pred = pd.concat([pred, umap_ranking], axis=1)

    return pred

def plot_roc(label, score, mean_fpr, ax=None, show_auc=False):
    fpr, tpr, thresh = roc_curve(label, score)
    auc = roc_auc_score(label, score)
    
    if ax is not None:
        if show_auc:
            display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
        else:
            display = RocCurveDisplay(fpr=fpr, tpr=tpr)
        display.plot(ax)
    
    tpr = np.interp(mean_fpr, fpr, tpr)
    tpr[0] = 0.0

    return auc, tpr
    
def plot_pr(label, score, ax=None, show_auc=False):
    precision, recall, thresh = precision_recall_curve(label, score)
    ap = average_precision_score(label, score)

    if ax is not None:
        if show_auc:
            display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap)
        else:
            display = PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot(ax)

    return ap

def barplot_err_seaborn(x, y, xerr=None, yerr=None, data=None, **kwargs):

    _data = []
    for i in data.index:

        _data_i = pd.concat([data.loc[i:i]]*3, ignore_index=True, sort=False)
        _row = data.loc[i]
        if xerr is not None:
            _data_i[x] = [_row[x]-_row[xerr], _row[x], _row[x]+_row[xerr]]
        if yerr is not None:
            _data_i[y] = [_row[y]-_row[yerr], _row[y], _row[y]+_row[yerr]]
        _data.append(_data_i)

    _data = pd.concat(_data, ignore_index=True, sort=False)

    _ax = sns.barplot(x=x, y=y, data=_data, ci='sd', **kwargs)

    return _ax

def precision_at_k(label, pred, k=10):
    label_set = set(label)
    pred_set = set(pred[:k])

    return len(label_set & pred_set) / k

def recall_at_k(label, pred, k=10):
    label_set = set(label)
    pred_set = set(pred[:k])
    
    return len(label_set & pred_set) / len(label_set)

def random_precision_at_k(true, predicted, top_k, ntimes=1000):
    results = []
    for r in range(ntimes):
        np.random.shuffle(predicted)
        results.append(precision_at_k(true, predicted, top_k))
    
    return np.mean(results), np.std(results)
