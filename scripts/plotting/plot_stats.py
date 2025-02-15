from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL

import wandb
import tempfile
import os
import sys
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from frag_gnn.utils.misc_utils import get_best_ckpt_from_wandb

def get_metrics_data(list_ids:list):
    # fiter to get inchikey splits

    '''
    filters = {"$and": [{"tags": "splits_exp2"}, 
                        {"tags": "inchikey"}, 
                        {"tags":"nist20"}, 
                        {"tags": "d4"},
                        {"tags": "wr00"}, 
                        {"State":"finished"},
                        {"tags": "splits"},
                        {'created_at': {
                        "$gt": '2023-12-07T17'
                                }}]}
    '''
    saved_dp = "./saved_ckpts"
    #filters = {"$and": [{"tags": "splits_exp2"},{"tags": "splits"},{"tags": "scaffold"}], "$not": {"tags": "gen"}}
    #runs = api.runs("frag-gnn/frag-gnn", include_sweeps=False, filters=filters)
    #for run in runs:
    #    print(run.id, run.name)

    all_metrics = []
    for run_id in list_ids:
        ckpt_fp, _  = get_best_ckpt_from_wandb(saved_dp, run_id)
    
        ckpt = torch.load(ckpt_fp)
        ot_metrics_types = [ 'train_mean_metrics',
                            'train_std_metrics',
                            'val_mean_metrics',
                            'val_std_metrics']
        
        metrics_data_d = { 'train_datapoint_metrics': {},
                    'val_datapoint_metrics': {},
                    'train_metrics': {},
                    'val_metrics':{}}
        
        train_num_datapoints = ckpt["state_dict"]['train_num_datapoints'].detach().cpu()
        val_num_datapoints = ckpt["state_dict"]['val_num_datapoints'].detach().cpu()
        
        for k,v in ckpt["state_dict"].items():

            if k.startswith('train_datapoint_metrics'):
                metrics_name = k.split('.')[1]
                metrics_data_d['train_datapoint_metrics'][metrics_name] = v[:val_num_datapoints].detach().cpu()

            if k.startswith('val_datapoint_metrics'):
                metrics_name = k.split('.')[1]
                metrics_data_d['val_datapoint_metrics'][metrics_name] = v[:val_num_datapoints].detach().cpu()

            for pre_fix in ot_metrics_types:
            # get val_datapoint_metrics
                if k.startswith(pre_fix):
                    metric_type = 'train_metrics' if 'train' in pre_fix else 'val_metrics'
                    metrics_name = k.split('.')[1]
                    metrics_value_type = 'mean' if 'mean' in pre_fix else 'std'
                    #print(k, metric_type, metrics_name, metrics_value_type)
                    if metrics_name not in metrics_data_d[metric_type]:
                        metrics_data_d[metric_type][metrics_name] = {"mean": None, "std": None, "epoch_num": ckpt_epoch_num}
                    metrics_data_d[metric_type][metrics_name][metrics_value_type] = v.detach().cpu()
    
        all_metrics.append({'data':metrics_data_d, 'run_id':run.id, 'name':run.name, 
                            'run_tags':run.tags, 
                            'train_num_datapoints':train_num_datapoints,
                            'val_num_datapoints':val_num_datapoints})
        #print(train_num_datapoints)
        #print(val_num_datapoints)
    return all_metrics

def plot_val_bar_plots(all_metrics, 
                       metrics_names, 
                       metrics_type,
                       title,
                       model_filter = None,
                       include_err_bar = False):

    bar_data = {}#{ m:[] for m in metrics_names}
    bar_err_data = {}#{ m:[] for m in metrics_names}
    model_names = []

    for metrics_name in metrics_names:

        for item in all_metrics:

            model_name = item['name']
            if model_name not in model_filter and model_filter is not None: 
                continue
            
            if model_filter is not None:
                model_name = model_filter[model_name]

            if model_name not in model_names:
                model_names.append(model_name)
                bar_data[model_name] = []
                bar_err_data[model_name] = []

            #print(item.keys())
      
            epoch_num = item['data'][metrics_type][metrics_name]['epoch_num']
            bar_data[model_name].append(float(item['data'][metrics_type][metrics_name]['mean'][epoch_num]))
            if include_err_bar:
                num_sample = float(item['val_num_datapoints'])
                bar_err_data[model_name].append(float(item['data'][metrics_type][metrics_name]['std'][epoch_num]) * 1.96 / np.sqrt(num_sample))
            #print(item['data'][metrics_type][metrics_name]['std'][epoch_num])

    x = np.arange(len(metrics_names))
    # print(x)
    width = 0.13  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(10,6))
    for attribute, values in bar_data.items():
        offset = width * (multiplier - len(model_names)/2 + 0.5)
        #print(x, values)
        if include_err_bar:
            rects = ax.bar(x + offset, values, width-0.05, yerr= bar_err_data[attribute], label=attribute, align='center')
        else:
            rects = ax.bar(x + offset, values, width-0.05, label=attribute, align='center')
        ax.bar_label(rects, fmt="{:.2f}", padding=3)
        multiplier += 1

    #print(model_names)
    metrics_display_names = ['Cosine Similarity\n(Hungarian Peak \n Matching 10ppm)','Cosine Similarity\n(Binned 0.01Da)', 'Sqrt Cosine Similarity\n(Binned 0.01Da)', 'Jensen-Shannon\n Similarity']   
    ax.set_xticks(x, metrics_display_names)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0.3, 1.0)
    plt.title(title)
    fig.savefig(f'./figs/{title}.png')

def plot_oos(metrics):

    oos_prob = metrics['data']['val_datapoint_metrics']['oos_prob']
    true_oos_prob = metrics['data']['val_datapoint_metrics']['true_oos_prob']
    cos_hun = metrics['data']['val_datapoint_metrics']['cos_hun']
    #wrecall = all_metrics[0]['data']['val_datapoint_metrics']['cos_hun'] 
    fig, ax = plt.subplots(layout='constrained', figsize=(10,6))

    #corr, _ = pearsonr(oos_prob, cos_hun)
    #print(corr)
    #ax.text(0.1, 0.95, f"{corr:{6}.{6}}", bbox=dict(facecolor='red', alpha=0.5))
    ax.scatter(oos_prob, cos_hun, s = [4.0] *len(oos_prob), label="True OOS", alpha=0.5)

    #corr, _ = pearsonr(true_oos_prob, cos_hun)
    #print(corr)
    #ax.text(0.1, 0.9, f"{corr:{6}.{6}}", bbox=dict(facecolor='red', alpha=0.5))
    ax.scatter(true_oos_prob, cos_hun, s = [4.0] *len(true_oos_prob), label="Predicted OOS", alpha=0.5)

    ax.set_xlabel("OOS")
    ax.set_ylabel("Cosine Similarity (Hungarian Peak Matching 10ppm)")
    ax.legend(loc='upper left', ncols=3)
    ax.set_xlim(0.0, 0.5)
    #plt.scatter(true_oos_prob, cos_hun, s = [0.3] *len(true_oos_prob), alpha=0.5)
    plt.title("OOS vs Cosine Similarity")
    plt.savefig('./figs/oos.png')
    
if __name__ == "__main__":

    inchikey_splits_runs = [
                       #'k0ykj0c9', #fraggnn_d3_nodemlp
                       #'tpo4z1yh', #fraggnn_d4_nodemlp
                       #'5j5qlwwa', # fraggnn_d3_nodemlp_mnce
                       # '4qtm9wd4', #fraggnn_d4_nodemlp_mnce
                       #'6vhbjjag', #fraggnn_d3
                       #'bwabv9wf', #fraggnn_d4
                       'lgeo7mjz',  #fraggnn_d3_nodemlp_mnce_nocp
                       'kme8fry7',  #fraggnn_d4_nodemlp_mnce_nocp
                       'f8p1azsm', #iceberg_inten_100
                       #'iri7yg0v', #iceberg_inten_all
    ]

    all_metrics = get_metrics_data(inchikey_splits_runs)

    metrics_names = ['cos_hun','cos_sim_0-01', 'cos_sim_sqrt_0-01', 'jss_hun']
    metrics_type = 'val_metrics'
    wanted_models = {
        #'fraggnn_d3_nodemlp':
        #'fraggnn_d4_nodemlp':
        'fraggnn_d3_nodemlp_mnce': 'FragGNN D3 NodeMLP',
        'fraggnn_d4_nodemlp_mnce': 'FragGNN D4 NodeMLP',
        'fraggnn_d3': 'FragGNN D3 DagGNN',
        'fraggnn_d4': 'FragGNN D4 DagGNN',
        'iceberg_inten_all': 'ICEBERG Upperbound',
        'iceberg_inten_100': 'ICEBERG',
    }
    plot_val_bar_plots(all_metrics, metrics_names, metrics_type, 'Validation Inchikey Split', wanted_models)

    scaffold_splits_runs = [
        '1kby1xpl', #fraggnn_d3_nodemlp_mnce
        'gmpfciq9', #fraggnn_d4_nodemlp_mnce
        #'pq1atl0y', #fraggnn_d3_nodemlp
        #'f788zzod', #fraggnn_d4_nodemlp
        #'kr3cu13k', #fraggnn_d3
        #'ieav1as2', #fraggnn_d4
        'xnmchi3a', #ceberg_inten_100
        'lf6jvg3h', #iceberg_inten_all
        #'xs44s8w2', #fraggnn_d3_sqrt
        #'9zvrcz9q', #fraggnn_d4_noedges
    ]
    all_metrics = get_metrics_data(scaffold_splits_runs)
    plot_val_bar_plots(all_metrics, metrics_names, metrics_type, 'Validation Scaffold Split', wanted_models)

    metrics = get_metrics_data(['4qtm9wd4'])[0]
    plot_oos(metrics)

    #plt.scatter(oos_prob, wrecall, s = [0.3] *len(true_oos_prob), alpha=0.5)
    #plt.scatter(true_oos_prob, cos_hun, s = [0.3] *len(true_oos_prob), alpha=0.5)
    #plt.savefig('./figs/oos.png')
    # cos_hun
    # cos_sim_0-01
    # ndcg_un_pess
    # node_ne
    #plt.plot(all_metrics[0]['data']['train_metrics']['cos_hun']['mean'])
    #plt.show()
    #print(all_metrics[0]['data']['train_metrics']['cos_hun']['mean'])