from frag_gnn.utils.misc_utils import get_ckpt_from_wandb
    
if __name__ == "__main__":

    inchikey_splits_runs = [
        'yylzfd6t', #fraggnn_d3
        'a64wr0sb', #fraggnn_d4
        'ylm3miyr', #nemis 
        '2ay7zwbm', #massformer
    ]
    
    save_dp = "./saved_ckpts/tmlr_inchikey"
    for run_id in inchikey_splits_runs:
        get_ckpt_from_wandb(save_dp, run_id)

    scaffold_splits_runs = [
        'ad8s7pkj', #'fraggnn_d3' 
        'hwan9uas', #'fraggnn_d4': 
        'n28n8vzx', # neims 
        'wf1zkb7n' # massformer'
    ]

    save_dp = "./saved_ckpts/tmlr_scaffold"
    for run_id in scaffold_splits_runs:
        get_ckpt_from_wandb(save_dp, run_id)