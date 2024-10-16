from frag_gnn.utils.misc_utils import get_best_ckpt_from_wandb
    
if __name__ == "__main__":

    inchikey_splits_runs = [

        'ohtiivyj', #fraggnn_d3_nodemlp_mnce_nocp
        'kme8fry7', #fraggnn_d4_nodemlp_mnce_nocp
        'f8p1azsm', #iceberg_inten_100
        '019gu0eq', #iceberg_gen
        'yjm4i8lm', #iceberg_inten_hun
        'gt17ukfh', #scaffold_fraggnn_d4_v2
        'poq59s7k', #scaffold_fraggnn_d3_v2
    ]
    
    save_dp = "./saved_ckpts/icml_inchikey"
    for run_id in inchikey_splits_runs:
        get_best_ckpt_from_wandb(save_dp, run_id)

    scaffold_splits_runs = [
        'lgeo7mjz', #fraggnn_d3_nodemlp_mnce
        'gmpfciq9', #fraggnn_d4_nodemlp_mnce
        'xnmchi3a', #iceberg_inten_100
        'dizgfqmp', #iceberg_gen
        'gzxjnb44', #scaffold_fraggnn_d4_v2
        'e1dehj4e', #scaffold_fraggnn_d3_v2
    ]

    save_dp = "./saved_ckpts/icml_scaffold"
    for run_id in scaffold_splits_runs:
        get_best_ckpt_from_wandb(save_dp, run_id)

    # nist20_inchikey_fraggnn_d4_v2 
    inchikey_splits_runs = [
        '6yilbgpd', 
        'moldmao1', 
        'p1v5us2q', 
        '3neggg68', 
        'tvby427i',
    ]

    save_dp = "./saved_ckpts/icml_inchikey_seeds"
    for run_id in inchikey_splits_runs:
        get_best_ckpt_from_wandb(save_dp, run_id)

    # nist20_inchikey_fraggnn_d3_v2 
    inchikey_splits_runs = [
        '2zm4rito', 
        'qi6ickzl', 
        '3l3nfgaf', 
        'h3hdj8xf', 
        '4hhonevu',
    ]

    save_dp = "./saved_ckpts/icml_inchikey_seeds"
    for run_id in inchikey_splits_runs:
        get_best_ckpt_from_wandb(save_dp, run_id)

    
    # nist20_inchikey_scaffold_d4_v2 
    inchikey_splits_runs = [
        'yuzyxy37', 
        'jppr9ago', 
        '2lnqlzmy', 
        'trhcwsow', 
        'ox038p0i',
    ]

    save_dp = "./saved_ckpts/icml_scaffold_seeds"
    for run_id in inchikey_splits_runs:
        get_best_ckpt_from_wandb(save_dp, run_id)
