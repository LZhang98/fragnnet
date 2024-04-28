from frag_gnn.utils.misc_utils import get_ckpt_from_wandb
    
if __name__ == "__main__":

    inchikey_splits_runs = [

        '0jeuscvs', #nist20_inchikey_fraggnn_d3_ce_lin
        'gym8mgqv', #nist20_inchikey_fraggnn_d3_ce
        'bzh7mub7', #nist20_inchikey_fraggnn_d4_ce_lin
    ]
    
    save_dp = "./saved_ckpts/ce_models"
    for run_id in inchikey_splits_runs:
        get_ckpt_from_wandb(save_dp, run_id)
        
    