from frag_gnn.utils.misc_utils import get_ckpt_from_wandb
    
if __name__ == "__main__":

    inchikey_splits_runs = [
        'f5w6bmjk', #nist20v2_d4_wr00_inchikey/fraggnn_d3/s0
        '8k7a121f', #nist20v2_d4_wr00_inchikey/fraggnn_d3/s1
        'e23qg6ua', #nist20v2_d4_wr00_inchikey/fraggnn_d3/s2
        'xdu3dui9', #nist20v2_d4_wr00_inchikey/fraggnn_d3/s3
        'kumfndkv', #nist20v2_d4_wr00_inchikey/fraggnn_d3/s4
        'vjrv1nzz', #nist20v2_d4_wr00_inchikey/fraggnn_d4/s0
        'dkztdob9', #nist20v2_d4_wr00_inchikey/fraggnn_d4/s1
        'rgw9m0jr', #nist20v2_d4_wr00_inchikey/fraggnn_d4/s2
        '1gs2g86m', #nist20v2_d4_wr00_inchikey/fraggnn_d4/s3
        'pxvc5qr4', #nist20v2_d4_wr00_inchikey/fraggnn_d4/s4
    ]
    
    save_dp = "./saved_ckpts/neurips_inchikey"
    for run_id in inchikey_splits_runs:
        get_ckpt_from_wandb(save_dp, run_id)

