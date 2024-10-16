from frag_gnn.utils.misc_utils import get_best_ckpt_from_wandb, get_wandb_runs_by_grp
	
if __name__ == "__main__":

	save_dp = "./saved_ckpts/"
	fragnnet_d3_runs = get_wandb_runs_by_grp("nist20v3_inchikey_fraggnn_d3")
	print(fragnnet_d3_runs)
	