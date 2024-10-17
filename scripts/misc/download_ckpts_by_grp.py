from frag_gnn.utils.misc_utils import get_best_ckpt_from_wandb, get_wandb_runs_by_grp
	
if __name__ == "__main__":

	save_dp = "./saved_ckpts/"
	iceberg_inchikey_gen_runs = get_wandb_runs_by_grp("nist20v3_inchikey_iceberg_gen")
	#print(fragnnet_d3_runs)
	for run in iceberg_inchikey_gen_runs:
		print(run.id, run.name)
	
	iceberg_inchikey_inten_runs = get_wandb_runs_by_grp("nist20v3_inchikey_iceberg_inten_100")
