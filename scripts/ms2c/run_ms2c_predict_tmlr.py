import argparse
from tqdm import tqdm

from frag_gnn.utils.misc_utils import booltype, get_ckpt_from_wandb
from frag_gnn.utils.ms2c_utils import load_model_and_init_config, run_prediction

import os
import pandas as pd
import json
import time


def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_dp", type=str,default="./ms2c_debug", help="")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50", help="")
	parser.add_argument("--frag_dp", type=str,default="./data/ms2c/pubchem/frags/d4_h4_isoFalse/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50", help="frag dp only used in fragnnet")

	parser.add_argument("--use_cahced_ckpt",type=booltype,default=False)
	parser.add_argument("--batch_size",type=int,default=32)
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")
	parser.add_argument("--wandb_run_id",type=str, default= '9gnx820u')
	# dest="model ckpt fp, when used, ignore wandb_run_ids"
	parser.add_argument("--model_ckpt",type=str, default=None)
	parser.add_argument("--start_row_idx", type=int, default=0)
	parser.add_argument("--num_row", type=int, default=0)
	#parser.add_argument("--group_by_target", type=booltype, default=True)
	args = parser.parse_args()
	return args


if __name__ == "__main__":

	args = get_args()
	timestr = time.strftime("%Y%m%d-%H%M%S")
	#save_dp = os.path.dirname(args.save_fp)
	os.makedirs(args.save_dp, exist_ok=True)
	json.dump(vars(args), open(f"{args.save_dp}/args_{timestr}.json", "w"), indent=2)

	if args.model_ckpt is None:
		os.makedirs(args.model_save_dp, exist_ok=True)	
		model_save_dp = args.model_save_dp
		print(f">> save model to {model_save_dp} from wandb")
		#assert args.wandb_run_id is not None
		print(f">> wandb run id: {args.wandb_run_id}")
		ckpt_fp = get_ckpt_from_wandb(model_save_dp, args.wandb_run_id, use_cached = args.use_cahced_ckpt)
	else:
		ckpt_fp = args.model_ckpt
  
	# preprocess load datasets
	print(f"> Loading datasets {args.proc_dp}")

	print("> preload mol_df")
	mol_df = pd.read_pickle(os.path.join(args.proc_dp,'mol_df.pkl.gz'))

	print("> preload spec_fp")
	spec_df = pd.read_pickle(os.path.join(args.proc_dp,'spec_df.pkl.gz'))

	print(f"> Loading models {ckpt_fp}")
	model, config_d, device = load_model_and_init_config(ckpt_fp = ckpt_fp, device = args.device, batch_size = args.batch_size)
 
	#if args.group_by_target:
	target_ids = spec_df['target_mol_id'].unique()
	if len(target_ids) < args.start_row_idx:
		exit()
	elif args.num_row > 0:
		msg_id_map_df = target_ids.loc[args.start_row_idx:args.start_row_idx+args.num_row]
	else:
		msg_id_map_df = target_ids.loc[args.start_row_idx:]

	for target_id in tqdm(target_ids):
		save_fp = os.path.join(args.save_dp, f"{target_id}.pkl.gz")

		if os.path.isfile(save_fp) and args.allow_cached:
			continue

		spec_df_grp = spec_df[ spec_df['target_mol_id'] == target_id]
		mol_df_grp = mol_df[ mol_df['mol_id'].isin(spec_df_grp['mol_id'])]
		
		df = run_prediction(model = model,   
					config_d = config_d,
					mol_df_fp = mol_df_grp,
					spec_df_fp = spec_df_grp,
					frag_dp = args.frag_dp,
					device = device)

		df.to_pickle(save_fp)
