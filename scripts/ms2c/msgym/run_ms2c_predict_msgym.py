import argparse
from tqdm import tqdm

from frag_gnn.utils.misc_utils import booltype, get_best_ckpt_from_wandb
from frag_gnn.utils.ms2c_utils import load_model_and_init_config, run_prediction

import os
import pandas as pd
import json
import time
import tempfile
import zipfile


def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_dp",type=str,default="./data/ms2c/msgym/fragnnet_predicted/retrieval_candidates_formula")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/msgym/proc/retrieval_candidates_formula")
	parser.add_argument("--frag_dp", type=str,default="./data/ms2c/msgym/frag/retrieval_candidates_formula_d3_h4")
 
	parser.add_argument("--use_cahced_ckpt",type=booltype,default=False)
	parser.add_argument("--batch_size",type=int,default=32)
	parser.add_argument("--zipped_frag",type=booltype,default=False)
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")
	parser.add_argument("--wandb_run_id",type=str, default= '9gnx820u')
	# dest="model ckpt fp, when used, ignore wandb_run_ids"
	parser.add_argument("--model_ckpt",type=str, default=None)
	parser.add_argument("--start_row_idx", type=int, default=0)
	parser.add_argument("--num_row", type=int, default=0)
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
		ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, args.wandb_run_id, use_cached = args.use_cahced_ckpt)
	else:
		ckpt_fp = args.model_ckpt
  
	# parser.add_argument("--msg_id_map", type=str,default="data/ms2c/msgym/proc/retrieval_candidates_formula/msg_id_map.csv")
	msg_id_map_fp = os.path.join(args.proc_dp,"msg_id_map.csv")
	msg_id_map_df = pd.read_csv(msg_id_map_fp)

	if len(msg_id_map_df) < args.start_row_idx:
		exit()
	elif args.num_row > 0:
		msg_id_map_df = msg_id_map_df.loc[args.start_row_idx:args.start_row_idx+args.num_row]
	else:
		msg_id_map_df = msg_id_map_df.loc[args.start_row_idx:]
  
	model, config_d, device = load_model_and_init_config(ckpt_fp = ckpt_fp, device = args.device, batch_size = args.batch_size)
	for idx,row in tqdm(msg_id_map_df.iterrows(), total=msg_id_map_df.shape[0]):

		mol_df_filename = row["mol_df"]
		spec_filename = row["spec_df"]

		mol_df_fp = os.path.join(args.proc_dp,mol_df_filename)
		spec_df_fp = os.path.join(args.proc_dp,spec_filename)
		save_fp = os.path.join(args.save_dp,spec_filename)
  
		if not os.path.isfile(mol_df_fp):
			continue
		if not os.path.isfile(spec_df_fp):
			continue

		if not args.zipped_frag:
			frag_dp = os.path.join(args.frag_dp, mol_df_filename[:-4])
			df = run_prediction(model = model,   
						config_d = config_d,
						mol_df_fp = mol_df_fp,
						spec_df_fp = spec_df_fp,
						frag_dp = frag_dp,
						device = device)
			df.to_pickle(save_fp)

		else:
			frag_zip_fp = os.path.join(args.frag_dp, mol_df_filename.replace(".pkl",".zip"))
			with tempfile.TemporaryDirectory() as tmp_dp, zipfile.ZipFile(frag_zip_fp, 'r') as zip_file:
				zip_file.extractall(tmp_dp)
					
				df = run_prediction(model = model,   
							config_d = config_d,
							mol_df_fp = mol_df_fp,
							spec_df_fp = spec_df_fp,
							frag_dp = tmp_dp,
							device = device)
				df.to_pickle(save_fp)
