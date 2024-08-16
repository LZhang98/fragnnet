import argparse
import torch as th
from tqdm import tqdm

from frag_gnn.runner import init_dataloader
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.utils.misc_utils import to_device
from frag_gnn.utils.nn_utils import get_pl_hparams, decompile_jit_ckpt
from frag_gnn.dataset import SpecMolFragDataset, SpecMolDataset
from frag_gnn.utils.misc_utils import booltype, get_ckpt_from_wandb
from frag_gnn.utils.ms2c_utils import load_model_and_init_config, run_prediction

import os
import multiprocessing
import numpy as np
import pandas as pd
from rdkit import Chem
import json
import time
	
def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_fp",type=str,default="./data/deepmet/neims_predicted/mol_df_1_predicted.pkl")
	parser.add_argument("--mol_df_fp",type=str,default="./data/deepmet/proc/deepmet_proc/mol_df_1.pkl.gz")
	parser.add_argument("--spec_df_fp",type=str,default="./data/deepmet/proc/deepmet_proc/spec_df_1.pkl.gz")
	parser.add_argument("--frag_dp", type=str,default="./data/deepmet/frags/mol_df_1")
	parser.add_argument("--use_cahced_ckpt",type=booltype,default=False)
	parser.add_argument("--batch_size",type=int,default=32)
 
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")
	parser.add_argument("--wandb_run_id",type=str, default= '0kxgjooa')
	# dest="model ckpt fp, when used, ignore wandb_run_ids"
	parser.add_argument("--model_ckpt",type=str, default=None)
	args = parser.parse_args()
	return args


if __name__ == "__main__":

	args = get_args()
	timestr = time.strftime("%Y%m%d-%H%M%S")
	save_dp = os.path.dirname(args.save_fp)
	os.makedirs(save_dp, exist_ok=True)
	json.dump(vars(args), open(f"{save_dp}/args_{timestr}.json", "w"), indent=2)

	if args.model_ckpt is None:
		os.makedirs(args.model_save_dp, exist_ok=True)	
		model_save_dp = args.model_save_dp
		print(f">> save model to {model_save_dp} from wandb")
		#assert args.wandb_run_id is not None
		print(f">> wandb run id: {args.wandb_run_id}")
		ckpt_fp = get_ckpt_from_wandb(model_save_dp, args.wandb_run_id, use_cached = args.use_cahced_ckpt)
	else:
		ckpt_fp = args.model_ckpt
	
	model, config_d, device = load_model_and_init_config(ckpt_fp = ckpt_fp, device = args.device, batch_size = args.batch_size)
	df = run_prediction(model = model,   
				config_d = config_d,
				mol_df_fp = args.mol_df_fp,
				spec_df_fp = args.spec_df_fp,
				frag_dp = args.frag_dp,
				device = args.device)
	df.to_pickle(args.save_fp)