
import argparse
import torch as th
from tqdm import tqdm

from frag_gnn.runner import init_dataloader
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.utils.misc_utils import to_device
from frag_gnn.utils.nn_utils import get_pl_hparams
from frag_gnn.dataset import SpecMolFragDataset, SpecMolDataset
from frag_gnn.utils.misc_utils import booltype, get_ckpt_from_wandb

import os
import multiprocessing
import numpy as np
import pandas as pd
from rdkit import Chem
import json
import time

model_type_to_model_cls = {
	"neims": NeimsPL,
	"frag_gnn": FragGNNPL,
}

model_type_ds_cls = {
	"neims": SpecMolDataset,
	"frag_gnn": SpecMolFragDataset,
}

def get_all_frags_name(frag_dp):
	return set([f.split('.')[0] for f in os.listdir(frag_dp)])
	
def run_prediction(ckpt_fp,   
				 save_fp,
				 mol_df_fp,
				 spec_df_fp,
				 frag_dp,
				 device = None,
				 batch_size = 32,
				 ):

	# load configs
	print(">> Loading configs")
	ckpt_hparams_d = get_pl_hparams(ckpt_fp)
	config_d= ckpt_hparams_d

	# set eval batch size
	config_d['eval_batch_size'] = batch_size
	print(f"> ckpt is compiled: {config_d['compile']}")
 	
	print(f"> using {device}")
	# disable metrics
	config_d['track_datapoint_metrics'] = False 
	config_d['auxiliary_scores'] = []  #['cos_hun','cos_sim']
	#config_d['eval_mz_bin_res'] = [0.01, 0.1]
	# let us see if ram will be an issue
	config_d['frag_params']['preload'] = True
	config_d['frag_params']['preprocess'] = False
	# disable sampler
	config_d["dynamic_batch_sampler"] = False
	config_d["group_sampler"] = False 
	config_d["simple_group_sampler"] = False
	# additional spec_params
	config_d['spec_params']['unique_id'] = True
	config_d['spec_params']["counts"] = False
 
	# config proc and check if frags are avaibale
	print(f"> mol_fp {mol_df_fp}")
	print(f"> spec_fp {spec_df_fp}")
	print(f"> frag_dp {config_d['frag_dp']}")

	print(">> filtering input without frags")
	all_frags = get_all_frags_name(frag_dp)
	mol_df  = pd.read_pickle(mol_df_fp)
	mol_df = mol_df[mol_df['mol_id'].isin(all_frags)]
	spec_df = pd.read_pickle(spec_df_fp)
	spec_df = spec_df[spec_df['mol_id'].isin(mol_df['mol_id'].to_list())]
	
	# config mol df, spec df and frag_dp
	config_d['mol_fp'] = mol_df
	config_d['spec_fp'] = spec_df
	config_d['frag_dp'] = frag_dp
 
	# load datasets
	print(">> Loading datasets")
	model_type = config_d['model_type']
	model_cls = model_type_to_model_cls[model_type]
	model_ds_cls = model_type_ds_cls[model_type]
	ds = model_ds_cls(split='predict_only', **config_d)
	print(f"> model supported inst type {ds.idx_to_inst_type}")
	print(f"> model supported prec type {ds.idx_to_prec_type}")
	print(">> init dataloader")
	dl = init_dataloader(ds, config_d)
 
	# config cuda and cpus
	if device is None:
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
	elif not th.cuda.is_available() and device != "cpu":
		device = "cpu"
   
	if device == "cpu":
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
		th.use_deterministic_algorithms(True)
	config_d['num_workers']= multiprocessing.cpu_count()
	
 	# load models
	print(">> Loading models")
	model = model_cls.load_from_checkpoint(ckpt_fp, map_location=device, strict=False, **config_d)
	model.eval()


	# make predictions on validation data
	print(">> Running inference")
	preds_l = []
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl),total=len(dl)):
			#print(batch_input)
			mol_smiles = batch_input['mol_smiles']
			spec_nces = batch_input['spec_nce']
			spec_ids = batch_input['spec_unique_id']
			mol_ids = batch_input['mol_id']
			spec_prec_types = batch_input['spec_prec_type']
			spec_inst_types = batch_input['spec_inst_type']
			batch_result = to_device(batch_input, device)

			batch_result = model.predict_step(**batch_input)
			batch_size = batch_input['batch_size'].detach().cpu()

			pred_mzs = batch_result["pred_mzs"].detach().cpu()
			pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()
			if "pred_oos_logprobs" in batch_result:
				pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()
	
			pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
			pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)
			for b_idx in range(batch_size):
				b_peak_mask = pred_batch_idxs == b_idx
				b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy()
				b_spec_id = spec_ids[b_idx]
				b_smi = str(mol_smiles[b_idx])
				b_nce = float(spec_nces[b_idx])
				b_mol_id = mol_ids[b_idx]
				b_spec_prec_type_idx = int(spec_prec_types[b_idx])
				b_spec_inst_type_idx = int(spec_inst_types[b_idx])
				#b_inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(b_smi))

				b_spec_prec_type = ds.idx_to_prec_type[b_spec_prec_type_idx]
				b_spec_inst_type = ds.idx_to_inst_type[b_spec_inst_type_idx]
				b_spec_mzs = pred_mzs[b_peak_mask]
				b_spec_ints = pred_ints[b_peak_mask]

				preds_l.append((b_spec_id.item(), b_mol_id, b_spec_mzs.tolist(),b_spec_ints.tolist(), b_pred_oos_logprob.item()))

	columns = ['spec_id','mol_id','pred_mzs','pred_ints','pred_oos']
	df = pd.DataFrame(preds_l, columns=columns)
	print(f">> save to {save_fp}")
	df.to_pickle(save_fp)
	del df
		
def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_fp",type=str,default="./data/deepmet/predicted/mol_df_1_predicted.pkl")
	parser.add_argument("--mol_df_fp",type=str,default="./data/deepmet/proc/mol_df_1.pkl.gz")
	parser.add_argument("--spec_df_fp",type=str,default="./data/deepmet/proc/spec_df_1.pkl.gz")
	parser.add_argument("--frag_dp", type=str,default="./data/deepmet/frags/mol_df_1")
	parser.add_argument("--use_cahced_ckpt",type=booltype,default=False)
	parser.add_argument("--batch_size",type=int,default=32)
 
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")
	parser.add_argument("--wandb_run_id",type=str, default= '0kxgjooa')
	args = parser.parse_args()
	return args


if __name__ == "__main__":

	args = get_args()
	timestr = time.strftime("%Y%m%d-%H%M%S")
	save_dp = os.path.dirname(args.save_fp)
	os.makedirs(save_dp, exist_ok=True)
	json.dump(vars(args), open(f"{save_dp}/args_{timestr}.json", "w"), indent=2)

	os.makedirs(args.model_save_dp, exist_ok=True)	
	model_save_dp = args.model_save_dp
	print(f">> save model to {model_save_dp} from wandb")
	assert args.wandb_run_id is not None

	print(f">> wandb run id: {args.wandb_run_id}")
	ckpt_fp = get_ckpt_from_wandb(model_save_dp, args.wandb_run_id, use_cached = args.use_cahced_ckpt)

	run_prediction(ckpt_fp = ckpt_fp,   
				save_fp = args.save_fp,
				mol_df_fp = args.mol_df_fp,
				spec_df_fp = args.spec_df_fp,
				frag_dp = args.frag_dp,
				device = args.device,
				batch_size = args.batch_size)