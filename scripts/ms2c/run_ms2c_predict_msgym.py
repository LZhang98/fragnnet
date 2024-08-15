import argparse
import torch as th
from tqdm import tqdm

from frag_gnn.runner import init_dataloader
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.utils.misc_utils import to_device
from frag_gnn.utils.nn_utils import get_pl_hparams, decompile_jit_ckpt
from frag_gnn.dataset import SpecMolFragDataset, SpecMolDataset
from frag_gnn.utils.misc_utils import booltype, get_ckpt_from_wandb

import os
import multiprocessing
import numpy as np
import pandas as pd
from rdkit import Chem
import json
import time
import tempfile
import zipfile

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

def load_model_and_init_config(ckpt_fp, device = None, batch_size = 32):
	
	# load ckpt
	ckpt = th.load(ckpt_fp, map_location=device)
	
	print(">> Loading configs")
	config_d = get_pl_hparams(ckpt)

	# set eval batch size
	config_d['eval_batch_size'] = batch_size
	# disable metrics
	config_d['track_datapoint_metrics'] = False 
	config_d['auxiliary_scores'] = []  #['cos_hun','cos_sim']
	#config_d['eval_mz_bin_res'] = [0.01, 0.1]
	# let us see if ram will be an issue
	config_d['frag_params']['preload'] = False
	config_d['frag_params']['preprocess'] = False
	# disable sampler
	config_d["dynamic_batch_sampler"] = False
	config_d["group_sampler"] = False 
	config_d["simple_group_sampler"] = False
	# additional spec_params
	config_d['spec_params']['unique_id'] = True
	config_d['spec_params']["counts"] = False
	# set num works and cpu stuff
	config_d['num_workers']= multiprocessing.cpu_count()
	if device == "cpu":
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
		th.use_deterministic_algorithms(True)
  
	# check class
	model_type = config_d['model_type']
	model_cls = model_type_to_model_cls[model_type]
	
   	# config cuda and cpus
	if device is None:
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
	elif not th.cuda.is_available() and device != "cpu":
		device = "cpu"
	print(f"> using {device}")
	# load configs
 
 	# load models
	print(">> Loading models")
	print(f"> ckpt was compiled: {config_d['compile']}")
	if config_d['compile']:
		ckpt = decompile_jit_ckpt(ckpt)
		config_d['compile'] = False

	model = model_cls(**config_d)
	model.load_state_dict(ckpt["state_dict"])
	model.to(device)
	#model.eval()
	return model, config_d, device
 
def run_prediction(model,
                 config_d,
				 mol_df_fp,
				 spec_df_fp,
				 frag_dp,
				 save_fp,
				 device
				 ):


	# config proc and check if frags are avaibale
	print(f"> mol_fp {mol_df_fp}")
	print(f"> spec_fp {spec_df_fp}")
	print(f"> frag_dp {frag_dp}")
	print(f"> save to {save_fp}")
 

	# check model class and ds class
	#model_cls = model_type_to_model_cls[model_type]
	# model_ds_cls = None #model_type_ds_cls[model_type]
	model_ds_cls = None
 
	mol_df  = pd.read_pickle(mol_df_fp)
	#print("> num of mols", len(mol_df))
	# only check for FragGNNet
	if isinstance(model,FragGNNPL):
		#print(">> filtering input without frags")
		all_frags = get_all_frags_name(frag_dp)
		#print("> num of frags: ", len(all_frags))
		mol_df = mol_df[mol_df['mol_id'].astype(str).isin(all_frags)]
		#print("> num of mols after filtering", len(mol_df))
		model_ds_cls = SpecMolFragDataset
  
	spec_df = pd.read_pickle(spec_df_fp)
	#print("> num of spec", len(spec_df))
	if  isinstance(model,NeimsPL):
		mz_max = config_d['mz_max']
		spec_df = spec_df[spec_df['prec_mz'] <= mz_max]
		#print("> [NeimsPL] num of spec after prec_mz filtering", len(spec_df))
		model_ds_cls = SpecMolDataset
	
	spec_df = spec_df[spec_df['mol_id'].isin(mol_df['mol_id'].to_list())]
	print("> num of spec after filtering", len(spec_df))
	
	# config mol df, spec df and frag_dp
	config_d['mol_fp'] = mol_df
	config_d['spec_fp'] = spec_df
	config_d['frag_dp'] = frag_dp
 
	# load datasets
	print(">> Loading datasets")
	ds = model_ds_cls(split='predict_only', **config_d)
	#print(f"> model supported inst type {ds.idx_to_inst_type}")
	#print(f"> model supported prec type {ds.idx_to_prec_type}")
	print(">> init dataloader")
	dl = init_dataloader(ds, config_d)
 
 	#model = model_cls.load_from_checkpoint(ckpt_fp, map_location=device, strict=False, **config_d)
	model.eval()

	# make predictions on validation data
	print(">> Running inference")
	preds_l = []
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl),total=len(dl)):
			#print(batch_input)
			mol_smiles = batch_input['mol_smiles']
			spec_ids = batch_input['spec_unique_id']
			mol_ids = batch_input['mol_id']
			spec_prec_types = batch_input['spec_prec_type']
			spec_inst_types = batch_input['spec_inst_type']
			batch_result = to_device(batch_input, device)

			batch_result = model.predict_step(**batch_input)
			batch_size = batch_input['batch_size'].detach().cpu()

			pred_mzs = batch_result["pred_mzs"].detach().cpu()
			pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()
			
			has_oos_b = "pred_oos_logprobs" in batch_result
			if has_oos_b:
				pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()
	
			pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
			pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)
			for b_idx in range(batch_size):
				b_peak_mask = pred_batch_idxs == b_idx
	
				b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy().item() if has_oos_b else np.nan
				b_spec_id = spec_ids[b_idx]
				b_smi = str(mol_smiles[b_idx])
				b_mol_id = mol_ids[b_idx]
				b_spec_prec_type_idx = int(spec_prec_types[b_idx])
				b_spec_inst_type_idx = int(spec_inst_types[b_idx])
				#b_inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(b_smi))

				b_spec_prec_type = ds.idx_to_prec_type[b_spec_prec_type_idx]
				b_spec_inst_type = ds.idx_to_inst_type[b_spec_inst_type_idx]
				b_spec_mzs = pred_mzs[b_peak_mask]
				b_spec_ints = pred_ints[b_peak_mask]

				preds_l.append((b_spec_id.item(), b_mol_id, b_spec_mzs.tolist(),b_spec_ints.tolist(), b_pred_oos_logprob))

	columns = ['spec_id','mol_id','pred_mzs','pred_ints','pred_oos']
	df = pd.DataFrame(preds_l, columns=columns)
	df.to_pickle(save_fp)
	del df
		
def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_dp",type=str,default="./data/msgym/fragnnet_predicted/retrieval_candidates_mass")
	parser.add_argument("--proc_dp", type=str,default="data/ms2c/msgym/proc/retrieval_candidates_mass")

	parser.add_argument("--frag_dp", type=str,default="./data/ms2c/msgym/frag/retrieval_candidates_mass_d3_h4")
	parser.add_argument("--use_cahced_ckpt",type=booltype,default=False)
	parser.add_argument("--batch_size",type=int,default=32)
	parser.add_argument("--zipped_frag",type=booltype,default=True)
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")
	parser.add_argument("--wandb_run_id",type=str, default= '9gnx820u')
	# dest="model ckpt fp, when used, ignore wandb_run_ids"
	parser.add_argument("--model_ckpt",type=str, default=None)
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
  
	# parser.add_argument("--msg_id_map", type=str,default="data/ms2c/msgym/proc/retrieval_candidates_formula/msg_id_map.csv")
	msg_id_map_fp = os.path.join(args.proc_dp,"msg_id_map.csv")
	msg_id_map_df = pd.read_csv(msg_id_map_fp)
	#print(df.head())

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
			run_prediction(model = model,   
						config_d = config_d,
						mol_df_fp = mol_df_fp,
						spec_df_fp = spec_df_fp,
						frag_dp = frag_dp,
						save_fp = save_fp,
						device = device)
		else:
			frag_zip_fp = os.path.join(args.frag_dp, mol_df_filename.replace(".pkl",".zip"))
			with tempfile.TemporaryDirectory() as tmp_dp, zipfile.ZipFile(frag_zip_fp, 'r') as zip_file:
				zip_file.extractall(tmp_dp)
					
				run_prediction(model = model,   
							config_d = config_d,
							mol_df_fp = mol_df_fp,
							spec_df_fp = spec_df_fp,
							frag_dp = tmp_dp,
							save_fp = save_fp,
							device = device)