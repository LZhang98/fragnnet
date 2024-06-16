
import argparse
import torch as th
from tqdm import tqdm

from frag_gnn.runner import load_config, init_dataloader
from frag_gnn.pl_model import FragGNNPL, NeimsPL, GNNPL
from frag_gnn.utils.misc_utils import to_device
from frag_gnn.utils.nn_utils import get_pl_hparams
from frag_gnn.dataset import SpecMolFragDataset, SpecMolDataset
from frag_gnn.utils.misc_utils import booltype, get_ckpt_from_wandb

import os
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

model_type_to_model_cls = {
	"neims": NeimsPL,
	"gnn": GNNPL,
	"frag_gnn": FragGNNPL,
}

model_type_ds_cls = {
	"neims": SpecMolDataset,
	"gnn": SpecMolDataset,
	"frag_gnn": SpecMolFragDataset,
}
  
def run_models(ckpt_fp,   
				 save_fp,
				 use_ckpt_config = True,
				 template_fp = None,
				 custom_fp = None,
				 device = None,
				 batch_size = 32,
				 ):

	# load configs
	print("> Loading configs")
	if use_ckpt_config:
		ckpt_hparams_d = get_pl_hparams(ckpt_fp)
		config_d= ckpt_hparams_d
	else:
		assert template_fp != None
		assert custom_fp != None
		config_d = load_config(template_fp, custom_fp)

	# set eval batch size
	config_d['eval_batch_size'] = batch_size
	print(f"> ckpt is compiled: {config_d['compile']}")
 	
	#config_d['eval_batch_size'] = batch_size
	# https://discuss.pytorch.org/t/pytorch-cannot-allocate-memory/134754
	#config_d['num_workers']= 0
	#config_d["compile"] = is_ckpt_compiled(ckpt_fp)
 
 	#config_d['num_workers']= 0
	#for key in config_d:
	#	if ckpt_hparams_d[key] != config_d[key]:
	#		print(key, ckpt_hparams_d[key],config_d[key])
	
	# load datasets
	print("> Loading datasets")
	model_type = config_d['model_type']
	model_cls = model_type_to_model_cls[model_type]
	model_ds_cls = model_type_ds_cls[model_type]
	print(f"> mol_fp {config_d['mol_fp']}")
	print(f"> spec_fp {config_d['spec_fp']}")
	print(f"> frag_dp {config_d['frag_dp']}")
	print(f"> split_dp {config_d['split_dp']}")

	ds = model_ds_cls(split='test', **config_d)
	print(ds.idx_to_inst_type)
	print(ds.idx_to_prec_type)
	dl = init_dataloader(ds, config_d)
 
	# load models
	print("> Loading models")

	if device is None:
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
	elif not th.cuda.is_available() and device != "cpu":
		device = "cpu"
   
	if device == "cpu":
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
		th.use_deterministic_algorithms(True)

	print(f"> using {device}")
	config_d['track_datapoint_metrics'] = False 
	config_d['auxiliary_scores'] = ['cos_hun','cos_sim']
	config_d['eval_mz_bin_res'] = [0.01, 0.1]
	config_d['frag_params']['preload'] = True
	config_d['frag_params']['preprocess'] = True

	model = model_cls.load_from_checkpoint(ckpt_fp, map_location=device, strict=False, **config_d)

	model.eval()
	# make predictions on validation data
	preds_l = []
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl),total=len(dl)):
			#print(batch_input)
			mol_smiles = batch_input['mol_smiles']
			spec_nces = batch_input['spec_nce']
			spec_prec_types = batch_input['spec_prec_type']
			spec_inst_types = batch_input['spec_inst_type']
			batch_result = to_device(batch_input, device)

			batch_result = model._common_step(batch_input, split="val")
			batch_size = batch_input['batch_size'].detach().cpu()

			#print(batch_result)
			#pred_mzs = batch_result["pred_mzs"].detach().cpu()
			#pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
			#pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()
			#pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()

			cos_sim_bin_lr = batch_result['cos_sim_0.1'].cpu()
			cos_sim_bin_hr = batch_result['cos_sim_0.01'].cpu()
			cos_sim_hun = batch_result['cos_hun'].cpu()
			#pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)
			for b_idx in range(batch_size):
				#b_peak_mask = pred_batch_idxs == b_idx
				# oos
				#b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy()
				b_smi = str(mol_smiles[b_idx])
				b_nce = float(spec_nces[b_idx])
				b_spec_prec_type_idx = int(spec_prec_types[b_idx])
				b_spec_inst_type_idx = int(spec_inst_types[b_idx])
				b_inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(b_smi))

				b_spec_prec_type = ds.idx_to_prec_type[b_spec_prec_type_idx]
				b_spec_inst_type = ds.idx_to_inst_type[b_spec_inst_type_idx] 
				preds_l.append((b_inchikey,b_smi,b_nce, b_spec_prec_type, b_spec_inst_type,
					cos_sim_bin_lr[b_idx].item(), cos_sim_bin_hr[b_idx].item(),cos_sim_hun[b_idx].item()))
				#plot_spectra_sparse()

	df = pd.DataFrame(preds_l, columns=['inchikey','smiles','nce',\
	 	'prec_type','inst_type','cos_sim_0.1','cos_sim_0.01', 'cos_sim_hun'])
 
	for metric in ['cos_sim_0.1','cos_sim_0.01','cos_sim_hun']:
		print(metric, df.loc[:, metric].mean(),  df.loc[:, metric].std())
	
	for agg_key in ['prec_type','inst_type']:
		for metric in ['cos_sim_0.1','cos_sim_0.01','cos_sim_hun']:
			#print(metric, df.loc[:, metric].mean(),  df.loc[:, metric].std())
			print(f"Aggregated by {agg_key}, {metric} \n", df.groupby(agg_key)[metric].agg(['mean','std','count']))

	print(f"> save csv to {save_fp}")
	df.to_csv(save_fp, index=False)


def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--template_fp",type=str,default="config/template.yml")
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_dp",type=str,default="./model_eval")
	parser.add_argument("--use_cahced_ckpt",type=booltype,default=False)
	parser.add_argument("--batch_size",type=int,default=32)
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")
	parser.add_argument("--wandb_run_ids",type=str, nargs='+', default= ['bzm1mv0p'])
 
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	os.makedirs(args.save_dp, exist_ok=True)	
	model_save_dp = args.model_save_dp
	print(f"> save model to {model_save_dp} from wandb")
	assert args.wandb_run_ids is not None

	for run_id in args.wandb_run_ids:
		#for run_id in inchikey_splits_runs:
		print(f"> wandb run id: {run_id}")
		ckpt_fp = get_ckpt_from_wandb(model_save_dp, run_id, use_cached = args.use_cahced_ckpt)

		#print(f"> {np_ckpt_fp}")
		save_fp = os.path.join(args.save_dp, run_id+'.csv')
		#print(f"> {custom_fp}")
		print(f"> {save_fp}")
		print(f"> {ckpt_fp}")
		run_models(ckpt_fp = ckpt_fp,   
					save_fp = save_fp,
					device = args.device,
					batch_size = args.batch_size)