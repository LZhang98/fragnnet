
import argparse
import torch as th
from tqdm import tqdm

from frag_gnn.runner import load_config, init_dataloader
from frag_gnn.pl_model import FragGNNPL, NeimsPL, GNNPL
from frag_gnn.utils.misc_utils import to_device
from frag_gnn.dataset import SpecMolFragDataset, SpecMolDataset
from frag_gnn.utils.misc_utils import booltype, get_best_ckpt_from_wandb

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
				 template_fp,
				 custom_fp,
				 device = None,
				 batch_size = 32,
				 ):


	# load configs
	print("> Loading configs")
	config_d = load_config(template_fp, custom_fp)

	# load datasets
	print("> Loading datasets")
	config_d['eval_batch_size'] = batch_size
	# https://discuss.pytorch.org/t/pytorch-cannot-allocate-memory/134754
	config_d['num_workers']= 0
	print(f"> ckpt is compiled: {config_d['compile']}")
 
	model_type = config_d['model_type']
	model_cls = model_type_to_model_cls[model_type]
	model_ds_cls = model_type_ds_cls[model_type]
 
	ds = model_ds_cls(split='test', **config_d)
	dl = init_dataloader(ds, config_d)
	# load models
	print("> Loading models")

	print(f"> {save_fp}")
	if device is None:
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
	elif not th.cuda.is_available() and device != "cpu":
		device = "cpu"
   
	if device == "cpu":
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
		th.use_deterministic_algorithms(True)

	print(f"> using {device}")
	#print(device)
	config_d['track_datapoint_metrics'] = False 
	config_d['auxiliary_scores'] = ['cos_hun','cos_sim']
	config_d['eval_mz_bin_res'] = [0.01, 0.1]
 
	model = model_cls.load_from_checkpoint(ckpt_fp, map_location=device, strict=False, **config_d)

	model.eval()
	# make predictions on validation data
	# print("Iterating over data")
	preds_l = []
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl),total=len(dl)):
			#print(batch_input)
			mol_smiles = batch_input['mol_smiles']
			spec_nces = batch_input['spec_ce']
   
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
			#cos_sim_hun = batch_result['cos_hun'].cpu()
			#pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)
			for b_idx in range(batch_size):
				#b_peak_mask = pred_batch_idxs == b_idx
				# oos
				#b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy()
    
				b_smi = str(mol_smiles[b_idx])
				b_nce = float(spec_nces[b_idx])
				b_inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(b_smi))
				preds_l.append((b_inchikey,b_smi,b_nce,cos_sim_bin_lr[b_idx].item(), cos_sim_bin_hr[b_idx].item()))
				#plot_spectra_sparse()

	df = pd.DataFrame(preds_l, columns=['inchikey','smiles','nce','cos_sim_0.1','cos_sim_0.01'])
	print('cos_sim_0.1', df.loc[:, 'cos_sim_0.1'].mean())
	print('cos_sim_0.01', df.loc[:, 'cos_sim_0.01'].mean())
	df.to_csv(save_fp, index=False)


def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--template_fp",type=str,default="config/template.yml")
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_dp",type=str,default="./model_eval")
	parser.add_argument("--use_cahced_ckpt",type=booltype,default=False)
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
 

	os.makedirs(args.save_dp, exist_ok=True)	
	model_save_dp = "./saved_ckpts/ms_gym_benchmark"

	#model_id_configs = [
	#	['lvsdbn5y','config/ms_gym_benchmark/d3_um_inst_ms_gym_benchmark_0.01.yml'],
	#	['hpnwdaoq','config/msg/msg_fp_0,1.yml'],
	#	['ajgc9p5x','config/msg/msg_fp_0,01.yml'],
	#	['9zxgadu4','config/msg/msg_gnn_0,1.yml'],
	#	['ht1o1wrs','config/msg/msg_gnn_0,01.yml'],
	#	['zka7gid2','config/msg/msg_fragnnet_0,1.yml'],
	#	['vclvoh0t ','config/msg/msg_fragnnet_0,01.yml'],
	#]
	model_id_configs = [
		['9gnx820u','config/msg/msg_fragnnet_0,01v2.yml']
	]
	batch_size = 32
	for run_id,custom_fp in model_id_configs:
		#for run_id in inchikey_splits_runs:
		print(f"> wandb run id: {run_id}")
		ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, run_id, use_cached = args.use_cahced_ckpt)
		
		#print(f"> {np_ckpt_fp}")
		save_fp = os.path.join(args.save_dp, run_id+'.csv')
		print(f"> {custom_fp}")
		print(f"> {save_fp}")
		print(f"> {ckpt_fp}")
		run_models(ckpt_fp = ckpt_fp,   
					save_fp = save_fp,
					template_fp = args.template_fp,
					custom_fp = custom_fp,
					device = args.device,
					batch_size = batch_size)