from frag_gnn.utils.misc_utils import get_ckpt_from_wandb




import argparse
import torch as th
from tqdm import tqdm

from frag_gnn.runner import load_config, init_dataloader
from frag_gnn.pl_model import FragGNNPL
from frag_gnn.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
from frag_gnn.utils.plot_utils import plot_spectra_sparse
from frag_gnn.utils.misc_utils import to_device
from frag_gnn.utils.data_utils import mol_to_inchikey
from frag_gnn.dataset import SpecMolFragDataset, , SpecMolDataset


import os
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

def run_frag_gnn(ckpt_fp,   
				 save_fp,
				 template_fp,
				 custom_fp,
				 device = None,
				 batch_size = 1,
				 ):

	# load configs
	print("> Loading configs")
	config_d = load_config(template_fp, custom_fp)

	# load datasets
	print("> Loading datasets")
	config_d['eval_batch_size'] = batch_size
	# https://discuss.pytorch.org/t/pytorch-cannot-allocate-memory/134754
	config_d['num_workers']= 0

	ds = SpecMolFragDataset(split='test', **config_d)
	dl = init_dataloader(ds, config_d)
	# load models
	print(">  Loading models")

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
	model = FragGNNPL.load_from_checkpoint(ckpt_fp, map_location=device, strict=False, **config_d)

	model.eval()
	# make predictions on validation data
	# print("Iterating over data")
	preds_l = []
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl),total=len(dl)):
			#print(batch_input)
			mol_smiles = batch_input['mol_smiles']
			spec_nces = batch_input['spec_nce']
   
			batch_result = to_device(batch_input, device)

			batch_result = model._common_step(batch_input,  split="val", no_log=True)
			batch_size = batch_input['batch_size'].detach().cpu()
			#print(batch_result)

			pred_mzs = batch_result["pred_mzs"].detach().cpu()
			pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
			pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()
			pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()
			cos_sim_bin = batch_result['cos_sim_0.01'].cpu()
			cos_sim_hun = batch_result['cos_hun'].cpu()
			#print(pred_mzs)
			#print(batch_result["pred_logprobs"])

			for b_idx in range(batch_size):
				b_peak_mask = pred_batch_idxs == b_idx
				# oos
				b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy()

				b_smi = str(mol_smiles[b_idx])
				b_nce = float(spec_nces[b_idx])
				b_inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(b_smi))
				preds_l.append((b_inchikey,b_smi,b_nce,cos_sim_bin[b_idx].item(), cos_sim_hun[b_idx].item()))

	df = pd.DataFrame(preds_l, columns=['inchikey','smiles','nce','cos_sim_bin','cos_sim_hun'])
	print('cos_sim_bin', df.loc[:, 'cos_sim_bin'].mean())
	print('cos_sim_hun', df.loc[:, 'cos_sim_hun'].mean())
	df.to_csv(save_fp, index=False)


def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--template_fp",type=str,default="config/template.yml")
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_dp",type=str,default="./model_eval")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	#save_dp = os.path.dirname(os.path.abspath(args.save_fp))
	os.makedirs(args.save_dp, exist_ok=True)	
	model_save_dp = "./saved_ckpts/ce_models"

	model_id_configs = [
		#['0jeuscvs','config/nist20_d4_wr00_inchikey_ce/fraggnn_d3_ce_lin.yml',32],
		['bzh7mub7','config/nist20_d4_wr00_inchikey_ce/fraggnn_d4_ce_lin.yml',4]
	]
	
	for run_id,custom_fp,batch_size in model_id_configs:
		#for run_id in inchikey_splits_runs:
		print(f"> {run_id}")
		ckpt_fp = get_ckpt_from_wandb(model_save_dp, run_id)
		print(f"> {ckpt_fp}")
		save_fp = os.path.join(args.save_dp, run_id+'.csv')
		print(f"> {custom_fp}")
		print(f"> {save_fp}")
		run_frag_gnn(ckpt_fp = ckpt_fp,   
					save_fp = save_fp,
					template_fp = args.template_fp,
					custom_fp = custom_fp,
					device = args.device,
					batch_size = 32)
