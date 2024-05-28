import argparse
import json
import torch as th
from tqdm import tqdm

from frag_gnn.runner import load_config
from frag_gnn.massformer.pl_model import MassFormerPL
from frag_gnn.dataset import SpecMolDataset, SpecMolFragDataset, init_dataloader

from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.massformer.pl_model import MassFormerPL
from frag_gnn.utils.misc_utils import to_device, booltype
from frag_gnn.dataset import SpecMolDataset, init_dataloader

import os
import multiprocessing
import pandas as pd

from tqdm import tqdm
import numpy as np

def run_prediction(ckpt_fp,   
				 save_dp,
				 template_fp,
				 custom_fp,
				 task_dp,
				 model_cls,
				 model_ds_cls,
				 frag_dp = None,
				 device = None,
				 batch_size = 1,
				 allow_cached = False,
				 disable_jit = False
				 ):

	# load configs
	print("> Loading configs")
	config_d = load_config(template_fp, custom_fp)
	config_d['split_dp'] = task_dp

	#config_d['spec_params']['prec_mz'] = False
	config_d['spec_params']['unique_id'] = False
	config_d['spec_params']["counts"] = False
	config_d['eval_batch_size'] = batch_size

	if frag_dp is not None:
		config_d['frag_dp'] = frag_dp
		print(f"> set frag_dp {frag_dp}")
	
	if disable_jit:
		print(f"> set compile to False")
		config_d['compile'] = False
  
	# load datasets
	print(f"> Loading datasets {task_dp}")

	print("> preload mol_df")
	mol_df = pd.read_pickle(os.path.join(task_dp,'mol_df.pkl.gz'))

	print("> preload spec_fp")
	spec_df = pd.read_pickle(os.path.join(task_dp,'spec_df.pkl.gz'))
	print(f"> Loading models {ckpt_fp}")

	#target_ids = spec_df['target_mol_id'].unique()

	print("> load model")
	model = model_cls.load_from_checkpoint(ckpt_fp, map_location=device, strict=False, **config_d)
	model.eval()
	model.freeze()
  
	for target_id in tqdm(target_ids):

		save_fp = os.path.join(save_dp, f"{target_id}.pkl.gz")

		if os.path.isfile(save_fp) and allow_cached:
			continue

		spec_df_grp = spec_df[ spec_df['target_mol_id'] == target_id]
		mol_df_grp = mol_df[ mol_df['mol_id'].isin(spec_df_grp['mol_id'])]

		cannot_df = None
		if isinstance(model, FragGNNPL):
			has_frags = []
			for mol_id in mol_df_grp['mol_id'].tolist():
				if os.path.isfile(os.path.join(config_d['frag_dp'],f'{mol_id}.pickle.bz2')) or \
					os.path.isfile(os.path.join(config_d['frag_dp'],f'{mol_id}.pickle')):
						has_frags.append(mol_id)

			if len(has_frags) < len(mol_df_grp):				
				#for mol_id in cannot_predict_mol_id:
				cannot_mol_df = mol_df_grp[~mol_df_grp['mol_id'].isin(has_frags)]
				cannot_spec_df = spec_df_grp[~spec_df_grp['mol_id'].isin(has_frags)]
				cannot_df = pd.merge(cannot_spec_df, cannot_mol_df, on="mol_id")
	
			mol_df_grp = mol_df_grp[mol_df_grp['mol_id'].isin(has_frags)]
			spec_df_grp = spec_df_grp[spec_df_grp['mol_id'].isin(has_frags)]

		config_d['mol_fp'] = mol_df_grp
		config_d['spec_fp'] = spec_df_grp

		split_type = 'predict'
		ds = model_ds_cls(split=split_type, **config_d)
  
		dl = init_dataloader(ds, config_d)
		# load models
		if device is None:
			device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
		
		if device == "cpu":
			th.set_num_threads(multiprocessing.cpu_count())
			th.multiprocessing.set_sharing_strategy('file_system')
			config_d['num_workers']= 0

		preds_l = []
		has_oos = False
		with th.no_grad():
			for _, batch_input in tqdm(enumerate(dl),total=len(dl), desc=f"> save fp: {save_fp}"):

				spec_unique_ids  = batch_input['spec_unique_id']
				mol_smiles = batch_input['mol_smiles']

				batch_result = to_device(batch_input, device)
				#batch_result = model.predict_step(**batch_input)
				batch_result = model._common_step(batch_input,  split="val", no_log = True)
				batch_size = batch_input['batch_size'].detach().cpu()

				pred_mzs = batch_result["pred_mzs"].detach().cpu()
				pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
				pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()

				# this only for fragnnet
				if "pred_oos_logprobs" in batch_result:
					pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()
					has_oos = True
				else:
					pred_oos_logprobs = None
	 
				cos_sim_bin_hr = batch_result['cos_sim_0.01'].cpu()
				#cos_sim_bin_lr = batch_result['cos_sim_0.1'].cpu()
				cos_sim_hun = batch_result['cos_hun'].cpu()
				
				for b_idx in range(batch_size):
					b_peak_mask = pred_batch_idxs == b_idx
					b_cos_sim_bin_hr = cos_sim_bin_hr[b_idx].numpy()
					#b_cos_sim_bin_lr = cos_sim_bin_lr[b_idx].numpy()
					b_cos_sim_hun = cos_sim_hun[b_idx].numpy()

					# peaks
					b_pred_mzs = pred_mzs[b_peak_mask]
					b_pred_ints = pred_ints[b_peak_mask]
					b_pred_ints = model.ints_untransform_func(b_pred_ints)
					b_pred_mzs = b_pred_mzs.numpy()
					b_pred_ints = b_pred_ints.numpy()
					row_data = [spec_unique_ids[b_idx].item(), 
					 			mol_smiles[b_idx], 
								b_pred_mzs, 
						  		b_pred_ints,
						   		b_cos_sim_bin_hr,
							 	b_cos_sim_hun]
					# oos
					if pred_oos_logprobs is not None:
						b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy()
						row_data.append(b_pred_oos_logprob)
					preds_l.append(row_data)

		columns=['spec_id','smiles','pred_mzs','pred_ints','cos_sim_bin', 'cos_sim_hun']
		if has_oos:
			columns.append('oos')
		
		if cannot_df is not None:
			for _,cannot_row in cannot_df.iterrows():
				row_data = [cannot_row['spec_id'],cannot_row['smiles'],[],[],0.0,0.0]#,np.nan]
				if has_oos:
					row_data.append(0.0)
				preds_l.append(row_data)
				
		df = pd.DataFrame(preds_l, columns=columns)
		if cannot_df is not None:
			df.sort_values(by=['spec_id'], inplace = True)
		#print(df)
		df.to_pickle(save_fp)
		
def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--template_fp",type=str,default="config/template.yml")
	parser.add_argument("--frag_dp", type=str,default="./data/ms2c/pubchem/frags/d4_h4_isoFalse/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50", help="frag dp only used in fragnnet")

	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--batch_size", type=int,default=16, help="")
	parser.add_argument("--model_type", type=str,default="fragnnet", help="")

	parser.add_argument("--custom_fp",type=str,default="./config/retrieval/scaffold_fraggnn_d4_v2.yml")
	parser.add_argument("--ckpt_fp", type=str,default="./saved_ckpts/icml_inchikey/inchikey_fraggnn_d4_v2/inchikey_fraggnn_d4_v2.ckpt", help="")
	parser.add_argument("--task_dp", type=str,default="./data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50", help="")
	parser.add_argument("--save_dp", type=str,default="./ms2c_debug", help="")
	parser.add_argument("--disable_jit", type=booltype, default=False , help="True to force use torch jit")
	parser.add_argument("--allow_cached", type=booltype, default=True , help="Skip Already computed")
	args = parser.parse_args()
	return args


if __name__ == "__main__":

	args = get_args()
	#TORCH_LOGS="recompiles"
	os.makedirs(args.save_dp, exist_ok=True)
	json.dump(vars(args), open(f"{args.save_dp}/args.json", "w"), indent=2)
	
	model_type_to_model_cls = {
		"neims": NeimsPL,
		"massformer": MassFormerPL,
		"fragnnet": FragGNNPL,
		"fraggnn": FragGNNPL,
	}

	model_type_ds_cls = {
		"neims": SpecMolDataset,
		"massformer": SpecMolDataset,
		"fragnnet": SpecMolFragDataset,
		"fraggnn": SpecMolFragDataset,
	}
	if args.model_type not in model_type_ds_cls or args.model_type not in model_type_to_model_cls: 
		print(f"> model_type {args.model_type} is not supported")
		exit()
  
	run_prediction(ckpt_fp = args.ckpt_fp,   
				 save_dp = args.save_dp,
				 template_fp = args.template_fp,
				 custom_fp = args.custom_fp,
				 task_dp = args.task_dp,
	 			 model_cls = model_type_to_model_cls[args.model_type],
				 model_ds_cls = model_type_ds_cls[args.model_type],
				 frag_dp  = args.frag_dp,
				 device = args.device,
				 batch_size = args.batch_size,
				 allow_cached = args.allow_cached,
	 			 disable_jit = args.disable_jit)
