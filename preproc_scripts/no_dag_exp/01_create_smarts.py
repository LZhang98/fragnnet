import torch as th
import torch.nn.functional as F
from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from frag_gnn.utils.frag_utils import load_frag_d, th_long_to_mask, th_mask_to_long, get_node_feats
import pandas as pd

from frag_gnn.dataset import SpecMolDataset, SpecMolFragDataset
from frag_gnn.utils.misc_utils import PPM, TOLERANCE_MIN_MZ
from frag_gnn.runner import load_config, init_dataloader, init_dataset
from frag_gnn.utils.spec_utils import calculate_match_mzs, scipy_linear_sum_assignment, round_aggregate_peaks, batched_bin_func, jss_helper
from frag_gnn.utils.data_utils import mol_from_smiles

import warnings

def compute_frag_peak_stats(peaks,formula_peak_mzs,formula_peak_probs,idx_by_h_delta,\
							 prec_mz, allowed_h_transfer, tolerance=0.01,\
							 prec_type="[M+H]+", is_ppm=False):
	"""_summary_

	Args:
		peaks (_type_): _description_
		formula_peak_mzs (_type_): _description_
		formula_peak_probs (_type_): _description_
		idx_by_h_delta (_type_): _description_
		allowed_h_transfer (_type_): _description_
		tolerance (float, optional): _description_. Defaults to 0.01.
		is_ppm (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""
	true_mzs, true_ints = list(zip(*peaks))
	theoretical_mzs = formula_peak_mzs
	theoretical_probs = formula_peak_probs

	allowed_idx = list(idx_by_h_delta[0])
	for h in range(1, allowed_h_transfer):
		allowed_idx += list(idx_by_h_delta[2 * h - 1])
		allowed_idx += list(idx_by_h_delta[2 * h])
	allowed_idx = list(set(allowed_idx))
	#print(allowed_idx)
	indices = th.tensor(allowed_idx)
	theoretical_mzs = th.index_select(theoretical_mzs,0, indices)
	theoretical_probs = th.index_select(theoretical_probs,0, indices)

	prec_mask = (theoretical_probs > 0.).type(th.float32)
	# account for adduct mass
	theoretical_mzs = theoretical_mzs + prec_mask * PREC_TYPE_TO_MASS_DIFF[prec_type]
	# compute overlap
	overlap_pred_idxs = []

	# check true_mzs
	for true_idx, true_mz in enumerate(true_mzs):
		mz_diffs = th.abs(theoretical_mzs-true_mz)
		if not is_ppm:
			mz_close = mz_diffs < tolerance 
		else:
			mz_close = mz_diffs < (true_mz * tolerance * PPM)
		if th.any(mz_close):
			pred_idx = th.nonzero(mz_close,as_tuple=False)
			overlap_pred_idxs.append(pred_idx)
	  
	# remove duplicates
	if len(overlap_pred_idxs) > 0:
		overlap_pred_idxs = th.unique(th.cat(overlap_pred_idxs,dim=0),dim=0)
	else:
		overlap_pred_idxs = th.zeros((0,2))

if __name__ == "__main__":
		
	# load config
	template_fp = "config/template.yml"
	custom_fp   = "config/debug/debug_tiny.yml"
	config_d  =load_config(template_fp, custom_fp) 

	config_d['train_batch_size'] = 1
	config_d['eval_batch_size'] = 1
	config_d['frag_params']['preload'] = False
	config_d['frag_params']['preprocess'] = False

	dses = init_dataset(config_d)
	train_ds = dses[0]
	#val_ds = dses[1]
	
	train_dl = init_dataloader(train_ds, config_d)
	#val_dl = init_dataloader(val_ds, config_d)
 
	# we can do megred ms vs dag
	# this should have same effect as doing it separately, but faster
	for batch_idx, data in enumerate(train_dl):
		# since we are using a collate function, the data is a dict
		# and used only one data per batch
  
		# 'data' is your input batch, 'target' is your label batch
		#print(f"Batch {batch_idx + 1}")
		#print(f"Data: {data}")
		#print(data)
		# not this should only with size one batch
		spec_mzs = data['spec_mzs']
		spec_prec_type = data['spec_prec_type']
		spec_inst_type = data['spec_inst_type']
		frag_pyg = data['frag_pyg']
		frag_formula_peak_mzs = data['frag_formula_peak_mzs']
		frag_formula_peak_idxs = data['frag_formula_peak_idxs']
		#print(frag_formula_peak_idxs)
		print("frag_formula_peak_sizes",data['frag_formula_peak_sizes'])
		#print(frag_formula_peak_mzs.shape)
		#print(frag_formula_peak_mzs.shape)

		spec_batch_idxs = data['spec_batch_idxs']
		spec_ce_batch_idxs = data['spec_ce_batch_idxs']

		for b_idx in range(data['batch_size'].item()):
			#b_spec_mask = spec_batch_idxs == b_idx
			#b_frag_formula_peak_mask = frag_formula_peak_idxs == b_idx
			mol_smiles = data['mol_smiles'][b_idx]
			#print(mol_smiles)
			#print("b_spec_mask shape",b_spec_mask.shape)
			#print("spec_mzs shape",spec_mzs.shape)
			#print("frag_formula_peak_mzs shape",frag_formula_peak_mzs.shape)
			b_mol = mol_from_smiles(mol_smiles)
   
			frag_pyg = data['frag_pyg'][b_idx]
			frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
			frag_edge_feat_idxs = frag_pyg.edge_feat_idxs[0]
			frag_node_mask = th_long_to_mask(get_node_feats(frag_pyg.x,frag_node_feat_idxs,"cc"))
			print(frag_node_mask)

			"""
			b_spec_prec_type = spec_prec_type[b_idx]
			b_true_mzs = spec_mzs[b_spec_mask]
			b_pred_mzs = frag_formula_peak_mzs[b_frag_formula_peak_mask]
			print(b_true_mzs)
			print(b_pred_mzs)
			relative =True
			b_match_mask = calculate_match_mzs(
				b_true_mzs,
				b_pred_mzs,
				tolerance= 10,
				relative=relative,
				tolerance_min_mz=TOLERANCE_MIN_MZ
			)
			b_true_match_mask = th.any(b_match_mask,dim=1)
			b_pred_match_mask = th.any(b_match_mask,dim=0)
			print(b_true_match_mask)
			print(b_pred_match_mask)
			print(b_pred_mzs[b_pred_match_mask])
			"""
		exit()
		#spec_prec_mass_diff = data['spec_prec_mass_diff']
		# 
		#spec_mzs
		#spec_ints
		#spec_prec_type
		#spec_inst_type
		#spec_prec_mass_diff
		#spec_ce
		#spec_prec_mz
		#spec_unique_id
		#group_id
		#mol_id
		#spec_per_mol
		#group_per_mol
		#spec_per_group
		#mol_smiles
		#mol_pyg
		#frag_pyg
		#frag_formula_peak_mzs
		#frag_formula_peak_probs
		#mol_num_nodes
		#frag_num_nodes
		#frag_formula_peak_idxs
		#frag_formula_sizes
		#frag_formula_cumsizes
		#frag_formula_peak_sizes
		#spec_batch_idxs
		#spec_ce_batch_idxs
		#batch_size
  
		#for k,v in data.items():
		#	print(f"{k}")
		#print(data.keys())
		#print(len(data['frag_pyg']))
		#print(type(data))
		exit(0)

	exit()
	# 
	frag_d = load_frag_d(2, frag_fp, is_compressed=True)
	# load peaks
	print(frag_d)
	#node_depth = frag_d['dag'].x[:,0:4]
	print(node_depth)
	frag_cc_longs = frag_d['dag'].x[:,4:6]
	node_base_formula_matrix = frag_d['dag'].x[:,6:7]
	node_formulae_matrix = frag_d['dag'].x[:, 7:8]
 
	print(node_base_formula_matrix)
	print(node_formulae_matrix)
	#print(frag_cc_longs)
 
	masks = th_long_to_mask(frag_cc_longs)
	print(masks)
	# Get column indices per row
	# this are index of atoms for each cc
	row_indices = [th.nonzero(row, as_tuple=True)[0] for row in masks]

	for i, cols in enumerate(row_indices):
		print(f"Row {i}: {cols}")
		print(node_base_formula_matrix[i])
		print(node_formulae_matrix[i])
	#num_nodes = frag_d['dag_num_nodes']
	#idx_to_formula = frag_d['idx_to_formula']
	# compute stats
	# compute_frag_peak_stats(peaks, formula_peak_mzs, formula_peak_probs, idx_by_h_delta, prec_mz, allowed_h_transfer, tolerance, prec_type, is_ppm)