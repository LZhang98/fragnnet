import os
import argparse

import torch as th
import torch.nn.functional as F
from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from frag_gnn.utils.frag_utils import th_long_to_mask, get_node_feats
import pandas as pd

from frag_gnn.utils.misc_utils import PPM, TOLERANCE_MIN_MZ
from frag_gnn.runner import load_config, init_dataloader, init_dataset
from frag_gnn.utils.spec_utils import calculate_match_mzs
from frag_gnn.utils.data_utils import mol_from_smiles
from frag_gnn.utils.formula_utils import H_MASS

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from copy import deepcopy
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def draw_highlighted_atoms(mol, atom_ids):
	
	local_mol = deepcopy(mol)  # Create a deep copy of the molecule to avoid modifying the original
	for i, atom in enumerate(local_mol.GetAtoms()):
		# For each atom, set the property "molAtomMapNumber" to a custom number, let's say, the index of the atom in the molecule
		atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
	
	# Create a list of atom indices to highlight
	highlight_atoms = set(atom_ids)

	# Generate image with highlighted atoms
	img = Draw.MolToImage(local_mol, highlightAtoms=highlight_atoms, size=(300, 300))

	return img

def atom_ids_to_smarts(mol, atom_ids, verbose=False):
	"""_summary_

	Args:
		mol (_type_): _description_
		atom_ids (_type_): _description_
		verbose (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""
 
	# Create an empty molecule and add only the atoms of interest
	q_mol = Chem.RWMol()

	# Add atoms that are in the atom_ids list to the new molecule
	atom_map = {}
	for atom_idx in atom_ids:
		atom = mol.GetAtomWithIdx(atom_idx)
		new_atom_idx = q_mol.AddAtom(atom)
		atom_map[atom_idx] = new_atom_idx

	# Now we add the bonds that correspond to the atoms in atom_ids
	for bond in mol.GetBonds():
		idx1 = bond.GetBeginAtomIdx()
		idx2 = bond.GetEndAtomIdx()

		# Check if both atoms are in the atom_ids list, if so, add the bond
		if idx1 in atom_ids and idx2 in atom_ids:
			q_mol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())
		elif idx1 in atom_ids or idx2 in atom_ids:
			dummy_atm = AllChem.Atom(0)
			if idx1 in atom_ids:
				q_mol.AddAtom(dummy_atm)
				q_mol.AddBond(atom_map[idx1], q_mol.GetNumAtoms() - 1, bond.GetBondType())
				q_mol.GetAtomWithIdx(atom_map[idx1]).GetImplicitValence()
				q_mol.GetAtomWithIdx(atom_map[idx1]).SetNumExplicitHs(0)
				q_mol.GetAtomWithIdx(atom_map[idx1]).SetNoImplicit(False)
			else:
				q_mol.AddAtom(dummy_atm)
				q_mol.AddBond(atom_map[idx2], q_mol.GetNumAtoms() - 1, bond.GetBondType())
				q_mol.GetAtomWithIdx(atom_map[idx2]).GetImplicitValence()
				q_mol.GetAtomWithIdx(atom_map[idx2]).SetNumExplicitHs(0)
				q_mol.GetAtomWithIdx(atom_map[idx2]).SetNoImplicit(False)

	# Get the substructure as a SMILES string
	q_mol.UpdatePropertyCache()
	mw = Descriptors.ExactMolWt(q_mol)
	formula = Chem.rdMolDescriptors.CalcMolFormula(q_mol)
	substructure_smiles = Chem.MolToSmiles(q_mol, canonical=True)
	# this need newer version #ignoreAtomMapNumbers=True)
	
	#https://www.rdkit.org/docs/GettingStartedInPython.html#atom-map-indices-in-smarts
	for i, atom in enumerate(q_mol.GetAtoms()):
		# For each atom, set the property "molAtomMapNumber" to a custom number, let's say, the index of the atom in the molecule
		if atom.GetAtomicNum() != 0:
			atom.SetProp("molAtomMapNumber", str(atom.GetIdx()+1))

	substructure_smarts = Chem.MolToSmarts(q_mol)
	substructure_smarts = substructure_smarts.replace("#0","!#1")
	if verbose:
		print(f"Substructure SMARTS: {substructure_smarts}")
		print(f"Substructure Canonical SMILES: {substructure_smiles}")
		print(f"Substructure base mass: {mw}")
		print(f"Substructure formula: {formula}")
  
	return substructure_smarts,substructure_smiles,mw,formula

def get_remaining_connected_atom_ids(adj_matrix, atom_ids_to_remove):
	"""
	Removes specified atom IDs from the adjacency matrix and groups the remaining atoms by connected components.

	Parameters:
		adj_matrix (np.array): The adjacency matrix of the molecule.
		atom_ids_to_remove (list): List of atom indices to be removed from the matrix.

	Returns:
		List of connected components: Each component is a list of atom indices.
	"""
	adj_matrix = adj_matrix.copy()  # Create a copy of the adjacency matrix to avoid modifying the original
	# Remove rows and columns corresponding to the atoms to be removed
	for atom_id in atom_ids_to_remove:
		adj_matrix[:, atom_id] = 0  # Set the column to zero
		adj_matrix[atom_id, :] = 0 # Set the row to zero
		
	# Convert the updated adjacency matrix to a sparse CSR matrix for efficient computation
	sparse_matrix = csr_matrix(adj_matrix)

	# Use scipy's connected_components to find the connected components in the graph
	num_components, labels = connected_components(sparse_matrix, directed=False, return_labels=True)

	# Group atom indices by connected components
	connected_atom_ids = [np.where(labels == i)[0].tolist() for i in range(num_components)]
	connected_atom_ids = [ tuple(atom_ids) for atom_ids in connected_atom_ids if atom_ids[0] not in set(atom_ids_to_remove)]
  
	return connected_atom_ids

def convert_mask(input_mask):

	batch_size = input_mask.shape[0]
	num_h = input_mask.shape[1] // 2
	output = th.ones((batch_size, 2 * num_h + 1), dtype=input_mask.dtype, device=input_mask.device)

	# Indices for the rules
	odd_indices = th.arange(1,2*num_h+1,2, dtype=th.long, device=input_mask.device)
	even_indices = th.arange(2,2*num_h+2,2, dtype=th.long, device=input_mask.device)
	#print(odd_indices)
	#print(even_indices)
	# For output positions 0,1,2,3: set to zero if input at 0,2,4,6 is zero
	output[:, :num_h] = (input_mask[:, odd_indices] != 0).to(input_mask.dtype)

	# For output positions 4,5,6,7: set to zero if input at 1,3,5,7 is zero
	output[:, num_h+1:] = (input_mask[:, even_indices] != 0).to(input_mask.dtype)

	return output

def process_b_idx(b_idx, spec_batch_idxs, spec_prec_type_str, data, num_hs):
	"""Process a single b_idx."""
	b_spec_mask = spec_batch_idxs == b_idx
	mol_smiles = data['mol_smiles'][b_idx]
	b_spec_prec_type_str = spec_prec_type_str[b_idx]
	b_mol = mol_from_smiles(mol_smiles)
	b_mol_adj = Chem.rdmolops.GetAdjacencyMatrix(b_mol)
 
	frag_pyg = data['frag_pyg'][b_idx]
	frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]

	frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
	frag_node_mask = th_long_to_mask(get_node_feats(frag_pyg.x, frag_node_feat_idxs, "cc"))
	frag_joint_formula_idxs = (get_node_feats(frag_pyg.x,frag_node_feat_idxs,"h_formulae_idx"))

 
	b_num_nodes = frag_node_mask.shape[0]
	b_num_mw_per_node = 2 * num_hs + 1
	b_pred_mzs = th.zeros(b_num_nodes * b_num_mw_per_node, dtype=th.float)
	b_frag_node_ids = th.zeros(b_num_nodes * b_num_mw_per_node, dtype=th.long)
	b_frag_node_hs = th.zeros(b_num_nodes * b_num_mw_per_node, dtype=th.long)
	b_smarts = []
	nl_smarts_d = []
 
	# create a smart for each fragment mask
	# map [0,-1,1,-2,2,-3,3,-4,4] to [-4,-3,-2,-1,0,1,2,3,4]
	# why I have it designed that way... I hate myself
	frag_h_mask = convert_mask(frag_joint_formula_idxs)
	
	for i in range(frag_node_mask.shape[0]):
		all_atom_ids = th.arange(0, frag_node_mask.shape[1]).to(th.long)
		frag_atom_ids = all_atom_ids[frag_node_mask[i]].tolist()
		substructure_smarts, substructure_smiles, mw, substructure_formula = atom_ids_to_smarts(b_mol, frag_atom_ids)

		b_frag_node_ids[i * b_num_mw_per_node:(i + 1) * b_num_mw_per_node] = i
		b_frag_node_hs[i * b_num_mw_per_node:(i + 1) * b_num_mw_per_node] = th.arange(-num_hs, num_hs + 1)
		b_pred_mzs[i * b_num_mw_per_node:(i + 1) * b_num_mw_per_node] = mw + H_MASS * th.arange(-num_hs, num_hs + 1) + PREC_TYPE_TO_MASS_DIFF[b_spec_prec_type_str]
		b_smarts.append([i, substructure_smarts, substructure_smiles, mw, substructure_formula, b_spec_prec_type_str])

		nl_atom_ids_l = get_remaining_connected_atom_ids(b_mol_adj, atom_ids_to_remove= frag_atom_ids)
		for nl_atom_ids in nl_atom_ids_l:
			nl_substructure_smart,nl_substructure_smiles,nl_mw,nl_formula= atom_ids_to_smarts(b_mol, nl_atom_ids)
			nl_smarts_d.append([i, nl_substructure_smart, nl_substructure_smiles, nl_mw, nl_formula, b_spec_prec_type_str])
	b_pred_mzs = b_pred_mzs * frag_h_mask.flatten()
	# match the dag mzs with the true mzs
	#b_true_mzs_raw = data['spec_mzs'][b_spec_mask]
 
	b_true_mzs = data['spec_mzs'][b_spec_mask]
	b_true_ints = data['spec_ints'][b_spec_mask]
	b_true_log_ints = th.log(F.normalize(b_true_ints, p=1, dim=0))

	relative = True
	b_match_mask = calculate_match_mzs(
		b_true_mzs,
		b_pred_mzs,
		tolerance=10 * PPM,
		relative=relative,
		tolerance_min_mz=TOLERANCE_MIN_MZ
	)

	b_pred_match_mask = th.any(b_match_mask, dim=0)
	#b_true_match_mask = th.any(b_match_mask, dim=1)

	matched_frag_node_id = b_frag_node_ids[b_pred_match_mask]
	matched_frag_h = b_frag_node_hs[b_pred_match_mask]
	matched_frag_mz = b_pred_mzs[b_pred_match_mask]
	matched_peak_id = b_pred_match_mask.nonzero().squeeze(1)

	fn_matches_d = {}
	for fn_id, fn_mh, fn_mz, fn_peak_id in zip(matched_frag_node_id, matched_frag_h, matched_frag_mz, matched_peak_id):
		node_id = fn_id.item()
		if node_id not in fn_matches_d:
			fn_matches_d[node_id] = [] #{'matached_frag':[],'matached_peak':[]}

		# match the true mzs with the predicted frag mzs
		fn_true_mz = b_true_mzs[b_match_mask[:, fn_peak_id]]
		fn_true_ints = b_true_log_ints[b_match_mask[:, fn_peak_id]]
		fn_matched_peaks = th.stack([fn_true_mz, fn_true_ints]).T
		#print(f"fn_matched_peaks: {fn_matched_peaks.tolist()}")
		#fn_matches_d[node_id]['matached_peak'] = fn_matched_peaks.tolist()
		fn_matches_d[node_id].append((fn_mh.item(), fn_matched_peaks.tolist()))
		
	b_smarts_df = pd.DataFrame(b_smarts, columns=["node_id", "smarts", "smiles", "mw", "formula", "prec_type",])
	b_smarts_df["mol_smiles"] = mol_smiles
	b_smarts_df["mol_inchikey"] = Chem.MolToInchiKey(b_mol)
	# only keep the matched fragments
	unique_matched_frag_node_id = set(th.unique(matched_frag_node_id).tolist())
	b_smarts_df = b_smarts_df[b_smarts_df["node_id"].isin(unique_matched_frag_node_id)]
	b_fn_matches_df = pd.DataFrame(fn_matches_d.items(), columns=["node_id", "matches"])
	b_smarts_df = b_smarts_df.merge(b_fn_matches_df, on="node_id", how="left")
 
	# nl dataframe
	b_nl_smarts_df = pd.DataFrame(nl_smarts_d, columns=["node_id", "smarts", "smiles", "mw", "formula", "prec_type"])
	return (b_smarts_df, b_nl_smarts_df)

def get_args():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--template_fp", type=str, default="config/template.yml", help="Path to the config file")
	parser.add_argument("--custom_fp", type=str, default="config/debug/debug_tiny.yml", help="Path to the custom config file")
	parser.add_argument("--output_dp", type=str, default="./data/smarts/debug_tiny", help="Path to the output file")
	#parser.add_argument("--custom_fp", type=str, default="config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi.yml", help="Path to the custom config file")
	#parser.add_argument("--output_dp", type=str, default="./data/smarts/fraggnn_d3_ma_mi", help="Path to the output file")
	parser.add_argument("--batch_size", type=int, default=40, help="Batch size for the dataloader")
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--split_names", type=str, nargs="+", default=['train','val','test'], help="split to process")
	parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing, -1 for all cores")
	return parser.parse_args()

if __name__ == "__main__":
		
	args = get_args()

	# load config
	config_d  =load_config(args.template_fp, args.custom_fp) 

	config_d['train_batch_size'] = args.batch_size
	config_d['eval_batch_size'] = args.batch_size
	config_d['frag_params']['preload'] = False
	config_d['frag_params']['preprocess'] = False
	config_d['spec_params']['prec_type_str'] = True
	config_d['num_workers'] = args.num_workers 
	config_d['dynamic_batch_sampler'] = None
	config_d['dynamic_batch_sampler'] = False

	num_hs = config_d['num_hs']
 
	# Create output directory
	os.makedirs(args.output_dp, exist_ok=True)	
	
	dses = init_dataset(config_d, splits = args.split_names)
	for split_name,dset in zip(args.split_names,dses):
		dl = init_dataloader(dset, config_d)
		smart_output_fp = os.path.join(args.output_dp, f"smarts_{split_name}.csv")
		nl_output_fp = os.path.join(args.output_dp, f"nl_smarts_{split_name}.csv")
		for batch_idx, data in tqdm(enumerate(dl), total=len(dl), desc="Processing batches"):
			spec_mzs = data['spec_mzs']
			spec_prec_type = data['spec_prec_type']
			spec_inst_type = data['spec_inst_type']
			spec_prec_type_str = data['spec_prec_type_str']

			spec_batch_idxs = data['spec_batch_idxs']

			#process_b_idx(0, spec_batch_idxs, spec_prec_type_str, data, num_hs)
			#exit()
			results = Parallel(n_jobs=-1)(
				delayed(process_b_idx)(b_idx, spec_batch_idxs, spec_prec_type_str, data, num_hs)
				for b_idx in range(data['batch_size'].item())
			)

			smarts_dfs_l = [result[0] for result in results]
			nl_smarts_dfs_l = [result[1] for result in results]
   
			# Combine results
			for ofp, smarts_df_l in zip([smart_output_fp, nl_output_fp], [smarts_dfs_l, nl_smarts_dfs_l]):
				smarts_df = pd.concat(smarts_df_l, ignore_index=True)
				if batch_idx == 0:
					smarts_df.to_csv(ofp, index=False, mode='w', header=True)
				else:
					smarts_df.to_csv(ofp, index=False, mode='a', header=0)