import os
import torch as th
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
import argparse

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

	substructure_smart = Chem.MolToSmarts(q_mol)
	substructure_smart = substructure_smart.replace("#0","!#1")
	if verbose:
		print(f"Substructure SMARTS: {substructure_smart}")
		print(f"Substructure Canonical SMILES: {substructure_smiles}")
		print(f"Substructure base mass: {mw}")
		print(f"Substructure formula: {formula}")
  
	return substructure_smart,substructure_smiles,mw,formula

def match_smart_substructure(target_smiles, substructure_smiles):
	print("Target SMILES:", target_smiles)
	print("Substructure SMARTS:", substructure_smiles)
	target_mol = Chem.MolFromSmiles(target_smiles)
	#Chem.SanitizeMol(target_mol)
	qmol = Chem.MolFromSmarts(substructure_smiles)

	ind_map = {}
	for atom in qmol.GetAtoms() :
		map_num = atom.GetAtomMapNum()
		if map_num:
			ind_map[map_num-1] = atom.GetIdx()
	
	for atom in qmol.GetAtoms():
		map_num = atom.GetAtomMapNum()
		if map_num:
			ind_map[map_num-1] = atom.GetIdx()
		
	map_list = [ind_map[x] for x in sorted(ind_map)]
	matches = []
	for match in target_mol.GetSubstructMatches( qmol ) :
		mas = [match[x] for x in map_list]
		matches.append(mas)
	
	return matches

def process_b_idx(b_idx, spec_batch_idxs, spec_prec_type_str, data, num_hs):
	"""Process a single b_idx."""
	b_spec_mask = spec_batch_idxs == b_idx
	mol_smiles = data['mol_smiles'][b_idx]
	b_spec_prec_type_str = spec_prec_type_str[b_idx]
	b_mol = mol_from_smiles(mol_smiles)

	frag_pyg = data['frag_pyg'][b_idx]
	frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
	frag_node_mask = th_long_to_mask(get_node_feats(frag_pyg.x, frag_node_feat_idxs, "cc"))

	b_pred_mzs = []
	b_frag_node_ids = []
	b_smarts = []

	for i in range(frag_node_mask.shape[0]):
		all_atom_ids = th.arange(0, frag_node_mask.shape[1]).to(th.long)
		frag_atom_ids = all_atom_ids[frag_node_mask[i]].tolist()
		substructure_smart, substructure_smiles, mw, substructure_formula = atom_ids_to_smarts(b_mol, frag_atom_ids)
		b_pred_mzs += [mw + H_MASS * j + PREC_TYPE_TO_MASS_DIFF[b_spec_prec_type_str] for j in range(-num_hs, num_hs + 1)]
		b_frag_node_ids += [i] * (2 * num_hs + 1)
		b_smarts.append([i, substructure_smart, substructure_smiles, mw, substructure_formula, b_spec_prec_type_str])

	b_true_mzs = data['spec_mzs'][b_spec_mask]
	b_pred_mzs = th.tensor(b_pred_mzs)
	b_frag_node_ids = th.tensor(b_frag_node_ids).to(th.long)

	relative = True
	b_match_mask = calculate_match_mzs(
		b_true_mzs,
		b_pred_mzs,
		tolerance=10 * PPM,
		relative=relative,
		tolerance_min_mz=TOLERANCE_MIN_MZ
	)

	b_pred_match_mask = th.any(b_match_mask, dim=0)
	matched_frag_node_id = th.unique(b_frag_node_ids[b_pred_match_mask])
	b_smarts_df = pd.DataFrame(b_smarts, columns=["node_id", "smarts", "smiles", "mw", "formula", "prec_type",])
	b_smarts_df["mol_smiles"] = mol_smiles
	b_smarts_df["mol_inchikey"] = Chem.MolToInchiKey(b_mol)
 
	b_smarts_df = b_smarts_df[b_smarts_df["node_id"].isin(matched_frag_node_id.tolist())]

	return b_smarts_df

def get_args():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("--template_fp", type=str, default="config/template.yml", help="Path to the config file")
	parser.add_argument("--custom_fp", type=str, default="config/exp_fraggnn_mi_ma/fraggnn_d3_mi_ma.yml", help="Path to the custom config file")
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--output_dp", type=str, default="./data/smarts/fraggnn_d3_mi_ma", help="Path to the output file")
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
	num_hs = config_d['num_hs']
 
	# Create output directory
	os.makedirs(args.output_dp, exist_ok=True)	
	
	split_names = ['train','val','test']
	dses = init_dataset(config_d, splits = split_names)

	for split_name,dset in zip(split_names,dses):
		dl = init_dataloader(dset, config_d)
		output_fp = os.path.join(args.output_dp, f"smarts_{split_name}.csv")
		for batch_idx, data in tqdm(enumerate(dl), total=len(dl), desc="Processing batches"):

			spec_mzs = data['spec_mzs']
			spec_prec_type = data['spec_prec_type']
			spec_inst_type = data['spec_inst_type']
			spec_prec_type_str = data['spec_prec_type_str']
			frag_pyg = data['frag_pyg']
			frag_formula_peak_mzs = data['frag_formula_peak_mzs']
			frag_formula_peak_idxs = data['frag_formula_peak_idxs']

			spec_batch_idxs = data['spec_batch_idxs']

			# Parallelize the loop
			results = Parallel(n_jobs=-1)(
				delayed(process_b_idx)(b_idx, spec_batch_idxs, spec_prec_type_str, data, num_hs)
				for b_idx in range(data['batch_size'].item())
			)

			# Combine results
			final_results = pd.concat(results, ignore_index=True)
			if batch_idx == 0:
				final_results.to_csv(output_fp, index=False, mode='w', header=True)
			else:
				final_results.to_csv(output_fp, index=False, mode='a', header=0)
