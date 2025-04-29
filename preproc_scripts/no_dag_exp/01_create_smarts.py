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

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from copy import deepcopy

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

def atom_ids_to_smarts(mol, atom_ids):
    # Create an empty molecule and add only the atoms of interest
    new_mol = Chem.RWMol()

    # Add atoms that are in the atom_ids list to the new molecule
    atom_map = {}
    for atom_idx in atom_ids:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_atom_idx = new_mol.AddAtom(atom)
        atom_map[atom_idx] = new_atom_idx

    # Now we add the bonds that correspond to the atoms in atom_ids
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()

        # Check if both atoms are in the atom_ids list, if so, add the bond
        if idx1 in atom_ids and idx2 in atom_ids:
            new_mol.AddBond(atom_map[idx1], atom_map[idx2], bond.GetBondType())
        elif idx1 in atom_ids or idx2 in atom_ids:
            dummy_atm = AllChem.Atom(0)
            if idx1 in atom_ids:
                new_mol.AddAtom(dummy_atm)
                new_mol.AddBond(atom_map[idx1], new_mol.GetNumAtoms() - 1, bond.GetBondType())
                new_mol.GetAtomWithIdx(atom_map[idx1]).GetImplicitValence()
                new_mol.GetAtomWithIdx(atom_map[idx1]).SetNumExplicitHs(0)
                new_mol.GetAtomWithIdx(atom_map[idx1]).SetNoImplicit(False)
            else:
                new_mol.AddAtom(dummy_atm)
                new_mol.AddBond(atom_map[idx2], new_mol.GetNumAtoms() - 1, bond.GetBondType())
                new_mol.GetAtomWithIdx(atom_map[idx2]).GetImplicitValence()
                new_mol.GetAtomWithIdx(atom_map[idx2]).SetNumExplicitHs(0)
                new_mol.GetAtomWithIdx(atom_map[idx2]).SetNoImplicit(False)

    # Get the substructure as a SMILES string
    new_mol.UpdatePropertyCache()
    mw = Descriptors.ExactMolWt(new_mol)
    substructure_smiles = Chem.MolToSmiles(new_mol, canonical=True)
    # this need newer version #ignoreAtomMapNumbers=True)
    
    #https://www.rdkit.org/docs/GettingStartedInPython.html#atom-map-indices-in-smarts
    for i, atom in enumerate(new_mol.GetAtoms()):
        # For each atom, set the property "molAtomMapNumber" to a custom number, let's say, the index of the atom in the molecule
        if atom.GetAtomicNum() != 0:
            atom.SetProp("molAtomMapNumber", str(atom.GetIdx()+1))

    substructure_smart = Chem.MolToSmarts(new_mol)
    substructure_smart = substructure_smart.replace("#0","!#1")
    print(f"Substructure SMARTS: {substructure_smart}")
    print(f"Substructure Canonical SMILES: {substructure_smiles}")
    print(f"Substructure base mass: {mw}")
    return substructure_smart,substructure_smiles,mw

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
    
    for atom in qmol.GetAtoms() :
        map_num = atom.GetAtomMapNum()
        if map_num:
            ind_map[map_num-1] = atom.GetIdx()
        
    map_list = [ind_map[x] for x in sorted(ind_map)]
    matches = []
    for match in target_mol.GetSubstructMatches( qmol ) :
        mas = [match[x] for x in map_list]
        matches.append(mas)
    
    return matches

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

		print(data['frag_formula_peak_idxs'])
		print(data['frag_formula_sizes'])
		print(data['frag_formula_cumsizes'])
      	#frag_formula_peak_idxs
		#frag_formula_sizes
		#frag_formula_cumsizes
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
   
			#for i in range(frag_node_mask.shape[0]):
			#	all_atom_ids = th.range(0,frag_node_mask.shape[1]-1).to(th.long)
			#	frag_atom_ids = all_atom_ids[frag_node_mask[i]].tolist()
			#	atom_ids_to_smarts(b_mol,frag_atom_ids)

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
		#exit()
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
