import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.MolStandardize as MolStandardize
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.Draw as Draw
import numpy as np
import torch as th
import networkx as nx
import scipy
import scipy.special
from scipy.special import comb
from pprint import pprint
import os
from collections import Counter
import pyximport
pyximport.install(language_level=3)

import compute_frags
from compute_frags import MAX_NUM_EDGES, MAX_NUM_NODES, MAX_EDGE_PER_NODE, DTYPE

ELEMENT_TO_VE = {
	"C": 4,
	"O": 2,
	"N": 3,
	"P": 3, # up to 5
	"S": 2, # up to 6
	"F": 1,
	"Cl": 1
}

ELEMENT_TO_IDX = { elem: idx for idx, elem in enumerate(ELEMENT_TO_VE.keys())}
ELEMENT_TO_IDX["H"] = len(ELEMENT_TO_VE.keys())
IDX_TO_ELEMENT = { idx: elem for elem, idx in ELEMENT_TO_IDX.items()}

MAX_H_TRANSFER = 4

def convert_to_int(cc):

	return sum([2**i for i in cc])

def convert_from_int(num_nodes,cc_int,bitmask=True):

	cc = []
	quot = cc_int
	for i in range(num_nodes):
		quot, rem = divmod(quot,2)
		if bitmask:
			cc.append(int(rem))
		else:
			if rem == 1:
				cc.append(i)
	cc = tuple(cc)
	return cc

def get_arrays_from_mol_d(mol_d):

	num_nodes = mol_d["atom_mask_arr"].shape[0]
	num_edges = mol_d["bond_mask_arr"].shape[0]
	assert num_nodes <= MAX_NUM_NODES, num_nodes
	assert num_edges <= MAX_NUM_EDGES, num_edges
	edges = np.zeros((MAX_NUM_EDGES,2),dtype=DTYPE)
	for bond_idx, bond in enumerate(mol_d["bonds"]):
		edges[bond_idx,0] = bond[0]
		edges[bond_idx,1] = bond[1]
	edge_to_node_idx = compute_frags.py_compute_node_to_edge_idx(num_nodes,num_edges,edges)
	# print(edge_to_node_idx)
	node_mask = np.zeros((MAX_NUM_NODES,),dtype=DTYPE)
	node_mask[:num_nodes] = 1
	edge_mask = np.zeros((MAX_NUM_EDGES,),dtype=DTYPE)
	edge_mask[:num_edges] = 1
	return num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx

def viz_atoms(mol,atom_idxs=None,color=(0,1,0),im_dir="tmp/viz_atoms/"):

	os.makedirs(im_dir,exist_ok=True)
	if atom_idxs == None:
		for atom in mol.GetAtoms():
			atom_idx = atom.GetIdx()
			# print(atom,atom_idx)
			im = Draw.MolToImage(mol,highlightAtoms=[atom_idx],highlightColor=color)
			im.save(os.path.join(im_dir,f"atom_{atom_idx:02d}.png"))
	else:
		im = Draw.MolToImage(mol,highlightAtoms=atom_idxs,highlightColor=color)
		im.save(os.path.join(im_dir,f"atom_{atom_idxs}.png"))

def extract_mol_info(smiles_or_mol,prec_type):

	if isinstance(smiles_or_mol,str):
		# rdkit mol stuff
		mol = Chem.MolFromSmiles(smiles_or_mol)
		# TODO: think about adding Hs
		rdMolStandardize.Cleanup(mol)
		te = rdMolStandardize.TautomerEnumerator()
		mol = te.Canonicalize(mol)
	else:
		assert isinstance(smiles_or_mol,Chem.rdchem.Mol), type(smiles_or_mol)
		mol = smiles_or_mol
	# some checks
	charge = rdmolops.GetFormalCharge(mol)
	assert charge == 0, charge
	# enumerate atoms
	sbond_arr = []
	ve_arr = []
	hs_arr = []
	elems = []
	num_hs = 0
	num_atoms = 0
	num_radicals = 0
	element_counts = {elem:0 for elem in ELEMENT_TO_VE.keys()}
	for atom in mol.GetAtoms():
		cur_idx = atom.GetIdx()
		cur_num_hs = atom.GetTotalNumHs()
		cur_deg = atom.GetTotalDegree()
		# cur_num_bonds = atom.GetNumBonds()
		cur_element = atom.GetSymbol()
		if cur_element not in element_counts:
			raise ValueError(f"Molecules with {cur_element} atom(s) currently not supported")
		element_counts[cur_element] += 1
		elems.append(cur_element)
		sbond_arr.append(cur_deg - cur_num_hs)
		ve_arr.append(ELEMENT_TO_VE[cur_element])
		hs_arr.append(cur_num_hs)
		num_hs += cur_num_hs
		num_atoms += 1
		num_radicals += atom.GetNumRadicalElectrons()
	element_counts["H"] = num_hs
	assert num_radicals == 0, num_radicals
	sbond_arr = np.array(sbond_arr)
	ve_arr = np.array(ve_arr)
	# enumerate bonds
	atoms_to_bonds = {}
	bonds = []
	num_bonds = 0
	adj = np.zeros((num_atoms, num_atoms), dtype=np.int32)
	for bond in mol.GetBonds():
		cur_idx = bond.GetIdx()
		from_idx = bond.GetBeginAtomIdx()
		to_idx = bond.GetEndAtomIdx()
		assert from_idx != to_idx
		adj[from_idx, to_idx] = 1
		bonds.append((from_idx, to_idx))
		if from_idx not in atoms_to_bonds:
			atoms_to_bonds[from_idx] = [cur_idx]
		else:
			atoms_to_bonds[from_idx].append(cur_idx)
		if to_idx not in atoms_to_bonds:
			atoms_to_bonds[to_idx] = [cur_idx]
		else:
			atoms_to_bonds[to_idx].append(cur_idx)
		num_bonds += 1
	if prec_type == "[M+H]+":
		prec_charge = 1
		prec_num_radicals = 0
	else:
		raise NotImplementedError(prec_type)
	mol_d = {}
	mol_d["mol"] = mol
	mol_d["prec_charge"] = prec_charge
	mol_d["prec_num_radicals"] = prec_num_radicals
	mol_d["num_hs"] = num_hs
	mol_d["sbond_arr"] = sbond_arr
	mol_d["ve_arr"] = ve_arr
	mol_d["hs_arr"] = hs_arr
	mol_d["elems"] = elems
	mol_d["bonds"] = bonds
	mol_d["bond_mask_arr"] = np.ones((num_bonds,), dtype=bool)
	mol_d["atom_mask_arr"] = np.ones((num_atoms,), dtype=bool) # can be computed
	mol_d["atoms_to_bonds"] = atoms_to_bonds # can be computed
	mol_d["element_counts"] = element_counts
	return mol_d

def compute_cc_cap(cc,ve_arr,sbond_arr,num_radicals):
	
	assert num_radicals == 0
	if not isinstance(cc,np.ndarray):
		cc = np.array(list(cc))
	cap_sbond_arr = sbond_arr[cc]
	cap_ve_arr = ve_arr[cc]
	cap_ve_mask = cap_ve_arr < cap_sbond_arr
	cap_ve_arr[cap_ve_mask] = cap_sbond_arr[cap_ve_mask]
	return np.sum(cap_ve_arr) - np.sum(cap_sbond_arr) - num_radicals

def compute_cc_floor(cc,ve_arr,sbond_arr,num_radicals,bonds,atoms_to_bonds,bond_mask_arr):

	assert num_radicals == 0
	diff_arr = np.maximum(ve_arr-sbond_arr,0)
	cc_atoms = list(cc)
	# print(diff_arr)
	# this computes a lower bound
	h_arr = np.copy(diff_arr)
	for atom_idx, atom in enumerate(cc_atoms):
		bond_idxs = atoms_to_bonds[atom]
		for bond_idx in bond_idxs:
			if h_arr[atom] == 0:
				break
			if not bond_mask_arr[bond_idx]:
				continue
			bond = bonds[bond_idx]
			if bond[0] == atom:
				other = bond[1]
			else:
				other = bond[0]
			# dont't form more than 3 bonds with anything!
			h_arr[atom] = max(0,h_arr[atom]-min(diff_arr[other],2))
	# print(h_arr)
	cc_floor = sum(h_arr[atom] for atom in cc_atoms)
	cc_floor -= num_radicals
	cc_floor = max(cc_floor,0) # why can cc_floor be negative?
	return cc_floor

def compute_all_no_mask(bonds):

	all_bond_breaks, all_actions = [], []
	for bond_idx, bond in enumerate(bonds):
		for charge_idx in [0,1]:
			for transfer in range(-MAX_H_TRANSFER,MAX_H_TRANSFER+1):
				all_actions.append((bond_idx,charge_idx,transfer))
		all_bond_breaks.append(bond_idx)
	return all_bond_breaks, all_actions

def compute_forward_mask(mol_d):

	ve_arr = mol_d["ve_arr"]
	sbond_arr = mol_d["sbond_arr"]
	hs_arr = mol_d["hs_arr"]
	prec_charge = mol_d["prec_charge"]
	prec_num_radicals = mol_d["prec_num_radicals"]
	num_hs = mol_d["num_hs"]
	bonds = mol_d["bonds"]
	bond_mask_arr = mol_d["bond_mask_arr"]
	atoms_to_bonds = mol_d["atoms_to_bonds"]
	assert prec_charge == 1 and prec_num_radicals == 0
	assert num_hs <= np.sum(hs_arr)
	# construct graph
	g_bonds = []
	for bond_idx, bond in enumerate(bonds):
		if bond_mask_arr[bond_idx]:
			g_bonds.append(bond)
	g = nx.Graph()
	g.add_edges_from(g_bonds)
	# verify that there is only one connected component
	all_ccs = list(nx.connected_components(g))
	assert len(all_ccs) == 1
	# break each bond and compute the resulting fragments
	all_bond_breaks, all_actions = compute_all_no_mask(bonds)
	# find the valid ones
	valid_bond_breaks = []
	valid_actions = []
	valid_children = []
	print(bonds)
	for bond_idx, bond in enumerate(bonds):
		# check if the bond is present in the parent
		if not bond_mask_arr[bond_idx]:
			continue
		# remove bond and modify graph
		print(f"> bond {bond_idx} - {bond}")
		g.remove_edge(bond[0],bond[1])
		bond_mask_arr[bond_idx] = False
		sbond_arr[bond[0]] -= 1
		sbond_arr[bond[1]] -= 1
		# iterate over connected components
		ccs = list(nx.connected_components(g))
		assert len(ccs) > 0 and len(ccs) <= 2
		# convention: order ccs by membership of the first bond atom
		if len(ccs) > 0 and bond[0] not in ccs[0]:
			ccs = [ccs[1],ccs[0]]
		else:
			assert bond[0] in ccs[0]
		all_ccs.extend(ccs)
		caps, floors, baselines = [], [], []
		for cc in ccs:
			cap = compute_cc_cap(cc,ve_arr,sbond_arr,0)
			floor = compute_cc_floor(cc,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)
			assert floor <= cap, (floor,cap)
			baseline = sum(hs_arr[atom] for atom in cc)
			assert baseline <= cap, (baseline,cap)
			# assert floor <= baseline, (floor,baseline)
			caps.append(cap)
			floors.append(floor)
			baselines.append(baseline)
		print(list((f,c) for f,c in zip(floors,caps)))
		print()
		if len(ccs) == 1:
			# single fragment,no hydrogen transfer
			# original fragment is still charged
			actions = [(bond_idx,0,0)]
			children = [(ccs[0],set())]
		else:
			# compute charged and neutral pairs
			actions, children = [], []
			for charge_idx in [0,1]:
				neutral_idx = 1-charge_idx
				charge_cc = ccs[charge_idx]
				charge_cap = caps[charge_idx]-prec_num_radicals
				charge_floor = floors[charge_idx]-prec_num_radicals
				charge_baseline = baselines[charge_idx]
				neutral_cc = ccs[neutral_idx]
				neutral_cap = caps[neutral_idx]
				neutral_floor = floors[neutral_idx]
				neutral_baseline = baselines[neutral_idx]
				# print(charge_idx,charge_cap,charge_floor,charge_cc)
				# print(neutral_idx,neutral_cap,neutral_floor,neutral_cc)
				# compute baselines relative to current hydrogen count
				total_baseline = charge_baseline+neutral_baseline
				charge_baseline_hs = int(np.around((charge_baseline/total_baseline)*num_hs))
				neutral_baseline_hs = int(np.around((neutral_baseline/total_baseline)*num_hs))
				assert charge_baseline_hs+neutral_baseline_hs == num_hs
				for transfer in range(-MAX_H_TRANSFER,MAX_H_TRANSFER+1):
					charge_hs = charge_baseline_hs+transfer
					neutral_hs = neutral_baseline_hs-transfer
					# print(transfer,charge_hs,neutral_hs)
					charge_valid = charge_hs >= charge_floor and charge_hs <= charge_cap
					neutral_valid = neutral_hs >= neutral_floor and neutral_hs <= neutral_cap
					if charge_valid and neutral_valid:
						actions.append((bond_idx,charge_idx,transfer))
						children.append(((charge_cc,charge_hs),(neutral_cc,neutral_hs)))
		valid_actions.extend(actions)
		valid_children.extend(children)
		if len(actions) > 0:
			valid_bond_breaks.append(bond_idx)
		# restore graph
		g.add_edge(bond[0],bond[1])
		bond_mask_arr[bond_idx] = True
		sbond_arr[bond[0]] += 1
		sbond_arr[bond[1]] += 1
	print(f">> Possible: # bonds = {len(all_bond_breaks)}, # actions = {len(all_actions)}")
	print(f">> Valid: # bonds = {len(valid_bond_breaks)}, # actions = {len(valid_actions)}")
	pprint(valid_actions)
	valid_action_mask = np.zeros((len(bonds),2,2*MAX_H_TRANSFER+1),dtype=bool)
	assert valid_action_mask.size == len(all_actions)
	for valid_action in valid_actions:
		dim_0 = valid_action[0]
		dim_1 = valid_action[1]
		dim_2 = valid_action[2]+MAX_H_TRANSFER
		valid_action_mask[dim_0,dim_1,dim_2] = True
	print(np.sum(valid_action_mask),valid_action_mask.size)
	mask_d = {
		"all_bond_breaks": all_bond_breaks,
		"all_actions": all_actions,
		"valid_bond_breaks": valid_bond_breaks,
		"valid_actions": valid_actions,
		"valid_children": valid_children,
		"valid_action_mask": valid_action_mask
	}
	return mask_d

def take_action(mol_d,action,children):

	bond_idx, charge_idx, transfer = action
	(charge_child,charge_hs), (neutral_child,neutral_hs) = children
	bonds = mol_d["bonds"] # never mutated
	bond_mask_arr = np.copy(mol_d["bond_mask_arr"])
	atom_mask_arr = np.copy(mol_d["atom_mask_arr"])
	atoms_to_bonds = mol_d["atoms_to_bonds"] # never mutated
	sbond_arr = np.copy(mol_d["sbond_arr"])
	ve_arr = np.copy(mol_d["ve_arr"])
	hs_arr = np.copy(mol_d["hs_arr"])
	num_hs = mol_d["num_hs"]
	# the broken bond
	removed_bond_idxs = [bond_idx]
	# all bonds from the neutral child
	for atom in neutral_child:
		for bond_idx in atoms_to_bonds[atom]:
			removed_bond_idxs.append(bond_idx)
		# discard atom statistics
		sbond_arr[atom] = 0.
		ve_arr[atom] = 0.
		hs_arr[atom] = 0.
		atom_mask_arr[atom] = False
	# removed all identified bonds
	for removed_bond_idx in removed_bond_idxs:
		removed_bond = bonds[removed_bond_idx]
		# update bond mask
		bond_mask_arr[removed_bond_idx] = False
		# update atom statistics
		if removed_bond[0] in charge_child:
			sbond_arr[removed_bond[0]] -= 1
		if removed_bond[1] in charge_child:
			sbond_arr[removed_bond[1]] -= 1
	# update h count
	num_hs = charge_hs
	# update dictionary
	new_mol_d = {**mol_d}
	new_mol_d["bond_mask_arr"] = bond_mask_arr
	new_mol_d["atom_mask_arr"] = atom_mask_arr
	new_mol_d["sbond_arr"] = sbond_arr
	new_mol_d["ve_arr"] = ve_arr
	new_mol_d["hs_arr"] = hs_arr
	new_mol_d["num_hs"] = num_hs
	return new_mol_d

def compute_state_arr(mol_d):

	# must be a flat vector
	# variable part: num_hs, bond_mask_arr (optionally: atom_mask_arr)
	# static part: bond_arr, sbond_arr, ve_arr, hs_arr
	atom_mask_arr = mol_d["atom_mask_arr"].astype(int)
	bond_mask_arr = mol_d["bond_mask_arr"].astype(int)
	num_hs_arr = np.array([mol_d["num_hs"]],dtype=int)
	state_arr = np.concatenate([atom_mask_arr,bond_mask_arr,num_hs],axis=0)
	return state_arr

def compute_approximate_formulae(ccs,mol_d,h_prior=False,max_h_transfer=MAX_H_TRANSFER,return_counts=False,max_depths=None):
	"""
	Note: this assumes fragments are fully defined by their node cover,
	not their edges. This might result in some undercounting.
	"""

	bonds = mol_d["bonds"] # never mutated
	atoms_to_bonds = mol_d["atoms_to_bonds"] # never mutated
	ve_arr = np.copy(mol_d["ve_arr"])
	sbond_arr = np.copy(mol_d["sbond_arr"])
	num_hs = mol_d["num_hs"]
	hs_arr = mol_d["hs_arr"]
	elems = mol_d["elems"]
	bond_mask_arr = mol_d["bond_mask_arr"]
	formulae = []
	formulae_to_counts = {}
	for cc in list(set(ccs)):
		base_formula = cc_to_formula(cc,elems)
		sbond_arr, bond_mask_arr = update_bonds(cc,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
		cap = compute_cc_cap(cc,ve_arr,sbond_arr,0)
		floor = compute_cc_floor(cc,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)
		assert floor <= cap, (floor,cap,cc)
		assert floor >= 0, floor
		if h_prior:
			num_hs_prior = 0
			for atom in cc:
				num_hs_prior += hs_arr[atom]
			floor = max(floor,num_hs_prior-max_h_transfer)
			cap = min(cap,num_hs_prior+max_h_transfer,num_hs)
		else:
			cap = min(cap,num_hs)
		for h in range(floor,cap+1):
			formula = np.copy(base_formula)
			formula[ELEMENT_TO_IDX["H"]] = h
			formula = tuple(formula)
			if formula in formulae_to_counts:
				formulae_to_counts[formula] += 1
			else:
				formulae_to_counts[formula] = 1
	if return_counts:
		return formulae_to_counts
	else:
		return set(formulae_to_counts.keys())

def update_bonds(frag,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds):

	sbond_arr = np.zeros_like(sbond_arr,dtype=int)
	bond_mask_arr = np.zeros_like(bond_mask_arr,dtype=bool)
	for atom in frag:
		for bond_idx in atoms_to_bonds[atom]:
			bond = bonds[bond_idx]
			if bond[0] in frag and bond[1] in frag:
				sbond_arr[atom] += 1
				bond_mask_arr[bond_idx] = True
	return sbond_arr, bond_mask_arr

def compute_approximate_cchs(ccs,mol_d,h_prior=False,max_h_transfer=MAX_H_TRANSFER):

	bonds = mol_d["bonds"] # never mutated
	atoms_to_bonds = mol_d["atoms_to_bonds"] # never mutated
	ve_arr = np.copy(mol_d["ve_arr"])
	sbond_arr = np.copy(mol_d["sbond_arr"])
	num_hs = mol_d["num_hs"]
	hs_arr = mol_d["hs_arr"]
	elems = mol_d["elems"]
	bond_mask_arr = mol_d["bond_mask_arr"]
	cchs = []
	all_cchs = set()
	for cc in list(set(ccs)):
		sbond_arr, bond_mask_arr = update_bonds(cc,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
		cap = compute_cc_cap(cc,ve_arr,sbond_arr,0)
		floor = compute_cc_floor(cc,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)
		assert floor <= cap, (floor,cap,cc)
		assert floor >= 0, floor
		if h_prior:
			num_hs_prior = 0
			for atom in cc:
				num_hs_prior += hs_arr[atom]
			floor = max(floor,num_hs_prior-max_h_transfer)
			cap = min(cap,num_hs_prior+max_h_transfer,num_hs)
			# cap = max(cap,floor)
		else:
			cap = min(cap,num_hs)
		# cc_cchs = []
		for h in range(floor,cap+1):
			# cc_cchs.append((cc,h))
			all_cchs.add((cc,h))
		# cchs.append(cc_cchs)
		# assert floor <= cap, (floor,cap,cc)
		# assert floor >= 0, floor
	return all_cchs

def cc_to_formula(cc,elems):
	# does not include h count

	formula_arr = np.zeros([len(ELEMENT_TO_IDX)],dtype=int)
	for atom in cc:
		elem = elems[atom]
		formula_arr[ELEMENT_TO_IDX[elem]] += 1
	return formula_arr

def get_formula_string(formula):

	formula_str = ""
	for idx, count in enumerate(formula):
		elem = IDX_TO_ELEMENT[idx]
		if count > 0:
			formula_str += elem
			if count > 1:
				formula_str += str(int(count))
	return formula_str

def ne_subgraph(nodes,edges):
	g = nx.Graph()
	g.add_nodes_from(nodes)
	for edge in edges:
		if edge[0] in nodes and edge[1] in nodes:
			g.add_edge(edge[0],edge[1])
	# print(nodes,g.nodes(),g.edges())
	return g

def compute_ccs_recursive(g,depth,max_depth):
	bonds = list(g.edges())
	if depth == max_depth:
		return []
	elif len(bonds) == 0:
		assert g.number_of_nodes() == 1
		return [tuple(list(g.nodes()))]
	else:
		all_ccs = []
		for bond_idx, bond in enumerate(bonds):
			# if depth == 0:
			# 	print(f"> {bond_idx} - {bond}")
			g.remove_edge(bond[0],bond[1])
			ccs = list(nx.connected_components(g))
			other_bonds = [bonds[idx] for idx in range(len(bonds)) if idx != bond_idx]
			sgs = [ne_subgraph(cc,other_bonds) for cc in ccs]
			assert len(sgs) <= 2 and len(sgs) > 0
			all_ccs.extend([tuple(list(sg.nodes())) for sg in sgs])
			for sg in sgs:
				all_ccs.extend(compute_ccs_recursive(sg,depth+1,max_depth))
			# print(">>", depth, all_ccs)
			g.add_edge(bond[0],bond[1])
		return all_ccs

def collect_ccs_depths(ccs,ccs_depths):

	ccs_to_depths = {}
	for cc_idx, cc in enumerate(ccs):
		if cc in ccs_to_depths:
			ccs_to_depths[cc].add(ccs_depths[cc_idx])
		else:
			ccs_to_depths[cc] = set([ccs_depths[cc_idx]])
	return ccs_to_depths

def collect_ccs_parents(ccs,ccs_parents):

	ccs_to_parents = {}
	for cc_idx, cc in enumerate(ccs):
		if cc in ccs_to_parents:
			ccs_to_parents[cc].add(ccs_parents[cc_idx])
		else:
			ccs_to_parents[cc] = set([ccs_parents[cc_idx]])
	return ccs_to_parents

