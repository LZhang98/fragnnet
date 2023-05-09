from copy import deepcopy
import networkx as nx
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
import rdkit.Chem.rdmolops as rdmolops
import torch as th
import torch_geometric as pyg

from frag_gnn.utils.misc_utils import get_pyg_memory_usage
import frag_gnn.frag.compute_frags as compute_frags
from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF, formula_to_peak_mzs
from frag_gnn.frag.compute_frags import (
	DTYPE,
	MAX_NUM_EDGES,
	MAX_NUM_NODES,
)

# full list of common elements
# NOTE: this maybe a bad idea to not cover all the element
# we should handle all the elemnet
ELEMENT_TO_VE = {
	"C": 4,
	"O": 2,
	"N": 3,
	"P": 3, # up to 5
	"S": 2, # up to 6
	"F": 1,
	"Cl": 1,
	"Br": 1,
	"I": 1,
	"Se": 2, # up to 6, same as S
	"Si": 4
}
NUM_HEAVY_ELEMENTS = len(ELEMENT_TO_VE.keys())
NUM_ELEMENTS = NUM_HEAVY_ELEMENTS+1

CANONICAL_ELEMENT_ORDER = ["C","H"] + sorted([elem for elem in ELEMENT_TO_VE.keys() if elem != "C" and elem != "H"])
CANONICAL_H_IDX = CANONICAL_ELEMENT_ORDER.index("H")

ELEMENT_TO_IDX = { elem: idx for idx, elem in enumerate(ELEMENT_TO_VE.keys())}
ELEMENT_TO_IDX["H"] = len(ELEMENT_TO_VE.keys())

IDX_TO_ELEMENT = { idx: elem for elem, idx in ELEMENT_TO_IDX.items()}

MAX_H_TRANSFER = 4
MAX_NUM_MZS_PER_FORMULA = 5

def convert_cc_mask_to_int(cc:list|np.ndarray)->int:
	"""covert a cc mask to an int

	Args: a list of 0 and 1s

	Returns:
		int: int presentation of input mask
	"""

	return sum([2**i for i in cc])

def convert_cc_int_to_mask(num_nodes:int,cc_int:int,bitmask=True) -> tuple[int]:
	"""_summary_

	Args:
		num_nodes (int): number of nodes
		cc_int (int): int version of cc mask
		bitmask (bool, optional): _description_. Defaults to True.

	Returns:
		list|np.ndarray: a list of 0 and 1s
	"""

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

def get_fraggen_input_arrays(mol_d:dict):
	"""_summary_

	Args:
		mol_d (_type_): _description_

	Returns:
		_type_: _description_
	"""
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

def extract_mol_info(smiles_or_mol, use_default_valence=False) ->dict:
	"""Method to exatract mol infomation in to dict

	Args:
		smiles_or_mol (_type_): _description_
		use_default_valence (bool, optional): _description_. Defaults to False.

	Raises:
		ValueError: _description_

	Returns:
		dict : 
		mol_d["mol"]: mol object
		mol_d["num_hs"]: number of totoal Hs
		mol_d["sbond_arr"]: single bond array
		mol_d["ve_arr"]: max velance array
		mol_d["hs_arr"]: max Hs per atom arary
		mol_d["elems"]: elemns per atom
		mol_d["bonds"]: a list of from atom - to atom pairs
		mol_d["bond_mask_arr"]: defualt bond mask, this should be just 1s
		mol_d["atom_mask_arr"] : defualt atom  mask, this should be just 1s
		mol_d["atoms_to_bonds"]: 
		mol_d["element_counts"] : atom count per element
	"""
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

	pt = Chem.GetPeriodicTable()
	#GetValenceList
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
		# number of single bond need to attach to this atom to keep atom connected
		# this equals to replace all all bond to single, and count how many single bond
		sbond_arr.append(cur_deg - cur_num_hs)
		#ve_arr.append(ELEMENT_TO_VE[cur_element])

		# set valence value for each atom
		# assumption we will use current valance unless
		# use default valence flag is set to true
		# not defualt valence can be -1 for transition metals
		# on paper we should not encounter them at all
		ve_value = atom.GetTotalValence()
		if use_default_valence:
			default_valence = pt.GetDefaultValence(cur_element)
			if default_valence != -1:
				ve_value = min(ve_value, default_valence)

		ve_arr.append(ve_value)

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
	#adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
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


	mol_d = {}
	mol_d["mol"] = mol
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

def compute_cc_h_cap(cc:list|np.ndarray,ve_arr:list[int],sbond_arr:list[int],num_radicals:int):
	"""compute max amount of Hs a cc can have.  
		For any ccs the max amount of Hs it can have is the congifcation where all the bond are single
		And all the atom has max amount of Hs
	Args:
		cc (list|np.ndarray): cc mask in list form
		ve_arr (list|np.ndarray): max velance each atom can have
		sbond_arr (_type_): _description_
		num_radicals (_type_): _description_

	Returns:
		_type_: _description_
	"""
	
	assert num_radicals == 0
	if not isinstance(cc,np.ndarray):
		cc = np.array(list(cc))
	cap_sbond_arr = sbond_arr[cc]
	cap_ve_arr = ve_arr[cc]
	cap_ve_mask = cap_ve_arr < cap_sbond_arr
	cap_ve_arr[cap_ve_mask] = cap_sbond_arr[cap_ve_mask]
	return np.sum(cap_ve_arr) - np.sum(cap_sbond_arr) - num_radicals

def compute_cc_h_floor(cc,ve_arr,sbond_arr,num_radicals,bonds,atoms_to_bonds,bond_mask_arr):

	"""compute lowest amount of Hs a cc can have
	NOTE: this does not consider if OG mol has a triple bond
	Args:
		cc (_type_): _description_
		ve_arr (_type_): _description_
		sbond_arr (_type_): _description_
		num_radicals (_type_): _description_

	Returns:
		_type_: _description_
	"""

	assert num_radicals == 0
	# if we could use update from single bond to double bond to get an electron pair
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

def compute_approximate_formula(cc:list|np.ndarray,mol_d:dict
				,max_h_transfer:int
				,formula_strs:bool=False) ->dict[int,str]|dict[int,np.ndarray]:
	"""given a connected component, comupute all forumla within give h shift

	Args:
		cc (list|np.ndarray): connected component
		mol_d (dict): mol dictionary
		max_h_transfer (_type_): max number of h movement allowed
		formula_strs (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""

	bonds = mol_d["bonds"] # never mutated
	atoms_to_bonds = mol_d["atoms_to_bonds"] # never mutated
	ve_arr = np.copy(mol_d["ve_arr"])
	sbond_arr = np.copy(mol_d["sbond_arr"])
	num_hs = mol_d["num_hs"]
	hs_arr = mol_d["hs_arr"]
	elems = mol_d["elems"]
	bond_mask_arr = mol_d["bond_mask_arr"]
	base_formula = cc_to_formula_arr(cc,elems)
	sbond_arr, bond_mask_arr = update_bonds(cc,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
	cap = compute_cc_h_cap(cc,ve_arr,sbond_arr,0)
	floor = compute_cc_h_floor(cc,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)

	# check floor and cap
	assert floor <= cap, (floor,cap,cc)
	assert floor >= 0, floor

	# check how many Hs each atom have on the mol
	num_hs_prior = 0
	for atom in cc:
		num_hs_prior += hs_arr[atom]

	# update min and max
	floor = max(floor,num_hs_prior-max_h_transfer)
	cap = min(cap,num_hs_prior+max_h_transfer,num_hs)

	delta_h_to_formula = {}
	for delta_h in range(-max_h_transfer,max_h_transfer+1):
		h = num_hs_prior + delta_h
		if h < floor or h > cap:
			# special invalid formula
			formula = np.zeros_like(base_formula)
		else:
			formula = np.copy(base_formula)
			formula[ELEMENT_TO_IDX["H"]] = h
			formula = tuple(formula)
		if formula_strs:
			formula = formula_arr_to_str(formula)
		delta_h_to_formula[h] = formula
	return delta_h_to_formula

def compute_approximate_formulae(ccs,mol_d,h_prior=False,max_h_transfer=MAX_H_TRANSFER,return_counts=False):
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
	formulae_to_counts = {}
	for cc in list(set(ccs)):
		base_formula = cc_to_formula_arr(cc,elems)
		sbond_arr, bond_mask_arr = update_bonds(cc,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
		cap = compute_cc_h_cap(cc,ve_arr,sbond_arr,0)
		floor = compute_cc_h_floor(cc,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)
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
	mol_d["elems"]
	bond_mask_arr = mol_d["bond_mask_arr"]
	all_cchs = set()
	for cc in list(set(ccs)):
		sbond_arr, bond_mask_arr = update_bonds(cc,sbond_arr,bond_mask_arr,bonds,atoms_to_bonds)
		cap = compute_cc_h_cap(cc,ve_arr,sbond_arr,0)
		floor = compute_cc_h_floor(cc,ve_arr,sbond_arr,0,bonds,atoms_to_bonds,bond_mask_arr)
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

def cc_to_formula_arr(cc,elems,bitmask=False):
	# does not include h count
	formula_arr = np.zeros([len(ELEMENT_TO_IDX)],dtype=int)
	if bitmask:
		for i in range(len(cc)):
			if cc[i]:
				elem = elems[i]
				formula_arr[ELEMENT_TO_IDX[elem]] += 1
	else:
		for atom in cc:
			elem = elems[atom]
			formula_arr[ELEMENT_TO_IDX[elem]] += 1
	return formula_arr

def formula_arr_to_str(formula):
	""" use canonical order """
	formula_str = ""
	elem_d = {}

	for idx, count in enumerate(formula):
		elem = IDX_TO_ELEMENT[idx]
		elem_d[elem] = count

	for elem in CANONICAL_ELEMENT_ORDER:
		if elem in elem_d:
			count = elem_d[elem]
			if count > 0:
				formula_str += elem
			if count > 1:
				formula_str += str(count)
	return formula_str

def ne_subgraph(nodes,edges):
	g = nx.Graph()
	g.add_nodes_from(nodes)
	for edge in edges:
		if edge[0] in nodes and edge[1] in nodes:
			g.add_edge(edge[0],edge[1])
	# print(nodes,g.nodes(),g.edges())
	return g

def collect_ccs_depths(ccs: list[int], ccs_depths: list[int]) -> dict[set[int]]:
	"""Method to collect depth for ccs

	Args:
		ccs (list[int]): list of connected components
		ccs_depths (list[int]): depths for each components

	Returns:
		dict[set[int]]: _description_
	"""

	ccs_to_depths = {}
	for cc_idx, cc in enumerate(ccs):
		depth = ccs_depths[cc_idx]
		if cc in ccs_to_depths:
			ccs_to_depths[cc].add(depth)
		else:
			ccs_to_depths[cc] = set([depth])
	return ccs_to_depths

def collect_ccs_parents(ccs: list[int],ccs_parents:list[int]) -> dict[set[int]]:
	"""Method to collect depth for ccs

	Args:
		ccs (list[int]): list of connected components
		ccs_depths (list[int]): parent ccs for each components

	Returns:
		dict[set[int]]: _description_
	"""

	ccs_to_parents = {}
	for cc_idx, cc in enumerate(ccs):
		parent = ccs_parents[cc_idx]
		if cc in ccs_to_parents:
			ccs_to_parents[cc].add(parent)
		else:
			ccs_to_parents[cc] = set([parent])
	return ccs_to_parents


def compute_frags_cython(mol_d,max_depth,parents=False,depths=False):

	assert not parents
	num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx = get_fraggen_input_arrays(mol_d)
	# time this part
	ccs, ccs_parents, _ = compute_frags.compute_ccs(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		edge_to_node_idx,
		max_depth,
		parents,
		depths,
		int(1e6)
	)
	if depths:
		ccs_to_parents = collect_ccs_parents(ccs,ccs_parents)
		return ccs_to_parents
	else:
		ccs = set(ccs)
		return ccs

def compute_frag_peak_stats(row, tolerance=0.01, prec_type="[M+H]+", 
			    is_abs_tolerance = True):
	"""_summary_

	Args:
		row (_type_): _description_
		tolerance (float, optional): _description_. Defaults to 0.01.
		prec_type (str, optional): _description_. Defaults to "[M+H]+".

	Returns:
		_type_: _description_
	"""

	true_mzs, true_ints = list(zip(*row["peaks"]))
	pred_mzs = row["formula_peak_mzs"]
	pred_probs = row["formula_peak_probs"]
	prec_mask = (pred_probs > 0.).type(th.float32)
	# account for adduct mass
	pred_mzs = pred_mzs + prec_mask*PREC_TYPE_TO_MASS_DIFF[prec_type]
	# compute overlap
	overlap_true_idxs = []
	overlap_true_ints = []
	overlap_pred_idxs = []
	overlap_pred_peak_counts = []
	overlap_pred_formula_counts = []
	for true_idx, true_mz in enumerate(true_mzs):
		mz_diffs = th.abs(pred_mzs-true_mz)
		mz_close = mz_diffs < tolerance
		if th.any(mz_close):
			pred_idx = th.nonzero(mz_close,as_tuple=False)
			num_formula_match = th.sum(th.any(mz_close,dim=1).type(th.int32)).item()
			num_peak_match = pred_idx.shape[0]
			overlap_true_idxs.append(true_idx)
			overlap_true_ints.append(true_ints[true_idx])
			overlap_pred_idxs.append(pred_idx)
			overlap_pred_peak_counts.append(num_peak_match)
			overlap_pred_formula_counts.append(num_formula_match)
	# remove duplicates
	if len(overlap_pred_idxs) > 0:
		overlap_pred_idxs = th.unique(th.cat(overlap_pred_idxs,dim=0),dim=0)
	else:
		overlap_pred_idxs = th.zeros((0,2))
	recall = len(overlap_true_idxs) / len(true_mzs)
	w_recall = np.sum(overlap_true_ints) / np.sum(true_ints)
	prec = len(overlap_pred_idxs) / th.sum((pred_probs>0.).type(th.int32)).item()
	if len(overlap_pred_peak_counts) > 0:
		ppt_peak = np.mean(overlap_pred_peak_counts)
		ppt_formula = np.mean(overlap_pred_formula_counts)
	else:
		ppt_peak = np.nan
		ppt_formula = np.nan
	return pd.Series([recall,w_recall,prec,ppt_peak,ppt_formula])

def get_node_feat_names_shapes(nx_g):

	node_feat_shapes = []
	node_feat_names = []
	dummy_node_idx = next(iter(nx_g.nodes()))
	dummy_node_feats = nx_g.nodes[dummy_node_idx]
	for k,v in dummy_node_feats.items():
		node_feat_names.append(k)
		node_feat_shapes.append(len(v))
	return node_feat_names, node_feat_shapes

def get_edge_feat_names_shapes(nx_g):

	edge_feat_shapes = []
	edge_feat_names = []
	dummy_edge_idx = next(iter(nx_g.edges()))
	dummy_edge_feats = nx_g.edges[dummy_edge_idx]
	for k,v in dummy_edge_feats.items():
		edge_feat_names.append(k)
		edge_feat_shapes.append(len(v))
	return edge_feat_names, edge_feat_shapes

def compute_dags(mol_d,max_depth,h_prior,max_h_transfer,frag_max_time):
	"""_summary_

	Args:
		mol_d (_type_): _description_
		max_depth (_type_): _description_
		h_prior (_type_): _description_
		max_h_transfer (_type_): _description_
		frag_max_time (_type_): _description_

	Raises:
		NotImplementedError: _description_
		NotImplementedError: _description_

	Returns:
		_type_: _description_
	"""

	assert h_prior in [True,False], h_prior
	num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx = get_fraggen_input_arrays(mol_d)
	# time the recursive part
	if frag_max_time is None:
		frag_max_time = int(1e6)
	
	node_mask = node_mask.astype(np.byte)
	edge_mask = edge_mask.astype(np.byte)
	#def _compute_frags():
	ccs, ccs_parents, ccs_depths = compute_frags.compute_ccs(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		edge_to_node_idx,
		max_depth,
		True,
		True,
		frag_max_time
	)

	assert len(ccs) == len(ccs_parents) == len(ccs_depths)
	ccs_to_parents = collect_ccs_parents(ccs,ccs_parents)
	ccs_to_depths = collect_ccs_depths(ccs,ccs_depths)
	
	#return ccs, ccs_to_parents, ccs_to_depths
	#ccs, ccs_to_parents, ccs_to_depths = _compute_frags()

	# construct graph
	nx_cc_g = nx.DiGraph()
	ccs_un = sorted(list(set(ccs)))
	assert len(ccs_un) == len(ccs_to_parents) == len(ccs_to_depths)
	assert -1 not in ccs_un
	nx_cc_g.add_nodes_from(ccs_un)

	# add node depth information
	# convert to a one hot encoding
	for cc, depths in ccs_to_depths.items():
		depth_arr = [0 for _ in range(max_depth+1)]
		for depth in depths:
			depth_arr[depth] = 1
		ccs_to_depths[cc] = depth_arr
	nx.set_node_attributes(nx_cc_g,ccs_to_depths,"depth")

	# add node mask information
	# convert to a one hot encoding
	ccs_to_mask = {cc : convert_cc_int_to_mask(num_nodes,cc,bitmask=True) for cc in ccs_un}
	assert len(ccs_to_mask) == len(ccs_to_parents) == len(ccs_to_depths)
	nx.set_node_attributes(nx_cc_g,ccs_to_mask,"mask")

	# add node h count and element information
	hs_arr = mol_d["hs_arr"]
	elems = mol_d["elems"]
	ccs_to_formula = {}
	for cc, cc_mask in ccs_to_mask.items():
		cc_h_count = sum(cc_mask[i]*hs_arr[i] for i in range(len(cc_mask)))
		cc_formula = cc_to_formula_arr(cc_mask,elems,bitmask=True).tolist()
		cc_formula[CANONICAL_H_IDX] = cc_h_count
		ccs_to_formula[cc] = cc_formula
	nx.set_node_attributes(nx_cc_g,ccs_to_formula,"base_formula")

	# map nodes to formulae
	ccs_to_formulae = {}
	formulae_un = set()
	for cc in ccs_un:
		cc_mask = convert_cc_int_to_mask(num_nodes,cc,bitmask=False)
		delta_h_to_formula = compute_approximate_formula(cc_mask,mol_d,max_h_transfer,formula_strs=True)
		# print(delta_h_to_formula)
		formulae = list(delta_h_to_formula.values())
		ccs_to_formulae[cc] = formulae
		formulae_un.update(formulae)
		
	# map nodes to formulae indices
	formula_to_idx = {formula:idx for idx,formula in enumerate(sorted(list(formulae_un)))}
	idx_to_formula = {idx:formula for formula,idx in formula_to_idx.items()}
	ccs_to_formulae_idx = {}
	for cc in ccs_un:
		formulae = ccs_to_formulae[cc]
		formulae_idx = []
		for formula in formulae:
			formula_idx = formula_to_idx[formula]
			formulae_idx.append(formula_idx)
		ccs_to_formulae_idx[cc] = formulae_idx
	nx.set_node_attributes(nx_cc_g,ccs_to_formulae_idx,"h_formulae_idx")
	# peak mzs array
	formula_peak_mzs = []
	formula_peak_probs = []
	peaks_for_element_cache = {}
	for formula, idx in formula_to_idx.items():
		if formula == "":
			peak_mzs = np.zeros(MAX_NUM_MZS_PER_FORMULA,dtype=np.float32)
			peak_probs = np.zeros(MAX_NUM_MZS_PER_FORMULA,dtype=np.float32)
		else:
			peak_mzs, peak_probs = formula_to_peak_mzs(formula,"",return_probs=True, peaks_for_element_cache = peaks_for_element_cache)
			peak_mzs, peak_probs = zip(*sorted(zip(peak_mzs, peak_probs), key=lambda x: x[1], reverse=True))
			peak_mzs = np.array(peak_mzs[:MAX_NUM_MZS_PER_FORMULA],dtype=np.float32)
			peak_mzs = np.pad(peak_mzs,(0,MAX_NUM_MZS_PER_FORMULA-len(peak_mzs)),"constant",constant_values=0)
			peak_probs = np.array(peak_probs[:MAX_NUM_MZS_PER_FORMULA],dtype=np.float32)
			peak_probs = np.pad(peak_probs,(0,MAX_NUM_MZS_PER_FORMULA-len(peak_probs)),"constant",constant_values=0)
		formula_peak_mzs.append(peak_mzs)
		formula_peak_probs.append(peak_probs)
	# save as bf16 for speed and lower ram usage
	formula_peak_mzs = th.as_tensor(np.stack(formula_peak_mzs,axis=0), dtype = th.bfloat16)
	formula_peak_probs = th.as_tensor(np.stack(formula_peak_probs,axis=0), dtype = th.bfloat16)

	# add edges
	edges_to_mask, edges_to_formula = {}, {}
	for cc, parents in ccs_to_parents.items():
		cc_mask = ccs_to_mask[cc]
		for parent in parents:
			# skip self edges
			if parent == cc or parent == -1:
				continue
			nx_cc_g.add_edge(parent,cc)
			parent_mask = ccs_to_mask[parent]
			diff_mask = [parent_mask[i]-cc_mask[i] for i in range(len(cc_mask))]
			edges_to_mask[(parent,cc)] = diff_mask
			diff_formula = cc_to_formula_arr(diff_mask,elems).tolist()
			edges_to_formula[(parent,cc)] = diff_formula
	nx.set_edge_attributes(nx_cc_g,edges_to_mask,"mask")
	nx.set_edge_attributes(nx_cc_g,edges_to_formula,"base_formula")

	# convert to pyg
	node_feat_names, node_feat_shapes = get_node_feat_names_shapes(nx_cc_g)
	edge_feat_names, edge_feat_shapes = get_edge_feat_names_shapes(nx_cc_g)
	pyg_cc_g = pyg.utils.convert.from_networkx(
		nx_cc_g, 
		group_node_attrs=node_feat_names, 
		group_edge_attrs=edge_feat_names,
	)
	
	#print(f'{get_pyg_memory_usage(pyg_cc_g)/1e6} MB')
	# TODO: can we reduce storage?
	pyg_cc_g.node_feat_idxs = th.cumsum(th.tensor([0]+node_feat_shapes,dtype=th.long),0).reshape(1,-1)
	#print(f'{get_pyg_memory_usage(pyg_cc_g)/1e6} MB')
	pyg_cc_g.edge_feat_idxs = th.cumsum(th.tensor([0]+edge_feat_shapes,dtype=th.long),0).reshape(1,-1)
	#print(f'{get_pyg_memory_usage(pyg_cc_g)/1e6} MB')
	
	dag_d = {}
	dag_d["max_depth"] = max_depth
	dag_d["h_prior"] = h_prior
	dag_d["max_h_transfer"] = max_h_transfer
	dag_d["dag"] = pyg_cc_g
	dag_d["formula_peak_mzs"] = formula_peak_mzs
	dag_d["formula_peak_probs"] = formula_peak_probs
	dag_d["idx_to_formula"] = idx_to_formula # useful for annotation

	return dag_d

def get_node_feats(node_feats,node_feat_idxs,key):

	if key == "depth":
		node_feats = node_feats[:,node_feat_idxs[0]:node_feat_idxs[1]]
	elif key == "mask":
		node_feats = node_feats[:,node_feat_idxs[1]:node_feat_idxs[2]]
	elif key == "base_formula":
		node_feats = node_feats[:,node_feat_idxs[2]:node_feat_idxs[3]]
	elif key == "h_formulae_idx":
		node_feats = node_feats[:,node_feat_idxs[3]:node_feat_idxs[4]]
	else:
		raise ValueError("key not found")
	return node_feats

def get_edge_feats(edge_feats,edge_feat_idxs,key):

	if key == "mask":
		key_edge_feats = edge_feats[:,edge_feat_idxs[0]:edge_feat_idxs[1]]
	elif key == "base_formula":
		key_edge_feats = edge_feats[:,edge_feat_idxs[1]:edge_feat_idxs[2]]
	else:
		raise ValueError("key not found")
	return key_edge_feats
