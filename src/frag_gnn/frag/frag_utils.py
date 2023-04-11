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
from pprint import pprint
import os
import pandas as pd

import frag_gnn.frag.compute_frags as compute_frags
from frag_gnn.frag.compute_frags import MAX_NUM_EDGES, MAX_NUM_NODES, MAX_EDGE_PER_NODE, DTYPE

ELEMENT_TO_VE = {
	"C": 4,
	"O": 2,
	"N": 3,
	"P": 3, # up to 5
	"S": 2, # up to 6
	"F": 1,
	"Cl": 1
}

CANONICAL_ELEMENT_ORDER = ["C","H"] + sorted([elem for elem in ELEMENT_TO_VE.keys() if elem != "C" and elem != "H"])

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

def extract_mol_info(smiles_or_mol):

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
		base_formula = cc_to_formula_arr(cc,elems)
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
		depth = ccs_depths[cc_idx]
		if cc in ccs_to_depths:
			ccs_to_depths[cc].add(depth)
		else:
			ccs_to_depths[cc] = set([depth])
	return ccs_to_depths

def collect_ccs_parents(ccs,ccs_parents):

	ccs_to_parents = {}
	for cc_idx, cc in enumerate(ccs):
		parent = ccs_parents[cc_idx]
		if cc in ccs_to_parents:
			ccs_to_parents[cc].add(parent)
		else:
			ccs_to_parents[cc] = set([parent])
	return ccs_to_parents

def compute_frags_python(mol_d,max_depth,parents=False,depths=False):

	raise NotImplementedError
	# g = nx.Graph()
	# g.add_edges_from(mol_d["bonds"])
	# # time this part
	# ccs_rec = compute_ccs_recursive(g,0,max_depth)
	# ccs_rec = [convert_to_int(cc) for cc in ccs_rec]
	# ccs_rec = set(ccs_rec)
	# return ccs_rec

def compute_frags_cython(mol_d,max_depth,parents=False,depths=False):

	assert not parents
	num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx = get_arrays_from_mol_d(mol_d)
	# time this part
	ccs, ccs_parents, ccs_depths = compute_frags.py_compute_ccs_recursive(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		edge_to_node_idx,
		0,
		max_depth,
		True,
		parents,
		depths
	)
	if depths:
		ccs_to_parents = collect_ccs_parents(ccs,ccs_parents)
		return ccs_to_parents
	else:
		ccs = set(ccs)
		return ccs

def compute_frag_peak_stats(row,tolerance=0.01):

	true_mzs, true_ints = list(zip(*row["peaks"]))
	pred_mz_d = row["pred_mz_d"]
	pred_mzs = np.array(list(pred_mz_d.keys()))
	pred_formulae = np.array(list(pred_mz_d.values()))
	# compute overlap
	overlap_true_idxs = []
	overlap_true_ints = []
	overlap_pred_idxs = []
	overlap_pred_peak_counts = []
	overlap_pred_formula_counts = []
	for true_idx, true_mz in enumerate(true_mzs):
		mz_diffs = np.abs(pred_mzs-true_mz)
		mz_close = mz_diffs < tolerance
		if np.any(mz_close):
			pred_idx = np.nonzero(mz_close)[0]
			overlap_true_idxs.append(true_idx)
			overlap_true_ints.append(true_ints[true_idx])
			overlap_pred_idxs.append(pred_idx)
			overlap_pred_peak_counts.append(len(pred_idx))
			overlap_pred_formula_counts.append(len(np.unique(pred_formulae[pred_idx])))
	# remove duplicates
	if len(overlap_pred_idxs) > 0:
		overlap_pred_idxs = np.unique(np.concatenate(overlap_pred_idxs,axis=0))
	else:
		overlap_pred_idxs = np.array([])
	recall = len(overlap_true_idxs) / len(true_mzs)
	w_recall = np.sum(overlap_true_ints) / np.sum(true_ints)
	prec = len(overlap_pred_idxs) / len(pred_mzs)
	if len(overlap_pred_peak_counts) > 0:
		ppt_peak = np.mean(overlap_pred_peak_counts)
		ppt_formula = np.mean(overlap_pred_formula_counts)
	else:
		ppt_peak = 0.
		ppt_formula = 0.
	return pd.Series([recall,w_recall,prec,ppt_peak,ppt_formula])

def compute_dags(mol_d,max_depth,h_prior,max_h_transfer,formula_dag=False,cch_dag=False):

	assert h_prior in [True,False], h_prior
	num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx = get_arrays_from_mol_d(mol_d)
	# time this part
	ccs, ccs_parents, ccs_depths = compute_frags.py_compute_ccs_recursive(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		edge_to_node_idx,
		0,
		max_depth,
		True,
		True,
		True
	)
	assert len(ccs) == len(ccs_parents) == len(ccs_depths)
	ccs_to_parents = collect_ccs_parents(ccs,ccs_parents)
	ccs_to_depths = collect_ccs_depths(ccs,ccs_depths)
	# pprint(ccs_to_parents)
	# pprint(ccs_to_depths)

	# construct graph
	cc_g = nx.DiGraph()
	ccs_un = sorted(list(set(ccs)))
	assert len(ccs_un) == len(ccs_to_parents) == len(ccs_to_depths)
	assert -1 not in ccs_un
	cc_g.add_nodes_from(ccs_un)
	# add node depth information
	nx.set_node_attributes(cc_g,ccs_to_depths,"depth")
	# add node mask information
	ccs_to_mask = {cc : np.array(convert_from_int(num_nodes,cc,bitmask=True)) for cc in ccs_un}
	assert len(ccs_to_mask) == len(ccs_to_parents) == len(ccs_to_depths)
	nx.set_node_attributes(cc_g,ccs_to_mask,"mask")
	# add node h count and element information
	hs_arr = mol_d["hs_arr"]
	elems = mol_d["elems"]
	ccs_to_hs, ccs_to_formula = {}, {}
	for cc, cc_mask in ccs_to_mask.items():
		cc_h_count = np.sum(cc_mask*hs_arr)
		ccs_to_hs[cc] = cc_h_count
		cc_formula = cc_to_formula_arr(cc_mask,elems,bitmask=True)
		ccs_to_formula[cc] = cc_formula
	nx.set_node_attributes(cc_g,ccs_to_hs,"hs")
	nx.set_node_attributes(cc_g,ccs_to_formula,"formula")
	# add edges
	edges_to_mask, edges_to_hs, edges_to_formula = {}, {}, {}
	for cc, parents in ccs_to_parents.items():
		cc_mask = ccs_to_mask[cc]
		cc_hs = ccs_to_hs[cc]
		for parent in parents:
			# skip self edges
			if parent == cc or parent == -1:
				continue
			cc_g.add_edge(parent,cc)
			parent_mask = ccs_to_mask[parent]
			diff_mask = parent_mask - cc_mask
			edges_to_mask[(parent,cc)] = diff_mask
			parent_hs = ccs_to_hs[parent]
			diff_hs = parent_hs - cc_hs
			edges_to_hs[(parent,cc)] = diff_hs
			diff_formula = cc_to_formula_arr(diff_mask,elems)
			edges_to_formula[(parent,cc)] = diff_formula
	nx.set_edge_attributes(cc_g,edges_to_mask,"mask")
	nx.set_edge_attributes(cc_g,edges_to_hs,"hs")
	nx.set_edge_attributes(cc_g,edges_to_formula,"formula")

	# compute formulae
	ccs_to_formulae = {}
	formulae_to_ccs = {}
	for cc_int in ccs_un:
		cc_mask = convert_from_int(num_nodes,cc_int,bitmask=False)
		formulae = compute_approximate_formulae([cc_mask],mol_d,h_prior=h_prior,max_h_transfer=max_h_transfer)
		formulae = [formula_arr_to_str(f) for f in formulae]
		ccs_to_formulae[cc_int] = formulae
		for formula in formulae:
			if formula in formulae_to_ccs:
				formulae_to_ccs[formula].append(cc_int)
			else:
				formulae_to_ccs[formula] = [cc_int]

	dag_d = {}
	dag_d["max_depth"] = max_depth
	dag_d["h_prior"] = h_prior
	dag_d["max_h_transfer"] = max_h_transfer
	dag_d["ccs_to_formulae"] = ccs_to_formulae
	dag_d["formulae_to_ccs"] = formulae_to_ccs
	dag_d["cc_g"] = cc_g

	if formula_dag:
		
		formulae_to_idx = {formula:idx for idx,formula in enumerate(sorted(list(formulae_to_ccs.keys())))}
		for k,v in ccs_to_formulae.items():
			ccs_to_formulae[k] = [formulae_to_idx[f] for f in v]
		formula_g = nx.DiGraph()
		formula_g.add_nodes_from(list(range(len(formulae_to_idx))))
		for cc,formulae in ccs_to_formulae.items():
			cc_out_edges = cc_g.out_edges(cc)
			for out_formula in formulae:
				for cc_out_edge in cc_out_edges:
					assert cc_out_edge[0] == cc
					for in_formula in ccs_to_formulae[cc_out_edge[1]]:
						formula_g.add_edge(out_formula,in_formula)
		dag_d["formula_g"] = formula_g

	if cch_dag:
		
		# identify nodes
		all_cchs = set()
		dropped_ccs = []
		for cc_int in ccs_un:
			cc_mask = convert_from_int(num_nodes,cc_int,bitmask=False)
			cchs = compute_approximate_cchs([cc_mask],mol_d,h_prior=h_prior,max_h_transfer=max_h_transfer)
			if len(cchs) == 0:
				dropped_ccs.append(cc_int)
				continue
			cchs = [(cc_int,cch[1]) for cch in cchs]
			all_cchs.update(cchs)
		# all_ccs = sorted(list(set([cch[0] for cch in all_cchs])))
		print(f"> dropped {len(dropped_ccs)} ccs: {dropped_ccs}")
		cch_to_idx = {}
		cc_to_cch_idx = {}
		for idx, cch in enumerate(all_cchs):
			cc, h = cch[0], cch[1]
			if cc not in cc_to_cch_idx:
				cc_to_cch_idx[cc] = [idx]
			else:
				cc_to_cch_idx[cc].append(idx)
			cch_to_idx[cch] = idx
		cch_g = nx.DiGraph()
		cch_g.add_nodes_from(list(cch_to_idx.values()))
		# set up edges
		for cch, src_idx in cch_to_idx.items():
			cc, h = cch[0], cch[1]
			cc_out_edges = cc_g.out_edges(cc)
			for cc_out_edge in cc_out_edges:
				src_cc = cc_out_edge[0]
				assert src_cc == cc, (src_cc,cc)
				dst_cc = cc_out_edge[1]
				for dst_idx in cc_to_cch_idx.get(dst_cc,tuple()):
					cch_g.add_edge(src_idx,dst_idx)
		dag_d["cch_g"] = cch_g

	return dag_d