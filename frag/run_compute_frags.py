import numpy as np
import networkx as nx
import time
from pprint import pprint
import pyximport
pyximport.install(language_level=3)

import frag_utils
from misc_utils import timeit
import compute_frags


def python_function(mol_d,max_depth):

	g = nx.Graph()
	g.add_edges_from(mol_d["bonds"])
	# time this part
	ccs_rec = frag_utils.compute_ccs_recursive(g,0,max_depth)
	ccs_rec = [frag_utils.convert_to_int(cc) for cc in ccs_rec]
	ccs_rec = set(ccs_rec)
	return ccs_rec

def cython_function(mol_d,max_depth):

	num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx = frag_utils.get_arrays_from_mol_d(mol_d)
	# time this part
	ccs_rec, ccs_parents_rec = compute_frags.py_compute_ccs_recursive(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		edge_to_node_idx,
		0,
		max_depth,
		True,
		False
	)
	ccs_rec = set(ccs_rec)
	return ccs_rec

def cython_dag(mol_d,max_depth):

	num_nodes, num_edges, node_mask, edges, edge_mask, edge_to_node_idx = frag_utils.get_arrays_from_mol_d(mol_d)
	# time this part
	ccs_rec, ccs_parents_rec = compute_frags.py_compute_ccs_recursive(
		num_nodes,
		num_edges,
		node_mask,
		edges,
		edge_mask,
		edge_to_node_idx,
		0,
		max_depth,
		True,
		True
	)
	assert len(ccs_rec) == len(ccs_parents_rec)
	# construct graph
	cc_g = nx.DiGraph()
	ccs_rec_un = sorted(list(set(ccs_rec)))
	cc_g.add_nodes_from(ccs_rec_un)
	for cc_idx, cc_parent in enumerate(ccs_parents_rec):
		src = cc_parent
		dst = ccs_rec[cc_idx]
		cc_g.add_edge(src,dst)
	ccs_to_formulae = {}
	all_formulae = set()
	for cc_int in ccs_rec_un:
		cc_mask = frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False)
		formulae = frag_utils.compute_approximate_formulae([cc_mask],mol_d)
		formulae = [frag_utils.get_formula_string(f) for f in formulae]
		ccs_to_formulae[cc_int] = formulae
		all_formulae.update(formulae)
	# print(np.mean([len(v) for v in ccs_to_formulae.values()]))
	formulae_to_idx = {formula:idx for idx,formula in enumerate(sorted(list(set(all_formulae))))}
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
	return cc_g, formula_g

if __name__ == "__main__":

	np.random.seed(420)

	# smiles = "CCO" # ethanol
	# smiles = "c1ccccc1" # benzene
	# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O" # aspirin
	smiles = "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C" # testosterone
	# smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O" # geddic acid
	# smiles = "CSCCC(C(=O)O)N" # methionine, S=2
	# smiles = "C1=CC(=CC=C1N)S(=O)(=O)N" # sulfanilamide, S=6
	# smiles = "COP(C1=CC=CC=C1)C2=CC=CC=C2" # methoxydiphenylphosphine, P=3
	# smiles = "CC1C(O1)P(=O)(O)O" # phosphomycin, P=5
	max_depth = 4
	mol_d = frag_utils.extract_mol_info(smiles,"[M+H]+")
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(num_nodes)
	print("> python")
	python_ccs = timeit(python_function)(mol_d,max_depth)
	print("> cython")
	cython_ccs = timeit(cython_function)(mol_d,max_depth)
	diff1_ccs = cython_ccs-python_ccs
	print(f"> python-cython: {len(diff1_ccs)}")
	diff2_ccs = python_ccs-cython_ccs
	print(f"> cython-python: {len(diff2_ccs)}")

	# cython_ccs = timeit(cython_function)(mol_d,max_depth)
	# print(f"> num frags: {len(cython_ccs)}")
	# ccs = [frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False) for cc_int in cython_ccs]
	# formulae = frag_utils.compute_approximate_formulae(ccs,mol_d,h_prior=False)
	# print(f"> num formulae: {len(formulae)}")
	# formulae = frag_utils.compute_approximate_formulae(ccs,mol_d,h_prior=True)
	# print(f"> num formulae: {len(formulae)}")
	# cchs = frag_utils.compute_approximate_cchs(ccs,mol_d,h_prior=False)
	# print(f"> num cchs: {len(cchs)}")
	# cchs = frag_utils.compute_approximate_cchs(ccs,mol_d,h_prior=True)
	# print(f"> num cchs: {len(cchs)}")

	# cc_g, formula_g = timeit(cython_dag)(mol_d,max_depth)
	# print("> cc dag")
	# print(cc_g.number_of_nodes(),cc_g.number_of_edges())
	# print("> formula dag")
	# print(formula_g.number_of_nodes(),formula_g.number_of_edges())
	# print(list(map(frag_utils.get_formula_string,cython_formulae)))

	# cc_g, formula_g = timeit(cython_dag)(mol_d,max_depth)
	# print("> cc dag")
	# print(cc_g.number_of_nodes(),cc_g.number_of_edges())
	# print("> formula dag")
	# print(formula_g.number_of_nodes(),formula_g.number_of_edges())
