import numpy as np
import networkx as nx
import time
from pprint import pprint
import pyximport
pyximport.install(language_level=3)
import argparse

import frag_utils
from misc_utils import timeit, booltype
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

def cython_dag(mol_d,max_depth,dag_type,h_prior):

	assert dag_type in ["formula","cch"], dag_type
	assert h_prior in [True,False], h_prior
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
	if dag_type == "formula":
		ccs_to_formulae = {}
		all_formulae = set()
		for cc_int in ccs_rec_un:
			cc_mask = frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False)
			formulae = frag_utils.compute_approximate_formulae([cc_mask],mol_d,h_prior=h_prior)
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
	else:
		# identify nodes
		all_cchs = set()
		dropped_ccs = []
		for cc_int in ccs_rec_un:
			cc_mask = frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False)
			cchs = frag_utils.compute_approximate_cchs([cc_mask],mol_d,h_prior=h_prior)
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
		return cc_g, cch_g

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--compound_idx",type=int,default=0)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--dag_type",type=str,default="formula")
	parser.add_argument("--h_prior",type=booltype,default=True)
	args = parser.parse_args()

	np.random.seed(420)

	COMPOUND_TO_SMILES = {
		0: "CCO", # ethanol
		1: "c1ccccc1", # benzene
		2: "CC(=O)OC1=CC=CC=C1C(=O)O", # aspirin
		3: "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", # testosterone
		4: "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O", # geddic acid
		5: "CSCCC(C(=O)O)N", # methionine, S=2
		6: "C1=CC(=CC=C1N)S(=O)(=O)N", # sulfanilamide, S=6
		7: "COP(C1=CC=CC=C1)C2=CC=CC=C2", # methoxydiphenylphosphine, P=3
		8: "CC1C(O1)P(=O)(O)O", # phosphomycin, P=5
	}

	max_depth = args.max_depth
	dag_type = args.dag_type
	h_prior = args.h_prior
	smiles = COMPOUND_TO_SMILES[args.compound_idx]
	mol_d = frag_utils.extract_mol_info(smiles,"[M+H]+")
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(num_nodes)

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

	ccs = timeit(cython_function)(mol_d,max_depth)
	ccs = [frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False) for cc_int in ccs]
	formulae = frag_utils.compute_approximate_formulae(ccs,mol_d,h_prior=h_prior)
	print("> formulae")
	print(len(formulae))

	cc_g, other_g = timeit(cython_dag)(mol_d,max_depth,dag_type,h_prior)
	print("> cc dag")
	cc_nn = cc_g.number_of_nodes()
	cc_ne = cc_g.number_of_edges()
	cc_deg = np.array(list(map(lambda x:x[1],list(cc_g.degree()))))
	print(cc_nn,cc_ne,cc_ne/(cc_nn*(cc_nn-1)))
	print(np.mean(cc_deg),np.std(cc_deg))
	print(f"> {dag_type} dag")
	other_nn = other_g.number_of_nodes()
	other_ne = other_g.number_of_edges()
	other_deg = np.array(list(map(lambda x:x[1],list(other_g.degree()))))
	print(other_nn,other_ne,other_ne/(other_nn*(other_nn-1)))
	print(np.mean(other_deg),np.std(other_deg))
	# print(list(map(frag_utils.get_formula_string,formulae)))
