import numpy as np
import time
from pprint import pprint
import argparse
import networkx as nx

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.misc_utils import timeit, booltype


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--compound_idx",type=int,default=2)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
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
	h_prior = args.h_prior
	max_h_transfer = args.max_h_transfer
	smiles = COMPOUND_TO_SMILES[args.compound_idx]
	mol_d = frag_utils.extract_mol_info(smiles)
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(num_nodes)

	ccs = timeit(frag_utils.compute_frags_cython)(mol_d,max_depth,depths=False)
	ccs = [frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False) for cc_int in ccs]
	formulae = frag_utils.compute_approximate_formulae(ccs,mol_d,h_prior=h_prior,max_h_transfer=max_h_transfer,return_counts=True)
	counts = np.array(list(formulae.values()))
	print("> formulae")
	print(len(formulae))
	print(np.mean(counts),np.std(counts))

	dag_d = timeit(frag_utils.compute_dags)(mol_d,max_depth,h_prior,max_h_transfer)
	cc_g = dag_d["cc_g"]
	print("> cc dag")
	cc_nn = cc_g.number_of_nodes()
	cc_ne = cc_g.number_of_edges()
	cc_deg = np.array(list(map(lambda x:x[1],list(cc_g.degree()))))
	is_dag = nx.is_directed_acyclic_graph(cc_g)
	print(f"is_dag: {is_dag}")
	print(f"num_nodes = {cc_nn}, num_edges = {cc_ne}, edge_frac = {cc_ne/(cc_nn*(cc_nn-1))}")
	print(f"avg_deg = {np.mean(cc_deg)}, std_deg = {np.std(cc_deg)}")
