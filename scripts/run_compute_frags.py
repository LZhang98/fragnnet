import numpy as np
import argparse
import torch_geometric as pyg
import pstats, cProfile
from pprint import pprint

import frag_gnn.utils.frag_utils as frag_utils
from frag_gnn.utils.misc_utils import timeit, booltype, get_pyg_memory_usage
from frag_gnn.utils.feat_utils import get_frag_graph

def run_frag(smiles, **kwargs):
	
	pprint(kwargs)
	
	mol_d = frag_utils.extract_mol_info(smiles)
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(f"> num heavy atoms {num_nodes}")
	dag_d = {}
	dag_d = timeit(frag_utils.compute_dags)(mol_d,**kwargs)
	# try:
	# 	dag_d = timeit(frag_utils.compute_dags)(mol_d,**kwargs)
	# except Exception as e:
	# 	print("> error when computing dag")
	# 	print(f'{e}')

	dag = dag_d["dag"]
	
	print("> dag")
	dag_pyg = dag_d["dag"]
	dag_nn = dag_d["dag_num_nodes"]
	dag_ne = dag_d["dag_num_edges"]
	dag_nn_nb = dag_d["dag_num_nodes_nb"]
	
	dag_indeg = pyg.utils.degree(dag.edge_index[1]).numpy()
	dag_outdeg = pyg.utils.degree(dag.edge_index[0]).numpy()
	dag_memory = get_pyg_memory_usage(dag)

	print(f"reached_depth = {dag_d['reached_depth']}")
	print(f"num_nodes = {dag_nn}, num_edges = {dag_ne}, edge_frac = {2*dag_ne/(dag_nn*(dag_nn-1))}")
	print(f"num_nodes_nb = {dag_nn_nb}")
	print(f"avg_indeg = {np.mean(dag_indeg)}, std_indeg = {np.std(dag_indeg)}")
	print(f"avg_outdeg = {np.mean(dag_outdeg)}, std_outdeg = {np.std(dag_outdeg)}")
	print(f"num unique forumla = {len(dag_d['idx_to_formula'])}")
	print(f"node feature size = {dag_d['node_feature_size']}, edge feature size = {dag_d['edge_feature_size']}")
	print(f"node edges by depth: {dag_d['dag_num_edges_by_depth']}")
	print(f"node node by depth: {dag_d['dag_num_nodes_by_depth']}")
	print(f"is directed = {dag_d['is_directed']}")
	print(f"memory = {dag_memory/1e6} MB")
	print(f"dag_pyg = {dag_pyg}")

	# get features
	frag_node_feats = [
		"cc",
		"base_formula",
		"depth",
		"h_formula_idx",
		"nb_iso_idx"
	]
	frag_edge_feats = [
		"cc",
		"base_formula"
	]
	bigraph = False
	dag_f_pyg = get_frag_graph(dag_pyg, frag_node_feats, frag_edge_feats, edges=edges, bigraph=bigraph)
	nb_iso_idx = frag_utils.get_node_feats(dag_f_pyg.x, dag_f_pyg.node_feat_idxs[0],"nb_iso_idx")
	import pdb; pdb.set_trace()

	return dag_d

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--compound_idx",type=int,default=2)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--isotopes",type=booltype,default=True)
	parser.add_argument("--nb_isomorphic",type=booltype,default=False)
	parser.add_argument("--b_isomorphic",type=booltype,default=False)
	parser.add_argument("--max_iterations",type=int,default=2)
	args = parser.parse_args()

	np.random.seed(420)

	IDX_TO_SMILES = {
		0: "CCO", # ethanol
		1: "c1ccccc1", # benzene
		2: "CC(=O)OC1=CC=CC=C1C(=O)O", # aspirin
		3: "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", # testosterone
		4: "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O", # geddic acid
		5: "CSCCC(C(=O)O)N", # methionine, S=2
		6: "C1=CC(=CC=C1N)S(=O)(=O)N", # sulfanilamide, S=6
		7: "COP(C1=CC=CC=C1)C2=CC=CC=C2", # methoxydiphenylphosphine, P=3
		8: "CC1C(O1)P(=O)(O)O", # phosphomycin, P=5
		9: 'CC(=O)NC1C(OC2CCC3(C)C(CCC4(C)C3CC=C3C5CC(C)(C)CCC5(C(=O)O)CCC34C)C2(C)C)OC(CO)C(O)C1O', # big one (47 atoms)
		10: 'CC(=O)NC1C(OC2OC(CC(O)C3OC(n4ccc(=O)[nH]c4=O)C(O)C3O)C(O)C(O)C2NC(=O)C=CCCCCCCCCCCCC(C)C)OC(CO)C(O)C1O', # big one (60 atoms)
		11: 'C12=C3C4=C5C6=C3C7=C1C8=C9C2=C4C1=C5C2=C6C7=C8C2=C91', #Fullerene-C20
		12: 'c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41', #	C60 fullerene
		13: 'CC(=O)OC12COC1CC(O)C1(C)C(=O)C(=O)C3C(C)C(OC(=O)C(O)C(NC(=O)c4ccccc4)c4ccccc4)CC(O)(C(OC(=O)c4ccccc4)C21)C3(C)C',
		14: 'c1ccc2[nH]nnc2c1'
	}

	smiles = IDX_TO_SMILES[args.compound_idx]
	frag_kw_d = dict(
		max_depth=args.max_depth,
		h_prior=args.h_prior,
		max_h_transfer=args.max_h_transfer,
		frag_max_time=None,
		isotopes=args.isotopes,
		nb_isomorphic=args.nb_isomorphic,
		b_isomorphic=args.b_isomorphic,
		max_iterations=args.max_iterations
	)
	dag_d = run_frag(smiles, **frag_kw_d)
	# import pdb; pdb.set_trace()

	# cProfile.runctx("run_frag(max_depth, h_prior, max_h_transfer, frag_max_time, smiles, isotopes)", globals(), locals(), "run_frag_exp.prof")
	# s = pstats.Stats("run_frag_exp.prof")
	# s.strip_dirs().sort_stats("time").print_stats()