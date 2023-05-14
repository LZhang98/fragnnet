import numpy as np
import argparse
import torch_geometric as pyg

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.utils.misc_utils import timeit, booltype, get_pyg_memory_usage

import pstats, cProfile

def run_frag(max_depth, h_prior, max_h_transfer, frag_max_time, smiles):
	mol_d = frag_utils.extract_mol_info(smiles)
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(num_nodes)

	dag_d = timeit(frag_utils.compute_dags)(mol_d,max_depth,h_prior,max_h_transfer,frag_max_time)
	dag = dag_d["dag"]
	
	print("> dag")
	#frag_utils.save_dag_d(dag_d, smiles + '.pickle.pbz2', is_compressed = True)
	#frag_utils.save_dag_d(dag_d, smiles + '.pickle', is_compressed = False)
	dag_nn = dag_d["dag_num_nodes"]
	dag_ne = dag_d["dag_num_edges"]
	
	dag_indeg = pyg.utils.degree(dag.edge_index[1]).numpy()
	dag_outdeg = pyg.utils.degree(dag.edge_index[0]).numpy()
	print(f"num_nodes = {dag_nn}, num_edges = {dag_ne}, edge_frac = {2*dag_ne/(dag_nn*(dag_nn-1))}")
	print(f"avg_indeg = {np.mean(dag_indeg)}, std_indeg = {np.std(dag_indeg)}")
	print(f"dag_outdeg = {np.mean(dag_outdeg)}, std_outdeg = {np.std(dag_outdeg)}")
	print(f"node feature size = {dag_d['node_feature_size']}, edge featuer size = {dag_d['edge_featuer_size']}")
	print(f"is diectred = {dag_d['is_directed']}")
	dag_memory = get_pyg_memory_usage(dag)
	print(f"memory = {dag_memory/1e6} MB")

	return dag_d

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--compound_idx",type=int,default=2)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
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
		13: 'CC(=O)OC12COC1CC(O)C1(C)C(=O)C(=O)C3C(C)C(OC(=O)C(O)C(NC(=O)c4ccccc4)c4ccccc4)CC(O)(C(OC(=O)c4ccccc4)C21)C3(C)C'	
	}

	max_depth = args.max_depth
	h_prior = args.h_prior
	max_h_transfer = args.max_h_transfer
	frag_max_time = None
	smiles = IDX_TO_SMILES[args.compound_idx]
	
	#dag_d = run_frag(max_depth, h_prior, max_h_transfer, frag_max_time, smiles)

	cProfile.runctx("run_frag(max_depth, h_prior, max_h_transfer, frag_max_time, smiles)", globals(), locals(), "run_frag_exp.prof")
	#s = pstats.Stats("run_frag_exp.prof")
	#s.strip_dirs().sort_stats("time").print_stats()