import numpy as np
import torch as th
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import torch_scatter as th_s
import time
from pprint import pprint, pformat
import argparse
import networkx as nx
import dgl
from copy import deepcopy
import warnings
import time

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.misc_utils import timeit, booltype
from frag_gnn.feat_utils import MolGraph, BatchedMolDAG
from frag_gnn.nn.nn_utils import GNN, JointPredictor, scatter_normalize
from frag_gnn.nn.loss import sparse_cross_entropy


def featurize_dag(dag):

	print(dag)
	return dag

def featurize_mol(mol):

	mg = MolGraph(pe_embed_k=0)
	mol_pyg = mg.get_pyg_graph(mol,bigraph=True)
	return mol_pyg

def torch_time(func,*args,**kwargs):
	if th.cuda.is_available():
		start = th.cuda.Event(enable_timing=True)
		end = th.cuda.Event(enable_timing=True)
		start.record()
		result = func(*args,**kwargs)
		end.record()
		th.cuda.synchronize()
		elapsed_time = start.elapsed_time(end)
	else:
		start = time.time_ns()
		result = func(*args,**kwargs)
		end = time.time_ns()
		elapsed_time = (end-start) / 1e6
	# print(f"> func {func.__name__}: time elapsed = {elapsed_time:.2f} ms")
	print(f"> func: time elapsed = {elapsed_time:.2f} ms")
	return result

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--compound_idx",type=int,default=2)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--vectorized",type=booltype,default=False)
	parser.add_argument("--device_id",type=int,choices=[-1,0],default=-1)
	parser.add_argument("--batch_size",type=int,default=2)
	parser.add_argument("--jit",type=booltype,default=False)
	parser.add_argument("--num_iters",type=int,default=10)
	args = parser.parse_args()


	np.random.seed(420)
	th.manual_seed(420)

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
		10: 'CC(=O)NC1C(OC2OC(CC(O)C3OC(n4ccc(=O)[nH]c4=O)C(O)C3O)C(O)C(O)C2NC(=O)C=CCCCCCCCCCCCC(C)C)OC(CO)C(O)C1O' # big one (60 atoms)
	}

	max_depth = args.max_depth
	h_prior = args.h_prior
	max_h_transfer = args.max_h_transfer
	vectorized = args.vectorized
	frag_max_time = None
	smiles = IDX_TO_SMILES[args.compound_idx]
	mol_d = frag_utils.extract_mol_info(smiles)
	num_nodes = mol_d["atom_mask_arr"].shape[0]

	if args.device_id == 0 and th.cuda.is_available():
		device = th.device("cuda:0")
	else:
		device = th.device("cpu")

	# prepare data
	batch_size = args.batch_size
	dag_d = timeit(frag_utils.compute_dags)(mol_d,max_depth,h_prior,max_h_transfer,frag_max_time)
	dag_pyg = featurize_dag(dag_d["dag"])
	mol_pyg = featurize_mol(mol_d["mol"])
	mol_pyg = mol_pyg.to(device)
	mol_pygs = [deepcopy(mol_pyg) for i in range(batch_size)]
	dag_pygs = [deepcopy(dag_pyg) for i in range(batch_size)]
	formula_peak_mzses = [dag_d["formula_peak_mzs"].detach().clone() for i in range(batch_size)]
	formula_peak_probses = [dag_d["formula_peak_probs"].detach().clone() for i in range(batch_size)]
	dag_node_feat_idxs = dag_d["node_feat_idxs"]
	dag_edge_feat_idxs = dag_d["edge_feat_idxs"]
	batched_dag = BatchedMolDAG(
		mol_pygs,
		dag_pygs,
		formula_peak_mzses,
		formula_peak_probses,
		dag_node_feat_idxs,
		dag_edge_feat_idxs,
		device=th.device("cpu")
	)
	batched_dag = batched_dag.to(device)
	print(batched_dag)
	# print(batched_dag.get_batched_mask("formula_peak_mzs"))
	batched_dag_d = batched_dag.__dict__()

	# # prepare models
	hidden_size = 256
	# mol_model = GNN(
	# 	hidden_size,
	# 	gnn_node_feats=mol_pyg.x.shape[1],
	# 	gnn_edge_feats=mol_pyg.edge_attr.shape[1],
	# )
	# mol_model.to(device)

	# # make predictions
	# mol_pred = mol_model(mol_pyg)
	# # simple test of differentiability
	# loss = th.sum(mol_pred)
	# loss.backward()

	mol_kwargs = {
		"hidden_size": hidden_size,
		"gnn_node_feats": mol_pyg.x.shape[1],
		"gnn_edge_feats": mol_pyg.edge_attr.shape[1],
		"mpnn_type": "GINE"
	}
	dag_kwargs = {
		"hidden_size": hidden_size,
		"mpnn_type": "GINE"
		# node/edge feats are inferred
	}
	mlp_kwargs = {
		"hidden_size": hidden_size,
		"dropout": 0.0,
		"num_layers": 2,
		"use_residuals": True
		# input/output sizes inferred
	}
	num_depth = dag_d["max_depth"]+1 # +1 for the root/precursor node
	num_hs = dag_d["max_h_transfer"]
	num_elements = frag_utils.NUM_ELEMENTS
	int_embedder = "abs-sines"
	joint_model = JointPredictor(
		num_depth,
		num_hs,
		num_elements,
		mol_kwargs,
		dag_kwargs,
		mlp_kwargs,
		int_embedder
	)
	joint_model.to(device)
	if args.jit:
		# import pdb; pdb.set_trace()
		_joint_model = th.jit.trace_module(
			joint_model,
			{"forward": deepcopy(batched_dag_d)},
			example_inputs_is_kwarg=True
		)
		joint_model_forward = _joint_model.forward
		# joint_model_forward = th.jit.trace(
		# 	joint_model.forward,
		# 	example_kwarg_inputs=deepcopy(batched_dag_d)
		# )
	else:
		joint_model_forward = joint_model.forward

	optim = th.optim.SGD(joint_model.parameters(), lr=0.)
	optim.zero_grad()

	for i in range(args.num_iters):
		print(f">> iter {i}")
		# pred_mzs, pred_logprobs, pred_batch_idxs = joint_model(batched_dag)
		pred_mzs, pred_logprobs, pred_batch_idxs = torch_time(
			joint_model_forward,
			**deepcopy(batched_dag_d)
		)
		# print(f"> avg of predicted logprobs = {pred_logprobs.mean():.4f}")
		# select random subset of pred peaks
		true_mask = th.rand_like(pred_mzs)<0.1
		true_batch_idxs = pred_batch_idxs[true_mask]
		# add slight mass jitter
		true_mzs = pred_mzs[true_mask] + 0.0002*th.rand_like(pred_mzs[true_mask]) - 0.0001
		# random intensities
		true_logprobs = scatter_normalize(th.log(th.rand_like(pred_logprobs[true_mask])),true_batch_idxs,dim=0)
		# calculate loss
		loss = torch_time(
			sparse_cross_entropy,
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			batch_size,
			variance=1e-3,
			vectorized=vectorized
		)
		# test of differentiability
		# print(f"> loss = {loss:.4f}")
		loss.backward()
		# zero out gradients
		optim.zero_grad()
	
	# get memory usage
	if args.device_id != -1 and th.cuda.is_available():
		cuda_max_memory = th.cuda.max_memory_allocated(device=th.device("cuda:0"))/1e9
		print(f"> GPU memory: {cuda_max_memory:.2f} GB")
