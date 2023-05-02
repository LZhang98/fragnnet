import numpy as np
import torch as th
import torch.nn.functional as F
import torch_scatter as th_s
import time
from pprint import pprint, pformat
import argparse
import networkx as nx
import dgl
from copy import deepcopy
import warnings

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.misc_utils import timeit, booltype
from frag_gnn.feat_utils import MolGraph, BatchedMolDAG
from frag_gnn.nn.nn_utils import GNN, JointPredictor, scatter_normalize
from frag_gnn.nn.loss import sparse_cross_entropy


def featurize_dag(dag):

	# cast the dag attributes to the right dtype
	nattrs = list(dag.ndata.keys())
	for nattr in nattrs:
		if nattr in ["mask"]:
			dag.ndata[nattr] = dag.ndata[nattr].float()
		else:
			dag.ndata[nattr] = dag.ndata[nattr].long()
	eattrs = list(dag.edata.keys())
	for eattr in eattrs:
		if eattr in ["mask"]:
			dag.edata[eattr] = dag.edata[eattr].float()
		else:
			dag.edata[eattr] = dag.edata[eattr].long()
	dag = dag.astype(th.int64)
	print(dag)
	return dag

def featurize_mol(mol):

	mg = MolGraph(pe_embed_k=0)
	mol_dgl = mg.get_dgl_graph(mol,bigraph=True)
	return mol_dgl


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--compound_idx",type=int,default=2)
	parser.add_argument("--max_depth",type=int,default=3)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--vectorized",type=booltype,default=False)
	parser.add_argument("--device_id",type=int,choices=[-1,0],default=-1)
	parser.add_argument("--batch_size",type=int,default=2)
	args = parser.parse_args()

    # this is a temporary workaround for dgl bug
	warnings.filterwarnings("ignore",category=UserWarning)

	np.random.seed(420)
	th.manual_seed(420)

    # mol_id=22041, smiles=O=C(Nc1nc(-c2ccccc2)cs1)c1ccncc1, wrecall=0.9814, depth=3
    # mol_id=7602, smiles=CCCCCCCCC=CCCCCCCCC(=O)NCC, wrecall=0.9947, depth=3
    # mol_id=11009, smiles=CNc1nc(C)c(C)c(OC(=O)N(C)C)n1, wrecall=0.9614, depth=3

    IDX_TO_SMILES = {
        0: "C1=CC=C(C=C1)C2=CSC(=N2)NC(=O)C3=CC=NC=C3",
        1: "CCCCCCCCC=CCCCCCCCC(=O)NCC",
        2: "CC1=C(N=C(N=C1OC(=O)N(C)C)NC)C",
    }

    IDX_TO_SPEC = {}

	max_depth = args.max_depth
	h_prior = args.h_prior
	max_h_transfer = args.max_h_transfer
	vectorized = args.vectorized
	frag_max_time = None
	smiles = IDX_TO_SMILES[args.compound_idx]
	spec = IDX_TO_SPEC[args.compound_idx]

    mol_d = frag_utils.extract_mol_info(smiles)
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(num_nodes)

	if args.device_id == 0:
		device = th.device("cuda:0")
	else:
		device = th.device("cpu")

	# prepare data
	batch_size = args.batch_size
	dag_d = timeit(frag_utils.compute_dags)(mol_d,max_depth,h_prior,max_h_transfer,frag_max_time)
	dag_dgl = featurize_dag(dag_d["dag"])
	mol_dgl = featurize_mol(mol_d["mol"])
	mol_dgl = mol_dgl.to(device)
	mol_dgls = [deepcopy(mol_dgl) for i in range(batch_size)]
	dag_dgls = [deepcopy(dag_dgl) for i in range(batch_size)]
	formula_peak_mzses = [dag_d["formula_peak_mzs"].detach().clone() for i in range(batch_size)]
	formula_peak_probses = [dag_d["formula_peak_probs"].detach().clone() for i in range(batch_size)]
	batched_dag = BatchedMolDAG(
		mol_dgls,
		dag_dgls,
		formula_peak_mzses,
		formula_peak_probses,
		device=th.device("cpu")
	)
	batched_dag = batched_dag.to(device)
	print(batched_dag)
	# print(batched_dag.get_batched_mask("formula_peak_mzs"))

	# prepare models
	hidden_size = 256
	mol_model = GNN(
		hidden_size,
		gnn_node_feats=mol_dgl.ndata["h"].shape[1],
		gnn_edge_feats=mol_dgl.edata["e"].shape[1],
	)
	mol_model.to(device)

	# make predictions
	mol_pred = mol_model(mol_dgl)
	# simple test of differentiability
	loss = th.sum(mol_pred)
	loss.backward()

	mol_kwargs = {
		"hidden_size": hidden_size,
		"gnn_node_feats": mol_dgl.ndata["h"].shape[1],
		"gnn_edge_feats": mol_dgl.edata["e"].shape[1],
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

	pred_mzs, pred_logprobs, pred_batch_idxs = joint_model(batched_dag)
	print(pred_logprobs)
	print(pred_logprobs.mean())
	# # simple test of differentiability
	# th.sum(pred_logprobs).backward()
	# select random subset of pred peaks
	true_mask = th.rand_like(pred_mzs)<0.1
	true_batch_idxs = pred_batch_idxs[true_mask]
	# add slight mass jitter
	true_mzs = pred_mzs[true_mask] + 0.0002*th.rand_like(pred_mzs[true_mask]) - 0.0001
	# random intensities
	true_logprobs = scatter_normalize(th.log(th.rand_like(pred_logprobs[true_mask])),true_batch_idxs,dim=0)
	# calculate loss
	loss = sparse_cross_entropy(
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
	print(loss)
	loss.backward()

	# get memory usage
	if th.cuda.is_available():
		cuda_max_memory = th.cuda.max_memory_allocated(device=th.device("cuda:0"))/1e9
		print(f"> GPU memory: {cuda_max_memory:.2f} GB")
