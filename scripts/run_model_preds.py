import numpy as np
import torch as th
import torch.nn.functional as F
import time
from pprint import pprint, pformat
import argparse
import networkx as nx
import dgl

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.misc_utils import timeit, booltype
from frag_gnn.mol_graph import MolGraph
from frag_gnn.nn.nn_utils import GNN, JointPredictor

class BatchedMolDAG:

	def __init__(self,mol_dgl_list,dag_dgl_list,dag_tensor_dict,device=None):

		assert len(mol_dgl_list) == len(dag_dgl_list), (len(mol_dgl_list),len(dag_dgl_list))
		self.batch_size = len(mol_dgl_list)
		if device is not None:
			self.device = device
		else:
			self.device = th.device("cpu")
		self.batched_mol_dgl = dgl.batch(mol_dgl_list)
		self.batched_dag_dgl = dgl.batch(dag_dgl_list)
		self.batched_mol_nodes, self.batched_dag_nodes = self._compute_batched_node_offsets(mol_dgl_list,dag_dgl_list)
		self.batched_tensor_dict = {}
		self.batched_size_dict = {}
		self.batched_cumsize_dict = {}
		self.batched_max_dict = {}
		for k,v in dag_tensor_dict.items():
			assert len(v) == self.batch_size, (len(v),self.batch_size)
			max_dim = max([dag_tensor.shape[0] for dag_tensor in v])
			self.batched_max_dict[k] = max_dim
			batched_tensor, batched_size = [], []
			for dag_tensor in v:
				dag_tensor_pad = F.pad(dag_tensor,(0,0,0,max_dim-dag_tensor.shape[0]),"constant",0)
				batched_tensor.append(dag_tensor_pad)
				dag_size = dag_tensor.shape[0]
				batched_size.append(dag_size)
			batched_tensor = th.stack(batched_tensor,dim=0).to(self.device)
			batched_size = th.tensor(batched_size,device=self.device)
			batched_cumsize = th.cumsum(th.tensor([0]+list(batched_size),device=self.device),dim=0)
			self.batched_tensor_dict[k] = batched_tensor
			self.batched_size_dict[k] = batched_size
			self.batched_cumsize_dict[k] = batched_cumsize

	def _compute_batched_node_offsets(self,mol_dgl_list,dag_dgl_list):

		# import pdb; pdb.set_trace()
		num_mol_nodes = [g.number_of_nodes() for g in mol_dgl_list]
		num_dag_nodes = [g.number_of_nodes() for g in dag_dgl_list]
		batched_mol_nodes = th.cumsum(th.tensor([0]+num_mol_nodes,device=self.device),dim=0)
		batched_dag_nodes = th.cumsum(th.tensor([0]+num_dag_nodes,device=self.device),dim=0)
		return batched_mol_nodes, batched_dag_nodes

	def get_batched_dgls(self):

		return self.batched_mol_dgl, self.batched_dag_dgl

	def get_batched_nodes(self):

		return self.batched_mol_nodes, self.batched_dag_nodes

	def get_batched_tensor(self,key):
		
		return self.batched_tensor_dict[key]

	def get_batched_mask(self,key):

		max_dim = self.batched_max_dict[key]
		batched_size = self.batched_size_dict[key]
		batched_mask = th.arange(0,max_dim,device=self.device).expand(self.batch_size,max_dim)
		batched_mask = (batched_mask < batched_size.unsqueeze(1)).float()
		return batched_mask

	def get_batched_size(self,key):

		return self.batched_size_dict[key]

	def get_batched_cumsize(self,key):

		return self.batched_cumsize_dict[key]

	def to(self,device):

		self.device = device
		for k in list(self.batched_dgl_dict.keys()):
			self.batched_dgl_dict[k] = self.batched_dgl_dict[k].to(device)
		for k in list(self.batched_tensor_dict.keys()):
			self.batched_tensor_dict[k] = self.batched_tensor_dict[k].to(device)
			self.batched_size_dict[k] = self.batched_size_dict[k].to(device)
		return self

	def __str__(self):

		tensor_keys = list(self.batched_tensor_dict.keys())
		str_format = f"BatchedDAG( batch_size={self.batch_size}, device={self.device}, tensor_keys={tensor_keys})"
		return str_format

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
		9: 'CC(=O)NC1C(OC2CCC3(C)C(CCC4(C)C3CC=C3C5CC(C)(C)CCC5(C(=O)O)CCC34C)C2(C)C)OC(CO)C(O)C1O', # big one (47 atoms)
		10: 'CC(=O)NC1C(OC2OC(CC(O)C3OC(n4ccc(=O)[nH]c4=O)C(O)C3O)C(O)C(O)C2NC(=O)C=CCCCCCCCCCCCC(C)C)OC(CO)C(O)C1O' # big one (60 atoms)
	}

	max_depth = args.max_depth
	h_prior = args.h_prior
	max_h_transfer = args.max_h_transfer
	frag_max_time = None
	smiles = COMPOUND_TO_SMILES[args.compound_idx]
	mol_d = frag_utils.extract_mol_info(smiles)
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	print(num_nodes)

	# prepare data
	dag_d = timeit(frag_utils.compute_dags)(mol_d,max_depth,h_prior,max_h_transfer,frag_max_time)
	dag_dgl = featurize_dag(dag_d["dag"])
	mol_dgl = featurize_mol(mol_d["mol"])
	batched_dag = BatchedMolDAG(
		[mol_dgl,mol_dgl],
		[dag_dgl,dag_dgl],
		{
			"formula_peak_mzs": [dag_d["formula_peak_mzs"],dag_d["formula_peak_mzs"]],
			"formula_peak_probs": [dag_d["formula_peak_probs"],dag_d["formula_peak_probs"]]
		},
		device=th.device("cpu")
	)
	print(batched_dag)
	# print(batched_dag.get_batched_mask("formula_peak_mzs"))

	# prepare models
	hidden_size = 256
	mol_model = GNN(
		hidden_size,
		gnn_node_feats=mol_dgl.ndata["h"].shape[1],
		gnn_edge_feats=mol_dgl.edata["e"].shape[1],
	)

	# make predictions
	mol_pred = mol_model(mol_dgl)
	print(f"mol_pred shape: {mol_pred.shape}")

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
	num_depth = dag_d["max_depth"]
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

	joint_pred = joint_model(batched_dag)
	print(joint_pred)