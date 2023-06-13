import copy
import math
import numpy as np
import scipy.sparse as sparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import scipy

from frag_gnn.nn.form_embedder import get_embedder
from frag_gnn.frag.frag_utils import get_node_feats, get_edge_feats, th_long_to_mask
from frag_gnn.utils.misc_utils import scatter_logsumexp, scatter_lognormalize, scatter_reduce, scatter_masked_softmax
from frag_gnn.utils.feat_utils import get_mol_feats_sizes


def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_lr_scheduler(
	optimizer, lr_decay_rate: float, decay_steps: int = 5000, warmup: int = 1000
):
	"""build_lr_scheduler.

	Args:
		optimizer:
		lr_decay_rate (float): lr_decay_rate
		decay_steps (int): decay_steps
		warmup_steps (int): warmup_steps
	"""

	def lr_lambda(step):
		if step >= warmup:
			# Adjust
			step = step - warmup
			rate = lr_decay_rate ** (step // decay_steps)
		else:
			rate = 1 - math.exp(-step / warmup)
		return rate

	scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
	return scheduler


class JointPredictor(nn.Module):

	def __init__(
		self,
		num_depth: int,
		num_hs: int,
		num_elements: int,
		int_embedder: str,
		mol_node_feats: list[str],
		mol_edge_feats: list[str],
		mol_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_dropout: float,
		frag_node_feats: list[str],
		frag_edge_feats: list[str],
		frag_hidden_size: int,
		frag_num_layers: int,
		frag_gnn_type: str,
		frag_dropout: float,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool
	):

		super().__init__()
		self.num_depth = num_depth
		self.num_hs = num_hs
		self.num_elements = num_elements
		# calculate node/edge feats sizes
		self.mol_node_feats = mol_node_feats
		self.mol_edge_feats = mol_edge_feats
		self.mol_pe_embed_k = mol_pe_embed_k
		self._compute_mol_feats_sizes() 
		mol_kwargs = {
			"node_feats_size": self.mol_node_feats_size,
			"edge_feats_size": self.mol_edge_feats_size,
			"hidden_size": mol_hidden_size,
			"num_layers": mol_num_layers,
			"gnn_type": mol_gnn_type,
			"dropout": mol_dropout
		}
		self.mol_embedder = GNN(**mol_kwargs)
		self.formula_embedder = get_embedder(int_embedder)
		self.depth_embedder = get_embedder(int_embedder)
		self.frag_node_feats = frag_node_feats
		self.frag_edge_feats = frag_edge_feats
		self._compute_frag_feats_sizes()
		frag_kwargs = {
			"node_feats_size": self.frag_node_feats_size,
			"edge_feats_size": self.frag_edge_feats_size,
			"hidden_size": frag_hidden_size,
			"num_layers": frag_num_layers,
			"gnn_type": frag_gnn_type,
			"dropout": frag_dropout
		}
		self.frag_embedder = GNN(**frag_kwargs)
		mlp_kwargs = {
			"input_size": self.frag_embedder.hidden_size,
			"output_size": 2*self.num_hs+1,
			"hidden_size": mlp_hidden_size,
			"num_layers": mlp_num_layers,
			"dropout": mlp_dropout,
			"use_residuals": mlp_use_residuals
		}
		self.output_module = MLPBlocks(**mlp_kwargs)

	def _compute_mol_feats_sizes(self):
		# these features don't rely on any model parameters
		self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
			self.mol_node_feats, 
			self.mol_edge_feats, 
			self.mol_pe_embed_k
		)

	def _compute_frag_feats_sizes(self):
		# these features do depend on model parameters
		# nodes
		self.frag_node_feats_size = 0
		if "cc" in self.frag_node_feats:
			self.frag_node_feats_size += self.mol_embedder.hidden_size
		if "base_formula" in self.frag_node_feats:
			self.frag_node_feats_size += self.num_elements*self.formula_embedder.num_dim
		if "depth" in self.frag_node_feats:
			self.frag_node_feats_size += self.num_depth*self.depth_embedder.num_dim
		# edges
		self.frag_edge_feats_size = 0
		if "cc" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.mol_embedder.hidden_size
		if "base_formula" in self.frag_edge_feats:
			self.frag_edge_feats_size += self.num_elements*self.formula_embedder.num_dim

	def compile_submodules(self,**kwargs):
		""" this doesn't work right now """
		self.mol_embedder = pyg.compile(self.mol_embedder,**kwargs)
		self.formula_embedder = th.compile(self.formula_embedder,**kwargs)
		self.depth_embedder = th.compile(self.depth_embedder,**kwargs)
		self.frag_embedder = pyg.compile(self.frag_embedder,**kwargs)

	def forward(
		self, 
		mol_pyg: pyg.data.Data,
		frag_pyg: pyg.data.Data,
		mol_num_nodes: th.Tensor,
		frag_num_nodes: th.Tensor,
		frag_formula_peak_idxs: th.Tensor,
		frag_formula_peak_mzs: th.Tensor,
		frag_formula_peak_probs: th.Tensor,
		frag_formula_sizes: th.Tensor,
		frag_formula_cumsizes: th.Tensor,
		frag_formula_peak_sizes: th.Tensor,
		**kwargs
	):

		# TODO add collision energy support
		
		device = mol_num_nodes.device

		mol_x, mol_edge_index, mol_edge_attr = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr
		frag_x, frag_edge_index, frag_edge_attr = frag_pyg.x, frag_pyg.edge_index, frag_pyg.edge_attr
		frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
		frag_edge_feat_idxs = frag_pyg.edge_feat_idxs[0]

		# get per-atom embeddings
		mol_embed = self.mol_embedder(
			mol_x,
			mol_edge_index,
			mol_edge_attr
		)
		
		# process dag
		frag_ndata, frag_edata = [], []
		batch_frag_num_nodes = frag_x.shape[0]
		batch_frag_num_edges = frag_edge_index.shape[1]
		# node atom embeddings
		if "cc" in self.frag_node_feats:
			frag_node_mask = th_long_to_mask(get_node_feats(frag_x,frag_node_feat_idxs,"cc"))
			frag_node_mask_idxs = th.nonzero(frag_node_mask).long()
			frag_node_offsets = mol_num_nodes[th.bucketize(frag_node_mask_idxs[:,0],frag_num_nodes,right=True)-1]
			frag_node_mask_idxs[:,1] = frag_node_mask_idxs[:,1] + frag_node_offsets
			frag_node_mask_embed = scatter_reduce(
				src=mol_embed[frag_node_mask_idxs[:,1]],
				index=frag_node_mask_idxs[:,0:1].expand(-1,mol_embed.shape[1]),
				reduce="sum",
				dim=0,
				dim_size=batch_frag_num_nodes
			)
			frag_ndata.append(frag_node_mask_embed)
		# edge atom embeddings
		if "cc" in self.frag_edge_feats:
			frag_edge_mask = th_long_to_mask(get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"cc"))
			frag_edge_mask_idxs = th.nonzero(frag_edge_mask).long()
			frag_edge_node_idxs = frag_edge_index[0][frag_edge_mask_idxs[:,0]]
			frag_edge_offsets = mol_num_nodes[th.bucketize(frag_edge_node_idxs,frag_num_nodes,right=True)-1]
			frag_edge_mask_idxs[:,1] = frag_edge_mask_idxs[:,1] + frag_edge_offsets
			frag_edge_mask_embed = scatter_reduce(
				src=mol_embed[frag_edge_mask_idxs[:,1]],
				index=frag_edge_mask_idxs[:,0:1].expand(-1,mol_embed.shape[1]),
				reduce="sum",
				dim=0,
				dim_size=batch_frag_num_edges
			)
			frag_edata.append(frag_edge_mask_embed)
		# node formulae
		if "base_formula" in self.frag_node_feats:
			frag_node_formula = self.formula_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"base_formula").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_node_formula)
		# edge formulae
		if "base_formula" in self.frag_edge_feats:
			frag_edge_formula = self.formula_embedder(
				get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"base_formula").reshape(batch_frag_num_edges,-1)
			)
			frag_edata.append(frag_edge_formula)
		# node depth
		if "depth" in self.frag_node_feats:
			frag_depth = self.depth_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"depth").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_depth)
		
		# get output formula aggregation 
		frag_formula_batch_idxs = th.bucketize(th.arange(frag_x.shape[0],device=device),frag_num_nodes,right=True)-1
		frag_formula_offsets = frag_formula_cumsizes[frag_formula_batch_idxs]
		frag_formula_idxs = (get_node_feats(frag_x,frag_node_feat_idxs,"h_formulae_idx")+frag_formula_offsets.unsqueeze(-1)).flatten()        
		
		# concatenate everything 
		frag_x = th.cat(frag_ndata,dim=-1)
		# concatenate everything 
		frag_edge_attr = th.cat(frag_edata,dim=-1)
		# apply GNN
		frag_embed = self.frag_embedder(
			frag_x,
			frag_edge_index,
			frag_edge_attr
		)

		# predict probabilities
		frag_formula_logits = self.output_module(frag_embed)

		# convert to logprobs
		frag_formula_logits = frag_formula_logits.flatten()
		frag_formula_logprobs = scatter_lognormalize(
			frag_formula_logits,
			frag_formula_idxs,
			dim=0
		)
		# aggregate by formula
		frag_formula_logprobs = scatter_logsumexp(
			frag_formula_logprobs,
			frag_formula_idxs,
			dim=0
		)

		# zero out NULL formula
		frag_formula_mask = th.ones_like(frag_formula_logprobs)
		frag_formula_mask[frag_formula_cumsizes[:-1]] = 0.
		frag_formula_batch_idxs = th.bucketize(
			th.arange(frag_formula_cumsizes[-1],device=device),
			frag_formula_cumsizes,
			right=True
		)-1
		
		# softmax with masked logits
		frag_formula_logprobs = scatter_masked_softmax(
			frag_formula_logprobs,
			frag_formula_mask,
			frag_formula_batch_idxs,
			dim=0,
			log=True
		)
		# print("frag_formula_logprobs",scatter_logsumexp(frag_formula_logprobs,frag_formula_batch_idxs,dim=0))

		# convert to spectrum
		spec_mzs = frag_formula_peak_mzs
		spec_logprobs = frag_formula_logprobs[frag_formula_peak_idxs] + th.log(frag_formula_peak_probs)

		# get batch idxs
		spec_batch_idxs = th.repeat_interleave(
			th.arange(frag_formula_peak_sizes.shape[0],device=device),
			frag_formula_peak_sizes
		)
		
		# log normalize
		# print("spec_un",scatter_logsumexp(spec_logprobs,spec_batch_idxs,dim=0))
		spec_logprobs = scatter_lognormalize(
			spec_logprobs,
			spec_batch_idxs,
			dim=0
		)
		# print("spec_n",scatter_logsumexp(spec_logprobs,spec_batch_idxs,dim=0))
		if th.any(th.abs(scatter_logsumexp(spec_logprobs,spec_batch_idxs,dim=0))>1.):
			import pdb; pdb.set_trace() 

		return spec_mzs, spec_logprobs, spec_batch_idxs

class GNN(nn.Module):

	def __init__(
		self,
		hidden_size: int,
		num_layers: int,
		node_feats_size: int,
		edge_feats_size: int,
		gnn_type: str,
		dropout: float,
		**kwargs
	):
		"""__init__.
		Args:
			hidden_size (int): Hidden size
			num_mol_layers (int): Number of layers to encode for the molecule
		"""
		super().__init__()
		self.hidden_size = hidden_size
		self.edge_feats_size = edge_feats_size
		self.node_feats_size = node_feats_size
		self.dropout = dropout

		self.gnn_type = gnn_type
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.input_project = nn.Linear(self.node_feats_size, self.hidden_size)

		if self.gnn_type == "GINE":
			self.gnn = GINE(
				hidden_size=self.hidden_size,
				edge_feats_size=self.edge_feats_size,
				node_feats_size=self.node_feats_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
			)
		else:
			raise NotImplementedError(self.gnn_type)

	def forward(self, x, edge_index, edge_attr):
		"""encode batch of molecule graph"""
		x = self.input_project(x)
		x = self.gnn(x, edge_index, edge_attr)
		return x


class GINE(nn.Module):
	def __init__(
		self,
		hidden_size: int,
		edge_feats_size: int,
		num_layers: int,
		dropout: float,
		**kwargs
	):
		"""GINE.

		Args:
			input_size (int): Size of edge features into the graph
			hidden_size (int): Hidden size
			edge_feats (int): Number of edge feats. Must be onehot!
			node_feats (int): Num of node feats (default 74)
			num_step_message_passing (int): Number of message passing steps
			dropout
		"""
		super().__init__()

		self.edge_transform = nn.Linear(edge_feats_size, hidden_size)

		self.layers = []
		for i in range(num_layers):
			apply_fn = nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size),
			)
			temp_layer = pyg.nn.conv.GINEConv(
				nn=apply_fn,
				eps=0.
			)
			self.layers.append(temp_layer)

		self.layers = nn.ModuleList(self.layers)
		self.bnorms = get_clones(nn.BatchNorm1d(hidden_size), num_layers)
		self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

	def forward(self, x, edge_index, edge_attr):
		"""forward.
		"""
		edge_attr = self.edge_transform(edge_attr)

		for dropout, layer, norm in zip(self.dropouts, self.layers, self.bnorms):
			layer_out = layer(x, edge_index, edge_attr)
			x = F.relu(dropout(norm(layer_out))) + x

		return x


class MLPBlocks(nn.Module):
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		dropout: float,
		num_layers: int,
		output_size: int = None,
		use_residuals: bool = False,
	):
		super().__init__()
		self.activation = nn.ReLU()
		self.dropout_layer = nn.Dropout(p=dropout)
		self.input_layer = nn.Linear(input_size, hidden_size)
		middle_layer = nn.Linear(hidden_size, hidden_size)
		self.layers = get_clones(middle_layer, num_layers - 1)

		self.output_layer = None
		self.output_size = output_size
		if self.output_size is not None:
			self.output_layer = nn.Linear(hidden_size, self.output_size)

		self.use_residuals = use_residuals

	def forward(self, x):
		output = x
		output = self.input_layer(x)
		output = self.dropout_layer(output)
		output = self.activation(output)
		old_op = output
		for layer_index, layer in enumerate(self.layers):
			output = layer(output)
			output = self.dropout_layer(output)
			output = self.activation(output)
			if self.use_residuals:
				output = output + old_op
				old_op = output

		if self.output_layer is not None:
			output = self.output_layer(output)
		return output


def random_walk_pe(g, k, eweight_name=None):
	"""Random Walk Positional Encoding, as introduced in
	`Graph Neural Networks with Learnable Structural and Positional Representations
	<https://arxiv.org/abs/2110.07875>`__

	This function computes the random walk positional encodings as landing probabilities
	from 1-step to k-step, starting from each node to itself.

	Parameters
	----------
	g : DGLGraph
		The input graph. Must be homogeneous.
	k : int
		The number of random walk steps. The paper found the best value to be 16 and 20
		for two experiments.
	eweight_name : str, optional
		The name to retrieve the edge weights. Default: None, not using the edge weights.

	Returns
	-------
	Tensor
		The random walk positional encodings of shape :math:`(N, k)`, where :math:`N` is the
		number of nodes in the input graph.

	Example
	-------
	>>> import dgl
	>>> g = dgl.graph(([0,1,1], [1,1,0]))
	>>> dgl.random_walk_pe(g, 2)
	tensor([[0.0000, 0.5000],
			[0.5000, 0.7500]])
	"""

	N = g.num_nodes  # number of nodes
	M = g.num_edges  # number of edges
	# sparse adjacency matrix
	A = scipy.sparse.csr_matrix(
		(
			th.ones_like(g.edge_index[0]).numpy(),
			(
				g.edge_index[0].numpy(),
				g.edge_index[1].numpy()
			)
		)
	)
	RW = A / (A.sum(1).reshape(-1,1) + 1e-30)  # 1-step transition probability

	# Iterate for k steps
	PE = [th.as_tensor(RW).diagonal()]
	RW_power = th.as_tensor(RW)
	for _ in range(k - 1):
	    RW_power = RW_power @ RW
	    PE.append(RW_power.diagonal().clone())
	PE = th.stack(PE, dim=-1)
	return PE

def scatter_masked_softmax(logits,mask,subset_idxs,mask_value=np.log(1e-4),dim=-1,log=True):
	# # clamp
	# logits = th.clamp(logits,min=np.log(1e-4))
	# apply mask
	masked_logits = mask*logits + (1-mask)*mask_value
	# normalize
	masked_logits = scatter_lognormalize(masked_logits,subset_idxs,dim)
	if not log:
		# exponentiate
		return th.exp(masked_logits)
	else:
		return masked_logits

if __name__ == "__main__":

	edge_index = th.tensor([[0,1,1],[1,1,0]],dtype=th.long)
	x = th.tensor([0,1],dtype=th.float)
	g = pyg.data.Data(x=x,edge_index=edge_index)
	print(g)
	rw_g = random_walk_pe(g,2)
	print(rw_g)
