import copy
import math
import numpy as np
import scipy.sparse as sparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import torch_geometric as pyg
import scipy
import torch_geometric
from torch_geometric.profile import get_gpu_memory_from_gc, get_gpu_memory_from_nvidia_smi
import logging
from typing import Literal

from frag_gnn.nn.form_embedder import get_embedder
from frag_gnn.frag.frag_utils import get_node_feats, get_edge_feats, th_long_to_mask
from frag_gnn.utils.misc_utils import scatter_logsumexp, scatter_logsoftmax, scatter_reduce, scatter_masked_softmax
from frag_gnn.utils.feat_utils import get_mol_feats_sizes


def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_lr_scheduler(
	optimizer, decay_rate: float, decay_steps: int = 5000, warmup_steps: int = 1000
):
	"""build_lr_scheduler.

	Args:
		optimizer:
		decay_rate (float): decay_rate
		decay_steps (int): decay_steps
		warmup_steps (int): warmup_steps
	"""

	def lr_lambda(step):
		if step >= warmup_steps:
			# Adjust
			step = step - warmup_steps
			rate = decay_rate ** (step // decay_steps)
		else:
			rate = 1 - math.exp(-step / warmup_steps)
		return rate

	scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
	return scheduler


class ZerosAggregation(nn.Module):
	""" Zero Aggreation module
		Just sums up features acroess all elements
	Args:
		nn (_type_): _description_
	"""
	def __init__(self):
		super().__init__()
		self.dummy_agg = pyg.nn.SumAggregation()

	def forward(self, x, batch):
		# sum x for the indices of elements for applying the aggregation. 
		# indices here work the same as reduce/sum
		agg = self.dummy_agg(x, batch)
		return 0.*agg


def build_pool_module(pool_type: str, node_dim: int):
	"""medthos build pooling layer for given pool_type and dim
		node dim is required for attention layer. 
		if pool type is not one of 'sum','mean','max','attention', return ZerosAggregation
	Args:
		pool_type (str): pooling type in ['sum','mean','max','attention']
		node_dim (int): node_dim, used for attention

	Returns:
		_type_: _description_
	"""

	if pool_type == "sum":
		pool_module = pyg.nn.SumAggregation()
	elif pool_type == "mean":
		pool_module = pyg.nn.MeanAggregation()
	elif pool_type == "max":
		pool_module = pyg.nn.MaxAggregation()
	elif pool_type == "attention":
		pool_module = pyg.nn.AttentionalAggregation(
			gate_nn = nn.Sequential(
				nn.Linear(node_dim, node_dim),
				nn.ReLU(),
				nn.Linear(node_dim, 1)
			)
		)
	else:
		assert pool_type == "none", pool_type
		# just return 0s
		pool_module = ZerosAggregation()
	return pool_module


class JointPredictor(nn.Module):
	"""Joint Predictor for ms prediction

	Args:
		nn (_type_): _description_
	"""

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
		mol_normalization: str,
		mol_dropout: float,
		mol_pool_type: str,
		frag_node_feats: list[str],
		frag_edge_feats: list[str],
		frag_hidden_size: int,
		frag_num_layers: int,
		frag_gnn_type: str,
		frag_normalization: str,
		frag_dropout: float,
		frag_pool_type: str,
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
			"dropout": mol_dropout,
			"normalization": mol_normalization,
		}

		# Mol GNN
		self.mol_embedder = GNN(**mol_kwargs)
		self.mol_pool = build_pool_module(mol_pool_type,mol_hidden_size)

		# interstage
		# self.interatge = th.nn.MultiheadAttention()

		# frag grahp GNN
		self.formula_embedder = get_embedder(int_embedder)
		self.depth_embedder = get_embedder(int_embedder)
		self.frag_node_feats = frag_node_feats
		self.frag_edge_feats = frag_edge_feats
		self._compute_frag_feats_sizes()

		# define frag-graph GNN to create frag-graph node level embedding
		frag_kwargs = {
			"node_feats_size": self.frag_node_feats_size,
			"edge_feats_size": self.frag_edge_feats_size,
			"hidden_size": frag_hidden_size,
			"num_layers": frag_num_layers,
			"gnn_type": frag_gnn_type,
			"dropout": frag_dropout,
			"normalization": frag_normalization
		}
		self.frag_embedder = GNN(**frag_kwargs)
		self.frag_pool = build_pool_module(frag_pool_type,frag_hidden_size)
		mlp_kwargs = {
			#input of starting node and ending nodes
			"input_size": 2*self.frag_embedder.hidden_size, 
			# num of allowed h transfer for both gain, loss and nothing
			"output_size": 2*self.num_hs+1,
			"hidden_size": mlp_hidden_size,
			"num_layers": mlp_num_layers,
			"dropout": mlp_dropout,
			"use_residuals": mlp_use_residuals
		}
		self.output_module = MLPBlocks(**mlp_kwargs)

	def _compute_mol_feats_sizes(self):
		""" method compute mol feature size
			these features don't rely on any model parameters
		"""
		self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
			self.mol_node_feats, 
			self.mol_edge_feats, 
			self.mol_pe_embed_k
		)

	def _compute_frag_feats_sizes(self):
		""" method compute frag-graph feature size
			these features do depend on model parameters
		"""
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
		""" pyg does not support dynamic shape compiling """
		# self.mol_embedder = pyg.compile(self.mol_embedder,**kwargs)
		self.formula_embedder = th.compile(self.formula_embedder,**kwargs)
		self.depth_embedder = th.compile(self.depth_embedder,**kwargs)
		# self.frag_embedder = pyg.compile(self.frag_embedder,**kwargs)

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
		"""forward methods for joint predictor

		Args:
			mol_pyg (pyg.data.Data): molecule pyg data object
			frag_pyg (pyg.data.Data): fragmentation graph pyg data object
			mol_num_nodes (th.Tensor): number of nodes in molecule graph
			frag_num_nodes (th.Tensor): number of nodes in fragmentation graph
			frag_formula_peak_idxs (th.Tensor): _description_
			frag_formula_peak_mzs (th.Tensor): _description_
			frag_formula_peak_probs (th.Tensor): _description_
			frag_formula_sizes (th.Tensor): _description_
			frag_formula_cumsizes (th.Tensor): _description_
			frag_formula_peak_sizes (th.Tensor): _description_

		Returns:
			_type_: _description_
		"""

		# TODO add collision energy support
		
		def get_mem():
			# th.cuda.empty_cache()
			# return f"{get_gpu_memory_from_nvidia_smi()[1]} MB"
			return None
		def log_msg(msg):
			# logging.info(msg)
			return None

		log_msg(f"initial: {get_mem()}")

		device = mol_num_nodes.device

		# mol_x: mol level node feature matrix
		# mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
		# edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		mol_x, mol_edge_index, mol_edge_attr, mol_batch = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr, mol_pyg.batch
		# frag_x: frag-graph level node feature matrix
		# frag_edge_index: frag graph connectivity in COO format with shape [2, num_edges]
		# frag_edge_attr: frag graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		frag_x, frag_edge_index, frag_edge_attr, frag_batch = frag_pyg.x, frag_pyg.edge_index, frag_pyg.edge_attr, frag_pyg.batch

		frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
		frag_edge_feat_idxs = frag_pyg.edge_feat_idxs[0]
		
		# get per-atom embeddings
		mol_embed_gnn = self.mol_embedder(
			mol_x,
			mol_batch,
			mol_edge_index,
			mol_edge_attr
		)
		mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn,mol_batch)
		log_msg(f"mol_embed_gnn: {get_mem()}")

		# process dag
		# create interstage
		frag_ndata, frag_edata = [], []
		batch_frag_num_nodes = frag_x.shape[0]
		batch_frag_num_edges = frag_edge_index.shape[1]
		# node atom embeddings for connected compoments
		if "cc" in self.frag_node_feats:
			frag_node_mask = th_long_to_mask(get_node_feats(frag_x,frag_node_feat_idxs,"cc"), device)
			frag_node_mask_idxs = th.nonzero(frag_node_mask).long()
			frag_node_offsets = mol_num_nodes[th.bucketize(frag_node_mask_idxs[:,0],frag_num_nodes,right=True)-1]
			frag_node_mask_idxs[:,1] = frag_node_mask_idxs[:,1] + frag_node_offsets
			frag_node_mask_embed = scatter_reduce(
				src=mol_embed_gnn[frag_node_mask_idxs[:,1]],
				index=frag_node_mask_idxs[:,0:1].expand(-1,mol_embed_gnn.shape[1]),
				reduce="sum",
				dim=0,
				dim_size=batch_frag_num_nodes
			)

			# frag_node_mask_embed can be one of following
			# 1. frag_node_mask_embed plus mol_embed_gnn_pool
			# 2. frag_node_mask_embed only, with out pooled embed for entire mol
			# 3. a linear layer between masked embed and pooled embed
			# 4 . attetion between frag embed and pooled mol embed
			frag_node_mask_embed = frag_node_mask_embed + mol_embed_gnn_pool[frag_batch]
			frag_ndata.append(frag_node_mask_embed)
		log_msg(f"frag_node_cc: {get_mem()}")

		# edge atom embeddings
		if "cc" in self.frag_edge_feats:
			# connected competents
			frag_edge_mask = th_long_to_mask(get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"cc"), device)
			frag_edge_mask_idxs = th.nonzero(frag_edge_mask).long()
			frag_edge_node_idxs = frag_edge_index[0][frag_edge_mask_idxs[:,0]]
			frag_edge_offsets = mol_num_nodes[th.bucketize(frag_edge_node_idxs,frag_num_nodes,right=True)-1]
			frag_edge_mask_idxs[:,1] = frag_edge_mask_idxs[:,1] + frag_edge_offsets
			frag_edge_mask_embed = scatter_reduce(
				src=mol_embed_gnn[frag_edge_mask_idxs[:,1]],
				index=frag_edge_mask_idxs[:,0:1].expand(-1,mol_embed_gnn.shape[1]),
				reduce="sum",
				dim=0,
				dim_size=batch_frag_num_edges
			)
			frag_edata.append(frag_edge_mask_embed)

		log_msg(f"frag_edge_cc: {get_mem()}")
		# node formulae
		if "base_formula" in self.frag_node_feats:
			frag_node_formula = self.formula_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"base_formula").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_node_formula)
		log_msg(f"frag_node_formula: {get_mem()}")
		# edge formulae
		if "base_formula" in self.frag_edge_feats:
			frag_edge_formula = self.formula_embedder(
				get_edge_feats(frag_edge_attr,frag_edge_feat_idxs,"base_formula").reshape(batch_frag_num_edges,-1)
			)
			frag_edata.append(frag_edge_formula)
		log_msg(f"frag_edge_formula: {get_mem()}")
		# node depth
		if "depth" in self.frag_node_feats:
			frag_depth = self.depth_embedder(
				get_node_feats(frag_x,frag_node_feat_idxs,"depth").reshape(batch_frag_num_nodes,-1)
			)
			frag_ndata.append(frag_depth)
		log_msg(f"frag_node_depth: {get_mem()}")
		# get output formula aggregation 
		frag_node_batch_idxs = th.bucketize(th.arange(frag_x.shape[0],device=device),frag_num_nodes,right=True)-1
		frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]
		frag_formula_idxs = (get_node_feats(frag_x,frag_node_feat_idxs,"h_formulae_idx")+frag_node_offsets.unsqueeze(-1)).flatten()

		# assmeble all features for dag
		# concatenate everything 
		frag_x = th.cat(frag_ndata,dim=-1)
		# concatenate everything 
		frag_edge_attr = th.cat(frag_edata,dim=-1)

		# define frag network
		frag_embed_gnn = self.frag_embedder(
			frag_x,
			frag_node_batch_idxs,
			frag_edge_index,
			frag_edge_attr
		)
		frag_embed_gnn_pool = self.frag_pool(frag_embed_gnn,frag_batch)
		frag_embed_node = self.frag_embedder.input_project(frag_x)
		frag_embed_node_pool = self.frag_pool(frag_embed_node,frag_batch)
		log_msg(f"frag_embed: {get_mem()}")

		# get frag dag embedding
		frag_embed = th.cat(
			[
				frag_embed_node - frag_embed_node_pool[frag_batch],
				frag_embed_gnn - frag_embed_gnn_pool[frag_batch]
			], dim=1
		)
		# predict probabilities
		frag_formula_logits = self.output_module(frag_embed)
		assert frag_formula_logits.shape[1] == 2*self.num_hs+1, (frag_formula_logits.shape[1],2*self.num_hs+1)

		# convert to logprobs
		frag_formula_batch_idxs = th.repeat_interleave(frag_node_batch_idxs,2*self.num_hs+1)
		frag_formula_logits = frag_formula_logits.flatten()
		frag_formula_logprobs = scatter_logsoftmax(
			frag_formula_logits,
			frag_formula_batch_idxs,
			dim=0
		)
		log_msg(f"frag_formula_logprobs_1: {get_mem()}")

		# aggregate by node
		frag_formula_mask = th.isin(frag_formula_idxs,frag_formula_cumsizes[:-1]).float()
		frag_node_formula_logits = (frag_formula_logprobs*frag_formula_mask + -1e9*(1-frag_formula_mask)).reshape(-1,2*self.num_hs+1)
		frag_node_logprobs = th.logsumexp(frag_node_formula_logits,dim=1)
		frag_node_logprobs = scatter_logsoftmax(
			frag_node_logprobs,
			frag_node_batch_idxs,
			dim=0
		)

		log_msg(f"frag_node_logprobs: {get_mem()}")

		# calculate conditional entropy
		frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logits,dim=1)
		# frag_node_formula_entropy = -th.sum(frag_node_formula_logprobs*th.exp(frag_node_formula_logprobs),dim=1)
		log_msg(f"frag_node_formula_logprobs: {get_mem()}")

		# aggregate by formula
		frag_formula_logprobs = scatter_logsumexp(
			frag_formula_logprobs,
			frag_formula_idxs,
			dim=0
		)
		log_msg(f"frag_formula_logprobs_2: {get_mem()}")

		# zero out NULL formula
		# TODO: could reuse masking from above
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
		log_msg(f"frag_formula_logprobs_3: {get_mem()}")

		# convert to spectrum
		frag_formula_offsets = th.repeat_interleave(
			frag_formula_cumsizes[:-1],
			frag_formula_sizes-1 # -1 is for NULL formulae
		)
		spec_mzs = frag_formula_peak_mzs
		spec_logprobs = frag_formula_logprobs[frag_formula_peak_idxs+frag_formula_offsets] + th.log(frag_formula_peak_probs)
		log_msg(f"spec_logprobs_1: {get_mem()}")

		# get batch idxs
		spec_batch_idxs = th.repeat_interleave(
			th.arange(frag_formula_peak_sizes.shape[0],device=device),
			frag_formula_peak_sizes
		)
		
		# log normalize
		spec_logprobs = scatter_logsoftmax(
			spec_logprobs,
			spec_batch_idxs,
			dim=0
		)
		log_msg(f"spec_logprobs_2: {get_mem()}")

		if th.any(th.abs(scatter_logsumexp(spec_logprobs,spec_batch_idxs,dim=0))>1.):
			import pdb; pdb.set_trace() 

		out_d = {
			"pred_mzs": spec_mzs,
			"pred_logprobs": spec_logprobs,
			"pred_batch_idxs": spec_batch_idxs,
			"pred_formula_logprobs": frag_formula_logprobs,
			"pred_formula_batch_idxs": frag_formula_batch_idxs,
			"pred_node_logprobs": frag_node_logprobs,
			"pred_node_batch_idxs": frag_node_batch_idxs,
			"pred_node_formula_logprobs": frag_node_formula_logprobs,
		}

		return out_d

class GNN(nn.Module):
	""" Generic GNN Class
	"""

	def __init__(
		self,
		hidden_size: int,
		num_layers: int,
		node_feats_size: int,
		edge_feats_size: int,
		gnn_type: str,
		dropout: float,
		normalization: str,
		**kwargs
	):
		"""_summary_

		Args:
			hidden_size (int): _description_
			num_layers (int): _description_
			node_feats_size (int): _description_
			edge_feats_size (int): _description_
			gnn_type (str): _description_
			dropout (float): _description_

		Raises:
			NotImplementedError: _description_
		"""
		super().__init__()
		self.hidden_size = hidden_size
		self.edge_feats_size = edge_feats_size
		self.node_feats_size = node_feats_size
		self.dropout = dropout
		self.normalization = normalization

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
				normalization=self.normalization
			)
		elif self.gnn_type == "GAT" or self.gnn_type == "GATv2":
			self.gnn = GAT(
				node_feats_size=self.node_feats_size,
				edge_feats_size=self.edge_feats_size,
				node_hidden_dim=self.hidden_size, 
			    num_step_message_passing =self.num_layers,
				num_step_set2set=self.num_layers,
				output_dim = self.hidden_size, # TODO this need to hook up correctly
				gat_heads = 1, # TODO this need to hook up correctly
				gat_dropout=self.dropout,
				is_v2 = True if self.gnn_type == "GATv2" else False
			)
		else:
			raise NotImplementedError(self.gnn_type)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""encode batch of molecule graph"""
		x = self.input_project(x)
		x = self.gnn(x, batch_index, edge_index, edge_attr)
		return x


class GINE(nn.Module):
	def __init__(
		self,
		hidden_size: int,
		node_feats_size: int,
		edge_feats_size: int,
		num_layers: int,
		dropout: float,
		normalization: str,
		**kwargs
	):
		"""GINE.
			Strategies for Pre-training Graph Neural Networks 
			https://arxiv.org/abs/1905.12265
		Args:
			hidden_size (int): Hidden layer size
			edge_feats (int): Number of edge feats. Must be onehot!
			num_layers (int): Number of message passing steps
			normalization (str): normalization type (batch, layer, none)
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
		# setup norm and norm_fn
		# norm_fn may or may not depend on the batch index
		if normalization == "batch":
			norm_mod = nn.BatchNorm1d(hidden_size)
			self.norm_fn = lambda x, b, n: n(x)
		elif normalization == "layer":
			norm_mod = nn.LayerNorm(hidden_size)
			self.norm_fn = lambda x, b, n: n(x)
		elif normalization == "graph":
			norm_mod = pyg.nn.GraphNorm(hidden_size)
			self.norm_fn = lambda x, b, n: n(x,b)
		else:
			assert normalization == "none", normalization
			norm_mod = nn.Identity()
			self.norm_fn = lambda x, b, n: n(x)
		self.norms = get_clones(norm_mod, num_layers)
		self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""forward.
		"""
		edge_attr = self.edge_transform(edge_attr)
		for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
			layer_out = layer(x, edge_index, edge_attr)
			layer_out = self.norm_fn(layer_out, batch_index, norm)
			x = F.relu(dropout(layer_out)) + x

		return x

class MPNN(nn.Module):
	def __init__(
		self,
		node_feats_size: int,
		edge_feats_size: int,
		output_size: int,
		node_hidden_dim=64,
		edge_hidden_dim=128,
		num_step_message_passing=6,
		num_step_set2set=6,
		**kwargs
	):
		"""Neural Message Passing for Quantum Chemistry
			https://arxiv.org/abs/1704.01212
		"""
		super().__init__()
		self.num_step_message_passing = num_step_message_passing
		self.lin0 = nn.Linear(node_feats_size, node_hidden_dim)
		edge_network = nn.Sequential(
			nn.Linear(edge_feats_size, edge_hidden_dim), nn.ReLU(),
			nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
		
		# MPNN 
		self.conv = pyg.nn.NNConv(node_hidden_dim,
			node_hidden_dim,
			edge_network,
			aggr='mean',
			root_weight=False)
		self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

		self.set2set = pyg.nn.Set2Set(node_hidden_dim,processing_steps=num_step_set2set)
		self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
		self.lin2 = nn.Linear(node_hidden_dim, output_size)
			
	def forward(self, x, batch_index, edge_index, edge_attr):
		"""forward.
		"""
		out = F.relu(self.lin0(x))
		h = out.unsqueeze(0)

		# for each setup, pass messages
		for _ in range(self.num_step_message_passing):
			m = F.relu(self.conv(out, edge_index, edge_attr))
			out, h = self.gru(m.unsqueeze(0), h)
			out = out.squeeze(0)

		out = self.set2set(out, batch_index)
		out = F.relu(self.lin1(out))
		out = self.lin2(out)
		return out

class GAT(nn.Module):
	""" Graph attention networks. https://arxiv.org/abs/1710.10903 
		Graph attention networks v2. (How Attentive are Graph Attention Networks?  https://arxiv.org/abs/2105.14491)
		With Set2Set for graph level prediction  (Order Matters: Sequence to sequence for sets) https://arxiv.org/abs/1511.06391
	"""

	def __init__(
		self,
		node_feats_size: int,
		edge_feats_size: int,
		node_hidden_dim: int,
		num_step_message_passing: int,
		num_step_set2set: int,
		output_dim: int,
		gat_heads: int = 1,
		gat_dropout: float = 0.0,
		is_v2: bool = True,
		is_concat: bool = True,
		**kwargs
	):
		"""_summary_
		Args:
			node_feats_size (int): _description_
			edge_feats_size (int): _description_
			node_hidden_dim (int): _description_
			num_step_message_passing (int): _description_
			num_step_set2set (int): _description_
			output_dim (int): _description_
			gat_heads (int, optional): _description_. Defaults to 1.
			gat_dropout (float, optional): _description_. Defaults to 0.0.
			is_v2 (bool, optional). use v2 GATConv, Defaults to True
			is_concat (bool, optional). concatenated multihead attetion, Defaults to True
		"""
		super(GAT, self).__init__()
		self.num_step_message_passing = num_step_message_passing
		self.lin0 = nn.Linear(node_feats_size, node_hidden_dim)

		gat_output_dim = node_hidden_dim//gat_heads if is_concat else node_hidden_dim

		if is_concat and node_hidden_dim % gat_heads != 0:
			raise ValueError(f"Ensure that the number of output channels of "
                             f"'GAT' (got '{node_hidden_dim}') is divisible "
                             f"by the number of heads (got '{gat_heads}')")	

		# hook up to GATCOnv
		GATConv_fn = pyg.nn.conv.GATConv if not is_v2 else pyg.nn.conv.GATv2Conv
		self.conv = GATConv_fn(node_hidden_dim, gat_output_dim, 
				heads  = gat_heads,
				edge_dim=edge_feats_size, 
				dropout=gat_dropout,
				concat=is_concat)

		if num_step_set2set:
			self.set2set = pyg.nn.Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
			# output of set2set is twice the dimensionality as the input
			self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
			self.lin2 = nn.Linear(node_hidden_dim, output_dim)
		else:
			self.set2set = None
			self.lin1 =  None
			self.lin2 = None

	def forward(self, x, batch_index, edge_index, edge_attr):
		out = F.relu(self.lin0(x))
		
		for i in range(self.num_step_message_passing):
			out = F.relu(self.conv(out, edge_index, edge_attr=edge_attr))

		# we don't need set2set for dag
		# but it maybe helful for molucle net
		if self.set2set is not None:
			out = self.set2set(out, batch_index)
			out = F.relu(self.lin1(out))
			out = self.lin2(out)
				
		return out
	
class MLPBlocks(nn.Module):
	"""Just Good Old Multilayer perceptron with residuals
		layer->dropout->activation(Relu)
		if residuals is True, add a skip connection between block
	Args:
		nn (_type_): _description_
	"""
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		dropout: float,
		num_layers: int,
		output_size: int = None,
		use_residuals: bool = False,
	):
		"""_summary_

		Args:
			input_size (int): input dimensions
			hidden_size (int): hidden_size layer dimensions
			dropout (float): dropout ratio
			num_layers (int): num of hidden layers + 1
			output_size (int, optional): output dimensions. Defaults to None.
			use_residuals (bool, optional): if residuals is True, add a skip connection between locks. Defaults to False.
		"""
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


def scatter_masked_softmax(logits,mask,subset_idxs,mask_value=np.log(1e-4),dim=-1,log=True):
	# # clamp
	# logits = th.clamp(logits,min=np.log(1e-4))
	# apply mask
	masked_logits = mask*logits + (1-mask)*mask_value
	# normalize
	masked_logits = scatter_logsoftmax(masked_logits,subset_idxs,dim)
	if not log:
		# exponentiate
		return th.exp(masked_logits)
	else:
		return masked_logits

