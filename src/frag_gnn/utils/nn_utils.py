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
from frag_gnn.utils.frag_utils import get_node_feats, get_edge_feats, th_long_to_mask
from frag_gnn.utils.misc_utils import scatter_logsumexp, scatter_logsoftmax, scatter_reduce, scatter_masked_softmax, scatter_masked_logsumexp, safelog
from frag_gnn.utils.feat_utils import get_mol_feats_sizes, get_mol_fp_size


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


def build_pool_module(pool_type: str, node_dim: int, num_step_set2set:int=3):
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
	elif pool_type == "set2set":
		"With Set2Set for graph level prediction  (Order Matters: Sequence to sequence for sets) https://arxiv.org/abs/1511.06391"
		pool_module = nn.Sequential(
			pyg.nn.Set2Set(node_dim,processing_steps=num_step_set2set),
			nn.Linear(2 * node_dim, node_dim),
		)
	elif pool_type == "softmax":
		pool_module = pyg.nn.SoftmaxAggregation(learn=True)
	elif pool_type == "powermean":
		pool_module = pyg.nn.PowerMeanAggregation(learn=True)
	elif pool_type == "mean_std_softmax":
		raise NotImplementedError
		# TODO: this need hook up in the way that size of input need change respect this
		#pool_module = pyg.nn.MultiAggregation(['mean', 'std',  pyg.nn.SoftmaxAggregation(learn=True)])
	else:
		assert pool_type == "none", pool_type
		# just return 0s
		pool_module = ZerosAggregation()
	return pool_module

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
		self.include_ce = kwargs.pop('include_ce', False)
		# if include ce here, add ce value to each node
		self.edge_feats_size = edge_feats_size if not self.include_ce else edge_feats_size + 1
		self.node_feats_size = node_feats_size if not self.include_ce else node_feats_size + 1
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
				edge_feats_size=self.edge_feats_size,
				node_feats_size=self.node_feats_size,
				node_hidden_dim=self.hidden_size, 
			    num_step_message_passing =self.num_layers,
				#num_step_set2set=self.num_layers,
				#output_dim = self.hidden_size, # TODO this need to hook up correctly
				gat_heads = 1, # TODO this need to hook up correctly
				gat_dropout=self.dropout,
				is_v2 = True if self.gnn_type == "GATv2" else False
			)
		else:
			raise NotImplementedError(self.gnn_type)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""encode batch of molecule graph"""
		z = self.input_project(x)
		z = self.gnn(z, batch_index, edge_index, edge_attr)
		return z


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

		#self.set2set = pyg.nn.Set2Set(node_hidden_dim,processing_steps=num_step_set2set)
		#self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
		#self.lin2 = nn.Linear(node_hidden_dim, output_size)
			
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

		#out = self.set2set(out, batch_index)
		#out = F.relu(self.lin1(out))
		#out = self.lin2(out)
		return out

class GAT(nn.Module):
	""" Graph attention networks. https://arxiv.org/abs/1710.10903 
		Graph attention networks v2. (How Attentive are Graph Attention Networks?  https://arxiv.org/abs/2105.14491)
	"""

	def __init__(
		self,
		node_feats_size: int,
		edge_feats_size: int,
		node_hidden_dim: int,
		num_step_message_passing: int,
		#num_step_set2set: int,
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
		#self.lin0 = nn.Linear(node_feats_size, node_hidden_dim)

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

	def forward(self, x, batch_index, edge_index, edge_attr):
		#out = F.relu(self.lin0(x))out
		out = x
		for i in range(self.num_step_message_passing):
			out = F.relu(self.conv(out, edge_index, edge_attr=edge_attr))

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
		normalization: str,
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

		# setup norm
		if normalization == "batch":
			norm_mod = nn.BatchNorm1d(hidden_size)
		elif normalization == "layer":
			norm_mod = nn.LayerNorm(hidden_size)
		else:
			assert normalization == "none", normalization
			norm_mod = nn.Identity()
		self.norms = get_clones(norm_mod, num_layers)

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
			output = self.norms[layer_index](output)
			if self.use_residuals:
				output = output + old_op
				old_op = output

		if self.output_layer is not None:
			output = self.output_layer(output)
		return output


class NeimsBlock(nn.Module):
	""" from the NEIMS paper (uses LeakyReLU instead of ReLU) """

	def __init__(self, in_dim, out_dim, dropout):

		super().__init__()
		bottleneck_factor = 0.5
		bottleneck_size = int(round(bottleneck_factor * out_dim))
		self.in_batch_norm = nn.BatchNorm1d(in_dim)
		self.in_activation = nn.LeakyReLU()
		self.in_linear = nn.Linear(in_dim, bottleneck_size)
		self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
		self.out_linear = nn.Linear(bottleneck_size, out_dim)
		self.out_activation = nn.LeakyReLU()
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):

		h = x
		h = self.in_batch_norm(h)
		h = self.in_activation(h)
		h = self.dropout(h)
		h = self.in_linear(h)
		h = self.out_batch_norm(h)
		h = self.out_activation(h)
		h = self.out_linear(h)
		return h


def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset):
	# adapted from NEIMS
	# raw_prediction is [B,D], prec_mass_idx is [B]

	max_idx = raw_prediction.shape[1]
	assert th.all(prec_mass_idx < max_idx)
	idx = th.arange(max_idx, device=prec_mass_idx.device)
	mask = (
		idx.unsqueeze(0) <= (
			prec_mass_idx.unsqueeze(1) +
			prec_mass_offset)).float()
	return mask * raw_prediction


def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
	# adapted from NEIMS
	# raw_prediction is [B,D], prec_mass_idx is [B]

	batch_size = raw_prediction.shape[0]
	max_idx = raw_prediction.shape[1]
	assert th.all(prec_mass_idx < max_idx)
	rev_prediction = th.flip(raw_prediction, dims=(1,))
	# convention is to shift right, so we express as negative to go left
	offset_idx = th.minimum(
		max_idx * th.ones_like(prec_mass_idx),
		prec_mass_idx + prec_mass_offset + 1)
	shifts = - (max_idx - offset_idx)
	gather_idx = th.arange(
		max_idx,
		device=raw_prediction.device).unsqueeze(0).expand(
		batch_size,
		max_idx)
	gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
	offset_rev_prediction = th.gather(rev_prediction, 1, gather_idx)
	# you could mask_prediction_by_mass here but it's unnecessary
	return offset_rev_prediction