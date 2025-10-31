"""
This module implements various neural network building blocks for graph-based and feed-forward models.
It includes implementations of Graph Neural Networks (GNNs), Multilayer Perceptrons (MLPs), and other
specialized layers for molecular and spectra.

- ZerosAggregation: Dummy aggregation module that outputs zero tensors.
- GNN: Generic wrapper for different types of Graph Neural Networks (e.g., GINE, MPNN, GAT, GGNN).
- GINE: Graph Isomorphism Network with Edge Features.
- MPNN: Neural Message Passing for Quantum Chemistry.
- GAT: Graph Attention Networks (v1 and v2).
- GGNN: Gated Graph Neural Networks.
- NodeMLP: Multilayer Perceptron for node-level ms/ms prediction.
- MLPBlocks: Multilayer Perceptron with optional residual connections.
- NeimsBlock: Specialized block from the NEIMS paper.
- LowRankDense: Low-rank approximation of dense layers.
- SpecFFN: Feed-forward network for spectrum prediction.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg 
from fragnnet.utils.nn_utils import get_clones
from fragnnet.utils.misc_utils import safelog
class ZerosAggregation(nn.Module):
	"""
	Zero Aggregation module.
	This module performs a dummy aggregation by summing up features across all elements
	and then multiplying the result by zero. Effectively, it outputs a tensor of zeros
	with the same shape as the aggregated result.

	Args:
		nn (torch.nn.Module): Base class for PyTorch modules.
	"""
	def __init__(self):
		super().__init__()
		self.dummy_agg = pyg.nn.SumAggregation()

	def forward(self, x, batch):
		"""
		Forward pass for zero aggregation.

		Args:
			x (torch.Tensor): Input features.
			batch (torch.Tensor): Batch indices for aggregation.

		Returns:
			torch.Tensor: Zero tensor with the same shape as the aggregated result.
		"""
		agg = self.dummy_agg(x, batch)
		return 0. * agg


class GNN(nn.Module):
	"""
	Generic GNN Class.
	This class serves as a wrapper for different types of Graph Neural Networks (GNNs),
	such as GINE, NodeMLP, MPNN, GAT, and GGNN.

	Args:
		hidden_size (int): Hidden layer size.
		num_layers (int): Number of GNN layers.
		node_feats_size (int): Size of node features.
		edge_feats_size (int): Size of edge features.
		gnn_type (str): Type of GNN to use (e.g., "GINE", "MPNN", "GAT").
		dropout (float): Dropout rate.
		normalization (str): Type of normalization to use ("batch", "layer", etc.).
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
		super().__init__()
		self.hidden_size = hidden_size
		self.edge_feats_size = edge_feats_size
		self.node_feats_size = node_feats_size
		self.dropout = dropout
		self.normalization = normalization
		self.gnn_type = gnn_type
		self.num_layers = num_layers

		# Input projection layer to map node features to hidden size
		self.input_project = nn.Linear(self.node_feats_size, self.hidden_size)

		# Initialize the appropriate GNN type
		if self.gnn_type == "GINE":
			self.gnn = GINE(
				hidden_size=self.hidden_size,
				edge_feats_size=self.edge_feats_size,
				node_feats_size=self.node_feats_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
				normalization=self.normalization
			)
		elif self.gnn_type == "NodeMLP":
			self.gnn = NodeMLP(
				hidden_size=self.hidden_size,
				edge_feats_size=self.edge_feats_size,
				node_feats_size=self.node_feats_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
				normalization=self.normalization
			)
		elif self.gnn_type == "MPNN":
			self.gnn = MPNN(
				node_feats_size=self.node_feats_size,
				edge_feats_size=self.edge_feats_size,
				hidden_size=self.hidden_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
				normalization=self.normalization
			)
		elif self.gnn_type == "GAT" or self.gnn_type == "GATv2":
			self.gnn = GAT(
				node_feats_size=self.node_feats_size,
				edge_feats_size=self.edge_feats_size,
				hidden_size=self.hidden_size,
				num_layers=self.num_layers,
				dropout=self.dropout,
				normalization=self.normalization,
				is_v2=(self.gnn_type == "GATv2")
			)
		elif self.gnn_type == "GGNN":
			self.gnn = GGNN(
				node_feats_size=self.node_feats_size,
				edge_feats_size=self.edge_feats_size,
				hidden_size=self.hidden_size,
				num_layers=self.num_layers
			)
		else:
			raise NotImplementedError(self.gnn_type)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""
		Forward pass for the GNN.

		Args:
			x (torch.Tensor): Node features.
			batch_index (torch.Tensor): Batch indices for nodes.
			edge_index (torch.Tensor): Edge indices.
			edge_attr (torch.Tensor): Edge features.

		Returns:
			torch.Tensor: Encoded node features.
		"""
		z = self.input_project(x)
		z = self.gnn(z, batch_index, edge_index, edge_attr)
		return z


def get_norm(normalization,hidden_size):

	# norm_fn may or may not depend on the batch index
	if normalization == "batch":
		norm_mod = nn.BatchNorm1d(hidden_size)
		norm_fn = lambda x, b, n: n(x)
	elif normalization == "layer":
		norm_mod = nn.LayerNorm(hidden_size)
		norm_fn = lambda x, b, n: n(x)
	elif normalization == "graph":
		norm_mod = pyg.nn.GraphNorm(hidden_size)
		norm_fn = lambda x, b, n: n(x,b)
	else:
		assert normalization == "none", normalization
		norm_mod = nn.Identity()
		norm_fn = lambda x, b, n: n(x)
	return norm_mod, norm_fn


class GINE(nn.Module):
	"""
	Graph Isomorphism Network with Edge Features (GINE).
	Implements the GINEConv layer as described in the paper:
	"Strategies for Pre-training Graph Neural Networks" (https://arxiv.org/abs/1905.12265).

	Args:
		hidden_size (int): Hidden layer size.
		node_feats_size (int): Size of node features.
		edge_feats_size (int): Size of edge features.
		num_layers (int): Number of GNN layers.
		dropout (float): Dropout rate.
		normalization (str): Type of normalization to use ("batch", "layer", etc.).
	"""
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
		super().__init__()
		assert edge_feats_size >= 0, edge_feats_size

		# Linear transformation for edge features
		self.edge_transform = nn.Linear(edge_feats_size, hidden_size)

		# Initialize GINE layers
		self.layers = []
		for i in range(num_layers):
			apply_fn = nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size),
			)
			temp_layer = pyg.nn.conv.GINEConv(
				nn=apply_fn,
				eps=0.,
				edge_dim=None
			)
			self.layers.append(temp_layer)

		self.layers = nn.ModuleList(self.layers)

		# Setup normalization and dropout
		norm_mod, norm_fn = get_norm(normalization, hidden_size)
		self.norm_fn = norm_fn
		self.norms = get_clones(norm_mod, num_layers)
		self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""
		Forward pass for GINE.

		Args:
			x (torch.Tensor): Node features.
			batch_index (torch.Tensor): Batch indices for nodes.
			edge_index (torch.Tensor): Edge indices.
			edge_attr (torch.Tensor): Edge features.

		Returns:
			torch.Tensor: Encoded node features.
		"""
		edge_attr = self.edge_transform(edge_attr)
		for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
			layer_out = layer(x, edge_index, edge_attr)
			layer_out = self.norm_fn(layer_out, batch_index, norm)
			x = F.relu(dropout(layer_out)) + x  # Residual connection
		return x


class MPNN(nn.Module):
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
		"""Neural Message Passing for Quantum Chemistry
			https://arxiv.org/abs/1704.01212
		"""
		super().__init__()
		assert edge_feats_size >= 0, edge_feats_size
		self.edge_transform = nn.Linear(edge_feats_size, hidden_size)
		# MPNN 
		self.layers = []
		for i in range(num_layers):
			edge_network = edge_network = nn.Sequential(
				nn.Linear(hidden_size, hidden_size), nn.ReLU(),
				nn.Linear(hidden_size, hidden_size * hidden_size))
			layer = pyg.nn.NNConv(
				hidden_size,
				hidden_size,
				edge_network,
				aggr='mean',
				root_weight=False)
			self.layers.append(layer)
		self.layers = nn.ModuleList(self.layers)
		# setup norms and dropout
		norm_mod, norm_fn = get_norm(normalization, hidden_size)
		self.norm_fn = norm_fn
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


class GAT(nn.Module):
	""" Graph attention networks. https://arxiv.org/abs/1710.10903 
		Graph attention networks v2. (How Attentive are Graph Attention Networks?  https://arxiv.org/abs/2105.14491)
	"""

	def __init__(
		self,
		node_feats_size: int,
		edge_feats_size: int,
		hidden_size: int,
		num_layers: int,
		dropout: float,
		normalization: str,
		num_gat_heads: int = 8,
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
		super().__init__()
		assert edge_feats_size >= 0, edge_feats_size
		self.edge_transform = nn.Linear(edge_feats_size, hidden_size)
		gat_output_size = hidden_size // num_gat_heads if is_concat else hidden_size
		if is_concat and hidden_size % num_gat_heads != 0:
			raise ValueError(f"Ensure that the number of output channels of "
							 f"'GAT' (got '{hidden_size}') is divisible "
							 f"by the number of heads (got '{num_gat_heads}')")
		GATConv_fn = pyg.nn.conv.GATConv if not is_v2 else pyg.nn.conv.GATv2Conv
		self.layers = []
		for i in range(num_layers):
			layer = GATConv_fn(
				in_channels=hidden_size,
				out_channels=gat_output_size,
				heads=num_gat_heads,
				edge_dim=hidden_size,
				dropout=gat_dropout,
				concat=is_concat)
			self.layers.append(layer)
		self.layers = nn.ModuleList(self.layers)
		# setup norms and dropout
		norm_mod, norm_fn = get_norm(normalization, hidden_size)
		self.norm_fn = norm_fn
		self.norms = get_clones(norm_mod, num_layers)
		self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

	def forward(self, x, batch_index, edge_index, edge_attr):
		edge_attr = self.edge_transform(edge_attr)
		for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
			layer_out = layer(x, edge_index, edge_attr)
			layer_out = self.norm_fn(layer_out, batch_index, norm)
			x = F.relu(dropout(layer_out)) + x
		return x


class GGNN(nn.Module):

	def __init__(
		self,
		node_feats_size: int,
		edge_feats_size: int,
		hidden_size: int,
		num_layers: int,
		**kwargs
	):
		"""
		"""
		super().__init__()
		assert edge_feats_size >= 0, edge_feats_size
		self.edge_transform = nn.Linear(edge_feats_size, 1)
		self.model = pyg.nn.conv.GatedGraphConv(
			out_channels=hidden_size,
			num_layers=num_layers
		)

	def forward(self, x, batch_index, edge_index, edge_attr):
		"""forward.
		"""
		edge_weight = self.edge_transform(edge_attr).squeeze(1)
		x = self.model(
			x=x,
			edge_index=edge_index,
			edge_weight=edge_weight
		)
		return x

class NodeMLP(nn.Module):

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
		"""NodeMLP
		"""
		super().__init__()

		assert edge_feats_size >= 0, edge_feats_size
		if edge_feats_size > 0:
			print("> Warning: NodeMLP does not use edge features, but got edge_feats_size > 0")

		self.layers = []
		for i in range(num_layers):
			apply_fn = nn.Sequential(
				nn.Linear(hidden_size, hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, hidden_size),
			)
			temp_layer = apply_fn
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
		for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
			layer_out = layer(x)
			layer_out = self.norm_fn(layer_out, batch_index, norm)
			x = F.relu(dropout(layer_out)) + x
		return x

class MLPBlocks(nn.Module):
	"""
		 Just Good Old Multilayer perceptron with residuals
		layer->dropout->activation(Relu)
		if residuals is True, add a skip connection between block
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
		""" MLP with optional residual connections.

		Args:
			input_size (int): Input feature dimension.
			hidden_size (int): Hidden layer dimension.
			normalization (str): Type of normalization ('batch', 'layer', or 'none').
			dropout (float): Dropout probability.
			num_layers (int): Number of hidden layers (excluding input/output).
			output_size (int, optional): Output feature dimension. If None, no output layer is used. If not None, use a linear layer to project to output_size.
			use_residuals (bool, optional): If True, adds skip connections between hidden layers.
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


class LowRankDense(nn.Module):
	"""
	Low-Rank Dense Layer. Ref: https://arxiv.org/pdf/2001.08885
	This class implements a low-rank approximation of a dense (fully connected) layer.
	Instead of a single weight matrix, it factorizes the weight matrix into two smaller
	matrices, reducing the number of parameters and computational cost.

	Args:
		input_dim (int): Dimension of the input features.
		output_dim (int): Dimension of the output features.
		rank (int): Rank of the factorization (controls the size of the intermediate layer).
		bias (bool): Whether to include a bias term in the layer.
	"""
	def __init__(self, input_dim, output_dim, rank, bias=True):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.rank = rank

		# First projection: input_dim -> rank
		self.proj1 = nn.Linear(input_dim, rank, bias=False)

		# Second projection: rank -> output_dim
		self.proj2 = nn.Linear(rank, output_dim, bias=bias)

	def forward(self, x):
		"""
		Forward pass for the Low-Rank Dense layer.

		Args:
			x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

		Returns:
			torch.Tensor: Output tensor of shape [batch_size, output_dim].
		"""
		# First projection: Reduce input dimension to rank
		x = self.proj1(x)

		# Second projection: Map rank dimension to output dimension
		x = self.proj2(x)

		return x

class SpecFFN(nn.Module):
	"""Spectral Feed-Forward Network.

	Args:
		nn (_type_): _description_
	"""
	def __init__(self, 
		input_size, 
		hidden_size,
		mz_max,
		mz_bin_res,
		num_layers,
		dropout, 
		prec_mz_offset,
		bidirectional,
		use_residuals,
		output_map_size,
		output_activation,
		log_min):

		super().__init__()

		self.input_size = input_size
		self.mz_max = mz_max
		self.mz_bin_res = mz_bin_res
		self.prec_mz_offset = prec_mz_offset
		self.bidirectional = bidirectional
		self.use_residuals = use_residuals
		self._compute_output_size()

		self.in_layer = nn.Linear(input_size, hidden_size)
		self.ff_layers = nn.ModuleList([])
		# self.ff_layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
		for i in range(num_layers):
			self.ff_layers.append(
				NeimsBlock(
					hidden_size,
					hidden_size,
					dropout))
		if output_map_size == -1:
			if self.bidirectional:
				# assumes gating, mass masking
				self.forw_out_layer = nn.Linear(hidden_size, self.output_size)
				self.rev_out_layer = nn.Linear(hidden_size, self.output_size)
				self.out_gate = nn.Sequential(
					*[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()])
			else:
				self.out_layer = nn.Linear(hidden_size, self.output_size)
				self.out_gate = nn.Sequential(
					*[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()])
		else:
			if self.bidirectional:
				# assumes gating, mass masking
				self.forw_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
				self.rev_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
				self.out_gate = nn.Sequential(
					*[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()])
			else:
				self.out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
				self.out_gate = nn.Sequential(
					*[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()])
		assert output_activation in ["relu", "sigmoid"], output_activation
		if output_activation == "relu":
			self.out_activation = nn.ReLU()
		else:
			self.out_activation = nn.Sigmoid()
		self.out_normalization = lambda x: F.normalize(x, p=1, dim=1)
		self.log_min = log_min

	def _compute_output_size(self):
		self.mz_bins = nn.Parameter(th.arange(self.mz_bin_res,self.mz_max+self.mz_bin_res,self.mz_bin_res, dtype=th.float32))
		self.mzs = nn.Parameter(self.mz_bins-0.5*self.mz_bin_res)
		self.output_size = self.mzs.shape[0]

	def _prec_mz_to_idx(self,prec_mz):

		prec_mz_idx = th.bucketize(prec_mz,self.mz_bins.to(prec_mz.device),right=True)
		assert th.max(prec_mz_idx) < self.output_size, (prec_mz_idx,self.output_size)
		return prec_mz_idx

	def forward(self, input_h, prec_mz):

		# get prec_mz_idxs
		prec_mz_idxs = self._prec_mz_to_idx(prec_mz)
		# process inputs
		fh = self.in_layer(input_h)
		# big MLP
		for layer in self.ff_layers:
			if self.use_residuals:
				fh = fh + layer(fh)
			else:
				fh = layer(fh)
		# bidirectional prediction
		if self.bidirectional:
			ff = self.forw_out_layer(fh)
			fr = reverse_prediction(
				self.rev_out_layer(fh),
				prec_mz_idxs,
				self.prec_mz_offset)
			fg = self.out_gate(fh)
			fo = ff * fg + fr * (1. - fg)
		else:
			# apply output layer
			fo = self.out_layer(fh)
			# apply gating
			fg = self.out_gate(fh)
			fo = fg * fo
		fo = self.out_activation(fo)
		fo = mask_prediction_by_mass(
			fo, prec_mz_idxs, self.prec_mz_offset)
		spec = self.out_normalization(fo)
		# handle all zeroes (set to first bin by default)
		all_zero_mask = th.max(spec,dim=1)[0]<=0.
		all_zero_bonus = th.zeros_like(spec)
		all_zero_bonus[all_zero_mask,0] = 1.
		spec = spec + all_zero_bonus
		# convert dense spectrum to sparse
		mask = spec>0.
		pred_mzs = (self.mzs.unsqueeze(0).expand(spec.shape[0],-1))[mask]
		pred_logprobs = safelog(spec, eps=self.log_min)[mask]
		pred_batch_idxs = th.arange(spec.shape[0],device=spec.device).unsqueeze(1).expand(-1,spec.shape[1])[mask]
		pred_specs = spec
		return pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs

class SetTransformer(nn.Module):
	"""
	Set Transformer implementation for processing sets of elements.
	

	Args:
		input_dim (int): Dimension of the input features.
		hidden_dim (int): Dimension of the hidden layers.
		output_dim (int): Dimension of the output embedding.
		num_heads (int): Number of attention heads.
		num_inds (int): Number of inducing points.
		num_blocks (int): Number of Set Transformer blocks.
		dropout (float): Dropout rate for regularization.
	"""
	def __init__(
		self,
		input_dim,
		hidden_dim,
		output_dim,
		num_heads=4,
		num_inds=32,
		num_blocks=2,
		dropout=0.1,
		return_nodes=False,  # If True, return per-node outputs; if False, return per-set outputs
	):
		super(SetTransformer, self).__init__()

		self.return_nodes = return_nodes
		self.inducing_points = nn.Parameter(th.randn(num_inds, input_dim))
		self.attention_blocks = nn.ModuleList([
			nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
			for _ in range(num_blocks)
		])
		
		# Decoder: maps inducing points back to nodes (only used if return_nodes=True)
		if return_nodes:
			self.decoder_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
			self.decoder_fc = nn.Sequential(
				nn.Linear(input_dim, hidden_dim),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(hidden_dim, output_dim)
			)
		else:
			self.fc = nn.Sequential(
				nn.Linear(input_dim, hidden_dim),
				nn.ReLU(),
				nn.Dropout(dropout),
				nn.Linear(hidden_dim, output_dim)
			)

	def forward(self, x, batch_idx=None):
		"""
		Forward pass for the Set Transformer.
		Args:
			x (Tensor): Input tensor. 
				- If batch_idx is None: shape (batch_size, set_size, input_dim) - fixed size sets
				- If batch_idx is provided: shape (total_elements, input_dim) - variable size sets
			# Input: (total_elements=85, input_dim=128)
			# batch_idx: [0,0,0,...,1,1,1,...,2,2,...,3,3,3,...]  (85 elements total)
			# Set 0: 25 elements, Set 1: 20 elements, Set 2: 15 elements, Set 3: 25 elementss
			batch_idx (Tensor, optional): Batch indices for variable-sized sets, shape (total_elements,).
				Each element indicates which set/graph the element belongs to.

		Returns:
			Tensor: Encoded set representation of shape (batch_size, output_dim).
		"""
		if batch_idx is None:
			# Fixed-size sets: (batch_size, set_size, input_dim)
			batch_size, set_size, _ = x.size()
			inducing_points = self.inducing_points.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_inds, input_dim)

			# SetTransformer architecture:
			# 1. Encoder: inputs attend to inducing points
			h, _ = self.attention_blocks[0](inducing_points, x, x)  # (B, num_inds, input_dim)
			
			# 2. Additional layers: inducing points attend to themselves
			for i in range(1, len(self.attention_blocks)):
				h, _ = self.attention_blocks[i](h, h, h)
			
			if self.return_nodes:
				# 3. Decoder: map inducing points back to per-node outputs
				x_decoded, _ = self.decoder_attn(x, h, h)  # (B, set_size, input_dim)
				x_out = self.decoder_fc(x_decoded)  # (B, set_size, output_dim)
				return x_out
			else:
				# 3. Aggregate: mean over inducing points for set-level output
				x = h.mean(dim=1)  # (B, input_dim)
				return self.fc(x)
		else:
			# Variable-size sets: (total_elements, input_dim) with batch_idx
			num_batches = int(batch_idx.max().item()) + 1
			output_list = []
			
			for batch_id in range(num_batches):
				# Extract elements for this batch
				mask = (batch_idx == batch_id)
				x_batch = x[mask].unsqueeze(0)  # (1, set_size_i, input_dim)
				
				# Apply SetTransformer: first attend from inputs to inducing points
				inducing_points = self.inducing_points.unsqueeze(0)  # (1, num_inds, input_dim)
				
				# Encoder: inputs attend to inducing points (queries=inducing_points, keys/values=inputs)
				h, _ = self.attention_blocks[0](inducing_points, x_batch, x_batch)  # (1, num_inds, input_dim)
				
				# Additional encoding layers: inducing points attend to themselves
				for i in range(1, len(self.attention_blocks)):
					h, _ = self.attention_blocks[i](h, h, h)
				
				if self.return_nodes:
					# Decoder: map inducing points back to per-node outputs
					# queries=original inputs, keys/values=encoded inducing points
					x_decoded, _ = self.decoder_attn(x_batch, h, h)  # (1, set_size_i, input_dim)
					x_out = self.decoder_fc(x_decoded.squeeze(0))  # (set_size_i, output_dim)
					output_list.append(x_out)
				else:
					# Aggregate: take mean of inducing points for set-level output
					x_agg = h.mean(dim=1)  # (1, input_dim)
					output_list.append(x_agg.squeeze(0))
			
			# Stack all batch outputs
			if self.return_nodes:
				x = th.cat(output_list, dim=0)  # (total_elements, output_dim)
			else:
				x = th.stack(output_list, dim=0)  # (num_batches, input_dim)
				x = self.fc(x)
			return x

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


class Set2SetPooling(nn.Module):
	"""Wrapper for Set2Set that handles batch argument correctly."""
	def __init__(self, node_dim: int, processing_steps: int = 3):
		super().__init__()
		self.set2set = pyg.nn.Set2Set(node_dim, processing_steps=processing_steps)
		self.linear = nn.Linear(2 * node_dim, node_dim)
	
	def forward(self, x, batch):
		"""Forward pass that accepts batch argument like other PyG aggregations."""
		x = self.set2set(x, batch)
		x = self.linear(x)
		return x


def build_pool_module(pool_type: str, node_dim: int, num_step_set2set:int=3):
	"""Method to build pooling layer for given pool_type and dim
		node dim is required for attention layer. 
		if pool type is not one of 'sum','mean','max','attention','set2set','set_transformer', return ZerosAggregation
	Args:
		pool_type (str): pooling type in ['sum','mean','max','attention','set2set','set_transformer','softmax','powermean','none']
		node_dim (int): node_dim, used for attention and set_transformer
		num_step_set2set (int): number of processing steps for set2set pooling

	Returns:
		nn.Module: pooling module
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
		pool_module = Set2SetPooling(node_dim, processing_steps=num_step_set2set)
	elif pool_type == "set_transformer":
		"With SetTransformer for graph level prediction https://arxiv.org/abs/1810.00825"
		pool_module = SetTransformer(
			input_dim=node_dim,
			hidden_dim=node_dim,
			output_dim=node_dim,
			num_heads=4,
			num_inds=16,  # 16 inducing points for pooling
			num_blocks=1,  # Single attention block for pooling
			dropout=0.1,
			return_nodes=False  # Return set-level output for pooling
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

