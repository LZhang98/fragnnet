"""
This module implements various neural network building blocks for graph-based and feed-forward models.
It includes implementations of Graph Neural Networks (GNNs), Multilayer Perceptrons (MLPs), and other
specialized layers for molecular and spectral data processing.

- ZerosAggregation: Dummy aggregation module that outputs zero tensors.
- GNN: Generic wrapper for different types of Graph Neural Networks (e.g., GINE, MPNN, GAT, GGNN).
- GINE: Graph Isomorphism Network with Edge Features.
- NodeMLP: Multilayer Perceptron for node-level processing.
- MPNN: Neural Message Passing for Quantum Chemistry.
- GAT: Graph Attention Networks (v1 and v2).
- GGNN: Gated Graph Neural Networks.
- MLPBlocks: Multilayer Perceptron with optional residual connections.
- NeimsBlock: Specialized block from the NEIMS paper.
- LowRankDense: Low-rank approximation of dense layers.
- SpecFFN: Feed-forward network for spectrum prediction.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg 
from fragnnet.utils.nn_utils import get_clones, get_norm


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
	"""
	Spectrum Feed-Forward Network (SpecFFN).
	This class implements a feed-forward neural network for predicting spectra
	from molecular or fragment-level features. It supports flexible input sizes,
	dropout, and optional residual connections.

	Args:
		input_size (int): Size of the input features.
		hidden_size (int): Size of the hidden layers.
		output_size (int): Size of the output layer (e.g., number of spectrum bins).
		num_layers (int): Number of layers in the feed-forward network.
		dropout (float): Dropout rate applied after each layer.
		residual (bool): Whether to use residual connections between layers.
	"""
	def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0, residual=False):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.residual = residual

		# Input layer: Maps input features to the hidden size
		self.input_layer = nn.Linear(input_size, hidden_size)

		# Hidden layers: A sequence of fully connected layers
		self.hidden_layers = nn.ModuleList([
			nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
		])

		# Output layer: Maps hidden features to the output size
		self.output_layer = nn.Linear(hidden_size, output_size)

		# Dropout layer
		self.dropout_layer = nn.Dropout(dropout)

	def forward(self, x):
		"""
		Forward pass for the SpecFFN.

		Args:
			x (torch.Tensor): Input tensor of shape [batch_size, input_size].

		Returns:
			torch.Tensor: Output tensor of shape [batch_size, output_size].
		"""
		# Apply the input layer
		x = F.relu(self.input_layer(x))

		# Apply hidden layers with optional residual connections
		for layer in self.hidden_layers:
			if self.residual:
				x = x + F.relu(layer(x))  # Residual connection
			else:
				x = F.relu(layer(x))
			x = self.dropout_layer(x)  # Apply dropout

		# Apply the output layer
		x = self.output_layer(x)
		return x