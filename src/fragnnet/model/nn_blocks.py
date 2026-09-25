"""
This module implements various neural network building blocks for graph-based and feed-forward models.
It includes implementations of Graph Neural Networks (GNNs), Multilayer Perceptrons (MLPs), and other
specialized layers for molecular and spectra.

- ZerosAggregation: Dummy aggregation module that outputs zero tensors.
- GNN: Generic wrapper for different types of Graph Neural Networks (e.g., GINE, MPNN, GAT, GGNN).
- GINE: Graph Isomorphism Network with Edge Features.
- GINE_GSA: GINE augmented with exact per-graph global self-attention.
- GINE_GRAPHORMER: GINE augmented with structurally biased global softmax attention.
- Graphormer: Global molecular attention with degree, distance, and bond biases.
- MPNN: Neural Message Passing for Quantum Chemistry.
- GAT: Graph Attention Networks (v1 and v2).
- GGNN: Gated Graph Neural Networks.
- NodeMLP: Multilayer Perceptron for node-level ms/ms prediction.
- MLPBlocks: Multilayer Perceptron with optional residual connections.
- NeimsBlock: Specialized block from the NEIMS paper.
- LowRankDense: Low-rank approximation of dense layers.
- SpecFFN: Feed-forward network for spectrum prediction.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import aggr

from fragnnet.utils.misc_utils import safelog
from fragnnet.utils.nn_utils import get_clones

logger = logging.getLogger(__name__)


class PoolWithDimSizeAdapter(nn.Module):
    """Normalize pooling modules to a common interface with optional ``dim_size``."""

    def __init__(self, pool_module: nn.Module) -> None:
        super().__init__()
        self.pool_module = pool_module

    def forward(
        self,
        x: th.Tensor,
        index: th.Tensor | None = None,
        ptr: th.Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
        **kwargs,
    ) -> th.Tensor:
        try:
            pooled = self.pool_module(
                x,
                index=index,
                ptr=ptr,
                dim_size=dim_size,
                dim=dim,
                **kwargs,
            )
        except TypeError as exc:
            if "dim_size" not in str(exc):
                raise
            pooled = self.pool_module(x, index)

        if dim_size is None or pooled.shape[0] == dim_size:
            return pooled

        out = x.new_zeros((dim_size, pooled.shape[1]))
        if pooled.shape[0] > 0:
            out[: pooled.shape[0]] = pooled
        return out


class ZerosAggregation(nn.Module):
    """
    Zero Aggregation module.
    This module performs a dummy aggregation by summing up features across all elements
    and then multiplying the result by zero. Effectively, it outputs a tensor of zeros
    with the same shape as the aggregated result.

    Args:
        nn (torch.nn.Module): Base class for PyTorch modules.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy_agg = aggr.SumAggregation()

    def forward(
        self,
        x: th.Tensor,
        index: th.Tensor | None = None,
        ptr: th.Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
        **kwargs,
    ) -> th.Tensor:
        """
        Forward pass for zero aggregation.

        Args:
            x (torch.Tensor): Input features.
            index (torch.Tensor | None): Group indices for aggregation.
            ptr (torch.Tensor | None): CSR-style segment pointers.
            dim_size (int | None): Explicit number of output groups.
            dim (int): Aggregation dimension.

        Returns:
            torch.Tensor: Zero tensor with the same shape as the aggregated result.
        """
        agg = self.dummy_agg(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, **kwargs)
        return th.zeros_like(agg)


_RESIDUAL_ACTIVATIONS = ("relu", "identity", "gelu")
GLOBAL_MESSAGE_SWAP_UNITS = ("atom", "molecule", "feature")


def _validate_residual_activation(activation: str) -> str:
    """Validate the activation applied to a GNN residual branch before the skip add."""
    if activation not in _RESIDUAL_ACTIVATIONS:
        raise ValueError(
            f"residual_activation must be one of {_RESIDUAL_ACTIVATIONS}, got {activation!r}"
        )
    return activation


def _residual_branch(activation: str, value: th.Tensor) -> th.Tensor:
    """Apply the configured branch activation.

    ``relu`` reproduces the historical behaviour, in which a layer can only ADD
    per-coordinate mass to the residual stream and can never attenuate it. ``identity``
    removes that restriction; ``gelu`` keeps a nonlinearity while allowing negative
    contributions.
    """
    if activation == "relu":
        return F.relu(value)
    if activation == "gelu":
        return F.gelu(value)
    return value


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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.residual_activation = _validate_residual_activation(
            kwargs.get("residual_activation", "relu")
        )
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
                normalization=self.normalization,
                residual_activation=self.residual_activation,
            )
        elif self.gnn_type == "GINE_GSA":
            self.gnn = GINE_GSA(
                hidden_size=self.hidden_size,
                edge_feats_size=self.edge_feats_size,
                node_feats_size=self.node_feats_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
                **kwargs,
            )
        elif self.gnn_type == "GINE_GRAPHORMER":
            self.gnn = GINEGraphormer(
                hidden_size=self.hidden_size,
                edge_feats_size=self.edge_feats_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
                num_heads=kwargs["num_heads"],
                **_graphormer_structural_kwargs(kwargs),
            )
        elif self.gnn_type == "GINE_GATED":
            self.gnn = GatedGINE(
                hidden_size=self.hidden_size,
                edge_feats_size=self.edge_feats_size,
                node_feats_size=self.node_feats_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
                ctx_dim=kwargs["ctx_dim"],
                gate_mode=kwargs["gate_mode"],
                gate_hidden_dim=kwargs["gate_hidden_dim"],
            )
        elif self.gnn_type == "Graphormer":
            self.gnn = Graphormer(
                hidden_size=self.hidden_size,
                edge_feats_size=self.edge_feats_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
                num_heads=kwargs["num_heads"],
                **_graphormer_structural_kwargs(kwargs),
            )
        elif self.gnn_type == "NodeMLP":
            self.gnn = NodeMLP(
                hidden_size=self.hidden_size,
                edge_feats_size=self.edge_feats_size,
                node_feats_size=self.node_feats_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
                residual_activation=self.residual_activation,
            )
        elif self.gnn_type == "MPNN":
            self.gnn = MPNN(
                node_feats_size=self.node_feats_size,
                edge_feats_size=self.edge_feats_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
            )
        elif self.gnn_type == "GAT" or self.gnn_type == "GATv2":
            self.gnn = GAT(
                node_feats_size=self.node_feats_size,
                edge_feats_size=self.edge_feats_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                normalization=self.normalization,
                is_v2=(self.gnn_type == "GATv2"),
            )
        elif self.gnn_type == "GGNN":
            self.gnn = GGNN(
                node_feats_size=self.node_feats_size,
                edge_feats_size=self.edge_feats_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
            )
        else:
            raise NotImplementedError(self.gnn_type)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        ctx: th.Tensor | None = None,
    ) -> th.Tensor:
        """
        Forward pass for the GNN.

        Args:
            x (torch.Tensor): Node features.
            batch_index (torch.Tensor): Batch indices for nodes.
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge features.
            ctx (torch.Tensor | None): Per-graph context of shape (num_graphs, ctx_dim).
                Required by gnn_type="GINE_GATED", ignored by every other type.

        Returns:
            torch.Tensor: Encoded node features.

        Raises:
            ValueError: If gnn_type="GINE_GATED" and ctx is None.
        """
        z = self.input_project(x)
        if self.gnn_type == "GINE_GATED":
            if ctx is None:
                raise ValueError("gnn_type='GINE_GATED' requires ctx, got None")
            z = self.gnn(z, batch_index, edge_index, edge_attr, ctx)
        else:
            z = self.gnn(z, batch_index, edge_index, edge_attr)
        return z


def _graphormer_structural_kwargs(kwargs: dict) -> dict:
    """Pick out the optional Graphormer structural-encoding sizes, if the caller set them.

    Omitted keys and ``None`` both fall through to the constructor defaults, so callers
    that do not care about these knobs can keep passing only ``num_heads``.
    """
    names = ("max_degree", "max_spatial_distance", "multi_hop_max_distance")
    return {name: kwargs[name] for name in names if kwargs.get(name) is not None}


def get_norm(normalization: str, hidden_size: int) -> tuple[nn.Module, Callable]:
    # norm_fn may or may not depend on the batch index
    if normalization == "batch":
        norm_mod = nn.BatchNorm1d(hidden_size)

        def norm_fn(x: th.Tensor, b: th.Tensor, n: nn.Module) -> th.Tensor:
            return n(x)

    elif normalization == "layer":
        norm_mod = nn.LayerNorm(hidden_size)

        def norm_fn(x: th.Tensor, b: th.Tensor, n: nn.Module) -> th.Tensor:
            return n(x)

    elif normalization == "graph":
        norm_mod = pyg.nn.GraphNorm(hidden_size)

        def norm_fn(x: th.Tensor, b: th.Tensor, n: nn.Module) -> th.Tensor:
            return n(x, b)

    else:
        assert normalization == "none", normalization
        norm_mod = nn.Identity()

        def norm_fn(x: th.Tensor, b: th.Tensor, n: nn.Module) -> th.Tensor:
            return n(x)

    return norm_mod, norm_fn


class GraphormerEncoderLayer(nn.Module):
    """Pre-norm transformer layer with an additive structural attention bias."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(
        self,
        x: th.Tensor,
        attention_bias: th.Tensor,
        padding_mask: th.Tensor,
    ) -> th.Tensor:
        """Mix dense graph tokens while excluding padded keys."""
        batch_size, num_tokens, hidden_size = x.shape
        normalized = self.attention_norm(x)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        query = query.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attention_logits = th.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_logits = attention_logits + attention_bias
        attention_logits = attention_logits.masked_fill(
            padding_mask[:, None, None, :],
            -th.inf,
        )
        attention = self.attention_dropout(F.softmax(attention_logits, dim=-1))
        attended = (
            th.matmul(attention, value)
            .transpose(1, 2)
            .reshape(
                batch_size,
                num_tokens,
                hidden_size,
            )
        )
        x = x + self.residual_dropout(self.attention_output(attended))
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


class Graphormer(nn.Module):
    """Graphormer molecular encoder operating directly on a PyG mini-batch.

    The encoder preserves the ordinary :class:`GNN` contract: it receives flat atom
    states and directed bonds, performs graph-scoped global attention, and returns one
    state per input atom in the original order. Its attention bias contains the three
    structural signals from Graphormer: in/out degree, shortest-path distance, and
    encoded bond features. A virtual graph token provides a global communication path
    but is removed from the returned atom states.

    Cost note: the structural bias is rebuilt on every forward pass and accounts for
    roughly a fifth of the step time (9.3 ms of a 42.5 ms forward+backward at batch 32,
    60 atoms, hidden 512, 5 layers on an RTX 2070 SUPER). Half of that is the
    Floyd-Warshall loop, which is kernel-launch bound rather than compute bound: it
    issues ``max_nodes`` sequential iterations over small tensors, so neither a narrower
    integer dtype nor min-plus repeated squaring speeds it up enough to justify their
    cost (squaring is exact but only 1.1-2.1x while needing an extra
    ``num_graphs * max_nodes ** 3`` workspace). Removing this cost needs the topology
    tables to be precomputed per molecule in the dataset, not a different kernel here.
    """

    def __init__(
        self,
        hidden_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        num_heads: int,
        max_degree: int = 64,
        max_spatial_distance: int = 128,
        multi_hop_max_distance: int = 5,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if normalization != "layer":
            raise ValueError("Graphormer requires mol_normalization: layer")
        if edge_feats_size <= 0:
            raise ValueError("Graphormer requires at least one edge feature")
        if max_degree < 1 or max_spatial_distance < 1 or multi_hop_max_distance < 1:
            raise ValueError(
                "max_degree, max_spatial_distance and multi_hop_max_distance must be >= 1, got "
                f"{max_degree}, {max_spatial_distance}, {multi_hop_max_distance}"
            )

        self.num_heads = num_heads
        self.max_degree = max_degree
        self.max_spatial_distance = max_spatial_distance
        self.multi_hop_max_distance = multi_hop_max_distance
        # Molecular graphs are bidirectional whenever mol_params.pyg_bigraph is True, which
        # makes the two degree tables receive identical indices. They are kept separate so
        # that pyg_bigraph=False, where every bond appears in one direction only, still gets
        # distinguishable in- and out-degree signals.
        self.in_degree_embedding = nn.Embedding(self.max_degree + 1, hidden_size)
        self.out_degree_embedding = nn.Embedding(self.max_degree + 1, hidden_size)
        self.spatial_embedding = nn.Embedding(self.max_spatial_distance + 2, num_heads)
        self.edge_bias = nn.Linear(edge_feats_size, num_heads, bias=False)
        self.edge_distance_mixing = nn.Parameter(
            th.empty(self.multi_hop_max_distance, num_heads, num_heads)
        )
        self.graph_token = nn.Parameter(th.zeros(1, 1, hidden_size))
        self.graph_token_bias = nn.Parameter(th.zeros(num_heads))
        self.layers = nn.ModuleList(
            [GraphormerEncoderLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        # Initializes everything built above; subclasses add their own modules afterwards
        # and keep whatever init suits their fan-in (see GINEGraphormer).
        self.apply(self._init_graphormer_parameters)
        nn.init.normal_(self.edge_distance_mixing, std=0.02)
        nn.init.normal_(self.graph_token, std=0.02)

    @staticmethod
    def _init_graphormer_parameters(module: nn.Module) -> None:
        """Use the small-normal initialization from the original Graphormer."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _shortest_path_distances(
        self,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        max_nodes: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return true all-pairs distances, next hops, and flat-to-dense atom indices.

        Distances are the real shortest-path lengths, never truncated to
        ``max_spatial_distance``: a disconnected pair is the only thing that comes back as
        ``unreachable_distance(max_nodes)``. Saturating the reachable distances onto the
        embedding's last bucket is the caller's job, so that a small
        ``max_spatial_distance`` shortens the embedding table without also telling the
        model that distant atoms belong to different molecules.
        """
        num_graphs = int(batch_index[-1].item()) + 1
        graph_sizes = th.bincount(batch_index, minlength=num_graphs)
        graph_offsets = th.cat((graph_sizes.new_zeros(1), graph_sizes.cumsum(0)[:-1]))
        local_indices = th.arange(batch_index.numel(), device=batch_index.device)
        local_indices = local_indices - graph_offsets[batch_index]

        unreachable = self.unreachable_distance(max_nodes)
        distances = th.full(
            (num_graphs, max_nodes, max_nodes),
            unreachable,
            dtype=th.long,
            device=batch_index.device,
        )
        diagonal = th.arange(max_nodes, device=batch_index.device)
        distances[:, diagonal, diagonal] = 0
        next_hop = th.full_like(distances, -1)
        next_hop[:, diagonal, diagonal] = diagonal
        source, target = edge_index
        edge_graphs = batch_index[source]
        distances[edge_graphs, local_indices[source], local_indices[target]] = 1
        next_hop[edge_graphs, local_indices[source], local_indices[target]] = local_indices[target]

        # Floyd-Warshall is cheap for molecular graphs and keeps preprocessing unchanged.
        for intermediate in range(max_nodes):
            through_intermediate = distances[:, :, intermediate].unsqueeze(2) + distances[
                :, intermediate, :
            ].unsqueeze(1)
            shorter_path = through_intermediate < distances
            candidate_next_hop = next_hop[:, :, intermediate].unsqueeze(2).expand_as(next_hop)
            next_hop = th.where(shorter_path, candidate_next_hop, next_hop)
            distances = th.where(shorter_path, through_intermediate, distances)
        return distances.clamp_max(unreachable), next_hop, local_indices

    @staticmethod
    def unreachable_distance(max_nodes: int) -> int:
        """Sentinel standing for "no path at all" between two atoms.

        Any real shortest path in a graph of ``max_nodes`` atoms visits at most
        ``max_nodes`` atoms, so ``max_nodes`` itself can never be a true distance and is
        free to mark disconnection.
        """
        return max(max_nodes, 1)

    def _multi_hop_edge_bias(
        self,
        distances: th.Tensor,
        next_hop: th.Tensor,
        direct_edge_bias: th.Tensor,
    ) -> th.Tensor:
        """Encode the first five bonds along one shortest path for every atom pair."""
        batch_size, max_nodes, _ = distances.shape
        batch_grid = th.arange(batch_size, device=distances.device)[:, None, None]
        batch_grid = batch_grid.expand(batch_size, max_nodes, max_nodes)
        source_grid = th.arange(max_nodes, device=distances.device)[None, :, None]
        source_grid = source_grid.expand(batch_size, max_nodes, max_nodes)
        target_grid = th.arange(max_nodes, device=distances.device)[None, None, :]
        target_grid = target_grid.expand(batch_size, max_nodes, max_nodes)

        hop_count = distances.clamp(max=self.multi_hop_max_distance)
        reachable = (distances > 0) & (distances < self.unreachable_distance(max_nodes))
        current_nodes = source_grid
        encoded_path = direct_edge_bias.new_zeros(direct_edge_bias.shape)
        for hop_index in range(self.multi_hop_max_distance):
            active = reachable & (hop_index < hop_count)
            following_nodes = next_hop[batch_grid, current_nodes, target_grid]
            safe_following_nodes = following_nodes.clamp_min(0)
            hop_edge_bias = direct_edge_bias[
                batch_grid,
                current_nodes,
                safe_following_nodes,
            ]
            hop_edge_bias = th.matmul(
                hop_edge_bias,
                self.edge_distance_mixing[hop_index],
            )
            encoded_path = encoded_path + hop_edge_bias.masked_fill(~active.unsqueeze(-1), 0.0)
            current_nodes = th.where(active, safe_following_nodes, current_nodes)
        return encoded_path / hop_count.clamp_min(1).unsqueeze(-1)

    def _structural_attention_bias(
        self,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        max_nodes: int,
    ) -> th.Tensor:
        """Build per-head spatial and multi-hop edge attention biases."""
        distances, next_hop, local_indices = self._shortest_path_distances(
            batch_index,
            edge_index,
            max_nodes,
        )
        # Saturate reachable pairs onto the last real bucket and reserve the row past it
        # for genuine disconnection, mirroring how max_degree clamps. Folding the two
        # together would tell the model that atoms farther than max_spatial_distance are
        # in a different molecule.
        unreachable = self.unreachable_distance(max_nodes)
        spatial_index = th.where(
            distances >= unreachable,
            th.full_like(distances, self.max_spatial_distance + 1),
            distances.clamp_max(self.max_spatial_distance),
        )
        spatial_bias = self.spatial_embedding(spatial_index)
        encoded_edges = self.edge_bias(edge_attr)
        bias_dtype = th.promote_types(spatial_bias.dtype, encoded_edges.dtype)
        spatial_bias = spatial_bias.to(dtype=bias_dtype)
        encoded_edges = encoded_edges.to(dtype=bias_dtype)
        direct_edge_bias = spatial_bias.new_zeros(
            (distances.shape[0], max_nodes, max_nodes, self.num_heads)
        )
        source, target = edge_index
        edge_graphs = batch_index[source]
        # Scatter all heads in one call: indexing the leading three axes leaves the head
        # axis to broadcast, which is ~9x faster than one index_put_ per head.
        direct_edge_bias.index_put_(
            (
                edge_graphs,
                local_indices[source],
                local_indices[target],
            ),
            encoded_edges,
            accumulate=True,
        )
        multi_hop_bias = self._multi_hop_edge_bias(
            distances,
            next_hop,
            direct_edge_bias,
        )
        return (spatial_bias + multi_hop_bias).permute(0, 3, 1, 2)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
        """Encode atoms with graph-local, structurally biased global attention."""
        if x.shape[0] == 0:
            return x
        self._validate_inputs(x, batch_index, edge_index, edge_attr)

        atom_states = x + self._degree_states(edge_index, x.shape[0])
        dense_x, valid_atoms = pyg.utils.to_dense_batch(atom_states, batch_index)
        batch_size, max_nodes, _ = dense_x.shape
        attention_bias, padding_mask = self._global_attention_inputs(
            batch_index,
            edge_index,
            edge_attr,
            valid_atoms,
            max_nodes,
        )

        dense_x = th.cat((self.graph_token.expand(batch_size, -1, -1), dense_x), dim=1)
        for layer in self.layers:
            dense_x = layer(dense_x, attention_bias, padding_mask)
        return self.output_norm(dense_x[:, 1:])[valid_atoms]

    def _validate_inputs(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> None:
        """Reject batches this encoder cannot interpret."""
        name = type(self).__name__
        if batch_index.shape != (x.shape[0],):
            raise ValueError("batch_index must contain one graph index per atom")
        if not th.all(batch_index[1:] >= batch_index[:-1]):
            raise ValueError(f"{name} requires atoms grouped by graph")
        if edge_index.shape[0] != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_index and edge_attr must describe the same directed bonds")
        if not th.equal(batch_index[edge_index[0]], batch_index[edge_index[1]]):
            raise ValueError(f"{name} received a bond connecting two batched graphs")

    def _degree_states(self, edge_index: th.Tensor, num_atoms: int) -> th.Tensor:
        """Embed each atom's in- and out-degree."""
        in_degree = th.bincount(edge_index[1], minlength=num_atoms).clamp_max(self.max_degree)
        out_degree = th.bincount(edge_index[0], minlength=num_atoms).clamp_max(self.max_degree)
        return self.in_degree_embedding(in_degree) + self.out_degree_embedding(out_degree)

    def _global_attention_inputs(
        self,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        valid_atoms: th.Tensor,
        max_nodes: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Build the virtual-token-padded attention bias and key padding mask."""
        structural_bias = self._structural_attention_bias(
            batch_index,
            edge_index,
            edge_attr,
            max_nodes,
        )
        batch_size = structural_bias.shape[0]
        attention_bias = structural_bias.new_zeros(
            (batch_size, self.num_heads, max_nodes + 1, max_nodes + 1)
        )
        attention_bias[:, :, 1:, 1:] = structural_bias
        attention_bias[:, :, 0, 1:] = self.graph_token_bias.view(1, -1, 1)
        attention_bias[:, :, 1:, 0] = self.graph_token_bias.view(1, -1, 1)
        padding_mask = th.cat(
            (
                th.zeros((batch_size, 1), dtype=th.bool, device=valid_atoms.device),
                ~valid_atoms,
            ),
            dim=1,
        )
        return attention_bias, padding_mask


class GINEGraphormer(Graphormer):
    """Interleave local GINE updates with structurally biased global attention.

    Each layer first updates flat atom states through real chemical bonds, then
    applies one Graphormer attention/FFN layer using degree, shortest-path, and
    multi-hop bond-path encodings. Structural attention biases are built once per
    forward pass and reused by every layer.
    """

    def __init__(
        self,
        hidden_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        num_heads: int,
        max_degree: int = 64,
        max_spatial_distance: int = 128,
        multi_hop_max_distance: int = 5,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            edge_feats_size=edge_feats_size,
            num_layers=num_layers,
            dropout=dropout,
            normalization=normalization,
            num_heads=num_heads,
            max_degree=max_degree,
            max_spatial_distance=max_spatial_distance,
            multi_hop_max_distance=multi_hop_max_distance,
        )
        self.local_edge_transform = nn.Linear(edge_feats_size, hidden_size)
        self.local_layers = nn.ModuleList(
            [
                pyg.nn.conv.GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                    ),
                    eps=0.0,
                    edge_dim=None,
                )
                for _ in range(num_layers)
            ]
        )
        self.local_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.local_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        # The local branch deliberately keeps PyTorch's fan-in-aware default init rather
        # than the Graphormer std=0.02 used by the attention stack. Graphormer's constant
        # std is tuned for fan_in == hidden_size; applying it to local_edge_transform,
        # whose fan_in is the handful of bond features, would shrink that projection's
        # output ~8x relative to the atom states it is summed with inside GINEConv
        # (relu(x_j + edge_attr)) and near-ablate bond features at init.

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
        """Return one locally and globally contextualized state per atom."""
        if x.shape[0] == 0:
            return x
        self._validate_inputs(x, batch_index, edge_index, edge_attr)

        # 1. Build graph structure once and initialize topology-aware atom states.
        atom_states = x + self._degree_states(edge_index, x.shape[0])
        _, valid_atoms = pyg.utils.to_dense_batch(atom_states, batch_index)
        batch_size, max_nodes = valid_atoms.shape
        attention_bias, padding_mask = self._global_attention_inputs(
            batch_index,
            edge_index,
            edge_attr,
            valid_atoms,
            max_nodes,
        )
        graph_tokens = self.graph_token.expand(batch_size, -1, -1)
        local_edge_attr = self.local_edge_transform(edge_attr)

        # 2. Alternate one real-bond GINE update with one global Graphormer layer.
        for local_layer, local_norm, local_dropout, global_layer in zip(
            self.local_layers,
            self.local_norms,
            self.local_dropouts,
            self.layers,
        ):
            local_out = local_layer(atom_states, edge_index, local_edge_attr)
            atom_states = atom_states + F.relu(local_dropout(local_norm(local_out)))
            dense_atoms, _ = pyg.utils.to_dense_batch(atom_states, batch_index)
            dense_states = th.cat((graph_tokens, dense_atoms), dim=1)
            dense_states = global_layer(dense_states, attention_bias, padding_mask)
            graph_tokens = dense_states[:, :1]
            atom_states = dense_states[:, 1:][valid_atoms]

        return self.output_norm(atom_states)


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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.residual_activation = _validate_residual_activation(
            kwargs.get("residual_activation", "relu")
        )
        if edge_feats_size < 0:
            raise ValueError(f"edge_feats_size must be non-negative, got {edge_feats_size}")

        # Linear transformation for edge features
        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)

        # Initialize GINE layers
        self.layers = []
        for _i in range(num_layers):
            apply_fn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            temp_layer = pyg.nn.conv.GINEConv(nn=apply_fn, eps=0.0, edge_dim=None)
            self.layers.append(temp_layer)

        self.layers = nn.ModuleList(self.layers)

        # Setup normalization and dropout
        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
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
            x = (
                _residual_branch(self.residual_activation, dropout(layer_out)) + x
            )  # Residual connection
        return x


class GatedGINEConv(nn.Module):
    """A GINE convolution whose per-edge messages are scaled by an externally supplied gate.

    Reproduces ``pyg.nn.conv.GINEConv(nn=..., eps=0.0, edge_dim=None)``, i.e.
    ``out_i = nn((1 + eps) * x_i + sum_j relu(x_j + e_ij))``, but weights each message by a
    scalar ``g_ij``. The gate is passed per call rather than owned here, so the caller decides
    whether it is static or recomputed per layer (see ``GatedGINE``).

    Implemented directly rather than by subclassing GINEConv: ``MessagePassing.propagate`` has
    its signature generated from the parent's ``message()`` at class-creation time and rejects
    the extra ``gate`` argument. Parameter names mirror GINEConv (``nn``, ``eps``) so the two
    stay state-dict compatible, which is what lets the identity-at-init test compare them.

    Attributes:
        nn: The GINE apply function, applied after aggregation.
        eps: Self-loop weighting, registered as a buffer to match GINEConv's state dict.
    """

    def __init__(self, nn_module: nn.Module, eps: float = 0.0) -> None:
        """Initialize the gated GINE convolution.

        Args:
            nn_module: Apply function mapping (hidden_size,) -> (hidden_size,) after aggregation.
            eps: Self-loop weighting; 0.0 matches the stock GINE setup.
        """
        super().__init__()
        self.nn = nn_module
        self.register_buffer("eps", th.tensor([eps]))

    def forward(
        self, x: th.Tensor, edge_index: th.Tensor, edge_attr: th.Tensor, gate: th.Tensor
    ) -> th.Tensor:
        """Forward pass with gated messages.

        Args:
            x: Node features of shape (num_nodes, hidden_size).
            edge_index: Edge connectivity of shape (2, num_edges); row 0 is the message source,
                row 1 the target (PyG's source_to_target convention).
            edge_attr: Edge features of shape (num_edges, hidden_size).
            gate: Per-edge message scale of shape (num_edges, 1), broadcast over hidden_size.

        Returns:
            Updated node features of shape (num_nodes, hidden_size).
        """
        src, dst = edge_index[0], edge_index[1]
        msg = F.relu(x[src] + edge_attr) * gate
        aggr = th.zeros_like(x).index_add_(0, dst, msg)
        return self.nn((1 + self.eps) * x + aggr)


class GatedGINE(nn.Module):
    """GINE whose messages are gated by a per-bond scalar conditioned on a context vector.

    Stock GINE shares one message function across every collision energy and adduct: context
    enters only as an extra constant channel concatenated onto every atom's input features, so
    the model must discover any energy-by-bond interaction inside a shared MLP. This variant
    makes that interaction structural -- each bond gets a scalar conductance
    ``g_ij = 1 + tanh(MLP([...]))`` bounded to [0, 2] that depends on the context, so which bonds
    transmit information can change with energy/adduct rather than only how much the result is
    rescaled afterwards.

    The gate readout is zero-initialized, so ``g_ij == 1`` exactly at init and the module is
    bit-identical to ``GINE`` on the first forward. It is therefore a strict superset: training
    can recover stock GINE by keeping the gate at 1.

    Two modes trade cost for expressiveness:
        - "static": one gate computed from (bond, context) before the layer loop and reused at
          every layer. Tests "bond identity x context" directly and is readable as a per-bond
          lability score.
        - "dynamic": a gate recomputed per layer from the live hidden states
          (x_src, x_dst, bond, context), so the gate also sees the electronic environment the
          GNN builds up. Subsumes "static" at the cost of one gate MLP per layer.

    Attributes:
        gate_mode: Either "static" or "dynamic".
        gate_mlps: Gate readout(s); length 1 for "static", num_layers for "dynamic".
        edge_transform: Shared linear mapping raw edge features to hidden_size.
    """

    def __init__(
        self,
        hidden_size: int,
        node_feats_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        ctx_dim: int,
        gate_mode: str,
        gate_hidden_dim: int,
        **kwargs: Any,
    ) -> None:
        """Initialize GatedGINE.

        Args:
            hidden_size: Hidden layer size.
            node_feats_size: Size of node features.
            edge_feats_size: Size of edge features.
            num_layers: Number of GINE layers.
            dropout: Dropout rate.
            normalization: Normalization type ("batch", "layer", "graph", "none").
            ctx_dim: Width of the per-graph context vector conditioning the gate.
            gate_mode: "static" (one gate, bond x context) or "dynamic" (per-layer, also sees
                the live hidden states).
            gate_hidden_dim: Hidden width of the gate MLP(s).

        Raises:
            ValueError: If gate_mode is not "static" or "dynamic", or ctx_dim is not positive.
        """
        super().__init__()
        self.residual_activation = _validate_residual_activation(
            kwargs.get("residual_activation", "relu")
        )
        if gate_mode not in ("static", "dynamic"):
            raise ValueError(f"gate_mode must be 'static' or 'dynamic', got {gate_mode!r}")
        if ctx_dim <= 0:
            raise ValueError(f"ctx_dim must be positive, got {ctx_dim}")
        self.gate_mode = gate_mode

        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)

        self.layers = nn.ModuleList(
            [
                GatedGINEConv(
                    nn_module=nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size),
                    ),
                    eps=0.0,
                )
                for _ in range(num_layers)
            ]
        )

        # static: [edge_attr, ctx]; dynamic: [x_src, x_dst, edge_attr, ctx]
        gate_in = (hidden_size + ctx_dim) if gate_mode == "static" else (3 * hidden_size + ctx_dim)
        num_gates = 1 if gate_mode == "static" else num_layers
        self.gate_mlps = nn.ModuleList(
            [self._build_gate_mlp(gate_in, gate_hidden_dim) for _ in range(num_gates)]
        )

        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    @staticmethod
    def _build_gate_mlp(in_dim: int, hidden_dim: int) -> nn.Module:
        """Build a gate readout whose output is zero at init (so the gate starts at 1.0)."""
        out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), out)

    @staticmethod
    def _gate(mlp: nn.Module, feats: th.Tensor) -> th.Tensor:
        """Map gate features to a per-edge scale bounded to [0, 2], equal to 1 at zero-init."""
        return 1.0 + th.tanh(mlp(feats))

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        ctx: th.Tensor,
    ) -> th.Tensor:
        """Forward pass with context-gated messages.

        Args:
            x: Node features of shape (num_nodes, hidden_size).
            batch_index: Graph assignment per node, shape (num_nodes,).
            edge_index: Edge connectivity of shape (2, num_edges).
            edge_attr: Raw edge features of shape (num_edges, edge_feats_size).
            ctx: Per-graph context vector of shape (num_graphs, ctx_dim).

        Returns:
            Encoded node features of shape (num_nodes, hidden_size).
        """
        edge_attr = self.edge_transform(edge_attr)
        # PyG's default source_to_target flow: messages run x[src] -> x[dst].
        src, dst = edge_index[0], edge_index[1]
        ctx_e = ctx[batch_index[src]]  # broadcast per-graph context to its edges

        static_gate = None
        if self.gate_mode == "static":
            static_gate = self._gate(self.gate_mlps[0], th.cat([edge_attr, ctx_e], dim=-1))

        for i, (dropout, layer, norm) in enumerate(zip(self.dropouts, self.layers, self.norms)):
            if static_gate is not None:
                gate = static_gate
            else:
                gate = self._gate(
                    self.gate_mlps[i], th.cat([x[src], x[dst], edge_attr, ctx_e], dim=-1)
                )
            layer_out = layer(x, edge_index, edge_attr, gate)
            layer_out = self.norm_fn(layer_out, batch_index, norm)
            x = (
                _residual_branch(self.residual_activation, dropout(layer_out)) + x
            )  # Residual connection
        return x


def _assert_graph_contiguous(batch_index: th.Tensor) -> None:
    """Validate that nodes are grouped by graph in ascending order.

    ``to_dense_batch`` derives each node's slot from its position relative to the
    per-graph cumulative count, so an unsorted ``batch_index`` would silently place
    nodes in the wrong rows and scramble features rather than raise. PyG's ``Batch``
    always produces sorted assignments; this guards against a hand-built batch.
    """
    if batch_index.numel() < 2:
        return
    is_sorted = (batch_index[1:] >= batch_index[:-1]).all()
    message = "batch_index must be sorted so nodes are contiguous within each graph"
    if th.compiler.is_compiling() or batch_index.device.type == "cuda":
        th._assert_async(is_sorted, message)
    elif not bool(is_sorted):
        raise ValueError(message)


def _graph_softmax_attn(
    Q: th.Tensor,
    K: th.Tensor,
    V: th.Tensor,
    batch_index: th.Tensor,
    num_graphs: int,
) -> th.Tensor:
    """Exact per-graph softmax attention on flat PyG node tensors.

    Nodes are packed into a dense ``(num_graphs, max_nodes, ...)`` batch so scaled
    dot-product attention runs independently per molecule with padded keys masked out.

    This replaces an ELU+1 linear-attention feature map. Linear attention costs
    ``O(N·H·D²)`` against ``O(N²·H·D)`` for softmax, so it only pays off once a graph
    holds more nodes than the head dimension. This encoder runs at ``head_dim`` 96 on
    molecules whose median heavy-atom count is 20 and whose 99th percentile is 82, so
    the crossover is never reached: the feature map had to materialize two ``(N, H, D, D)``
    intermediates (about 144 KiB per node), measuring 2-5x slower and 15-20x heavier than
    the exact attention below while also being an approximation.

    Args:
        Q: Queries of shape (N, num_heads, head_dim).
        K: Keys of shape (N, num_heads, head_dim).
        V: Values of shape (N, num_heads, head_dim).
        batch_index: Graph assignments of shape (N,), sorted ascending.
        num_graphs: Number of graphs the dense batch is sized by. Passed in rather than
            derived here so the enclosing forward pays one host sync instead of one per
            layer.

    Returns:
        Attended values of shape (N, num_heads, head_dim), in the input node order.
    """
    _assert_graph_contiguous(batch_index)
    num_nodes, num_heads, head_dim = Q.shape
    dense_queries, node_mask = pyg.utils.to_dense_batch(
        Q.reshape(num_nodes, num_heads * head_dim),
        batch_index,
        batch_size=num_graphs,
    )
    dense_keys, _ = pyg.utils.to_dense_batch(
        K.reshape(num_nodes, num_heads * head_dim),
        batch_index,
        batch_size=num_graphs,
    )
    dense_values, _ = pyg.utils.to_dense_batch(
        V.reshape(num_nodes, num_heads * head_dim),
        batch_index,
        batch_size=num_graphs,
    )
    max_nodes = dense_queries.shape[1]

    def split_heads(dense: th.Tensor) -> th.Tensor:
        return dense.view(num_graphs, max_nodes, num_heads, head_dim).transpose(1, 2)

    # Padded slots are masked as keys. A graph with no nodes would leave a row with every
    # key masked, which softmax turns into NaN, so slot 0 is always left attendable: for a
    # non-empty graph it already holds a real node, and for an empty graph it yields a zero
    # row that the final mask discards anyway.
    key_mask = node_mask.clone()
    key_mask[:, 0] = True
    attended = F.scaled_dot_product_attention(
        split_heads(dense_queries),
        split_heads(dense_keys),
        split_heads(dense_values),
        attn_mask=key_mask[:, None, None, :],
    )
    return attended.transpose(1, 2)[node_mask]


class GINE_GSA(nn.Module):
    """GINE augmented with exact per-graph global self-attention.

    After each GINEConv message-passing step, a graph-scoped attention pass mixes
    information across all nodes in the same molecule, capturing long-range interactions
    that local message passing misses.

    This was once ``GINE_EFA``, whose global pass used an ELU+1 linear-attention feature
    map (Euclidean Fast Attention). The attention is now exact softmax
    (see :func:`_graph_softmax_attn` for why), so the EFA name no longer described it.
    The rename is deliberately breaking: ``gnn_type: GINE_EFA`` and any checkpoint
    carrying ``efa_*`` parameters now fail loudly rather than loading under changed math.

    Args:
        hidden_size (int): Hidden layer size.
        node_feats_size (int): Size of node features.
        edge_feats_size (int): Size of edge features.
        num_layers (int): Number of GNN + attention layers.
        dropout (float): Dropout rate.
        normalization (str): Normalization type ("batch", "layer", "graph", "none").
        num_heads (int): Number of attention heads. Must divide hidden_size. Defaults to 4.
        global_message_substitution (float): Training-only probability that an atom's whole
            normalized global-attention message is replaced, per layer, by the batch mean
            message, which carries no molecule identity. 0 (default) disables it.
        global_message_noise_std (float): Training-only Gaussian noise on the normalized
            global-attention message, as a multiple of each feature's batch standard
            deviation. 0 (default) disables it.
        global_message_swap (float): Training-only probability, per layer, that a message is
            replaced by another molecule's real message ("swap noise", a feature-level CutMix
            that keeps the original label). 0 (default) disables it.
        global_message_swap_unit (str): What one swap draw replaces: ``atom`` (an atom's whole
            message, from a random atom in the batch), ``molecule`` (every atom of a molecule,
            each from a random atom of one other molecule) or ``feature`` (one feature of one
            atom, from the same feature of a random atom in the batch).
    """

    def __init__(
        self,
        hidden_size: int,
        node_feats_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        num_heads: int = 4,
        global_message_substitution: float = 0.0,
        global_message_noise_std: float = 0.0,
        global_message_swap: float = 0.0,
        global_message_swap_unit: str = "atom",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.residual_activation = _validate_residual_activation(
            kwargs.get("residual_activation", "relu")
        )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if edge_feats_size < 0:
            raise ValueError(f"edge_feats_size must be non-negative, got {edge_feats_size}")
        if not 0.0 <= global_message_substitution < 1.0:
            raise ValueError(
                f"global_message_substitution must be in [0, 1), got {global_message_substitution}"
            )
        if global_message_noise_std < 0.0:
            raise ValueError(
                f"global_message_noise_std must be nonnegative, got {global_message_noise_std}"
            )
        if not 0.0 <= global_message_swap < 1.0:
            raise ValueError(f"global_message_swap must be in [0, 1), got {global_message_swap}")
        if global_message_swap_unit not in GLOBAL_MESSAGE_SWAP_UNITS:
            raise ValueError(
                f"global_message_swap_unit must be one of {GLOBAL_MESSAGE_SWAP_UNITS}, "
                f"got {global_message_swap_unit!r}"
            )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.global_message_substitution = global_message_substitution
        self.global_message_noise_std = global_message_noise_std
        self.global_message_swap = global_message_swap
        self.global_message_swap_unit = global_message_swap_unit

        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)

        gine_layers = []
        for _ in range(num_layers):
            apply_fn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            gine_layers.append(pyg.nn.conv.GINEConv(nn=apply_fn, eps=0.0, edge_dim=None))
        self.layers = nn.ModuleList(gine_layers)

        self.gsa_q = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.gsa_k = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.gsa_v = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.gsa_out = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )

        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.gsa_norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
        """Forward pass: local GINEConv → global self-attention → residual sum, per layer.

        Args:
            x: Node features of shape (N, hidden_size).
            batch_index: Graph assignments of shape (N,).
            edge_index: Edge connectivity of shape (2, E).
            edge_attr: Edge features of shape (E, edge_feats_size).

        Returns:
            Updated node features of shape (N, hidden_size).
        """
        edge_attr = self.edge_transform(edge_attr)
        # One host sync for the whole encoder instead of one per attention pass.
        num_graphs = int(batch_index.max().item()) + 1
        for dropout, layer, norm, gsa_norm, q_proj, k_proj, v_proj, out_proj in zip(
            self.dropouts,
            self.layers,
            self.norms,
            self.gsa_norms,
            self.gsa_q,
            self.gsa_k,
            self.gsa_v,
            self.gsa_out,
        ):
            # Local message passing (same residual pattern as GINE)
            local_out = layer(x, edge_index, edge_attr)
            local_out = self.norm_fn(local_out, batch_index, norm)
            local_out = _residual_branch(self.residual_activation, dropout(local_out)) + x

            # Exact global self-attention within each graph
            N, C = local_out.shape
            Q = q_proj(local_out).view(N, self.num_heads, self.head_dim)
            K = k_proj(local_out).view(N, self.num_heads, self.head_dim)
            V = v_proj(local_out).view(N, self.num_heads, self.head_dim)
            gsa_out = _graph_softmax_attn(Q, K, V, batch_index, num_graphs).reshape(N, C)
            gsa_out = dropout(out_proj(F.relu(gsa_out)))
            gsa_out = self.norm_fn(gsa_out, batch_index, gsa_norm)
            # After the norm, so its statistics neither absorb the noise nor carry it into eval.
            if self.training:
                gsa_out = self._perturb_global_message(gsa_out, batch_index, num_graphs)

            x = local_out + gsa_out
        return x

    def _perturb_global_message(
        self, message: th.Tensor, batch_index: th.Tensor, num_graphs: int
    ) -> th.Tensor:
        """Apply the training-only global-message noise; draw nothing when it is off.

        Applied in order. Substitution sets ``m_i <- mean_j(m_j)`` with probability ``p`` per
        atom, the mean taken over every atom in the batch and detached. Swap then replaces
        messages by detached real messages of other atoms (see ``_swap_global_message``).
        Gaussian noise then sets ``m_i <- m_i + sigma * s * eps_i`` with ``eps_i ~ N(0, I)`` and
        ``s`` the detached per-feature standard deviation of ``m`` over the batch.

        Args:
            message: Normalized global-attention messages, shape (N, hidden_size).
            batch_index: Graph assignment of each atom, sorted, shape (N,).
            num_graphs: Number of graphs in the batch.

        Returns:
            Perturbed messages, shape (N, hidden_size).
        """
        if self.global_message_substitution > 0.0:
            replace = (
                th.rand(message.shape[0], 1, device=message.device)
                < self.global_message_substitution
            )
            batch_mean = message.detach().mean(dim=0, keepdim=True)
            message = th.where(replace, batch_mean, message)
        if self.global_message_swap > 0.0:
            message = self._swap_global_message(message, batch_index, num_graphs)
        if self.global_message_noise_std > 0.0:
            feature_std = message.detach().std(dim=0, unbiased=False, keepdim=True)
            message = message + self.global_message_noise_std * feature_std * th.randn_like(message)
        return message

    def _swap_global_message(
        self, message: th.Tensor, batch_index: th.Tensor, num_graphs: int
    ) -> th.Tensor:
        """Replace messages by other atoms' real messages with probability ``p``.

        Unlike substitution by the batch mean, a swapped message is an in-distribution
        message carrying another molecule's identity, so the head cannot tell when to trust it.
        Donors are detached, so a swapped entry gets no gradient. The label is not mixed.

        - ``atom``: ``m_i <- m_j`` with probability ``p`` per atom, ``j`` uniform over the
          batch (``j`` is in atom ``i``'s own molecule with probability ``n_mol / N``).
        - ``molecule``: with probability ``p`` per molecule, each of its atoms takes the message
          of a uniformly drawn atom of one donor molecule, never the molecule itself. A batch
          of one molecule has no donor and is returned unchanged.
        - ``feature``: ``m_ic <- m_jc`` with probability ``p`` per entry, ``j`` drawn per entry.

        Args:
            message: Normalized global-attention messages, shape (N, hidden_size).
            batch_index: Graph assignment of each atom, sorted, shape (N,).
            num_graphs: Number of graphs in the batch.

        Returns:
            Messages with the swapped entries replaced, shape (N, hidden_size).
        """
        donors = message.detach()
        num_atoms, device = message.shape[0], message.device
        if self.global_message_swap_unit == "feature":
            source = th.randint(num_atoms, message.shape, device=device)
            swap = th.rand(message.shape, device=device) < self.global_message_swap
            return th.where(swap, donors.gather(0, source), message)
        if self.global_message_swap_unit == "atom":
            source = th.randint(num_atoms, (num_atoms,), device=device)
            swap = th.rand(num_atoms, 1, device=device) < self.global_message_swap
            return th.where(swap, donors[source], message)
        if num_graphs < 2:
            return message
        # 1. Pair each molecule with a different donor molecule.
        shift = th.randint(1, num_graphs, (num_graphs,), device=device)
        donor_graph = (th.arange(num_graphs, device=device) + shift) % num_graphs
        # 2. Draw each atom's donor atom uniformly within its molecule's donor.
        counts = th.bincount(batch_index, minlength=num_graphs)
        starts = th.cumsum(counts, dim=0) - counts
        atom_donor_graph = donor_graph[batch_index]
        offset = (th.rand(num_atoms, device=device) * counts[atom_donor_graph]).long()
        source = starts[atom_donor_graph] + offset
        # 3. Swap whole molecules.
        swap = th.rand(num_graphs, device=device) < self.global_message_swap
        return th.where(swap[batch_index].unsqueeze(1), donors[source], message)


class MPNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        node_feats_size: int,
        edge_feats_size: int,
        num_layers: int,
        dropout: float,
        normalization: str,
        **kwargs: Any,
    ) -> None:
        """Neural Message Passing for Quantum Chemistry
        https://arxiv.org/abs/1704.01212
        """
        super().__init__()
        self.residual_activation = _validate_residual_activation(
            kwargs.get("residual_activation", "relu")
        )
        if edge_feats_size < 0:
            raise ValueError(f"edge_feats_size must be non-negative, got {edge_feats_size}")
        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)
        # MPNN
        self.layers = []
        for _i in range(num_layers):
            edge_network = edge_network = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * hidden_size),
            )
            layer = pyg.nn.NNConv(
                hidden_size, hidden_size, edge_network, aggr="mean", root_weight=False
            )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        # setup norms and dropout
        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
        """forward."""
        edge_attr = self.edge_transform(edge_attr)
        for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
            layer_out = layer(x, edge_index, edge_attr)
            layer_out = self.norm_fn(layer_out, batch_index, norm)
            x = _residual_branch(self.residual_activation, dropout(layer_out)) + x
        return x


class GAT(nn.Module):
    """Graph attention networks. https://arxiv.org/abs/1710.10903
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
        **kwargs: Any,
    ) -> None:
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
        self.residual_activation = _validate_residual_activation(
            kwargs.get("residual_activation", "relu")
        )
        if edge_feats_size < 0:
            raise ValueError(f"edge_feats_size must be non-negative, got {edge_feats_size}")
        self.edge_transform = nn.Linear(edge_feats_size, hidden_size)
        gat_output_size = hidden_size // num_gat_heads if is_concat else hidden_size
        if is_concat and hidden_size % num_gat_heads != 0:
            raise ValueError(
                f"Ensure that the number of output channels of "
                f"'GAT' (got '{hidden_size}') is divisible "
                f"by the number of heads (got '{num_gat_heads}')"
            )
        GATConv_fn = pyg.nn.conv.GATConv if not is_v2 else pyg.nn.conv.GATv2Conv
        self.layers = []
        for _i in range(num_layers):
            layer = GATConv_fn(
                in_channels=hidden_size,
                out_channels=gat_output_size,
                heads=num_gat_heads,
                edge_dim=hidden_size,
                dropout=gat_dropout,
                concat=is_concat,
            )
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        # setup norms and dropout
        norm_mod, norm_fn = get_norm(normalization, hidden_size)
        self.norm_fn = norm_fn
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
        edge_attr = self.edge_transform(edge_attr)
        for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
            layer_out = layer(x, edge_index, edge_attr)
            layer_out = self.norm_fn(layer_out, batch_index, norm)
            x = _residual_branch(self.residual_activation, dropout(layer_out)) + x
        return x


class GGNN(nn.Module):
    def __init__(
        self,
        node_feats_size: int,
        edge_feats_size: int,
        hidden_size: int,
        num_layers: int,
        **kwargs: Any,
    ) -> None:
        """ """
        super().__init__()
        if edge_feats_size < 0:
            raise ValueError(f"edge_feats_size must be non-negative, got {edge_feats_size}")
        self.edge_transform = nn.Linear(edge_feats_size, 1)
        self.model = pyg.nn.conv.GatedGraphConv(out_channels=hidden_size, num_layers=num_layers)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
        """forward."""
        edge_weight = self.edge_transform(edge_attr).squeeze(1)
        x = self.model(x=x, edge_index=edge_index, edge_weight=edge_weight)
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
        **kwargs: Any,
    ) -> None:
        """NodeMLP"""
        super().__init__()
        self.residual_activation = _validate_residual_activation(
            kwargs.get("residual_activation", "relu")
        )

        if edge_feats_size < 0:
            raise ValueError(f"edge_feats_size must be non-negative, got {edge_feats_size}")
        if edge_feats_size > 0:
            logger.warning(
                "NodeMLP does not use edge features, but got edge_feats_size > 0, this most likely indicates a misconfiguration and wasted VRAM."
            )

        self.layers = []
        for _i in range(num_layers):
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
            self.norm_fn = lambda x, b, n: n(x, b)
        else:
            if normalization != "none":
                raise ValueError(f"unknown normalization {normalization!r}")
            norm_mod = nn.Identity()
            self.norm_fn = lambda x, b, n: n(x)
        self.norms = get_clones(norm_mod, num_layers)
        self.dropouts = get_clones(nn.Dropout(dropout), num_layers)

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
    ) -> th.Tensor:
        """forward."""
        for dropout, layer, norm in zip(self.dropouts, self.layers, self.norms):
            layer_out = layer(x)
            layer_out = self.norm_fn(layer_out, batch_index, norm)
            x = _residual_branch(self.residual_activation, dropout(layer_out)) + x
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
        output_size: int | None = None,
        use_residuals: bool = False,
    ) -> None:
        """MLP with optional residual connections.

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
            if normalization != "none":
                raise ValueError(f"unknown normalization {normalization!r}")
            norm_mod = nn.Identity()
        self.norms = get_clones(norm_mod, num_layers - 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
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
    """from the NEIMS paper (uses LeakyReLU instead of ReLU)"""

    def __init__(
        self, in_dim: int, out_dim: int, dropout: float, bottleneck_factor: float = 0.5
    ) -> None:
        super().__init__()
        bottleneck_size = int(round(bottleneck_factor * out_dim))
        self.in_batch_norm = nn.BatchNorm1d(in_dim)
        self.in_activation = nn.LeakyReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
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

    def __init__(self, input_dim: int, output_dim: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # First projection: input_dim -> rank
        self.proj1 = nn.Linear(input_dim, rank, bias=False)

        # Second projection: rank -> output_dim
        self.proj2 = nn.Linear(rank, output_dim, bias=bias)

    def forward(self, x: th.Tensor) -> th.Tensor:
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

    def __init__(
        self,
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
        log_min,
        bottleneck_factor=0.5,
    ) -> None:
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
        for _i in range(num_layers):
            self.ff_layers.append(NeimsBlock(hidden_size, hidden_size, dropout, bottleneck_factor))
        if output_map_size == -1:
            if self.bidirectional:
                # assumes gating, mass masking
                self.forw_out_layer = nn.Linear(hidden_size, self.output_size)
                self.rev_out_layer = nn.Linear(hidden_size, self.output_size)
                self.out_gate = nn.Sequential(
                    *[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()]
                )
            else:
                self.out_layer = nn.Linear(hidden_size, self.output_size)
                self.out_gate = nn.Sequential(
                    *[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()]
                )
        else:
            if self.bidirectional:
                # assumes gating, mass masking
                self.forw_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.rev_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.out_gate = nn.Sequential(
                    *[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()]
                )
            else:
                self.out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.out_gate = nn.Sequential(
                    *[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()]
                )
        if output_activation not in ["relu", "sigmoid"]:
            raise ValueError(
                f"unknown output_activation {output_activation!r}; expected 'relu' or 'sigmoid'"
            )
        if output_activation == "relu":
            self.out_activation = nn.ReLU()
        else:
            self.out_activation = nn.Sigmoid()
        self.out_normalization = lambda x: F.normalize(x, p=1, dim=1)
        self.log_min = log_min

    def _compute_output_size(self):
        mz_bins = th.arange(
            self.mz_bin_res, self.mz_max + self.mz_bin_res, self.mz_bin_res, dtype=th.float32
        )
        self.register_buffer("mz_bins", mz_bins, persistent=False)
        self.register_buffer("mzs", mz_bins - 0.5 * self.mz_bin_res, persistent=False)
        self.output_size = mz_bins.shape[0]

    def _prec_mz_to_idx(self, prec_mz):
        prec_mz_idx = th.bucketize(prec_mz, self.mz_bins.to(prec_mz.device), right=True)
        assert th.max(prec_mz_idx) < self.output_size, (prec_mz_idx, self.output_size)
        return prec_mz_idx

    def forward(
        self, input_h: th.Tensor, prec_mz: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
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
            fr = reverse_prediction(self.rev_out_layer(fh), prec_mz_idxs, self.prec_mz_offset)
            fg = self.out_gate(fh)
            fo = ff * fg + fr * (1.0 - fg)
        else:
            # apply output layer
            fo = self.out_layer(fh)
            # apply gating
            fg = self.out_gate(fh)
            fo = fg * fo
        fo = self.out_activation(fo)
        fo = mask_prediction_by_mass(fo, prec_mz_idxs, self.prec_mz_offset)
        spec = self.out_normalization(fo)
        # handle all zeroes (set to first bin by default)
        all_zero_mask = th.max(spec, dim=1)[0] <= 0.0
        all_zero_bonus = th.zeros_like(spec)
        all_zero_bonus[all_zero_mask, 0] = 1.0
        spec = spec + all_zero_bonus
        # convert dense spectrum to sparse
        mask = spec > 0.0
        pred_mzs = (self.mzs.unsqueeze(0).expand(spec.shape[0], -1))[mask]
        pred_logprobs = safelog(spec, eps=self.log_min)[mask]
        pred_batch_idxs = (
            th.arange(spec.shape[0], device=spec.device)
            .unsqueeze(1)
            .expand(-1, spec.shape[1])[mask]
        )
        pred_specs = spec
        return pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs


def _pack_variable_length(x: th.Tensor, batch_idx: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Pack flat (total, dim) into padded (batch, max_len, dim) with boolean mask.

    Args:
        x: Flat feature tensor of shape (total_elements, dim).
        batch_idx: Batch index per element of shape (total_elements,).

    Returns:
        x_padded: Padded tensor of shape (batch_size, max_len, dim).
        mask: Boolean mask of shape (batch_size, max_len), True = padding position.
    """
    batch_size = int(batch_idx.max().item()) + 1
    set_sizes = batch_idx.bincount(minlength=batch_size)  # (batch_size,)
    max_size = int(set_sizes.max().item())

    cum_starts = th.cat([th.zeros(1, dtype=th.long, device=x.device), set_sizes.cumsum(0)[:-1]])
    local_pos = th.arange(x.shape[0], device=x.device) - cum_starts[batch_idx]

    x_padded = x.new_zeros(batch_size, max_size, x.shape[-1])
    x_padded[batch_idx, local_pos] = x

    arange = th.arange(max_size, device=x.device).unsqueeze(0)
    mask = arange >= set_sizes.unsqueeze(1)  # True = padding
    return x_padded, mask


def _unpack_variable_length(x_padded: th.Tensor, batch_idx: th.Tensor) -> th.Tensor:
    """Inverse of _pack_variable_length: strips padding, returns (total, dim).

    Args:
        x_padded: Padded tensor of shape (batch_size, max_len, dim).
        batch_idx: Original batch indices of shape (total_elements,).

    Returns:
        Flat tensor of shape (total_elements, dim).
    """
    batch_size = x_padded.shape[0]
    set_sizes = batch_idx.bincount(minlength=batch_size)
    return th.cat([x_padded[i, : int(set_sizes[i].item())] for i in range(batch_size)], dim=0)


class _MAB(nn.Module):
    """Multi-head Attention Block: Q attends to KV with residual + LayerNorm + FFN.

    Args:
        dim: Feature dimension (Q, K, V all share the same dim).
        num_heads: Number of attention heads.
        dropout: Dropout applied inside attention and after the attention sublayer.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        Q: th.Tensor,
        KV: th.Tensor,
        kv_padding_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        """Args:
            Q: Query tensor of shape (B, Nq, dim).
            KV: Key/value tensor of shape (B, Nkv, dim).
            kv_padding_mask: Boolean mask (B, Nkv), True = padding (ignored).

        Returns:
            Updated query tensor of shape (B, Nq, dim).
        """
        H, _ = self.attn(Q, KV, KV, key_padding_mask=kv_padding_mask)
        Q = self.norm1(Q + self.drop(H))
        Q = self.norm2(Q + self.ffn(Q))
        return Q


class _ISAB(nn.Module):
    """Induced Set Attention Block (ISAB) from the Set Transformer paper.

    Each block runs two MABs:
      H = MAB(I, X)     — inducing points compress the input set
      out = MAB(X, H)   — input reads back from the compressed representation

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_inds: Number of learnable inducing points.
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, num_heads: int, num_inds: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.inducing_points = nn.Parameter(th.randn(1, num_inds, dim))
        self.mab_i = _MAB(dim, num_heads, dropout)  # I <- X
        self.mab_x = _MAB(dim, num_heads, dropout)  # X <- H

    def forward(self, X: th.Tensor, x_padding_mask: th.Tensor | None = None) -> th.Tensor:
        """Args:
            X: Input of shape (B, N, dim).
            x_padding_mask: Boolean mask (B, N), True = padding.

        Returns:
            Updated X of shape (B, N, dim).
        """
        B = X.shape[0]
        inducing = self.inducing_points.expand(B, -1, -1)
        H = self.mab_i(inducing, X, kv_padding_mask=x_padding_mask)  # (B, num_inds, dim)
        return self.mab_x(X, H)  # (B, N, dim) — H has no padding


class _PMA(nn.Module):
    """Pooling by Multi-head Attention (PMA) with k learned seed vectors.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        num_seeds: Number of output vectors (1 produces a single set embedding).
        dropout: Dropout rate.
    """

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.seeds = nn.Parameter(th.randn(1, num_seeds, dim))
        self.mab = _MAB(dim, num_heads, dropout)

    def forward(self, X: th.Tensor) -> th.Tensor:
        """Args:
            X: Input of shape (B, N, dim).

        Returns:
            Pooled representation of shape (B, num_seeds, dim).
        """
        return self.mab(self.seeds.expand(X.shape[0], -1, -1), X)


class SetTransformer(nn.Module):
    """Set Transformer with ISAB encoder blocks and PMA pooling.

    Implements the architecture from "Set Transformer: A Framework for
    Attention-based Permutation-Invariant Neural Networks" (Lee et al., 2019).
    Each ISAB block runs two cross-attention MABs and achieves O(N·m) complexity
    where m = num_inds. Final pooling uses PMA (learned seed vectors) instead of
    mean pooling.

    Args:
        input_dim: Feature dimension entering the ISAB blocks (after any external
            input projection).
        hidden_dim: Hidden dimension used in the output MLP projector.
        output_dim: Dimension of the output embedding.
        num_heads: Number of attention heads (must divide input_dim).
        num_inds: Number of inducing points per ISAB block.
        num_blocks: Number of ISAB blocks.
        dropout: Dropout rate applied inside attention and FFN sublayers.
        return_nodes: If True, return per-element outputs of shape
            (total_elements, output_dim); if False, return per-set embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_inds: int = 32,
        num_blocks: int = 2,
        dropout: float = 0.1,
        return_nodes: bool = False,
    ) -> None:
        super().__init__()
        self.return_nodes = return_nodes
        self.isab_blocks = nn.ModuleList(
            [_ISAB(input_dim, num_heads, num_inds, dropout) for _ in range(num_blocks)]
        )
        if return_nodes:
            self.out_proj = nn.Linear(input_dim, output_dim)
        else:
            self.pma = _PMA(input_dim, num_heads, num_seeds=1, dropout=dropout)
            self.out_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: th.Tensor, batch_idx: th.Tensor | None = None) -> th.Tensor:
        """Forward pass for the Set Transformer.

        Args:
            x: Input tensor.
                - Fixed-size path (batch_idx is None): shape (B, N, input_dim).
                - Variable-size path (batch_idx provided): shape (total, input_dim).
            batch_idx: Batch index per element of shape (total,). Required for
                variable-size sets; omit for uniformly-sized batches.

        Returns:
            - return_nodes=False: set embeddings of shape (B, output_dim).
            - return_nodes=True: per-element outputs of shape (total, output_dim).
        """
        if batch_idx is None:
            for isab in self.isab_blocks:
                x = isab(x)
            if self.return_nodes:
                return self.out_proj(x)
            return self.out_proj(self.pma(x).squeeze(1))

        x_padded, mask = _pack_variable_length(x, batch_idx)
        for isab in self.isab_blocks:
            x_padded = isab(x_padded, x_padding_mask=mask)

        if self.return_nodes:
            return _unpack_variable_length(self.out_proj(x_padded), batch_idx)
        return self.out_proj(self.pma(x_padded).squeeze(1))


def reverse_prediction(
    raw_prediction: th.Tensor, prec_mass_idx: th.Tensor, prec_mass_offset: int
) -> th.Tensor:
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    rev_prediction = th.flip(raw_prediction, dims=(1,))
    # convention is to shift right, so we express as negative to go left
    offset_idx = th.minimum(
        max_idx * th.ones_like(prec_mass_idx), prec_mass_idx + prec_mass_offset + 1
    )
    shifts = -(max_idx - offset_idx)
    gather_idx = (
        th.arange(max_idx, device=raw_prediction.device).unsqueeze(0).expand(batch_size, max_idx)
    )
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
    offset_rev_prediction = th.gather(rev_prediction, 1, gather_idx)
    # you could mask_prediction_by_mass here but it's unnecessary
    return offset_rev_prediction


class Set2SetPooling(nn.Module):
    """Wrapper for Set2Set that handles batch argument correctly."""

    def __init__(self, node_dim: int, processing_steps: int = 3) -> None:
        super().__init__()
        self.set2set = pyg.nn.Set2Set(node_dim, processing_steps=processing_steps)
        self.linear = nn.Linear(2 * node_dim, node_dim)

    def forward(self, x: th.Tensor, batch: th.Tensor) -> th.Tensor:
        """Forward pass that accepts batch argument like other PyG aggregations."""
        x = self.set2set(x, batch)
        x = self.linear(x)
        return x


def build_pool_module(pool_type: str, node_dim: int) -> nn.Module:
    """Method to build pooling layer for given pool_type and dim
        node dim is required for attention layer.
    Args:
        pool_type (str): pooling type in ['sum','mean','max','attention','set2set','set_transformer','softmax','none']
        node_dim (int): node_dim, used for attention and set_transformer
    Returns:
        nn.Module: pooling module
    """

    if pool_type == "sum":
        pool_module = aggr.SumAggregation()
    elif pool_type == "mean":
        pool_module = aggr.MeanAggregation()
    elif pool_type == "max":
        pool_module = aggr.MaxAggregation()
    elif pool_type == "attention":
        pool_module = aggr.AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, 1))
        )
    elif pool_type == "set_transformer":
        # use SetTransformerAggregation from pyg
        # TODO pass channels, num_seed_points, num_encoder_blocks, num_decoder_blocks, heads, concat, layer_norm, dropout
        raise NotImplementedError
        # pool_module = aggr.SetTransformerAggregation(
        #   channels=node_dim,
        #   num_seed_points=16,
        #   num_encoder_blocks=1,
        #   num_decoder_blocks=1,
        #   heads=4,
        #   concat=True,
        #   layer_norm=False,
        #   dropout= 0.1
        # )
    elif pool_type == "equilibrium":
        raise NotImplementedError
        # TODO pass in_channels and out_channels
        # pool_module = aggr.EquilibriumAggregation(
        #   in_channels=node_dim,
        #   out_channels=node_dim
        # )
    elif pool_type == "softmax":
        pool_module = aggr.SoftmaxAggregation(learn=True)
    elif pool_type == "mean_max_sum":
        # raise NotImplementedError
        pool_module = aggr.MultiAggregation(["mean", "max", "sum"])
    elif pool_type == "mean_std_softmax":
        # raise NotImplementedError
        pool_module = aggr.MultiAggregation(["mean", "std", aggr.SoftmaxAggregation(learn=True)])
    else:
        assert pool_type == "none", pool_type
        # just return 0s
        pool_module = ZerosAggregation()
    return PoolWithDimSizeAdapter(pool_module)


def mask_prediction_by_mass(
    raw_prediction: th.Tensor, prec_mass_idx: th.Tensor, prec_mass_offset: int
) -> th.Tensor:
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    idx = th.arange(max_idx, device=prec_mass_idx.device)
    mask = (idx.unsqueeze(0) <= (prec_mass_idx.unsqueeze(1) + prec_mass_offset)).float()
    return mask * raw_prediction
