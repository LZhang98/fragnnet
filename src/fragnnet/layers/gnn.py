"""Graph neural network encoders selected by ``gnn_type``.

- GNN: dispatches ``gnn_type`` to an encoder below, ``Graphormer``/``GINE_GRAPHORMER``
  (layers/graphormer.py) or ``GINE_GSA`` (layers/gine_gsa.py).
- GINE: Graph Isomorphism Network with Edge Features.
- GatedGINE: GINE with context-gated messages (``GINE_GATED``).
- MPNN: Neural Message Passing for Quantum Chemistry.
- NodeMLP: per-node MLP without message passing.
"""

import logging
from typing import Any

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from fragnnet.layers.gine_gsa import GINE_GSA
from fragnnet.layers.graphormer import GINEGraphormer, Graphormer
from fragnnet.layers.norm_residual import get_norm, residual_branch, validate_residual_activation
from fragnnet.utils.nn_utils import get_clones

logger = logging.getLogger(__name__)


class GNN(nn.Module):
    """
    Generic GNN Class.
    This class serves as a wrapper for different types of Graph Neural Networks (GNNs),
    such as GINE, GINE_GSA, Graphormer, NodeMLP and MPNN.

    Args:
        hidden_size (int): Hidden layer size.
        num_layers (int): Number of GNN layers.
        node_feats_size (int): Size of node features.
        edge_feats_size (int): Size of edge features.
        gnn_type (str): Type of GNN to use (e.g., "GINE", "MPNN", "NodeMLP").
        dropout (float): Dropout rate.
        normalization (str): Type of normalization to use ("batch", "layer", etc.).
        atom_mask (float): Training-only probability, drawn per atom and forward pass, that an
            atom's projected input is replaced by a learned mask vector (attribute masking,
            Hu et al. 2020). The atom keeps its bonds, so the encoder must infer it from its
            neighbours. 0 (default) creates no mask vector and draws no random numbers.
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
        atom_mask: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.residual_activation = validate_residual_activation(
            kwargs.get("residual_activation", "relu")
        )
        if not 0.0 <= atom_mask < 1.0:
            raise ValueError(f"atom_mask must be in [0, 1), got {atom_mask}")
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
        else:
            raise NotImplementedError(self.gnn_type)

        # Created last and zero-initialized, so it draws no random numbers and every other
        # parameter initializes exactly as without it.
        self.atom_mask = atom_mask
        if atom_mask > 0.0:
            self.atom_mask_embedding = nn.Parameter(th.zeros(self.hidden_size))

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
                Required by gnn_type="GINE_GATED" and by a "GINE_GSA" built with a context
                token (``context_token_dim > 0``), ignored by every other type.

        Returns:
            torch.Tensor: Encoded node features.

        Raises:
            ValueError: If gnn_type="GINE_GATED" and ctx is None.
        """
        z = self.input_project(x)
        if self.training and self.atom_mask > 0.0:
            masked = th.rand(z.shape[0], 1, device=z.device) < self.atom_mask
            z = th.where(masked, self.atom_mask_embedding, z)
        if self.gnn_type == "GINE_GATED":
            if ctx is None:
                raise ValueError("gnn_type='GINE_GATED' requires ctx, got None")
            z = self.gnn(z, batch_index, edge_index, edge_attr, ctx)
        elif self.gnn_type == "GINE_GSA" and self.gnn.context_token_dim > 0:
            z = self.gnn(z, batch_index, edge_index, edge_attr, context=ctx)
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
        self.residual_activation = validate_residual_activation(
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
                residual_branch(self.residual_activation, dropout(layer_out)) + x
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
        self.residual_activation = validate_residual_activation(
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
                residual_branch(self.residual_activation, dropout(layer_out)) + x
            )  # Residual connection
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
        **kwargs: Any,
    ) -> None:
        """Neural Message Passing for Quantum Chemistry
        https://arxiv.org/abs/1704.01212
        """
        super().__init__()
        self.residual_activation = validate_residual_activation(
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
            x = residual_branch(self.residual_activation, dropout(layer_out)) + x
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
        self.residual_activation = validate_residual_activation(
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
            x = residual_branch(self.residual_activation, dropout(layer_out)) + x
        return x
