"""Graphormer: global molecular attention with degree, distance and bond biases, and
the GINE-Graphormer hybrid (``gnn_type`` ``Graphormer`` and ``GINE_GRAPHORMER``).
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg


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
