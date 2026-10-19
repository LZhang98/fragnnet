"""GINE_GSA: GINE with exact per-graph global self-attention (``gnn_type: GINE_GSA``).

MASSter's recipe layer; it stays a shared layer until its move to ``models/masster``
(docs/plans/repo_reorg_plan.md, Phase 8 follow-ups).
"""

from typing import Any

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

from fragnnet.layers.norm_residual import get_norm, residual_branch, validate_residual_activation
from fragnnet.utils.nn_utils import get_clones

GLOBAL_MESSAGE_SWAP_UNITS = ("atom", "molecule", "feature")


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


def _nodes_per_graph(batch_index: th.Tensor, num_graphs: int) -> th.Tensor:
    """Node count of each graph; a sync-free ``bincount`` for ids known to be in range."""
    return batch_index.new_zeros(num_graphs).index_add_(0, batch_index, th.ones_like(batch_index))


def _graph_softmax_attn(
    Q: th.Tensor,
    K: th.Tensor,
    V: th.Tensor,
    batch_index: th.Tensor,
    num_graphs: int,
    extra_keys: th.Tensor | None = None,
    extra_values: th.Tensor | None = None,
    max_nodes: int | None = None,
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

    With ``extra_keys``/``extra_values``, each graph gains one more key/value slot that every
    node of that graph attends to inside the same softmax (a per-graph memory token, e.g. the
    spectrum's context). The slot issues no query and is never updated.

    Args:
        Q: Queries of shape (N, num_heads, head_dim).
        K: Keys of shape (N, num_heads, head_dim).
        V: Values of shape (N, num_heads, head_dim).
        batch_index: Graph assignments of shape (N,), sorted ascending.
        num_graphs: Number of graphs the dense batch is sized by. Passed in rather than
            derived here so the enclosing forward pays one host sync instead of one per
            layer.
        extra_keys: Optional per-graph key of shape (num_graphs, num_heads, head_dim).
        extra_values: Per-graph value of the same shape, given exactly with ``extra_keys``.
        max_nodes: Largest node count of any graph, when the caller already holds it on the
            host; derived here (one host sync) when omitted. Must not be below the true maximum.

    Returns:
        Attended values of shape (N, num_heads, head_dim), in the input node order.

    Raises:
        ValueError: If only one of ``extra_keys``/``extra_values`` is given, or their shape
            is not ``(num_graphs, num_heads, head_dim)``.
    """
    _assert_graph_contiguous(batch_index)
    num_nodes, num_heads, head_dim = Q.shape
    # The packing PyG's to_dense_batch performs, with the slot layout computed once for Q, K and
    # V and without its host syncs: node i of graph g lands in dense row g * max_nodes + slot.
    # batch_index values are in [0, num_graphs): num_graphs comes from batch_index.max().
    counts = _nodes_per_graph(batch_index, num_graphs)
    if max_nodes is None:
        max_nodes = int(counts.max())
    slot = th.arange(num_nodes, device=Q.device) - (counts.cumsum(0) - counts)[batch_index]
    flat_index = slot + batch_index * max_nodes
    node_mask = th.zeros(num_graphs * max_nodes, dtype=th.bool, device=Q.device)
    node_mask = node_mask.index_fill_(0, flat_index, True).view(num_graphs, max_nodes)

    def pack(nodes: th.Tensor) -> th.Tensor:
        dense = nodes.new_zeros(num_graphs * max_nodes, num_heads * head_dim)
        dense = dense.index_copy(0, flat_index, nodes.reshape(num_nodes, num_heads * head_dim))
        return dense.view(num_graphs, max_nodes, num_heads * head_dim)

    dense_queries, dense_keys, dense_values = pack(Q), pack(K), pack(V)

    def split_heads(dense: th.Tensor) -> th.Tensor:
        return dense.view(num_graphs, dense.shape[1], num_heads, head_dim).transpose(1, 2)

    # Padded slots are masked as keys. A graph with no nodes would leave a row with every
    # key masked, which softmax turns into NaN, so slot 0 is always left attendable: for a
    # non-empty graph it already holds a real node, and for an empty graph it yields a zero
    # row that the final mask discards anyway.
    key_mask = node_mask.clone()
    key_mask[:, 0] = True
    if (extra_keys is None) != (extra_values is None):
        raise ValueError("extra_keys and extra_values must be given together")
    if extra_keys is not None:
        expected = (num_graphs, num_heads, head_dim)
        if extra_keys.shape != expected or extra_values.shape != expected:
            raise ValueError(f"extra_keys and extra_values must have shape {expected}")
        # The per-graph slot goes after the node slots and is always attendable.
        dense_keys = th.cat((dense_keys, extra_keys.reshape(num_graphs, 1, -1)), dim=1)
        dense_values = th.cat((dense_values, extra_values.reshape(num_graphs, 1, -1)), dim=1)
        key_mask = th.cat((key_mask, key_mask.new_ones((num_graphs, 1))), dim=1)
    attended = F.scaled_dot_product_attention(
        split_heads(dense_queries),
        split_heads(dense_keys),
        split_heads(dense_values),
        attn_mask=key_mask[:, None, None, :],
    )
    return attended.transpose(1, 2)[batch_index, slot]


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
        bond_dropout (float): Training-only probability, drawn per chemical bond and forward
            pass, that the bond is removed from local message passing in every layer, both
            directions together (DropEdge). Kept messages are scaled by 1/(1 - p), so each atom's
            summed message keeps its expectation. Global attention is untouched. 0 (default)
            disables it.
        context_token_dim (int): Width of a per-graph context vector attended by every layer's
            global attention. Each layer projects the context to one extra key and value per
            graph, inside the same softmax as the graph's atoms, so each atom weighs the context
            per layer and head. The projections are built on a forked RNG, so every other
            parameter initializes as without them. 0 (default) creates nothing.
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
        bond_dropout: float = 0.0,
        context_token_dim: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.residual_activation = validate_residual_activation(
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
        if not 0.0 <= bond_dropout < 1.0:
            raise ValueError(f"bond_dropout must be in [0, 1), got {bond_dropout}")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.global_message_substitution = global_message_substitution
        self.global_message_noise_std = global_message_noise_std
        self.global_message_swap = global_message_swap
        self.global_message_swap_unit = global_message_swap_unit
        self.bond_dropout = bond_dropout

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

        if context_token_dim < 0:
            raise ValueError(f"context_token_dim must be nonnegative, got {context_token_dim}")
        self.context_token_dim = context_token_dim
        self.gsa_context_k: nn.ModuleList | None = None
        self.gsa_context_v: nn.ModuleList | None = None
        if context_token_dim > 0:
            # Forked RNG: enabling the token leaves every other parameter's initialization as is.
            with th.random.fork_rng(devices=[]):
                self.gsa_context_k = nn.ModuleList(
                    [nn.Linear(context_token_dim, hidden_size) for _ in range(num_layers)]
                )
                self.gsa_context_v = nn.ModuleList(
                    [nn.Linear(context_token_dim, hidden_size) for _ in range(num_layers)]
                )

    def forward(
        self,
        x: th.Tensor,
        batch_index: th.Tensor,
        edge_index: th.Tensor,
        edge_attr: th.Tensor,
        context: th.Tensor | None = None,
    ) -> th.Tensor:
        """Forward pass: local GINEConv → global self-attention → residual sum, per layer.

        Args:
            x: Node features of shape (N, hidden_size).
            batch_index: Graph assignments of shape (N,).
            edge_index: Edge connectivity of shape (2, E).
            edge_attr: Edge features of shape (E, edge_feats_size).
            context: Per-graph context of shape (num_graphs, context_token_dim), attended by
                every layer's global attention; given exactly when ``context_token_dim > 0``.

        Returns:
            Updated node features of shape (N, hidden_size).

        Raises:
            ValueError: If ``context`` is given without a context token, missing with one, or
                not shaped ``(num_graphs, context_token_dim)``.
        """
        edge_attr = self.edge_transform(edge_attr)
        # GINEConv's message is relu(x_j + e_ij), and relu(w x_j + w e_ij) = w relu(x_j + e_ij)
        # for w > 0. Passing the pair (w x, x) with w e therefore scales the kept messages by w
        # and leaves the root term (1 + eps) x_i unscaled.
        drop_bonds = self.training and self.bond_dropout > 0.0
        if drop_bonds:
            edge_index, edge_attr = self._drop_bonds(edge_index, edge_attr, x.shape[0])
            message_scale = 1.0 / (1.0 - self.bond_dropout)
            edge_attr = edge_attr * message_scale
        # One host sync for the whole encoder instead of one per attention pass.
        # The largest graph is the longest run of equal ids, found without knowing num_graphs
        # so both values come back in the same transfer (batch_index is sorted, see
        # _assert_graph_contiguous).
        positions = th.arange(batch_index.shape[0], device=batch_index.device)
        is_start = th.ones_like(batch_index, dtype=th.bool)
        is_start[1:] = batch_index[1:] != batch_index[:-1]
        run_slot = positions - th.cummax(th.where(is_start, positions, 0), dim=0).values
        num_graphs, max_nodes = (
            th.stack((batch_index.max(), run_slot.max() + 1)).tolist()
        )
        num_graphs += 1
        if (self.context_token_dim > 0) != (context is not None):
            raise ValueError("context must be given exactly when context_token_dim is positive")
        if context is not None and context.shape != (num_graphs, self.context_token_dim):
            raise ValueError(
                f"context must have shape ({num_graphs}, {self.context_token_dim}), "
                f"got {tuple(context.shape)}"
            )
        for layer_index, (
            dropout,
            layer,
            norm,
            gsa_norm,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        ) in enumerate(
            zip(
                self.dropouts,
                self.layers,
                self.norms,
                self.gsa_norms,
                self.gsa_q,
                self.gsa_k,
                self.gsa_v,
                self.gsa_out,
            )
        ):
            # Local message passing (same residual pattern as GINE)
            local_out = layer((x * message_scale, x) if drop_bonds else x, edge_index, edge_attr)
            local_out = self.norm_fn(local_out, batch_index, norm)
            local_out = residual_branch(self.residual_activation, dropout(local_out)) + x

            # Exact global self-attention within each graph, plus the context slot when on.
            N, C = local_out.shape
            Q = q_proj(local_out).view(N, self.num_heads, self.head_dim)
            K = k_proj(local_out).view(N, self.num_heads, self.head_dim)
            V = v_proj(local_out).view(N, self.num_heads, self.head_dim)
            context_keys = context_values = None
            if context is not None:
                heads = (num_graphs, self.num_heads, self.head_dim)
                context_keys = self.gsa_context_k[layer_index](context).view(heads)
                context_values = self.gsa_context_v[layer_index](context).view(heads)
            gsa_out = _graph_softmax_attn(
                Q, K, V, batch_index, num_graphs, context_keys, context_values, max_nodes
            ).reshape(N, C)
            gsa_out = dropout(out_proj(F.relu(gsa_out)))
            gsa_out = self.norm_fn(gsa_out, batch_index, gsa_norm)
            # After the norm, so its statistics neither absorb the noise nor carry it into eval.
            if self.training:
                gsa_out = self._perturb_global_message(gsa_out, batch_index, num_graphs)

            x = local_out + gsa_out
        return x

    def _drop_bonds(
        self, edge_index: th.Tensor, edge_attr: th.Tensor, num_nodes: int
    ) -> tuple[th.Tensor, th.Tensor]:
        """Remove each chemical bond, both of its directed edges together, with probability p.

        A bond is the unordered atom pair of its edges, so no edge ordering is assumed.

        Args:
            edge_index: Directed edges, shape (2, E).
            edge_attr: Transformed edge features, shape (E, hidden_size).
            num_nodes: Number of atoms in the batch.

        Returns:
            The kept edges, shape (2, E'), and their features, shape (E', hidden_size).
        """
        low = th.minimum(edge_index[0], edge_index[1])
        high = th.maximum(edge_index[0], edge_index[1])
        bonds, bond_of_edge = th.unique(low * num_nodes + high, return_inverse=True)
        kept_bonds = th.rand(bonds.shape[0], device=edge_index.device) >= self.bond_dropout
        kept = kept_bonds[bond_of_edge]
        return edge_index[:, kept], edge_attr[kept]

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
