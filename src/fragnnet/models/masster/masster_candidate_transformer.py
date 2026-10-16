"""Candidate intensity transformer: set attention over a precursor's candidate fragments."""

from __future__ import annotations

import math
from typing import Literal

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fragnnet.utils.misc_utils import _validate_tensor_condition


def _build_feedforward(
    model_dim: int,
    feedforward_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build the position-wise feed-forward block shared by both attention layers."""
    return nn.Sequential(
        nn.Linear(model_dim, feedforward_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(feedforward_dim, model_dim),
    )


def _substitute_set_message(
    candidate_updates: th.Tensor,
    candidate_padding_mask: th.Tensor,
    probability: float,
) -> th.Tensor:
    """Replace a random share of candidates' set messages with their set's mean message.

    ``candidate_updates`` is ``[molecules, candidates, dim]``: what each candidate receives from
    its candidate set in one attention layer, before the residual add. Each real candidate
    independently, with ``probability``, receives instead the detached mean of the messages over
    its own set's real candidates. It keeps the set summary and loses whatever it retrieved from
    the set for itself. Padded slots are left out of the mean and never replaced.
    """
    real = ~candidate_padding_mask
    weights = real.unsqueeze(-1).to(candidate_updates.dtype)
    set_mean = (candidate_updates * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    replace = (
        th.rand(candidate_padding_mask.shape, device=candidate_updates.device) < probability
    ) & real
    return th.where(
        replace.unsqueeze(-1),
        set_mean.detach().expand_as(candidate_updates),
        candidate_updates,
    )


def _drop_set_keys(candidate_padding_mask: th.Tensor, probability: float) -> th.Tensor:
    """Hide a random share of each set's real candidates from the keys of its set summary.

    Returns a key padding mask, True on padded slots and on hidden candidates. Each real
    candidate is hidden independently with ``probability``. A set whose real candidates would
    all be hidden keeps its first one visible, so no latent query attends over an empty key set.
    Hidden candidates still query the summary and are still predicted; only what the summary
    carries about them is removed.
    """
    real = ~candidate_padding_mask
    draws = th.rand(candidate_padding_mask.shape, device=candidate_padding_mask.device)
    hidden = real & (draws < probability)
    emptied = ~(real & ~hidden).any(dim=1, keepdim=True)
    first_real = real & (real.cumsum(dim=1) == 1)
    return candidate_padding_mask | (hidden & ~(emptied & first_real))


class _InducedCandidateAttentionLayer(nn.Module):
    """Exchange information between candidate tokens through a small latent set."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        norm_placement: Literal["post", "pre"] = "post",
        qk_norm: bool = False,
        ce_feature_dim: int = 0,
        set_message_substitution: float = 0.0,
        set_key_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_placement not in {"post", "pre"}:
            raise ValueError(f"unknown norm_placement: {norm_placement}")
        if qk_norm:
            raise ValueError("qk_norm requires attention_type='full'")
        if ce_feature_dim:
            raise ValueError("ce_film requires attention_type='full'")
        if not 0.0 <= set_message_substitution < 1.0:
            raise ValueError(
                f"set_message_substitution must be in [0, 1), got {set_message_substitution}"
            )
        if not 0.0 <= set_key_dropout < 1.0:
            raise ValueError(f"set_key_dropout must be in [0, 1), got {set_key_dropout}")
        self.norm_placement = norm_placement
        # Training-only probability that a candidate's set message is replaced by its set's
        # detached mean message (arm sm1); 0 draws no random numbers.
        self.set_message_substitution = float(set_message_substitution)
        # Training-only probability that a candidate is hidden from the keys the inducing
        # tokens summarize, while it is still predicted (arm sd1); 0 draws no random numbers.
        self.set_key_dropout = float(set_key_dropout)
        self.latent_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.candidate_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.latent_attention_norm = nn.LayerNorm(model_dim)
        self.latent_feedforward_norm = nn.LayerNorm(model_dim)
        self.candidate_attention_norm = nn.LayerNorm(model_dim)
        self.candidate_feedforward_norm = nn.LayerNorm(model_dim)
        self.latent_feedforward = _build_feedforward(
            model_dim,
            feedforward_dim,
            dropout,
        )
        self.candidate_feedforward = _build_feedforward(
            model_dim,
            feedforward_dim,
            dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        candidate_tokens: th.Tensor,
        inducing_tokens: th.Tensor | None,
        candidate_padding_mask: th.Tensor,
        pair_bias: th.Tensor | None = None,
        ce_features: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        if pair_bias is not None:
            raise ValueError("induced candidate attention cannot apply a pairwise bias")
        if ce_features is not None:
            raise ValueError("induced candidate attention cannot apply CE modulation")
        if inducing_tokens is None:
            raise ValueError("induced candidate attention requires a latent token set")
        # Pre-norm normalizes the query stream only; keys and values arrive already
        # normalized from the previous block, so no extra norm modules are introduced and
        # the parameter count is identical across the two placements.
        pre = self.norm_placement == "pre"
        latent_query = self.latent_attention_norm(inducing_tokens) if pre else inducing_tokens
        key_padding_mask = candidate_padding_mask
        if self.training and self.set_key_dropout > 0.0:
            key_padding_mask = _drop_set_keys(candidate_padding_mask, self.set_key_dropout)
        latent_updates, _ = self.latent_attention(
            latent_query,
            candidate_tokens,
            candidate_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if pre:
            inducing_tokens = inducing_tokens + self.dropout(latent_updates)
            inducing_tokens = inducing_tokens + self.dropout(
                self.latent_feedforward(self.latent_feedforward_norm(inducing_tokens))
            )
        else:
            inducing_tokens = self.latent_attention_norm(
                inducing_tokens + self.dropout(latent_updates)
            )
            inducing_tokens = self.latent_feedforward_norm(
                inducing_tokens + self.dropout(self.latent_feedforward(inducing_tokens))
            )

        candidate_query = (
            self.candidate_attention_norm(candidate_tokens) if pre else candidate_tokens
        )
        candidate_updates, _ = self.candidate_attention(
            candidate_query,
            inducing_tokens,
            inducing_tokens,
            need_weights=False,
        )
        if self.training and self.set_message_substitution > 0.0:
            candidate_updates = _substitute_set_message(
                candidate_updates,
                candidate_padding_mask,
                self.set_message_substitution,
            )
        if pre:
            candidate_tokens = candidate_tokens + self.dropout(candidate_updates)
            candidate_tokens = candidate_tokens + self.dropout(
                self.candidate_feedforward(self.candidate_feedforward_norm(candidate_tokens))
            )
            return candidate_tokens, inducing_tokens
        candidate_tokens = self.candidate_attention_norm(
            candidate_tokens + self.dropout(candidate_updates)
        )
        candidate_tokens = self.candidate_feedforward_norm(
            candidate_tokens + self.dropout(self.candidate_feedforward(candidate_tokens))
        )
        return candidate_tokens, inducing_tokens


class _FullCandidateAttentionLayer(nn.Module):
    """Let every candidate attend directly to every other candidate in its molecule.

    The induced layer routes all candidate-to-candidate information through a fixed
    latent set, which caps the rank of that interaction at the inducing-point count.
    At the configured unique-fragment budget the quadratic cost is affordable, so this
    layer keeps the pairwise interaction exact. Normalization placement, feed-forward
    shape, and dropout match :class:`_InducedCandidateAttentionLayer` so that an A/B
    between the two isolates the attention topology.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        norm_placement: Literal["post", "pre"] = "post",
        qk_norm: bool = False,
        ce_feature_dim: int = 0,
        set_message_substitution: float = 0.0,
        set_key_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_placement not in {"post", "pre"}:
            raise ValueError(f"unknown norm_placement: {norm_placement}")
        if set_message_substitution:
            raise ValueError("set_message_substitution requires attention_type='induced'")
        if set_key_dropout:
            raise ValueError("set_key_dropout requires attention_type='induced'")
        self.norm_placement = norm_placement
        self.candidate_attention = _BiasedSelfAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            ce_feature_dim=ce_feature_dim,
        )
        self.candidate_attention_norm = nn.LayerNorm(model_dim)
        self.candidate_feedforward_norm = nn.LayerNorm(model_dim)
        self.candidate_feedforward = _build_feedforward(
            model_dim,
            feedforward_dim,
            dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        candidate_tokens: th.Tensor,
        inducing_tokens: th.Tensor | None,
        candidate_padding_mask: th.Tensor,
        pair_bias: th.Tensor | None = None,
        ce_features: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        # Padded slots are masked as keys, so they never reach a real candidate even
        # though their own rows keep accumulating values. Every molecule contributes at
        # least one real candidate, so no query row attends over a fully masked key set.
        pre = self.norm_placement == "pre"
        attention_input = (
            self.candidate_attention_norm(candidate_tokens) if pre else candidate_tokens
        )
        candidate_updates = self.candidate_attention(
            attention_input,
            candidate_padding_mask,
            pair_bias=pair_bias,
            ce_features=ce_features,
        )
        if pre:
            candidate_tokens = candidate_tokens + self.dropout(candidate_updates)
            candidate_tokens = candidate_tokens + self.dropout(
                self.candidate_feedforward(self.candidate_feedforward_norm(candidate_tokens))
            )
            return candidate_tokens, inducing_tokens
        candidate_tokens = self.candidate_attention_norm(
            candidate_tokens + self.dropout(candidate_updates)
        )
        candidate_tokens = self.candidate_feedforward_norm(
            candidate_tokens + self.dropout(self.candidate_feedforward(candidate_tokens))
        )
        return candidate_tokens, inducing_tokens


class _PositionwiseCandidateLayer(nn.Module):
    """The feed-forward half of a candidate layer, with no exchange between candidates.

    Each token is updated from itself alone, so candidates compete only through the output
    normalization. This is the allocator's no-attention floor. Its feed-forward shape, dropout and
    normalization placement match the attention layers, so an A/B isolates what candidate
    exchange adds.
    """

    def __init__(
        self,
        model_dim: int,
        _num_heads: int,
        feedforward_dim: int,
        dropout: float,
        norm_placement: Literal["post", "pre"] = "post",
        qk_norm: bool = False,
        ce_feature_dim: int = 0,
        set_message_substitution: float = 0.0,
        set_key_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if norm_placement not in {"post", "pre"}:
            raise ValueError(f"unknown norm_placement: {norm_placement}")
        if qk_norm or ce_feature_dim or set_message_substitution or set_key_dropout:
            raise ValueError(
                "attention_type='none' has no attention for qk_norm, ce_film, "
                "set_message_substitution or set_key_dropout to act on"
            )
        self.norm_placement = norm_placement
        self.candidate_feedforward_norm = nn.LayerNorm(model_dim)
        self.candidate_feedforward = _build_feedforward(
            model_dim,
            feedforward_dim,
            dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        candidate_tokens: th.Tensor,
        inducing_tokens: th.Tensor | None,
        candidate_padding_mask: th.Tensor | None,
        pair_bias: th.Tensor | None = None,
        ce_features: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor | None]:
        if self.norm_placement == "pre":
            update = self.candidate_feedforward(self.candidate_feedforward_norm(candidate_tokens))
            return candidate_tokens + self.dropout(update), inducing_tokens
        candidate_tokens = self.candidate_feedforward_norm(
            candidate_tokens + self.dropout(self.candidate_feedforward(candidate_tokens))
        )
        return candidate_tokens, inducing_tokens


# Common neutral-loss masses (Da). Used only to initialize the learnable RBF centres of
# :class:`PairwiseMassBias`; the model is free to move them.
_NEUTRAL_LOSS_CENTRES = (
    0.0,
    2.0157,  # H2
    17.0265,  # NH3
    18.0106,  # H2O
    27.9949,  # CO
    28.0313,  # C2H4
    43.9898,  # CO2
    46.0055,  # HCOOH
)


class PairwiseMassBias(nn.Module):
    """Additive attention bias on the signed mass difference between two candidates.

    Candidate competition in MS/MS is structured along a mass axis: a fragment and its
    water-loss or CO-loss daughter compete for the same ion current, and that relation is
    a fixed mass offset regardless of the molecule. Content-based attention over GNN
    embeddings has to rediscover it for every scaffold, so the relation is supplied here
    as a radial-basis bias, the MS/MS analog of the Gaussian distance kernels used by
    3-D molecular transformers.

    The bias is shared across heads. A per-head variant would require materializing an
    ``(molecules, heads, candidates, candidates)`` tensor, which at the configured
    candidate budget costs one head-multiple more memory than the attention itself.
    """

    def __init__(
        self,
        num_centres: int = len(_NEUTRAL_LOSS_CENTRES),
        signed: bool = True,
        ce_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if num_centres <= 0:
            raise ValueError("num_centres must be positive")
        if ce_feature_dim < 0:
            raise ValueError("ce_feature_dim must be nonnegative")
        centres = th.tensor(_NEUTRAL_LOSS_CENTRES, dtype=th.float32)
        if num_centres <= centres.numel():
            centres = centres[:num_centres]
        else:
            extra = th.linspace(
                float(centres[-1]) + 10.0,
                float(centres[-1]) + 10.0 * (num_centres - centres.numel()),
                num_centres - centres.numel(),
            )
            centres = th.cat([centres, extra])
        self.signed = signed
        if signed:
            # Losing a neutral and gaining one are different events, so mirror the centres.
            centres = th.cat([centres, -centres[1:]])
        self.centres = nn.Parameter(centres.clone())
        self.log_widths = nn.Parameter(th.full((centres.numel(),), math.log(4.0)))
        self.weights = nn.Parameter(th.zeros(centres.numel()))
        # Collision energy changes WHICH neutral-loss relations dominate, so the RBF
        # weights may depend on it. Zero-initialized, so enabling this keeps the module a
        # no-op at step 0, and after training w_r(CE) is directly readable.
        self.ce_weights: nn.Linear | None = None
        if ce_feature_dim > 0:
            self.ce_weights = nn.Linear(ce_feature_dim, centres.numel())
            nn.init.zeros_(self.ce_weights.weight)
            nn.init.zeros_(self.ce_weights.bias)

    def forward(
        self,
        masses: th.Tensor,
        valid: th.Tensor,
        ce_features: th.Tensor | None = None,
    ) -> th.Tensor:
        """Return an additive ``(molecules, 1, candidates, candidates)`` attention bias.

        Args:
            masses: ``(molecules, candidates)`` neutral candidate masses; padded slots may
                hold anything, they are zeroed by ``valid``.
            valid: ``(molecules, candidates)`` boolean, True for real candidates.

        Returns:
            The bias, broadcastable over attention heads.
        """
        if masses.ndim != 2 or valid.shape != masses.shape:
            raise ValueError("masses and valid must be aligned (molecules, candidates) tensors")
        if (self.ce_weights is not None) != (ce_features is not None):
            raise ValueError("ce_features must be supplied exactly when CE conditioning is on")
        deltas = masses.unsqueeze(2) - masses.unsqueeze(1)
        if not self.signed:
            deltas = deltas.abs()
        widths = self.log_widths.exp().clamp(min=1e-3)
        weights = self.weights.unsqueeze(0)
        if self.ce_weights is not None and ce_features is not None:
            if ce_features.shape[0] != masses.shape[0]:
                raise ValueError("ce_features must contain one row per molecule")
            weights = weights + self.ce_weights(ce_features.to(masses))
        bias = deltas.new_zeros(deltas.shape)
        # Accumulate one centre at a time: materializing (..., num_centres) would cost
        # num_centres times the pair tensor, which dominates at the candidate budget.
        for index in range(self.centres.numel()):
            offset = (deltas - self.centres[index]) / widths[index]
            bias = bias + weights[:, index].reshape(-1, 1, 1) * th.exp(-0.5 * offset * offset)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        return (bias * pair_valid).unsqueeze(1)


class _BiasedSelfAttention(nn.Module):
    """Batch-first multi-head self-attention accepting an additive pairwise bias.

    Parameter names mirror :class:`torch.nn.MultiheadAttention` (``in_proj_weight``,
    ``in_proj_bias``, ``out_proj``) so that a checkpoint trained with the stock module
    loads into this one unchanged. ``qk_norm`` adds new parameters and is therefore only
    checkpoint-compatible in the direction of enabling it.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float,
        qk_norm: bool = False,
        ce_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if ce_feature_dim < 0:
            raise ValueError("ce_feature_dim must be nonnegative")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(th.empty(3 * model_dim, model_dim))
        self.in_proj_bias = nn.Parameter(th.zeros(3 * model_dim))
        nn.init.xavier_uniform_(self.in_proj_weight)
        self.out_proj = nn.Linear(model_dim, model_dim)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        # FiLM on the query and key streams from collision energy: CE then changes WHICH
        # candidates attend to which, not just how each is embedded. Zero-initialized, so
        # the modulation starts as the identity.
        self.ce_film: nn.Linear | None = None
        if ce_feature_dim > 0:
            self.ce_film = nn.Linear(ce_feature_dim, 4 * model_dim)
            nn.init.zeros_(self.ce_film.weight)
            nn.init.zeros_(self.ce_film.bias)

    def forward(
        self,
        tokens: th.Tensor,
        key_padding_mask: th.Tensor,
        pair_bias: th.Tensor | None = None,
        ce_features: th.Tensor | None = None,
    ) -> th.Tensor:
        """Attend over candidates.

        Args:
            tokens: ``(molecules, candidates, model_dim)``.
            key_padding_mask: ``(molecules, candidates)`` boolean, True where padded.
            pair_bias: optional additive bias broadcastable to
                ``(molecules, heads, candidates, candidates)``.

        Returns:
            The attention output, shaped like ``tokens``.
        """
        num_molecules, num_candidates, _ = tokens.shape
        projected = F.linear(tokens, self.in_proj_weight, self.in_proj_bias)
        query, key, value = projected.chunk(3, dim=-1)

        def split(tensor: th.Tensor) -> th.Tensor:
            return tensor.reshape(
                num_molecules, num_candidates, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = split(query), split(key), split(value)
        if (self.ce_film is not None) != (ce_features is not None):
            raise ValueError("ce_features must be supplied exactly when ce_film is on")
        if self.ce_film is not None and ce_features is not None:
            modulation = self.ce_film(ce_features.to(tokens))
            query_scale, query_shift, key_scale, key_shift = (
                part.reshape(num_molecules, self.num_heads, 1, self.head_dim)
                for part in modulation.chunk(4, dim=-1)
            )
            query = query * (1.0 + query_scale) + query_shift
            key = key * (1.0 + key_scale) + key_shift
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        attention_mask = th.zeros(
            (num_molecules, 1, 1, num_candidates),
            dtype=tokens.dtype,
            device=tokens.device,
        ).masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        if pair_bias is not None:
            attention_mask = attention_mask + pair_bias
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(num_molecules, num_candidates, self.model_dim)
        return self.out_proj(attended)


def candidate_graph_edges(
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    component_batch_indices: th.Tensor,
    spectrum_atom_offsets: th.Tensor,
    spectrum_num_atoms: th.Tensor,
    *,
    num_components: int,
    chunk_size: int = 64,
) -> th.Tensor:
    """Directed attention edges of the candidate graph (arm ``cg1``).

    Within one spectrum, a query candidate is linked to a key candidate when the key is a direct
    parent of the query (it strictly contains the query's atoms with no candidate strictly
    between them), a direct child, or its complement (disjoint, and together the whole
    precursor: the two sides of one acyclic cut). A candidate between two cuts lies inside both
    single-cut sides and so keeps both parents. The intact precursor contains every other
    candidate, so as a key it would hand each candidate the whole set's summary in one hop: it
    receives from its direct children and sends to none. Self-loops are left to the caller.

    With ``M`` the candidates-by-atoms membership of a spectrum, ``M M^T`` counts every pair's
    shared atoms; ``i`` lies strictly inside ``j`` when they share all of ``i``'s atoms and ``j``
    is larger, and the containment matrix times itself marks the pairs with a candidate
    between. Spectra are densified ``chunk_size`` at a time to bound memory.

    Args:
        atom_indices: Global atom index of each (component, atom) membership row, shape ``(M,)``.
        atom_component_indices: Component of each membership row, shape ``(M,)``.
        component_batch_indices: Spectrum of each component, shape ``(C,)``.
        spectrum_atom_offsets: First global atom index of each spectrum's graph, shape ``(B,)``.
        spectrum_num_atoms: Atom count of each spectrum's precursor, shape ``(B,)``.
        num_components: ``C``.
        chunk_size: Spectra densified together.

    Returns:
        Long tensor ``(2, E)`` of ``(query component, key component)`` pairs, without
        self-loops.
    """
    index_inputs = (
        atom_indices,
        atom_component_indices,
        component_batch_indices,
        spectrum_atom_offsets,
        spectrum_num_atoms,
    )
    if any(t.dtype != th.long or t.ndim != 1 for t in index_inputs):
        raise ValueError("candidate_graph_edges expects 1-D torch.long index tensors")
    if atom_indices.shape != atom_component_indices.shape:
        raise ValueError("atom_indices and atom_component_indices must have matching shapes")
    if component_batch_indices.shape != (num_components,):
        raise ValueError("component_batch_indices must contain one spectrum per component")
    if spectrum_atom_offsets.shape != spectrum_num_atoms.shape:
        raise ValueError("spectrum_atom_offsets and spectrum_num_atoms must have matching shapes")
    device = atom_indices.device
    batch_size = int(spectrum_num_atoms.shape[0])
    empty = th.zeros((2, 0), dtype=th.long, device=device)
    if num_components == 0:
        return empty
    _validate_tensor_condition(
        ((atom_component_indices >= 0) & (atom_component_indices < num_components)).all(),
        "atom_component_indices references a missing component",
    )
    _validate_tensor_condition(
        ((component_batch_indices >= 0) & (component_batch_indices < batch_size)).all(),
        "component_batch_indices references a missing spectrum",
    )
    counts = th.bincount(component_batch_indices, minlength=batch_size)
    order = th.argsort(component_batch_indices, stable=True)
    starts = counts.cumsum(0) - counts
    position = th.empty_like(component_batch_indices)
    position[order] = (
        th.arange(num_components, device=device) - starts[component_batch_indices[order]]
    )
    table = th.full((batch_size, int(counts.max())), -1, dtype=th.long, device=device)
    table[component_batch_indices, position] = th.arange(num_components, device=device)
    member_spectrum = component_batch_indices[atom_component_indices]
    member_position = position[atom_component_indices]
    member_atom = atom_indices - spectrum_atom_offsets[member_spectrum]
    edges = [empty]
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        chunk_counts = counts[start:stop]
        width = int(chunk_counts.max())
        if width == 0:
            continue
        whole = spectrum_num_atoms[start:stop]
        keep = (member_spectrum >= start) & (member_spectrum < stop)
        membership = th.zeros((stop - start, width, int(whole.max())), device=device)
        membership[member_spectrum[keep] - start, member_position[keep], member_atom[keep]] = 1.0
        shared = membership @ membership.transpose(1, 2)
        size = membership.sum(2)
        valid = th.arange(width, device=device).unsqueeze(0) < chunk_counts.unsqueeze(1)
        pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
        # 1. inside[i, j]: candidate i lies strictly inside candidate j.
        inside = (shared == size.unsqueeze(2)) & (size.unsqueeze(2) < size.unsqueeze(1)) & pairs
        between = (inside.float() @ inside.float()) > 0
        parent = inside & ~between
        # 2. Complements: disjoint, and together the whole precursor.
        whole = whole.to(size.dtype)[:, None, None]
        complement = (shared == 0) & (size.unsqueeze(2) + size.unsqueeze(1) == whole) & pairs
        # 3. Queries read parents, children and complements; the precursor never sends.
        links = parent | parent.transpose(1, 2) | complement
        links = links & ~((size == whole[:, :, 0]) & valid).unsqueeze(1)
        spectrum, query, key = th.nonzero(links, as_tuple=True)
        spectrum = spectrum + start
        edges.append(th.stack([table[spectrum, query], table[spectrum, key]]))
    return th.cat(edges, dim=1)


class CandidateIntensityTransformerHead(nn.Module):
    """Contextualize a molecule's component candidates before intensity scoring.

    ``attention_type`` selects how candidates exchange information. ``induced`` routes
    every exchange through a fixed latent set, keeping attention linear in the number
    of component pathways, which matters when a molecule contributes thousands of
    candidates. ``full`` lets candidates attend to each other directly: the cost is
    quadratic in the candidate count, but the pairwise interaction is exact rather than
    capped at the inducing-point rank. ``none`` keeps only the feed-forward sublayers, so
    each candidate is scored from its own token. With ``full``, a positive ``mass_window``
    (Da) restricts each candidate to the candidates within that neutral-mass distance, and
    ``candidate_graph`` restricts it to its chemical neighbours (see
    :func:`candidate_graph_edges`).
    ``mz_positional_encoding`` and ``nl_positional_encoding`` add a sinusoidal position to
    each token: its neutral mass ``m`` and its neutral loss ``M - m`` from the precursor's
    neutral mass ``M``. With both on, each takes half the token width.

    Note:
        Two inputs degenerate in unique-fragment mode, where the caller passes identity
        ``component_mask_indices`` and ``num_masks`` equal to the component count. The
        per-mask component count is then one everywhere, so its ``log1p`` input feature
        is a constant, and ``component_residual_head`` receives no gradient because the
        caller reads the mask-pooled residual instead. Both remain live in the legacy
        pathway-token mode, so they are retained for checkpoint and mode compatibility.
    """

    def __init__(
        self,
        fragment_embedding_dim: int,
        context_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        num_inducing_points: int,
        num_ion_modes: int,
        num_h_shifts: int,
        dropout: float,
        use_base_probability_feature: bool,
        attention_type: Literal["induced", "full", "none"] = "induced",
        separate_ion_h_support_head: bool = False,
        predict_oos: bool = False,
        initial_oos_probability: float = 0.1,
        candidate_feature_dim: int = 0,
        unique_fragment_tokens: bool = False,
        norm_placement: Literal["post", "pre"] = "post",
        qk_norm: bool = False,
        mass_pair_bias: bool = False,
        mass_pair_bias_centres: int = len(_NEUTRAL_LOSS_CENTRES),
        mz_positional_encoding: bool = False,
        mass_window: float = 0.0,
        ce_pair_bias: bool = False,
        ce_film: bool = False,
        ce_feature_dim: int = 0,
        input_dropout: float = 0.0,
        sink_dim: int = 0,
        set_message_substitution: float = 0.0,
        set_key_dropout: float = 0.0,
        connectivity_embedding_dim: int = 0,
        nl_positional_encoding: bool = False,
        candidate_graph: bool = False,
        rule_pathway_dim: int = 0,
        rewire_pathway_dim: int = 0,
    ) -> None:
        super().__init__()
        dimensions = (
            fragment_embedding_dim,
            model_dim,
            num_layers,
            num_heads,
            feedforward_dim,
            num_ion_modes,
            num_h_shifts,
        )
        if min(dimensions) <= 0:
            raise ValueError("transformer dimensions and layer counts must be positive")
        if attention_type not in {"induced", "full", "none"}:
            raise ValueError(f"unknown attention_type: {attention_type}")
        if not math.isfinite(mass_window) or mass_window < 0.0:
            raise ValueError(f"mass_window must be finite and nonnegative, got {mass_window}")
        if mass_window > 0.0 and attention_type != "full":
            raise ValueError(
                "mass_window requires attention_type='full'; the induced layer exchanges "
                "information through latents that carry no mass of their own"
            )
        if mass_pair_bias and attention_type != "full":
            raise ValueError(
                "mass_pair_bias requires attention_type='full'; the induced layer exchanges "
                "information through latents that carry no mass of their own"
            )
        if candidate_graph and attention_type != "full":
            raise ValueError(
                "candidate_graph requires attention_type='full'; it masks direct "
                "candidate-to-candidate attention"
            )
        if qk_norm and attention_type != "full":
            raise ValueError("qk_norm requires attention_type='full'")
        if ce_film and attention_type != "full":
            raise ValueError("ce_film requires attention_type='full'")
        if ce_pair_bias and not mass_pair_bias:
            raise ValueError("ce_pair_bias modulates mass_pair_bias, which is off")
        if (ce_pair_bias or ce_film) and ce_feature_dim <= 0:
            raise ValueError("CE-conditioned attention requires a positive ce_feature_dim")
        self.ce_conditioned = ce_pair_bias or ce_film
        if attention_type == "induced" and num_inducing_points <= 0:
            raise ValueError("num_inducing_points must be positive for induced attention")
        if context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if mz_positional_encoding and model_dim % 2 != 0:
            raise ValueError("mz_positional_encoding needs an even model_dim (sin/cos pairs)")
        if nl_positional_encoding and model_dim % 2 != 0:
            raise ValueError("nl_positional_encoding needs an even model_dim (sin/cos pairs)")
        if mz_positional_encoding and nl_positional_encoding and model_dim % 4 != 0:
            raise ValueError(
                "mz_positional_encoding with nl_positional_encoding needs model_dim divisible "
                "by 4 (each encoding takes half the width, in sin/cos pairs)"
            )
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if not math.isfinite(initial_oos_probability) or not (0.0 < initial_oos_probability < 1.0):
            raise ValueError("initial_oos_probability must be finite and in (0, 1)")
        if candidate_feature_dim < 0:
            raise ValueError("candidate_feature_dim must be nonnegative")
        if not 0.0 <= input_dropout < 1.0:
            raise ValueError(f"input_dropout must be in [0, 1), got {input_dropout}")
        if sink_dim < 0:
            raise ValueError(f"sink_dim must be nonnegative, got {sink_dim}")
        if sink_dim > 0 and not unique_fragment_tokens:
            raise ValueError("the memorization sink requires unique-fragment tokens")
        if connectivity_embedding_dim < 0:
            raise ValueError(
                f"connectivity_embedding_dim must be nonnegative, got {connectivity_embedding_dim}"
            )
        if rule_pathway_dim < 0:
            raise ValueError(f"rule_pathway_dim must be nonnegative, got {rule_pathway_dim}")
        if rewire_pathway_dim < 0:
            raise ValueError(f"rewire_pathway_dim must be nonnegative, got {rewire_pathway_dim}")

        self.fragment_embedding_dim = fragment_embedding_dim
        self.context_dim = context_dim
        self.num_ion_modes = num_ion_modes
        self.num_h_shifts = num_h_shifts
        self.use_base_probability_feature = use_base_probability_feature
        self.attention_type = attention_type
        self.predict_oos = predict_oos
        # Descriptor columns the caller builds from the resolved `candidate_features` specs;
        # the head needs only their total width, and zero means the token carries no descriptor
        # block at all.
        self.candidate_feature_dim = candidate_feature_dim
        # In unique-fragment mode every token is its own mask, so the per-mask component count is
        # 1 everywhere and the caller has no separate charge factor to report. Both inputs are
        # then constants and are dropped rather than projected.
        self.unique_fragment_tokens = unique_fragment_tokens
        input_dim = 3 * fragment_embedding_dim + context_dim + candidate_feature_dim
        if not unique_fragment_tokens:
            input_dim += 1
        if use_base_probability_feature:
            input_dim += 1 if unique_fragment_tokens else 2
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.SiLU(),
            nn.LayerNorm(model_dim),
        )
        # Training-only dropout on the projected token, the first layer every candidate route
        # passes through. Applied in `forward` so `input_projection` keeps its state-dict keys.
        self.input_dropout = float(input_dropout)
        self.inducing_tokens: nn.Parameter | None = None
        if attention_type == "induced":
            self.inducing_tokens = nn.Parameter(th.empty(1, num_inducing_points, model_dim))
            nn.init.normal_(self.inducing_tokens, std=model_dim**-0.5)
        layer_type = {
            "induced": _InducedCandidateAttentionLayer,
            "full": _FullCandidateAttentionLayer,
            "none": _PositionwiseCandidateLayer,
        }[attention_type]
        self.layers = nn.ModuleList(
            [
                layer_type(
                    model_dim,
                    num_heads,
                    feedforward_dim,
                    dropout,
                    norm_placement=norm_placement,
                    qk_norm=qk_norm,
                    ce_feature_dim=ce_feature_dim if ce_film else 0,
                    set_message_substitution=set_message_substitution,
                    set_key_dropout=set_key_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ce_film = ce_film
        self.mass_pair_bias = (
            PairwiseMassBias(
                num_centres=mass_pair_bias_centres,
                ce_feature_dim=ce_feature_dim if ce_pair_bias else 0,
            )
            if mass_pair_bias
            else None
        )
        # The standard transformer sinusoidal encoding, with each candidate's neutral mass in Da
        # (mz) or its neutral loss from the precursor in Da (nl) as its position. With both on
        # they split the token width, mass first, as a 2-D positional encoding splits it by axis.
        # Summed over the same dimensions they would not separate: per frequency w,
        # PE(m) + PE(M - m) = 2 cos(w (m - M / 2)) (sin(w M / 2), cos(w M / 2)).
        # Non-persistent buffers: no parameters, no initialization draws, and an unchanged state
        # dict.
        self.mz_positional_encoding = mz_positional_encoding
        self.nl_positional_encoding = nl_positional_encoding
        encoding_width = (
            model_dim // 2 if mz_positional_encoding and nl_positional_encoding else model_dim
        )
        for buffer_name, enabled in (
            ("mz_inverse_frequencies", mz_positional_encoding),
            ("nl_inverse_frequencies", nl_positional_encoding),
        ):
            if enabled:
                self.register_buffer(
                    buffer_name,
                    10000.0
                    ** (-th.arange(0, encoding_width, 2, dtype=th.float64) / encoding_width),
                    persistent=False,
                )
        # Neutral-mass radius (Da) of each candidate's attention; 0 attends over the whole set.
        self.mass_window = float(mass_window)
        # Attend only along candidate-graph edges (arm cg1); the caller supplies the edges.
        self.candidate_graph = candidate_graph
        # Each candidate's neutral mass must be passed whenever a mass-aware input is on.
        self.uses_token_masses = (
            mass_pair_bias
            or mz_positional_encoding
            or nl_positional_encoding
            or self.mass_window > 0.0
        )
        self.component_residual_head = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.component_residual_head.weight)
        nn.init.zeros_(self.component_residual_head.bias)
        self.mask_residual_head = nn.Linear(model_dim, 1)
        nn.init.zeros_(self.mask_residual_head.weight)
        nn.init.zeros_(self.mask_residual_head.bias)
        self.ion_hydrogen_head = nn.Linear(
            model_dim,
            num_ion_modes * num_h_shifts,
        )
        self.ion_h_support_head: nn.Linear | None = None
        if separate_ion_h_support_head:
            with th.random.fork_rng(devices=[]):
                self.ion_h_support_head = nn.Linear(
                    model_dim,
                    num_ion_modes * num_h_shifts,
                )
            with th.no_grad():
                self.ion_h_support_head.weight.copy_(self.ion_hydrogen_head.weight)
                self.ion_h_support_head.bias.copy_(self.ion_hydrogen_head.bias)
        self.oos_head = nn.Linear(model_dim, 1) if predict_oos else None
        if self.oos_head is not None:
            nn.init.zeros_(self.oos_head.weight)
            nn.init.constant_(
                self.oos_head.bias,
                math.log(initial_oos_probability / (1.0 - initial_oos_probability)),
            )
        # Memorization sink: the shared feature map `phi` that a per-molecule vector reads, as
        # in `forward`'s `sink_vectors`. Built on a forked RNG so enabling it leaves every other
        # parameter's initialization unchanged.
        self.sink_dim = sink_dim
        self.sink_projection: nn.Linear | None = None
        if sink_dim > 0:
            with th.random.fork_rng(devices=[]):
                self.sink_projection = nn.Linear(model_dim, sink_dim, bias=False)
        # Retained-fragment connectivity branch: a zero-initialized residual added right after
        # input_projection (see `forward`), so at construction it changes no other parameter's
        # initialization and, whatever the branch computes, contributes exactly nothing.
        self.connectivity_embedding_dim = connectivity_embedding_dim
        self.connectivity_residual: nn.Linear | None = None
        if connectivity_embedding_dim > 0:
            with th.random.fork_rng(devices=[]):
                self.connectivity_residual = nn.Linear(connectivity_embedding_dim, model_dim)
            nn.init.zeros_(self.connectivity_residual.weight)
            nn.init.zeros_(self.connectivity_residual.bias)
        # SMARTS rule provenance: one column per rule of the vocabulary, 1 where some mechanism
        # of that rule produces the fragment. Each column's weight is that rule's embedding,
        # so the product sums the fragment's rule embeddings. Zero-initialized and without a
        # bias, it adds nothing to any token at construction, as the connectivity branch.
        self.rule_pathway_dim = rule_pathway_dim
        self.rule_pathway_residual: nn.Linear | None = None
        if rule_pathway_dim > 0:
            with th.random.fork_rng(devices=[]):
                self.rule_pathway_residual = nn.Linear(rule_pathway_dim, model_dim, bias=False)
            nn.init.zeros_(self.rule_pathway_residual.weight)
        # Rewire rule provenance (arm a0): one column per REWIRE_RULE_VOCABULARY rule, zero-init
        # and bias-free like the SMARTS branch. Absent (no module, no RNG draw) when 0.
        self.rewire_pathway_dim = rewire_pathway_dim
        self.rewire_pathway_residual: nn.Linear | None = None
        if rewire_pathway_dim > 0:
            with th.random.fork_rng(devices=[]):
                self.rewire_pathway_residual = nn.Linear(rewire_pathway_dim, model_dim, bias=False)
            nn.init.zeros_(self.rewire_pathway_residual.weight)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, th.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Initialize a newly enabled support head from checkpoint intensity weights.

        Raises:
            RuntimeError: If the checkpoint carries the removed atom-decoder weights.
        """
        legacy_atom_decoder_keys = sorted(
            key
            for key in state_dict
            if key.startswith((f"{prefix}atom_decoder", f"{prefix}atom_projection"))
        )
        if legacy_atom_decoder_keys:
            raise RuntimeError(
                "this checkpoint was trained with the fragment-to-atom decoder, which was "
                "removed on 2026-09-19; it cannot be loaded by the current model "
                f"(found {len(legacy_atom_decoder_keys)} legacy keys, e.g. "
                f"{legacy_atom_decoder_keys[0]})"
            )
        if self.ion_h_support_head is not None:
            for parameter_name in ("weight", "bias"):
                support_key = f"{prefix}ion_h_support_head.{parameter_name}"
                intensity_key = f"{prefix}ion_hydrogen_head.{parameter_name}"
                if support_key not in state_dict and intensity_key in state_dict:
                    state_dict[support_key] = state_dict[intensity_key].clone()
        if self.oos_head is not None:
            # A checkpoint trained before the escape existed carries no oos_head. Restoring
            # the constructor's initialization lets the escape be switched on over an existing
            # base checkpoint instead of forcing a run from scratch; it starts, as a fresh head
            # does, at zero weight and the configured initial escape probability.
            for parameter_name in ("weight", "bias"):
                escape_key = f"{prefix}oos_head.{parameter_name}"
                if escape_key not in state_dict:
                    state_dict[escape_key] = getattr(self.oos_head, parameter_name).detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _sinusoidal_encoding(positions: th.Tensor, inverse_frequencies: th.Tensor) -> th.Tensor:
        """Interleave ``sin`` and ``cos`` of each position times each inverse frequency.

        Args:
            positions: Positions in Da, shape (num_components,).
            inverse_frequencies: ``w_i``, shape (d / 2,), float64.

        Returns:
            ``[sin(p w_0), cos(p w_0), sin(p w_1), ...]``, shape (num_components, d), in float64.
        """
        angles = positions.to(inverse_frequencies)[:, None] * inverse_frequencies
        return th.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)

    def _mz_encoding(self, masses: th.Tensor) -> th.Tensor:
        """Sinusoidal encoding of each candidate's neutral mass.

        ``PE(m)[2i] = sin(m w_i)`` and ``PE(m)[2i + 1] = cos(m w_i)``, with
        ``w_i = 10000^(-2i / d)`` and ``m`` in Da. ``d`` is the token width, or half of it when
        ``nl_positional_encoding`` is also on. The shortest wavelength is 2 pi Da, so the
        encoding resolves nominal mass, not the mass defect that would name a formula. As with
        text positions, ``PE(m + dm)`` is a fixed rotation of ``PE(m)`` for each ``dm``, so
        attention can express a mass offset such as a neutral loss.

        Args:
            masses: Neutral candidate masses in Da, shape (num_components,).

        Returns:
            The encoding, shape (num_components, d), in float64.
        """
        return self._sinusoidal_encoding(masses, self.mz_inverse_frequencies)

    def _nl_encoding(self, losses: th.Tensor) -> th.Tensor:
        """Sinusoidal encoding of each candidate's neutral loss from its precursor.

        The :meth:`_mz_encoding` formula with the loss ``M - m`` in Da as the position, where
        ``M`` is the precursor's neutral mass: 0 for the precursor itself, 18.011 for a water
        loss. A loss recurs across molecules where a fragment mass does not, and a mass offset
        between two candidates is still a fixed rotation (of opposite sign).

        Args:
            losses: Neutral losses in Da, shape (num_components,).

        Returns:
            The encoding, shape (num_components, d), in float64.
        """
        return self._sinusoidal_encoding(losses, self.nl_inverse_frequencies)

    def _mass_window_bias(self, masses: th.Tensor, padding_mask: th.Tensor) -> th.Tensor:
        """Additive attention mask that keeps each candidate within its neutral-mass window.

        A real candidate ``i`` may attend to ``j`` only when ``|m_i - m_j| <= mass_window``, which
        always includes ``i`` itself. Padded query rows keep every key, so no row is fully masked;
        their outputs are discarded, and padded keys stay hidden by the key padding mask.

        Args:
            masses: Neutral masses in Da, shape (molecules, candidates); padded slots may hold
                anything.
            padding_mask: Shape (molecules, candidates), True where padded.

        Returns:
            The bias, 0 inside the window and -inf outside, shape
            (molecules, 1, candidates, candidates).
        """
        within = (masses.unsqueeze(2) - masses.unsqueeze(1)).abs() <= self.mass_window
        within = within | padding_mask.unsqueeze(2)
        return masses.new_zeros(within.shape).masked_fill(~within, float("-inf")).unsqueeze(1)

    @staticmethod
    def _candidate_graph_bias(
        edges: th.Tensor,
        num_tokens: int,
        bucket_token_indices: th.Tensor,
        local_batch_indices: th.Tensor,
        positions: th.Tensor,
        padding_mask: th.Tensor,
        dtype: th.dtype,
    ) -> th.Tensor:
        """Additive attention mask that keeps each candidate on its candidate-graph edges.

        A real candidate attends to itself and to the keys of its ``(query, key)`` edges. Padded
        query rows keep every key, so no row is fully masked; their outputs are discarded, and
        padded keys stay hidden by the key padding mask.

        Args:
            edges: ``(2, E)`` long ``(query, key)`` token rows, both in one molecule.
            num_tokens: Number of token rows over the whole batch.
            bucket_token_indices: Token rows of this bucket, shape ``(T,)``.
            local_batch_indices: Bucket-local molecule of each of those rows, shape ``(T,)``.
            positions: Padded slot of each of those rows, shape ``(T,)``.
            padding_mask: Shape ``(molecules, candidates)``, True where padded.
            dtype: Dtype of the returned bias.

        Returns:
            The bias, 0 on an edge or the diagonal and -inf elsewhere, shape
            ``(molecules, 1, candidates, candidates)``.
        """
        molecule = edges.new_full((num_tokens,), -1)
        molecule[bucket_token_indices] = local_batch_indices
        slot = edges.new_full((num_tokens,), -1)
        slot[bucket_token_indices] = positions
        query, key = edges[:, molecule[edges[0]] >= 0]
        _validate_tensor_condition(
            (molecule[key] == molecule[query]).all(),
            "candidate_graph_edges links candidates of different molecules",
        )
        num_molecules, num_candidates = padding_mask.shape
        allowed = (
            th.eye(num_candidates, dtype=th.bool, device=padding_mask.device)
            .unsqueeze(0)
            .repeat(num_molecules, 1, 1)
        )
        allowed[molecule[query], slot[query], slot[key]] = True
        allowed = allowed | padding_mask.unsqueeze(2)
        return (
            th.zeros(allowed.shape, dtype=dtype, device=allowed.device)
            .masked_fill(~allowed, float("-inf"))
            .unsqueeze(1)
        )

    def _contextualize_by_molecule(
        self,
        tokens: th.Tensor,
        component_batch_indices: th.Tensor,
        batch_size: int,
        token_masses: th.Tensor | None = None,
        ce_features: th.Tensor | None = None,
        candidate_graph_edges: th.Tensor | None = None,
    ) -> th.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if component_batch_indices.shape != (tokens.shape[0],):
            raise ValueError("component_batch_indices must contain one index per token")
        if component_batch_indices.dtype != th.long:
            raise ValueError("component_batch_indices must have dtype torch.long")
        if tokens.shape[0] == 0:
            raise ValueError("at least one component token is required")
        _validate_tensor_condition(
            ((component_batch_indices >= 0) & (component_batch_indices < batch_size)).all(),
            "component_batch_indices references a missing molecule",
        )

        # 1. Group similarly sized candidate sets to avoid padding every
        # molecule to the largest precursor in the batch.
        order = th.argsort(component_batch_indices, stable=True)
        sorted_batch_indices = component_batch_indices[order]
        counts = th.bincount(sorted_batch_indices, minlength=batch_size)
        _validate_tensor_condition(
            (counts > 0).all(),
            "every molecule must contain at least one component token",
        )
        count_values = counts.tolist()
        bucketed_molecules: dict[int, list[int]] = {}
        for molecule_index, count in enumerate(count_values):
            bucket_limit = 1 << (count - 1).bit_length()
            bucketed_molecules.setdefault(bucket_limit, []).append(molecule_index)

        contextualized = tokens.new_empty(tokens.shape)
        for bucket_limit, molecule_indices in bucketed_molecules.items():
            molecule_index_tensor = th.tensor(
                molecule_indices,
                dtype=th.long,
                device=tokens.device,
            )
            token_mask = th.isin(sorted_batch_indices, molecule_index_tensor)
            sorted_token_indices = th.nonzero(token_mask, as_tuple=False).flatten()
            bucket_token_indices = order[sorted_token_indices]
            bucket_batch_indices = sorted_batch_indices[sorted_token_indices]
            local_batch_indices = th.searchsorted(
                molecule_index_tensor,
                bucket_batch_indices,
            )
            bucket_counts = counts[molecule_index_tensor]
            offsets = bucket_counts.cumsum(0) - bucket_counts
            positions = (
                th.arange(
                    bucket_token_indices.shape[0],
                    device=tokens.device,
                )
                - offsets[local_batch_indices]
            )
            padded_tokens = tokens.new_zeros((len(molecule_indices), bucket_limit, tokens.shape[1]))
            padded_tokens[local_batch_indices, positions] = tokens[bucket_token_indices]
            padding_mask = th.ones(
                (len(molecule_indices), bucket_limit),
                dtype=th.bool,
                device=tokens.device,
            )
            padding_mask[local_batch_indices, positions] = False
            # Spectrum-level CE rows, gathered onto this bucket's molecules.
            bucket_ce_features = None if ce_features is None else ce_features[molecule_index_tensor]
            pair_bias = None
            if self.mass_pair_bias is not None or self.mass_window > 0.0:
                if token_masses is None:
                    raise ValueError(
                        "mass_pair_bias or mass_window is enabled but token masses were not "
                        "provided"
                    )
                padded_masses = token_masses.new_zeros((len(molecule_indices), bucket_limit))
                padded_masses[local_batch_indices, positions] = token_masses[bucket_token_indices]
            if self.mass_pair_bias is not None:
                pair_bias = self.mass_pair_bias(
                    padded_masses,
                    ~padding_mask,
                    ce_features=(
                        bucket_ce_features if self.mass_pair_bias.ce_weights is not None else None
                    ),
                )
            if self.mass_window > 0.0:
                window_bias = self._mass_window_bias(padded_masses, padding_mask)
                pair_bias = window_bias if pair_bias is None else pair_bias + window_bias
            if candidate_graph_edges is not None:
                graph_bias = self._candidate_graph_bias(
                    candidate_graph_edges,
                    tokens.shape[0],
                    bucket_token_indices,
                    local_batch_indices,
                    positions,
                    padding_mask,
                    tokens.dtype,
                )
                pair_bias = graph_bias if pair_bias is None else pair_bias + graph_bias

            # 2. Contextualize candidates, either through a fixed-size inducing set or
            # by direct candidate-to-candidate attention.
            inducing_tokens = (
                None
                if self.inducing_tokens is None
                else self.inducing_tokens.expand(len(molecule_indices), -1, -1)
            )
            for layer in self.layers:
                padded_tokens, inducing_tokens = layer(
                    padded_tokens,
                    inducing_tokens,
                    padding_mask,
                    pair_bias=pair_bias,
                    ce_features=bucket_ce_features if self.ce_film else None,
                )
            contextualized[bucket_token_indices] = padded_tokens[
                local_batch_indices,
                positions,
            ]
        return contextualized

    def forward(
        self,
        fragment_embeddings: th.Tensor,
        complement_embeddings: th.Tensor,
        boundary_embeddings: th.Tensor,
        charge_log_probabilities: th.Tensor,
        mask_log_probabilities: th.Tensor,
        component_batch_indices: th.Tensor,
        component_mask_indices: th.Tensor,
        *,
        batch_size: int,
        num_masks: int,
        context: th.Tensor | None = None,
        return_support_logits: bool = False,
        component_neutral_masses: th.Tensor | None = None,
        ce_features: th.Tensor | None = None,
        candidate_features: th.Tensor | None = None,
        connectivity_embeddings: th.Tensor | None = None,
        sink_vectors: th.Tensor | None = None,
        precursor_neutral_masses: th.Tensor | None = None,
        candidate_graph_edges: th.Tensor | None = None,
        rule_pathway_features: th.Tensor | None = None,
        rewire_pathway_features: th.Tensor | None = None,
    ) -> tuple[th.Tensor, ...]:
        """Return intensity logits and, when enabled, one OOS logit per molecule.

        ``sink_vectors`` are the memorization-sink rows of each candidate's molecule, shape
        ``(num_components, sink_dim)``. When given, each mask residual gains
        ``sink_vector · phi(token)``, averaged over the mask's components like the mask token;
        with ``None`` (evaluation, or before the sink starts) the head is the shared model.
        ``precursor_neutral_masses`` holds each spectrum's intact neutral mass in Da, shape
        ``(batch_size,)``, and is required exactly when ``nl_positional_encoding`` is on.
        ``candidate_graph_edges`` holds the ``(2, E)`` long ``(query, key)`` component rows of
        :func:`candidate_graph_edges`, required exactly when ``candidate_graph`` is on.
        ``rule_pathway_features`` holds each component's 0/1 SMARTS rule indicators, shape
        ``(num_components, rule_pathway_dim)``, required exactly when ``rule_pathway_dim`` is
        positive.
        ``rewire_pathway_features`` likewise holds the 0/1 rewire rule indicators, shape
        ``(num_components, rewire_pathway_dim)``.
        """
        embeddings = (fragment_embeddings, complement_embeddings, boundary_embeddings)
        num_components = fragment_embeddings.shape[0]
        if any(
            embedding.ndim != 2 or embedding.shape != (num_components, self.fragment_embedding_dim)
            for embedding in embeddings
        ):
            raise ValueError(
                "fragment, complement, and boundary embeddings must have matching "
                f"shape (num_components, {self.fragment_embedding_dim})"
            )
        scalar_features = (charge_log_probabilities, mask_log_probabilities)
        if any(feature.shape != (num_components,) for feature in scalar_features):
            raise ValueError("candidate scalar features must contain one value per component")
        if component_mask_indices.shape != (num_components,):
            raise ValueError("component_mask_indices must contain one index per component")
        if component_mask_indices.dtype != th.long:
            raise ValueError("component_mask_indices must have dtype torch.long")
        if num_masks <= 0:
            raise ValueError("num_masks must be positive")
        _validate_tensor_condition(
            ((component_mask_indices >= 0) & (component_mask_indices < num_masks)).all(),
            "component_mask_indices references a missing mask",
        )
        mask_component_counts = th.bincount(
            component_mask_indices,
            minlength=num_masks,
        )
        _validate_tensor_condition(
            (mask_component_counts > 0).all(),
            "every mask must contain at least one component",
        )
        if self.context_dim == 0:
            if context is not None:
                raise ValueError("context was provided but context_dim is zero")
        elif context is None or context.shape != (num_components, self.context_dim):
            raise ValueError(f"context must have shape ({num_components}, {self.context_dim})")

        if (self.candidate_feature_dim > 0) != (candidate_features is not None):
            raise ValueError(
                "candidate_features must be supplied exactly when candidate_feature_dim is positive"
            )
        if candidate_features is not None and candidate_features.shape != (
            num_components,
            self.candidate_feature_dim,
        ):
            raise ValueError(
                f"candidate_features must have shape ({num_components}, "
                f"{self.candidate_feature_dim})"
            )
        if (self.connectivity_embedding_dim > 0) != (connectivity_embeddings is not None):
            raise ValueError(
                "connectivity_embeddings must be supplied exactly when "
                "connectivity_embedding_dim is positive"
            )
        if connectivity_embeddings is not None and connectivity_embeddings.shape != (
            num_components,
            self.connectivity_embedding_dim,
        ):
            raise ValueError(
                f"connectivity_embeddings must have shape ({num_components}, "
                f"{self.connectivity_embedding_dim})"
            )
        if (self.rule_pathway_dim > 0) != (rule_pathway_features is not None):
            raise ValueError(
                "rule_pathway_features must be supplied exactly when rule_pathway_dim is positive"
            )
        if rule_pathway_features is not None and rule_pathway_features.shape != (
            num_components,
            self.rule_pathway_dim,
        ):
            raise ValueError(
                f"rule_pathway_features must have shape ({num_components}, "
                f"{self.rule_pathway_dim})"
            )
        if (self.rewire_pathway_dim > 0) != (rewire_pathway_features is not None):
            raise ValueError(
                "rewire_pathway_features must be supplied exactly when rewire_pathway_dim is "
                "positive"
            )
        if rewire_pathway_features is not None and rewire_pathway_features.shape != (
            num_components,
            self.rewire_pathway_dim,
        ):
            raise ValueError(
                f"rewire_pathway_features must have shape ({num_components}, "
                f"{self.rewire_pathway_dim})"
            )

        features = [*embeddings]
        if context is not None:
            features.append(context)
        if not self.unique_fragment_tokens:
            features.append(
                th.log1p(
                    mask_component_counts[component_mask_indices].to(fragment_embeddings)
                ).unsqueeze(1)
            )
        if self.use_base_probability_feature:
            if not self.unique_fragment_tokens:
                features.append(charge_log_probabilities.to(fragment_embeddings).unsqueeze(1))
            features.append(mask_log_probabilities.to(fragment_embeddings).unsqueeze(1))
        if candidate_features is not None:
            features.append(candidate_features.to(fragment_embeddings))
        if (sink_vectors is not None) and (
            self.sink_projection is None or sink_vectors.shape != (num_components, self.sink_dim)
        ):
            raise ValueError(
                f"sink_vectors must have shape ({num_components}, {self.sink_dim}) and need a "
                "head built with sink_dim > 0"
            )
        tokens = self.input_projection(th.cat(features, dim=1))
        if self.connectivity_residual is not None:
            tokens = tokens + self.connectivity_residual(connectivity_embeddings.to(tokens))
        if self.rule_pathway_residual is not None:
            tokens = tokens + self.rule_pathway_residual(rule_pathway_features.to(tokens))
        if self.rewire_pathway_residual is not None:
            tokens = tokens + self.rewire_pathway_residual(rewire_pathway_features.to(tokens))
        if self.input_dropout > 0.0:
            tokens = F.dropout(tokens, p=self.input_dropout, training=self.training)
        if self.ce_conditioned != (ce_features is not None):
            raise ValueError(
                "ce_features must be supplied exactly when CE-conditioned attention is on"
            )
        if ce_features is not None and ce_features.shape[0] != batch_size:
            raise ValueError("ce_features must contain one row per spectrum")
        if self.uses_token_masses != (component_neutral_masses is not None):
            raise ValueError(
                "component_neutral_masses must be supplied exactly when mass_pair_bias, "
                "mass_window, nl_positional_encoding or mz_positional_encoding is on"
            )
        if component_neutral_masses is not None and component_neutral_masses.shape != (
            num_components,
        ):
            raise ValueError("component_neutral_masses must contain one mass per component")
        if self.candidate_graph != (candidate_graph_edges is not None):
            raise ValueError(
                "candidate_graph_edges must be supplied exactly when candidate_graph is on"
            )
        if candidate_graph_edges is not None and (
            candidate_graph_edges.ndim != 2
            or candidate_graph_edges.shape[0] != 2
            or candidate_graph_edges.dtype != th.long
        ):
            raise ValueError("candidate_graph_edges must be a (2, E) torch.long tensor")
        if self.nl_positional_encoding != (precursor_neutral_masses is not None):
            raise ValueError(
                "precursor_neutral_masses must be supplied exactly when nl_positional_encoding "
                "is on"
            )
        if precursor_neutral_masses is not None and precursor_neutral_masses.shape != (batch_size,):
            raise ValueError("precursor_neutral_masses must contain one mass per spectrum")
        positions = []
        if self.mz_positional_encoding:
            positions.append(self._mz_encoding(component_neutral_masses))
        if self.nl_positional_encoding:
            # The loss is taken in float64, so its rounding does not depend on the mass dtype.
            precursor_masses = precursor_neutral_masses.to(th.float64)[component_batch_indices]
            losses = precursor_masses - component_neutral_masses.to(th.float64)
            positions.append(self._nl_encoding(losses))
        if positions:
            # Added after the token dropout, so a dropped token keeps its position.
            tokens = tokens + th.cat(positions, dim=1).to(tokens)
        if self.attention_type == "none":
            # Nothing crosses candidates, so the tokens need no per-molecule padding.
            for layer in self.layers:
                tokens, _ = layer(tokens, None, None)
        else:
            tokens = self._contextualize_by_molecule(
                tokens,
                component_batch_indices,
                batch_size,
                token_masses=(
                    None
                    if component_neutral_masses is None
                    else component_neutral_masses.to(tokens)
                ),
                ce_features=ce_features,
                candidate_graph_edges=candidate_graph_edges,
            )
        component_residuals = self.component_residual_head(tokens).squeeze(1)
        mask_token_sums = tokens.new_zeros((num_masks, tokens.shape[1])).index_add(
            0,
            component_mask_indices,
            tokens,
        )
        mask_tokens = mask_token_sums / mask_component_counts.to(tokens).unsqueeze(1)
        mask_residuals = self.mask_residual_head(mask_tokens).squeeze(1)
        if sink_vectors is not None:
            # Bilinear in the sink row, so a zero row (unseen molecule) adds exactly nothing.
            sink_logits = (sink_vectors.to(tokens) * self.sink_projection(tokens)).sum(dim=1)
            mask_residuals = mask_residuals + tokens.new_zeros(num_masks).index_add(
                0,
                component_mask_indices,
                sink_logits,
            ) / mask_component_counts.to(tokens)
        ion_hydrogen_logits = self.ion_hydrogen_head(tokens).reshape(
            num_components,
            self.num_ion_modes,
            self.num_h_shifts,
        )
        outputs = (component_residuals, mask_residuals, ion_hydrogen_logits)
        if return_support_logits:
            ion_h_support_logits = (
                ion_hydrogen_logits
                if self.ion_h_support_head is None
                else self.ion_h_support_head(tokens).reshape_as(ion_hydrogen_logits)
            )
            outputs = (*outputs, ion_h_support_logits)
        if self.oos_head is None:
            return outputs

        token_sums = tokens.new_zeros((batch_size, tokens.shape[1])).index_add(
            0,
            component_batch_indices,
            tokens,
        )
        token_counts = th.bincount(
            component_batch_indices,
            minlength=batch_size,
        ).to(tokens)
        # A spectrum can contribute no components, which would divide by zero here; its
        # pooled token is then the zero vector and the head falls back to its bias.
        molecule_tokens = token_sums / token_counts.clamp_min(1.0).unsqueeze(1)
        outputs = (*outputs, self.oos_head(molecule_tokens).squeeze(1))
        return outputs
