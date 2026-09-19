"""Learnable heads for one-pass bond-mask fragment generation."""

from __future__ import annotations

import math
from typing import Literal

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _validate_tensor_condition(condition: th.Tensor, message: str) -> None:
    """Validate tensor data without synchronizing CUDA after every check."""
    if th.compiler.is_compiling() or condition.device.type == "cuda":
        th._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _build_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
        raise ValueError("head dimensions must be positive")
    if dropout < 0.0 or dropout > 1.0:
        raise ValueError(f"dropout must be in [0, 1], got {dropout}")
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


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


class _InducedCandidateAttentionLayer(nn.Module):
    """Exchange information between candidate tokens through a small latent set."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
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
    ) -> tuple[th.Tensor, th.Tensor]:
        if inducing_tokens is None:
            raise ValueError("induced candidate attention requires a latent token set")
        latent_updates, _ = self.latent_attention(
            inducing_tokens,
            candidate_tokens,
            candidate_tokens,
            key_padding_mask=candidate_padding_mask,
            need_weights=False,
        )
        inducing_tokens = self.latent_attention_norm(inducing_tokens + self.dropout(latent_updates))
        inducing_tokens = self.latent_feedforward_norm(
            inducing_tokens + self.dropout(self.latent_feedforward(inducing_tokens))
        )

        candidate_updates, _ = self.candidate_attention(
            candidate_tokens,
            inducing_tokens,
            inducing_tokens,
            need_weights=False,
        )
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
    ) -> None:
        super().__init__()
        self.candidate_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
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
    ) -> tuple[th.Tensor, th.Tensor | None]:
        # Padded slots are masked as keys, so they never reach a real candidate even
        # though their own rows keep accumulating values. Every molecule contributes at
        # least one real candidate, so no query row attends over a fully masked key set.
        candidate_updates, _ = self.candidate_attention(
            candidate_tokens,
            candidate_tokens,
            candidate_tokens,
            key_padding_mask=candidate_padding_mask,
            need_weights=False,
        )
        candidate_tokens = self.candidate_attention_norm(
            candidate_tokens + self.dropout(candidate_updates)
        )
        candidate_tokens = self.candidate_feedforward_norm(
            candidate_tokens + self.dropout(self.candidate_feedforward(candidate_tokens))
        )
        return candidate_tokens, inducing_tokens


class IonModeProposalHead(nn.Module):
    """Score CRF/CMF support from proposal-side fragment features."""

    def __init__(
        self,
        fragment_embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_ion_modes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if fragment_embedding_dim <= 0 or context_dim < 0 or num_ion_modes <= 1:
            raise ValueError("ion-mode proposal dimensions are invalid")
        self.fragment_embedding_dim = fragment_embedding_dim
        self.context_dim = context_dim
        self.head = _build_head(
            3 * fragment_embedding_dim + context_dim + 2,
            hidden_dim,
            num_ion_modes,
            dropout,
        )

    def forward(
        self,
        fragment_embeddings: th.Tensor,
        complement_embeddings: th.Tensor,
        boundary_embeddings: th.Tensor,
        cut_counts: th.Tensor,
        fragment_sizes: th.Tensor,
        context: th.Tensor | None,
    ) -> th.Tensor:
        """Return one proposal logit per fragment and ion mode."""
        embeddings = (fragment_embeddings, complement_embeddings, boundary_embeddings)
        num_fragments = fragment_embeddings.shape[0]
        if any(
            embedding.shape != (num_fragments, self.fragment_embedding_dim)
            for embedding in embeddings
        ):
            raise ValueError("proposal fragment embeddings must have matching shapes")
        if cut_counts.shape != (num_fragments,) or fragment_sizes.shape != (num_fragments,):
            raise ValueError("proposal scalar features must contain one value per fragment")
        if self.context_dim == 0:
            if context is not None:
                raise ValueError("context was provided but context_dim is zero")
            features = [*embeddings]
        else:
            if context is None or context.shape != (num_fragments, self.context_dim):
                raise ValueError(f"context must have shape ({num_fragments}, {self.context_dim})")
            features = [*embeddings, context]
        features.extend(
            (
                cut_counts.to(fragment_embeddings).unsqueeze(1),
                th.log1p(fragment_sizes.to(fragment_embeddings)).unsqueeze(1),
            )
        )
        return self.head(th.cat(features, dim=1))


class CandidateIntensityTransformerHead(nn.Module):
    """Contextualize a molecule's component candidates before intensity scoring.

    ``attention_type`` selects how candidates exchange information. ``induced`` routes
    every exchange through a fixed latent set, keeping attention linear in the number
    of component pathways, which matters when a molecule contributes thousands of
    candidates. ``full`` lets candidates attend to each other directly: the cost is
    quadratic in the candidate count, but the pairwise interaction is exact rather than
    capped at the inducing-point rank.

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
        attention_type: Literal["induced", "full"] = "induced",
        separate_ion_h_support_head: bool = False,
        predict_oos: bool = False,
        initial_oos_probability: float = 0.1,
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
        if attention_type not in {"induced", "full"}:
            raise ValueError(f"unknown attention_type: {attention_type}")
        if attention_type == "induced" and num_inducing_points <= 0:
            raise ValueError("num_inducing_points must be positive for induced attention")
        if context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if not math.isfinite(initial_oos_probability) or not (0.0 < initial_oos_probability < 1.0):
            raise ValueError("initial_oos_probability must be finite and in (0, 1)")

        self.fragment_embedding_dim = fragment_embedding_dim
        self.context_dim = context_dim
        self.num_ion_modes = num_ion_modes
        self.num_h_shifts = num_h_shifts
        self.use_base_probability_feature = use_base_probability_feature
        self.attention_type = attention_type
        self.predict_oos = predict_oos
        input_dim = 3 * fragment_embedding_dim + context_dim + 3
        if use_base_probability_feature:
            input_dim += 2
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.SiLU(),
            nn.LayerNorm(model_dim),
        )
        self.inducing_tokens: nn.Parameter | None = None
        if attention_type == "induced":
            self.inducing_tokens = nn.Parameter(th.empty(1, num_inducing_points, model_dim))
            nn.init.normal_(self.inducing_tokens, std=model_dim**-0.5)
        layer_type = (
            _InducedCandidateAttentionLayer
            if attention_type == "induced"
            else _FullCandidateAttentionLayer
        )
        self.layers = nn.ModuleList(
            [
                layer_type(
                    model_dim,
                    num_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(num_layers)
            ]
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
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _contextualize_by_molecule(
        self,
        tokens: th.Tensor,
        component_batch_indices: th.Tensor,
        batch_size: int,
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
        cut_counts: th.Tensor,
        component_sizes: th.Tensor,
        charge_log_probabilities: th.Tensor,
        mask_log_probabilities: th.Tensor,
        component_batch_indices: th.Tensor,
        component_mask_indices: th.Tensor,
        *,
        batch_size: int,
        num_masks: int,
        context: th.Tensor | None = None,
        return_support_logits: bool = False,
    ) -> tuple[th.Tensor, ...]:
        """Return intensity logits and, when enabled, one OOS logit per molecule."""
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
        scalar_features = (
            cut_counts,
            component_sizes,
            charge_log_probabilities,
            mask_log_probabilities,
        )
        if any(feature.shape != (num_components,) for feature in scalar_features):
            raise ValueError("candidate scalar features must contain one value per component")
        if component_sizes.dtype != th.long:
            raise ValueError("component_sizes must have dtype torch.long")
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

        features = [*embeddings]
        if context is not None:
            features.append(context)
        features.extend(
            (
                cut_counts.to(fragment_embeddings).unsqueeze(1),
                th.log1p(component_sizes.to(fragment_embeddings)).unsqueeze(1),
                th.log1p(
                    mask_component_counts[component_mask_indices].to(fragment_embeddings)
                ).unsqueeze(1),
            )
        )
        if self.use_base_probability_feature:
            features.extend(
                (
                    charge_log_probabilities.to(fragment_embeddings).unsqueeze(1),
                    mask_log_probabilities.to(fragment_embeddings).unsqueeze(1),
                )
            )
        tokens = self.input_projection(th.cat(features, dim=1))
        tokens = self._contextualize_by_molecule(
            tokens,
            component_batch_indices,
            batch_size,
        )
        component_residuals = self.component_residual_head(tokens).squeeze(1)
        mask_token_sums = tokens.new_zeros((num_masks, tokens.shape[1])).index_add(
            0,
            component_mask_indices,
            tokens,
        )
        mask_tokens = mask_token_sums / mask_component_counts.to(tokens).unsqueeze(1)
        mask_residuals = self.mask_residual_head(mask_tokens).squeeze(1)
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
        molecule_tokens = token_sums / token_counts.unsqueeze(1)
        return (*outputs, self.oos_head(molecule_tokens).squeeze(1))


def _validate_context(
    context: th.Tensor | None,
    batch_indices: th.Tensor | None,
    *,
    context_dim: int,
    num_items: int,
    name: str,
) -> th.Tensor | None:
    if context_dim == 0:
        if context is not None or batch_indices is not None:
            raise ValueError(f"{name} context inputs were provided but context_dim is zero")
        return None
    if context is None or batch_indices is None:
        raise ValueError(f"{name} context and batch indices are required when context_dim > 0")
    if context.ndim != 2 or context.shape[1] != context_dim:
        raise ValueError(
            f"{name} context must have shape (num_graphs, {context_dim}), got {context.shape}"
        )
    if batch_indices.ndim != 1 or batch_indices.shape[0] != num_items:
        raise ValueError(f"{name} batch indices must contain one index per item")
    if batch_indices.dtype != th.long:
        raise ValueError(f"{name} batch indices must have dtype torch.long")
    if num_items > 0:
        _validate_tensor_condition(
            ((batch_indices >= 0) & (batch_indices < context.shape[0])).all(),
            f"{name} batch indices reference a missing context row",
        )
    return context[batch_indices]


class EndpointBondBreakHead(nn.Module):
    """Predict one endpoint-order-invariant break logit per precursor bond."""

    def __init__(
        self,
        atom_embedding_dim: int,
        bond_feature_dim: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
        num_mixtures: int = 1,
    ) -> None:
        super().__init__()
        if atom_embedding_dim <= 0:
            raise ValueError("atom_embedding_dim must be positive")
        if bond_feature_dim < 0 or context_dim < 0:
            raise ValueError("bond_feature_dim and context_dim must be nonnegative")
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive")
        self.atom_embedding_dim = atom_embedding_dim
        self.bond_feature_dim = bond_feature_dim
        self.context_dim = context_dim
        self.num_mixtures = num_mixtures
        input_dim = 2 * atom_embedding_dim + bond_feature_dim + context_dim
        self.head = _build_head(input_dim, hidden_dim, num_mixtures, dropout)

    def forward(
        self,
        atom_embeddings: th.Tensor,
        bond_endpoints: th.Tensor,
        bond_features: th.Tensor | None = None,
        *,
        context: th.Tensor | None = None,
        bond_batch_indices: th.Tensor | None = None,
    ) -> th.Tensor:
        """Return one Bernoulli logit per bond and latent mixture."""
        if atom_embeddings.ndim != 2 or atom_embeddings.shape[1] != self.atom_embedding_dim:
            raise ValueError(
                "atom_embeddings must have shape "
                f"(num_atoms, {self.atom_embedding_dim}), got {atom_embeddings.shape}"
            )
        if bond_endpoints.ndim != 2 or bond_endpoints.shape[0] != 2:
            raise ValueError("bond_endpoints must have shape (2, num_bonds)")
        if bond_endpoints.dtype != th.long:
            raise ValueError("bond_endpoints must have dtype torch.long")

        num_bonds = bond_endpoints.shape[1]
        if num_bonds > 0:
            _validate_tensor_condition(
                ((bond_endpoints >= 0) & (bond_endpoints < atom_embeddings.shape[0])).all(),
                "bond_endpoints references a missing atom embedding",
            )

        if self.bond_feature_dim == 0:
            if bond_features is not None:
                raise ValueError("bond_features were provided but bond_feature_dim is zero")
        elif (
            bond_features is None
            or bond_features.ndim != 2
            or bond_features.shape != (num_bonds, self.bond_feature_dim)
        ):
            raise ValueError(
                f"bond_features must have shape ({num_bonds}, {self.bond_feature_dim})"
            )

        expanded_context = _validate_context(
            context,
            bond_batch_indices,
            context_dim=self.context_dim,
            num_items=num_bonds,
            name="bond",
        )

        left = atom_embeddings[bond_endpoints[0]]
        right = atom_embeddings[bond_endpoints[1]]
        features = [left + right, th.abs(left - right)]
        if bond_features is not None:
            features.append(bond_features)
        if expanded_context is not None:
            features.append(expanded_context)
        logits = self.head(th.cat(features, dim=-1))
        return logits.squeeze(-1) if self.num_mixtures == 1 else logits


class BondFocusedBreakHead(nn.Module):
    """Score cleavage after alternating bond-to-atom and atom-to-bond updates.

    Chemical bonds maintain explicit hidden states. Pooling those states at
    their endpoint atoms gives mean-field line-graph-style propagation: bonds
    communicate when they share an atom, but only through a degree-normalized
    incident-bond mean rather than learned neighbour-specific messages.
    Symmetric endpoint features preserve invariance to the orientation of
    ``bond_endpoints``.
    """

    def __init__(
        self,
        atom_embedding_dim: int,
        bond_feature_dim: int,
        context_dim: int,
        bond_state_dim: int,
        head_hidden_dim: int,
        num_layers: int,
        dropout: float,
        num_mixtures: int = 1,
    ) -> None:
        super().__init__()
        if min(atom_embedding_dim, bond_state_dim, head_hidden_dim) <= 0:
            raise ValueError("atom, bond-state, and head dimensions must be positive")
        if bond_feature_dim < 0 or context_dim < 0:
            raise ValueError("bond feature and context dimensions must be nonnegative")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.atom_embedding_dim = atom_embedding_dim
        self.bond_feature_dim = bond_feature_dim
        self.context_dim = context_dim
        self.bond_state_dim = bond_state_dim
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.atom_project = nn.Linear(atom_embedding_dim, bond_state_dim)
        self.bond_project = nn.Sequential(
            nn.Linear(
                2 * atom_embedding_dim + bond_feature_dim + context_dim,
                bond_state_dim,
            ),
            nn.SiLU(),
            nn.LayerNorm(bond_state_dim),
        )
        self.atom_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * bond_state_dim, bond_state_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(bond_state_dim, bond_state_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.bond_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * bond_state_dim, bond_state_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(bond_state_dim, bond_state_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.atom_norms = nn.ModuleList([nn.LayerNorm(bond_state_dim) for _ in range(num_layers)])
        self.bond_norms = nn.ModuleList([nn.LayerNorm(bond_state_dim) for _ in range(num_layers)])
        self.head = _build_head(
            bond_state_dim + context_dim,
            head_hidden_dim,
            num_mixtures,
            dropout,
        )

    def forward(
        self,
        atom_embeddings: th.Tensor,
        bond_endpoints: th.Tensor,
        bond_features: th.Tensor | None = None,
        *,
        context: th.Tensor | None = None,
        bond_batch_indices: th.Tensor | None = None,
    ) -> th.Tensor:
        """Return one endpoint-order-invariant logit per chemical bond."""
        if atom_embeddings.ndim != 2 or atom_embeddings.shape[1] != self.atom_embedding_dim:
            raise ValueError(
                "atom_embeddings must have shape "
                f"(num_atoms, {self.atom_embedding_dim}), got {atom_embeddings.shape}"
            )
        if bond_endpoints.ndim != 2 or bond_endpoints.shape[0] != 2:
            raise ValueError("bond_endpoints must have shape (2, num_bonds)")
        if bond_endpoints.dtype != th.long:
            raise ValueError("bond_endpoints must have dtype torch.long")
        num_bonds = bond_endpoints.shape[1]
        if num_bonds > 0:
            _validate_tensor_condition(
                ((bond_endpoints >= 0) & (bond_endpoints < atom_embeddings.shape[0])).all(),
                "bond_endpoints references a missing atom embedding",
            )
        if self.bond_feature_dim == 0:
            if bond_features is not None:
                raise ValueError("bond_features were provided but bond_feature_dim is zero")
        elif (
            bond_features is None
            or bond_features.ndim != 2
            or bond_features.shape != (num_bonds, self.bond_feature_dim)
        ):
            raise ValueError(
                f"bond_features must have shape ({num_bonds}, {self.bond_feature_dim})"
            )

        expanded_context = _validate_context(
            context,
            bond_batch_indices,
            context_dim=self.context_dim,
            num_items=num_bonds,
            name="bond",
        )
        left = atom_embeddings[bond_endpoints[0]]
        right = atom_embeddings[bond_endpoints[1]]
        initial_bond_features = [left + right, th.abs(left - right)]
        if bond_features is not None:
            initial_bond_features.append(bond_features)
        if expanded_context is not None:
            initial_bond_features.append(expanded_context)
        atom_states = self.atom_project(atom_embeddings)
        bond_states = self.bond_project(th.cat(initial_bond_features, dim=-1))
        # CUDA autocast may keep the normalized bond states in FP32 while the
        # direct atom projection is BF16. Aggregation requires matching dtypes.
        state_dtype = th.promote_types(atom_states.dtype, bond_states.dtype)
        atom_states = atom_states.to(dtype=state_dtype)
        bond_states = bond_states.to(dtype=state_dtype)

        # 1. Alternate mean-field bond/atom message passing through shared endpoints.
        endpoint_indices = bond_endpoints.flatten()
        endpoint_counts = atom_states.new_zeros(atom_states.shape[0]).index_add(
            0,
            endpoint_indices,
            atom_states.new_ones(2 * num_bonds),
        )
        for atom_update, bond_update, atom_norm, bond_norm in zip(
            self.atom_updates,
            self.bond_updates,
            self.atom_norms,
            self.bond_norms,
        ):
            endpoint_bond_states = th.cat([bond_states, bond_states], dim=0)
            atom_messages = atom_states.new_zeros(atom_states.shape).index_add(
                0,
                endpoint_indices,
                endpoint_bond_states,
            )
            atom_messages = atom_messages / endpoint_counts.clamp(min=1).unsqueeze(1)
            atom_states = atom_norm(
                atom_states + atom_update(th.cat([atom_states, atom_messages], dim=1))
            )
            left_state = atom_states[bond_endpoints[0]]
            right_state = atom_states[bond_endpoints[1]]
            bond_messages = th.cat(
                [
                    bond_states,
                    left_state + right_state,
                    th.abs(left_state - right_state),
                ],
                dim=1,
            )
            bond_states = bond_norm(bond_states + bond_update(bond_messages))

        # 2. Condition the final cleavage score on acquisition context.
        head_features = [bond_states]
        if expanded_context is not None:
            head_features.append(expanded_context)
        logits = self.head(th.cat(head_features, dim=1))
        return logits.squeeze(-1) if self.num_mixtures == 1 else logits


class AtomChargeAffinityHead(nn.Module):
    """Predict one charge-affinity logit per precursor atom."""

    def __init__(
        self,
        atom_embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if atom_embedding_dim <= 0 or context_dim < 0:
            raise ValueError("atom_embedding_dim must be positive and context_dim nonnegative")
        self.atom_embedding_dim = atom_embedding_dim
        self.context_dim = context_dim
        self.head = _build_head(atom_embedding_dim + context_dim, hidden_dim, 1, dropout)

    def forward(
        self,
        atom_embeddings: th.Tensor,
        *,
        context: th.Tensor | None = None,
        atom_batch_indices: th.Tensor | None = None,
    ) -> th.Tensor:
        """Return atom charge-affinity logits with shape ``(num_atoms,)``."""
        if atom_embeddings.ndim != 2 or atom_embeddings.shape[1] != self.atom_embedding_dim:
            raise ValueError(
                "atom_embeddings must have shape "
                f"(num_atoms, {self.atom_embedding_dim}), got {atom_embeddings.shape}"
            )
        expanded_context = _validate_context(
            context,
            atom_batch_indices,
            context_dim=self.context_dim,
            num_items=atom_embeddings.shape[0],
            name="atom",
        )
        features = atom_embeddings
        if expanded_context is not None:
            features = th.cat([features, expanded_context], dim=-1)
        return self.head(features).squeeze(-1)


def cut_mask_log_probabilities(
    bond_logits: th.Tensor,
    cut_mask: th.Tensor,
) -> th.Tensor:
    """Recompute differentiable Bernoulli log probabilities for selected masks.

    Top-K mask identities are selected from detached logits by the CPU ranking
    utility. This function evaluates those fixed masks from the original logits
    so gradients still reach the bond head.
    """
    if bond_logits.ndim != 1:
        raise ValueError("bond_logits must be one-dimensional")
    if cut_mask.ndim != 2 or cut_mask.shape[1] != bond_logits.shape[0]:
        raise ValueError("cut_mask must have shape (num_masks, num_bonds)")
    if cut_mask.dtype != th.bool:
        raise ValueError("cut_mask must have dtype torch.bool")
    cut_log_probabilities = F.logsigmoid(bond_logits).unsqueeze(0)
    intact_log_probabilities = F.logsigmoid(-bond_logits).unsqueeze(0)
    return th.where(cut_mask, cut_log_probabilities, intact_log_probabilities).sum(dim=1)


def _pool_indexed_embeddings(
    item_embeddings: th.Tensor,
    item_indices: th.Tensor,
    component_indices: th.Tensor,
    *,
    num_components: int,
    reduction: Literal["sum", "mean"] = "mean",
    require_nonempty: bool,
) -> th.Tensor:
    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must have shape (num_items, embedding_dim)")
    if num_components < 0:
        raise ValueError(f"num_components must be nonnegative, got {num_components}")
    if item_indices.ndim != 1 or component_indices.shape != item_indices.shape:
        raise ValueError("item_indices and component_indices must be matching vectors")
    if item_indices.dtype != th.long or component_indices.dtype != th.long:
        raise ValueError("item_indices and component_indices must have dtype torch.long")
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"unknown component pooling reduction: {reduction}")
    if item_indices.numel() > 0:
        _validate_tensor_condition(
            ((item_indices >= 0) & (item_indices < item_embeddings.shape[0])).all(),
            "item_indices references a missing item embedding",
        )
        _validate_tensor_condition(
            ((component_indices >= 0) & (component_indices < num_components)).all(),
            "component_indices references a missing component",
        )

    pooled = item_embeddings.new_zeros((num_components, item_embeddings.shape[1])).index_add(
        0,
        component_indices,
        item_embeddings[item_indices],
    )
    if reduction == "mean" or require_nonempty:
        component_sizes = th.zeros(
            num_components,
            dtype=th.long,
            device=component_indices.device,
        ).index_add(0, component_indices, th.ones_like(component_indices))
        if require_nonempty:
            _validate_tensor_condition(
                (component_sizes > 0).all(),
                "every component must contain at least one pooled item",
            )
    if reduction == "mean":
        pooled = pooled / component_sizes.clamp(min=1).to(dtype=pooled.dtype).unsqueeze(1)
    return pooled


def pool_component_embeddings(
    item_embeddings: th.Tensor,
    item_indices: th.Tensor,
    component_indices: th.Tensor,
    *,
    num_components: int,
    reduction: Literal["sum", "mean"] = "mean",
) -> th.Tensor:
    """Pool sparse item/component membership pairs into component embeddings."""
    return _pool_indexed_embeddings(
        item_embeddings,
        item_indices,
        component_indices,
        num_components=num_components,
        reduction=reduction,
        require_nonempty=True,
    )


def pool_component_and_complement_embeddings(
    atom_embeddings: th.Tensor,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    component_batch_indices: th.Tensor,
    atom_batch_indices: th.Tensor,
    *,
    num_components: int,
    num_molecules: int,
    reduction: Literal["sum", "mean"] = "mean",
) -> tuple[th.Tensor, th.Tensor]:
    """Pool components and derive their complements from precursor totals.

    Every component belongs to one precursor, so its complement sum is the
    precursor sum minus the component sum. This avoids materializing the much
    larger complement-atom/component membership relation.
    """
    if atom_embeddings.ndim != 2:
        raise ValueError("atom_embeddings must be a matrix")
    if atom_indices.ndim != 1 or atom_component_indices.shape != atom_indices.shape:
        raise ValueError("atom_indices and atom_component_indices must be matching vectors")
    if atom_indices.dtype != th.long or atom_component_indices.dtype != th.long:
        raise ValueError("atom and component membership indices must have dtype torch.long")
    if component_batch_indices.shape != (num_components,):
        raise ValueError("component_batch_indices must contain one index per component")
    if atom_batch_indices.shape != (atom_embeddings.shape[0],):
        raise ValueError("atom_batch_indices must contain one index per atom")
    if component_batch_indices.dtype != th.long or atom_batch_indices.dtype != th.long:
        raise ValueError("batch indices must have dtype torch.long")
    if num_components < 0 or num_molecules < 0:
        raise ValueError("component and molecule counts must be nonnegative")
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"unknown component pooling reduction: {reduction}")
    if atom_indices.numel() > 0:
        _validate_tensor_condition(
            ((atom_indices >= 0) & (atom_indices < atom_embeddings.shape[0])).all(),
            "atom_indices references a missing atom embedding",
        )
        _validate_tensor_condition(
            ((atom_component_indices >= 0) & (atom_component_indices < num_components)).all(),
            "atom_component_indices references a missing component",
        )
    if num_components > 0:
        _validate_tensor_condition(
            ((component_batch_indices >= 0) & (component_batch_indices < num_molecules)).all(),
            "component_batch_indices references a missing molecule",
        )

    component_sums = atom_embeddings.new_zeros(
        (num_components, atom_embeddings.shape[1])
    ).index_add(0, atom_component_indices, atom_embeddings[atom_indices])
    component_sizes = th.zeros(
        num_components,
        dtype=th.long,
        device=atom_embeddings.device,
    ).index_add(0, atom_component_indices, th.ones_like(atom_component_indices))
    _validate_tensor_condition(
        (component_sizes > 0).all(),
        "every component must contain at least one atom",
    )

    precursor_sums = atom_embeddings.new_zeros((num_molecules, atom_embeddings.shape[1])).index_add(
        0, atom_batch_indices, atom_embeddings
    )
    precursor_sizes = th.zeros(
        num_molecules,
        dtype=th.long,
        device=atom_embeddings.device,
    ).index_add(0, atom_batch_indices, th.ones_like(atom_batch_indices))
    complement_sizes = precursor_sizes[component_batch_indices] - component_sizes
    complement_sums = precursor_sums[component_batch_indices] - component_sums
    complement_sums = th.where(
        (complement_sizes > 0).unsqueeze(1),
        complement_sums,
        th.zeros_like(complement_sums),
    )

    if reduction == "sum":
        return component_sums, complement_sums
    component_means = component_sums / component_sizes.to(component_sums).unsqueeze(1)
    complement_means = complement_sums / complement_sizes.clamp(min=1).to(
        complement_sums
    ).unsqueeze(1)
    return component_means, complement_means


def pool_boundary_bond_embeddings(
    bond_embeddings: th.Tensor,
    boundary_bond_indices: th.Tensor,
    component_indices: th.Tensor,
    *,
    num_components: int,
    reduction: Literal["sum", "mean"] = "mean",
) -> th.Tensor:
    """Pool sparse boundary-bond/component pairs into component embeddings."""
    return _pool_indexed_embeddings(
        bond_embeddings,
        boundary_bond_indices,
        component_indices,
        num_components=num_components,
        reduction=reduction,
        require_nonempty=False,
    )


def pool_complement_embeddings(
    atom_embeddings: th.Tensor,
    complement_atom_indices: th.Tensor,
    component_indices: th.Tensor,
    *,
    num_components: int,
    reduction: Literal["sum", "mean"] = "mean",
) -> th.Tensor:
    """Pool atoms outside each component, allowing an empty precursor complement."""
    return _pool_indexed_embeddings(
        atom_embeddings,
        complement_atom_indices,
        component_indices,
        num_components=num_components,
        reduction=reduction,
        require_nonempty=False,
    )


def _segment_logsumexp(
    values: th.Tensor,
    segment_indices: th.Tensor,
    num_segments: int,
) -> th.Tensor:
    """Compute a stable logsumexp for sparse one-dimensional segments."""
    maxima = values.new_full((num_segments,), -th.inf).scatter_reduce(
        0,
        segment_indices,
        values,
        reduce="amax",
        include_self=True,
    )
    shifted_exponentials = th.exp(values - maxima[segment_indices])
    sums = shifted_exponentials.new_zeros(num_segments).index_add(
        0,
        segment_indices,
        shifted_exponentials,
    )
    return maxima + th.log(sums)


def component_charge_log_probabilities(
    atom_charge_logits: th.Tensor,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    component_group_indices: th.Tensor,
    *,
    pooling: Literal["logsumexp", "logmeanexp"] = "logsumexp",
) -> th.Tensor:
    """Normalize charge retention within each cut mask's components.

    Args:
        atom_charge_logits: Charge-affinity logits of shape ``(num_atoms,)``.
        atom_indices: Atom index for every sparse atom/component membership pair.
        atom_component_indices: Component index for every membership pair.
        component_group_indices: Cut-mask group index for every component.
        pooling: ``logsumexp`` treats every atom as an additional potential
            charge site. ``logmeanexp`` removes the corresponding component-size
            prior.
    """
    if atom_charge_logits.ndim != 1:
        raise ValueError("atom_charge_logits must be one-dimensional")
    if atom_indices.ndim != 1 or atom_component_indices.shape != atom_indices.shape:
        raise ValueError("atom_indices and atom_component_indices must be matching vectors")
    if atom_indices.dtype != th.long or atom_component_indices.dtype != th.long:
        raise ValueError("atom and component membership indices must have dtype torch.long")
    if component_group_indices.ndim != 1 or component_group_indices.dtype != th.long:
        raise ValueError("component_group_indices must be a torch.long vector")
    num_components = component_group_indices.shape[0]
    if num_components == 0:
        raise ValueError("component_group_indices must contain at least one component")
    if atom_indices.numel() > 0:
        _validate_tensor_condition(
            ((atom_indices >= 0) & (atom_indices < atom_charge_logits.shape[0])).all(),
            "atom_indices references a missing atom charge logit",
        )
        _validate_tensor_condition(
            ((atom_component_indices >= 0) & (atom_component_indices < num_components)).all(),
            "atom_component_indices references a missing component",
        )
    _validate_tensor_condition(
        ((component_group_indices >= 0) & (component_group_indices < num_components)).all(),
        "component_group_indices references an invalid group",
    )
    component_sizes = th.zeros(
        num_components,
        dtype=th.long,
        device=atom_component_indices.device,
    ).index_add(0, atom_component_indices, th.ones_like(atom_component_indices))
    _validate_tensor_condition(
        (component_sizes > 0).all(),
        "every component must contain at least one atom",
    )
    if pooling not in {"logsumexp", "logmeanexp"}:
        raise ValueError(f"unknown charge pooling mode: {pooling}")

    component_logits = _segment_logsumexp(
        atom_charge_logits[atom_indices],
        atom_component_indices,
        num_components,
    )
    if pooling == "logmeanexp":
        component_logits = component_logits - th.log(
            component_sizes.to(dtype=component_logits.dtype)
        )
    group_log_normalizers = _segment_logsumexp(
        component_logits,
        component_group_indices,
        num_components,
    )
    return component_logits - group_log_normalizers[component_group_indices]


def latent_pathway_charge_log_probabilities(
    atom_charge_logits: th.Tensor,
    unique_atom_indices: th.Tensor,
    atom_unique_component_indices: th.Tensor,
    pathway_unique_component_indices: th.Tensor,
    pathway_mask_indices: th.Tensor,
    *,
    num_unique_components: int,
    num_masks: int,
    pooling: Literal["logsumexp", "logmeanexp"] = "logsumexp",
) -> th.Tensor:
    """Normalize charge retention per mask while scoring each atom set once.

    Multiple cut masks can produce the same component atom set. Its unnormalized
    charge-affinity logit is identical in every occurrence, so this function pools
    atom logits once per unique component, broadcasts those logits to the latent
    component pathways, and only then normalizes among each mask's components.
    """
    if atom_charge_logits.ndim != 1:
        raise ValueError("atom_charge_logits must be one-dimensional")
    if (
        unique_atom_indices.ndim != 1
        or atom_unique_component_indices.shape != unique_atom_indices.shape
    ):
        raise ValueError(
            "unique_atom_indices and atom_unique_component_indices must be matching vectors"
        )
    index_tensors = (
        unique_atom_indices,
        atom_unique_component_indices,
        pathway_unique_component_indices,
        pathway_mask_indices,
    )
    if any(index.dtype != th.long or index.ndim != 1 for index in index_tensors):
        raise ValueError("all membership and pathway indices must be torch.long vectors")
    if pathway_unique_component_indices.shape != pathway_mask_indices.shape:
        raise ValueError("pathway unique-component and mask indices must match")
    if pathway_unique_component_indices.numel() == 0:
        raise ValueError("at least one component pathway is required")

    if num_unique_components <= 0 or num_masks <= 0:
        raise ValueError("unique-component and mask counts must be positive")
    _validate_tensor_condition(
        ((unique_atom_indices >= 0) & (unique_atom_indices < atom_charge_logits.shape[0])).all(),
        "unique_atom_indices references a missing atom charge logit",
    )
    _validate_tensor_condition(
        (
            (atom_unique_component_indices >= 0)
            & (atom_unique_component_indices < num_unique_components)
        ).all(),
        "atom_unique_component_indices references a missing unique component",
    )
    _validate_tensor_condition(
        (
            (pathway_unique_component_indices >= 0)
            & (pathway_unique_component_indices < num_unique_components)
        ).all(),
        "pathway_unique_component_indices references a missing unique component",
    )
    _validate_tensor_condition(
        ((pathway_mask_indices >= 0) & (pathway_mask_indices < num_masks)).all(),
        "pathway_mask_indices references a missing mask",
    )
    if pooling not in {"logsumexp", "logmeanexp"}:
        raise ValueError(f"unknown charge pooling mode: {pooling}")

    unique_component_sizes = th.zeros(
        num_unique_components,
        dtype=th.long,
        device=atom_unique_component_indices.device,
    ).index_add(
        0,
        atom_unique_component_indices,
        th.ones_like(atom_unique_component_indices),
    )
    _validate_tensor_condition(
        (unique_component_sizes > 0).all(),
        "every unique component must contain at least one atom",
    )
    unique_component_logits = _segment_logsumexp(
        atom_charge_logits[unique_atom_indices],
        atom_unique_component_indices,
        num_unique_components,
    )
    if pooling == "logmeanexp":
        unique_component_logits = unique_component_logits - th.log(
            unique_component_sizes.to(dtype=unique_component_logits.dtype)
        )

    pathway_logits = unique_component_logits[pathway_unique_component_indices]
    mask_log_normalizers = _segment_logsumexp(
        pathway_logits,
        pathway_mask_indices,
        num_masks,
    )
    return pathway_logits - mask_log_normalizers[pathway_mask_indices]


def masked_ion_hydrogen_log_probabilities(
    logits: th.Tensor,
    valid_channel_mask: th.Tensor,
) -> th.Tensor:
    """Mask invalid joint ion/H channels and normalize each fragment."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape (num_fragments, num_ion_modes, num_h_shifts)")
    if valid_channel_mask.shape != logits.shape or valid_channel_mask.dtype != th.bool:
        raise ValueError("valid_channel_mask must be boolean and match logits.shape")
    if logits.shape[0] == 0:
        return logits
    _validate_tensor_condition(
        valid_channel_mask.flatten(start_dim=1).any(dim=1).all(),
        "every fragment must have at least one valid ion/H channel",
    )

    flat_logits = logits.flatten(start_dim=1)
    flat_mask = valid_channel_mask.flatten(start_dim=1)
    flat_log_probabilities = th.log_softmax(flat_logits.masked_fill(~flat_mask, -th.inf), dim=1)
    return flat_log_probabilities.reshape_as(logits)
