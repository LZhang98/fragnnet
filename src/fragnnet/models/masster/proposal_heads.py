"""Proposal-side heads: bond-break, atom charge-affinity and ion-mode logits."""

from __future__ import annotations

from typing import Literal

import torch as th
import torch.nn as nn

from fragnnet.utils.misc_utils import _validate_tensor_condition


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
        endpoint_aggregation: Literal["mean", "sum", "sqrt"] = "mean",
    ) -> None:
        super().__init__()
        if min(atom_embedding_dim, bond_state_dim, head_hidden_dim) <= 0:
            raise ValueError("atom, bond-state, and head dimensions must be positive")
        if endpoint_aggregation not in {"mean", "sum", "sqrt"}:
            raise ValueError(f"unknown endpoint_aggregation: {endpoint_aggregation}")
        self.endpoint_aggregation = endpoint_aggregation
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
            # Degree normalization. ``mean`` is the historical default; GIN's expressiveness
            # argument prefers ``sum``, since a mean cannot distinguish a quaternary carbon
            # from a methyl. ``sqrt`` keeps some degree signal without the raw scale growth.
            if self.endpoint_aggregation == "mean":
                atom_messages = atom_messages / endpoint_counts.clamp(min=1).unsqueeze(1)
            elif self.endpoint_aggregation == "sqrt":
                atom_messages = atom_messages / endpoint_counts.clamp(min=1).sqrt().unsqueeze(1)
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
