"""Learnable heads for one-pass bond-mask fragment generation."""

from __future__ import annotations

from typing import Literal

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _validate_tensor_condition(condition: th.Tensor, message: str) -> None:
    """Raise on invalid tensor data without breaking compiled forward graphs."""
    if th.compiler.is_compiling():
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
    ) -> None:
        super().__init__()
        if atom_embedding_dim <= 0:
            raise ValueError("atom_embedding_dim must be positive")
        if bond_feature_dim < 0 or context_dim < 0:
            raise ValueError("bond_feature_dim and context_dim must be nonnegative")
        self.atom_embedding_dim = atom_embedding_dim
        self.bond_feature_dim = bond_feature_dim
        self.context_dim = context_dim
        input_dim = 2 * atom_embedding_dim + bond_feature_dim + context_dim
        self.head = _build_head(input_dim, hidden_dim, 1, dropout)

    def forward(
        self,
        atom_embeddings: th.Tensor,
        bond_endpoints: th.Tensor,
        bond_features: th.Tensor | None = None,
        *,
        context: th.Tensor | None = None,
        bond_batch_indices: th.Tensor | None = None,
    ) -> th.Tensor:
        """Return independent Bernoulli logits with shape ``(num_bonds,)``."""
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
                "bond_features must have shape "
                f"({num_bonds}, {self.bond_feature_dim})"
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
        return self.head(th.cat(features, dim=-1)).squeeze(-1)


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
    sums = values.new_zeros(num_segments).index_add(
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
            (
                (atom_component_indices >= 0)
                & (atom_component_indices < num_components)
            ).all(),
            "atom_component_indices references a missing component",
        )
    _validate_tensor_condition(
        (
            (component_group_indices >= 0)
            & (component_group_indices < num_components)
        ).all(),
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


class JointIonHydrogenHead(nn.Module):
    """Predict joint ion-mode and hydrogen-transfer logits per charged fragment."""

    def __init__(
        self,
        fragment_embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_ion_modes: int,
        num_h_shifts: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if fragment_embedding_dim <= 0 or context_dim < 0:
            raise ValueError("fragment_embedding_dim must be positive and context_dim nonnegative")
        if num_ion_modes <= 0 or num_h_shifts <= 0:
            raise ValueError("num_ion_modes and num_h_shifts must be positive")
        self.fragment_embedding_dim = fragment_embedding_dim
        self.context_dim = context_dim
        self.num_ion_modes = num_ion_modes
        self.num_h_shifts = num_h_shifts
        input_dim = 3 * fragment_embedding_dim + context_dim + 1
        self.head = _build_head(
            input_dim,
            hidden_dim,
            num_ion_modes * num_h_shifts,
            dropout,
        )

    def forward(
        self,
        fragment_embeddings: th.Tensor,
        complement_embeddings: th.Tensor,
        boundary_embeddings: th.Tensor,
        cut_counts: th.Tensor,
        *,
        context: th.Tensor | None = None,
    ) -> th.Tensor:
        """Return logits shaped ``(num_fragments, num_ion_modes, num_h_shifts)``."""
        embeddings = (fragment_embeddings, complement_embeddings, boundary_embeddings)
        if any(
            embedding.ndim != 2
            or embedding.shape[1] != self.fragment_embedding_dim
            or embedding.shape[0] != fragment_embeddings.shape[0]
            for embedding in embeddings
        ):
            raise ValueError(
                "fragment, complement, and boundary embeddings must have matching "
                f"shape (num_fragments, {self.fragment_embedding_dim})"
            )
        num_fragments = fragment_embeddings.shape[0]
        if cut_counts.ndim != 1 or cut_counts.shape[0] != num_fragments:
            raise ValueError("cut_counts must contain one value per fragment")
        if self.context_dim == 0:
            if context is not None:
                raise ValueError("context was provided but context_dim is zero")
        elif (
            context is None
            or context.ndim != 2
            or context.shape != (num_fragments, self.context_dim)
        ):
            raise ValueError(
                f"context must have shape ({num_fragments}, {self.context_dim})"
            )

        features = [*embeddings, cut_counts.to(fragment_embeddings).unsqueeze(-1)]
        if context is not None:
            features.append(context)
        logits = self.head(th.cat(features, dim=-1))
        return logits.reshape(num_fragments, self.num_ion_modes, self.num_h_shifts)


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
