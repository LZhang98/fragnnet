"""Log-probabilities of selected cut masks, component charge retention and ion/H channels."""

from __future__ import annotations

from typing import Literal

import torch as th
import torch.nn.functional as F

from fragnnet.utils.misc_utils import _validate_tensor_condition


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
