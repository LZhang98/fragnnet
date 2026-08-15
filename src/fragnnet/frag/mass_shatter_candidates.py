"""Batch top-ranked bond masks into sparse MASShatter component tensors."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch as th
import torch.nn.functional as F

from fragnnet.frag.bond_mask_generator import (
    CutMaskComponents,
    RankedCutMask,
    connected_component_atom_masks,
    iter_ranked_cut_masks,
    top_k_cut_masks,
    translate_cut_masks,
)


class MassShatterTopologyCache:
    """Bounded LRU cache for static cut-mask component translations."""

    def __init__(self, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self.max_entries = max_entries
        self._entries: OrderedDict[
            tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]], tuple[int, ...]
        ] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_components(
        self,
        num_atoms: int,
        bond_endpoints: tuple[tuple[int, int], ...],
        broken_bond_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Return cached connected components for one static graph/cut mask."""
        key = (num_atoms, bond_endpoints, broken_bond_indices)
        components = self._entries.get(key)
        if components is not None:
            self._entries.move_to_end(key)
            self.hits += 1
            return components

        self.misses += 1
        components = connected_component_atom_masks(
            num_atoms,
            tuple(zip(*bond_endpoints)) if bond_endpoints else ((), ()),
            broken_bond_indices,
        )
        self._entries[key] = components
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return components


@dataclass(frozen=True)
class MassShatterCandidateBatch:
    """Sparse component pathways selected from a batch of precursor graphs."""

    mask_log_probabilities: th.Tensor
    mask_batch_indices: th.Tensor
    component_mask_indices: th.Tensor
    component_batch_indices: th.Tensor
    component_cut_counts: th.Tensor
    atom_indices: th.Tensor
    atom_component_indices: th.Tensor
    complement_atom_indices: th.Tensor
    complement_component_indices: th.Tensor
    boundary_bond_indices: th.Tensor
    boundary_component_indices: th.Tensor

    @property
    def num_components(self) -> int:
        """Return the number of retained component pathways."""
        return int(self.component_mask_indices.shape[0])


def _long_tensor(values: list[int], device: th.device) -> th.Tensor:
    return th.tensor(values, dtype=th.long, device=device)


def build_mass_shatter_candidates(
    bond_logits: th.Tensor,
    chemical_bond_index: th.Tensor,
    atom_batch_indices: th.Tensor,
    atom_ptr: th.Tensor,
    *,
    max_masks: int,
    explore_masks: int,
    exploration_temperature: float,
    include_single_bond_cuts: bool,
    generator: th.Generator | None = None,
    topology_cache: MassShatterTopologyCache | None = None,
) -> MassShatterCandidateBatch:
    """Select masks on CPU and return sparse pathway tensors on the model device.

    Mask identities are deliberately selected from detached logits. Their log
    probabilities are then recomputed from the original logits, preserving the
    gradient to every selected cut/intact bond decision.
    """
    if bond_logits.ndim != 1:
        raise ValueError("bond_logits must be one-dimensional")
    if chemical_bond_index.shape != (2, bond_logits.shape[0]):
        raise ValueError("chemical_bond_index must have shape (2, num_chemical_bonds)")
    if chemical_bond_index.dtype != th.long:
        raise ValueError("chemical_bond_index must have dtype torch.long")
    if atom_batch_indices.ndim != 1 or atom_batch_indices.dtype != th.long:
        raise ValueError("atom_batch_indices must be a torch.long vector")
    if atom_ptr.ndim != 1 or atom_ptr.dtype != th.long or atom_ptr.numel() < 2:
        raise ValueError("atom_ptr must be a torch.long CSR pointer")
    if max_masks <= 0:
        raise ValueError(f"max_masks must be positive, got {max_masks}")
    if explore_masks < 0 or explore_masks >= max_masks:
        raise ValueError(
            "explore_masks must be nonnegative and smaller than max_masks, "
            f"got {explore_masks} and {max_masks}"
        )
    if not math.isfinite(exploration_temperature) or exploration_temperature <= 0.0:
        raise ValueError(
            "exploration_temperature must be positive, "
            f"got {exploration_temperature}"
        )

    device = bond_logits.device
    num_molecules = atom_ptr.numel() - 1
    if atom_batch_indices.numel() != int(atom_ptr[-1]):
        raise ValueError("atom_ptr does not describe atom_batch_indices")

    if bond_logits.numel() == 0:
        bond_batch_indices = th.empty(0, dtype=th.long, device=device)
    else:
        if bool(
            ((chemical_bond_index < 0) | (chemical_bond_index >= atom_batch_indices.shape[0])).any()
        ):
            raise ValueError("chemical_bond_index references a missing atom")
        left_batch = atom_batch_indices[chemical_bond_index[0]]
        right_batch = atom_batch_indices[chemical_bond_index[1]]
        if not bool((left_batch == right_batch).all()):
            raise ValueError("a chemical bond cannot connect different molecules")
        bond_batch_indices = left_batch

    mask_batch_indices: list[int] = []
    mask_bond_indices: list[list[int]] = []
    component_mask_indices: list[int] = []
    component_batch_indices: list[int] = []
    component_cut_counts: list[int] = []
    atom_indices: list[int] = []
    atom_component_indices: list[int] = []
    complement_atom_indices: list[int] = []
    complement_component_indices: list[int] = []
    boundary_bond_indices: list[int] = []
    boundary_component_indices: list[int] = []

    # 1. Select and translate masks independently for each precursor.
    for molecule_index in range(num_molecules):
        atom_start = int(atom_ptr[molecule_index])
        atom_end = int(atom_ptr[molecule_index + 1])
        num_atoms = atom_end - atom_start
        if num_atoms <= 0:
            raise ValueError("every precursor must contain at least one atom")

        molecule_bond_indices = th.nonzero(
            bond_batch_indices == molecule_index,
            as_tuple=False,
        ).flatten()
        molecule_logits = bond_logits[molecule_bond_indices]
        local_endpoints = chemical_bond_index[:, molecule_bond_indices] - atom_start
        local_endpoints_cpu = local_endpoints.detach().cpu()
        global_bond_indices = molecule_bond_indices.detach().cpu().tolist()
        detached_logits = molecule_logits.detach().cpu().tolist()
        exploit_masks = max_masks - explore_masks
        ranked_masks = list(
            top_k_cut_masks(
                detached_logits,
                exploit_masks,
                include_empty=True,
            )
        )
        selected_bond_sets = {ranked_mask.broken_bond_indices for ranked_mask in ranked_masks}
        forced_bond_sets = [()]
        if include_single_bond_cuts:
            forced_bond_sets.extend(
                (bond_index,) for bond_index in range(len(detached_logits))
            )
        reserved_bond_sets = set(forced_bond_sets)

        # 2. Sample unique masks from the tempered current policy.
        proposal_probabilities = th.sigmoid(
            molecule_logits.detach().cpu() / exploration_temperature
        )
        max_sampling_attempts = max(32, 8 * explore_masks)
        sampled_count = 0
        for _ in range(max_sampling_attempts):
            if sampled_count == explore_masks or proposal_probabilities.numel() == 0:
                break
            sampled_cut_mask = th.rand(
                proposal_probabilities.shape,
                generator=generator,
            ) < proposal_probabilities
            broken_bond_indices = tuple(
                th.nonzero(sampled_cut_mask, as_tuple=False).flatten().tolist()
            )
            if broken_bond_indices in selected_bond_sets | reserved_bond_sets:
                continue
            ranked_masks.append(
                RankedCutMask(
                    broken_bond_indices=broken_bond_indices,
                    log_probability=0.0,
                    log_probability_from_map=0.0,
                )
            )
            selected_bond_sets.add(broken_bond_indices)
            sampled_count += 1

        # 3. Deterministically refill rejected or duplicate exploration draws.
        if len(ranked_masks) < max_masks:
            for refill_mask in iter_ranked_cut_masks(detached_logits, include_empty=True):
                broken_bond_indices = refill_mask.broken_bond_indices
                if broken_bond_indices in selected_bond_sets | reserved_bond_sets:
                    continue
                ranked_masks.append(refill_mask)
                selected_bond_sets.add(broken_bond_indices)
                if len(ranked_masks) == max_masks:
                    break

        # 4. Append forced masks outside the learned proposal budget.
        for broken_bond_indices in forced_bond_sets:
            if broken_bond_indices in selected_bond_sets:
                continue
            ranked_masks.append(
                RankedCutMask(
                    broken_bond_indices=broken_bond_indices,
                    log_probability=0.0,
                    log_probability_from_map=0.0,
                )
            )
            selected_bond_sets.add(broken_bond_indices)
        if topology_cache is None:
            translated_masks = translate_cut_masks(
                ranked_masks,
                num_atoms=num_atoms,
                bond_endpoints=local_endpoints_cpu.tolist(),
            )
        else:
            local_bond_endpoints = tuple(zip(*local_endpoints_cpu.tolist()))
            translated_masks = tuple(
                CutMaskComponents(
                    cut_mask=ranked_mask,
                    component_atom_masks=topology_cache.get_components(
                        num_atoms,
                        local_bond_endpoints,
                        ranked_mask.broken_bond_indices,
                    ),
                )
                for ranked_mask in ranked_masks
            )

        # 5. Record the selected masks; score every one together below.
        for translated_mask in translated_masks:
            mask_index = len(mask_batch_indices)
            mask_batch_indices.append(molecule_index)
            mask_bond_indices.append(
                [
                    global_bond_indices[local_bond_index]
                    for local_bond_index in translated_mask.cut_mask.broken_bond_indices
                ]
            )

            component_masks = translated_mask.component_atom_masks
            local_atom_components: dict[int, int] = {}
            for component_mask in component_masks:
                component_index = len(component_mask_indices)
                component_mask_indices.append(mask_index)
                component_batch_indices.append(molecule_index)
                component_cut_counts.append(translated_mask.cut_mask.cut_count)

                for local_atom_index in range(num_atoms):
                    global_atom_index = atom_start + local_atom_index
                    if component_mask & (1 << local_atom_index):
                        atom_indices.append(global_atom_index)
                        atom_component_indices.append(component_index)
                        local_atom_components[local_atom_index] = component_index
                    else:
                        complement_atom_indices.append(global_atom_index)
                        complement_component_indices.append(component_index)

            # 6. Attach each truly separating cut bond to both adjacent components.
            for local_bond_index in translated_mask.cut_mask.broken_bond_indices:
                left = int(local_endpoints_cpu[0, local_bond_index])
                right = int(local_endpoints_cpu[1, local_bond_index])
                left_component = local_atom_components[left]
                right_component = local_atom_components[right]
                if left_component == right_component:
                    continue
                global_bond_index = global_bond_indices[local_bond_index]
                boundary_bond_indices.extend([global_bond_index, global_bond_index])
                boundary_component_indices.extend([left_component, right_component])

    if not mask_batch_indices:
        raise ValueError("candidate generation produced no cut masks")

    # 7. Score every selected mask in one batched operation. A broken bond
    # changes its contribution from logsigmoid(-logit) to logsigmoid(logit),
    # whose difference is exactly logit.
    mask_batch_indices_t = _long_tensor(mask_batch_indices, device)
    base_log_probabilities = F.logsigmoid(-bond_logits)
    molecule_log_probabilities = th.zeros(
        num_molecules,
        dtype=bond_logits.dtype,
        device=device,
    )
    if bond_logits.numel() > 0:
        molecule_log_probabilities.index_add_(
            0,
            bond_batch_indices,
            base_log_probabilities,
        )
    mask_log_probabilities = molecule_log_probabilities[mask_batch_indices_t]
    mask_indices = [
        mask_index
        for mask_index, bond_indices in enumerate(mask_bond_indices)
        for _ in bond_indices
    ]
    broken_bond_indices = [
        bond_index for bond_indices in mask_bond_indices for bond_index in bond_indices
    ]
    if broken_bond_indices:
        mask_log_probabilities = mask_log_probabilities.index_add(
            0,
            _long_tensor(mask_indices, device),
            bond_logits[_long_tensor(broken_bond_indices, device)],
        )

    return MassShatterCandidateBatch(
        mask_log_probabilities=mask_log_probabilities,
        mask_batch_indices=mask_batch_indices_t,
        component_mask_indices=_long_tensor(component_mask_indices, device),
        component_batch_indices=_long_tensor(component_batch_indices, device),
        component_cut_counts=_long_tensor(component_cut_counts, device),
        atom_indices=_long_tensor(atom_indices, device),
        atom_component_indices=_long_tensor(atom_component_indices, device),
        complement_atom_indices=_long_tensor(complement_atom_indices, device),
        complement_component_indices=_long_tensor(
            complement_component_indices,
            device,
        ),
        boundary_bond_indices=_long_tensor(boundary_bond_indices, device),
        boundary_component_indices=_long_tensor(
            boundary_component_indices,
            device,
        ),
    )
