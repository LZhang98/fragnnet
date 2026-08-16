"""Batch top-ranked bond masks into sparse MASShatter component tensors."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
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
class MassShatterCandidateTimings:
    """Wall-clock milliseconds spent in candidate-generation phases."""

    input_transfer_ms: float
    mask_selection_ms: float
    topology_ms: float
    assembly_ms: float
    output_transfer_ms: float
    total_ms: float


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
    boundary_bond_indices: th.Tensor
    boundary_component_indices: th.Tensor
    timings: MassShatterCandidateTimings

    @property
    def num_components(self) -> int:
        """Return the number of retained component pathways."""
        return int(self.component_mask_indices.shape[0])


def _packed_long_tensors(
    values: Sequence[Sequence[int] | np.ndarray | th.Tensor],
    device: th.device,
) -> tuple[th.Tensor, ...]:
    """Create all candidate index tensors with one host-to-device transfer."""
    cpu_tensors = tuple(th.as_tensor(value, dtype=th.long, device="cpu") for value in values)
    sizes = tuple(tensor.numel() for tensor in cpu_tensors)
    packed_cpu = th.cat(cpu_tensors) if sum(sizes) > 0 else th.empty(0, dtype=th.long)
    packed = packed_cpu.to(device=device)
    return tuple(packed.split(sizes))


def _unpack_component_memberships(
    component_atom_masks: Sequence[int],
    num_atoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode Python bitmasks through NumPy's compiled unpack/nonzero kernels."""
    byte_width = (num_atoms + 7) // 8
    packed = np.frombuffer(
        b"".join(
            int(mask).to_bytes(byte_width, byteorder="little") for mask in component_atom_masks
        ),
        dtype=np.uint8,
    ).reshape(len(component_atom_masks), byte_width)
    membership = np.unpackbits(packed, axis=1, bitorder="little")[:, :num_atoms]
    component_indices, atom_indices = np.nonzero(membership)
    return component_indices, atom_indices


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
        raise ValueError(f"exploration_temperature must be positive, got {exploration_temperature}")

    total_start = perf_counter()
    device = bond_logits.device
    num_bonds = bond_logits.numel()

    # 1. Copy the dynamic logits and static graph indices once. Previously the
    # per-molecule loop issued several synchronizing GPU-to-CPU copies.
    input_transfer_start = perf_counter()
    if device.type == "cpu":
        bond_logits_cpu = bond_logits.detach()
        chemical_bond_index_cpu = chemical_bond_index.detach()
        atom_batch_indices_cpu = atom_batch_indices.detach()
        atom_ptr_cpu = atom_ptr.detach()
    else:
        packed_graph_cpu = (
            th.cat(
                (
                    chemical_bond_index.reshape(-1),
                    atom_batch_indices,
                    atom_ptr,
                )
            )
            .detach()
            .cpu()
        )
        chemical_bond_end = 2 * num_bonds
        atom_batch_end = chemical_bond_end + atom_batch_indices.numel()
        chemical_bond_index_cpu = packed_graph_cpu[:chemical_bond_end].reshape(2, num_bonds)
        atom_batch_indices_cpu = packed_graph_cpu[chemical_bond_end:atom_batch_end]
        atom_ptr_cpu = packed_graph_cpu[atom_batch_end:]
        bond_logits_cpu = bond_logits.detach().cpu()
    input_transfer_ms = (perf_counter() - input_transfer_start) * 1000.0

    num_molecules = atom_ptr_cpu.numel() - 1
    atom_ptr_values = atom_ptr_cpu.tolist()
    if atom_batch_indices.numel() != atom_ptr_values[-1]:
        raise ValueError("atom_ptr does not describe atom_batch_indices")
    chemical_bond_rows = chemical_bond_index_cpu.tolist()
    atom_batch_values = atom_batch_indices_cpu.tolist()
    bond_batch_values: list[int] = []
    bond_indices_by_molecule: list[list[int]] = [[] for _ in range(num_molecules)]
    for bond_index, (left, right) in enumerate(zip(*chemical_bond_rows)):
        if (
            left < 0
            or right < 0
            or left >= len(atom_batch_values)
            or right >= len(atom_batch_values)
        ):
            raise ValueError("chemical_bond_index references a missing atom")
        molecule_index = atom_batch_values[left]
        if molecule_index < 0 or molecule_index >= num_molecules:
            raise ValueError("atom_batch_indices references a missing molecule")
        if molecule_index != atom_batch_values[right]:
            raise ValueError("a chemical bond cannot connect different molecules")
        bond_batch_values.append(molecule_index)
        bond_indices_by_molecule[molecule_index].append(bond_index)
    detached_logits_all = bond_logits_cpu.tolist()

    mask_batch_indices: list[int] = []
    mask_bond_indices: list[list[int]] = []
    component_mask_indices: list[int] = []
    component_batch_indices: list[int] = []
    component_cut_counts: list[int] = []
    atom_index_chunks: list[np.ndarray] = []
    atom_component_index_chunks: list[np.ndarray] = []
    boundary_bond_indices: list[int] = []
    boundary_component_indices: list[int] = []
    mask_selection_ms = 0.0
    topology_ms = 0.0
    assembly_ms = 0.0

    # 2. Select and translate masks independently for each precursor.
    for molecule_index in range(num_molecules):
        atom_start = atom_ptr_values[molecule_index]
        atom_end = atom_ptr_values[molecule_index + 1]
        num_atoms = atom_end - atom_start
        if num_atoms <= 0:
            raise ValueError("every precursor must contain at least one atom")

        selection_start = perf_counter()
        global_bond_indices = bond_indices_by_molecule[molecule_index]
        detached_logits = [detached_logits_all[index] for index in global_bond_indices]
        local_endpoints = tuple(
            tuple(chemical_bond_rows[row][index] - atom_start for index in global_bond_indices)
            for row in range(2)
        )
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
            forced_bond_sets.extend((bond_index,) for bond_index in range(len(detached_logits)))
        reserved_bond_sets = set(forced_bond_sets)

        # Sample unique masks from the tempered current policy.
        if explore_masks > 0:
            proposal_probabilities = th.sigmoid(
                bond_logits_cpu[global_bond_indices] / exploration_temperature
            )
            max_sampling_attempts = max(32, 8 * explore_masks)
            sampled_count = 0
            for _ in range(max_sampling_attempts):
                if sampled_count == explore_masks or proposal_probabilities.numel() == 0:
                    break
                sampled_cut_mask = (
                    th.rand(
                        proposal_probabilities.shape,
                        generator=generator,
                    )
                    < proposal_probabilities
                )
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

        # Deterministically refill rejected or duplicate exploration draws.
        if len(ranked_masks) < max_masks:
            for refill_mask in iter_ranked_cut_masks(detached_logits, include_empty=True):
                broken_bond_indices = refill_mask.broken_bond_indices
                if broken_bond_indices in selected_bond_sets | reserved_bond_sets:
                    continue
                ranked_masks.append(refill_mask)
                selected_bond_sets.add(broken_bond_indices)
                if len(ranked_masks) == max_masks:
                    break

        # Append forced masks outside the learned proposal budget.
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
        mask_selection_ms += (perf_counter() - selection_start) * 1000.0

        # 3. Translate selected masks into connected components.
        topology_start = perf_counter()
        if topology_cache is None:
            translated_masks = translate_cut_masks(
                ranked_masks,
                num_atoms=num_atoms,
                bond_endpoints=local_endpoints,
            )
        else:
            local_bond_endpoints = tuple(zip(*local_endpoints))
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
        topology_ms += (perf_counter() - topology_start) * 1000.0

        # 4. Decode all component bitmasks for this molecule in compiled NumPy
        # kernels, then derive boundary membership from the dense local map.
        assembly_start = perf_counter()
        component_start = len(component_mask_indices)
        local_component_masks: list[int] = []
        local_component_owners: list[int] = []
        for local_mask_index, translated_mask in enumerate(translated_masks):
            mask_index = len(mask_batch_indices)
            mask_batch_indices.append(molecule_index)
            mask_bond_indices.append(
                [
                    global_bond_indices[local_bond_index]
                    for local_bond_index in translated_mask.cut_mask.broken_bond_indices
                ]
            )
            num_local_components = len(translated_mask.component_atom_masks)
            local_component_masks.extend(translated_mask.component_atom_masks)
            local_component_owners.extend([local_mask_index] * num_local_components)
            component_mask_indices.extend([mask_index] * num_local_components)
            component_batch_indices.extend([molecule_index] * num_local_components)
            component_cut_counts.extend([translated_mask.cut_mask.cut_count] * num_local_components)

        local_component_indices, local_atom_indices = _unpack_component_memberships(
            local_component_masks,
            num_atoms,
        )
        global_component_indices = local_component_indices + component_start
        atom_index_chunks.append(local_atom_indices + atom_start)
        atom_component_index_chunks.append(global_component_indices)
        mask_atom_components = np.empty(
            (len(translated_masks), num_atoms),
            dtype=np.int64,
        )
        component_owners = np.asarray(local_component_owners, dtype=np.int64)[
            local_component_indices
        ]
        mask_atom_components[
            component_owners,
            local_atom_indices,
        ] = global_component_indices

        for local_mask_index, translated_mask in enumerate(translated_masks):
            # Attach each truly separating cut bond to both adjacent components.
            for local_bond_index in translated_mask.cut_mask.broken_bond_indices:
                left = local_endpoints[0][local_bond_index]
                right = local_endpoints[1][local_bond_index]
                left_component = int(mask_atom_components[local_mask_index, left])
                right_component = int(mask_atom_components[local_mask_index, right])
                if left_component == right_component:
                    continue
                global_bond_index = global_bond_indices[local_bond_index]
                boundary_bond_indices.extend([global_bond_index, global_bond_index])
                boundary_component_indices.extend([left_component, right_component])
        assembly_ms += (perf_counter() - assembly_start) * 1000.0

    if not mask_batch_indices:
        raise ValueError("candidate generation produced no cut masks")

    atom_indices = np.concatenate(atom_index_chunks)
    atom_component_indices = np.concatenate(atom_component_index_chunks)
    mask_indices = [
        mask_index
        for mask_index, bond_indices in enumerate(mask_bond_indices)
        for _ in bond_indices
    ]
    broken_bond_indices = [
        bond_index for bond_indices in mask_bond_indices for bond_index in bond_indices
    ]

    # 5. Materialize every long index through one packed host-to-device copy.
    output_transfer_start = perf_counter()
    (
        bond_batch_indices,
        mask_batch_indices_t,
        component_mask_indices_t,
        component_batch_indices_t,
        component_cut_counts_t,
        atom_indices_t,
        atom_component_indices_t,
        boundary_bond_indices_t,
        boundary_component_indices_t,
        mask_indices_t,
        broken_bond_indices_t,
    ) = _packed_long_tensors(
        (
            bond_batch_values,
            mask_batch_indices,
            component_mask_indices,
            component_batch_indices,
            component_cut_counts,
            atom_indices,
            atom_component_indices,
            boundary_bond_indices,
            boundary_component_indices,
            mask_indices,
            broken_bond_indices,
        ),
        device,
    )
    output_transfer_ms = (perf_counter() - output_transfer_start) * 1000.0

    # 6. Score every selected mask in one batched operation. A broken bond
    # changes its contribution from logsigmoid(-logit) to logsigmoid(logit),
    # whose difference is exactly logit.
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
    if broken_bond_indices_t.numel() > 0:
        mask_log_probabilities = mask_log_probabilities.index_add(
            0,
            mask_indices_t,
            bond_logits[broken_bond_indices_t],
        )

    return MassShatterCandidateBatch(
        mask_log_probabilities=mask_log_probabilities,
        mask_batch_indices=mask_batch_indices_t,
        component_mask_indices=component_mask_indices_t,
        component_batch_indices=component_batch_indices_t,
        component_cut_counts=component_cut_counts_t,
        atom_indices=atom_indices_t,
        atom_component_indices=atom_component_indices_t,
        boundary_bond_indices=boundary_bond_indices_t,
        boundary_component_indices=boundary_component_indices_t,
        timings=MassShatterCandidateTimings(
            input_transfer_ms=input_transfer_ms,
            mask_selection_ms=mask_selection_ms,
            topology_ms=topology_ms,
            assembly_ms=assembly_ms,
            output_transfer_ms=output_transfer_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        ),
    )
