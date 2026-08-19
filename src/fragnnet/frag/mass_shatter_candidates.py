"""Batch top-ranked bond masks into sparse MASShatter component tensors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch as th
import torch.nn.functional as F

from fragnnet.frag.bond_mask_generator import (
    RankedCutMask,
    iter_ranked_cut_masks,
    top_k_cut_masks,
    translate_cut_masks,
)


@dataclass(frozen=True)
class MassShatterStaticGraph:
    """CPU graph indices retained across Lightning's device transfer."""

    chemical_bond_index: th.Tensor
    atom_batch_indices: th.Tensor
    atom_ptr: th.Tensor


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
    mask_cut_counts: th.Tensor
    component_mask_indices: th.Tensor
    component_batch_indices: th.Tensor
    component_cut_counts: th.Tensor
    atom_indices: th.Tensor
    atom_component_indices: th.Tensor
    boundary_bond_indices: th.Tensor
    boundary_component_indices: th.Tensor
    exploration_mask_count: th.Tensor
    exploration_mean_cut_count: th.Tensor
    proposal_single_cut_fraction: th.Tensor
    unique_component_count: th.Tensor
    timings: MassShatterCandidateTimings

    @property
    def num_components(self) -> int:
        """Return the number of retained component pathways."""
        return int(self.component_mask_indices.shape[0])


def _packed_long_tensors(
    values: Sequence[Sequence[int] | np.ndarray],
    device: th.device,
) -> tuple[th.Tensor, ...]:
    """Create all candidate index tensors with one host-to-device transfer."""
    cpu_arrays = tuple(np.asarray(value, dtype=np.int64).reshape(-1) for value in values)
    sizes = tuple(array.size for array in cpu_arrays)
    packed_array = (
        np.concatenate(cpu_arrays) if sum(sizes) > 0 else np.empty(0, dtype=np.int64)
    )
    packed = th.from_numpy(packed_array).to(device=device)
    return tuple(packed.split(sizes))


def _unpack_component_memberships(
    component_atom_masks: Sequence[int],
    num_atoms: int,
    expected_memberships: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode component bitmasks into flat component/atom membership arrays."""
    if num_atoms <= 256:
        from fragnnet.frag.mass_shatter_core import unpack_component_memberships_raw

        return unpack_component_memberships_raw(
            list(component_atom_masks),
            num_atoms,
            expected_memberships,
        )
    byte_width = (num_atoms + 7) // 8
    packed = np.frombuffer(
        b"".join(
            int(mask).to_bytes(byte_width, byteorder="little") for mask in component_atom_masks
        ),
        dtype=np.uint8,
    ).reshape(len(component_atom_masks), byte_width)
    membership = np.unpackbits(packed, axis=1, bitorder="little")[:, :num_atoms]
    component_indices, atom_indices = np.nonzero(membership)
    if component_indices.size != expected_memberships:
        raise ValueError("component masks do not partition every atom exactly once")
    return component_indices, atom_indices


def build_mass_shatter_candidates(
    bond_logits: th.Tensor,
    chemical_bond_index: th.Tensor,
    atom_batch_indices: th.Tensor,
    atom_ptr: th.Tensor,
    *,
    max_masks: int,
    explore_masks: int,
    exploration_mode: str,
    exploration_temperature: float,
    exploration_cut_count_weights: Sequence[float],
    include_single_bond_cuts: bool,
    min_mask_probability: float = 0.0,
    generator: th.Generator | None = None,
    static_graph: MassShatterStaticGraph | None = None,
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
    if not math.isfinite(min_mask_probability) or not 0.0 <= min_mask_probability <= 1.0:
        raise ValueError(
            "min_mask_probability must be finite and in [0, 1], "
            f"got {min_mask_probability}"
        )
    if explore_masks < 0:
        raise ValueError(f"explore_masks must be nonnegative, got {explore_masks}")
    if exploration_mode not in {"bernoulli_replace", "bounded_additive"}:
        raise ValueError(f"unknown exploration mode: {exploration_mode}")
    if exploration_mode == "bernoulli_replace" and explore_masks >= max_masks:
        raise ValueError(
            "bernoulli_replace requires explore_masks smaller than max_masks, "
            f"got {explore_masks} and {max_masks}"
        )
    if not math.isfinite(exploration_temperature) or exploration_temperature <= 0.0:
        raise ValueError(f"exploration_temperature must be positive, got {exploration_temperature}")
    cut_count_weights = tuple(float(weight) for weight in exploration_cut_count_weights)
    if (
        len(cut_count_weights) < 3
        or any(not math.isfinite(weight) or weight < 0.0 for weight in cut_count_weights)
        or cut_count_weights[0] != 0.0
        or cut_count_weights[1] != 0.0
        or sum(cut_count_weights[2:]) <= 0.0
    ):
        raise ValueError(
            "exploration_cut_count_weights must contain nonnegative weights indexed by "
            "cut count, with zero weights for 0/1 cuts and positive mass above 1"
        )

    total_start = perf_counter()
    device = bond_logits.device
    num_bonds = bond_logits.numel()

    # 1. Copy only dynamic logits when Lightning retained a CPU graph sidecar.
    input_transfer_start = perf_counter()
    if static_graph is not None:
        chemical_bond_index_cpu = static_graph.chemical_bond_index
        atom_batch_indices_cpu = static_graph.atom_batch_indices
        atom_ptr_cpu = static_graph.atom_ptr
        bond_logits_cpu = bond_logits.detach().cpu()
    elif device.type == "cpu":
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

    mask_batch_index_chunks: list[np.ndarray] = []
    mask_cut_count_chunks: list[np.ndarray] = []
    mask_index_chunks: list[np.ndarray] = []
    broken_bond_index_chunks: list[np.ndarray] = []
    component_mask_index_chunks: list[np.ndarray] = []
    component_batch_index_chunks: list[np.ndarray] = []
    component_cut_count_chunks: list[np.ndarray] = []
    atom_index_chunks: list[np.ndarray] = []
    atom_component_index_chunks: list[np.ndarray] = []
    boundary_bond_index_chunks: list[np.ndarray] = []
    boundary_component_index_chunks: list[np.ndarray] = []
    exploration_mask_counts: list[int] = []
    exploration_mean_cut_counts: list[float] = []
    proposal_single_cut_fractions: list[float] = []
    unique_component_counts: list[int] = []
    mask_selection_ms = 0.0
    topology_ms = 0.0
    assembly_ms = 0.0
    num_masks_so_far = 0
    num_components_so_far = 0

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
        exploit_masks = (
            max_masks - explore_masks
            if exploration_mode == "bernoulli_replace"
            else max_masks
        )
        ranked_masks = list(
            top_k_cut_masks(
                detached_logits,
                exploit_masks,
                include_empty=True,
                min_probability=min_mask_probability,
            )
        )
        if min_mask_probability > 0.0 and not ranked_masks:
            raise ValueError(
                "mass_shatter_min_mask_probability excludes every learned mask "
                f"for precursor {molecule_index}"
            )
        selected_bond_sets = {ranked_mask.broken_bond_indices for ranked_mask in ranked_masks}
        forced_bond_sets = [()]
        if include_single_bond_cuts:
            forced_bond_sets.extend((bond_index,) for bond_index in range(len(detached_logits)))
        reserved_bond_sets = set(forced_bond_sets)
        unavailable_bond_sets = selected_bond_sets | reserved_bond_sets

        sampled_count = 0
        sampled_cut_count = 0
        if explore_masks > 0 and exploration_mode == "bernoulli_replace":
            # Legacy complete-mask Bernoulli exploration retained for old runs.
            proposal_probabilities = th.sigmoid(
                bond_logits_cpu[global_bond_indices] / exploration_temperature
            )
            max_sampling_attempts = max(32, 8 * explore_masks)
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
                if broken_bond_indices in unavailable_bond_sets:
                    continue
                ranked_masks.append(
                    RankedCutMask(
                        broken_bond_indices=broken_bond_indices,
                        log_probability=0.0,
                        log_probability_from_map=0.0,
                    )
                )
                selected_bond_sets.add(broken_bond_indices)
                unavailable_bond_sets.add(broken_bond_indices)
                sampled_count += 1
                sampled_cut_count += len(broken_bond_indices)

        # Preserve the historical fixed total budget for legacy experiments.
        if exploration_mode == "bernoulli_replace" and len(ranked_masks) < max_masks:
            min_log_probability = (
                math.log(min_mask_probability)
                if min_mask_probability > 0.0
                else -math.inf
            )
            for refill_mask in iter_ranked_cut_masks(detached_logits, include_empty=True):
                if refill_mask.log_probability < min_log_probability:
                    break
                broken_bond_indices = refill_mask.broken_bond_indices
                if broken_bond_indices in unavailable_bond_sets:
                    continue
                ranked_masks.append(refill_mask)
                selected_bond_sets.add(broken_bond_indices)
                unavailable_bond_sets.add(broken_bond_indices)
                if len(ranked_masks) == max_masks:
                    break

        if explore_masks > 0 and exploration_mode == "bounded_additive":
            # Draw cardinalities and Gumbel top-k masks in one compiled call.
            valid_weight_count = min(len(cut_count_weights), len(detached_logits) + 1)
            valid_weights = cut_count_weights[:valid_weight_count]
            if sum(valid_weights) > 0.0:
                supported_cut_counts = {
                    cut_count
                    for cut_count, weight in enumerate(valid_weights)
                    if weight > 0.0
                }
                proposal_space_size = sum(
                    math.comb(len(detached_logits), cut_count)
                    for cut_count in supported_cut_counts
                )
                unavailable_proposal_count = sum(
                    len(bond_set) in supported_cut_counts
                    for bond_set in unavailable_bond_sets
                )
                target_sample_count = min(
                    explore_masks,
                    proposal_space_size - unavailable_proposal_count,
                )
                if target_sample_count > 0:
                    from fragnnet.frag.mass_shatter_core import bounded_additive_masks_raw

                    max_sampling_attempts = max(32, 8 * explore_masks)
                    proposal_logits = [
                        detached_logits[index] / exploration_temperature
                        for index in range(len(detached_logits))
                    ]
                    sampler_seed = int(
                        th.randint(
                            0,
                            2**63 - 1,
                            (),
                            generator=generator,
                            dtype=th.int64,
                        ).item()
                    )
                    sampled_masks = bounded_additive_masks_raw(
                        proposal_logits,
                        valid_weights,
                        unavailable_bond_sets,
                        target_sample_count,
                        max_sampling_attempts,
                        sampler_seed,
                    )
                    for broken_bond_indices in sampled_masks:
                        ranked_masks.append(
                            RankedCutMask(
                                broken_bond_indices=broken_bond_indices,
                                log_probability=0.0,
                                log_probability_from_map=0.0,
                            )
                        )
                        selected_bond_sets.add(broken_bond_indices)
                        unavailable_bond_sets.add(broken_bond_indices)
                        sampled_count += 1
                        sampled_cut_count += len(broken_bond_indices)

        exploration_mask_counts.append(sampled_count)
        exploration_mean_cut_counts.append(
            sampled_cut_count / sampled_count if sampled_count > 0 else 0.0
        )
        proposal_single_cut_fractions.append(
            sum(mask.cut_count == 1 for mask in ranked_masks) / len(ranked_masks)
        )

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
        translated_masks = translate_cut_masks(
            ranked_masks,
            num_atoms=num_atoms,
            bond_endpoints=local_endpoints,
        )
        topology_ms += (perf_counter() - topology_start) * 1000.0

        # 4. Decode all component bitmasks for this molecule in compiled NumPy
        # kernels, then derive boundary membership from the dense local map.
        assembly_start = perf_counter()
        mask_start = num_masks_so_far
        component_start = num_components_so_far
        num_local_masks = len(translated_masks)
        local_mask_indices = np.arange(
            mask_start,
            mask_start + num_local_masks,
            dtype=np.int64,
        )
        local_cut_counts = np.fromiter(
            (translated_mask.cut_mask.cut_count for translated_mask in translated_masks),
            dtype=np.int64,
            count=num_local_masks,
        )
        local_component_counts = np.fromiter(
            (len(translated_mask.component_atom_masks) for translated_mask in translated_masks),
            dtype=np.int64,
            count=num_local_masks,
        )
        local_component_masks = [
            component_mask
            for translated_mask in translated_masks
            for component_mask in translated_mask.component_atom_masks
        ]
        local_component_owners = np.repeat(
            np.arange(num_local_masks, dtype=np.int64),
            local_component_counts,
        )
        mask_batch_index_chunks.append(
            np.full(num_local_masks, molecule_index, dtype=np.int64)
        )
        mask_cut_count_chunks.append(local_cut_counts)
        component_mask_index_chunks.append(
            np.repeat(local_mask_indices, local_component_counts)
        )
        component_batch_index_chunks.append(
            np.full(len(local_component_masks), molecule_index, dtype=np.int64)
        )
        component_cut_count_chunks.append(
            np.repeat(local_cut_counts, local_component_counts)
        )
        num_masks_so_far += num_local_masks
        num_components_so_far += len(local_component_masks)
        unique_component_counts.append(len(set(local_component_masks)))

        local_component_indices, local_atom_indices = _unpack_component_memberships(
            local_component_masks,
            num_atoms,
            num_local_masks * num_atoms,
        )
        global_component_indices = local_component_indices + component_start
        atom_index_chunks.append(local_atom_indices + atom_start)
        atom_component_index_chunks.append(global_component_indices)
        mask_atom_components = np.empty(
            (len(translated_masks), num_atoms),
            dtype=np.int64,
        )
        component_owners = local_component_owners[local_component_indices]
        mask_atom_components[
            component_owners,
            local_atom_indices,
        ] = global_component_indices
        local_broken_bond_indices = np.fromiter(
            (
                local_bond_index
                for translated_mask in translated_masks
                for local_bond_index in translated_mask.cut_mask.broken_bond_indices
            ),
            dtype=np.int64,
            count=int(local_cut_counts.sum()),
        )
        local_broken_bond_owners = np.repeat(
            np.arange(num_local_masks, dtype=np.int64),
            local_cut_counts,
        )
        global_bond_indices_array = np.asarray(global_bond_indices, dtype=np.int64)
        global_broken_bond_indices = global_bond_indices_array[
            local_broken_bond_indices
        ]
        mask_index_chunks.append(local_mask_indices[local_broken_bond_owners])
        broken_bond_index_chunks.append(global_broken_bond_indices)

        if local_broken_bond_indices.size > 0:
            local_endpoints_array = np.asarray(local_endpoints, dtype=np.int64)
            left_components = mask_atom_components[
                local_broken_bond_owners,
                local_endpoints_array[0, local_broken_bond_indices],
            ]
            right_components = mask_atom_components[
                local_broken_bond_owners,
                local_endpoints_array[1, local_broken_bond_indices],
            ]
            separating = left_components != right_components
            separating_global_bonds = global_broken_bond_indices[separating]
            boundary_bond_index_chunks.append(
                np.repeat(separating_global_bonds, 2)
            )
            boundary_component_index_chunks.append(
                np.column_stack(
                    (left_components[separating], right_components[separating])
                ).reshape(-1)
            )
        assembly_ms += (perf_counter() - assembly_start) * 1000.0

    if num_masks_so_far == 0:
        raise ValueError("candidate generation produced no cut masks")

    mask_batch_indices = np.concatenate(mask_batch_index_chunks)
    mask_cut_counts = np.concatenate(mask_cut_count_chunks)
    component_mask_indices = np.concatenate(component_mask_index_chunks)
    component_batch_indices = np.concatenate(component_batch_index_chunks)
    component_cut_counts = np.concatenate(component_cut_count_chunks)
    atom_indices = np.concatenate(atom_index_chunks)
    atom_component_indices = np.concatenate(atom_component_index_chunks)
    boundary_bond_indices = (
        np.concatenate(boundary_bond_index_chunks)
        if boundary_bond_index_chunks
        else np.empty(0, dtype=np.int64)
    )
    boundary_component_indices = (
        np.concatenate(boundary_component_index_chunks)
        if boundary_component_index_chunks
        else np.empty(0, dtype=np.int64)
    )
    mask_indices = np.concatenate(mask_index_chunks)
    broken_bond_indices = np.concatenate(broken_bond_index_chunks)

    # 5. Materialize every long index through one packed host-to-device copy.
    output_transfer_start = perf_counter()
    (
        bond_batch_indices,
        mask_batch_indices_t,
        mask_cut_counts_t,
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
            mask_cut_counts,
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
        mask_cut_counts=mask_cut_counts_t,
        component_mask_indices=component_mask_indices_t,
        component_batch_indices=component_batch_indices_t,
        component_cut_counts=component_cut_counts_t,
        atom_indices=atom_indices_t,
        atom_component_indices=atom_component_indices_t,
        boundary_bond_indices=boundary_bond_indices_t,
        boundary_component_indices=boundary_component_indices_t,
        exploration_mask_count=th.tensor(
            exploration_mask_counts,
            dtype=bond_logits.dtype,
            device=device,
        ),
        exploration_mean_cut_count=th.tensor(
            exploration_mean_cut_counts,
            dtype=bond_logits.dtype,
            device=device,
        ),
        proposal_single_cut_fraction=th.tensor(
            proposal_single_cut_fractions,
            dtype=bond_logits.dtype,
            device=device,
        ),
        unique_component_count=th.tensor(
            unique_component_counts,
            dtype=bond_logits.dtype,
            device=device,
        ),
        timings=MassShatterCandidateTimings(
            input_transfer_ms=input_transfer_ms,
            mask_selection_ms=mask_selection_ms,
            topology_ms=topology_ms,
            assembly_ms=assembly_ms,
            output_transfer_ms=output_transfer_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        ),
    )
