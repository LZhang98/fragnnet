"""Benchmark-only GPU topology assembly after the existing exact CPU mask selector.

This private hybrid preserves the CPU candidate contract; it is not a public CUDA
backend. Native topology is optional and imported only when explicitly requested.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch as th

from fragnnet.frag.masster_candidates import (
    MassterCandidateBatch,
    MassterCandidateTimings,
    _score_mask_log_probabilities,
)

if TYPE_CHECKING:
    from fragnnet.frag.masster_candidates import _TopologyInput
    from fragnnet.frag.masster_cuda_topology import CudaTopologyBatch


def _validate_grouped_topology(
    requests: Sequence[_TopologyInput],
    representative_of: Sequence[int],
    counts_by_representative: dict[int, int],
) -> None:
    """Apply the reference graph and retained-CSR-prefix checks before view reuse."""
    for index, leader in enumerate(representative_of):
        if index == leader:
            continue
        request, reference = requests[index], requests[leader]
        masks = counts_by_representative[leader]
        ptr_size = masks + 1
        broken_size = int(reference.mask_ptr[masks])
        if (
            request.num_atoms != reference.num_atoms
            or request.endpoints != reference.endpoints
            or request.mask_ptr.size < ptr_size
            or not np.array_equal(request.mask_ptr[:ptr_size], reference.mask_ptr[:ptr_size])
            or request.broken_bonds.size < broken_size
            or not np.array_equal(
                request.broken_bonds[:broken_size], reference.broken_bonds[:broken_size]
            )
        ):
            raise ValueError("candidate_group_ids can only group identical molecular topology")


def _expand_topology_views(
    topology: CudaTopologyBatch,
    representative_of: Sequence[int],
    topology_representatives: Sequence[int],
    atom_ptr_values: Sequence[int],
    bond_indices_by_molecule: Sequence[Sequence[int]],
) -> tuple[dict[str, th.Tensor], np.ndarray, float, float]:
    """Gather ragged representative arrays and apply all view offsets in six batches.

    Host counts determine output sizes. All index arithmetic on candidate arrays
    runs on their device, with no per-molecule tensor operations or shape reads.
    """
    device = topology.arrays[0].device
    packing_start = perf_counter()
    position = {leader: index for index, leader in enumerate(topology_representatives)}
    view_to_rep = np.asarray([position[leader] for leader in representative_of], dtype=np.int64)
    lengths = np.asarray([count.field_lengths for count in topology.counts], dtype=np.int64)
    view_lengths = lengths[view_to_rep]
    source_offsets = np.cumsum(lengths, axis=0) - lengths
    view_offsets = np.cumsum(view_lengths, axis=0) - view_lengths
    bonds_per_view = np.asarray(
        [len(indices) for indices in bond_indices_by_molecule], dtype=np.int64
    )
    bond_offsets = np.cumsum(bonds_per_view) - bonds_per_view
    # Rows 0:6 identify views and their global atom/mask/component/unique/bond starts.
    metadata_rows = [
        np.arange(len(representative_of), dtype=np.int64),
        np.asarray(atom_ptr_values[:-1], dtype=np.int64),
        view_offsets[:, 0],
        view_offsets[:, 1],
        view_offsets[:, 10],
        bond_offsets,
    ]
    domains = ((0,), (1, 2, 9), (3, 4), (5, 6), (7, 8), (10,))
    for fields in domains:
        field = fields[0]
        metadata_rows.extend(
            (
                view_lengths[:, field],
                source_offsets[view_to_rep, field] - view_offsets[:, field],
            )
        )
    metadata_cpu = np.stack(metadata_rows)
    bond_lookup_cpu = np.asarray(
        [bond for indices in bond_indices_by_molecule for bond in indices], dtype=np.int64
    )
    packing_ms = (perf_counter() - packing_start) * 1000.0
    transfer_start = perf_counter()
    metadata = th.from_numpy(metadata_cpu).to(device)
    bond_lookup = th.from_numpy(bond_lookup_cpu).to(device)
    transfer_ms = (perf_counter() - transfer_start) * 1000.0

    # Gather each distinct ragged domain once; fields in a domain share its index map.
    remap_start = perf_counter()
    gathered: dict[int, th.Tensor] = {}
    owners: list[th.Tensor] = []
    for domain, fields in enumerate(domains):
        total = int(view_lengths[:, fields[0]].sum())
        owner = th.repeat_interleave(metadata[0], metadata[6 + 2 * domain], output_size=total)
        source_indices = th.arange(total, device=device) + metadata[7 + 2 * domain][owner]
        owners.append(owner)
        for field in fields:
            gathered[field] = topology.arrays[field][source_indices]
    masks, components, atoms, broken, boundaries, unique = owners
    component_mask_indices = gathered[1] + metadata[2][components]
    mask_indices = gathered[5] + metadata[2][broken]
    broken_bond_indices = bond_lookup[gathered[6] + metadata[5][broken]]
    arrays = {
        "mask_batch_indices": masks,
        "mask_cut_counts": gathered[0],
        "mechanism_mask_indices": th.arange(masks.numel(), dtype=th.long, device=device),
        "mechanism_rule_indices": th.zeros_like(masks),
        "mechanism_boundary_bond_indices": broken_bond_indices,
        "boundary_mechanism_indices": mask_indices,
        "component_mask_indices": component_mask_indices,
        "component_mechanism_indices": component_mask_indices,
        "component_batch_indices": components,
        "component_cut_counts": gathered[2],
        # Ordinary components are connected, so they form no bond.
        "component_formed_bond_counts": th.zeros_like(gathered[2]),
        "component_unique_indices": gathered[9] + metadata[4][components],
        "unique_representative_component_indices": gathered[10] + metadata[3][unique],
        "atom_indices": gathered[3] + metadata[1][atoms],
        "atom_component_indices": gathered[4] + metadata[3][atoms],
        "boundary_bond_indices": bond_lookup[gathered[7] + metadata[5][boundaries]],
        "boundary_component_indices": gathered[8] + metadata[3][boundaries],
        "mask_indices": mask_indices,
        "broken_bond_indices": broken_bond_indices,
    }
    packing_ms += (perf_counter() - remap_start) * 1000.0
    return arrays, view_to_rep, packing_ms, transfer_ms


def _build_cuda_candidates(
    *,
    bond_logits: th.Tensor,
    topology_inputs: Sequence[_TopologyInput],
    topology_representatives: Sequence[int],
    representative_of: Sequence[int],
    atom_ptr_values: Sequence[int],
    bond_indices_by_molecule: Sequence[Sequence[int]],
    bond_batch_values: Sequence[int],
    proposal_single_cut_fractions: Sequence[float],
    selected_mask_counts: Sequence[int],
    total_start: float,
    input_transfer_ms: float,
    mask_selection_ms: float,
    forced_merge_ms: float,
    statistics: list[dict] | None = None,
) -> MassterCandidateBatch:
    """Build the standard candidate batch without detaching mask probabilities."""
    from fragnnet.frag.masster_cuda_topology import build_cuda_topology

    device = bond_logits.device
    topology_start = perf_counter()
    topology = build_cuda_topology(
        [topology_inputs[index] for index in topology_representatives], device
    )
    topology_ms = (perf_counter() - topology_start) * 1000.0
    _validate_grouped_topology(
        topology_inputs,
        representative_of,
        dict(
            zip(
                topology_representatives,
                (count.num_masks for count in topology.counts),
                strict=True,
            )
        ),
    )
    arrays, view_to_rep, packing_ms, transfer_ms = _expand_topology_views(
        topology,
        representative_of,
        topology_representatives,
        atom_ptr_values,
        bond_indices_by_molecule,
    )
    view_counts = [topology.counts[index] for index in view_to_rep]
    unique_counts = tuple(count.num_unique_components for count in view_counts)
    transfer_start = perf_counter()
    bond_batch_indices = th.tensor(bond_batch_values, dtype=th.long, device=device)
    diagnostics = th.tensor(
        [proposal_single_cut_fractions, unique_counts], dtype=bond_logits.dtype, device=device
    )
    transfer_ms += (perf_counter() - transfer_start) * 1000.0
    probability_start = perf_counter()
    probabilities = _score_mask_log_probabilities(
        bond_logits,
        bond_batch_indices,
        arrays["mask_batch_indices"],
        arrays.pop("mask_indices"),
        arrays.pop("broken_bond_indices"),
        len(representative_of),
        None,
    )
    probability_ms = (perf_counter() - probability_start) * 1000.0
    if statistics is not None:
        statistics.append(
            {
                "representative_indices": list(topology_representatives),
                "physical_mask_counts": list(topology.physical_mask_counts),
                "logical_scanned_mask_counts": [
                    count.scanned_mask_count for count in topology.counts
                ],
                "retained_mask_counts": [count.num_masks for count in topology.counts],
                "scratch_bytes": topology.scratch_bytes,
                "native_load_ms": topology.native_load_ms,
                "topology_call_ms": topology_ms,
                "timing_note": "First call includes lazy native compilation if needed; topology includes host count/status synchronization. View remapping is timed as GPU enqueue work.",
            }
        )
    return MassterCandidateBatch(
        mask_log_probabilities=probabilities,
        **arrays,
        proposal_single_cut_fraction=diagnostics[0],
        unique_component_count=diagnostics[1],
        component_counts=tuple(count.num_components for count in view_counts),
        unique_component_counts=unique_counts,
        selected_mask_counts=tuple(selected_mask_counts),
        retained_mask_counts=tuple(count.num_masks for count in view_counts),
        scanned_mask_counts=tuple(count.scanned_mask_count for count in view_counts),
        timings=MassterCandidateTimings(
            input_transfer_ms=input_transfer_ms,
            mask_selection_ms=mask_selection_ms,
            topology_assembly_ms=topology_ms,
            forced_merge_ms=forced_merge_ms,
            smarts_assembly_ms=0.0,
            batch_packing_ms=packing_ms,
            device_transfer_ms=transfer_ms,
            probability_scoring_ms=probability_ms,
            output_transfer_ms=packing_ms + transfer_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        ),
    )
