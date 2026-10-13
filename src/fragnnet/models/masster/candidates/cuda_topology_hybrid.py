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

from fragnnet.models.masster.candidates.builder import (
    MassterCandidateBatch,
    MassterCandidateTimings,
    _build_cpu_candidates,
    _score_mask_log_probabilities,
)
from fragnnet.models.masster.candidates.cuda_candidates import _expand_topology_views

if TYPE_CHECKING:
    from fragnnet.models.masster.candidates.builder import _TopologyInput


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
    from fragnnet.models.masster.candidates.cuda_topology import build_cuda_topology

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


def build_masster_candidates_cuda_topology(
    bond_logits: th.Tensor,
    *args,
    max_unique_components: int | None,
    min_mask_relative_probability: float = 0.0,
    statistics: list[dict] | None = None,
    **kwargs,
) -> MassterCandidateBatch:
    """Benchmark entry point: CPU mask selection, then GPU topology assembly.

    Takes the arguments of ``build_masster_candidates`` (without ``candidate_backend``) and
    returns the same fields; only the post-selection topology construction runs on CUDA.
    """
    if bond_logits.device.type != "cuda" or bond_logits.dtype != th.float32:
        raise ValueError("CUDA topology prototype requires CUDA float32 bond logits")
    if bond_logits.ndim != 1 or max_unique_components is None:
        raise ValueError("CUDA topology prototype requires one mixture and unique mode")
    if min_mask_relative_probability != 0.0:
        raise ValueError("CUDA topology prototype requires disabled mask probability floors")

    def topology_builder(*, smarts_templates, **topology_kwargs):
        if smarts_templates is not None and any(smarts_templates):
            raise ValueError("CUDA topology prototype does not support SMARTS mechanisms")
        return _build_cuda_candidates(statistics=statistics, **topology_kwargs)

    return _build_cpu_candidates(
        bond_logits,
        *args,
        max_unique_components=max_unique_components,
        min_mask_relative_probability=min_mask_relative_probability,
        **kwargs,
        topology_builder=topology_builder,
    )
