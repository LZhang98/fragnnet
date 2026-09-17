"""Optional CUDA candidates: exact ranking, topology and differentiable scoring.

Static graph metadata may be prepared on CPU. Proposal logits and ranked-mask CSR
remain on CUDA; only grouped-equality flags and native ragged counts synchronize.
Selected explicitly through ``mass_shatter_candidate_backend: cuda``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch as th

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterCandidateBatch,
    MassShatterCandidateTimings,
    MassShatterStaticGraph,
    _score_mask_log_probabilities,
)
from fragnnet.frag.mass_shatter_cuda_candidates import _expand_topology_views
from fragnnet.frag.mass_shatter_cuda_topology import CudaTopologyGraph

if TYPE_CHECKING:
    from fragnnet.frag.mass_shatter_templates import MassShatterTemplate


@dataclass(frozen=True)
class _CudaGraphBatch:
    """Static layouts; no proposal logits or ranked masks enter host metadata."""

    graphs: tuple[CudaTopologyGraph, ...]
    atom_ptr_values: list[int]
    bond_indices_by_molecule: list[list[int]]
    bond_batch_values: list[int]
    representatives: list[int]
    representative_of: list[int]
    representative_bond_indices: list[int]
    comparison_view_bonds: list[int]
    comparison_leader_bonds: list[int]


def _prepare_graph_batch(
    chemical_bond_index: th.Tensor,
    atom_batch_indices: th.Tensor,
    atom_ptr: th.Tensor,
    group_values: list[int] | None,
    max_unique_components: int,
) -> _CudaGraphBatch:
    """Validate host graph structure and compute deterministic representative layouts."""
    if any(
        tensor.device.type != "cpu"
        for tensor in (chemical_bond_index, atom_batch_indices, atom_ptr)
    ):
        raise ValueError("static graph preparation requires CPU graph metadata")
    pointers = atom_ptr.tolist()
    atoms = atom_batch_indices.tolist()
    edges = chemical_bond_index.tolist()
    num_molecules = len(pointers) - 1
    if pointers[0] != 0 or len(atoms) != pointers[-1]:
        raise ValueError("atom_ptr does not describe atom_batch_indices")
    for index, (start, end) in enumerate(zip(pointers[:-1], pointers[1:], strict=True)):
        if start >= end or any(molecule != index for molecule in atoms[start:end]):
            raise ValueError(
                "atom_ptr requires nonempty contiguous molecule spans in atom_batch_indices"
            )
    bonds: list[list[int]] = [[] for _ in range(num_molecules)]
    bond_batches = []
    for index, (left, right) in enumerate(zip(*edges, strict=True)):
        if left < 0 or right < 0 or left >= len(atoms) or right >= len(atoms):
            raise ValueError("chemical_bond_index references a missing atom")
        molecule = atoms[left]
        if molecule < 0 or molecule >= num_molecules:
            raise ValueError("atom_batch_indices references a missing molecule")
        if molecule != atoms[right]:
            raise ValueError("a chemical bond cannot connect different molecules")
        bonds[molecule].append(index)
        bond_batches.append(molecule)
    graphs, representatives, representative_of = [], [], []
    representative_bonds, compare_views, compare_leaders = [], [], []
    first_of_group: dict[int, int] = {}
    graph_by_representative: dict[int, CudaTopologyGraph] = {}
    for index in range(num_molecules):
        num_atoms = pointers[index + 1] - pointers[index]
        if num_atoms <= 0:
            raise ValueError("every precursor must contain at least one atom")
        endpoints = tuple(
            tuple(row[bond] - pointers[index] for bond in bonds[index]) for row in edges
        )
        graph = CudaTopologyGraph(num_atoms, endpoints, max_unique_components)
        leader = (
            index if group_values is None else first_of_group.setdefault(group_values[index], index)
        )
        representative_of.append(leader)
        if leader == index:
            representatives.append(index)
            representative_bonds.extend(bonds[index])
            graphs.append(graph)
            graph_by_representative[index] = graph
        else:
            if graph != graph_by_representative[leader]:
                raise ValueError("candidate_group_ids can only group identical molecular topology")
            compare_views.extend(bonds[index])
            compare_leaders.extend(bonds[leader])
    return _CudaGraphBatch(
        graphs=tuple(graphs),
        atom_ptr_values=pointers,
        bond_indices_by_molecule=bonds,
        bond_batch_values=bond_batches,
        representatives=representatives,
        representative_of=representative_of,
        representative_bond_indices=representative_bonds,
        comparison_view_bonds=compare_views,
        comparison_leader_bonds=compare_leaders,
    )


def build_mass_shatter_candidates_cuda(
    bond_logits: th.Tensor,
    chemical_bond_index: th.Tensor,
    atom_batch_indices: th.Tensor,
    atom_ptr: th.Tensor,
    *,
    max_masks: int,
    include_single_bond_cuts: bool,
    min_mask_relative_probability: float = 0.0,
    max_unique_components: int | None = None,
    max_enumerated_masks: int | None = None,
    mask_selection_threads: int = 1,
    static_graph: MassShatterStaticGraph | None = None,
    smarts_templates: Sequence[Sequence[MassShatterTemplate]] | None = None,
    mixture_log_weights: th.Tensor | None = None,
    candidate_group_ids: th.Tensor | None = None,
    _statistics: list[dict] | None = None,
) -> MassShatterCandidateBatch:
    """Build exact single-mixture candidates without copying proposal logits to CPU.

    Supported inputs are finite FP32 CUDA logits, unique mode, zero mask floors and
    no SMARTS mechanisms. Explicit native size/workspace limits apply. Every grouped
    view must have identical local graph structure and proposal logits. Selection
    detaches only its discrete decisions; shared probability scoring uses live logits.
    """
    if (
        bond_logits.device.type != "cuda"
        or bond_logits.dtype != th.float32
        or bond_logits.ndim != 1
    ):
        raise ValueError("CUDA candidate prototype requires one-dimensional CUDA float32 logits")
    if mixture_log_weights is not None:
        raise ValueError("CUDA candidate prototype supports one mixture without mixture weights")
    if not math.isfinite(min_mask_relative_probability) or min_mask_relative_probability != 0.0:
        raise ValueError("CUDA candidate prototype requires disabled mask probability floors")
    if max_masks <= 0 or max_unique_components is None or max_unique_components <= 0:
        raise ValueError(
            "CUDA candidate prototype requires positive mask and unique-component targets"
        )
    if max_enumerated_masks is None or max_enumerated_masks < max_masks:
        raise ValueError("CUDA candidate prototype requires an enumeration budget >= max_masks")
    if mask_selection_threads < 1:
        raise ValueError("mask_selection_threads must be at least 1")
    if (
        chemical_bond_index.shape != (2, bond_logits.shape[0])
        or chemical_bond_index.dtype != th.long
    ):
        raise ValueError(
            "chemical_bond_index must be torch.long with shape (2, num_chemical_bonds)"
        )
    if atom_batch_indices.ndim != 1 or atom_batch_indices.dtype != th.long:
        raise ValueError("atom_batch_indices must be a torch.long vector")
    if atom_ptr.ndim != 1 or atom_ptr.dtype != th.long or atom_ptr.numel() < 2:
        raise ValueError("atom_ptr must be a torch.long CSR pointer")
    num_molecules = atom_ptr.numel() - 1
    if smarts_templates is None and static_graph is not None:
        smarts_templates = static_graph.smarts_templates
    if smarts_templates is not None:
        if len(smarts_templates) != num_molecules:
            raise ValueError("smarts_templates must contain one sequence per precursor")
        if any(smarts_templates):
            raise ValueError("CUDA candidate prototype does not support SMARTS mechanisms")
    if candidate_group_ids is not None and (
        candidate_group_ids.shape != (num_molecules,) or candidate_group_ids.dtype != th.long
    ):
        raise ValueError("candidate_group_ids must be a torch.long vector with one id per molecule")

    # 1. Prepare static layouts and check grouped logits with one batched device flag.
    total_start = perf_counter()
    device = bond_logits.device
    if static_graph is None:
        original_graph_tensors = (chemical_bond_index, atom_batch_indices, atom_ptr)
        if len({tensor.device for tensor in original_graph_tensors}) == 1:
            graph_lengths = [tensor.numel() for tensor in original_graph_tensors]
            packed_graph_cpu = (
                th.cat([tensor.reshape(-1) for tensor in original_graph_tensors]).detach().cpu()
            )
            graph_parts = packed_graph_cpu.split(graph_lengths)
            graph_tensors = (graph_parts[0].reshape_as(chemical_bond_index), *graph_parts[1:])
        else:
            graph_tensors = tuple(tensor.detach().cpu() for tensor in original_graph_tensors)
    else:
        graph_tensors = (
            static_graph.chemical_bond_index,
            static_graph.atom_batch_indices,
            static_graph.atom_ptr,
        )
    group_values = (
        None if candidate_group_ids is None else candidate_group_ids.detach().cpu().tolist()
    )
    graph_batch = _prepare_graph_batch(*graph_tensors, group_values, max_unique_components)
    host_indices = (
        graph_batch.representative_bond_indices,
        graph_batch.comparison_view_bonds,
        graph_batch.comparison_leader_bonds,
        graph_batch.bond_batch_values,
    )
    lengths = [len(indices) for indices in host_indices]
    packed = th.from_numpy(
        np.asarray([index for indices in host_indices for index in indices], dtype=np.int64)
    ).to(device)
    representative_bonds, compare_views, compare_leaders, bond_batches = packed.split(lengths)
    detached_logits = bond_logits.detach()
    if (
        compare_views.numel()
        and not th.all(detached_logits[compare_views] == detached_logits[compare_leaders]).item()
    ):
        raise ValueError("candidate_group_ids can only group identical proposal logits")
    representative_logits = detached_logits[representative_bonds]
    input_ms = (perf_counter() - total_start) * 1000.0

    # 2. Select exact forced-prefix CSR on GPU, then feed it directly to GPU topology.
    from fragnnet.frag.mass_shatter_cuda_selection import rank_batch_cuda
    from fragnnet.frag.mass_shatter_cuda_topology import build_cuda_topology_device_masks

    selection_start = perf_counter()
    ranked = rank_batch_cuda(
        representative_logits,
        [len(graph.endpoints[0]) for graph in graph_batch.graphs],
        ranked_mask_budget=max_enumerated_masks,
        include_single_bond_cuts=include_single_bond_cuts,
    )
    selection_ms = (perf_counter() - selection_start) * 1000.0
    topology_start = perf_counter()
    topology = build_cuda_topology_device_masks(
        graph_batch.graphs,
        ranked.mask_ptr,
        ranked.broken_bonds,
        mask_counts=[count.num_masks for count in ranked.counts],
        forced_mask_counts=[count.forced_mask_count for count in ranked.counts],
    )
    topology_ms = (perf_counter() - topology_start) * 1000.0

    # 3. Expand representative outputs to every view, then score from original logits.
    arrays, view_to_rep, packing_ms, transfer_ms = _expand_topology_views(
        topology,
        graph_batch.representative_of,
        graph_batch.representatives,
        graph_batch.atom_ptr_values,
        graph_batch.bond_indices_by_molecule,
    )
    view_counts = [topology.counts[index] for index in view_to_rep]
    selected_counts = [ranked.counts[index].num_masks for index in view_to_rep]
    unique_counts = tuple(count.num_unique_components for count in view_counts)
    transfer_start = perf_counter()
    view_to_rep_t = th.from_numpy(view_to_rep).to(device)
    unique_counts_t = th.tensor(unique_counts, dtype=bond_logits.dtype, device=device)
    transfer_ms += (perf_counter() - transfer_start) * 1000.0
    probability_start = perf_counter()
    proposal_fraction = ranked.device_counts[view_to_rep_t, 2].to(
        bond_logits.dtype
    ) / ranked.device_counts[view_to_rep_t, 1].to(bond_logits.dtype)
    probabilities = _score_mask_log_probabilities(
        bond_logits,
        bond_batches,
        arrays["mask_batch_indices"],
        arrays.pop("mask_indices"),
        arrays.pop("broken_bond_indices"),
        num_molecules,
        None,
    )
    probability_ms = (perf_counter() - probability_start) * 1000.0
    if _statistics is not None:
        _statistics.append(
            {
                "representative_indices": graph_batch.representatives,
                "selected_mask_counts": [count.num_masks for count in ranked.counts],
                "learned_mask_counts": [count.learned_mask_count for count in ranked.counts],
                "learned_single_cut_counts": [
                    count.learned_single_cut_count for count in ranked.counts
                ],
                "physical_mask_counts": list(topology.physical_mask_counts),
                "logical_scanned_mask_counts": [
                    count.scanned_mask_count for count in topology.counts
                ],
                "selection_scratch_bytes": ranked.scratch_bytes,
                "topology_scratch_bytes": topology.scratch_bytes,
                "selection_native_load_ms": ranked.native_load_ms,
                "topology_native_load_ms": topology.native_load_ms,
                "selection_call_ms": selection_ms,
                "topology_call_ms": topology_ms,
                "timing_note": "Native load includes cold compilation if needed; ranking includes forced merging and a batched host count read. Graph metadata and grouped-equality flags may synchronize; logits and ranked masks remain on CUDA.",
            }
        )
    return MassShatterCandidateBatch(
        mask_log_probabilities=probabilities,
        **arrays,
        proposal_single_cut_fraction=proposal_fraction,
        unique_component_count=unique_counts_t,
        component_counts=tuple(count.num_components for count in view_counts),
        unique_component_counts=unique_counts,
        selected_mask_counts=tuple(selected_counts),
        retained_mask_counts=tuple(count.num_masks for count in view_counts),
        scanned_mask_counts=tuple(count.scanned_mask_count for count in view_counts),
        timings=MassShatterCandidateTimings(
            input_transfer_ms=input_ms,
            mask_selection_ms=selection_ms,
            topology_assembly_ms=topology_ms,
            forced_merge_ms=0.0,
            smarts_assembly_ms=0.0,
            batch_packing_ms=packing_ms,
            device_transfer_ms=transfer_ms,
            probability_scoring_ms=probability_ms,
            output_transfer_ms=packing_ms + transfer_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        ),
    )
