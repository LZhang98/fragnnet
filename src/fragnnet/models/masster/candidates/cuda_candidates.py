"""CUDA candidates: exact ranking, topology and differentiable scoring.

Status: PRODUCTION. This is the backend of the current recipe (`i4`) and nearly every MASSter
experiment config, selected with ``masster_candidate_backend: cuda``; the template default
stays ``cpu``. Supports finite FP32 logits, one mixture, unique-fragment mode and no mask
floor; unsupported requests raise instead of falling back to CPU.

Static graph metadata may be prepared on CPU. Proposal logits and ranked-mask CSR
remain on CUDA; only grouped-equality flags and native ragged counts synchronize.
SMARTS template cuts are forced masks in the ranking prefix, and their alternative
assemblies are appended per precursor in the CPU backend's layout.
Selected explicitly through ``masster_candidate_backend: cuda``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch as th

from fragnnet.models.masster.candidates.builder import (
    MassterCandidateBatch,
    MassterCandidateTimings,
    MassterStaticGraph,
    _packed_long_chunk_tensors,
    _score_mask_log_probabilities,
)
from fragnnet.models.masster.candidates.cuda_topology import (
    CudaTopologyBatch,
    CudaTopologyCounts,
    CudaTopologyGraph,
)
from fragnnet.models.masster.candidates.smarts_templates import (
    MassterTemplate,
    _connected_piece_count,
)


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


def _template_forced_masks(
    templates: Sequence[MassterTemplate],
    num_bonds: int,
    include_single_bond_cuts: bool,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return extra forced cut sets and each template's molecule-local mask index.

    Mirrors the CPU forced list ``dict.fromkeys([(), *singles, *template_cuts])``: the
    empty mask is index 0, forced single cuts follow in bond order, then every other
    distinct template cut in first-occurrence order.
    """
    base_forced_count = 1 + (num_bonds if include_single_bond_cuts else 0)
    extra: dict[tuple[int, ...], int] = {}
    positions = []
    for template in templates:
        bond_set = tuple(template.broken_bond_indices)
        if not bond_set:
            positions.append(0)
        elif include_single_bond_cuts and len(bond_set) == 1:
            positions.append(1 + bond_set[0])
        else:
            positions.append(base_forced_count + extra.setdefault(bond_set, len(extra)))
    return tuple(extra), tuple(positions)


def _append_smarts_mechanisms(
    arrays: dict[str, th.Tensor],
    view_templates: Sequence[Sequence[MassterTemplate]],
    view_mask_positions: Sequence[Sequence[int]],
    view_counts: Sequence[CudaTopologyCounts],
    view_endpoints: Sequence[tuple[tuple[int, ...], ...]],
    atom_ptr_values: Sequence[int],
    bond_indices_by_molecule: Sequence[Sequence[int]],
) -> tuple[dict[str, th.Tensor], list[int], list[int]]:
    """Insert SMARTS mechanisms after each view's ordinary rows, as the CPU backend does.

    Each template becomes one extra mechanism under its forced cut mask, with one
    component per template side. A side whose atoms are connected in the precursor is
    exactly an ordinary component of that mask, since the template cuts every crossing
    bond and no internal one; its unique index is read on the device from that
    component. A disconnected side can come from no ordinary mechanism, so it is a new
    unique fragment, deduplicated per view in template order. Every other value is
    host-known, and rows are placed so each view reads [ordinary rows, SMARTS rows].

    Args:
        arrays: Expanded ordinary candidate fields, without ``mask_indices`` and
            ``broken_bond_indices``; mechanisms equal masks.
        view_templates: Eligible templates for each view.
        view_mask_positions: Molecule-local forced mask index of each template.
        view_counts: Ordinary per-view field lengths.
        view_endpoints: Molecule-local bond endpoints for each view.
        atom_ptr_values: Global atom CSR pointer over views.
        bond_indices_by_molecule: Global chemical bond ids of each view, in local order.

    Returns:
        The updated fields, then the added component and unique-fragment counts per view.
    """
    device = arrays["mask_batch_indices"].device
    num_views = len(view_counts)
    # Columns: masks/mechanisms, mechanism bonds, components, atoms, boundaries, uniques.
    base_lengths = np.asarray(
        [
            (
                count.num_masks,
                count.num_broken_bonds,
                count.num_components,
                count.num_atom_memberships,
                count.num_boundary_memberships,
                count.num_unique_components,
            )
            for count in view_counts
        ],
        dtype=np.int64,
    ).reshape(num_views, 6)
    base_starts = np.cumsum(base_lengths, axis=0) - base_lengths
    extra_lengths = np.zeros_like(base_lengths)

    # 1. Build SMARTS rows in final coordinates, one view at a time.
    mechanism_rows: list[int] = []  # (mask, rule, position) triples
    mechanism_bond_rows: list[int] = []  # (bond, mechanism, position)
    # (mask, mechanism, view, cut count, unique, position, formed bonds)
    component_rows: list[int] = []
    atom_rows: list[int] = []  # (atom, component, position)
    boundary_rows: list[int] = []  # (bond, component, position)
    unique_rows: list[int] = []  # (representative component, position)
    query_rows: list[int] = []  # (component row, mask, atom, atom count, unique shift)
    for view, (templates, positions) in enumerate(
        zip(view_templates, view_mask_positions, strict=True)
    ):
        if not templates:
            continue
        final_starts = base_starts[view] + extra_lengths[:view].sum(axis=0)
        extra_starts = final_starts + base_lengths[view]
        added = np.zeros(6, dtype=np.int64)
        global_bonds = bond_indices_by_molecule[view]
        atom_start = atom_ptr_values[view]
        num_atoms = atom_ptr_values[view + 1] - atom_start
        new_uniques: dict[tuple[int, ...], int] = {}
        for template, local_mask in zip(templates, positions, strict=True):
            mask_index = int(base_starts[view, 0] + local_mask)
            mechanism_index = int(extra_starts[0] + added[0])
            mechanism_rows += (mask_index, template.rule_index, mechanism_index)
            added[0] += 1
            cut_bonds = [global_bonds[bond] for bond in template.broken_bond_indices]
            for bond in cut_bonds:
                mechanism_bond_rows += (bond, mechanism_index, int(extra_starts[1] + added[1]))
                added[1] += 1
            for component in template.components:
                component_index = int(extra_starts[2] + added[2])
                added[2] += 1
                num_pieces = _connected_piece_count(component, view_endpoints[view])
                if num_pieces == 1:
                    unique_index = -1
                    query_rows += (
                        len(component_rows) // 7,
                        mask_index,
                        atom_start + component[0],
                        num_atoms,
                        int(final_starts[5] - base_starts[view, 5]),
                    )
                else:
                    unique_index = new_uniques.get(tuple(component), -1)
                    if unique_index < 0:
                        unique_index = int(extra_starts[5] + added[5])
                        new_uniques[tuple(component)] = unique_index
                        unique_rows += (component_index, unique_index)
                        added[5] += 1
                component_rows += (
                    mask_index,
                    mechanism_index,
                    view,
                    len(cut_bonds),
                    unique_index,
                    component_index,
                    num_pieces - 1,
                )
                for atom in component:
                    atom_rows += (atom_start + atom, component_index, int(extra_starts[3] + added[3]))
                    added[3] += 1
                for bond in cut_bonds:
                    boundary_rows += (bond, component_index, int(extra_starts[4] + added[4]))
                    added[4] += 1
        extra_lengths[view] = added
    (
        mechanism_extra,
        mechanism_bond_extra,
        component_extra,
        atom_extra,
        boundary_extra,
        unique_extra,
        query_extra,
        before_extra,
    ), *_ = _packed_long_chunk_tensors(
        (
            (mechanism_rows,),
            (mechanism_bond_rows,),
            (component_rows,),
            (atom_rows,),
            (boundary_rows,),
            (unique_rows,),
            (query_rows,),
            ((np.cumsum(extra_lengths, axis=0) - extra_lengths).T,),
        ),
        device,
    )
    mechanism_extra = mechanism_extra.view(-1, 3)
    mechanism_bond_extra = mechanism_bond_extra.view(-1, 3)
    component_extra = component_extra.view(-1, 7)
    atom_extra = atom_extra.view(-1, 3)
    boundary_extra = boundary_extra.view(-1, 3)
    unique_extra = unique_extra.view(-1, 2)
    query_extra = query_extra.view(-1, 5)

    # 2. Read connected sides' unique indices from their ordinary component. Components
    # are sorted by mask and atoms by component, and each mask's components partition its
    # molecule's atoms, so a mask's memberships are one contiguous block of that length.
    component_unique_extra = component_extra[:, 4].clone()
    if query_rows:
        query_lengths = query_rows[3::5]
        query_owner = th.repeat_interleave(
            th.arange(len(query_lengths), device=device),
            query_extra[:, 3],
            output_size=sum(query_lengths),
        )
        block_starts = th.searchsorted(
            arrays["atom_component_indices"],
            th.searchsorted(arrays["component_mask_indices"], query_extra[:, 1].contiguous()),
        )
        entry_offsets = th.from_numpy(
            np.concatenate([np.arange(length, dtype=np.int64) for length in query_lengths])
        ).to(device)
        entries = block_starts[query_owner] + entry_offsets
        matches = th.where(
            arrays["atom_indices"][entries] == query_extra[query_owner, 2],
            arrays["atom_component_indices"][entries],
            -1,
        )
        found = th.full_like(query_extra[:, 0], -1).scatter_reduce(
            0, query_owner, matches, reduce="amax", include_self=True
        )
        th._assert_async(
            (found >= 0).all(), "a connected SMARTS side must be an ordinary component"
        )
        component_unique_extra[query_extra[:, 0]] = (
            arrays["component_unique_indices"][found] + query_extra[:, 4]
        )

    # 3. Shift ordinary index values past earlier views' SMARTS rows, then interleave.
    component_owner = arrays["component_batch_indices"]
    owners = {
        "mechanism": arrays["mask_batch_indices"],
        "mechanism_bond": arrays["mask_batch_indices"][arrays["boundary_mechanism_indices"]],
        "component": component_owner,
        "atom": component_owner[arrays["atom_component_indices"]],
        "boundary": component_owner[arrays["boundary_component_indices"]],
        "unique": component_owner[arrays["unique_representative_component_indices"]],
    }
    before = dict(zip(owners, before_extra.view(6, -1), strict=True))
    mechanisms_before = before["mechanism"]
    components_before = before["component"]
    uniques_before = before["unique"]

    def interleave(
        domain: str,
        name: str,
        extra_values: th.Tensor,
        extra_positions: th.Tensor,
        shift: th.Tensor | None = None,
    ) -> th.Tensor:
        values = arrays[name]
        owner = owners[domain]
        if shift is not None:
            values = values + shift[owner]
        output = values.new_empty(values.numel() + extra_values.numel())
        output[th.arange(values.numel(), device=device) + before[domain][owner]] = values
        output[extra_positions] = extra_values
        return output

    updated = dict(arrays)
    updated["mechanism_mask_indices"] = interleave(
        "mechanism", "mechanism_mask_indices", mechanism_extra[:, 0], mechanism_extra[:, 2]
    )
    updated["mechanism_rule_indices"] = interleave(
        "mechanism", "mechanism_rule_indices", mechanism_extra[:, 1], mechanism_extra[:, 2]
    )
    updated["mechanism_boundary_bond_indices"] = interleave(
        "mechanism_bond",
        "mechanism_boundary_bond_indices",
        mechanism_bond_extra[:, 0],
        mechanism_bond_extra[:, 2],
    )
    updated["boundary_mechanism_indices"] = interleave(
        "mechanism_bond",
        "boundary_mechanism_indices",
        mechanism_bond_extra[:, 1],
        mechanism_bond_extra[:, 2],
        mechanisms_before,
    )
    component_positions = component_extra[:, 5]
    for column, name, shift in (
        (0, "component_mask_indices", None),
        (1, "component_mechanism_indices", mechanisms_before),
        (2, "component_batch_indices", None),
        (3, "component_cut_counts", None),
        (6, "component_formed_bond_counts", None),
    ):
        updated[name] = interleave(
            "component", name, component_extra[:, column], component_positions, shift
        )
    updated["component_unique_indices"] = interleave(
        "component",
        "component_unique_indices",
        component_unique_extra,
        component_positions,
        uniques_before,
    )
    updated["atom_indices"] = interleave("atom", "atom_indices", atom_extra[:, 0], atom_extra[:, 2])
    updated["atom_component_indices"] = interleave(
        "atom", "atom_component_indices", atom_extra[:, 1], atom_extra[:, 2], components_before
    )
    updated["boundary_bond_indices"] = interleave(
        "boundary", "boundary_bond_indices", boundary_extra[:, 0], boundary_extra[:, 2]
    )
    updated["boundary_component_indices"] = interleave(
        "boundary",
        "boundary_component_indices",
        boundary_extra[:, 1],
        boundary_extra[:, 2],
        components_before,
    )
    updated["unique_representative_component_indices"] = interleave(
        "unique",
        "unique_representative_component_indices",
        unique_extra[:, 0],
        unique_extra[:, 1],
        components_before,
    )
    return updated, extra_lengths[:, 2].tolist(), extra_lengths[:, 5].tolist()


def build_masster_candidates_cuda(
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
    static_graph: MassterStaticGraph | None = None,
    smarts_templates: Sequence[Sequence[MassterTemplate]] | None = None,
    mixture_log_weights: th.Tensor | None = None,
    candidate_group_ids: th.Tensor | None = None,
    _statistics: list[dict] | None = None,
) -> MassterCandidateBatch:
    """Build exact single-mixture candidates without copying proposal logits to CPU.

    Supported inputs are finite FP32 CUDA logits, unique mode and zero mask floors.
    Explicit native size/workspace limits apply. Every grouped view must have identical
    local graph structure, proposal logits and SMARTS template cuts. Selection detaches
    only its discrete decisions; shared probability scoring uses live logits. SMARTS
    templates produce the CPU backend's fields exactly: their cuts are forced masks, and
    their mechanisms and components follow each view's ordinary rows.
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
    if smarts_templates is not None and len(smarts_templates) != num_molecules:
        raise ValueError("smarts_templates must contain one sequence per precursor")
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
    # Template cuts join each representative's forced prefix; grouped views share one.
    view_templates = (
        ((),) * num_molecules
        if smarts_templates is None
        else tuple(tuple(templates) for templates in smarts_templates)
    )
    view_forced = [
        _template_forced_masks(
            templates,
            len(graph_batch.bond_indices_by_molecule[view]),
            include_single_bond_cuts,
        )
        for view, templates in enumerate(view_templates)
    ]
    if any(
        view_forced[view][0] != view_forced[leader][0]
        for view, leader in enumerate(graph_batch.representative_of)
    ):
        raise ValueError("candidate_group_ids can only group identical molecular topology")
    extra_forced_bond_sets = (
        [view_forced[representative][0] for representative in graph_batch.representatives]
        if any(view_templates)
        else None
    )
    input_ms = (perf_counter() - total_start) * 1000.0

    # 2. Select exact forced-prefix CSR on GPU, then feed it directly to GPU topology.
    from fragnnet.models.masster.candidates.cuda_mask_ranking import rank_batch_cuda
    from fragnnet.models.masster.candidates.cuda_topology import (
        build_cuda_topology_device_masks,
    )

    selection_start = perf_counter()
    ranked = rank_batch_cuda(
        representative_logits,
        [len(graph.endpoints[0]) for graph in graph_batch.graphs],
        ranked_mask_budget=max_enumerated_masks,
        include_single_bond_cuts=include_single_bond_cuts,
        extra_forced_bond_sets=extra_forced_bond_sets,
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
    transfer_start = perf_counter()
    view_to_rep_t = th.from_numpy(view_to_rep).to(device)
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

    # 4. Append SMARTS mechanisms after each view's ordinary rows.
    smarts_start = perf_counter()
    extra_components = extra_uniques = [0] * num_molecules
    if extra_forced_bond_sets is not None:
        arrays, extra_components, extra_uniques = _append_smarts_mechanisms(
            arrays,
            view_templates,
            [positions for _, positions in view_forced],
            view_counts,
            [graph_batch.graphs[index].endpoints for index in view_to_rep],
            graph_batch.atom_ptr_values,
            graph_batch.bond_indices_by_molecule,
        )
    component_counts = tuple(
        count.num_components + extra
        for count, extra in zip(view_counts, extra_components, strict=True)
    )
    unique_counts = tuple(
        count.num_unique_components + extra
        for count, extra in zip(view_counts, extra_uniques, strict=True)
    )
    unique_counts_t = th.tensor(unique_counts, dtype=bond_logits.dtype, device=device)
    smarts_ms = (perf_counter() - smarts_start) * 1000.0
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
    return MassterCandidateBatch(
        mask_log_probabilities=probabilities,
        **arrays,
        proposal_single_cut_fraction=proposal_fraction,
        unique_component_count=unique_counts_t,
        component_counts=component_counts,
        unique_component_counts=unique_counts,
        selected_mask_counts=tuple(selected_counts),
        retained_mask_counts=tuple(count.num_masks for count in view_counts),
        scanned_mask_counts=tuple(count.scanned_mask_count for count in view_counts),
        timings=MassterCandidateTimings(
            input_transfer_ms=input_ms,
            mask_selection_ms=selection_ms,
            topology_assembly_ms=topology_ms,
            forced_merge_ms=0.0,
            smarts_assembly_ms=smarts_ms,
            batch_packing_ms=packing_ms,
            device_transfer_ms=transfer_ms,
            probability_scoring_ms=probability_ms,
            output_transfer_ms=packing_ms + transfer_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        ),
    )
