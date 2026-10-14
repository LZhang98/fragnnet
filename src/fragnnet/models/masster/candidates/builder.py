"""Batch top-ranked bond masks into sparse MASSter component tensors."""

from __future__ import annotations

import atexit
import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch as th
import torch.nn.functional as F

if TYPE_CHECKING:
    from fragnnet.models.masster.candidates.smarts_templates import MassterTemplate


_CPU_POOL: ThreadPoolExecutor | None = None
_CPU_POOL_THREADS = 0
_CPU_POOL_LOCK = Lock()
_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def _reset_cpu_pool_after_fork() -> None:
    global _CPU_POOL, _CPU_POOL_THREADS, _CPU_POOL_LOCK
    # Parent threads do not survive fork; never submit to their inherited executor.
    _CPU_POOL = None
    _CPU_POOL_THREADS = 0
    _CPU_POOL_LOCK = Lock()


def _shutdown_cpu_pool() -> None:
    global _CPU_POOL, _CPU_POOL_THREADS
    with _CPU_POOL_LOCK:
        if _CPU_POOL is not None:
            _CPU_POOL.shutdown(wait=True)
            _CPU_POOL = None
            _CPU_POOL_THREADS = 0


atexit.register(_shutdown_cpu_pool)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cpu_pool_after_fork)


def _map_cpu_tasks(
    function: Callable[[_Input], _Output], tasks: Sequence[_Input], threads: int
) -> list[_Output]:
    """Run non-nested numeric tasks on one persistent executor per process.

    The coordinator lock also serializes simultaneous callers that request different
    thread counts. Workers never acquire it; failures propagate through ``map``.
    """
    global _CPU_POOL, _CPU_POOL_THREADS
    if threads == 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    with _CPU_POOL_LOCK:
        if _CPU_POOL is None or threads != _CPU_POOL_THREADS:
            if _CPU_POOL is not None:
                _CPU_POOL.shutdown(wait=True)
            _CPU_POOL = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="masster")
            _CPU_POOL_THREADS = threads
        return list(_CPU_POOL.map(function, tasks))


@dataclass(frozen=True)
class MassterStaticGraph:
    """CPU graph indices retained across Lightning's device transfer."""

    chemical_bond_index: th.Tensor
    atom_batch_indices: th.Tensor
    atom_ptr: th.Tensor
    smarts_templates: tuple[tuple[MassterTemplate, ...], ...] | None = None


@dataclass(frozen=True)
class MassterCandidateTimings:
    """Coordinator wall times in milliseconds, with concurrent workers measured once.

    ``topology_assembly_ms`` covers local dispatch/join. ``batch_packing_ms`` includes
    serial global-index remapping plus packed-buffer construction.
    """

    input_transfer_ms: float
    mask_selection_ms: float
    topology_assembly_ms: float
    forced_merge_ms: float
    smarts_assembly_ms: float
    batch_packing_ms: float
    device_transfer_ms: float
    probability_scoring_ms: float
    output_transfer_ms: float
    total_ms: float


@dataclass(frozen=True)
class MassterCandidateBatch:
    """Sparse component pathways selected from a batch of precursor graphs."""

    mask_log_probabilities: th.Tensor
    mask_batch_indices: th.Tensor
    mask_cut_counts: th.Tensor
    mechanism_mask_indices: th.Tensor
    mechanism_rule_indices: th.Tensor
    mechanism_boundary_bond_indices: th.Tensor
    boundary_mechanism_indices: th.Tensor
    component_mask_indices: th.Tensor
    component_mechanism_indices: th.Tensor
    component_batch_indices: th.Tensor
    component_cut_counts: th.Tensor
    component_formed_bond_counts: th.Tensor
    component_unique_indices: th.Tensor
    unique_representative_component_indices: th.Tensor
    atom_indices: th.Tensor
    atom_component_indices: th.Tensor
    boundary_bond_indices: th.Tensor
    boundary_component_indices: th.Tensor
    proposal_single_cut_fraction: th.Tensor
    unique_component_count: th.Tensor
    component_counts: tuple[int, ...]
    unique_component_counts: tuple[int, ...]
    selected_mask_counts: tuple[int, ...]
    retained_mask_counts: tuple[int, ...]
    scanned_mask_counts: tuple[int, ...]
    timings: MassterCandidateTimings

    @property
    def num_components(self) -> int:
        """Return the number of retained component pathways."""
        return int(self.component_mask_indices.shape[0])

    @property
    def num_mechanisms(self) -> int:
        """Return the number of ordinary and SMARTS mechanisms."""
        return int(self.mechanism_mask_indices.shape[0])

    @property
    def num_unique_components(self) -> int:
        """Return the number of distinct precursor atom subsets."""
        return int(self.unique_representative_component_indices.shape[0])


@dataclass(frozen=True)
class _TopologyCache:
    """Molecule-local component arrays shared by repeated graph views."""

    num_atoms: int
    endpoints: tuple[tuple[int, ...], tuple[int, ...]]
    mask_ptr: np.ndarray
    broken_bonds: np.ndarray
    cut_counts: np.ndarray
    component_mask_indices: np.ndarray
    component_cut_counts: np.ndarray
    atom_indices: np.ndarray
    atom_component_indices: np.ndarray
    mask_indices: np.ndarray
    broken_bond_indices: np.ndarray
    boundary_bond_indices: np.ndarray
    boundary_component_indices: np.ndarray
    num_components: int
    unique_component_count: int
    scanned_mask_count: int
    component_unique_indices: np.ndarray
    unique_representative_component_indices: np.ndarray


@dataclass(frozen=True)
class _TopologyInput:
    """Validated molecule-local inputs; workers produce only zero-based indices."""

    num_atoms: int
    endpoints: tuple[tuple[int, ...], tuple[int, ...]]
    mask_ptr: np.ndarray
    broken_bonds: np.ndarray
    forced_mask_count: int
    max_unique_components: int | None


def _build_local_topology(request: _TopologyInput) -> _TopologyCache:
    from fragnnet.models.masster.candidates.cut_mask_kernels import build_component_arrays_raw

    (
        cut_counts, component_mask_indices, component_cut_counts, atom_indices,
        atom_component_indices, mask_indices, broken_bond_indices, boundary_bond_indices,
        boundary_component_indices, num_components, unique_count, component_unique_indices,
        representatives, scanned_mask_count,
    ) = build_component_arrays_raw(
        request.num_atoms,
        np.asarray(request.endpoints, dtype=np.int64).T,
        (request.mask_ptr, request.broken_bonds),
        np.arange(len(request.endpoints[0]), dtype=np.int64),
        0, 0, 0,
        request.max_unique_components,
        request.forced_mask_count,
    )
    num_masks = cut_counts.size
    return _TopologyCache(
        num_atoms=request.num_atoms,
        endpoints=request.endpoints,
        mask_ptr=request.mask_ptr[:num_masks + 1],
        broken_bonds=request.broken_bonds[:request.mask_ptr[num_masks]],
        cut_counts=cut_counts,
        component_mask_indices=component_mask_indices,
        component_cut_counts=component_cut_counts,
        atom_indices=atom_indices,
        atom_component_indices=atom_component_indices,
        mask_indices=mask_indices,
        broken_bond_indices=broken_bond_indices,
        boundary_bond_indices=boundary_bond_indices,
        boundary_component_indices=boundary_component_indices,
        num_components=num_components,
        unique_component_count=unique_count,
        scanned_mask_count=scanned_mask_count,
        # Raw views have E*num_atoms capacity; do not retain every worker's full
        # allocation across a large batch when early stopping kept a short prefix.
        component_unique_indices=component_unique_indices.copy(),
        unique_representative_component_indices=representatives.copy(),
    )


def _offset_local_indices(values: np.ndarray, offset: int, last_view: bool) -> np.ndarray:
    """Consume private local buffers on their last view instead of copying them."""
    return np.add(values, offset, out=values if last_view else None)


@dataclass(frozen=True)
class _SmartsLookupCache:
    """Molecule-local SMARTS lookup data reusable by grouped graph views."""

    bond_set_to_mask_index: dict[tuple[int, ...], int]
    atom_set_to_unique_index: dict[tuple[int, ...], int]


def _packed_long_chunk_tensors(
    fields: Sequence[Sequence[Sequence[int] | np.ndarray]],
    device: th.device,
) -> tuple[tuple[th.Tensor, ...], float, float]:
    """Fill one packed host buffer from chunks, then transfer it once."""
    packing_start = perf_counter()
    sizes = tuple(sum(np.asarray(chunk).size for chunk in chunks) for chunks in fields)
    packed_array = np.empty(sum(sizes), dtype=np.int64)
    field_start = 0
    for chunks, field_size in zip(fields, sizes, strict=True):
        cursor = field_start
        for chunk in chunks:
            array = np.asarray(chunk, dtype=np.int64).reshape(-1)
            packed_array[cursor : cursor + array.size] = array
            cursor += array.size
        if cursor != field_start + field_size:
            raise RuntimeError("packed candidate field size changed during assembly")
        field_start = cursor
    packing_ms = (perf_counter() - packing_start) * 1000.0
    transfer_start = perf_counter()
    packed = th.from_numpy(packed_array).to(device=device)
    transfer_ms = (perf_counter() - transfer_start) * 1000.0
    return tuple(packed.split(sizes)), packing_ms, transfer_ms


def _top_mixture_bond_sets(
    logits: np.ndarray,
    mixture_log_weights: np.ndarray,
    limit: int,
    min_relative_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deduplicated CSR pool ranked by exact Bernoulli-mixture probability.

    Pools each component's exact top-k masks, deduplicates them, scores every
    survivor under the full mixture, and ranks by descending probability with
    lexicographic tie-breaks. The work runs in a compiled kernel that holds no
    Python state, so the per-molecule body executes without the GIL.

    Args:
        logits: Per-bond cut logits of shape (num_bonds, num_mixtures).
        mixture_log_weights: Log-normalized mixture weights of shape (num_mixtures,).
        limit: Maximum number of bond sets to return.
        min_relative_probability: Probability floor relative to the MAP mask; 0 disables.

    Returns:
        Tuple of ``(mask_ptr, broken_bonds)`` CSR arrays of dtype int64, where
        ``mask_ptr`` has shape (num_selected + 1,) and ``broken_bonds`` holds the
        concatenated cut bond indices. Bond sets are ordered by descending mixture
        probability, ties broken by ascending lexicographic bond order.
    """
    from fragnnet.models.masster.candidates.cut_mask_kernels import top_mixture_bond_sets_raw

    return top_mixture_bond_sets_raw(
        np.ascontiguousarray(logits, dtype=np.float64),
        np.ascontiguousarray(mixture_log_weights, dtype=np.float64),
        limit,
        0.0,
        min_relative_probability,
    )


def _select_masks_batch(
    logits_by_molecule: list[np.ndarray],
    mixture_log_weights: np.ndarray | None,
    candidate_group_values: list[int] | None,
    limit: int,
    min_relative_probability: float,
    threads: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Select ranked cut masks for every precursor, sharing grouped views.

    Repeated collision-energy views of one precursor carry the same proposal logits,
    so only the first member of each ``candidate_group`` is selected and the rest
    alias its arrays. Selection itself runs in a kernel that releases the GIL, so
    ``threads`` above one is real parallelism and never changes the result.

    Args:
        logits_by_molecule: Per-precursor cut logits, each of shape
            (num_bonds, num_mixtures).
        mixture_log_weights: Log-normalized mixture weights of shape
            (num_molecules, num_mixtures), or None for a single Bernoulli component.
        candidate_group_values: One group id per precursor, or None when no views
            are shared.
        limit: Maximum ranked masks to select per precursor.
        min_relative_probability: Floor relative to the MAP mask; 0 disables.
        threads: Worker threads for selection. 1 selects inline.

    Returns:
        One ``(mask_ptr, broken_bonds)`` CSR pair per precursor. Grouped views share
        the representative's arrays, which are never mutated downstream.

    Raises:
        ValueError: If precursors sharing a ``candidate_group`` disagree on their
            proposal logits or mixture weights.
    """
    num_molecules = len(logits_by_molecule)

    # 1. Pick one representative per group and check the rest agree with it. The
    #    grouping contract is a precondition, so it is validated once here rather
    #    than re-checked on every cache hit during assembly.
    representatives: list[int] = []
    representative_of: list[int] = [0] * num_molecules
    if candidate_group_values is None:
        representatives = list(range(num_molecules))
        representative_of = representatives
    else:
        first_of_group: dict[int, int] = {}
        for molecule_index in range(num_molecules):
            group = candidate_group_values[molecule_index]
            leader = first_of_group.get(group)
            if leader is None:
                first_of_group[group] = molecule_index
                representatives.append(molecule_index)
                representative_of[molecule_index] = molecule_index
                continue
            if not np.array_equal(logits_by_molecule[leader], logits_by_molecule[molecule_index]):
                raise ValueError("candidate_group_ids can only group identical proposal logits")
            if mixture_log_weights is not None and not np.array_equal(
                mixture_log_weights[leader], mixture_log_weights[molecule_index]
            ):
                raise ValueError("candidate_group_ids can only group identical mixture weights")
            representative_of[molecule_index] = leader

    # 2. A single Bernoulli component is the degenerate one-component mixture, so
    #    both cases go through one kernel instead of two parallel code paths.
    single_weight = np.zeros(1, dtype=np.float64)

    def select(molecule_index: int) -> tuple[np.ndarray, np.ndarray]:
        weights = (
            single_weight if mixture_log_weights is None else mixture_log_weights[molecule_index]
        )
        return _top_mixture_bond_sets(
            logits_by_molecule[molecule_index],
            weights,
            limit,
            min_relative_probability,
        )

    selected = _map_cpu_tasks(select, representatives, threads)

    by_representative = dict(zip(representatives, selected))
    return [by_representative[representative_of[i]] for i in range(num_molecules)]


def _merge_forced_masks(
    local_mask_ptr: np.ndarray,
    local_broken_bonds: np.ndarray,
    forced_bond_sets: list[tuple[int, ...]],
    learned_cut_counts: np.ndarray,
    learned_mask_count: int,
    base_forced_mask_count: int,
    unique_fragment_mode: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge the always-required cut masks into a precursor's ranked selection.

    Two policies, differing only in where the forced masks land:

    - unique-fragment mode puts them first and drops their ranked duplicates, so a
      prefix can be truncated at the unique-fragment target without ever losing the
      intact or single-cut support;
    - otherwise they are appended, and only those absent from the ranked selection.

    Args:
        local_mask_ptr: CSR pointer over the ranked masks.
        local_broken_bonds: Concatenated cut bonds for the ranked masks.
        forced_bond_sets: Required bond sets, deduplicated, in their historical order.
        learned_cut_counts: Cut count per ranked mask.
        learned_mask_count: Number of ranked masks.
        base_forced_mask_count: Number of forced masks contributed by the empty
            mask and optional complete single-cut set, before SMARTS masks.
        unique_fragment_mode: Whether unique-fragment enumeration is active.

    Returns:
        Tuple of ``(mask_ptr, broken_bonds, forced_mask_count)``, where
        ``forced_mask_count`` is the number of forced masks placed at the front and is
        zero outside unique-fragment mode.
    """
    forced_mask_count = 0
    if unique_fragment_mode:
        # Put the required low-cut masks first and remove their ranked
        # duplicates. A prefix can then be stopped at the unique-fragment
        # target without ever dropping the intact or single-cut support.
        forced_cut_counts = np.asarray(
            [len(bond_set) for bond_set in forced_bond_sets],
            dtype=np.int64,
        )
        forced_bonds = np.asarray(
            [bond for bond_set in forced_bond_sets for bond in bond_set],
            dtype=np.int64,
        )
        keep_ranked_masks = np.ones(learned_mask_count, dtype=np.bool_)
        keep_ranked_masks[learned_cut_counts == 0] = False
        if base_forced_mask_count > 1:
            # The base prefix contains every single cut, so cardinality alone
            # identifies all of its ranked duplicates.
            keep_ranked_masks[learned_cut_counts == 1] = False

        template_forced_by_count: dict[int, set[tuple[int, ...]]] = {}
        for bond_set in forced_bond_sets[base_forced_mask_count:]:
            template_forced_by_count.setdefault(len(bond_set), set()).add(bond_set)
        if template_forced_by_count:
            candidate_indices = np.flatnonzero(
                np.isin(
                    learned_cut_counts,
                    np.fromiter(template_forced_by_count, dtype=np.int64),
                )
            )
            for index in candidate_indices:
                bond_set = tuple(
                    local_broken_bonds[local_mask_ptr[index] : local_mask_ptr[index + 1]]
                )
                if bond_set in template_forced_by_count[len(bond_set)]:
                    keep_ranked_masks[index] = False
        keep_ranked_bonds = np.repeat(keep_ranked_masks, learned_cut_counts)
        retained_ranked_cut_counts = learned_cut_counts[keep_ranked_masks]
        retained_ranked_bonds = local_broken_bonds[keep_ranked_bonds]
        combined_cut_counts = np.concatenate((forced_cut_counts, retained_ranked_cut_counts))
        local_mask_ptr = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(combined_cut_counts, dtype=np.int64),
            )
        )
        local_broken_bonds = np.concatenate((forced_bonds, retained_ranked_bonds))
        forced_mask_count = forced_cut_counts.size
    else:
        extra_cut_counts: list[int] = []
        extra_bonds: list[int] = []
        missing_bond_sets = set(forced_bond_sets)
        forced_cut_counts = np.fromiter(
            {len(bond_set) for bond_set in forced_bond_sets},
            dtype=np.int64,
        )
        candidate_indices = np.flatnonzero(np.isin(learned_cut_counts, forced_cut_counts))
        for index in candidate_indices:
            missing_bond_sets.discard(
                tuple(local_broken_bonds[local_mask_ptr[index] : local_mask_ptr[index + 1]])
            )
            if not missing_bond_sets:
                break
        for bond_set in forced_bond_sets:
            if bond_set not in missing_bond_sets:
                continue
            extra_cut_counts.append(len(bond_set))
            extra_bonds.extend(bond_set)
        if extra_cut_counts:
            local_mask_ptr = np.concatenate(
                (
                    local_mask_ptr,
                    local_mask_ptr[-1] + np.cumsum(extra_cut_counts, dtype=np.int64),
                )
            )
            if extra_bonds:
                local_broken_bonds = np.concatenate(
                    (local_broken_bonds, np.asarray(extra_bonds, dtype=np.int64))
                )

    return local_mask_ptr, local_broken_bonds, forced_mask_count


def _score_mask_log_probabilities(
    bond_logits: th.Tensor,
    bond_batch_indices: th.Tensor,
    mask_batch_indices_t: th.Tensor,
    mask_indices_t: th.Tensor,
    broken_bond_indices_t: th.Tensor,
    num_molecules: int,
    mixture_log_weights: th.Tensor | None,
) -> th.Tensor:
    """Recompute exact selected-mask probabilities from live logits for autograd."""
    num_mixtures = 1 if bond_logits.ndim == 1 else bond_logits.shape[1]
    device = bond_logits.device
    base_log_probabilities = F.logsigmoid(-bond_logits)
    probability_logits = bond_logits.to(base_log_probabilities)
    molecule_log_probabilities = th.zeros(
        (num_molecules,) if num_mixtures == 1 else (num_molecules, num_mixtures),
        dtype=base_log_probabilities.dtype,
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
            probability_logits[broken_bond_indices_t],
        )
    if num_mixtures > 1:
        if mixture_log_weights is None:
            raise RuntimeError("mixture weights disappeared during probability evaluation")
        mask_log_probabilities = th.logsumexp(
            mask_log_probabilities + mixture_log_weights[mask_batch_indices_t],
            dim=1,
        )
    return mask_log_probabilities


def build_masster_candidates(
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
    candidate_backend: str = "cpu",
    _cuda_full_stats: list[dict] | None = None,
) -> MassterCandidateBatch:
    """Select masks on CPU and return sparse pathway tensors on the model device.

    Mask identities are deliberately selected from detached logits. Their exact
    Bernoulli or Bernoulli-mixture log probabilities are then recomputed from the
    original logits, preserving gradients to every selected cut/intact decision and
    latent mixture weight. When ``candidate_group_ids`` is provided, repeated
    collision-energy views with the same group id reuse exact ranked masks and
    connected-component topology while retaining separate differentiable
    probabilities and output candidates. ``mask_selection_threads`` above one
    runs selection and local topology/assembly on the same persistent CPU pool.
    Numeric kernels release the GIL; ordering and selected masks stay unchanged.
    ``candidate_backend='cuda'`` selects exact device construction for finite FP32,
    one-mixture, unique-mode inputs with no mask floor; SMARTS templates are supported and
    give the CPU backend's fields exactly. CUDA compilation is lazy and optional;
    unsupported requests raise instead of falling back.
    When ``max_unique_components`` is set, deterministic enumeration continues up to
    ``max_enumerated_masks`` and keeps the shortest prefix that reaches the requested
    number of distinct atom sets. The probability floor applies only to learned ranked
    masks; required intact and single-cut masks remain forced.
    """
    if candidate_backend not in ("cpu", "cuda"):
        raise ValueError(f"unknown candidate_backend: {candidate_backend!r}; use cpu or cuda")
    if candidate_backend == "cuda":
        from fragnnet.models.masster.candidates.cuda_candidates import (
            build_masster_candidates_cuda,
        )

        return build_masster_candidates_cuda(
            bond_logits,
            chemical_bond_index,
            atom_batch_indices,
            atom_ptr,
            max_masks=max_masks,
            include_single_bond_cuts=include_single_bond_cuts,
            min_mask_relative_probability=min_mask_relative_probability,
            max_unique_components=max_unique_components,
            max_enumerated_masks=max_enumerated_masks,
            mask_selection_threads=mask_selection_threads,
            static_graph=static_graph,
            smarts_templates=smarts_templates,
            mixture_log_weights=mixture_log_weights,
            candidate_group_ids=candidate_group_ids,
            _statistics=_cuda_full_stats,
        )
    if _cuda_full_stats is not None:
        raise ValueError("full CUDA statistics require candidate_backend='cuda'")
    return _build_cpu_candidates(
        bond_logits,
        chemical_bond_index,
        atom_batch_indices,
        atom_ptr,
        max_masks=max_masks,
        include_single_bond_cuts=include_single_bond_cuts,
        min_mask_relative_probability=min_mask_relative_probability,
        max_unique_components=max_unique_components,
        max_enumerated_masks=max_enumerated_masks,
        mask_selection_threads=mask_selection_threads,
        static_graph=static_graph,
        smarts_templates=smarts_templates,
        mixture_log_weights=mixture_log_weights,
        candidate_group_ids=candidate_group_ids,
    )


def _build_cpu_candidates(
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
    topology_builder: Callable[..., MassterCandidateBatch] | None = None,
) -> MassterCandidateBatch:
    """CPU candidate construction; ``topology_builder`` is the benchmark-only GPU topology hook."""
    if bond_logits.ndim not in {1, 2}:
        raise ValueError("bond_logits must have shape (num_bonds,) or (num_bonds, mixtures)")
    if bond_logits.ndim == 2 and bond_logits.shape[1] < 2:
        raise ValueError("two-dimensional bond_logits must contain at least two mixtures")
    num_mixtures = 1 if bond_logits.ndim == 1 else bond_logits.shape[1]
    if num_mixtures <= 0:
        raise ValueError("bond_logits must contain at least one mixture")
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
    if mask_selection_threads < 1:
        raise ValueError(f"mask_selection_threads must be at least 1, got {mask_selection_threads}")
    if max_unique_components is not None and max_unique_components <= 0:
        raise ValueError(
            f"max_unique_components must be positive when enabled, got {max_unique_components}"
        )
    if max_enumerated_masks is not None and max_enumerated_masks < max_masks:
        raise ValueError(
            "max_enumerated_masks must be at least max_masks, "
            f"got {max_enumerated_masks} and {max_masks}"
        )
    if max_unique_components is not None and max_enumerated_masks is None:
        raise ValueError("max_enumerated_masks is required when max_unique_components is enabled")
    if (
        not math.isfinite(min_mask_relative_probability)
        or not 0.0 <= min_mask_relative_probability <= 1.0
    ):
        raise ValueError(
            "min_mask_relative_probability must be finite and in [0, 1], "
            f"got {min_mask_relative_probability}"
        )
    if num_mixtures == 1 and mixture_log_weights is not None:
        raise ValueError("mixture_log_weights must be omitted for one Bernoulli component")
    if num_mixtures > 1 and mixture_log_weights is None:
        raise ValueError("mixture_log_weights are required for multiple Bernoulli components")

    total_start = perf_counter()
    device = bond_logits.device
    num_bonds = bond_logits.shape[0]

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
    num_molecules = atom_ptr_cpu.numel() - 1
    if candidate_group_ids is None:
        candidate_group_values = None
    else:
        if candidate_group_ids.shape != (num_molecules,) or candidate_group_ids.dtype != th.long:
            raise ValueError(
                "candidate_group_ids must be a torch.long vector with one id per molecule"
            )
        candidate_group_values = candidate_group_ids.detach().cpu().tolist()
    input_transfer_ms = (perf_counter() - input_transfer_start) * 1000.0
    if mixture_log_weights is not None:
        if mixture_log_weights.shape != (num_molecules, num_mixtures):
            raise ValueError("mixture_log_weights must have shape (num_molecules, num_mixtures)")
        mixture_log_weights_cpu = mixture_log_weights.detach().to(
            device="cpu",
            dtype=th.float64,
        )
        mixture_log_normalizers = np.logaddexp.reduce(
            mixture_log_weights_cpu.numpy(),
            axis=1,
        )
        if not np.isfinite(mixture_log_weights_cpu.numpy()).all() or not np.allclose(
            mixture_log_normalizers,
            0.0,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("mixture_log_weights must be finite and log-normalized")
    else:
        mixture_log_weights_cpu = None
    if smarts_templates is None and static_graph is not None:
        smarts_templates = static_graph.smarts_templates
    if smarts_templates is not None and len(smarts_templates) != num_molecules:
        raise ValueError("smarts_templates must contain one sequence per precursor")
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
    mechanism_mask_index_chunks: list[np.ndarray] = []
    mechanism_rule_index_chunks: list[np.ndarray] = []
    mechanism_boundary_bond_index_chunks: list[np.ndarray] = []
    boundary_mechanism_index_chunks: list[np.ndarray] = []
    component_mask_index_chunks: list[np.ndarray] = []
    component_mechanism_index_chunks: list[np.ndarray] = []
    component_batch_index_chunks: list[np.ndarray] = []
    component_cut_count_chunks: list[np.ndarray] = []
    component_formed_bond_count_chunks: list[np.ndarray] = []
    component_unique_index_chunks: list[np.ndarray] = []
    unique_representative_component_index_chunks: list[np.ndarray] = []
    atom_index_chunks: list[np.ndarray] = []
    atom_component_index_chunks: list[np.ndarray] = []
    boundary_bond_index_chunks: list[np.ndarray] = []
    boundary_component_index_chunks: list[np.ndarray] = []
    proposal_single_cut_fractions: list[float] = []
    component_counts: list[int] = []
    unique_component_counts: list[int] = []
    selected_mask_counts: list[int] = []
    retained_mask_counts: list[int] = []
    scanned_mask_counts: list[int] = []
    mask_selection_ms = 0.0
    remap_ms = 0.0
    forced_merge_ms = 0.0
    smarts_assembly_ms = 0.0
    num_masks_so_far = 0
    num_mechanisms_so_far = 0
    num_components_so_far = 0
    num_unique_components_so_far = 0
    smarts_lookup_cache: dict[int, _SmartsLookupCache] = {}
    forced_merge_cache: dict[
        tuple[int, tuple[tuple[int, ...], ...]],
        tuple[np.ndarray, np.ndarray, int],
    ] = {}

    # 2. Select every precursor's ranked masks up front. Selection depends only on
    # the proposal logits, so hoisting it out of the assembly loop is what lets
    # grouped views share one selection and lets the pool run threads across
    # precursors.
    learned_mask_limit = max_enumerated_masks if max_unique_components is not None else max_masks
    logits_by_molecule = [
        np.asarray(
            [detached_logits_all[index] for index in bond_indices_by_molecule[molecule_index]],
            dtype=np.float64,
        ).reshape(-1, num_mixtures)
        for molecule_index in range(num_molecules)
    ]
    selection_start = perf_counter()
    selected_masks = _select_masks_batch(
        logits_by_molecule,
        None if mixture_log_weights_cpu is None else mixture_log_weights_cpu.numpy(),
        candidate_group_values,
        learned_mask_limit,
        min_mask_relative_probability,
        mask_selection_threads,
    )
    mask_selection_ms += (perf_counter() - selection_start) * 1000.0

    # 3. Merge forced masks before dispatching molecule-local topology workers.
    topology_inputs: list[_TopologyInput] = []
    topology_representatives: list[int] = []
    representative_of: list[int] = []
    first_of_group: dict[int, int] = {}
    for molecule_index in range(num_molecules):
        atom_start = atom_ptr_values[molecule_index]
        atom_end = atom_ptr_values[molecule_index + 1]
        num_atoms = atom_end - atom_start
        if num_atoms <= 0:
            raise ValueError("every precursor must contain at least one atom")
        candidate_group = (
            None if candidate_group_values is None else int(candidate_group_values[molecule_index])
        )

        global_bond_indices = bond_indices_by_molecule[molecule_index]
        local_endpoints = tuple(
            tuple(chemical_bond_rows[row][index] - atom_start for index in global_bond_indices)
            for row in range(2)
        )
        molecule_templates = (
            tuple(smarts_templates[molecule_index]) if smarts_templates is not None else ()
        )
        template_bond_sets = [
            tuple(template.broken_bond_indices) for template in molecule_templates
        ]
        forced_bond_sets = [()]
        # NOTE: single-bond cuts are forced only for a single Bernoulli component.
        # Mixtures have never forced them -- the pre-refactor code derived the bond
        # count from a list it left empty when num_mixtures > 1 -- and no shipped
        # config combines the two, so the behavior is preserved rather than changed
        # here. See `git show e65339ed:docs/plans/masster/masster_candidate_pathway_refactor.md`.
        if include_single_bond_cuts and num_mixtures == 1:
            forced_bond_sets.extend((bond_index,) for bond_index in range(len(global_bond_indices)))
        base_forced_mask_count = len(forced_bond_sets)
        forced_bond_sets.extend(template_bond_sets)
        forced_bond_sets = list(dict.fromkeys(forced_bond_sets))
        local_mask_ptr, local_broken_bonds = selected_masks[molecule_index]
        learned_mask_count = local_mask_ptr.size - 1
        learned_cut_counts = np.diff(local_mask_ptr)
        proposal_single_cut_fractions.append(
            float(np.count_nonzero(learned_cut_counts == 1)) / learned_mask_count
        )

        forced_merge_start = perf_counter()
        forced_merge_key = (
            None
            if candidate_group is None
            else (candidate_group, tuple(forced_bond_sets))
        )
        merged_masks = (
            None
            if forced_merge_key is None
            else forced_merge_cache.get(forced_merge_key)
        )
        if merged_masks is None:
            merged_masks = _merge_forced_masks(
                local_mask_ptr,
                local_broken_bonds,
                forced_bond_sets,
                learned_cut_counts,
                learned_mask_count,
                base_forced_mask_count,
                max_unique_components is not None,
            )
            if forced_merge_key is not None:
                forced_merge_cache[forced_merge_key] = merged_masks
        local_mask_ptr, local_broken_bonds, forced_mask_count = merged_masks
        forced_merge_ms += (perf_counter() - forced_merge_start) * 1000.0
        selected_mask_counts.append(local_mask_ptr.size - 1)
        topology_inputs.append(_TopologyInput(
            num_atoms=num_atoms,
            endpoints=local_endpoints,
            mask_ptr=local_mask_ptr,
            broken_bonds=local_broken_bonds,
            forced_mask_count=forced_mask_count,
            max_unique_components=max_unique_components,
        ))
        leader = (
            molecule_index if candidate_group is None
            else first_of_group.setdefault(candidate_group, molecule_index)
        )
        representative_of.append(leader)
        if leader == molecule_index:
            topology_representatives.append(leader)

    if topology_builder is not None:
        return topology_builder(
            bond_logits=bond_logits,
            topology_inputs=topology_inputs,
            topology_representatives=topology_representatives,
            representative_of=representative_of,
            atom_ptr_values=atom_ptr_values,
            bond_indices_by_molecule=bond_indices_by_molecule,
            bond_batch_values=bond_batch_values,
            proposal_single_cut_fractions=proposal_single_cut_fractions,
            selected_mask_counts=selected_mask_counts,
            total_start=total_start,
            input_transfer_ms=input_transfer_ms,
            mask_selection_ms=mask_selection_ms,
            forced_merge_ms=forced_merge_ms,
            smarts_templates=smarts_templates,
        )

    # One coordinator interval measures concurrent construction, never a sum of workers.
    topology_start = perf_counter()
    local_topologies = _map_cpu_tasks(
        _build_local_topology,
        [topology_inputs[index] for index in topology_representatives],
        mask_selection_threads,
    )
    topology_assembly_ms = (perf_counter() - topology_start) * 1000.0
    topology_by_representative = dict(zip(topology_representatives, local_topologies, strict=True))
    local_topologies.clear()
    last_view_of = {leader: index for index, leader in enumerate(representative_of)}

    # 4. Apply global offsets in input order, then append any SMARTS mechanisms.
    for molecule_index, request in enumerate(topology_inputs):
        atom_start = atom_ptr_values[molecule_index]
        num_atoms = request.num_atoms
        local_endpoints = request.endpoints
        global_bond_indices = bond_indices_by_molecule[molecule_index]
        local_mask_ptr, local_broken_bonds = request.mask_ptr, request.broken_bonds
        molecule_templates = (
            tuple(smarts_templates[molecule_index]) if smarts_templates is not None else ()
        )
        candidate_group = (
            None if candidate_group_values is None else candidate_group_values[molecule_index]
        )
        mask_start = num_masks_so_far
        component_start = num_components_so_far
        leader = representative_of[molecule_index]
        cached_topology = topology_by_representative[leader]
        if leader != molecule_index:
            retained_ptr_size = cached_topology.mask_ptr.size
            retained_broken_size = cached_topology.broken_bonds.size
            if (
                cached_topology.num_atoms != num_atoms
                or cached_topology.endpoints != local_endpoints
                or local_mask_ptr.size < retained_ptr_size
                or not np.array_equal(cached_topology.mask_ptr, local_mask_ptr[:retained_ptr_size])
                or local_broken_bonds.size < retained_broken_size
                or not np.array_equal(
                    cached_topology.broken_bonds, local_broken_bonds[:retained_broken_size]
                )
            ):
                raise ValueError("candidate_group_ids can only group identical molecular topology")
        remap_start = perf_counter()
        last_view = last_view_of[leader] == molecule_index
        global_bond_array = np.asarray(global_bond_indices, dtype=np.int64)
        local_cut_counts = cached_topology.cut_counts
        local_component_mask_indices = _offset_local_indices(
            cached_topology.component_mask_indices, mask_start, last_view
        )
        local_component_cut_counts = cached_topology.component_cut_counts
        local_component_formed_bond_counts = np.zeros(
            cached_topology.num_components, dtype=np.int64
        )
        local_atom_indices = _offset_local_indices(cached_topology.atom_indices, atom_start, last_view)
        local_atom_component_indices = _offset_local_indices(
            cached_topology.atom_component_indices, component_start, last_view
        )
        local_mask_indices = _offset_local_indices(
            cached_topology.mask_indices, mask_start, last_view
        )
        # Bond lists are strictly increasing; the common contiguous case is a shift.
        if global_bond_indices and (
            global_bond_indices[-1] - global_bond_indices[0] + 1 == len(global_bond_indices)
        ):
            local_broken_bond_indices = _offset_local_indices(
                cached_topology.broken_bond_indices, global_bond_indices[0], last_view
            )
            local_boundary_bond_indices = _offset_local_indices(
                cached_topology.boundary_bond_indices, global_bond_indices[0], last_view
            )
        else:
            local_broken_bond_indices = global_bond_array[cached_topology.broken_bond_indices]
            local_boundary_bond_indices = global_bond_array[cached_topology.boundary_bond_indices]
        local_boundary_component_indices = _offset_local_indices(
            cached_topology.boundary_component_indices, component_start, last_view
        )
        num_local_components = cached_topology.num_components
        unique_component_count = cached_topology.unique_component_count
        local_scanned_mask_count = cached_topology.scanned_mask_count
        local_component_unique_indices = cached_topology.component_unique_indices
        local_unique_representative_component_indices = (
            cached_topology.unique_representative_component_indices
        )
        if last_view:
            del topology_by_representative[leader]
        remap_ms += (perf_counter() - remap_start) * 1000.0
        num_local_masks = local_cut_counts.size
        retained_mask_counts.append(num_local_masks)
        scanned_mask_counts.append(local_scanned_mask_count)

        # Each cut mask always has one ordinary connected-component mechanism.
        # SMARTS matches add alternative atom partitions under that same mask;
        # they never create a second copy of the mask probability.
        mechanism_start = num_mechanisms_so_far
        local_mechanism_mask_indices = np.arange(
            mask_start,
            mask_start + num_local_masks,
            dtype=np.int64,
        )
        local_mechanism_rule_indices = np.zeros(num_local_masks, dtype=np.int64)
        local_component_mechanism_indices = (
            local_component_mask_indices - mask_start + mechanism_start
        )
        local_mechanism_boundary_bond_indices = local_broken_bond_indices.copy()
        local_boundary_mechanism_indices = local_mask_indices - mask_start + mechanism_start

        smarts_start = perf_counter()
        if molecule_templates:
            from fragnnet.models.masster.candidates.smarts_templates import formed_bond_count

            smarts_lookup = (
                None
                if candidate_group is None
                else smarts_lookup_cache.get(candidate_group)
            )
            if smarts_lookup is None:
                bond_set_to_mask_index: dict[tuple[int, ...], int] = {}
                for local_index in range(num_local_masks):
                    bond_start = local_mask_ptr[local_index]
                    bond_end = local_mask_ptr[local_index + 1]
                    bond_set = tuple(
                        int(bond)
                        for bond in local_broken_bonds[bond_start:bond_end]
                    )
                    bond_set_to_mask_index[bond_set] = local_index

                representative_components = (
                    local_unique_representative_component_indices + component_start
                )
                membership_starts = np.searchsorted(
                    local_atom_component_indices,
                    representative_components,
                    side="left",
                )
                membership_ends = np.searchsorted(
                    local_atom_component_indices,
                    representative_components,
                    side="right",
                )
                atom_set_to_unique_index = {
                    tuple(
                        int(atom) - atom_start
                        for atom in local_atom_indices[start:end]
                    ): unique_index
                    for unique_index, (start, end) in enumerate(
                        zip(membership_starts, membership_ends, strict=True)
                    )
                }
                smarts_lookup = _SmartsLookupCache(
                    bond_set_to_mask_index=bond_set_to_mask_index,
                    atom_set_to_unique_index=atom_set_to_unique_index,
                )
                if candidate_group is not None:
                    smarts_lookup_cache[candidate_group] = smarts_lookup
            atom_set_to_unique = smarts_lookup.atom_set_to_unique_index.copy()

            extra_mechanism_masks: list[int] = []
            extra_mechanism_rules: list[int] = []
            extra_mechanism_boundary_bonds: list[int] = []
            extra_boundary_mechanisms: list[int] = []
            extra_component_masks: list[int] = []
            extra_component_mechanisms: list[int] = []
            extra_component_cut_counts: list[int] = []
            extra_component_formed_bond_counts: list[int] = []
            extra_component_unique_indices: list[int] = []
            extra_atom_indices: list[int] = []
            extra_atom_component_indices: list[int] = []
            extra_boundary_bonds: list[int] = []
            extra_boundary_components: list[int] = []
            extra_representatives: list[int] = []

            for template in molecule_templates:
                local_mask_index = smarts_lookup.bond_set_to_mask_index.get(
                    tuple(template.broken_bond_indices)
                )
                if local_mask_index is None:
                    continue
                mask_index = mask_start + local_mask_index
                mechanism_index = mechanism_start + num_local_masks + len(extra_mechanism_masks)
                extra_mechanism_masks.append(mask_index)
                extra_mechanism_rules.append(template.rule_index)
                for local_bond_index in template.broken_bond_indices:
                    extra_mechanism_boundary_bonds.append(global_bond_indices[local_bond_index])
                    extra_boundary_mechanisms.append(mechanism_index)

                for component in template.components:
                    component_index = (
                        component_start + num_local_components + len(extra_component_masks)
                    )
                    local_atoms = tuple(component)
                    global_atoms = tuple(atom_start + atom for atom in local_atoms)
                    unique_index = atom_set_to_unique.get(local_atoms)
                    if unique_index is None:
                        unique_index = unique_component_count
                        unique_component_count += 1
                        atom_set_to_unique[local_atoms] = unique_index
                        # Representatives are molecule-local; component_start is
                        # added once when the chunk is appended.
                        extra_representatives.append(component_index - component_start)
                    extra_component_masks.append(mask_index)
                    extra_component_mechanisms.append(mechanism_index)
                    extra_component_cut_counts.append(len(template.broken_bond_indices))
                    extra_component_formed_bond_counts.append(
                        formed_bond_count(local_atoms, local_endpoints)
                    )
                    extra_component_unique_indices.append(unique_index)
                    extra_atom_indices.extend(global_atoms)
                    extra_atom_component_indices.extend([component_index] * len(global_atoms))
                    for local_bond_index in template.broken_bond_indices:
                        extra_boundary_bonds.append(global_bond_indices[local_bond_index])
                        extra_boundary_components.append(component_index)

            if extra_mechanism_masks:
                local_mechanism_mask_indices = np.concatenate(
                    (local_mechanism_mask_indices, np.asarray(extra_mechanism_masks))
                )
                local_mechanism_rule_indices = np.concatenate(
                    (local_mechanism_rule_indices, np.asarray(extra_mechanism_rules))
                )
                local_mechanism_boundary_bond_indices = np.concatenate(
                    (
                        local_mechanism_boundary_bond_indices,
                        np.asarray(extra_mechanism_boundary_bonds, dtype=np.int64),
                    )
                )
                local_boundary_mechanism_indices = np.concatenate(
                    (
                        local_boundary_mechanism_indices,
                        np.asarray(extra_boundary_mechanisms, dtype=np.int64),
                    )
                )
                local_component_mask_indices = np.concatenate(
                    (local_component_mask_indices, np.asarray(extra_component_masks))
                )
                local_component_mechanism_indices = np.concatenate(
                    (
                        local_component_mechanism_indices,
                        np.asarray(extra_component_mechanisms),
                    )
                )
                local_component_cut_counts = np.concatenate(
                    (
                        local_component_cut_counts,
                        np.asarray(extra_component_cut_counts),
                    )
                )
                local_component_formed_bond_counts = np.concatenate(
                    (
                        local_component_formed_bond_counts,
                        np.asarray(extra_component_formed_bond_counts, dtype=np.int64),
                    )
                )
                local_component_unique_indices = np.concatenate(
                    (
                        local_component_unique_indices,
                        np.asarray(extra_component_unique_indices),
                    )
                )
                local_atom_indices = np.concatenate(
                    (local_atom_indices, np.asarray(extra_atom_indices))
                )
                local_atom_component_indices = np.concatenate(
                    (
                        local_atom_component_indices,
                        np.asarray(extra_atom_component_indices),
                    )
                )
                local_boundary_bond_indices = np.concatenate(
                    (local_boundary_bond_indices, np.asarray(extra_boundary_bonds))
                )
                local_boundary_component_indices = np.concatenate(
                    (
                        local_boundary_component_indices,
                        np.asarray(extra_boundary_components),
                    )
                )
                local_unique_representative_component_indices = np.concatenate(
                    (
                        local_unique_representative_component_indices,
                        np.asarray(extra_representatives, dtype=np.int64),
                    )
                )
                num_local_components += len(extra_component_masks)
        smarts_assembly_ms += (perf_counter() - smarts_start) * 1000.0

        mask_batch_index_chunks.append(np.full(num_local_masks, molecule_index, dtype=np.int64))
        mask_cut_count_chunks.append(local_cut_counts)
        mechanism_mask_index_chunks.append(local_mechanism_mask_indices)
        mechanism_rule_index_chunks.append(local_mechanism_rule_indices)
        if local_mechanism_boundary_bond_indices.size > 0:
            mechanism_boundary_bond_index_chunks.append(local_mechanism_boundary_bond_indices)
            boundary_mechanism_index_chunks.append(local_boundary_mechanism_indices)
        component_mask_index_chunks.append(local_component_mask_indices)
        component_mechanism_index_chunks.append(local_component_mechanism_indices)
        component_batch_index_chunks.append(
            np.full(num_local_components, molecule_index, dtype=np.int64)
        )
        component_cut_count_chunks.append(local_component_cut_counts)
        component_formed_bond_count_chunks.append(local_component_formed_bond_counts)
        component_unique_index_chunks.append(
            local_component_unique_indices + num_unique_components_so_far
        )
        unique_representative_component_index_chunks.append(
            local_unique_representative_component_indices + component_start
        )
        atom_index_chunks.append(local_atom_indices)
        atom_component_index_chunks.append(local_atom_component_indices)
        mask_index_chunks.append(local_mask_indices)
        broken_bond_index_chunks.append(local_broken_bond_indices)
        if local_boundary_bond_indices.size > 0:
            boundary_bond_index_chunks.append(local_boundary_bond_indices)
            boundary_component_index_chunks.append(local_boundary_component_indices)
        num_masks_so_far += num_local_masks
        num_mechanisms_so_far += local_mechanism_mask_indices.size
        num_components_so_far += num_local_components
        num_unique_components_so_far += unique_component_count
        component_counts.append(num_local_components)
        unique_component_counts.append(unique_component_count)

    if num_masks_so_far == 0:
        raise ValueError("candidate generation produced no cut masks")

    # 5. Fill one packed CPU buffer directly from molecule-local chunks. This
    # avoids per-field concatenations followed by a second cross-field copy.
    packed_tensors, batch_packing_ms, device_transfer_ms = _packed_long_chunk_tensors(
        (
            (bond_batch_values,),
            mask_batch_index_chunks,
            mask_cut_count_chunks,
            mechanism_mask_index_chunks,
            mechanism_rule_index_chunks,
            mechanism_boundary_bond_index_chunks,
            boundary_mechanism_index_chunks,
            component_mask_index_chunks,
            component_mechanism_index_chunks,
            component_batch_index_chunks,
            component_cut_count_chunks,
            component_formed_bond_count_chunks,
            component_unique_index_chunks,
            unique_representative_component_index_chunks,
            atom_index_chunks,
            atom_component_index_chunks,
            boundary_bond_index_chunks,
            boundary_component_index_chunks,
            mask_index_chunks,
            broken_bond_index_chunks,
        ),
        device,
    )
    (
        bond_batch_indices,
        mask_batch_indices_t,
        mask_cut_counts_t,
        mechanism_mask_indices_t,
        mechanism_rule_indices_t,
        mechanism_boundary_bond_indices_t,
        boundary_mechanism_indices_t,
        component_mask_indices_t,
        component_mechanism_indices_t,
        component_batch_indices_t,
        component_cut_counts_t,
        component_formed_bond_counts_t,
        component_unique_indices_t,
        unique_representative_component_indices_t,
        atom_indices_t,
        atom_component_indices_t,
        boundary_bond_indices_t,
        boundary_component_indices_t,
        mask_indices_t,
        broken_bond_indices_t,
    ) = packed_tensors
    batch_packing_ms += remap_ms
    output_transfer_ms = batch_packing_ms + device_transfer_ms

    # 6. Score every selected mask in one batched operation. A broken bond
    # changes its contribution from logsigmoid(-logit) to logsigmoid(logit),
    # whose difference is exactly logit.
    probability_start = perf_counter()
    mask_log_probabilities = _score_mask_log_probabilities(
        bond_logits, bond_batch_indices, mask_batch_indices_t, mask_indices_t,
        broken_bond_indices_t, num_molecules, mixture_log_weights,
    )
    probability_scoring_ms = (perf_counter() - probability_start) * 1000.0

    return MassterCandidateBatch(
        mask_log_probabilities=mask_log_probabilities,
        mask_batch_indices=mask_batch_indices_t,
        mask_cut_counts=mask_cut_counts_t,
        mechanism_mask_indices=mechanism_mask_indices_t,
        mechanism_rule_indices=mechanism_rule_indices_t,
        mechanism_boundary_bond_indices=mechanism_boundary_bond_indices_t,
        boundary_mechanism_indices=boundary_mechanism_indices_t,
        component_mask_indices=component_mask_indices_t,
        component_mechanism_indices=component_mechanism_indices_t,
        component_batch_indices=component_batch_indices_t,
        component_cut_counts=component_cut_counts_t,
        component_formed_bond_counts=component_formed_bond_counts_t,
        component_unique_indices=component_unique_indices_t,
        unique_representative_component_indices=(unique_representative_component_indices_t),
        atom_indices=atom_indices_t,
        atom_component_indices=atom_component_indices_t,
        boundary_bond_indices=boundary_bond_indices_t,
        boundary_component_indices=boundary_component_indices_t,
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
        component_counts=tuple(component_counts),
        unique_component_counts=tuple(unique_component_counts),
        selected_mask_counts=tuple(selected_mask_counts),
        retained_mask_counts=tuple(retained_mask_counts),
        scanned_mask_counts=tuple(scanned_mask_counts),
        timings=MassterCandidateTimings(
            input_transfer_ms=input_transfer_ms,
            mask_selection_ms=mask_selection_ms,
            topology_assembly_ms=topology_assembly_ms,
            forced_merge_ms=forced_merge_ms,
            smarts_assembly_ms=smarts_assembly_ms,
            batch_packing_ms=batch_packing_ms,
            device_transfer_ms=device_transfer_ms,
            probability_scoring_ms=probability_scoring_ms,
            output_transfer_ms=output_transfer_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        ),
    )
