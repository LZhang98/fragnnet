"""Batch top-ranked bond masks into sparse MASShatter component tensors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch as th
import torch.nn.functional as F

from fragnnet.frag.bond_mask_generator import (
    RankedCutMask,
    iter_ranked_cut_masks,
    top_k_cut_masks,
)

if TYPE_CHECKING:
    from fragnnet.frag.mass_shatter_templates import MassShatterTemplate


@dataclass(frozen=True)
class MassShatterStaticGraph:
    """CPU graph indices retained across Lightning's device transfer."""

    chemical_bond_index: th.Tensor
    atom_batch_indices: th.Tensor
    atom_ptr: th.Tensor
    smarts_templates: tuple[tuple[MassShatterTemplate, ...], ...] | None = None


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
    mechanism_mask_indices: th.Tensor
    mechanism_rule_indices: th.Tensor
    mechanism_boundary_bond_indices: th.Tensor
    boundary_mechanism_indices: th.Tensor
    component_mask_indices: th.Tensor
    component_mechanism_indices: th.Tensor
    component_batch_indices: th.Tensor
    component_cut_counts: th.Tensor
    component_unique_indices: th.Tensor
    unique_representative_component_indices: th.Tensor
    atom_indices: th.Tensor
    atom_component_indices: th.Tensor
    boundary_bond_indices: th.Tensor
    boundary_component_indices: th.Tensor
    exploration_mask_count: th.Tensor
    exploration_mean_cut_count: th.Tensor
    proposal_single_cut_fraction: th.Tensor
    unique_component_count: th.Tensor
    component_counts: tuple[int, ...]
    unique_component_counts: tuple[int, ...]
    timings: MassShatterCandidateTimings

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
class _RankedMaskCache:
    """Learned masks shared by repeated collision-energy views."""

    logits: np.ndarray
    mixture_log_weights: np.ndarray | None
    mask_ptr: np.ndarray
    broken_bonds: np.ndarray


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
    component_unique_indices: np.ndarray
    unique_representative_component_indices: np.ndarray


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


def _top_mixture_bond_sets(
    logits: np.ndarray,
    mixture_log_weights: np.ndarray,
    limit: int,
    min_probability: float,
    min_relative_probability: float,
) -> list[tuple[int, ...]]:
    """Return a deduplicated pool ranked by exact Bernoulli-mixture probability."""
    from fragnnet.frag.mass_shatter_core import top_k_cut_masks_csr_raw

    bond_set_pool: set[tuple[int, ...]] = set()
    for mixture_index in range(logits.shape[1]):
        mask_ptr, broken_bonds, _, _ = top_k_cut_masks_csr_raw(
            logits[:, mixture_index].tolist(),
            limit,
            True,
            0.0,
            0.0,
        )
        bond_set_pool.update(
            tuple(int(index) for index in broken_bonds[mask_ptr[i] : mask_ptr[i + 1]])
            for i in range(mask_ptr.size - 1)
        )

    bond_sets = sorted(bond_set_pool)
    intact_log_probabilities = -np.logaddexp(0.0, logits).sum(axis=0)
    component_log_probabilities = np.broadcast_to(
        intact_log_probabilities,
        (len(bond_sets), logits.shape[1]),
    ).copy()
    cut_counts = np.asarray([len(bond_set) for bond_set in bond_sets], dtype=np.int64)
    flat_bond_indices = np.asarray(
        [bond_index for bond_set in bond_sets for bond_index in bond_set],
        dtype=np.int64,
    )
    if flat_bond_indices.size > 0:
        np.add.at(
            component_log_probabilities,
            np.repeat(np.arange(len(bond_sets)), cut_counts),
            logits[flat_bond_indices],
        )
    mixture_log_probabilities = np.logaddexp.reduce(
        mixture_log_weights + component_log_probabilities,
        axis=1,
    )
    scored_bond_sets = list(zip(mixture_log_probabilities.tolist(), bond_sets))
    scored_bond_sets.sort(key=lambda item: (-item[0], item[1]))
    if not scored_bond_sets:
        return []

    minimum_log_probability = (
        math.log(min_probability) if min_probability > 0.0 else -math.inf
    )
    minimum_relative_log_probability = (
        math.log(min_relative_probability)
        if min_relative_probability > 0.0
        else -math.inf
    )
    map_log_probability = scored_bond_sets[0][0]
    return [
        bond_set
        for log_probability, bond_set in scored_bond_sets
        if log_probability >= minimum_log_probability
        and log_probability - map_log_probability >= minimum_relative_log_probability
    ][:limit]


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
    min_mask_relative_probability: float = 0.0,
    max_unique_components: int | None = None,
    max_enumerated_masks: int | None = None,
    generator: th.Generator | None = None,
    static_graph: MassShatterStaticGraph | None = None,
    smarts_templates: Sequence[Sequence[MassShatterTemplate]] | None = None,
    mixture_log_weights: th.Tensor | None = None,
    candidate_group_ids: th.Tensor | None = None,
) -> MassShatterCandidateBatch:
    """Select masks on CPU and return sparse pathway tensors on the model device.

    Mask identities are deliberately selected from detached logits. Their exact
    Bernoulli or Bernoulli-mixture log probabilities are then recomputed from the
    original logits, preserving gradients to every selected cut/intact decision and
    latent mixture weight. When ``candidate_group_ids`` is provided, repeated
    collision-energy views with the same group id reuse exact ranked masks and
    connected-component topology while retaining separate differentiable
    probabilities and output candidates. When
    ``max_unique_components`` is set, deterministic enumeration continues up to
    ``max_enumerated_masks`` and keeps the shortest prefix that reaches the requested
    number of distinct atom sets. Probability floors apply only to learned ranked
    masks; required intact and single-cut masks remain forced.
    """
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
    if max_unique_components is not None and max_unique_components <= 0:
        raise ValueError(
            "max_unique_components must be positive when enabled, "
            f"got {max_unique_components}"
        )
    if max_enumerated_masks is not None and max_enumerated_masks < max_masks:
        raise ValueError(
            "max_enumerated_masks must be at least max_masks, "
            f"got {max_enumerated_masks} and {max_masks}"
        )
    if max_unique_components is not None and max_enumerated_masks is None:
        raise ValueError(
            "max_enumerated_masks is required when max_unique_components is enabled"
        )
    if not math.isfinite(min_mask_probability) or not 0.0 <= min_mask_probability <= 1.0:
        raise ValueError(
            "min_mask_probability must be finite and in [0, 1], "
            f"got {min_mask_probability}"
        )
    if (
        not math.isfinite(min_mask_relative_probability)
        or not 0.0 <= min_mask_relative_probability <= 1.0
    ):
        raise ValueError(
            "min_mask_relative_probability must be finite and in [0, 1], "
            f"got {min_mask_relative_probability}"
        )
    if explore_masks < 0:
        raise ValueError(f"explore_masks must be nonnegative, got {explore_masks}")
    if num_mixtures == 1 and mixture_log_weights is not None:
        raise ValueError("mixture_log_weights must be omitted for one Bernoulli component")
    if num_mixtures > 1 and mixture_log_weights is None:
        raise ValueError("mixture_log_weights are required for multiple Bernoulli components")
    if num_mixtures > 1 and explore_masks > 0:
        raise ValueError("Bernoulli mixtures do not support exploration masks")
    if max_unique_components is not None and explore_masks > 0:
        raise ValueError("unique-component enumeration does not support exploration masks")
    if candidate_group_ids is not None and explore_masks > 0:
        raise ValueError("candidate_group_ids does not support exploration masks")
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
        if (
            candidate_group_ids.shape != (num_molecules,)
            or candidate_group_ids.dtype != th.long
        ):
            raise ValueError(
                "candidate_group_ids must be a torch.long vector with one id per molecule"
            )
        candidate_group_values = candidate_group_ids.detach().cpu().tolist()
    input_transfer_ms = (perf_counter() - input_transfer_start) * 1000.0
    if mixture_log_weights is not None:
        if mixture_log_weights.shape != (num_molecules, num_mixtures):
            raise ValueError(
                "mixture_log_weights must have shape (num_molecules, num_mixtures)"
            )
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
    component_unique_index_chunks: list[np.ndarray] = []
    unique_representative_component_index_chunks: list[np.ndarray] = []
    atom_index_chunks: list[np.ndarray] = []
    atom_component_index_chunks: list[np.ndarray] = []
    boundary_bond_index_chunks: list[np.ndarray] = []
    boundary_component_index_chunks: list[np.ndarray] = []
    exploration_mask_counts: list[int] = []
    exploration_mean_cut_counts: list[float] = []
    proposal_single_cut_fractions: list[float] = []
    component_counts: list[int] = []
    unique_component_counts: list[int] = []
    mask_selection_ms = 0.0
    topology_ms = 0.0
    assembly_ms = 0.0
    num_masks_so_far = 0
    num_mechanisms_so_far = 0
    num_components_so_far = 0
    num_unique_components_so_far = 0
    ranked_mask_cache: dict[int, _RankedMaskCache] = {}
    topology_cache: dict[int, _TopologyCache] = {}

    # 2. Select and assemble masks independently for each precursor.
    for molecule_index in range(num_molecules):
        atom_start = atom_ptr_values[molecule_index]
        atom_end = atom_ptr_values[molecule_index + 1]
        num_atoms = atom_end - atom_start
        if num_atoms <= 0:
            raise ValueError("every precursor must contain at least one atom")
        candidate_group = (
            None
            if candidate_group_values is None
            else int(candidate_group_values[molecule_index])
        )

        selection_start = perf_counter()
        global_bond_indices = bond_indices_by_molecule[molecule_index]
        local_logits = np.asarray(
            [detached_logits_all[index] for index in global_bond_indices],
            dtype=np.float64,
        )
        if num_mixtures == 1:
            detached_logits = local_logits.reshape(-1).tolist()
        else:
            local_logits = local_logits.reshape(-1, num_mixtures)
            detached_logits = []
        local_endpoints = tuple(
            tuple(chemical_bond_rows[row][index] - atom_start for index in global_bond_indices)
            for row in range(2)
        )
        molecule_templates = (
            tuple(smarts_templates[molecule_index])
            if smarts_templates is not None
            else ()
        )
        template_bond_sets = [
            tuple(template.broken_bond_indices) for template in molecule_templates
        ]
        forced_bond_sets = [()]
        if include_single_bond_cuts:
            forced_bond_sets.extend(
                (bond_index,) for bond_index in range(len(detached_logits))
            )
        forced_bond_sets.extend(template_bond_sets)
        forced_bond_sets = list(dict.fromkeys(forced_bond_sets))
        if explore_masks == 0:
            # The production Transformer path stays numeric from exact top-k
            # selection through topology assembly. Only forced empty/single
            # masks are appended here, in their historical order.
            from fragnnet.frag.mass_shatter_core import top_k_cut_masks_csr_raw

            learned_mask_limit = (
                max_enumerated_masks
                if max_unique_components is not None
                else max_masks
            )
            cached_ranked_masks = (
                None
                if candidate_group is None
                else ranked_mask_cache.get(candidate_group)
            )
            local_mixture_log_weights = (
                None
                if mixture_log_weights_cpu is None
                else mixture_log_weights_cpu[molecule_index].numpy()
            )
            if cached_ranked_masks is not None:
                if not np.array_equal(cached_ranked_masks.logits, local_logits) or (
                    cached_ranked_masks.mixture_log_weights is None
                ) != (local_mixture_log_weights is None):
                    raise ValueError(
                        "candidate_group_ids can only group identical proposal logits"
                    )
                if (
                    local_mixture_log_weights is not None
                    and not np.array_equal(
                        cached_ranked_masks.mixture_log_weights,
                        local_mixture_log_weights,
                    )
                ):
                    raise ValueError(
                        "candidate_group_ids can only group identical mixture weights"
                    )
                local_mask_ptr = cached_ranked_masks.mask_ptr
                local_broken_bonds = cached_ranked_masks.broken_bonds
            elif num_mixtures == 1:
                local_mask_ptr, local_broken_bonds, _, _ = top_k_cut_masks_csr_raw(
                    detached_logits,
                    learned_mask_limit,
                    True,
                    min_mask_probability,
                    min_mask_relative_probability,
                )
            else:
                if mixture_log_weights_cpu is None:
                    raise RuntimeError("mixture weights disappeared during mask selection")
                mixture_bond_sets = _top_mixture_bond_sets(
                    local_logits,
                    mixture_log_weights_cpu[molecule_index].numpy(),
                    learned_mask_limit,
                    min_mask_probability,
                    min_mask_relative_probability,
                )
                mixture_cut_counts = np.asarray(
                    [len(bond_set) for bond_set in mixture_bond_sets],
                    dtype=np.int64,
                )
                local_mask_ptr = np.concatenate(
                    (
                        np.zeros(1, dtype=np.int64),
                        np.cumsum(mixture_cut_counts, dtype=np.int64),
                    )
                )
                local_broken_bonds = np.asarray(
                    [bond for bond_set in mixture_bond_sets for bond in bond_set],
                    dtype=np.int64,
                )
            if cached_ranked_masks is None and candidate_group is not None:
                ranked_mask_cache[candidate_group] = _RankedMaskCache(
                    logits=local_logits.copy(),
                    mixture_log_weights=(
                        None
                        if local_mixture_log_weights is None
                        else local_mixture_log_weights.copy()
                    ),
                    mask_ptr=local_mask_ptr.copy(),
                    broken_bonds=local_broken_bonds.copy(),
                )
            learned_mask_count = local_mask_ptr.size - 1
            if min_mask_probability > 0.0 and learned_mask_count == 0:
                raise ValueError(
                    "mass_shatter_min_mask_probability excludes every learned mask "
                    f"for precursor {molecule_index}"
                )

            learned_cut_counts = np.diff(local_mask_ptr)
            proposal_single_cut_fractions.append(
                float(np.count_nonzero(learned_cut_counts == 1)) / learned_mask_count
            )
            exploration_mask_counts.append(0)
            exploration_mean_cut_counts.append(0.0)

            forced_mask_count = 0
            if max_unique_components is not None:
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
                if molecule_templates:
                    forced_bond_set_lookup = set(forced_bond_sets)
                    ranked_bond_sets = [
                        tuple(
                            local_broken_bonds[
                                local_mask_ptr[index] : local_mask_ptr[index + 1]
                            ]
                        )
                        for index in range(learned_mask_count)
                    ]
                    keep_ranked_masks = np.asarray(
                        [
                            bond_set not in forced_bond_set_lookup
                            for bond_set in ranked_bond_sets
                        ],
                        dtype=np.bool_,
                    )
                else:
                    # Without SMARTS templates, forced masks are exactly the empty
                    # mask and, when requested, every single cut. Remove their ranked
                    # duplicates by cardinality instead of materializing thousands of
                    # Python tuples per precursor.
                    maximum_forced_cut_count = 1 if len(forced_bond_sets) > 1 else 0
                    keep_ranked_masks = learned_cut_counts > maximum_forced_cut_count
                keep_ranked_bonds = np.repeat(keep_ranked_masks, learned_cut_counts)
                retained_ranked_cut_counts = learned_cut_counts[keep_ranked_masks]
                retained_ranked_bonds = local_broken_bonds[keep_ranked_bonds]
                combined_cut_counts = np.concatenate(
                    (forced_cut_counts, retained_ranked_cut_counts)
                )
                local_mask_ptr = np.concatenate(
                    (
                        np.zeros(1, dtype=np.int64),
                        np.cumsum(combined_cut_counts, dtype=np.int64),
                    )
                )
                local_broken_bonds = np.concatenate(
                    (forced_bonds, retained_ranked_bonds)
                )
                forced_mask_count = forced_cut_counts.size
            else:
                extra_cut_counts: list[int] = []
                extra_bonds: list[int] = []
                selected_bond_sets = {
                    tuple(local_broken_bonds[local_mask_ptr[index] : local_mask_ptr[index + 1]])
                    for index in range(learned_mask_count)
                }
                for bond_set in forced_bond_sets:
                    if bond_set in selected_bond_sets:
                        continue
                    extra_cut_counts.append(len(bond_set))
                    extra_bonds.extend(bond_set)
                if extra_cut_counts:
                    local_mask_ptr = np.concatenate(
                        (
                            local_mask_ptr,
                            local_mask_ptr[-1]
                            + np.cumsum(extra_cut_counts, dtype=np.int64),
                        )
                    )
                    if extra_bonds:
                        local_broken_bonds = np.concatenate(
                            (local_broken_bonds, np.asarray(extra_bonds, dtype=np.int64))
                        )
            broken_mask_input: object = (local_mask_ptr, local_broken_bonds)
            num_local_masks = local_mask_ptr.size - 1
            selected_local_bond_sets = (
                [
                    tuple(
                        int(bond)
                        for bond in local_broken_bonds[
                            local_mask_ptr[index] : local_mask_ptr[index + 1]
                        ]
                    )
                    for index in range(num_local_masks)
                ]
                if molecule_templates
                else None
            )
        else:
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
                    min_relative_probability=min_mask_relative_probability,
                )
            )
            if min_mask_probability > 0.0 and not ranked_masks:
                raise ValueError(
                    "mass_shatter_min_mask_probability excludes every learned mask "
                    f"for precursor {molecule_index}"
                )
            selected_bond_sets = {
                ranked_mask.broken_bond_indices for ranked_mask in ranked_masks
            }
            reserved_bond_sets = set(forced_bond_sets)
            unavailable_bond_sets = selected_bond_sets | reserved_bond_sets

            sampled_count = 0
            sampled_cut_count = 0
            if exploration_mode == "bernoulli_replace":
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
                min_relative_log_probability = (
                    math.log(min_mask_relative_probability)
                    if min_mask_relative_probability > 0.0
                    else -math.inf
                )
                for refill_mask in iter_ranked_cut_masks(detached_logits, include_empty=True):
                    if (
                        refill_mask.log_probability < min_log_probability
                        or refill_mask.log_probability_from_map
                        < min_relative_log_probability
                    ):
                        break
                    broken_bond_indices = refill_mask.broken_bond_indices
                    if broken_bond_indices in unavailable_bond_sets:
                        continue
                    ranked_masks.append(refill_mask)
                    selected_bond_sets.add(broken_bond_indices)
                    unavailable_bond_sets.add(broken_bond_indices)
                    if len(ranked_masks) == max_masks:
                        break

            if exploration_mode == "bounded_additive":
                # Draw cardinalities and Gumbel top-k masks in one compiled call.
                valid_weight_count = min(
                    len(cut_count_weights), len(detached_logits) + 1
                )
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
                        from fragnnet.frag.mass_shatter_core import (
                            bounded_additive_masks_raw,
                        )

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
            broken_mask_input = [
                ranked_mask.broken_bond_indices for ranked_mask in ranked_masks
            ]
            num_local_masks = len(ranked_masks)
            selected_local_bond_sets = list(broken_mask_input)
        mask_selection_ms += (perf_counter() - selection_start) * 1000.0

        # 3. Build connected components and every flat candidate index without
        # materializing Python dicts, arbitrary-width bitmasks, or dataclasses.
        from fragnnet.frag.mass_shatter_core import build_component_arrays_raw

        mask_start = num_masks_so_far
        component_start = num_components_so_far
        cached_topology = (
            None if candidate_group is None else topology_cache.get(candidate_group)
        )
        if cached_topology is None:
            (
                local_cut_counts,
                local_component_mask_indices,
                local_component_cut_counts,
                local_atom_indices,
                local_atom_component_indices,
                local_mask_indices,
                local_broken_bond_indices,
                local_boundary_bond_indices,
                local_boundary_component_indices,
                num_local_components,
                unique_component_count,
                local_topology_ms,
                local_assembly_ms,
                local_component_unique_indices,
                local_unique_representative_component_indices,
            ) = build_component_arrays_raw(
                num_atoms,
                np.asarray(local_endpoints, dtype=np.int64).T,
                broken_mask_input,
                global_bond_indices,
                atom_start,
                mask_start,
                component_start,
            )
            if candidate_group is not None:
                if not isinstance(broken_mask_input, tuple):
                    raise RuntimeError("shared candidate masks must use CSR arrays")
                global_to_local_bond = {
                    int(global_index): local_index
                    for local_index, global_index in enumerate(global_bond_indices)
                }
                local_broken_output = np.asarray(
                    [
                        global_to_local_bond[int(global_index)]
                        for global_index in local_broken_bond_indices
                    ],
                    dtype=np.int64,
                )
                local_boundary_output = np.asarray(
                    [
                        global_to_local_bond[int(global_index)]
                        for global_index in local_boundary_bond_indices
                    ],
                    dtype=np.int64,
                )
                topology_cache[candidate_group] = _TopologyCache(
                    num_atoms=num_atoms,
                    endpoints=local_endpoints,
                    mask_ptr=broken_mask_input[0].copy(),
                    broken_bonds=broken_mask_input[1].copy(),
                    cut_counts=local_cut_counts.copy(),
                    component_mask_indices=(
                        local_component_mask_indices - mask_start
                    ),
                    component_cut_counts=local_component_cut_counts.copy(),
                    atom_indices=local_atom_indices - atom_start,
                    atom_component_indices=(
                        local_atom_component_indices - component_start
                    ),
                    mask_indices=local_mask_indices - mask_start,
                    broken_bond_indices=local_broken_output,
                    boundary_bond_indices=local_boundary_output,
                    boundary_component_indices=(
                        local_boundary_component_indices - component_start
                    ),
                    num_components=num_local_components,
                    unique_component_count=unique_component_count,
                    component_unique_indices=local_component_unique_indices.copy(),
                    unique_representative_component_indices=(
                        local_unique_representative_component_indices.copy()
                    ),
                )
        else:
            if not isinstance(broken_mask_input, tuple):
                raise RuntimeError("shared candidate masks must use CSR arrays")
            if (
                cached_topology.num_atoms != num_atoms
                or cached_topology.endpoints != local_endpoints
                or not np.array_equal(cached_topology.mask_ptr, broken_mask_input[0])
                or not np.array_equal(
                    cached_topology.broken_bonds,
                    broken_mask_input[1],
                )
            ):
                raise ValueError(
                    "candidate_group_ids can only group identical molecular topology"
                )
            remap_start = perf_counter()
            global_bond_array = np.asarray(global_bond_indices, dtype=np.int64)
            local_cut_counts = cached_topology.cut_counts
            local_component_mask_indices = (
                cached_topology.component_mask_indices + mask_start
            )
            local_component_cut_counts = cached_topology.component_cut_counts
            local_atom_indices = cached_topology.atom_indices + atom_start
            local_atom_component_indices = (
                cached_topology.atom_component_indices + component_start
            )
            local_mask_indices = cached_topology.mask_indices + mask_start
            local_broken_bond_indices = global_bond_array[
                cached_topology.broken_bond_indices
            ]
            local_boundary_bond_indices = global_bond_array[
                cached_topology.boundary_bond_indices
            ]
            local_boundary_component_indices = (
                cached_topology.boundary_component_indices + component_start
            )
            num_local_components = cached_topology.num_components
            unique_component_count = cached_topology.unique_component_count
            local_component_unique_indices = cached_topology.component_unique_indices
            local_unique_representative_component_indices = (
                cached_topology.unique_representative_component_indices
            )
            local_topology_ms = 0.0
            local_assembly_ms = (perf_counter() - remap_start) * 1000.0
        if (
            max_unique_components is not None
            and unique_component_count > max_unique_components
        ):
            # Unique representatives are ordered by first occurrence. Keep the
            # shortest mask prefix that reaches the target, while retaining every
            # forced low-cut mask at the front of the sequence.
            target_representative = local_unique_representative_component_indices[
                max_unique_components - 1
            ]
            target_mask = (
                local_component_mask_indices[target_representative] - mask_start
            )
            num_local_masks = max(int(target_mask) + 1, forced_mask_count)
            mask_end = mask_start + num_local_masks
            num_local_components = int(
                np.searchsorted(local_component_mask_indices, mask_end)
            )
            component_end = component_start + num_local_components
            atom_membership_end = int(
                np.searchsorted(local_atom_component_indices, component_end)
            )
            broken_bond_end = int(np.searchsorted(local_mask_indices, mask_end))
            boundary_keep = local_boundary_component_indices < component_end

            local_cut_counts = local_cut_counts[:num_local_masks]
            local_component_mask_indices = local_component_mask_indices[
                :num_local_components
            ]
            local_component_cut_counts = local_component_cut_counts[
                :num_local_components
            ]
            local_component_unique_indices = local_component_unique_indices[
                :num_local_components
            ]
            local_atom_indices = local_atom_indices[:atom_membership_end]
            local_atom_component_indices = local_atom_component_indices[
                :atom_membership_end
            ]
            local_mask_indices = local_mask_indices[:broken_bond_end]
            local_broken_bond_indices = local_broken_bond_indices[:broken_bond_end]
            local_boundary_bond_indices = local_boundary_bond_indices[boundary_keep]
            local_boundary_component_indices = local_boundary_component_indices[
                boundary_keep
            ]
            representative_keep = (
                local_unique_representative_component_indices < num_local_components
            )
            local_unique_representative_component_indices = (
                local_unique_representative_component_indices[representative_keep]
            )
            unique_component_count = len(
                local_unique_representative_component_indices
            )
        if selected_local_bond_sets is not None:
            selected_local_bond_sets = selected_local_bond_sets[:num_local_masks]

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
        local_boundary_mechanism_indices = (
            local_mask_indices - mask_start + mechanism_start
        )

        if molecule_templates:
            if selected_local_bond_sets is None:
                raise RuntimeError("selected bond sets disappeared for SMARTS templates")
            bond_set_to_mask = {
                bond_set: mask_start + local_index
                for local_index, bond_set in enumerate(selected_local_bond_sets)
            }
            component_atoms: dict[int, list[int]] = {
                component_start + index: [] for index in range(num_local_components)
            }
            for atom_index, component_index in zip(
                local_atom_indices.tolist(),
                local_atom_component_indices.tolist(),
            ):
                component_atoms[component_index].append(atom_index)
            atom_set_to_unique = {
                tuple(atoms): int(local_component_unique_indices[component_index - component_start])
                for component_index, atoms in component_atoms.items()
            }

            extra_mechanism_masks: list[int] = []
            extra_mechanism_rules: list[int] = []
            extra_mechanism_boundary_bonds: list[int] = []
            extra_boundary_mechanisms: list[int] = []
            extra_component_masks: list[int] = []
            extra_component_mechanisms: list[int] = []
            extra_component_cut_counts: list[int] = []
            extra_component_unique_indices: list[int] = []
            extra_atom_indices: list[int] = []
            extra_atom_component_indices: list[int] = []
            extra_boundary_bonds: list[int] = []
            extra_boundary_components: list[int] = []
            extra_representatives: list[int] = []

            for template in molecule_templates:
                mask_index = bond_set_to_mask.get(tuple(template.broken_bond_indices))
                if mask_index is None:
                    continue
                mechanism_index = (
                    mechanism_start + num_local_masks + len(extra_mechanism_masks)
                )
                extra_mechanism_masks.append(mask_index)
                extra_mechanism_rules.append(template.rule_index)
                for local_bond_index in template.broken_bond_indices:
                    extra_mechanism_boundary_bonds.append(
                        global_bond_indices[local_bond_index]
                    )
                    extra_boundary_mechanisms.append(mechanism_index)

                for component in template.components:
                    component_index = (
                        component_start + num_local_components + len(extra_component_masks)
                    )
                    global_atoms = tuple(atom_start + atom for atom in component)
                    unique_index = atom_set_to_unique.get(global_atoms)
                    if unique_index is None:
                        unique_index = unique_component_count
                        unique_component_count += 1
                        atom_set_to_unique[global_atoms] = unique_index
                        # Representatives are molecule-local; component_start is
                        # added once when the chunk is appended.
                        extra_representatives.append(component_index - component_start)
                    extra_component_masks.append(mask_index)
                    extra_component_mechanisms.append(mechanism_index)
                    extra_component_cut_counts.append(len(template.broken_bond_indices))
                    extra_component_unique_indices.append(unique_index)
                    extra_atom_indices.extend(global_atoms)
                    extra_atom_component_indices.extend(
                        [component_index] * len(global_atoms)
                    )
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

        topology_ms += local_topology_ms
        assembly_ms += local_assembly_ms
        mask_batch_index_chunks.append(
            np.full(num_local_masks, molecule_index, dtype=np.int64)
        )
        mask_cut_count_chunks.append(local_cut_counts)
        mechanism_mask_index_chunks.append(local_mechanism_mask_indices)
        mechanism_rule_index_chunks.append(local_mechanism_rule_indices)
        if local_mechanism_boundary_bond_indices.size > 0:
            mechanism_boundary_bond_index_chunks.append(
                local_mechanism_boundary_bond_indices
            )
            boundary_mechanism_index_chunks.append(local_boundary_mechanism_indices)
        component_mask_index_chunks.append(local_component_mask_indices)
        component_mechanism_index_chunks.append(local_component_mechanism_indices)
        component_batch_index_chunks.append(
            np.full(num_local_components, molecule_index, dtype=np.int64)
        )
        component_cut_count_chunks.append(local_component_cut_counts)
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

    # 4. Concatenate the per-precursor numeric chunks into one CPU batch.
    mask_batch_indices = np.concatenate(mask_batch_index_chunks)
    mask_cut_counts = np.concatenate(mask_cut_count_chunks)
    mechanism_mask_indices = np.concatenate(mechanism_mask_index_chunks)
    mechanism_rule_indices = np.concatenate(mechanism_rule_index_chunks)
    mechanism_boundary_bond_indices = (
        np.concatenate(mechanism_boundary_bond_index_chunks)
        if mechanism_boundary_bond_index_chunks
        else np.empty(0, dtype=np.int64)
    )
    boundary_mechanism_indices = (
        np.concatenate(boundary_mechanism_index_chunks)
        if boundary_mechanism_index_chunks
        else np.empty(0, dtype=np.int64)
    )
    component_mask_indices = np.concatenate(component_mask_index_chunks)
    component_mechanism_indices = np.concatenate(component_mechanism_index_chunks)
    component_batch_indices = np.concatenate(component_batch_index_chunks)
    component_cut_counts = np.concatenate(component_cut_count_chunks)
    component_unique_indices = np.concatenate(component_unique_index_chunks)
    unique_representative_component_indices = np.concatenate(
        unique_representative_component_index_chunks
    )
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
        mechanism_mask_indices_t,
        mechanism_rule_indices_t,
        mechanism_boundary_bond_indices_t,
        boundary_mechanism_indices_t,
        component_mask_indices_t,
        component_mechanism_indices_t,
        component_batch_indices_t,
        component_cut_counts_t,
        component_unique_indices_t,
        unique_representative_component_indices_t,
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
            mechanism_mask_indices,
            mechanism_rule_indices,
            mechanism_boundary_bond_indices,
            boundary_mechanism_indices,
            component_mask_indices,
            component_mechanism_indices,
            component_batch_indices,
            component_cut_counts,
            component_unique_indices,
            unique_representative_component_indices,
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
        (num_molecules,) if num_mixtures == 1 else (num_molecules, num_mixtures),
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
    if num_mixtures > 1:
        if mixture_log_weights is None:
            raise RuntimeError("mixture weights disappeared during probability evaluation")
        mask_log_probabilities = th.logsumexp(
            mask_log_probabilities
            + mixture_log_weights[mask_batch_indices_t],
            dim=1,
        )

    return MassShatterCandidateBatch(
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
        component_unique_indices=component_unique_indices_t,
        unique_representative_component_indices=(
            unique_representative_component_indices_t
        ),
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
        component_counts=tuple(component_counts),
        unique_component_counts=tuple(unique_component_counts),
        timings=MassShatterCandidateTimings(
            input_transfer_ms=input_transfer_ms,
            mask_selection_ms=mask_selection_ms,
            topology_ms=topology_ms,
            assembly_ms=assembly_ms,
            output_transfer_ms=output_transfer_ms,
            total_ms=(perf_counter() - total_start) * 1000.0,
        ),
    )
