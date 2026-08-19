"""Differential tests for fused MASShatter topology and candidate assembly."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
import torch as th
import torch.nn.functional as F

from fragnnet.frag.bond_mask_generator import (
    connected_component_atom_masks,
    iter_ranked_cut_masks,
)
from fragnnet.frag.mass_shatter_candidates import build_mass_shatter_candidates
from fragnnet.frag.mass_shatter_core import build_component_arrays_raw

BOUNDED_CUT_COUNT_WEIGHTS = (0.0, 0.0, 0.35, 0.30, 0.20, 0.10, 0.05)


def _legacy_component_arrays(
    num_atoms: int,
    bond_endpoints: np.ndarray,
    broken_bond_sets: Sequence[Sequence[int]],
    global_bond_indices: np.ndarray,
    *,
    atom_offset: int = 0,
    mask_offset: int = 0,
    component_offset: int = 0,
) -> tuple[tuple[np.ndarray, ...], int, int]:
    """Reproduce the former Python-bitmask assembly as a test oracle."""
    endpoint_rows = bond_endpoints.T.tolist()
    component_masks_by_mask = [
        connected_component_atom_masks(num_atoms, endpoint_rows, broken_bonds)
        for broken_bonds in broken_bond_sets
    ]
    component_masks = [
        component_mask
        for mask_components in component_masks_by_mask
        for component_mask in mask_components
    ]
    component_counts = np.asarray(
        [len(mask_components) for mask_components in component_masks_by_mask],
        dtype=np.int64,
    )
    cut_counts = np.asarray(
        [len(broken_bonds) for broken_bonds in broken_bond_sets],
        dtype=np.int64,
    )
    component_mask_indices = np.repeat(
        np.arange(
            mask_offset,
            mask_offset + len(broken_bond_sets),
            dtype=np.int64,
        ),
        component_counts,
    )
    component_cut_counts = np.repeat(cut_counts, component_counts)

    atom_indices: list[int] = []
    atom_component_indices: list[int] = []
    mask_atom_components = np.empty(
        (len(broken_bond_sets), num_atoms),
        dtype=np.int64,
    )
    local_component_index = 0
    for mask_index, mask_components in enumerate(component_masks_by_mask):
        for component_mask in mask_components:
            global_component_index = component_offset + local_component_index
            for atom_index in range(num_atoms):
                if component_mask & (1 << atom_index):
                    atom_indices.append(atom_offset + atom_index)
                    atom_component_indices.append(global_component_index)
                    mask_atom_components[mask_index, atom_index] = global_component_index
            local_component_index += 1

    mask_indices: list[int] = []
    broken_bond_indices: list[int] = []
    boundary_bond_indices: list[int] = []
    boundary_component_indices: list[int] = []
    for mask_index, broken_bonds in enumerate(broken_bond_sets):
        for local_bond_index in broken_bonds:
            global_bond_index = int(global_bond_indices[local_bond_index])
            mask_indices.append(mask_offset + mask_index)
            broken_bond_indices.append(global_bond_index)
            left, right = bond_endpoints[local_bond_index]
            left_component = int(mask_atom_components[mask_index, left])
            right_component = int(mask_atom_components[mask_index, right])
            if left_component == right_component:
                continue
            boundary_bond_indices.extend((global_bond_index, global_bond_index))
            boundary_component_indices.extend((left_component, right_component))

    arrays = (
        cut_counts,
        component_mask_indices,
        component_cut_counts,
        np.asarray(atom_indices, dtype=np.int64),
        np.asarray(atom_component_indices, dtype=np.int64),
        np.asarray(mask_indices, dtype=np.int64),
        np.asarray(broken_bond_indices, dtype=np.int64),
        np.asarray(boundary_bond_indices, dtype=np.int64),
        np.asarray(boundary_component_indices, dtype=np.int64),
    )
    return arrays, len(component_masks), len(set(component_masks))


def _assert_matches_legacy(
    num_atoms: int,
    bond_endpoints: np.ndarray,
    broken_bond_sets: list[tuple[int, ...]],
    global_bond_indices: np.ndarray,
    *,
    atom_offset: int = 0,
    mask_offset: int = 0,
    component_offset: int = 0,
) -> tuple[np.ndarray, ...]:
    expected_arrays, expected_component_count, expected_unique_count = (
        _legacy_component_arrays(
            num_atoms,
            bond_endpoints,
            broken_bond_sets,
            global_bond_indices,
            atom_offset=atom_offset,
            mask_offset=mask_offset,
            component_offset=component_offset,
        )
    )
    actual = build_component_arrays_raw(
        num_atoms,
        bond_endpoints,
        broken_bond_sets,
        global_bond_indices,
        atom_offset,
        mask_offset,
        component_offset,
    )

    actual_arrays = actual[:9]
    for actual_array, expected_array in zip(actual_arrays, expected_arrays, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)
        assert actual_array.dtype == np.int64
    assert actual[9] == expected_component_count
    assert actual[10] == expected_unique_count
    assert actual[11] >= 0.0
    assert actual[12] >= 0.0
    return actual_arrays


@pytest.mark.parametrize("seed", range(8))
def test_fused_component_arrays_match_legacy_on_random_multigraphs(seed: int) -> None:
    rng = np.random.default_rng(9183 + seed)
    num_atoms = int(rng.integers(1, 13))
    num_bonds = int(rng.integers(0, 3 * num_atoms + 1))
    bond_endpoints = rng.integers(
        0,
        num_atoms,
        size=(num_bonds, 2),
        dtype=np.int64,
    )
    global_bond_indices = (
        rng.permutation(num_bonds).astype(np.int64) * 3 + 11
    )
    broken_bond_sets: list[tuple[int, ...]] = [(), tuple(range(num_bonds))]
    for _ in range(10):
        cut_count = int(rng.integers(0, num_bonds + 1))
        broken_bond_sets.append(
            tuple(
                sorted(
                    rng.choice(num_bonds, size=cut_count, replace=False).tolist()
                )
            )
        )

    _assert_matches_legacy(
        num_atoms,
        bond_endpoints,
        broken_bond_sets,
        global_bond_indices,
        atom_offset=17,
        mask_offset=23,
        component_offset=31,
    )


def test_fused_component_arrays_preserve_redundant_cut_boundary_order() -> None:
    bond_endpoints = np.asarray([(0, 1), (1, 2), (2, 0)], dtype=np.int64)
    broken_bond_sets = [(2,), (2, 0), (1, 0, 2)]
    global_bond_indices = np.asarray([10, 4, 8], dtype=np.int64)

    arrays = _assert_matches_legacy(
        3,
        bond_endpoints,
        broken_bond_sets,
        global_bond_indices,
    )

    assert arrays[5].tolist() == [0, 1, 1, 2, 2, 2]
    assert arrays[6].tolist() == [8, 8, 10, 4, 10, 8]
    # The first mask breaks one ring bond but leaves its endpoints connected.
    assert arrays[7].tolist() == [8, 8, 10, 10, 4, 4, 10, 10, 8, 8]
    assert arrays[8].tolist() == [2, 1, 1, 2, 4, 5, 3, 4, 5, 3]


def test_fused_component_arrays_preserve_bitmask_order_above_64_atoms() -> None:
    # Breaking bond 0 yields {1, ..., 69} and {0, 70}. Numeric Python bitmask
    # order puts the former first because max atom 69 precedes max atom 70,
    # even though the latter contains the smaller minimum atom.
    bond_endpoints = np.asarray(
        [(0, 1), (0, 70), *((atom, atom + 1) for atom in range(1, 69))],
        dtype=np.int64,
    )
    arrays = _assert_matches_legacy(
        71,
        bond_endpoints,
        [(0,)],
        np.arange(bond_endpoints.shape[0], dtype=np.int64),
    )

    assert arrays[3].tolist() == [*range(1, 70), 0, 70]
    assert arrays[4].tolist() == [*[0] * 69, 1, 1]
    assert arrays[8].tolist() == [1, 0]


def test_fused_component_arrays_count_multiword_components_exactly() -> None:
    num_atoms = 66
    bond_endpoints = np.asarray(
        [(atom, atom + 1) for atom in range(num_atoms - 1)],
        dtype=np.int64,
    )
    broken_bond_sets = [(), (64,), (63, 64), (63, 64)]

    _assert_matches_legacy(
        num_atoms,
        bond_endpoints,
        broken_bond_sets,
        np.arange(num_atoms - 1, dtype=np.int64),
    )
    actual = build_component_arrays_raw(
        num_atoms,
        bond_endpoints,
        broken_bond_sets,
        np.arange(num_atoms - 1, dtype=np.int64),
        0,
        0,
        0,
    )

    # Singleton atoms 64 and 65 have identical low 64-bit words, while the
    # repeated final mask must not add new unique components.
    assert actual[9] == 9
    assert actual[10] == 5


def test_fused_component_arrays_match_legacy_above_256_atoms() -> None:
    num_atoms = 257
    bond_endpoints = np.asarray(
        [(atom, atom + 1) for atom in range(num_atoms - 1)],
        dtype=np.int64,
    )
    broken_bond_sets = [(), (255,), (254, 255), (64, 128, 255)]

    arrays = _assert_matches_legacy(
        num_atoms,
        bond_endpoints,
        broken_bond_sets,
        np.arange(500, 500 + num_atoms - 1, dtype=np.int64),
        atom_offset=1000,
        mask_offset=20,
        component_offset=30,
    )

    assert len(arrays[3]) == len(broken_bond_sets) * num_atoms
    assert 1000 + 256 in arrays[3]
    actual = build_component_arrays_raw(
        num_atoms,
        bond_endpoints,
        broken_bond_sets,
        np.arange(500, 500 + num_atoms - 1, dtype=np.int64),
        1000,
        20,
        30,
    )
    assert actual[9] == 10
    assert actual[10] == 8


@pytest.mark.parametrize(
    (
        "logits",
        "max_masks",
        "min_mask_probability",
        "include_single_bond_cuts",
    ),
    [
        ([0.0, 0.0, 0.0, 0.0], 5, 0.0, True),
        ([math.log(4.0), math.log(1.5)], 4, 0.2, True),
        ([2.0, 2.0, 2.0], 1, 0.0, True),
        ([-2.0, -3.0, -4.0], 3, 0.0, False),
    ],
)
def test_no_exploration_numeric_path_matches_legacy_selection_and_assembly(
    logits: list[float],
    max_masks: int,
    min_mask_probability: float,
    include_single_bond_cuts: bool,
) -> None:
    num_bonds = len(logits)
    num_atoms = num_bonds + 1
    bond_endpoints = np.asarray(
        [(atom, atom + 1) for atom in range(num_bonds)],
        dtype=np.int64,
    )

    min_log_probability = (
        math.log(min_mask_probability)
        if min_mask_probability > 0.0
        else -math.inf
    )
    learned_masks = []
    for ranked_mask in iter_ranked_cut_masks(logits, include_empty=True):
        if ranked_mask.log_probability < min_log_probability:
            break
        learned_masks.append(ranked_mask.broken_bond_indices)
        if len(learned_masks) == max_masks:
            break
    selected_masks = set(learned_masks)
    expected_masks = list(learned_masks)
    forced_masks = [()]
    if include_single_bond_cuts:
        forced_masks.extend((bond_index,) for bond_index in range(num_bonds))
    expected_masks.extend(mask for mask in forced_masks if mask not in selected_masks)

    expected_arrays, expected_component_count, expected_unique_count = (
        _legacy_component_arrays(
            num_atoms,
            bond_endpoints,
            expected_masks,
            np.arange(num_bonds, dtype=np.int64),
        )
    )
    generator = th.Generator().manual_seed(9183)
    initial_generator_state = generator.get_state().clone()
    bond_logits = th.tensor(logits, dtype=th.float64, requires_grad=True)
    candidates = build_mass_shatter_candidates(
        bond_logits,
        chemical_bond_index=th.from_numpy(bond_endpoints.T.copy()),
        atom_batch_indices=th.zeros(num_atoms, dtype=th.long),
        atom_ptr=th.tensor([0, num_atoms], dtype=th.long),
        max_masks=max_masks,
        min_mask_probability=min_mask_probability,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=BOUNDED_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=include_single_bond_cuts,
        generator=generator,
    )

    assert th.equal(generator.get_state(), initial_generator_state)
    expected_mask_count = len(expected_masks)
    expected_component_batch_indices = np.zeros(
        expected_component_count,
        dtype=np.int64,
    )
    expected_tensors = {
        "mask_batch_indices": np.zeros(expected_mask_count, dtype=np.int64),
        "mask_cut_counts": expected_arrays[0],
        "component_mask_indices": expected_arrays[1],
        "component_batch_indices": expected_component_batch_indices,
        "component_cut_counts": expected_arrays[2],
        "atom_indices": expected_arrays[3],
        "atom_component_indices": expected_arrays[4],
        "boundary_bond_indices": expected_arrays[7],
        "boundary_component_indices": expected_arrays[8],
    }
    for field_name, expected_array in expected_tensors.items():
        assert th.equal(
            getattr(candidates, field_name),
            th.from_numpy(expected_array),
        )

    bond_batch_indices = th.zeros(num_bonds, dtype=th.long)
    molecule_log_probability = bond_logits.new_zeros(1).index_add(
        0,
        bond_batch_indices,
        F.logsigmoid(-bond_logits),
    )
    expected_log_probabilities = molecule_log_probability[
        th.zeros(expected_mask_count, dtype=th.long)
    ]
    if expected_arrays[6].size > 0:
        expected_log_probabilities = expected_log_probabilities.index_add(
            0,
            th.from_numpy(expected_arrays[5]),
            bond_logits[th.from_numpy(expected_arrays[6])],
        )
    assert th.equal(candidates.mask_log_probabilities, expected_log_probabilities)
    assert candidates.exploration_mask_count.tolist() == [0.0]
    assert candidates.exploration_mean_cut_count.tolist() == [0.0]
    assert candidates.proposal_single_cut_fraction.tolist() == pytest.approx(
        [sum(len(mask) == 1 for mask in learned_masks) / len(learned_masks)]
    )
    assert candidates.unique_component_count.tolist() == [
        float(expected_unique_count)
    ]
    assert candidates.num_components == expected_component_count
