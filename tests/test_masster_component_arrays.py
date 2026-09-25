"""Differential tests for fused MASSter topology and candidate assembly."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch as th
import torch.nn.functional as F

from fragnnet.frag.bond_mask_generator import (
    connected_component_atom_masks,
    iter_ranked_cut_masks,
)
from fragnnet.frag.masster_candidates import build_masster_candidates
from fragnnet.frag.masster_core import build_component_arrays_raw

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
    expected_arrays, expected_component_count, expected_unique_count = _legacy_component_arrays(
        num_atoms,
        bond_endpoints,
        broken_bond_sets,
        global_bond_indices,
        atom_offset=atom_offset,
        mask_offset=mask_offset,
        component_offset=component_offset,
    )
    actual = build_component_arrays_raw(
        num_atoms,
        bond_endpoints,
        _mask_csr(broken_bond_sets),
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
    return actual_arrays


def _mask_csr(
    broken_bond_sets: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack test cut masks into the production CSR representation."""
    cut_counts = np.asarray([len(mask) for mask in broken_bond_sets], dtype=np.int64)
    mask_ptr = np.empty(len(broken_bond_sets) + 1, dtype=np.int64)
    mask_ptr[0] = 0
    np.cumsum(cut_counts, out=mask_ptr[1:])
    broken_bonds = np.asarray(
        [bond for mask in broken_bond_sets for bond in mask],
        dtype=np.int64,
    )
    return mask_ptr, broken_bonds


def _full_build_then_slice(
    full: tuple[object, ...],
    target: int,
    forced_mask_count: int,
    *,
    mask_offset: int,
    component_offset: int,
) -> tuple[tuple[np.ndarray, ...], int, int]:
    """Apply the removed Python truncation block as an exact test oracle."""
    arrays = tuple(np.asarray(value) for value in full[:9])
    component_count = int(full[9])
    unique_count = int(full[10])
    component_unique_indices = np.asarray(full[11])
    representatives = np.asarray(full[12])
    if unique_count <= target:
        return (*arrays, component_unique_indices, representatives), component_count, unique_count

    target_component = int(representatives[target - 1])
    target_mask = int(arrays[1][target_component]) - mask_offset
    retained_mask_count = max(target_mask + 1, forced_mask_count)
    mask_end = mask_offset + retained_mask_count
    component_count = int(np.searchsorted(arrays[1], mask_end))
    component_end = component_offset + component_count
    atom_end = int(np.searchsorted(arrays[4], component_end))
    broken_end = int(np.searchsorted(arrays[5], mask_end))
    boundary_keep = arrays[8] < component_end
    representatives = representatives[representatives < component_count]
    expected = (
        arrays[0][:retained_mask_count],
        arrays[1][:component_count],
        arrays[2][:component_count],
        arrays[3][:atom_end],
        arrays[4][:atom_end],
        arrays[5][:broken_end],
        arrays[6][:broken_end],
        arrays[7][boundary_keep],
        arrays[8][boundary_keep],
        component_unique_indices[:component_count],
        representatives,
    )
    return expected, component_count, representatives.size


def _assert_early_build_matches_full_slice(
    num_atoms: int,
    bond_endpoints: np.ndarray,
    broken_bond_sets: Sequence[Sequence[int]],
    target: int,
    forced_mask_count: int,
    *,
    atom_offset: int = 0,
    mask_offset: int = 0,
    component_offset: int = 0,
) -> tuple[object, ...]:
    """Compare early topology against the prior full-build-then-slice path."""
    mask_csr = _mask_csr(broken_bond_sets)
    global_bond_indices = np.arange(
        101,
        101 + bond_endpoints.shape[0],
        dtype=np.int64,
    )
    full = build_component_arrays_raw(
        num_atoms,
        bond_endpoints,
        mask_csr,
        global_bond_indices,
        atom_offset,
        mask_offset,
        component_offset,
    )
    expected, expected_component_count, expected_unique_count = _full_build_then_slice(
        full,
        target,
        forced_mask_count,
        mask_offset=mask_offset,
        component_offset=component_offset,
    )
    actual = build_component_arrays_raw(
        num_atoms,
        bond_endpoints,
        mask_csr,
        global_bond_indices,
        atom_offset,
        mask_offset,
        component_offset,
        target,
        forced_mask_count,
    )
    actual_arrays = (*actual[:9], actual[11], actual[12])
    for actual_array, expected_array in zip(actual_arrays, expected, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)
    assert actual[9] == expected_component_count
    assert actual[10] == expected_unique_count
    assert 1 <= actual[13] <= len(broken_bond_sets)
    return actual


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
    global_bond_indices = rng.permutation(num_bonds).astype(np.int64) * 3 + 11
    broken_bond_sets: list[tuple[int, ...]] = [(), tuple(range(num_bonds))]
    for _ in range(10):
        cut_count = int(rng.integers(0, num_bonds + 1))
        broken_bond_sets.append(
            tuple(sorted(rng.choice(num_bonds, size=cut_count, replace=False).tolist()))
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


def test_concurrent_component_arrays_preserve_shared_inputs_and_independent_offsets() -> None:
    """Workers sharing immutable CSR inputs must keep their scratch buffers private."""
    num_atoms = 71
    endpoints = np.asarray(
        [(atom, atom + 1) for atom in range(num_atoms - 1)] + [(0, num_atoms - 1)],
        dtype=np.int64,
    )
    masks = [(), (0,), (0, 69), (0, 69, 70), (63, 64, 70)] * 8
    mask_csr = _mask_csr(masks)
    original_csr = tuple(array.copy() for array in mask_csr)
    global_bonds = np.arange(endpoints.shape[0], dtype=np.int64) * 3 + 17

    def construct(offset: int) -> tuple[object, ...]:
        return build_component_arrays_raw(
            num_atoms, endpoints, mask_csr, global_bonds, offset, offset * 2, offset * 3
        )

    offsets = list(range(0, 800, 100))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(construct, offsets))

    for result, offset in zip(results, offsets, strict=True):
        expected, component_count, unique_count = _legacy_component_arrays(
            num_atoms,
            endpoints,
            masks,
            global_bonds,
            atom_offset=offset,
            mask_offset=offset * 2,
            component_offset=offset * 3,
        )
        for actual, reference in zip(result[:9], expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
        assert result[9:11] == (component_count, unique_count)
        np.testing.assert_array_equal(result[11], results[0][11])
        np.testing.assert_array_equal(result[12], results[0][12])
        assert result[13] == len(masks)
    for actual, original in zip(mask_csr, original_csr, strict=True):
        np.testing.assert_array_equal(actual, original)


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
        _mask_csr(broken_bond_sets),
        np.arange(num_atoms - 1, dtype=np.int64),
        0,
        0,
        0,
    )

    # Singleton atoms 64 and 65 have identical low 64-bit words, while the
    # repeated final mask must not add new unique components.
    assert actual[9] == 9
    assert actual[10] == 5


def test_fused_component_arrays_maps_duplicate_paths_to_first_representative() -> None:
    actual = build_component_arrays_raw(
        3,
        np.asarray([(0, 1), (1, 2)], dtype=np.int64),
        _mask_csr([(), (0,), (0, 1)]),
        np.asarray([0, 1], dtype=np.int64),
        0,
        0,
        0,
    )

    assert actual[11].tolist() == [0, 1, 2, 1, 3, 4]
    assert actual[12].tolist() == [0, 1, 2, 4, 5]


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
        _mask_csr(broken_bond_sets),
        np.arange(500, 500 + num_atoms - 1, dtype=np.int64),
        1000,
        20,
        30,
    )
    assert actual[9] == 10
    assert actual[10] == 8


@pytest.mark.parametrize(
    ("target", "expected_retained", "expected_scanned"),
    [
        (1, 1, 2),
        (2, 2, 2),
        (3, 2, 4),
        (4, 4, 4),
        (5, 4, 4),
        (6, 4, 4),
    ],
)
def test_early_topology_matches_full_slice_across_budget_boundaries(
    target: int,
    expected_retained: int,
    expected_scanned: int,
) -> None:
    endpoints = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
    masks = [(), (0,), (0,), (1,)]

    actual = _assert_early_build_matches_full_slice(
        4,
        endpoints,
        masks,
        target,
        0,
        atom_offset=7,
        mask_offset=11,
        component_offset=13,
    )

    assert actual[0].size == expected_retained
    assert actual[13] == expected_scanned


@pytest.mark.parametrize("forced_mask_count", [2, 3, 4])
def test_early_topology_finishes_forced_prefix(
    forced_mask_count: int,
) -> None:
    endpoints = np.asarray([(0, 1), (1, 2), (2, 3)], dtype=np.int64)
    masks = [(), (0,), (1,), (2,), (0, 2)]

    actual = _assert_early_build_matches_full_slice(
        4,
        endpoints,
        masks,
        target=1,
        forced_mask_count=forced_mask_count,
    )

    assert actual[0].size == forced_mask_count
    assert actual[13] == forced_mask_count


def test_early_topology_retains_all_masks_when_target_is_reached_exactly() -> None:
    endpoints = np.asarray([(0, 1), (1, 2)], dtype=np.int64)
    masks = [(), (), (0,), (0,)]

    actual = _assert_early_build_matches_full_slice(
        3,
        endpoints,
        masks,
        target=3,
        forced_mask_count=1,
    )

    assert actual[0].size == len(masks)
    assert actual[13] == len(masks)


def test_early_topology_handles_zero_bond_graph_and_duplicate_empty_masks() -> None:
    actual = _assert_early_build_matches_full_slice(
        1,
        np.empty((0, 2), dtype=np.int64),
        [(), (), ()],
        target=1,
        forced_mask_count=1,
    )

    assert actual[0].tolist() == [0, 0, 0]
    assert actual[13] == 3


def test_early_topology_preserves_redundant_ring_and_multigraph_boundaries() -> None:
    endpoints = np.asarray(
        [(0, 1), (0, 1), (1, 2), (2, 0)],
        dtype=np.int64,
    )
    masks = [(), (0,), (1,), (0, 1), (2, 3), (0, 2, 3)]

    _assert_early_build_matches_full_slice(
        3,
        endpoints,
        masks,
        target=2,
        forced_mask_count=3,
        atom_offset=5,
        mask_offset=9,
        component_offset=12,
    )


def test_early_topology_preserves_multiword_signatures_above_256_atoms() -> None:
    num_atoms = 257
    endpoints = np.asarray(
        [(atom, atom + 1) for atom in range(num_atoms - 1)],
        dtype=np.int64,
    )
    masks = [(), (255,), (255,), (128,), (64, 128, 255)]

    actual = _assert_early_build_matches_full_slice(
        num_atoms,
        endpoints,
        masks,
        target=3,
        forced_mask_count=1,
        atom_offset=1000,
        mask_offset=20,
        component_offset=30,
    )

    assert 1000 + 256 in actual[3]


@pytest.mark.parametrize("seed", range(12))
def test_early_topology_matches_full_slice_on_random_multigraphs(seed: int) -> None:
    rng = np.random.default_rng(48271 + seed)
    num_atoms = int(rng.integers(1, 20))
    num_bonds = int(rng.integers(0, 3 * num_atoms + 1))
    endpoints = rng.integers(
        0,
        num_atoms,
        size=(num_bonds, 2),
        dtype=np.int64,
    )
    masks: list[tuple[int, ...]] = [()]
    for _ in range(20):
        cut_count = int(rng.integers(0, num_bonds + 1))
        masks.append(tuple(sorted(rng.choice(num_bonds, size=cut_count, replace=False).tolist())))
    target = int(rng.integers(1, num_atoms * 4 + 1))
    forced_mask_count = int(rng.integers(0, len(masks) + 1))

    _assert_early_build_matches_full_slice(
        num_atoms,
        endpoints,
        masks,
        target,
        forced_mask_count,
        atom_offset=17,
        mask_offset=23,
        component_offset=31,
    )


@pytest.mark.parametrize(
    ("target", "forced_mask_count", "message"),
    [
        (0, 0, "max_unique_components must be positive"),
        (-1, 0, "max_unique_components must be positive"),
        (1, -1, "forced_mask_count must be in"),
        (1, 3, "forced_mask_count must be in"),
    ],
)
def test_early_topology_validates_budget_arguments(
    target: int,
    forced_mask_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_component_arrays_raw(
            2,
            np.asarray([(0, 1)], dtype=np.int64),
            _mask_csr([(), (0,)]),
            np.asarray([0], dtype=np.int64),
            0,
            0,
            0,
            target,
            forced_mask_count,
        )


@pytest.mark.parametrize(
    (
        "logits",
        "max_masks",
        "include_single_bond_cuts",
    ),
    [
        ([0.0, 0.0, 0.0, 0.0], 5, True),
        ([1.0, 0.5], 4, True),
        ([2.0, 2.0, 2.0], 1, True),
        ([-2.0, -3.0, -4.0], 3, False),
    ],
)
def test_no_exploration_numeric_path_matches_legacy_selection_and_assembly(
    logits: list[float],
    max_masks: int,
    include_single_bond_cuts: bool,
) -> None:
    num_bonds = len(logits)
    num_atoms = num_bonds + 1
    bond_endpoints = np.asarray(
        [(atom, atom + 1) for atom in range(num_bonds)],
        dtype=np.int64,
    )

    learned_masks = []
    for ranked_mask in iter_ranked_cut_masks(logits, include_empty=True):
        learned_masks.append(ranked_mask.broken_bond_indices)
        if len(learned_masks) == max_masks:
            break
    selected_masks = set(learned_masks)
    expected_masks = list(learned_masks)
    forced_masks = [()]
    if include_single_bond_cuts:
        forced_masks.extend((bond_index,) for bond_index in range(num_bonds))
    expected_masks.extend(mask for mask in forced_masks if mask not in selected_masks)

    expected_arrays, expected_component_count, expected_unique_count = _legacy_component_arrays(
        num_atoms,
        bond_endpoints,
        expected_masks,
        np.arange(num_bonds, dtype=np.int64),
    )
    bond_logits = th.tensor(logits, dtype=th.float64, requires_grad=True)
    candidates = build_masster_candidates(
        bond_logits,
        chemical_bond_index=th.from_numpy(bond_endpoints.T.copy()),
        atom_batch_indices=th.zeros(num_atoms, dtype=th.long),
        atom_ptr=th.tensor([0, num_atoms], dtype=th.long),
        max_masks=max_masks,
        include_single_bond_cuts=include_single_bond_cuts,
    )

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
    assert candidates.proposal_single_cut_fraction.tolist() == pytest.approx(
        [sum(len(mask) == 1 for mask in learned_masks) / len(learned_masks)]
    )
    assert candidates.unique_component_count.tolist() == [float(expected_unique_count)]
    assert candidates.num_components == expected_component_count
