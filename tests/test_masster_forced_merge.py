"""Differential and cache tests for MASSter forced-mask merging."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
import torch as th

from fragnnet.frag import masster_candidates
from fragnnet.frag.masster_templates import MassterTemplate


def _pack_masks(masks: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    cut_counts = np.asarray([len(mask) for mask in masks], dtype=np.int64)
    mask_ptr = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(cut_counts, dtype=np.int64))
    )
    broken_bonds = np.asarray(
        [bond for mask in masks for bond in mask],
        dtype=np.int64,
    )
    return mask_ptr, broken_bonds


def _unpack_masks(
    mask_ptr: np.ndarray,
    broken_bonds: np.ndarray,
) -> list[tuple[int, ...]]:
    return [
        tuple(broken_bonds[mask_ptr[index] : mask_ptr[index + 1]])
        for index in range(mask_ptr.size - 1)
    ]


def _reference_merge(
    ranked_masks: Sequence[tuple[int, ...]],
    forced_masks: Sequence[tuple[int, ...]],
    unique_fragment_mode: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Express the forced-mask contract without using the optimized path."""
    ranked_set = set(ranked_masks)
    forced_set = set(forced_masks)
    if unique_fragment_mode:
        merged = [*forced_masks, *(mask for mask in ranked_masks if mask not in forced_set)]
        forced_mask_count = len(forced_masks)
    else:
        merged = [*ranked_masks, *(mask for mask in forced_masks if mask not in ranked_set)]
        forced_mask_count = 0
    mask_ptr, broken_bonds = _pack_masks(merged)
    return mask_ptr, broken_bonds, forced_mask_count


def _run_merge(
    ranked_masks: Sequence[tuple[int, ...]],
    base_forced_masks: Sequence[tuple[int, ...]],
    template_masks: Sequence[tuple[int, ...]],
    unique_fragment_mode: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    forced_masks = list(dict.fromkeys([*base_forced_masks, *template_masks]))
    mask_ptr, broken_bonds = _pack_masks(ranked_masks)
    original_mask_ptr = mask_ptr.copy()
    original_broken_bonds = broken_bonds.copy()

    actual = masster_candidates._merge_forced_masks(
        mask_ptr,
        broken_bonds,
        forced_masks,
        np.diff(mask_ptr),
        len(ranked_masks),
        len(base_forced_masks),
        unique_fragment_mode,
    )

    np.testing.assert_array_equal(mask_ptr, original_mask_ptr)
    np.testing.assert_array_equal(broken_bonds, original_broken_bonds)
    assert actual[0].dtype == np.int64
    assert actual[1].dtype == np.int64
    assert actual[0].flags.c_contiguous
    assert actual[1].flags.c_contiguous
    return actual


@pytest.mark.parametrize("seed", range(100))
def test_filtered_merge_matches_reference_on_random_ranked_pools(seed: int) -> None:
    rng = np.random.default_rng(83047 + seed)
    num_bonds = int(rng.integers(0, 9))
    all_masks = [
        tuple(bond for bond in range(num_bonds) if bits & (1 << bond))
        for bits in range(1 << num_bonds)
    ]
    ranked_order = rng.permutation(len(all_masks))
    ranked_count = int(rng.integers(1, len(all_masks) + 1))
    ranked_masks = [all_masks[int(index)] for index in ranked_order[:ranked_count]]
    base_forced_masks: list[tuple[int, ...]] = [()]
    if bool(rng.integers(0, 2)):
        base_forced_masks.extend((bond,) for bond in range(num_bonds))
    template_count = int(rng.integers(0, min(8, len(all_masks)) + 1))
    template_masks = [
        all_masks[int(index)]
        for index in rng.integers(0, len(all_masks), size=template_count)
    ]
    forced_masks = list(dict.fromkeys([*base_forced_masks, *template_masks]))
    unique_fragment_mode = bool(rng.integers(0, 2))

    actual = _run_merge(
        ranked_masks,
        base_forced_masks,
        template_masks,
        unique_fragment_mode,
    )
    expected = _reference_merge(ranked_masks, forced_masks, unique_fragment_mode)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def _small_ranked_pool(order: int) -> list[tuple[int, ...]]:
    masks = [
        tuple(bond for bond in range(3) if bits & (1 << bond))
        for bits in range(8)
    ]
    if order == 0:
        return masks
    if order == 1:
        return masks[::-1]
    if order == 2:
        return masks[::2]
    return [*masks[3:], *masks[:3]][:5]


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
@pytest.mark.parametrize("include_single_cuts", [False, True])
@pytest.mark.parametrize("ranked_order", range(4))
@pytest.mark.parametrize("template_bits", range(8))
def test_filtered_merge_matches_reference_exhaustively_on_three_bonds(
    unique_fragment_mode: bool,
    include_single_cuts: bool,
    ranked_order: int,
    template_bits: int,
) -> None:
    ranked_masks = _small_ranked_pool(ranked_order)
    base_forced_masks: list[tuple[int, ...]] = [()]
    if include_single_cuts:
        base_forced_masks.extend((bond,) for bond in range(3))
    template_mask = tuple(bond for bond in range(3) if template_bits & (1 << bond))
    template_masks = [template_mask]
    forced_masks = list(dict.fromkeys([*base_forced_masks, *template_masks]))

    actual = _run_merge(
        ranked_masks,
        base_forced_masks,
        template_masks,
        unique_fragment_mode,
    )
    expected = _reference_merge(ranked_masks, forced_masks, unique_fragment_mode)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_filtered_merge_handles_bondless_ranked_pool(unique_fragment_mode: bool) -> None:
    actual = _run_merge([()], [()], [()], unique_fragment_mode)

    assert _unpack_masks(actual[0], actual[1]) == [()]
    assert actual[2] == (1 if unique_fragment_mode else 0)


@pytest.mark.parametrize(
    ("candidate_groups", "expected_calls"),
    [
        (None, 4),
        ([5, 5, 5, 5], 1),
        ([5, 5, 9, 9], 2),
    ],
)
@pytest.mark.parametrize("unique_fragment_mode", [False, True])
@pytest.mark.parametrize("with_smarts", [False, True])
def test_grouped_views_reuse_forced_merge_by_signature(
    monkeypatch: pytest.MonkeyPatch,
    candidate_groups: list[int] | None,
    expected_calls: int,
    unique_fragment_mode: bool,
    with_smarts: bool,
) -> None:
    original_merge = masster_candidates._merge_forced_masks
    calls = 0

    def counted_merge(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(masster_candidates, "_merge_forced_masks", counted_merge)
    chemical_bond_index = th.tensor(
        [[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
         [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]],
        dtype=th.long,
    )
    template = MassterTemplate(
        broken_bond_indices=(0, 2),
        components=((0, 3), (1, 2)),
        rule_index=3,
        rule_name="cache_test",
    )
    kwargs = {
        "chemical_bond_index": chemical_bond_index,
        "atom_batch_indices": th.arange(4).repeat_interleave(4),
        "atom_ptr": th.tensor([0, 4, 8, 12, 16], dtype=th.long),
        "max_masks": 4,
        "include_single_bond_cuts": True,
        "smarts_templates": tuple((template,) for _ in range(4)) if with_smarts else None,
    }
    if unique_fragment_mode:
        kwargs.update(max_unique_components=4, max_enumerated_masks=16)
    if candidate_groups is not None:
        kwargs["candidate_group_ids"] = th.tensor(candidate_groups, dtype=th.long)

    masster_candidates.build_masster_candidates(
        th.tensor([0.6, -0.2, 0.1] * 4),
        **kwargs,
    )

    assert calls == expected_calls
