"""Parity tests for the numeric MASSter top-k mask kernel."""

from __future__ import annotations

import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from fragnnet.models.masster.candidates.cut_mask_kernels import (
    top_k_cut_masks_csr_raw,
    top_mixture_bond_sets_raw,
)
from fragnnet.models.masster.candidates.reference_cut_masks import iter_ranked_cut_masks


def _decode_masks(mask_ptr: np.ndarray, broken_bonds: np.ndarray) -> list[tuple[int, ...]]:
    return [
        tuple(broken_bonds[mask_ptr[index] : mask_ptr[index + 1]].tolist())
        for index in range(mask_ptr.size - 1)
    ]


@pytest.mark.parametrize("include_empty", [False, True])
@pytest.mark.parametrize("num_bonds", range(9))
def test_numeric_top_k_matches_python_enumerator(
    num_bonds: int,
    include_empty: bool,
) -> None:
    rng = np.random.default_rng(9183 + num_bonds)
    logits = rng.normal(size=num_bonds) + np.arange(num_bonds) * 1e-4
    k = 40
    expected = list(
        itertools.islice(
            iter_ranked_cut_masks(logits, include_empty=include_empty),
            k,
        )
    )

    mask_ptr, broken_bonds, log_probabilities, relative_probabilities = (
        top_k_cut_masks_csr_raw(logits, k, include_empty, 0.0)
    )

    assert mask_ptr.dtype == np.int64
    assert broken_bonds.dtype == np.int64
    assert log_probabilities.dtype == np.float64
    assert relative_probabilities.dtype == np.float64
    assert mask_ptr[0] == 0
    assert mask_ptr[-1] == broken_bonds.size
    assert np.all(mask_ptr[1:] >= mask_ptr[:-1])
    assert _decode_masks(mask_ptr, broken_bonds) == [
        mask.broken_bond_indices for mask in expected
    ]
    np.testing.assert_array_equal(
        log_probabilities,
        [mask.log_probability for mask in expected],
    )
    if expected:
        expected_relative = np.asarray(
            [mask.log_probability_from_map for mask in expected],
            dtype=np.float64,
        )
        if _decode_masks(mask_ptr, broken_bonds)[0] == tuple(
            index for index, logit in enumerate(logits) if logit > 0.0
        ):
            expected_relative[0] = 0.0
        np.testing.assert_array_equal(relative_probabilities, expected_relative)


def test_numeric_top_k_preserves_zero_cost_tie_order_and_signed_zero() -> None:
    mask_ptr, broken_bonds, _, relative_probabilities = top_k_cut_masks_csr_raw(
        np.asarray([0.0, -0.0, 0.0, -0.0])[::2],
        8,
        True,
        0.0,
    )

    assert _decode_masks(mask_ptr, broken_bonds) == [(), (0,), (0, 1), (1,)]
    assert not np.signbit(relative_probabilities[0])
    assert np.signbit(relative_probabilities[1:]).all()


def test_numeric_top_k_respects_floor_and_handles_k_beyond_state_space() -> None:
    logits = [math.log(4.0), math.log(1.5)]
    mask_ptr, broken_bonds, log_probabilities, _ = top_k_cut_masks_csr_raw(
        logits,
        100,
        True,
        0.2,
    )

    assert _decode_masks(mask_ptr, broken_bonds) == [(0, 1), (0,)]
    np.testing.assert_allclose(np.exp(log_probabilities), [0.48, 0.32])


@pytest.mark.parametrize("num_bonds", [0, 1, 7, 65, 129])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("limit", [1, 257])
def test_single_mixture_without_floors_matches_python_ranked_prefix(
    num_bonds: int, tied: bool, limit: int
) -> None:
    """The selector preserves heap ties and multiword mask identities."""
    rng = np.random.default_rng(218 + num_bonds)
    logits = (
        np.resize(np.asarray([0.0, -0.0, 0.5, -0.5]), num_bonds)
        if tied
        else rng.normal(size=num_bonds)
    )
    expected = [
        mask.broken_bond_indices for mask in itertools.islice(iter_ranked_cut_masks(logits), limit)
    ]

    ptr, bonds = top_mixture_bond_sets_raw(logits.reshape(-1, 1), np.zeros(1), limit, 0.0, 0.0)

    assert ptr.dtype == bonds.dtype == np.int64
    assert ptr[0] == 0 and ptr[-1] == bonds.size
    assert _decode_masks(ptr, bonds) == expected


@pytest.mark.parametrize(
    ("absolute_floor", "relative_floor", "expected"),
    [
        (0.2, 0.0, [(0, 1), (0,)]),
        (0.0, 0.5, [(0, 1), (0,)]),
        (0.1, 0.0, [(0, 1), (0,), (1,)]),
        (1.0, 0.0, []),
        (0.0, 1.0, [(0, 1)]),
    ],
)
def test_single_mixture_keeps_enabled_probability_floors(
    absolute_floor: float, relative_floor: float, expected: list[tuple[int, ...]]
) -> None:
    # The four mask probabilities are 0.48, 0.32, 0.12, and 0.08.
    logits = np.asarray([[math.log(4.0)], [math.log(1.5)]])
    ptr, bonds = top_mixture_bond_sets_raw(logits, np.zeros(1), 100, absolute_floor, relative_floor)

    assert _decode_masks(ptr, bonds) == expected


@pytest.mark.parametrize(
    ("logits", "weight", "limit"),
    [
        ([1e308, 1e308], 0.0, 4),
        ([-1e308, -1e308], 0.0, 4),
        ([1e308, -1e308], 0.0, 4),
        ([-1e308, -1e308], -1e308, 4),
        ([1e308, 1e308], 1e308, 4),
        ([1.0, -1.0], np.finfo(np.float64).max, 4),
        ([], -np.finfo(np.float64).max, 4),
        ([np.finfo(np.float32).max, -np.finfo(np.float32).max], 0.0, 4),
        ([1.0, np.nextafter(1.0, 2.0), -1.0], 0.0, 8),
    ],
)
def test_single_mixture_retains_legacy_extreme_float64_filtering(
    logits: list[float], weight: float, limit: int
) -> None:
    """Finite inputs may overflow old scores; disabled floors still reject NaN."""
    masks = [
        mask.broken_bond_indices for mask in itertools.islice(iter_ranked_cut_masks(logits), limit)
    ]
    intact_log_probability = 0.0
    for logit in logits:
        intact_log_probability -= (
            logit + math.log1p(math.exp(-logit)) if logit > 0.0 else math.log1p(math.exp(logit))
        )
    scores = []
    for mask in masks:
        score = intact_log_probability
        for bond in mask:
            score += logits[bond]
        scores.append(score + float(weight))
    expected = [
        mask
        for mask, score in zip(masks, scores)
        if score >= -math.inf and score - scores[0] >= -math.inf
    ]

    ptr, bonds = top_mixture_bond_sets_raw(
        np.asarray(logits, dtype=np.float64).reshape(-1, 1),
        np.asarray([weight]),
        limit,
        0.0,
        0.0,
    )

    assert _decode_masks(ptr, bonds) == expected


@pytest.mark.parametrize("threads", [2, 4])
def test_single_mixture_threaded_csr_matches_independent_ranked_prefix(threads: int) -> None:
    rng = np.random.default_rng(248)
    logits_by_molecule = [rng.normal(size=(size, 1)) for size in [0, 1, 6, 30, 65, 129]]
    weights = np.zeros(1)

    def select(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return top_mixture_bond_sets_raw(logits, weights, 257, 0.0, 0.0)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        outputs = list(pool.map(select, logits_by_molecule))

    for logits, (ptr, bonds) in zip(logits_by_molecule, outputs):
        expected = [
            mask.broken_bond_indices
            for mask in itertools.islice(iter_ranked_cut_masks(logits[:, 0]), 257)
        ]
        assert _decode_masks(ptr, bonds) == expected
