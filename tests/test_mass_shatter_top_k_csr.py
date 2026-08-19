"""Parity tests for the numeric MASShatter top-k mask kernel."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from fragnnet.frag.bond_mask_generator import iter_ranked_cut_masks
from fragnnet.frag.mass_shatter_core import top_k_cut_masks_csr_raw


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

