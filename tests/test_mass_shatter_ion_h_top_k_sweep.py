"""Tests for checkpoint-only MASShatter ion/H top-k sweeps."""

import pytest

from scripts.eval.mass_shatter_ion_h_top_k_sweep import (
    validate_joint_relative_thresholds,
    validate_top_ks,
)


def test_validate_top_ks_keeps_dense_reference_and_requested_order() -> None:
    assert validate_top_ks([0, 1, 2, 3]) == [0, 1, 2, 3]


@pytest.mark.parametrize("top_ks", [[], [-1, 0], [0, 1, 1], [1, 2]])
def test_validate_top_ks_rejects_invalid_values(top_ks: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_top_ks(top_ks)


def test_validate_joint_relative_thresholds_keeps_requested_order() -> None:
    assert validate_joint_relative_thresholds([0.001, 0.01, 0.1]) == [
        0.001,
        0.01,
        0.1,
    ]


@pytest.mark.parametrize("thresholds", [[0.0], [-0.1], [1.1], [0.1, 0.1]])
def test_validate_joint_relative_thresholds_rejects_invalid_values(
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError):
        validate_joint_relative_thresholds(thresholds)
