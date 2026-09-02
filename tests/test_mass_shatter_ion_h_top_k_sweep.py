"""Tests for checkpoint-only MASShatter ion/H top-k sweeps."""

import pytest

from scripts.eval.mass_shatter_ion_h_top_k_sweep import (
    build_selections,
    metric_names,
    validate_fragment_relative_thresholds,
    validate_joint_relative_thresholds,
    validate_top_ks,
)


def test_validate_top_ks_keeps_dense_reference_and_requested_order() -> None:
    assert validate_top_ks([0, 1, 2, 3]) == [0, 1, 2, 3]


@pytest.mark.parametrize("top_ks", [[], [-1, 0], [0, 1, 1], [1, 2]])
def test_validate_top_ks_rejects_invalid_values(top_ks: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_top_ks(top_ks)


def test_validate_fragment_relative_thresholds_keeps_requested_order() -> None:
    assert validate_fragment_relative_thresholds([0.0, 0.001, 0.01]) == [0.0, 0.001, 0.01]


@pytest.mark.parametrize("thresholds", [[], [-0.1, 0.0], [0.0, 1.1], [0.0, 0.1, 0.1], [0.1]])
def test_validate_fragment_relative_thresholds_rejects_invalid_values(
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError):
        validate_fragment_relative_thresholds(thresholds)


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


def test_metric_names_only_requests_configured_auxiliary_scores() -> None:
    names = metric_names(
        {
            "auxiliary_scores": ["cos_sim", "precision", "recall"],
            "eval_mz_bin_res": [0.01],
        }
    )

    assert "precision" in names
    assert "recall" in names
    assert "cos_sim_0.01" in names
    assert "wprecision" not in names
    assert "wrecall" not in names
    assert "fragment_count_before_cutoff" in names
    assert "fragment_count_after_cutoff" in names
    assert "retained_fragment_probability" in names
    assert "fragment_cutoff_pruned_fraction" in names


def test_build_selections_crosses_fragment_thresholds_with_top_ks() -> None:
    selections = build_selections(
        fragment_relative_thresholds=[0.0, 0.01],
        top_ks=[0, 4],
        joint_relative_thresholds=[],
    )

    assert selections == [
        ("dense", 0.0, 0, 0.0),
        ("top_k_4", 0.0, 4, 0.0),
        ("fragment_relative_0.01", 0.01, 0, 0.0),
        ("fragment_relative_0.01_top_k_4", 0.01, 4, 0.0),
    ]
