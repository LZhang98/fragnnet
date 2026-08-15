"""Tests for checkpoint-only MASShatter mask-budget sweeps."""

import pytest
import torch as th

from scripts.eval.mass_shatter_mask_budget_sweep import (
    WeightedMetricAccumulator,
    sample_weights,
    validate_mask_budgets,
)


def test_validate_mask_budgets_keeps_requested_order() -> None:
    assert validate_mask_budgets([100, 256, 512]) == [100, 256, 512]


@pytest.mark.parametrize("mask_budgets", [[], [0], [-1, 100], [100, 100]])
def test_validate_mask_budgets_rejects_invalid_values(mask_budgets: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_mask_budgets(mask_budgets)


def test_sample_weights_match_epoch_weighting_modes() -> None:
    spec_per_group = th.tensor([2.0, 4.0])
    spec_per_mol = th.tensor([5.0, 10.0])
    group_per_mol = th.tensor([3.0, 2.0])

    assert sample_weights("none", spec_per_group, spec_per_mol, group_per_mol).tolist() == [
        1.0,
        1.0,
    ]
    assert sample_weights("group", spec_per_group, spec_per_mol, group_per_mol).tolist() == [
        0.5,
        0.25,
    ]
    assert sample_weights("mol", spec_per_group, spec_per_mol, group_per_mol).tolist() == [
        0.2,
        0.1,
    ]
    assert sample_weights(
        "group_mol", spec_per_group, spec_per_mol, group_per_mol
    ).tolist() == pytest.approx([1 / 6, 1 / 8])


def test_metric_accumulator_computes_global_weighted_mean() -> None:
    accumulator = WeightedMetricAccumulator(("recall", "candidate_wrecall"))
    accumulator.update(
        {
            "spec_per_group": th.tensor([1.0, 2.0]),
            "spec_per_mol": th.tensor([1.0, 2.0]),
            "group_per_mol": th.ones(2),
            "recall": th.tensor([0.2, 0.8]),
            "candidate_wrecall": th.tensor([0.5, 1.0]),
        },
        weight_mode="group",
    )

    assert accumulator.sample_count == 2
    assert accumulator.means()["recall"] == pytest.approx(0.4)
    assert accumulator.means()["candidate_wrecall"] == pytest.approx(2 / 3)
