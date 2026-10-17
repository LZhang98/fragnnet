"""Check accurate-mass scoring and peak aggregation on known spectrum pairs."""

import numpy as np
import pytest

from scripts.debug.ei.evaluate_hr_ei_checkpoint import aggregate, score


def test_same_formula_contributions_are_summed():
    mz, weights = aggregate(np.array([43.01839, 43.01839, 50.0]), np.array([0.2, 0.3, 0.5]))
    assert mz == pytest.approx([43.01839, 50.0])
    assert weights == pytest.approx([0.5, 0.5])


def test_exact_and_nominal_are_different():
    pm, pi = np.array([43.0184]), np.array([1.0])
    tm, ti = np.array([43.0542]), np.array([1.0])
    assert score(pm, pi, tm, ti, 5, 0.0005)["cosine"] == 0
    assert score(np.rint(pm), pi, np.rint(tm), ti, 0, 0)["cosine"] == 1


def test_assignment_is_one_to_one_and_keeps_unmatched_norm():
    result = score(
        np.array([100.0, 100.0002, 200.0]), np.ones(3), np.array([100.0001]), np.ones(1), 5, 0.0005
    )
    assert result["cosine"] == pytest.approx(1 / np.sqrt(3))
    assert result["target_intensity_recall"] == 1


def test_ppm_and_floor_tolerances():
    args = (np.array([500.004]), np.ones(1), np.array([500.0]), np.ones(1))
    assert score(*args, 5, 0.0005)["cosine"] == 0
    assert score(*args, 10, 0.001)["cosine"] == 1
    assert (
        score(np.array([40.0004]), np.ones(1), np.array([40.0]), np.ones(1), 5, 0.0005)["cosine"]
        == 1
    )


def test_nominal_comparison_tolerates_float_roundoff():
    result = score(np.array([69.0]), np.ones(1), np.array([69.00000000000001]), np.ones(1), 0, 0)
    assert result["cosine"] == 1
