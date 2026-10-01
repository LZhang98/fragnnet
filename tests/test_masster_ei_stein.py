"""Regression checks for EI Stein weighted similarity."""

import numpy as np
import pytest

from scripts.debug.analyze_masster_ei_checkpoint import _spectrum_rows, _stein_similarity


def test_stein_mass_weight_and_alignment() -> None:
    pred_mzs = np.array([10, 20])
    pred_intensities = np.array([1.0, 1.0])
    target_mzs = np.array([10, 20])
    target_intensities = np.array([1.0, 0.0])
    expected = 1 / np.sqrt(1 + 8**2)
    assert _stein_similarity(
        pred_mzs, pred_intensities, target_mzs, target_intensities
    ) == pytest.approx(expected)
    assert _stein_similarity(
        pred_mzs, pred_intensities, pred_mzs, pred_intensities
    ) == pytest.approx(1.0)
    assert _stein_similarity(
        np.array([30]), np.array([1.0]), target_mzs, target_intensities
    ) == 0.0


def test_stein_intensity_power_and_score_only_path() -> None:
    pred_mzs = np.array([10, 20])
    pred_intensities = np.array([0.2, 0.8])
    target_mzs = np.array([10, 20])
    target_intensities = np.array([0.5, 0.5])
    pred_vector = pred_mzs**3 * pred_intensities**0.6
    target_vector = target_mzs**3 * target_intensities**0.6
    expected = np.dot(pred_vector, target_vector) / (
        np.linalg.norm(pred_vector) * np.linalg.norm(target_vector)
    )
    rows, info = _spectrum_rows(
        7, pred_mzs, pred_intensities, np.array([0, 1]), np.array([0, 0]),
        target_mzs, target_intensities, include_peak_rows=False,
    )
    assert rows == []
    assert info["stein_full"] == pytest.approx(expected)
    assert info["cosine_full"] != pytest.approx(expected)
