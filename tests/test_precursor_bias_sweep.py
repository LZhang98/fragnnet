"""Tests for the post-hoc precursor bias diagnostic."""

import numpy as np
import pytest

from scripts.debug.masster_memorization.precursor_bias_sweep import Spectrum, evaluate_spectrum


def _spectrum(true_ints: list[float], pred_ints: list[float]) -> Spectrum:
    return Spectrum(
        mol_id=1,
        true_mzs=np.array([100.0, 150.0]),
        true_ints=np.asarray(true_ints),
        pred_mzs=np.array([100.0, 150.0]),
        pred_ints=np.asarray(pred_ints),
        prec_mz=150.0,
    )


def test_positive_bias_can_correct_precursor_overcall() -> None:
    spectrum = _spectrum(true_ints=[80.0, 20.0], pred_ints=[40.0, 60.0])

    baseline = evaluate_spectrum(spectrum, bias=0.0)
    corrected = evaluate_spectrum(spectrum, bias=1.0)

    assert baseline["pred_base_is_prec"]
    assert not baseline["base_hit"]
    assert not corrected["pred_base_is_prec"]
    assert corrected["base_hit"]
    assert corrected["cos_hun"] > baseline["cos_hun"]


def test_zero_bias_is_identity() -> None:
    spectrum = _spectrum(true_ints=[25.0, 75.0], pred_ints=[25.0, 75.0])

    metrics = evaluate_spectrum(spectrum, bias=0.0)

    assert metrics["true_base_is_prec"]
    assert metrics["pred_base_is_prec"]
    assert metrics["base_hit"]
    assert metrics["mean_mz_error"] == pytest.approx(0.0)
    assert metrics["cos_hun"] == pytest.approx(1.0)
