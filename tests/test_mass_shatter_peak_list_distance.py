"""Tests for direct, non-Gaussian MASShatter peak-list objectives."""

import pytest
import torch as th

from fragnnet.model.loss import sparse_peak_list_distance


@pytest.mark.parametrize(
    "objective",
    ["cosine_distance", "jss_distance", "kl_distance"],
)
@pytest.mark.parametrize("matching", ["closest", "hungarian"])
def test_identical_spectra_have_zero_loss(objective: str, matching: str) -> None:
    mzs = th.tensor([100.0, 150.0])
    intensities = th.tensor([1.0, 3.0])
    batch_indices = th.zeros(2, dtype=th.long)

    primary, entropy = sparse_peak_list_distance(
        mzs,
        intensities,
        batch_indices,
        mzs,
        intensities,
        batch_indices,
        objective=objective,
        matching=matching,
        tolerance=0.01,
        relative=False,
    )

    assert th.allclose(primary, th.zeros(1), atol=1e-7)
    assert th.allclose(entropy, th.zeros(1), atol=1e-7)


def test_closest_matching_sums_many_predictions_at_one_observed_peak() -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([3.0, 1.0])
    pred_mzs = th.tensor([99.996, 100.004, 150.0])
    pred_intensities = th.tensor([1.0, 2.0, 1.0], requires_grad=True)
    true_batch_indices = th.zeros(2, dtype=th.long)
    pred_batch_indices = th.zeros(3, dtype=th.long)

    primary, _ = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        true_batch_indices,
        pred_mzs,
        pred_intensities,
        pred_batch_indices,
        objective="cosine_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
    )
    primary.sum().backward()

    assert th.allclose(primary, th.zeros(1), atol=1e-7)
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


def test_hungarian_is_one_to_one_and_mz_only() -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([1.0, 1.0])
    pred_mzs = th.tensor([99.999, 100.002, 150.0])
    pred_batch_indices = th.zeros(3, dtype=th.long)
    true_batch_indices = th.zeros(2, dtype=th.long)

    low_first, _ = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        true_batch_indices,
        pred_mzs,
        th.tensor([0.1, 100.0, 1.0]),
        pred_batch_indices,
        objective="cosine_distance",
        matching="hungarian",
        tolerance=0.01,
        relative=False,
    )
    high_first, _ = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        true_batch_indices,
        pred_mzs,
        th.tensor([100.0, 0.1, 1.0]),
        pred_batch_indices,
        objective="cosine_distance",
        matching="hungarian",
        tolerance=0.01,
        relative=False,
    )

    assert low_first > high_first


def test_four_decimal_clustering_sums_intensity_and_keeps_gradient() -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([3.0, 1.0])
    pred_mzs = th.tensor([100.00001, 100.00002, 150.0])
    pred_intensities = th.tensor([1.0, 2.0, 1.0], requires_grad=True)
    true_batch_indices = th.zeros(2, dtype=th.long)
    pred_batch_indices = th.zeros(3, dtype=th.long)

    primary, _ = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        true_batch_indices,
        pred_mzs,
        pred_intensities,
        pred_batch_indices,
        objective="jss_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
        decimals=4,
    )
    primary.sum().backward()

    assert th.allclose(primary, th.zeros(1), atol=1e-7)
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


@pytest.mark.parametrize("matching", ["closest", "hungarian"])
def test_zero_entries_are_valid_and_negative_entries_are_rejected(matching: str) -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([1.0, 1.0])
    pred_mzs = th.tensor([100.0, 150.0])
    batch_indices = th.zeros(2, dtype=th.long)

    primary, entropy = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        batch_indices,
        pred_mzs,
        th.tensor([1.0, 0.0]),
        batch_indices,
        objective="jss_distance",
        matching=matching,
        tolerance=0.01,
        relative=False,
    )
    assert th.isfinite(primary).all()
    assert th.isfinite(entropy).all()

    with pytest.raises(ValueError, match="nonnegative"):
        sparse_peak_list_distance(
            true_mzs,
            true_intensities,
            batch_indices,
            pred_mzs,
            th.tensor([1.0, -0.1]),
            batch_indices,
            objective="cosine_distance",
            matching=matching,
            tolerance=0.01,
            relative=False,
        )


def test_unmatched_prediction_is_retained_and_suppressed() -> None:
    pred_intensities = th.tensor([1.0, 1.0], requires_grad=True)
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0]),
        th.tensor([1.0]),
        th.tensor([0]),
        th.tensor([100.0, 200.0]),
        pred_intensities,
        th.tensor([0, 0]),
        objective="cosine_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
    )
    primary.sum().backward()

    assert primary.item() > 0
    assert pred_intensities.grad is not None
    assert pred_intensities.grad[1].item() > 0


def test_kl_smoothing_keeps_missing_predicted_support_finite() -> None:
    pred_intensities = th.tensor([1.0, 0.0], requires_grad=True)
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0, 150.0]),
        th.tensor([0.5, 0.5]),
        th.tensor([0, 0]),
        th.tensor([100.0, 150.0]),
        pred_intensities,
        th.tensor([0, 0]),
        objective="kl_distance",
        matching="hungarian",
        tolerance=0.01,
        relative=False,
        kl_epsilon=1e-8,
    )

    assert th.isfinite(primary).all()
    assert primary.item() > 0.0
    primary.sum().backward()
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


@pytest.mark.parametrize(
    "objective", ["cosine_distance", "jss_distance", "kl_distance"]
)
def test_instrument_tolerance_is_passed_to_peak_matching(objective: str) -> None:
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0, 100.0]),
        th.ones(2),
        th.tensor([0, 1]),
        th.tensor([100.003, 100.003]),
        th.ones(2),
        th.tensor([0, 1]),
        objective=objective,
        matching="hungarian",
        tol_per_sample=th.tensor([1.0e-5, 2.0e-5]),
        min_mz_per_sample=th.tensor([200.0, 200.0]),
    )

    assert primary[0] > 0.0
    assert primary[1] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("pred_dtype", [th.float16, th.bfloat16])
def test_cosine_accepts_mixed_precision_predictions(pred_dtype: th.dtype) -> None:
    pred_intensities = th.tensor([1.0, 2.0], dtype=pred_dtype, requires_grad=True)
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0, 150.0]),
        th.tensor([1.0, 2.0]),
        th.tensor([0, 0]),
        th.tensor([100.0, 150.0]),
        pred_intensities,
        th.tensor([0, 0]),
        objective="cosine_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
    )
    primary.sum().backward()

    assert th.allclose(primary, th.zeros(1), atol=1e-7)
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()
