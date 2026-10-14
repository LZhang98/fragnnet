"""Tests for the vectorized ``peak_marginal`` sparse cross-entropy path.

``peak_marginal`` replaces the Gaussian m/z density with a hard tolerance window, so a
candidate inside the window contributes its full log-probability regardless of its m/z
offset. These tests pin that behaviour, its equivalence with the sequential
implementation, and its handling of per-instrument tolerances.
"""

import math

import pytest
import torch as th

from fragnnet.loss.cross_entropy import get_sparse_cross_entropy_fn

TOLERANCE = 1e-5
TOLERANCE_MIN_MZ = 200.0


def _make_fn(dist: str, vectorized: bool, loss_batch_size: int = 8, pm_multiple: float = 1.0):
    return get_sparse_cross_entropy_fn(
        dist=dist,
        vectorized=vectorized,
        tolerance=TOLERANCE,
        relative=True,
        tolerance_min_mz=TOLERANCE_MIN_MZ,
        oos_tolerance_multiple=1.0,
        gaussian_renormalize=False,
        pm_tolerance_multiple=pm_multiple,
        loss_batch_size=loss_batch_size,
    )


def _single_spectrum_inputs() -> tuple:
    """One true peak at m/z 300 with two in-window candidates and one outside.

    The tolerance at m/z 300 is 1e-5 * 300 = 0.003 Da, so 300.0 and 300.001 match while
    300.01 does not.
    """
    true_mzs = th.tensor([300.0], dtype=th.float32)
    true_logprobs = th.tensor([0.0], dtype=th.float32)  # intensity 1.0
    true_batch_idxs = th.tensor([0], dtype=th.long)
    pred_mzs = th.tensor([300.0, 300.001, 300.01], dtype=th.float32)
    pred_logprobs = th.log(th.tensor([0.5, 0.3, 0.2], dtype=th.float32))
    pred_batch_idxs = th.tensor([0, 0, 0], dtype=th.long)
    pred_oos_logprobs = th.tensor([math.log(1e-6)], dtype=th.float32)
    return (
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        pred_oos_logprobs,
    )


def _multi_spectrum_inputs() -> tuple:
    true_mzs = th.tensor([150.0, 300.0, 301.5, 500.0, 520.25], dtype=th.float32)
    true_ints = th.tensor([0.6, 0.4, 0.25, 0.5, 0.25], dtype=th.float32)
    true_logprobs = th.log(true_ints)
    true_batch_idxs = th.tensor([0, 0, 1, 2, 2], dtype=th.long)
    # 300.005 and 500.009 sit at ~1.7 sigma: inside the Gaussian's reach but outside the
    # tolerance window, so they discriminate a truncated mixture from an untruncated one.
    pred_mzs = th.tensor(
        [150.0005, 300.002, 300.005, 412.0, 301.5, 301.502, 499.999, 500.009, 520.25, 900.0],
        dtype=th.float32,
    )
    pred_logprobs = th.log(
        th.tensor(
            [0.25, 0.3, 0.2, 0.25, 0.7, 0.3, 0.4, 0.15, 0.3, 0.15],
            dtype=th.float32,
        )
    )
    pred_batch_idxs = th.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=th.long)
    pred_oos_logprobs = th.log(th.tensor([0.01, 0.02, 0.03], dtype=th.float32))
    return (
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        pred_oos_logprobs,
    )


def test_peak_marginal_aggregates_in_window_candidates_by_logsumexp() -> None:
    """A true peak's score is the log-sum of every candidate inside the window."""
    args = _single_spectrum_inputs()
    ios_ce, _, _, _ = _make_fn("peak_marginal", vectorized=True)(*args)
    # 300.0 and 300.001 match (0.5 + 0.3); 300.01 is outside the 0.003 Da window.
    assert ios_ce.item() == pytest.approx(-math.log(0.8), abs=1e-5)


def test_peak_marginal_ignores_offset_inside_window_but_gaussian_does_not() -> None:
    """Moving a matched candidate within the window changes Gaussian CE, not peak-marginal."""
    (
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        pred_oos_logprobs,
    ) = _single_spectrum_inputs()
    shifted_pred_mzs = pred_mzs.clone()
    shifted_pred_mzs[1] = 300.0025  # still inside the 0.003 Da window

    def run(fn, mzs):
        return fn(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            mzs,
            pred_logprobs,
            pred_batch_idxs,
            pred_oos_logprobs,
        )[0].item()

    pm_fn = _make_fn("peak_marginal", vectorized=True)
    gauss_fn = _make_fn("gaussian", vectorized=True)
    assert run(pm_fn, pred_mzs) == pytest.approx(run(pm_fn, shifted_pred_mzs), abs=1e-6)
    assert run(gauss_fn, pred_mzs) != pytest.approx(run(gauss_fn, shifted_pred_mzs), abs=1e-6)


@pytest.mark.parametrize("dist", ["gaussian", "peak_marginal"])
@pytest.mark.parametrize("loss_batch_size", [1, 8])
def test_vectorized_matches_sequential(dist: str, loss_batch_size: int) -> None:
    """The vectorized path reproduces the sequential path for both distributions."""
    args = _multi_spectrum_inputs()
    vec = _make_fn(dist, vectorized=True, loss_batch_size=loss_batch_size)(*args)
    seq = _make_fn(dist, vectorized=False)(*args)
    for vec_value, seq_value in zip(vec[:3], seq[:3], strict=True):
        assert th.allclose(vec_value, seq_value, atol=1e-5)


def test_unmatched_true_peak_is_excluded_and_reported_as_unmatched() -> None:
    """A true peak with no candidate in range adds no in-support loss."""
    true_mzs = th.tensor([300.0, 700.0], dtype=th.float32)
    true_logprobs = th.log(th.tensor([0.75, 0.25], dtype=th.float32))
    true_batch_idxs = th.tensor([0, 0], dtype=th.long)
    pred_mzs = th.tensor([300.0], dtype=th.float32)
    pred_logprobs = th.tensor([0.0], dtype=th.float32)
    pred_batch_idxs = th.tensor([0], dtype=th.long)
    pred_oos_logprobs = th.tensor([math.log(0.25)], dtype=th.float32)

    ios_ce, _, true_oos_logprobs, _ = _make_fn("peak_marginal", vectorized=True)(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        pred_oos_logprobs,
    )
    # Only the matched peak contributes: -0.75 * log(1.0) == 0.
    assert ios_ce.item() == pytest.approx(0.0, abs=1e-6)
    # The unmatched peak carries all of the unmatched intensity.
    assert true_oos_logprobs.exp().item() == pytest.approx(0.25, abs=1e-5)


def test_per_sample_tolerance_widens_the_window() -> None:
    """Per-instrument tolerance changes which candidates a true peak can match."""
    true_mzs = th.tensor([300.0], dtype=th.float32)
    true_logprobs = th.tensor([0.0], dtype=th.float32)
    true_batch_idxs = th.tensor([0], dtype=th.long)
    pred_mzs = th.tensor([300.008], dtype=th.float32)
    pred_logprobs = th.tensor([0.0], dtype=th.float32)
    pred_batch_idxs = th.tensor([0], dtype=th.long)
    pred_oos_logprobs = th.tensor([math.log(1e-6)], dtype=th.float32)
    fn = _make_fn("peak_marginal", vectorized=True)

    args = (
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        pred_oos_logprobs,
    )
    # FT-like: 1e-5 * 300 = 0.003 Da -> 0.008 Da offset is unmatched.
    _, _, ft_unmatched, _ = fn(
        *args,
        tol_per_sample=th.tensor([1e-5], dtype=th.float32),
        min_mz_per_sample=th.tensor([200.0], dtype=th.float32),
    )
    # QTOF-like: 2e-5 * 750 = 0.015 Da floor -> the same candidate now matches.
    _, _, qtof_unmatched, _ = fn(
        *args,
        tol_per_sample=th.tensor([2e-5], dtype=th.float32),
        min_mz_per_sample=th.tensor([750.0], dtype=th.float32),
    )
    assert ft_unmatched.exp().item() == pytest.approx(1.0, abs=1e-5)
    assert qtof_unmatched.exp().item() == pytest.approx(0.0, abs=1e-5)


def test_unknown_distribution_is_rejected() -> None:
    with pytest.raises(NotImplementedError):
        _make_fn("laplace", vectorized=True)


@pytest.mark.parametrize("dist", ["gaussian", "peak_marginal"])
def test_per_sample_tolerance_matching_the_global_is_a_no_op(dist: str) -> None:
    """Enabling per-instrument tolerance with the global values changes nothing.

    FT-only configs set ``inst_type_loss_tol`` to the same values as the global
    tolerance, which switches the loss from the scalar path to the per-peak tensor
    path. That switch must be numerically inert.
    """
    args = _multi_spectrum_inputs()
    batch_size = int(args[6].numel())
    fn = _make_fn(dist, vectorized=True)
    scalar = fn(*args)
    per_sample = fn(
        *args,
        tol_per_sample=th.full((batch_size,), TOLERANCE, dtype=th.float32),
        min_mz_per_sample=th.full((batch_size,), TOLERANCE_MIN_MZ, dtype=th.float32),
    )
    for scalar_value, per_sample_value in zip(scalar[:3], per_sample[:3], strict=True):
        assert th.allclose(scalar_value, per_sample_value, atol=1e-5)


def test_sequential_gaussian_truncates_at_the_tolerance_window() -> None:
    """A candidate outside the window contributes nothing, in either implementation.

    ``gaussian_ce`` previously summed over every candidate while still applying the
    ``gaussian_renormalize`` truncation correction, so it disagreed with
    ``sparse_cross_entropy_vec`` for candidates in the 1-3 sigma band.
    """
    true_mzs = th.tensor([300.0], dtype=th.float32)
    true_logprobs = th.tensor([0.0], dtype=th.float32)
    true_batch_idxs = th.tensor([0], dtype=th.long)
    # Tolerance at m/z 300 is 0.003 Da; 300.005 is ~1.7 sigma away and out of window.
    in_window_only = th.tensor([300.0], dtype=th.float32)
    with_outsider = th.tensor([300.0, 300.005], dtype=th.float32)
    oos = th.tensor([math.log(1e-6)], dtype=th.float32)

    def run(vectorized: bool, pred_mzs: th.Tensor, pred_probs: list[float]) -> float:
        return _make_fn("gaussian", vectorized=vectorized)(
            true_mzs,
            true_logprobs,
            true_batch_idxs,
            pred_mzs,
            th.log(th.tensor(pred_probs, dtype=th.float32)),
            th.zeros(len(pred_probs), dtype=th.long),
            oos,
        )[0].item()

    for vectorized in (True, False):
        assert run(vectorized, in_window_only, [0.5]) == pytest.approx(
            run(vectorized, with_outsider, [0.5, 0.5]), abs=1e-5
        )
