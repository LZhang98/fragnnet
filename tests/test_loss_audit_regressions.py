"""Regression cases for the shared loss audit's matching defects."""

import math

import pytest
import torch as th

from fragnnet.loss.cross_entropy import get_sparse_cross_entropy_fn
from fragnnet.loss.spectrum_distance import sparse_cosine_distance_hungarian


@pytest.mark.parametrize("log_distance", [False, True])
@pytest.mark.parametrize("equal_intensities", [False, True])
def test_hungarian_cosine_excludes_forced_infeasible_assignment(
    log_distance: bool, equal_intensities: bool
) -> None:
    """A 2-to-1 and 1-to-2 match graph permits only two of three assignments."""
    true_mzs = th.tensor([100.0, 100.001, 101.0])
    pred_mzs = th.tensor([100.0, 101.0, 101.001])
    true_probs = th.tensor([1 / 3] * 3 if equal_intensities else [0.2, 0.3, 0.5])
    pred_probs = th.tensor([1 / 3] * 3 if equal_intensities else [0.4, 0.35, 0.25])
    pred_lp = pred_probs.log().requires_grad_()
    batch_idxs = th.zeros(3, dtype=th.long)
    actual = sparse_cosine_distance_hungarian(
        true_mzs,
        true_probs.log(),
        batch_idxs,
        pred_mzs,
        pred_lp,
        batch_idxs,
        tolerance=0.01,
        relative=False,
        log_distance=log_distance,
    )
    # With unequal intensities the optimal feasible pairs are (1, 0) and (2, 1).
    reference_lp = pred_probs.log().requires_grad_()
    reference_probs = reference_lp.exp()
    similarity = (true_probs[1] * reference_probs[0] + true_probs[2] * reference_probs[1]) / (
        true_probs.norm() * reference_probs.norm()
    )
    expected = th.log1p(-similarity) if log_distance else 1.0 - similarity
    assert actual.shape == (1,)
    assert actual.device == pred_lp.device
    assert actual.item() == pytest.approx(expected.item(), abs=1e-6)
    if equal_intensities:
        assert actual.item() == pytest.approx(math.log(1 / 3) if log_distance else 1 / 3)
    else:
        # Equal weights have tied assignments, so only compare gradients for the unique optimum.
        actual_grad = th.autograd.grad(actual.sum(), pred_lp)[0]
        expected_grad = th.autograd.grad(expected, reference_lp)[0]
        assert th.allclose(actual_grad, expected_grad, atol=1e-6)


@pytest.mark.parametrize("log_distance", [False, True])
def test_hungarian_cosine_no_matches_has_zero_gradient(log_distance: bool) -> None:
    pred_lp = th.tensor([0.0], requires_grad=True)
    idx = th.zeros(1, dtype=th.long)
    loss = sparse_cosine_distance_hungarian(
        th.tensor([100.0]),
        th.tensor([0.0]),
        idx,
        th.tensor([200.0]),
        pred_lp,
        idx,
        tolerance=0.01,
        relative=False,
        log_distance=log_distance,
    )
    assert loss.item() == pytest.approx(0.0 if log_distance else 1.0)
    assert th.equal(th.autograd.grad(loss.sum(), pred_lp)[0], th.zeros_like(pred_lp))


@pytest.mark.parametrize("relative,per_sample", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("chunk_size", [1, 8])
def test_untruncated_gaussian_ce_matches_sequential_values_and_gradients(
    relative: bool, per_sample: bool, chunk_size: int
) -> None:
    """Negative window means every candidate contributes, including outside one sigma."""
    true_mzs = th.tensor([100.0, 100.03, 300.0])
    pred_mzs = th.tensor([100.0, 100.04, 300.05])
    idx = th.tensor([0, 0, 1])
    true_lp = th.tensor([0.6, 0.4, 1.0]).log()
    pred_lp = th.tensor([0.3, 0.7, 1.0]).log().requires_grad_()
    oos_lp = th.tensor([-2.0, -3.0], requires_grad=True)
    kwargs = {}
    if per_sample:
        kwargs = {
            "tol_per_sample": th.tensor([1e-4, 2e-4]),
            "min_mz_per_sample": th.tensor([200.0, 400.0]),
        }
    results = []
    gradients = []
    for vectorized in (False, True):
        fn = get_sparse_cross_entropy_fn(
            dist="gaussian",
            vectorized=vectorized,
            tolerance=1e-4 if relative else 0.01,
            relative=relative,
            tolerance_min_mz=200.0,
            oos_tolerance_multiple=-1,
            gaussian_renormalize=False,
            pm_tolerance_multiple=1.0,
            loss_batch_size=chunk_size,
        )
        result = fn(true_mzs, true_lp, idx, pred_mzs, pred_lp, idx, oos_lp, **kwargs)
        results.append(result)
        gradients.append(th.autograd.grad((result[0] + result[1]).sum(), (pred_lp, oos_lp)))
    for sequential, vectorized in zip(results[0][:3], results[1][:3], strict=True):
        assert vectorized.shape == (2,)
        assert th.allclose(vectorized, sequential, atol=1e-5)
    assert th.equal(results[1][1], th.zeros(2))
    assert th.equal(results[1][2].exp(), th.zeros(2))
    for sequential, vectorized in zip(gradients[0], gradients[1], strict=True):
        assert th.isfinite(vectorized).all()
        assert th.allclose(vectorized, sequential, atol=1e-5)
    assert gradients[1][0].abs().sum() > 0
