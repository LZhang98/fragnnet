"""Numerical regressions with independent entropy, Gaussian and assignment references."""

import itertools
import math

import pytest
import torch as th

from fragnnet.loss.cross_entropy import get_sparse_cross_entropy_fn
from fragnnet.loss.entropy import sparse_conditional_entropy_fn, sparse_entropy_fn
from fragnnet.loss.spectrum_distance import (
    sparse_jensen_shannon_divergence_hungarian,
    sparse_jensen_shannon_divergence_hungarian_vec,
)


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("shift", [0.0, -25.0, -60.0])
@pytest.mark.parametrize("chunk", [1, 3])
def test_jsd_tiny_mass_matches_exhaustive_assignment(vectorized, shift, chunk):
    """Ambiguous all-to-all matches must maximize normalized JS gain at every scale."""
    fn = (
        sparse_jensen_shannon_divergence_hungarian_vec
        if vectorized
        else sparse_jensen_shannon_divergence_hungarian
    )
    p = th.tensor([0.1, 0.65, 0.25])
    logits = th.tensor([0.55, 0.15, 0.3]).log().requires_grad_()
    mz = th.tensor([100.0, 100.01, 100.02])
    idx = th.zeros(3, dtype=th.long)
    actual = fn(
        mz,
        p.log() + shift,
        idx,
        mz,
        logits + shift,
        idx,
        tolerance=0.1,
        relative=False,
        **({"loss_batch_size": chunk} if vectorized else {}),
    )
    q = logits.softmax(0)
    costs = []
    for order in itertools.permutations(range(3)):
        r = q[list(order)]
        m = (p + r) / 2
        costs.append((0.5 * (p * (p / m).log() + r * (r / m).log()).sum()) / math.log(2))
    expected = th.stack(costs).min()
    th.testing.assert_close(actual[0], expected, atol=3e-6, rtol=3e-5)
    (actual_grad,) = th.autograd.grad(actual.sum(), logits, retain_graph=True)
    (expected_grad,) = th.autograd.grad(expected, logits)
    th.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=1e-4)


@pytest.mark.parametrize("zero", [-th.inf, -1000.0])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("oos", [False, True])
def test_entropy_zero_mass_has_finite_values_and_gradients(zero, renormalize, oos):
    lp = th.tensor([math.log(0.25), zero, math.log(0.75)]).requires_grad_()
    outside = th.tensor([zero], requires_grad=True) if oos else None
    entropy, normalized = sparse_entropy_fn(lp, th.zeros(3, dtype=th.long), outside, renormalize)
    expected = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
    th.testing.assert_close(entropy, th.tensor([expected]))
    th.testing.assert_close(normalized, th.tensor([expected / math.log(4 if oos else 3)]))
    entropy.sum().backward()
    assert th.isfinite(lp.grad).all()
    assert lp.grad[1] == 0
    if oos:
        assert th.isfinite(outside.grad).all() and outside.grad.item() == 0


@pytest.mark.parametrize("zero", [-th.inf, -1000.0])
def test_conditional_entropy_zero_mass(zero):
    marginal = th.tensor([math.log(0.6), math.log(0.4)], requires_grad=True)
    conditional = th.tensor([math.log(0.5), math.log(0.5), zero, 0.0], requires_grad=True)
    entropy, normalized = sparse_conditional_entropy_fn(
        marginal,
        th.tensor([0, 0]),
        conditional,
        th.tensor([0, 0, 1, 1]),
        th.zeros(4, dtype=th.long),
    )
    th.testing.assert_close(entropy, th.tensor([0.6 * math.log(2)]))
    th.testing.assert_close(normalized, th.tensor([0.6]))
    entropy.sum().backward()
    assert th.isfinite(marginal.grad).all() and th.isfinite(conditional.grad).all()
    assert conditional.grad[2] == 0


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("multiple,renormalize", [(-1, False), (3, False), (3, True)])
@pytest.mark.parametrize("chunk", [1, 4])
def test_gaussian_per_spectrum_width_against_normal_density(
    vectorized, relative, multiple, renormalize, chunk
):
    pred_mz = th.tensor([100.0, 200.0])
    true_mz = pred_mz + th.tensor([0.05, 0.1])
    idx = th.arange(2)
    widths = th.tensor([0.2, 0.4])
    tolerances = widths / pred_mz if relative else widths
    lp = th.tensor([-0.3, -0.7], requires_grad=True)
    fn = get_sparse_cross_entropy_fn(
        "gaussian", vectorized, 0.01, relative, 50.0, multiple, renormalize, 1.0, chunk
    )
    actual = fn(
        true_mz,
        th.zeros(2),
        idx,
        pred_mz,
        lp,
        idx,
        th.tensor([-2.0, -2.0]),
        tol_per_sample=tolerances,
        min_mz_per_sample=th.tensor([50.0, 50.0]),
    )
    density = th.distributions.Normal(pred_mz, widths).log_prob(true_mz)
    correction = math.log(math.erf(multiple / math.sqrt(2))) if renormalize else 0.0
    expected = -density - lp + correction
    th.testing.assert_close(actual[0], expected, rtol=2e-5, atol=2e-5)
    th.testing.assert_close(actual[1], th.zeros(2))
    (grad,) = th.autograd.grad(actual[0].sum(), lp)
    th.testing.assert_close(grad, -th.ones(2))


@pytest.mark.parametrize("zero", [-th.inf, -1000.0])
@pytest.mark.parametrize("dist", ["gaussian", "peak_marginal"])
def test_sequential_oos_entropy_ignores_zero_mass_peaks(zero, dist):
    true_lp = th.tensor([math.log(0.5), math.log(0.2), zero, math.log(0.3)], requires_grad=True)
    fn = get_sparse_cross_entropy_fn(dist, False, 0.1, False, 1.0, 3.0, False, 1.0, 2)
    result = fn(
        th.tensor([100.0, 200.0, 250.0, 300.0]),
        true_lp,
        th.zeros(4, dtype=th.long),
        th.tensor([100.0]),
        th.tensor([0.0]),
        th.zeros(1, dtype=th.long),
        th.tensor([-2.0]),
    )
    expected_entropy = -0.4 * math.log(0.4) - 0.6 * math.log(0.6)
    th.testing.assert_close(result[2], th.tensor([math.log(0.5)]))
    th.testing.assert_close(result[3], th.tensor([expected_entropy]))
    result[3].sum().backward()
    assert th.isfinite(true_lp.grad).all()
    assert true_lp.grad[2] == 0


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_oos", [False, True])
def test_entropy_all_zero_mass_is_zero(renormalize, with_oos):
    lp = th.full((3,), -th.inf, requires_grad=True)
    oos = th.tensor([-th.inf], requires_grad=True) if with_oos else None
    values = sparse_entropy_fn(lp, th.zeros(3, dtype=th.long), oos, renormalize)
    for value in values:
        th.testing.assert_close(value, th.zeros(1))
    sum(value.sum() for value in values).backward()
    th.testing.assert_close(lp.grad, th.zeros(3))


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("shift", [0.0, -30.0])
@pytest.mark.parametrize("case", ["disjoint", "missing_true", "missing_pred", "zero_pred"])
def test_jsd_degenerate_rows_keep_distance_one(vectorized, shift, case):
    fn = (
        sparse_jensen_shannon_divergence_hungarian_vec
        if vectorized
        else sparse_jensen_shannon_divergence_hungarian
    )
    tm = th.tensor([] if case == "missing_true" else [100.0])
    pm = th.tensor([] if case == "missing_pred" else [200.0 if case == "disjoint" else 100.0])
    tl = th.full_like(tm, shift)
    pl = th.full_like(pm, -th.inf if case == "zero_pred" else shift).requires_grad_()
    actual = fn(
        tm,
        tl,
        th.zeros(len(tm), dtype=th.long),
        pm,
        pl,
        th.zeros(len(pm), dtype=th.long),
        tolerance=0.1,
        relative=False,
    )
    th.testing.assert_close(actual, th.ones(1))
    if pl.numel():
        actual.sum().backward()
        th.testing.assert_close(pl.grad, th.zeros_like(pl))


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("true_shift,pred_shift", [(-60.0, 0.0), (0.0, -60.0), (-60.0, -60.0)])
def test_jsd_tiny_mass_preserves_ambiguous_feasible_assignment(vectorized, true_shift, pred_shift):
    """The optimal JS matching differs from the unnormalized tiny-mass product matching."""
    fn = (
        sparse_jensen_shannon_divergence_hungarian_vec
        if vectorized
        else sparse_jensen_shannon_divergence_hungarian
    )
    p = th.tensor([0.3942, 0.601, 0.0048])
    logits = th.tensor([0.7866, 0.0482, 0.1652]).log().requires_grad_()
    tm, pm = th.tensor([100.0, 100.04, 100.08]), th.tensor([100.01, 100.05, 100.09])
    rows = th.zeros(3, dtype=th.long)
    actual = fn(
        tm,
        p.log() + true_shift,
        rows,
        pm,
        logits + pred_shift,
        rows,
        tolerance=0.07,
        relative=False,
    )
    q = logits.softmax(0)
    feasible = (tm[:, None] - pm[None, :]).abs() < 0.07
    costs = []
    for order in itertools.permutations(range(3)):
        if all(feasible[i, j] for i, j in enumerate(order)):
            r = q[list(order)]
            m = (p + r) / 2
            costs.append(0.5 * (p * (p / m).log() + r * (r / m).log()).sum() / math.log(2))
    expected = th.stack(costs).min()
    th.testing.assert_close(actual[0], expected, atol=3e-6, rtol=3e-5)
    (grad,) = th.autograd.grad(actual.sum(), logits, retain_graph=True)
    (ref_grad,) = th.autograd.grad(expected, logits)
    th.testing.assert_close(grad, ref_grad, atol=3e-6, rtol=3e-5)
