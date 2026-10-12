"""Independent dense references and invariants for shared loss preparation."""

import math

import pytest
import torch as th
import torch.nn.functional as F

from fragnnet.loss.cross_entropy import get_sparse_cross_entropy_fn
from fragnnet.loss.padding import (
    build_loss_chunks,
    build_padded_batch_layout,
    pad_flat_with_layout,
    paired_batch_offsets,
)
from fragnnet.loss.pairwise import (
    get_cross_jss_contrastive,
    get_pairwise_cossim,
    get_pairwise_cross_entropy,
    get_pairwise_jss_sim,
)
from fragnnet.loss.peak_list import (
    sparse_outside_window_intensity_fraction,
    sparse_peak_list_distance,
)
from fragnnet.utils.misc_utils import EPS


def _dense(logprobs, rows, columns, batch_size=3, width=8):
    flat = logprobs.new_zeros(batch_size * width).index_add(
        0, rows * width + columns, logprobs.exp()
    )
    return flat.reshape(batch_size, width)


def _jss(p, q):
    mixture = (p[:, None, :] + q[None, :, :]) * 0.5
    h_mix = -(mixture * (mixture + EPS).log()).sum(-1)
    h_p = -(p * (p + EPS).log()).sum(-1)
    h_q = -(q * (q + EPS).log()).sum(-1)
    return (1 - (h_mix - (h_p[:, None] + h_q[None, :]) / 2) / math.log(2)).clamp(0, 1)


def _assert_value_gradient(actual, expected, leaf, atol=3e-6):
    th.testing.assert_close(actual, expected, atol=atol, rtol=2e-5)
    weights = th.linspace(0.2, 1.3, actual.numel()).reshape(actual.shape).to(actual)
    (actual_grad,) = th.autograd.grad((actual * weights).sum(), leaf, retain_graph=True)
    (expected_grad,) = th.autograd.grad((expected * weights).sum(), leaf)
    th.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=5e-5)


@pytest.mark.parametrize("kind", ["cosine", "jss", "ce"])
@pytest.mark.parametrize("chunk", [1, 2, 8, None])
@pytest.mark.parametrize("resolution", [0.1, 0.25])
@pytest.mark.parametrize("shuffle", [False, True])
def test_pairwise_dense_reference_and_gradient(kind, chunk, resolution, shuffle):
    rows = th.tensor([0, 0, 0, 1, 1, 2, 2])
    cols = th.tensor([1, 1, 4, 1, 6, 3, 7])  # duplicates, overlap and disjoint support
    lp = th.tensor([0.1, 0.4, 0.5, 0.8, 0.2, 0.3, 0.7]).log().requires_grad_()
    order = th.tensor([6, 2, 4, 0, 5, 1, 3]) if shuffle else th.arange(7)
    fn = {
        "cosine": get_pairwise_cossim,
        "jss": get_pairwise_jss_sim,
        "ce": get_pairwise_cross_entropy,
    }[kind]
    actual = fn(
        (cols[order] + 0.3) * resolution,
        lp[order],
        rows[order],
        3,
        8 * resolution,
        resolution,
        chunk_size=chunk,
    )
    dense = _dense(lp, rows, cols)
    if kind == "cosine":
        normalized = F.normalize(dense, p=2, dim=1)
        expected = normalized @ normalized.T
    else:
        normalized = dense / dense.sum(1, keepdim=True)
        expected = (
            _jss(normalized, normalized)
            if kind == "jss"
            else -normalized @ (normalized + EPS).log().T
        )
    _assert_value_gradient(actual, expected, lp)
    if kind != "ce":
        th.testing.assert_close(actual, actual.T)
    else:
        assert not th.allclose(actual, actual.T)


@pytest.mark.parametrize("chunk", [1, 2, None])
@pytest.mark.parametrize("sqrt", [False, True])
@pytest.mark.parametrize("temperature", [0.07, 0.5])
def test_cross_jss_shared_bins_dense_reference(chunk, sqrt, temperature):
    rows = th.tensor([0, 0, 1, 1, 2, 2])
    true_cols, pred_cols = th.tensor([1, 2, 2, 4, 5, 7]), th.tensor([1, 3, 2, 2, 4, 6])
    p_lp = th.tensor([0.3, 0.7, 0.8, 0.2, 0.6, 0.4]).log()
    q_lp = th.tensor([0.4, 0.6, 0.7, 0.3, 0.2, 0.8]).log().requires_grad_()
    actual = get_cross_jss_contrastive(
        (true_cols + 0.2) / 4,
        p_lp,
        rows,
        (pred_cols + 0.2) / 4,
        q_lp,
        rows,
        3,
        2.0,
        0.25,
        temperature,
        sqrt,
        chunk,
    )
    p, q = _dense(p_lp, rows, true_cols), _dense(q_lp, rows, pred_cols)
    if sqrt:
        # A finite floor avoids undefined derivatives at empty bins; exact zeros stay zero.
        p = th.where(p > 0, p.clamp_min(1e-20).sqrt(), 0)
        q = th.where(q > 0, q.clamp_min(1e-20).sqrt(), 0)
    p, q = p / p.sum(1, keepdim=True), q / q.sum(1, keepdim=True)
    expected = F.cross_entropy(_jss(p, q) / temperature, th.arange(3), reduction="none")
    _assert_value_gradient(actual, expected, q_lp, atol=1e-5)


@pytest.mark.parametrize("counts", [[2, 0, 3], [0, 0, 0], [1], [5, 1, 2]])
@pytest.mark.parametrize("sorted_input", [False, True])
@pytest.mark.parametrize("fill", [0.0, -7.0])
def test_reused_layout_values_masks_and_gradients(counts, sorted_input, fill):
    rows = th.repeat_interleave(th.arange(len(counts)), th.tensor(counts))
    values = th.arange(rows.numel(), dtype=th.float64).requires_grad_()
    if not sorted_input:
        rows = rows.flip(0)
    order, indices, valid, width = build_padded_batch_layout(
        rows, len(counts), assume_sorted=sorted_input
    )
    actual = pad_flat_with_layout(values, order, indices, len(counts), width, fill)
    expected = values.new_full((len(counts), max(counts)), fill)
    for row in range(len(counts)):
        expected[row, : counts[row]] = values[rows == row]
    th.testing.assert_close(actual, expected)
    assert valid.sum(1).tolist() == counts
    # Reuse the exact same layout for another field.
    th.testing.assert_close(
        pad_flat_with_layout(values * 2, order, indices, len(counts), width, fill)[valid],
        actual[valid] * 2,
    )
    if values.numel():
        actual.sum().backward()
        th.testing.assert_close(values.grad, th.ones_like(values))


@pytest.mark.parametrize("chunk", [1, 2, 4, 20])
def test_offsets_and_adaptive_chunks_preserve_empty_rows_and_bound_pairs(chunk):
    tc, pc = th.tensor([1, 0, 9, 2]), th.tensor([8, 0, 1, 4])
    true, pred = th.repeat_interleave(th.arange(4), tc), th.repeat_interleave(th.arange(4), pc)
    a, b, t, p = paired_batch_offsets(true, pred, 4)
    th.testing.assert_close(a, tc)
    th.testing.assert_close(b, pc)
    assert t == [0, 1, 1, 10, 12] and p == [0, 8, 8, 9, 13]
    chunks = build_loss_chunks(t, p, chunk)
    assert [i for lo, hi in chunks for i in range(lo, hi)] == list(range(4))
    for lo, hi in chunks:
        assert hi - lo <= chunk
        assert (hi - lo) * max(tc[lo:hi]) * max(pc[lo:hi]) <= sum(tc[lo:hi]) * sum(pc[lo:hi])


@pytest.mark.parametrize("dist", ["gaussian", "peak_marginal"])
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("per_sample", [False, True])
@pytest.mark.parametrize("chunk", [1, 2, 7])
def test_ragged_ce_dispatch_values_and_prediction_gradients(dist, relative, per_sample, chunk):
    tm, pm = th.tensor([100.0, 120.0, 130.0, 200.0, 210.0, 250.0]), th.tensor(
        [100.01, 130.01, 199.99, 210.02]
    )
    ti, pi = th.tensor([0, 0, 0, 1, 1, 1]), th.tensor([0, 0, 1, 1])
    tl = th.tensor([0.2, 0.3, 0.5, 0.1, 0.2, 0.7]).log()
    pl = th.tensor([0.4, 0.6, 0.7, 0.3]).log().requires_grad_()
    oos = th.tensor([-2.0, -3.0], requires_grad=True)
    tolerance = 0.001 if relative else 0.15
    kwargs = (
        {
            "tol_per_sample": th.tensor([tolerance, tolerance * 2]),
            "min_mz_per_sample": th.tensor([80.0, 220.0]),
        }
        if per_sample
        else {}
    )
    result = []
    for vectorized in [False, True]:
        fn = get_sparse_cross_entropy_fn(
            dist, vectorized, tolerance, relative, 100.0, 3.0, True, 2.0, chunk
        )
        result.append(fn(tm, tl, ti, pm, pl, pi, oos, **kwargs))
    for actual, expected in zip(result[1][:3], result[0][:3]):
        th.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    _assert_value_gradient(result[1][0] + result[1][1], result[0][0] + result[0][1], pl)
    th.testing.assert_close(result[1][3], th.zeros(2))  # documented path-specific contract
    assert result[0][3][0] == 0 and result[0][3][1] == 0  # one OOS peak per spectrum


@pytest.mark.parametrize(
    "objective", ["cosine_distance", "hellinger_distance", "jss_distance", "kl_distance"]
)
@pytest.mark.parametrize("matching", ["closest", "hungarian"])
@pytest.mark.parametrize("dump", [False, True])
@pytest.mark.parametrize("chunk", [1, 3])
def test_peak_list_entropy_and_primary_dense_reference(objective, matching, dump, chunk):
    rows = th.tensor([0, 0, 1, 1])
    mz = th.tensor([100.0, 150.0, 200.0, 250.0])
    p = th.tensor([0.3, 0.7, 0.6, 0.4]).reshape(2, 2)
    q = th.tensor([0.4, 0.6, 0.2, 0.8], requires_grad=True)
    qdist = q.reshape(2, 2)
    if dump:
        qdist = th.cat((qdist, qdist.new_full((2, 1), 0.2)), dim=1)
        p = th.cat((p, p.new_zeros((2, 1))), dim=1)
    actual, entropy = sparse_peak_list_distance(
        mz,
        p[:, :2].flatten(),
        rows,
        mz,
        q,
        rows,
        objective=objective,
        matching=matching,
        relative=False,
        tolerance=0.1,
        loss_batch_size=chunk,
        pred_dump_intensities=th.full((2,), 0.2) if dump else None,
    )
    qdist = qdist / qdist.sum(1, keepdim=True)
    if objective == "cosine_distance":
        expected = 1 - (F.normalize(p, dim=1) * F.normalize(qdist, dim=1)).sum(1)
    elif objective == "hellinger_distance":
        expected = 1 - th.where(p > 0, (p * qdist).clamp_min(1e-20).sqrt(), 0).sum(1)
    elif objective == "jss_distance":
        expected = 1 - _jss(p, qdist).diagonal()
    else:
        smoothed = (1 - 1e-4) * qdist + 1e-4 * p
        expected = (p * (p.clamp_min(1e-20).log() - smoothed.clamp_min(1e-20).log())).sum(1)
    h_p = -(p * p.clamp_min(1e-20).log()).sum(1)
    h_q = -(qdist * qdist.log()).sum(1)
    _assert_value_gradient(actual + entropy, expected + (h_p - h_q).square(), q, atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 3, 9])
@pytest.mark.parametrize("shuffle", [False, True])
def test_leakage_reused_layout_ragged_empty_rows(chunk, shuffle):
    tm, ti = th.tensor([100.0, 200.0, 210.0]), th.tensor([0, 2, 2])
    pm, pi = th.tensor([100.0, 130.0, 180.0, 200.0, 240.0]), th.tensor([0, 0, 1, 2, 2])
    values = th.tensor([0.4, 0.6, 1.0, 0.7, 0.3], requires_grad=True)
    order = th.tensor([4, 1, 3, 0, 2]) if shuffle else th.arange(5)
    actual = sparse_outside_window_intensity_fraction(
        tm,
        ti,
        pm[order],
        values[order],
        pi[order],
        batch_size=3,
        relative=False,
        tolerance=0.1,
        loss_batch_size=chunk,
    )
    expected = th.stack(
        (values[1] / values[:2].sum(), values[2] * 0 + 1, values[4] / values[3:].sum())
    )
    _assert_value_gradient(actual, expected, values)


@pytest.mark.parametrize("attribution", ["gaussian_posterior", "hungarian"])
@pytest.mark.parametrize("use_prior", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_channel_targets_gaussian_reference_and_detached_prior(attribution, use_prior, relative):
    from fragnnet.loss.channel_support import observational_channel_targets

    masses = th.tensor([[[100.0, 100.1]], [[200.0, 200.1]]])
    weights = th.tensor([[[0.1, 0.9]], [[0.8, 0.2]]]).log().requires_grad_()
    peaks = th.tensor([100.01, 200.08])
    tol = 0.002 if relative else 0.2
    actual = observational_channel_targets(
        true_mzs=peaks,
        true_batch_idxs=th.tensor([0, 1]),
        channel_mzs=masses,
        channel_log_weights=weights,
        chemical_valid_channels=th.ones_like(masses, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0, 1]),
        batch_size=2,
        tolerance=tol,
        relative=relative,
        tolerance_min_mz=50.0,
        attribution=attribution,
        use_intensity_prior=use_prior,
    )
    std = masses * tol if relative else th.full_like(masses, tol)
    log_likelihood = -0.5 * ((peaks[:, None, None] - masses) / std).square() - std.log()
    if use_prior:
        log_likelihood = log_likelihood + weights.detach()
    expected = (
        log_likelihood.softmax(-1)
        if attribution == "gaussian_posterior"
        else F.one_hot(log_likelihood.argmax(-1), 2).float()
    )
    th.testing.assert_close(actual, expected)
    assert not actual.requires_grad


@pytest.mark.parametrize("chunk", [0, -1])
def test_adaptive_chunks_reject_nonpositive_limits(chunk):
    with pytest.raises(ValueError, match="must be > 0"):
        build_loss_chunks([0, 1], [0, 1], chunk)


def test_dispatch_rejects_unknown_distribution():
    with pytest.raises(NotImplementedError, match="not implemented"):
        get_sparse_cross_entropy_fn("invalid", True, 0.1, False, 1.0, 3.0, False, 1.0, 1)


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_ragged_ce_values_and_gradients():
    with th.device("cuda"):
        test_ragged_ce_dispatch_values_and_prediction_gradients("gaussian", False, True, 2)
        test_pairwise_dense_reference_and_gradient("jss", 2, 0.25, True)
        test_peak_list_entropy_and_primary_dense_reference("hellinger_distance", "closest", True, 1)
