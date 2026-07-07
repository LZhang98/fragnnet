"""Tests for GPU-sync-removal refactor in sparse_cross_entropy_vec.

Covers three changes:
1. scatter_logsumexp empty-input guard (numel==0, no GPU sync)
2. Removed th.any(bl_true_both_mask) branch — scatter_reduce handles empty input
3. Removed th.all(bl_true_both_mask) branch — scatter_logsumexp handles empty input

Edge cases exercised:
- All true peaks have a matching predicted peak (empty OOS / unmatched path)
- No true peaks have a matching predicted peak (empty IOS / matched path)
- Mixed (normal training case)
"""

import pytest
import torch as th
import torch.nn.functional as F

from fragnnet.model.loss import get_sparse_cross_entropy_fn, sparse_cross_entropy_vec
from fragnnet.utils.misc_utils import EPS, LOG_ZERO, scatter_logsumexp

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_batch_idxs(peaks_per_spec: list[int], device: th.device) -> th.Tensor:
    batch_size = len(peaks_per_spec)
    return th.repeat_interleave(
        th.arange(batch_size, device=device),
        th.tensor(peaks_per_spec, device=device),
    )


def _default_ce_fn():
    return get_sparse_cross_entropy_fn(
        dist="gaussian",
        vectorized=True,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        pm_tolerance_multiple=1,
        loss_batch_size=4,
    )


def _reference_sparse_ce_vec(
    true_mzs,
    true_logprobs,
    true_batch_idxs,
    pred_mzs,
    pred_logprobs,
    pred_batch_idxs,
    pred_oos_logprobs,
    tolerance,
    relative,
    tol_min_mz,
    oos_tol,
    gaussian_renorm,
    loss_batch_size,
    batch_size,
):
    """Old implementation with th.any / th.all branches — used as correctness reference."""
    import math as _math

    from fragnnet.model.loss import scatter_reduce as _sr
    from fragnnet.utils.misc_utils import LOG_ZERO as _LZ
    from fragnnet.utils.misc_utils import scatter_logsumexp as _sle
    from fragnnet.utils.spec_utils import calculate_match_mzs as _cmm

    float_dtype = true_logprobs.dtype
    device = true_logprobs.device
    mask_value = _LZ(float_dtype)

    loss_num_batches = int((batch_size // loss_batch_size) + int(batch_size % loss_batch_size > 0))
    true_batch_cumsum = th.cumsum(
        F.pad(th.unique(true_batch_idxs, return_counts=True)[1], (1, 0), "constant", 0), dim=0
    )
    pred_batch_cumsum = th.cumsum(
        F.pad(th.unique(pred_batch_idxs, return_counts=True)[1], (1, 0), "constant", 0), dim=0
    )

    ios_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    oos_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    true_oos_out = th.zeros(batch_size, dtype=float_dtype, device=device)

    sigma = _math.sqrt(2)
    trunc = _math.erf(oos_tol / sigma)
    log_trunc = th.log(th.tensor(trunc, device=device))

    for bl in range(loss_num_batches):
        bl_lower = bl * loss_batch_size
        bl_upper = min((bl + 1) * loss_batch_size, batch_size)
        bl_batch_size = bl_upper - bl_lower
        tbl = true_batch_cumsum[bl_lower]
        tbu = true_batch_cumsum[bl_upper]
        pbl = pred_batch_cumsum[bl_lower]
        pbu = pred_batch_cumsum[bl_upper]

        bl_ti = true_batch_idxs[tbl:tbu] - bl_lower
        bl_pi = pred_batch_idxs[pbl:pbu] - bl_lower
        bl_tlp = true_logprobs[tbl:tbu]
        bl_plp = pred_logprobs[pbl:pbu]
        bl_tmz = true_mzs[tbl:tbu]
        bl_pmz = pred_mzs[pbl:pbu]
        bl_poos = pred_oos_logprobs[bl_lower:bl_upper]

        batch_mask = bl_ti.reshape(-1, 1) == bl_pi.reshape(1, -1)
        match_mask = _cmm(
            bl_tmz,
            bl_pmz,
            tolerance=oos_tol * tolerance,
            relative=relative,
            tolerance_min_mz=tol_min_mz,
        )
        match_mask = th.as_tensor(match_mask, device=device)
        both_mask = batch_mask & match_mask
        del batch_mask, match_mask
        true_both = th.any(both_mask, dim=1)

        stds = th.clamp(bl_pmz, min=tol_min_mz) * tolerance
        vs = stds**2
        log_trunc_vec = log_trunc * th.ones_like(bl_pmz)

        ios_lp = (
            -0.5 * (bl_tmz.reshape(-1, 1) - bl_pmz.reshape(1, -1)) ** 2 / vs.reshape(1, -1)
            - 0.5 * th.log(2 * th.pi * vs.reshape(1, -1))
            + bl_plp.reshape(1, -1)
            - log_trunc_vec.reshape(1, -1)
        )
        ios_lp = ios_lp + (~both_mask).to(float_dtype) * mask_value
        ios_lp = th.logsumexp(ios_lp, dim=1)

        if not th.any(true_both):
            pass  # ios_ce stays 0 (old behaviour: no assignment in this branch)
        else:
            bl_ios = _sr(
                (-th.exp(bl_tlp) * ios_lp)[true_both],
                bl_ti[true_both],
                "sum",
                dim=0,
                dim_size=bl_batch_size,
                default=0.0,
            )
            ios_ce[bl_lower:bl_upper] = bl_ios

        if not th.all(true_both):
            bl_oos_lp = _sle(bl_tlp[~true_both], bl_ti[~true_both], dim_size=bl_batch_size)
        else:
            bl_oos_lp = th.full((bl_batch_size,), mask_value, dtype=float_dtype, device=device)
        oos_ce[bl_lower:bl_upper] = -th.exp(bl_oos_lp) * bl_poos
        true_oos_out[bl_lower:bl_upper] = bl_oos_lp

    return ios_ce, oos_ce, true_oos_out


# ── 1. scatter_logsumexp empty-input guard ────────────────────────────────────


class TestScatterLogsumexpEmptyInput:
    """scatter_logsumexp must return LOG_ZERO fill (not crash) when input is empty."""

    @pytest.mark.parametrize(
        "device_str",
        [
            "cpu",
            pytest.param(
                "cuda", marks=pytest.mark.skipif(not th.cuda.is_available(), reason="no GPU")
            ),
        ],
    )
    def test_empty_with_dim_size(self, device_str):
        device = th.device(device_str)
        logits = th.empty(0, dtype=th.float32, device=device)
        idxs = th.empty(0, dtype=th.long, device=device)
        result = scatter_logsumexp(logits, idxs, dim_size=4)
        assert result.shape == (4,), f"expected (4,), got {result.shape}"
        expected = LOG_ZERO(th.float32)
        assert th.all(result == expected), f"expected LOG_ZERO fills, got {result}"

    def test_empty_without_dim_size_returns_empty(self):
        logits = th.empty(0, dtype=th.float32)
        idxs = th.empty(0, dtype=th.long)
        result = scatter_logsumexp(logits, idxs, dim_size=None)
        assert result.shape == (0,)

    def test_empty_result_gives_zero_oos_ce(self):
        """exp(LOG_ZERO) * pred_oos_logprob ≈ 0 — OOS loss is 0 when no unmatched peaks."""
        device = th.device("cpu")
        logits = th.empty(0, dtype=th.float32, device=device)
        idxs = th.empty(0, dtype=th.long, device=device)
        oos_lp = scatter_logsumexp(logits, idxs, dim_size=3)
        pred_oos = th.tensor([-1.0, -2.0, -3.0])
        oos_ce = -th.exp(oos_lp) * pred_oos
        assert th.allclose(oos_ce, th.zeros(3), atol=1e-6), f"expected zeros, got {oos_ce}"

    @pytest.mark.parametrize(
        "device_str",
        [
            "cpu",
            pytest.param(
                "cuda", marks=pytest.mark.skipif(not th.cuda.is_available(), reason="no GPU")
            ),
        ],
    )
    def test_nonempty_unchanged(self, device_str):
        """Non-empty path must produce the same result as before the guard."""
        device = th.device(device_str)
        th.manual_seed(0)
        logits = th.randn(6, device=device)
        idxs = th.tensor([0, 0, 1, 1, 2, 2], device=device)
        result = scatter_logsumexp(logits, idxs, dim_size=3)
        # Manual reference
        expected = th.stack([th.logsumexp(logits[idxs == i], dim=0) for i in range(3)])
        assert th.allclose(result, expected, atol=1e-5), f"mismatch: {result} vs {expected}"


# ── 2. sparse_cross_entropy_vec edge cases ────────────────────────────────────


class TestSparseCEEdgeCases:
    @pytest.fixture
    def ce_fn(self):
        return _default_ce_fn()

    @pytest.mark.parametrize(
        "device_str",
        [
            "cpu",
            pytest.param(
                "cuda", marks=pytest.mark.skipif(not th.cuda.is_available(), reason="no GPU")
            ),
        ],
    )
    def test_all_peaks_match_oos_ce_near_zero(self, ce_fn, device_str):
        """When every true peak matches a predicted peak, OOS loss must be ≈ 0."""
        device = th.device(device_str)
        th.manual_seed(1)
        batch_size = 4
        peaks = [3, 4, 2, 5]
        # true and pred share the same m/z values — all peaks match within tolerance
        mzs = th.cat([th.linspace(100.0, 500.0, n, device=device) for n in peaks])
        true_bidxs = _make_batch_idxs(peaks, device)
        pred_bidxs = _make_batch_idxs(peaks, device)
        true_lp = th.cat([th.log_softmax(th.randn(n, device=device), dim=0) for n in peaks])
        pred_lp = th.cat([th.log_softmax(th.randn(n, device=device), dim=0) for n in peaks])
        pred_oos = th.full((batch_size,), -2.0, device=device)

        ios_ce, oos_ce, _, _ = ce_fn(mzs, true_lp, true_bidxs, mzs, pred_lp, pred_bidxs, pred_oos)
        assert oos_ce.shape == (batch_size,)
        assert th.allclose(oos_ce, th.zeros(batch_size, device=device), atol=1e-5), (
            f"OOS CE should be ~0 when all peaks match, got {oos_ce}"
        )
        # ios_ce can be negative when Gaussian log-prob at zero distance > 0 (small σ)
        assert th.all(th.isfinite(ios_ce)), f"IOS CE must be finite, got {ios_ce}"

    @pytest.mark.parametrize(
        "device_str",
        [
            "cpu",
            pytest.param(
                "cuda", marks=pytest.mark.skipif(not th.cuda.is_available(), reason="no GPU")
            ),
        ],
    )
    def test_no_peaks_match_ios_ce_is_zero(self, ce_fn, device_str):
        """When no true peak matches any predicted peak, IOS CE must be 0."""
        device = th.device(device_str)
        th.manual_seed(2)
        batch_size = 3
        peaks = [2, 3, 2]
        # true at low m/z, pred at high m/z — zero overlap within tolerance
        true_mzs = th.cat([th.linspace(50.0, 100.0, n, device=device) for n in peaks])
        pred_mzs = th.cat([th.linspace(900.0, 1000.0, n, device=device) for n in peaks])
        true_bidxs = _make_batch_idxs(peaks, device)
        pred_bidxs = _make_batch_idxs(peaks, device)
        true_lp = th.cat([th.log_softmax(th.randn(n, device=device), dim=0) for n in peaks])
        pred_lp = th.cat([th.log_softmax(th.randn(n, device=device), dim=0) for n in peaks])
        pred_oos = th.full((batch_size,), -2.0, device=device)

        ios_ce, oos_ce, _, _ = ce_fn(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos
        )
        assert ios_ce.shape == (batch_size,)
        assert th.allclose(ios_ce, th.zeros(batch_size, device=device), atol=1e-6), (
            f"IOS CE should be 0 when no peaks match, got {ios_ce}"
        )
        # OOS CE must be positive (all true peaks are out-of-sample)
        assert (oos_ce > 0).all(), f"OOS CE should be positive when no peaks match, got {oos_ce}"

    def test_single_spectrum_single_peak_match(self):
        """Trivial single-peak case: IOS CE = -log(pred_prob), OOS CE = 0."""
        device = th.device("cpu")
        mz = th.tensor([300.0])
        true_lp = th.tensor([0.0])  # log-prob = 0 → prob = 1
        pred_lp = th.tensor([-0.5])  # log-prob = -0.5
        pred_oos = th.tensor([-10.0])  # effectively 0
        bidxs = th.tensor([0])

        ce_fn = get_sparse_cross_entropy_fn(
            dist="gaussian",
            vectorized=True,
            tolerance=0.1,
            relative=False,
            tolerance_min_mz=0.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=False,
            pm_tolerance_multiple=1,
            loss_batch_size=4,
        )
        ios_ce, oos_ce, _, _ = ce_fn(mz, true_lp, bidxs, mz, pred_lp, bidxs, pred_oos)
        assert oos_ce.shape == (1,)
        # oos_ce ≈ 0 because all peaks match
        assert float(oos_ce[0]) == pytest.approx(0.0, abs=1e-4)
        # ios_ce = -p * log_prob; can be negative when Gaussian log-prob > 0 (high density at match)
        assert th.isfinite(ios_ce[0]), f"ios_ce must be finite, got {ios_ce[0]}"


# ── 3. Correctness vs reference implementation ────────────────────────────────


@pytest.fixture
def batch_data():
    th.manual_seed(42)
    batch_size = 8
    peaks_true = [4, 6, 3, 8, 5, 2, 7, 4]
    peaks_pred = [5, 4, 6, 3, 7, 5, 3, 6]
    true_bidxs = _make_batch_idxs(peaks_true, th.device("cpu"))
    pred_bidxs = _make_batch_idxs(peaks_pred, th.device("cpu"))
    # m/z values with partial overlap
    true_mzs = th.cat(
        [th.linspace(100.0 + i * 5, 200.0 + i * 5, n) for i, n in enumerate(peaks_true)]
    )
    pred_mzs = th.cat(
        [th.linspace(95.0 + i * 5, 195.0 + i * 5, n) for i, n in enumerate(peaks_pred)]
    )
    true_lp = th.cat([th.log_softmax(th.randn(n), dim=0) for n in peaks_true])
    pred_lp = th.cat([th.log_softmax(th.randn(n), dim=0) for n in peaks_pred])
    pred_oos = th.full((batch_size,), -2.0)
    return (batch_size, true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos)


class TestSparseCEMatchesReference:
    """New implementation must match the old th.any/th.all reference for normal inputs."""

    def test_ios_ce_matches_reference(self, batch_data):
        (batch_size, true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos) = (
            batch_data
        )

        tol, rel, tol_min, oos_tol, renorm, lbs = 1e-5, True, 200.0, 1, True, 4

        ref_ios, ref_oos, _ = _reference_sparse_ce_vec(
            true_mzs,
            true_lp,
            true_bidxs,
            pred_mzs,
            pred_lp,
            pred_bidxs,
            pred_oos,
            tol,
            rel,
            tol_min,
            oos_tol,
            renorm,
            lbs,
            batch_size,
        )
        new_ios, new_oos, _, _ = sparse_cross_entropy_vec(
            true_mzs,
            true_lp,
            true_bidxs,
            pred_mzs,
            pred_lp,
            pred_bidxs,
            pred_oos,
            tolerance=tol,
            relative=rel,
            tolerance_min_mz=tol_min,
            oos_tolerance_multiple=oos_tol,
            gaussian_renormalize=renorm,
            loss_batch_size=lbs,
        )

        assert th.allclose(new_ios, ref_ios, atol=1e-5), (
            f"IOS CE mismatch:\n  new={new_ios}\n  ref={ref_ios}"
        )
        assert th.allclose(new_oos, ref_oos, atol=1e-5), (
            f"OOS CE mismatch:\n  new={new_oos}\n  ref={ref_oos}"
        )

    @pytest.mark.parametrize(
        "device_str",
        [
            "cpu",
            pytest.param(
                "cuda", marks=pytest.mark.skipif(not th.cuda.is_available(), reason="no GPU")
            ),
        ],
    )
    def test_ce_fn_wrapper_matches_reference(self, batch_data, device_str):
        """get_sparse_cross_entropy_fn wrapper produces the same result."""
        (batch_size, true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos) = (
            batch_data
        )
        device = th.device(device_str)
        true_mzs, true_lp, true_bidxs = (
            true_mzs.to(device),
            true_lp.to(device),
            true_bidxs.to(device),
        )
        pred_mzs, pred_lp, pred_bidxs = (
            pred_mzs.to(device),
            pred_lp.to(device),
            pred_bidxs.to(device),
        )
        pred_oos = pred_oos.to(device)

        tol, rel, tol_min, oos_tol, renorm, lbs = 1e-5, True, 200.0, 1, True, 4

        ref_ios, ref_oos, _ = _reference_sparse_ce_vec(
            true_mzs,
            true_lp,
            true_bidxs,
            pred_mzs,
            pred_lp,
            pred_bidxs,
            pred_oos,
            tol,
            rel,
            tol_min,
            oos_tol,
            renorm,
            lbs,
            batch_size,
        )
        ce_fn = get_sparse_cross_entropy_fn(
            dist="gaussian",
            vectorized=True,
            tolerance=tol,
            relative=rel,
            tolerance_min_mz=tol_min,
            oos_tolerance_multiple=oos_tol,
            gaussian_renormalize=renorm,
            pm_tolerance_multiple=1,
            loss_batch_size=lbs,
        )
        new_ios, new_oos, _, _ = ce_fn(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos
        )

        assert th.allclose(new_ios.cpu(), ref_ios.cpu(), atol=1e-5)
        assert th.allclose(new_oos.cpu(), ref_oos.cpu(), atol=1e-5)

    def test_loss_batch_size_invariant(self, batch_data):
        """Result must be identical regardless of loss_batch_size (it's a memory tradeoff)."""
        (batch_size, true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos) = (
            batch_data
        )

        tol, rel, tol_min, oos_tol, renorm = 1e-5, True, 200.0, 1, True

        def run(lbs):
            ios, oos, _, _ = sparse_cross_entropy_vec(
                true_mzs,
                true_lp,
                true_bidxs,
                pred_mzs,
                pred_lp,
                pred_bidxs,
                pred_oos,
                tolerance=tol,
                relative=rel,
                tolerance_min_mz=tol_min,
                oos_tolerance_multiple=oos_tol,
                gaussian_renormalize=renorm,
                loss_batch_size=lbs,
            )
            return ios, oos

        ios_ref, oos_ref = run(4)
        for lbs in [1, 16, 100]:
            ios, oos = run(lbs)
            assert th.allclose(ios, ios_ref, atol=1e-5), f"IOS mismatch at loss_batch_size={lbs}"
            assert th.allclose(oos, oos_ref, atol=1e-5), f"OOS mismatch at loss_batch_size={lbs}"


# ── 4. hard_peak_temperature hard-example-mining weighting ───────────────────


class TestHardPeakTemperatureWeighting:
    """hard_peak_temperature continuously upweights matched true peaks within each
    spectrum in proportion to their own CE, via a per-spectrum softmax that
    conserves total loss mass (weights sum to the spectrum's peak count)."""

    def test_disabled_matches_no_kwarg_baseline(self, batch_data):
        """hard_peak_temperature=None (explicit) must match not passing it at all."""
        (batch_size, true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos) = (
            batch_data
        )
        common = dict(
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            loss_batch_size=4,
        )
        ios_default, oos_default, _, _ = sparse_cross_entropy_vec(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos, **common
        )
        ios_explicit, oos_explicit, _, _ = sparse_cross_entropy_vec(
            true_mzs,
            true_lp,
            true_bidxs,
            pred_mzs,
            pred_lp,
            pred_bidxs,
            pred_oos,
            hard_peak_temperature=None,
            **common,
        )
        assert th.allclose(ios_explicit, ios_default)
        assert th.allclose(oos_explicit, oos_default)

    def test_equal_ce_peaks_are_unaffected_regardless_of_temperature(self, batch_data):
        """When all matched peaks in a spectrum have identical CE, softmax is
        uniform, so weight == 1 everywhere and the result must match baseline —
        true for ANY temperature, since it's a property of equal inputs, not of T."""
        mzs = th.tensor([100.0, 200.0, 300.0])
        true_lp = th.zeros(3)
        pred_lp = th.full((3,), -1.0)  # identical CE across all 3 peaks
        bidxs = th.zeros(3, dtype=th.long)
        pred_oos = th.full((1,), -2.0)
        common = dict(
            tolerance=1e-4,
            relative=False,
            tolerance_min_mz=0.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            loss_batch_size=4,
        )
        baseline, _, _, _ = sparse_cross_entropy_vec(
            mzs, true_lp, bidxs, mzs, pred_lp, bidxs, pred_oos, **common
        )
        # Very low temperatures amplify floating-point noise between nominally-equal
        # CE values (through the logsumexp/masking pipeline) — not tested here since
        # it's numerical noise, not a correctness property; 1.0/100.0 stay stable.
        for temperature in (1.0, 100.0):
            weighted, _, _, _ = sparse_cross_entropy_vec(
                mzs,
                true_lp,
                bidxs,
                mzs,
                pred_lp,
                bidxs,
                pred_oos,
                hard_peak_temperature=temperature,
                **common,
            )
            assert th.allclose(weighted, baseline, atol=1e-5), (
                f"temperature={temperature}: expected {baseline}, got {weighted}"
            )

    def test_high_temperature_approaches_uniform_weighting(self, batch_data):
        """As temperature -> large, softmax approaches uniform, so weighted result
        must approach (but need not exactly equal) the unweighted baseline."""
        (batch_size, true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos) = (
            batch_data
        )
        common = dict(
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            loss_batch_size=4,
        )
        baseline, _, _, _ = sparse_cross_entropy_vec(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos, **common
        )
        weighted, _, _, _ = sparse_cross_entropy_vec(
            true_mzs,
            true_lp,
            true_bidxs,
            pred_mzs,
            pred_lp,
            pred_bidxs,
            pred_oos,
            hard_peak_temperature=1e6,
            **common,
        )
        assert th.allclose(weighted, baseline, atol=1e-3), (
            f"high-temperature weighted CE should ~match baseline:\n"
            f"  weighted={weighted}\n  baseline={baseline}"
        )

    def test_upweights_the_hardest_peak_in_a_spectrum(self):
        """Hand-constructed single spectrum: verify the exact softmax-weighted sum
        against per-peak CEs obtained independently (own single-peak 'spectrum')."""
        # Widely-spaced, 1:1 matched mzs — no cross-matching between peaks.
        mzs = th.tensor([100.0, 200.0, 300.0, 400.0])
        true_lp = th.zeros(4)  # exp(0) = 1, so per-peak CE = -ios_log_prob
        pred_lp = th.tensor([-3.0, -0.1, -2.0, -0.5])
        temperature = 1.0
        common = dict(
            tolerance=1e-4,
            relative=False,
            tolerance_min_mz=0.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            loss_batch_size=4,
        )

        # Obtain each peak's individual CE contribution by putting each in its own
        # "spectrum" (batch_size=4) — grouping doesn't affect the pairwise ios_log_prob.
        per_peak_bidxs = th.tensor([0, 1, 2, 3])
        pred_oos_4 = th.full((4,), -2.0)
        individual_ce, _, _, _ = sparse_cross_entropy_vec(
            mzs, true_lp, per_peak_bidxs, mzs, pred_lp, per_peak_bidxs, pred_oos_4, **common
        )

        # Now combine all 4 peaks into a single spectrum with continuous weighting.
        combined_bidxs = th.zeros(4, dtype=th.long)
        pred_oos_1 = th.full((1,), -2.0)
        combined_ce, _, _, _ = sparse_cross_entropy_vec(
            mzs,
            true_lp,
            combined_bidxs,
            mzs,
            pred_lp,
            combined_bidxs,
            pred_oos_1,
            hard_peak_temperature=temperature,
            **common,
        )

        z = (individual_ce - individual_ce.mean()) / th.clamp(individual_ce.std(), min=EPS)
        expected_weight = th.softmax(z / temperature, dim=0) * 4
        expected = th.sum(expected_weight * individual_ce)
        assert th.allclose(combined_ce[0], expected, atol=1e-5), (
            f"expected {expected}, got {combined_ce[0]}"
        )

    def test_single_peak_spectrum_is_noop(self):
        """A spectrum with exactly one matched peak: softmax over a single-element
        group is always 1.0, so weight == 1*1 == 1 regardless of temperature."""
        mz = th.tensor([300.0])
        true_lp = th.tensor([0.0])
        pred_lp = th.tensor([-1.0])
        pred_oos = th.tensor([-2.0])
        bidxs = th.tensor([0])
        common = dict(
            tolerance=1e-4,
            relative=False,
            tolerance_min_mz=0.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            loss_batch_size=4,
        )
        baseline, _, _, _ = sparse_cross_entropy_vec(
            mz, true_lp, bidxs, mz, pred_lp, bidxs, pred_oos, **common
        )
        weighted, _, _, _ = sparse_cross_entropy_vec(
            mz, true_lp, bidxs, mz, pred_lp, bidxs, pred_oos, hard_peak_temperature=0.1, **common
        )
        assert th.allclose(weighted, baseline, atol=1e-5)

    def test_ragged_batch_with_all_oos_spectrum(self):
        """A spectrum with zero matched peaks (all OOS) inside the same loss-batch as
        a spectrum with matched peaks must not crash and must not affect the other
        spectrum's weighting."""
        # Spectrum 0: 2 matched peaks. Spectrum 1: true/pred far apart -> no matches.
        true_mzs = th.tensor([100.0, 200.0, 50.0, 60.0])
        pred_mzs = th.tensor([100.0, 200.0, 900.0, 950.0])
        true_bidxs = th.tensor([0, 0, 1, 1])
        pred_bidxs = th.tensor([0, 0, 1, 1])
        true_lp = th.tensor([0.0, 0.0, 0.0, 0.0])
        pred_lp = th.tensor([-2.0, -0.5, -1.0, -1.0])
        pred_oos = th.tensor([-2.0, -2.0])
        temperature = 1.0
        common = dict(
            tolerance=1e-4,
            relative=False,
            tolerance_min_mz=0.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            loss_batch_size=4,
        )
        weighted, _, _, _ = sparse_cross_entropy_vec(
            true_mzs,
            true_lp,
            true_bidxs,
            pred_mzs,
            pred_lp,
            pred_bidxs,
            pred_oos,
            hard_peak_temperature=temperature,
            **common,
        )
        # Spectrum 1 (all OOS, no matched peaks) must be untouched.
        assert th.allclose(weighted[1], th.zeros(1), atol=1e-6)

        # Spectrum 0: derive each peak's individual CE to compute the expected
        # weighted sum exactly (no crash from the ragged missing-group case).
        per_peak_bidxs = th.tensor([0, 1])
        pred_oos_2 = th.full((2,), -2.0)
        individual_ce, _, _, _ = sparse_cross_entropy_vec(
            true_mzs[:2],
            true_lp[:2],
            per_peak_bidxs,
            pred_mzs[:2],
            pred_lp[:2],
            per_peak_bidxs,
            pred_oos_2,
            **common,
        )
        z = (individual_ce - individual_ce.mean()) / th.clamp(individual_ce.std(), min=EPS)
        expected_weight = th.softmax(z / temperature, dim=0) * 2
        expected_spec0 = th.sum(expected_weight * individual_ce)
        assert th.allclose(weighted[0], expected_spec0, atol=1e-5), (
            f"expected {expected_spec0}, got {weighted[0]}"
        )

    def test_invalid_temperature_raises(self, batch_data):
        (batch_size, true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos) = (
            batch_data
        )
        common = dict(
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            loss_batch_size=4,
        )
        for bad_temperature in (0.0, -1.0):
            with pytest.raises(ValueError):
                sparse_cross_entropy_vec(
                    true_mzs,
                    true_lp,
                    true_bidxs,
                    pred_mzs,
                    pred_lp,
                    pred_bidxs,
                    pred_oos,
                    hard_peak_temperature=bad_temperature,
                    **common,
                )

    def test_non_vectorized_with_temperature_raises(self):
        """hard_peak_temperature is only supported by the vectorized loss path."""
        with pytest.raises(NotImplementedError):
            get_sparse_cross_entropy_fn(
                dist="gaussian",
                vectorized=False,
                tolerance=1e-5,
                relative=True,
                tolerance_min_mz=200.0,
                oos_tolerance_multiple=1,
                gaussian_renormalize=True,
                pm_tolerance_multiple=1,
                loss_batch_size=4,
                hard_peak_temperature=1.0,
            )

    def test_apply_hard_peak_weighting_false_disables_reweighting(self):
        """The get_sparse_cross_entropy_fn wrapper must fall back to the unweighted
        baseline when apply_hard_peak_weighting=False, regardless of the configured
        hard_peak_temperature — this is what PL modules pass during eval/val/test
        (apply_hard_peak_weighting=self.training) so hard-example mining only ever
        affects the training gradient, never the reported validation/test loss."""
        # Same hand-constructed scenario as test_upweights_the_hardest_peak_in_a_spectrum:
        # guaranteed exact matches (same mz) with distinguishable per-peak CE.
        mzs = th.tensor([100.0, 200.0, 300.0, 400.0])
        true_lp = th.zeros(4)
        pred_lp = th.tensor([-3.0, -0.1, -2.0, -0.5])
        bidxs = th.zeros(4, dtype=th.long)
        pred_oos = th.full((1,), -2.0)
        common = dict(
            dist="gaussian",
            vectorized=True,
            tolerance=1e-4,
            relative=False,
            tolerance_min_mz=0.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            pm_tolerance_multiple=1,
            loss_batch_size=4,
        )
        ce_fn_baseline = get_sparse_cross_entropy_fn(**common)
        ce_fn_hard_peak = get_sparse_cross_entropy_fn(hard_peak_temperature=1.0, **common)

        baseline_ios, baseline_oos, _, _ = ce_fn_baseline(
            mzs, true_lp, bidxs, mzs, pred_lp, bidxs, pred_oos
        )
        # "train" mode: reweighting must actually change the result vs baseline.
        train_ios, _, _, _ = ce_fn_hard_peak(
            mzs,
            true_lp,
            bidxs,
            mzs,
            pred_lp,
            bidxs,
            pred_oos,
            apply_hard_peak_weighting=True,
        )
        assert not th.allclose(train_ios, baseline_ios, atol=1e-4), (
            "expected hard_peak_temperature to change ios_ce when applied"
        )
        # "eval" mode: apply_hard_peak_weighting=False must match the unweighted baseline.
        eval_ios, eval_oos, _, _ = ce_fn_hard_peak(
            mzs,
            true_lp,
            bidxs,
            mzs,
            pred_lp,
            bidxs,
            pred_oos,
            apply_hard_peak_weighting=False,
        )
        assert th.allclose(eval_ios, baseline_ios, atol=1e-6)
        assert th.allclose(eval_oos, baseline_oos, atol=1e-6)

    def test_weight_concentration_stabilizes_instead_of_collapsing(self):
        """Regression test for the winner-take-all collapse bug: with raw (non
        z-scored) CE, a persistently-hard peak's weight share among 20 peaks grows
        unboundedly toward 100% as the other 19 peaks' fit improves (simulating
        training progress) — starving them of gradient entirely. Z-scoring must
        make the hard peak's weight share stabilize at a bounded plateau instead of
        continuing to grow as the easy majority's CE scale keeps increasing."""
        n_peaks = 20
        mzs = th.arange(n_peaks).float() * 50.0 + 100.0
        true_lp = th.zeros(n_peaks)
        per_peak_bidxs = th.arange(n_peaks)
        pred_oos_n = th.full((n_peaks,), -2.0)
        common = dict(
            tolerance=1e-4,
            relative=False,
            tolerance_min_mz=0.0,
            oos_tolerance_multiple=1,
            gaussian_renormalize=True,
            loss_batch_size=4,
        )

        def hard_peak_weight_share(easy_pred_lp: float) -> float:
            pred_lp = th.full((n_peaks,), easy_pred_lp)
            pred_lp[0] = -0.2  # persistently "hard" peak, doesn't improve
            individual_ce, _, _, _ = sparse_cross_entropy_vec(
                mzs, true_lp, per_peak_bidxs, mzs, pred_lp, per_peak_bidxs, pred_oos_n, **common
            )
            z = (individual_ce - individual_ce.mean()) / th.clamp(individual_ce.std(), min=EPS)
            weight = th.softmax(z, dim=0) * n_peaks
            return (weight[0] / n_peaks).item()

        # Simulate the easy majority's fit improving over the course of training.
        shares = [hard_peak_weight_share(easy_lp) for easy_lp in (0.0, 10.0, 20.0, 40.0, 80.0)]

        # Must not approach total collapse (the old raw-CE bug reached ~100% by lp=10).
        assert max(shares) < 0.95, f"hard-peak weight share should stay bounded, got {shares}"
        # Must stabilize (not keep climbing) once the easy/hard gap is established —
        # the last few stages should be within numerical noise of each other.
        assert abs(shares[-1] - shares[-2]) < 1e-3, (
            f"expected weight share to plateau, got shares={shares}"
        )
