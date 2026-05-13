"""Benchmark GPU-sync-removal refactor in sparse_cross_entropy_vec.

Old path:
  - true_batch_cumsum / pred_batch_cumsum kept as GPU tensors
  - Loop body indexes them with Python ints → th.Tensor.__getitem__(int) forces .item() GPU sync
  - th.any(bl_true_both_mask) and th.all(bl_true_both_mask) are evaluated as Python bools → 2 more syncs/iter
  - bl_log_trunc_factors = th.log(th.tensor(...) * th.ones_like(...)) creates a full tensor every iter

New path:
  - .tolist() on cumsum tensors → Python-int slice indices, zero sync
  - scatter_reduce / scatter_logsumexp called unconditionally (both empty-safe) → no th.any/th.all
  - _LOG_2PI and _log_trunc_factor are Python scalars, no tensor allocation

Sync count per training step (batch_size=256, loss_batch_size=16 → 16 loop iters):
  GPU tensor index syncs: 4 / iter × 16 = 64
  th.any / th.all syncs:  2 / iter × 16 = 32
  Total eliminated: 96 GPU→CPU stalls

Usage:
    conda run -n FRAGNNET-GPU python scripts/debug/bench_loss_syncs.py
"""

import gc
import math
import time

import torch as th
import torch.nn.functional as F

from fragnnet.model.loss import sparse_cross_entropy_vec
from fragnnet.utils.misc_utils import (
    EPS,
    LOG_ZERO,
    scatter_logsumexp,
    scatter_reduce,
)
from fragnnet.utils.spec_utils import calculate_match_mzs

# ── Configuration ─────────────────────────────────────────────────────────────
BATCH_SIZE = 256
PEAKS_PER_SPEC = 50
LOSS_BATCH_SIZE = 16          # production value
TOLERANCE = 1e-5
TOLERANCE_MIN_MZ = 200.0
OOS_TOLERANCE_MULTIPLE = 1
GAUSSIAN_RENORMALIZE = True
RELATIVE = True

DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
WARMUP = 5
REPEATS = 30


# ── Old implementation (with GPU syncs) ───────────────────────────────────────

def _sparse_ce_old(
    true_mzs, true_logprobs, true_batch_idxs,
    pred_mzs, pred_logprobs, pred_batch_idxs, pred_oos_logprobs,
    batch_size,
    loss_batch_size=LOSS_BATCH_SIZE,
):
    """Old implementation: GPU tensor cumsum indexing + th.any/th.all + per-iter tensor alloc."""
    float_dtype = true_logprobs.dtype
    device = true_logprobs.device
    mask_value = LOG_ZERO(float_dtype)

    loss_num_batches = int((batch_size // loss_batch_size) + int(batch_size % loss_batch_size > 0))

    # GPU tensors — each [bl_lower] / [bl_upper] index forces a .item() GPU sync
    true_batch_cumsum = th.cat([
        true_batch_idxs.new_zeros(1),
        th.bincount(true_batch_idxs, minlength=batch_size).cumsum(0),
    ])
    pred_batch_cumsum = th.cat([
        pred_batch_idxs.new_zeros(1),
        th.bincount(pred_batch_idxs, minlength=batch_size).cumsum(0),
    ])

    ios_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    oos_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    true_oos_logprobs = th.zeros(batch_size, dtype=float_dtype, device=device)

    sigma = math.sqrt(2)
    trunc_factor = math.erf(OOS_TOLERANCE_MULTIPLE / sigma)

    for bl in range(loss_num_batches):
        bl_lower = bl * loss_batch_size
        bl_upper = min((bl + 1) * loss_batch_size, batch_size)
        # GPU tensor indexing → .item() sync (×4 per iteration)
        bl_true_lower = true_batch_cumsum[bl_lower]
        bl_true_upper = true_batch_cumsum[bl_upper]
        bl_pred_lower = pred_batch_cumsum[bl_lower]
        bl_pred_upper = pred_batch_cumsum[bl_upper]
        bl_batch_size = bl_upper - bl_lower

        bl_ti = true_batch_idxs[bl_true_lower:bl_true_upper] - bl_lower
        bl_pi = pred_batch_idxs[bl_pred_lower:bl_pred_upper] - bl_lower
        bl_tlp = true_logprobs[bl_true_lower:bl_true_upper]
        bl_plp = pred_logprobs[bl_pred_lower:bl_pred_upper]
        bl_tmz = true_mzs[bl_true_lower:bl_true_upper]
        bl_pmz = pred_mzs[bl_pred_lower:bl_pred_upper]
        bl_poos = pred_oos_logprobs[bl_lower:bl_upper]

        bl_batch_mask = bl_ti.reshape(-1, 1) == bl_pi.reshape(1, -1)
        bl_match_mask = calculate_match_mzs(
            bl_tmz, bl_pmz,
            tolerance=OOS_TOLERANCE_MULTIPLE * TOLERANCE,
            relative=RELATIVE, tolerance_min_mz=TOLERANCE_MIN_MZ,
        )
        bl_match_mask = th.as_tensor(bl_match_mask, device=device)
        bl_both_mask = bl_batch_mask & bl_match_mask
        del bl_batch_mask, bl_match_mask
        bl_true_both = th.any(bl_both_mask, dim=1)

        bl_stds = th.clamp(bl_pmz, min=TOLERANCE_MIN_MZ) * TOLERANCE
        bl_vars = bl_stds ** 2

        # per-iter tensor allocation (ones_like + th.log + th.tensor)
        bl_log_trunc = th.log(th.tensor(trunc_factor, device=device) * th.ones_like(bl_pmz))

        bl_ios_lp = (
            -0.5 * (bl_tmz.reshape(-1, 1) - bl_pmz.reshape(1, -1)) ** 2 / bl_vars.reshape(1, -1)
            - 0.5 * th.log(2 * th.pi * bl_vars.reshape(1, -1))
            + bl_plp.reshape(1, -1) - bl_log_trunc.reshape(1, -1)
        )
        bl_ios_lp = bl_ios_lp + (~bl_both_mask).to(float_dtype) * mask_value
        bl_ios_lp = th.logsumexp(bl_ios_lp, dim=1)

        # th.any() → GPU sync (×1 per iteration)
        if not th.any(bl_true_both):
            pass  # ios_ce stays 0
        else:
            ios_ce[bl_lower:bl_upper] = scatter_reduce(
                (-th.exp(bl_tlp) * bl_ios_lp)[bl_true_both],
                bl_ti[bl_true_both], "sum", dim=0, dim_size=bl_batch_size, default=0.0,
            )

        # th.all() → GPU sync (×1 per iteration)
        if not th.all(bl_true_both):
            bl_oos_lp = scatter_logsumexp(
                bl_tlp[~bl_true_both], bl_ti[~bl_true_both], dim_size=bl_batch_size,
            )
        else:
            bl_oos_lp = th.full((bl_batch_size,), LOG_ZERO(float_dtype), dtype=float_dtype, device=device)

        oos_ce[bl_lower:bl_upper] = -th.exp(bl_oos_lp) * bl_poos
        true_oos_logprobs[bl_lower:bl_upper] = bl_oos_lp

    return ios_ce, oos_ce, true_oos_logprobs


# ── Data generation ───────────────────────────────────────────────────────────

def make_batch(batch_size, peaks_per_spec, device):
    """Create realistic sparse spectrum batch."""
    total_true = batch_size * peaks_per_spec
    total_pred = batch_size * peaks_per_spec

    true_mzs = (th.rand(total_true, device=device) * 900 + 50).sort().values
    pred_mzs = (th.rand(total_pred, device=device) * 900 + 50).sort().values
    true_bidxs = th.repeat_interleave(
        th.arange(batch_size, device=device), peaks_per_spec
    )
    pred_bidxs = th.repeat_interleave(
        th.arange(batch_size, device=device), peaks_per_spec
    )
    true_lp = th.cat([
        th.log_softmax(th.randn(peaks_per_spec, device=device), dim=0)
        for _ in range(batch_size)
    ])
    pred_lp = th.cat([
        th.log_softmax(th.randn(peaks_per_spec, device=device), dim=0)
        for _ in range(batch_size)
    ])
    pred_oos = th.full((batch_size,), -2.0, device=device)
    return true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos


# ── Timing helpers ────────────────────────────────────────────────────────────

def sync():
    if DEVICE.type == "cuda":
        th.cuda.synchronize()


def timed_run(fn, args, repeats, warmup):
    for _ in range(warmup):
        fn(*args)
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    sync()
    return (time.perf_counter() - t0) * 1000 / repeats  # ms per call


def peak_mb_run(fn, args):
    gc.collect()
    if DEVICE.type == "cuda":
        th.cuda.empty_cache()
        th.cuda.reset_peak_memory_stats()
    baseline = th.cuda.memory_allocated() / 1024**2 if DEVICE.type == "cuda" else 0.0
    fn(*args)
    sync()
    if DEVICE.type == "cuda":
        return th.cuda.max_memory_allocated() / 1024**2 - baseline
    return float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"batch_size={BATCH_SIZE}, peaks_per_spec={PEAKS_PER_SPEC}, loss_batch_size={LOSS_BATCH_SIZE}")
    print(f"Loop iterations per call: {BATCH_SIZE // LOSS_BATCH_SIZE}")
    print(f"GPU syncs eliminated per call: 4×{BATCH_SIZE // LOSS_BATCH_SIZE} (cumsum) + "
          f"2×{BATCH_SIZE // LOSS_BATCH_SIZE} (th.any/all) = "
          f"{6 * (BATCH_SIZE // LOSS_BATCH_SIZE)} total")
    print()

    data = make_batch(BATCH_SIZE, PEAKS_PER_SPEC, DEVICE)
    true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos = data

    def run_old():
        return _sparse_ce_old(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos,
            batch_size=BATCH_SIZE,
        )

    def run_new():
        return sparse_cross_entropy_vec(
            true_mzs, true_lp, true_bidxs, pred_mzs, pred_lp, pred_bidxs, pred_oos,
            tolerance=TOLERANCE, relative=RELATIVE, tolerance_min_mz=TOLERANCE_MIN_MZ,
            oos_tolerance_multiple=OOS_TOLERANCE_MULTIPLE,
            gaussian_renormalize=GAUSSIAN_RENORMALIZE, loss_batch_size=LOSS_BATCH_SIZE,
        )

    # ── Correctness check ─────────────────────────────────────────────────────
    ios_old, oos_old, _ = run_old()
    ios_new, oos_new, _, _ = run_new()
    ios_match = th.allclose(ios_old.cpu(), ios_new.cpu(), atol=1e-5)
    oos_match = th.allclose(oos_old.cpu(), oos_new.cpu(), atol=1e-5)
    print(f"Correctness: ios_ce {'✓' if ios_match else '✗ MISMATCH'}  "
          f"oos_ce {'✓' if oos_match else '✗ MISMATCH'}")
    print()

    # ── Speed ─────────────────────────────────────────────────────────────────
    old_ms = timed_run(run_old, (), REPEATS, WARMUP)
    new_ms = timed_run(run_new, (), REPEATS, WARMUP)
    speedup = old_ms / new_ms

    print("── Speed ──────────────────────────────────────────────────────")
    print(f"  old (with GPU syncs):    {old_ms:.3f} ms/call")
    print(f"  new (sync-free):         {new_ms:.3f} ms/call   ×{speedup:.2f} speedup")
    print()

    # ── Peak memory ──────────────────────────────────────────────────────────
    if DEVICE.type == "cuda":
        old_mb = peak_mb_run(run_old, ())
        new_mb = peak_mb_run(run_new, ())
        print("── Peak VRAM ΔMB ──────────────────────────────────────────────")
        print(f"  old: {old_mb:.1f} MB")
        print(f"  new: {new_mb:.1f} MB  (saved {old_mb - new_mb:.1f} MB)")
        print()

    # ── Sweep over peaks_per_spec ─────────────────────────────────────────────
    print("── Sweep: peaks_per_spec × speedup ────────────────────────────")
    print(f"  {'peaks/spec':>10}  {'old ms':>8}  {'new ms':>8}  {'speedup':>8}")
    for pps in [20, 50, 100, 200]:
        d = make_batch(BATCH_SIZE, pps, DEVICE)

        def _old():
            return _sparse_ce_old(*d, batch_size=BATCH_SIZE)

        def _new():
            return sparse_cross_entropy_vec(
                *d, tolerance=TOLERANCE, relative=RELATIVE,
                tolerance_min_mz=TOLERANCE_MIN_MZ,
                oos_tolerance_multiple=OOS_TOLERANCE_MULTIPLE,
                gaussian_renormalize=GAUSSIAN_RENORMALIZE,
                loss_batch_size=LOSS_BATCH_SIZE,
            )

        o = timed_run(_old, (), REPEATS, WARMUP)
        n = timed_run(_new, (), REPEATS, WARMUP)
        print(f"  {pps:>10}  {o:>8.3f}  {n:>8.3f}  {o/n:>8.2f}×")

    # ── Sweep over loss_batch_size ────────────────────────────────────────────
    print()
    print("── Sweep: loss_batch_size (more iters = more syncs in old) ────")
    print(f"  {'lbs':>6}  {'iters':>6}  {'old ms':>8}  {'new ms':>8}  {'speedup':>8}")
    d = make_batch(BATCH_SIZE, PEAKS_PER_SPEC, DEVICE)
    for lbs in [4, 8, 16, 32, 64]:
        iters = math.ceil(BATCH_SIZE / lbs)

        def _old_fn(_lbs=lbs):
            return _sparse_ce_old(*d, batch_size=BATCH_SIZE, loss_batch_size=_lbs)

        def _new_fn(_lbs=lbs):
            return sparse_cross_entropy_vec(
                *d, tolerance=TOLERANCE, relative=RELATIVE,
                tolerance_min_mz=TOLERANCE_MIN_MZ,
                oos_tolerance_multiple=OOS_TOLERANCE_MULTIPLE,
                gaussian_renormalize=GAUSSIAN_RENORMALIZE,
                loss_batch_size=_lbs,
            )

        o = timed_run(_old_fn, (), REPEATS, WARMUP)
        n = timed_run(_new_fn, (), REPEATS, WARMUP)
        print(f"  {lbs:>6}  {iters:>6}  {o:>8.3f}  {n:>8.3f}  {o/n:>8.2f}×")
