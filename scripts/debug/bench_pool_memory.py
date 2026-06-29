"""Benchmark scatter (index-based) vs segment_csr (ptr-based) pooling.

PyG's Aggregation.reduce uses segment_csr ONLY when ptr is not None AND index is None
(or _deterministic=True). Passing both index and ptr keeps the scatter path active.

Compares memory and speed for mean/max pool types, reflecting the change in
fragnnet_model.py that calls frag_pool(embed, index=None, ptr=frag_ptr) for mean/max.

Usage:
    conda run -n FRAGNNET-GPU python scripts/debug/bench_pool_memory.py
"""

import gc
import time

import torch as th
from torch_geometric.nn import aggr

# ── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZE = 256  # molecules per batch
FRAGS_PER_MOL = 40  # average fragments per molecule
HIDDEN = 256  # fragment embedding dimension
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
WARMUP = 5
REPEATS = 20

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_batch(
    batch_size: int, frags_per_mol: int, hidden: int, device: th.device
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Return (x, index, ptr) for a uniform batch."""
    num_frags = batch_size * frags_per_mol
    x = th.randn(num_frags, hidden, device=device)
    index = th.repeat_interleave(th.arange(batch_size, device=device), frags_per_mol)
    counts = th.full((batch_size,), frags_per_mol, dtype=th.long, device=device)
    ptr = th.cat([counts.new_zeros(1), counts.cumsum(0)])
    return x, index, ptr


def reset_peak() -> None:
    if DEVICE.type == "cuda":
        th.cuda.synchronize()
        th.cuda.reset_peak_memory_stats()


def peak_mb() -> float:
    if DEVICE.type == "cuda":
        th.cuda.synchronize()
        return th.cuda.max_memory_allocated() / 1024**2
    return float("nan")


def current_mb() -> float:
    if DEVICE.type == "cuda":
        return th.cuda.memory_allocated() / 1024**2
    return float("nan")


def run_trial(pool_fn, x, index, ptr, use_ptr: bool) -> float:
    """Run one forward pass and return peak GPU memory delta in MB."""
    gc.collect()
    if DEVICE.type == "cuda":
        th.cuda.empty_cache()
    baseline = current_mb()
    reset_peak()
    if use_ptr:
        # index=None activates segment_csr path in PyG's Aggregation.reduce
        _ = pool_fn(x, index=None, ptr=ptr, dim_size=BATCH_SIZE)
    else:
        _ = pool_fn(x, index=index, dim_size=BATCH_SIZE)
    if DEVICE.type == "cuda":
        th.cuda.synchronize()
    return peak_mb() - baseline


def bench(
    pool_fn, x: th.Tensor, index: th.Tensor, ptr: th.Tensor, label: str
) -> None:
    """Warm up then report mean ± std peak memory."""
    for _ in range(WARMUP):
        run_trial(pool_fn, x, index, ptr, use_ptr=False)
        run_trial(pool_fn, x, index, ptr, use_ptr=True)

    scatter_peaks = [run_trial(pool_fn, x, index, ptr, use_ptr=False) for _ in range(REPEATS)]
    segcsr_peaks = [run_trial(pool_fn, x, index, ptr, use_ptr=True) for _ in range(REPEATS)]

    scatter_t = th.tensor(scatter_peaks)
    segcsr_t = th.tensor(segcsr_peaks)

    print(f"\n── {label} ──────────────────────────────────────────")
    print(f"  scatter  (index only) peak ΔMB: {scatter_t.mean():.3f} ± {scatter_t.std():.3f}")
    print(f"  segment_csr (ptr only) peak ΔMB: {segcsr_t.mean():.3f} ± {segcsr_t.std():.3f}")
    saved = scatter_t.mean() - segcsr_t.mean()
    pct = saved / scatter_t.mean() * 100 if scatter_t.mean() > 0 else 0.0
    print(f"  saved: {saved:.3f} MB  ({pct:.1f}%)")


def bench_speed(
    pool_fn, x: th.Tensor, index: th.Tensor, ptr: th.Tensor, label: str
) -> None:
    """Time scatter vs segment_csr with CPU-side wall clock (after sync)."""
    if DEVICE.type != "cuda":
        print(f"\n── {label} speed (CPU — skipping timing) ──")
        return

    def timed(use_ptr: bool) -> float:
        for _ in range(WARMUP):
            if use_ptr:
                pool_fn(x, index=None, ptr=ptr, dim_size=BATCH_SIZE)
            else:
                pool_fn(x, index=index, dim_size=BATCH_SIZE)
        th.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEATS):
            if use_ptr:
                pool_fn(x, index=None, ptr=ptr, dim_size=BATCH_SIZE)
            else:
                pool_fn(x, index=index, dim_size=BATCH_SIZE)
        th.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / REPEATS  # ms per call

    scatter_ms = timed(use_ptr=False)
    segcsr_ms = timed(use_ptr=True)
    speedup = scatter_ms / segcsr_ms
    print(f"\n── {label} speed ───────────────────────────────────")
    print(f"  scatter (index only):   {scatter_ms:.3f} ms/call")
    print(f"  segment_csr (ptr only): {segcsr_ms:.3f} ms/call  (×{speedup:.2f} speedup)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(
        f"Shape: {BATCH_SIZE} mols × {FRAGS_PER_MOL} frags = "
        f"{BATCH_SIZE * FRAGS_PER_MOL} nodes, hidden={HIDDEN}"
    )
    print(
        f"index tensor: ({BATCH_SIZE * FRAGS_PER_MOL},)  "
        f"ptr tensor: ({BATCH_SIZE + 1},)"
    )

    x, index, ptr = make_batch(BATCH_SIZE, FRAGS_PER_MOL, HIDDEN, DEVICE)
    sum_agg = aggr.SumAggregation().to(DEVICE)
    mean_agg = aggr.MeanAggregation().to(DEVICE)
    max_agg = aggr.MaxAggregation().to(DEVICE)

    bench(sum_agg, x, index, ptr, "SumAggregation")
    bench(mean_agg, x, index, ptr, "MeanAggregation")
    bench(max_agg, x, index, ptr, "MaxAggregation")

    bench_speed(sum_agg, x, index, ptr, "SumAggregation")
    bench_speed(mean_agg, x, index, ptr, "MeanAggregation")
    bench_speed(max_agg, x, index, ptr, "MaxAggregation")

    # Larger batch to amplify differences
    print("\n\n=== Larger batch (×4) ===")
    BATCH_SIZE_L = BATCH_SIZE * 4
    x2, index2, ptr2 = make_batch(BATCH_SIZE_L, FRAGS_PER_MOL, HIDDEN, DEVICE)

    def _bench_large(pool_fn, label):
        for _ in range(WARMUP):
            pool_fn(x2, index=index2, dim_size=BATCH_SIZE_L)
            pool_fn(x2, index=None, ptr=ptr2, dim_size=BATCH_SIZE_L)
        scatter_peaks = []
        segcsr_peaks = []
        for _ in range(REPEATS):
            gc.collect()
            if DEVICE.type == "cuda":
                th.cuda.empty_cache()
            b = current_mb()
            reset_peak()
            pool_fn(x2, index=index2, dim_size=BATCH_SIZE_L)
            if DEVICE.type == "cuda":
                th.cuda.synchronize()
            scatter_peaks.append(peak_mb() - b)

            gc.collect()
            if DEVICE.type == "cuda":
                th.cuda.empty_cache()
            b = current_mb()
            reset_peak()
            pool_fn(x2, index=None, ptr=ptr2, dim_size=BATCH_SIZE_L)
            if DEVICE.type == "cuda":
                th.cuda.synchronize()
            segcsr_peaks.append(peak_mb() - b)
        s = th.tensor(scatter_peaks)
        g = th.tensor(segcsr_peaks)
        saved = s.mean() - g.mean()
        pct = saved / s.mean() * 100 if s.mean() > 0 else 0.0
        print(f"\n── {label} (large) ──────────────────────────────")
        print(f"  scatter   peak ΔMB: {s.mean():.3f} ± {s.std():.3f}")
        print(f"  segment_csr peak ΔMB: {g.mean():.3f} ± {g.std():.3f}")
        print(f"  saved: {saved:.3f} MB  ({pct:.1f}%)")

    _bench_large(sum_agg, "SumAggregation")
    _bench_large(mean_agg, "MeanAggregation")
    _bench_large(max_agg, "MaxAggregation")

    if DEVICE.type == "cuda":
        def timed_large(pool_fn, use_ptr: bool) -> float:
            for _ in range(WARMUP):
                if use_ptr:
                    pool_fn(x2, index=None, ptr=ptr2, dim_size=BATCH_SIZE_L)
                else:
                    pool_fn(x2, index=index2, dim_size=BATCH_SIZE_L)
            th.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                if use_ptr:
                    pool_fn(x2, index=None, ptr=ptr2, dim_size=BATCH_SIZE_L)
                else:
                    pool_fn(x2, index=index2, dim_size=BATCH_SIZE_L)
            th.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000 / REPEATS

        for pool_fn, label in [(sum_agg, "SumAggregation"), (mean_agg, "MeanAggregation"), (max_agg, "MaxAggregation")]:
            s_ms = timed_large(pool_fn, use_ptr=False)
            g_ms = timed_large(pool_fn, use_ptr=True)
            print(f"\n── {label} (large) speed ────────────────────────")
            print(f"  scatter:     {s_ms:.3f} ms/call")
            print(f"  segment_csr: {g_ms:.3f} ms/call  (×{s_ms / g_ms:.2f} speedup)")
