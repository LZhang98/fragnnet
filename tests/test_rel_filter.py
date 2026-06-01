import torch

from fragnnet.utils.spec_utils import batched_rel_filter_func


def _make_batch(spectra: list[list[tuple[float, float]]]):
    """Convert list of (mz, int) spectra into flat tensors."""
    mzs, ints, batch_idxs = [], [], []
    for i, peaks in enumerate(spectra):
        for mz, intensity in peaks:
            mzs.append(mz)
            ints.append(intensity)
            batch_idxs.append(i)
    return (
        torch.tensor(mzs, dtype=torch.float),
        torch.tensor(ints, dtype=torch.float),
        torch.tensor(batch_idxs, dtype=torch.long),
    )


def test_rel_filter_drops_below_threshold():
    # Spectrum: max=800, so 1% threshold = 8.0. Peaks at 4.0 and 2.0 should be dropped.
    mzs, ints, batch_idxs = _make_batch([[(100.0, 800.0), (200.0, 4.0), (300.0, 2.0), (400.0, 100.0)]])
    out_mzs, out_ints, out_batch = batched_rel_filter_func(mzs, ints, batch_idxs, rel_thresh=0.01)
    assert out_mzs.tolist() == [100.0, 400.0]
    assert out_ints.tolist() == [800.0, 100.0]
    assert out_batch.tolist() == [0, 0]


def test_rel_filter_keeps_boundary_peak():
    # Peak exactly at threshold (= max * rel_thresh) is dropped (strict >).
    mzs, ints, batch_idxs = _make_batch([[(100.0, 1000.0), (200.0, 10.0), (300.0, 9.9)]])
    out_mzs, _, _ = batched_rel_filter_func(mzs, ints, batch_idxs, rel_thresh=0.01)
    # 10.0 == 1% of 1000 → dropped (not strictly greater); 9.9 < 10 → also dropped
    assert out_mzs.tolist() == [100.0]


def test_rel_filter_zero_threshold_noop():
    mzs, ints, batch_idxs = _make_batch([[(100.0, 50.0), (200.0, 1.0)]])
    out_mzs, out_ints, out_batch = batched_rel_filter_func(mzs, ints, batch_idxs, rel_thresh=0.0)
    assert out_mzs.tolist() == [100.0, 200.0]


def test_rel_filter_single_peak_survives():
    # Single peak is always its own max → ratio = 1.0, always > any rel_thresh < 1.
    mzs, ints, batch_idxs = _make_batch([[(300.0, 42.0)]])
    out_mzs, _, _ = batched_rel_filter_func(mzs, ints, batch_idxs, rel_thresh=0.99)
    assert out_mzs.tolist() == [300.0]


def test_rel_filter_per_spectrum_independent():
    # Two spectra with very different max values; threshold must be computed per-spectrum.
    # Spectrum 0: max=1000, 1% = 10.  Peak at 5.0 should be dropped.
    # Spectrum 1: max=10,   1% = 0.1. Peak at 5.0 should be kept.
    mzs, ints, batch_idxs = _make_batch([
        [(100.0, 1000.0), (200.0, 5.0)],
        [(300.0, 10.0), (400.0, 5.0)],
    ])
    out_mzs, _, out_batch = batched_rel_filter_func(mzs, ints, batch_idxs, rel_thresh=0.01)
    assert 200.0 not in out_mzs.tolist(), "peak below 1% of spectrum-0 max should be dropped"
    assert 400.0 in out_mzs.tolist(), "peak above 1% of spectrum-1 max should be kept"


def test_rel_filter_mz_max():
    # mz_max=250 should drop peaks at mz >= 250 independently of intensity.
    mzs, ints, batch_idxs = _make_batch([[(100.0, 999.0), (300.0, 999.0)]])
    out_mzs, _, _ = batched_rel_filter_func(mzs, ints, batch_idxs, rel_thresh=0.0, mz_max=250.0)
    assert out_mzs.tolist() == [100.0]
