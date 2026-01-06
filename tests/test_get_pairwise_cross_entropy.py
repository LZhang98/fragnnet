import sys
from pathlib import Path
import torch as th

# ensure src is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fragnnet.model.loss import get_pairwise_cross_entropy


def test_pairwise_cross_entropy_identical_peaks_zero_ce():
    mzs = th.tensor([75.0, 75.0])
    logprobs = th.tensor([0.0, 0.0])
    batch_idxs = th.tensor([0, 1], dtype=th.long)
    batch_size = 2
    mz_max = 200.0
    mz_bin_res = 1.0

    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    expected = th.zeros((2, 2), dtype=CE.dtype)
    assert th.allclose(CE, expected, atol=1e-6)


def test_pairwise_cross_entropy_empty_input():
    mzs = th.tensor([], dtype=th.float32)
    logprobs = th.tensor([], dtype=th.float32)
    batch_idxs = th.tensor([], dtype=th.long)
    batch_size = 3
    mz_max = 100.0
    mz_bin_res = 1.0

    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    assert CE.shape == (batch_size, batch_size)
    assert th.allclose(CE, th.zeros_like(CE))
