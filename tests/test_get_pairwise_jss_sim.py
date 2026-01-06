import sys
from pathlib import Path
import torch as th

# ensure src is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fragnnet.model.loss import get_pairwise_jss_sim


def test_pairwise_jss_sim_identical_peaks():
    mzs = th.tensor([50.0, 50.0])
    logprobs = th.tensor([0.0, 0.0])
    batch_idxs = th.tensor([0, 1], dtype=th.long)
    batch_size = 2
    mz_max = 100.0
    mz_bin_res = 1.0

    JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    expected = th.ones((2, 2), dtype=JSS.dtype)
    assert th.allclose(JSS, expected, atol=1e-6)


def test_pairwise_jss_sim_empty_input():
    mzs = th.tensor([], dtype=th.float32)
    logprobs = th.tensor([], dtype=th.float32)
    batch_idxs = th.tensor([], dtype=th.long)
    batch_size = 4
    mz_max = 100.0
    mz_bin_res = 1.0

    JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    assert JSS.shape == (batch_size, batch_size)
    assert th.allclose(JSS, th.zeros_like(JSS))
