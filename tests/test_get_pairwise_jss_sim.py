import sys
from pathlib import Path
import torch as th
import pytest
import math

# ensure src is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fragnnet.model.loss import get_pairwise_jss_sim


def test_pairwise_jss_sim_single_peak():
    """Test JSS similarity for single-peak spectra."""
    # Create two spectra, each with a single peak
    # Spectrum 0: mz=[50], logprob=[0]
    # Spectrum 1: mz=[100], logprob=[0]
    mzs = th.tensor([50.0, 100.0])
    logprobs = th.tensor([0.0, 0.0])
    batch_idxs = th.tensor([0, 1], dtype=th.long)
    batch_size = 2
    mz_max = 200.0
    mz_bin_res = 1.0

    JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    # Diagonal should be 1 (self-similarity)
    assert th.allclose(th.diag(JSS), th.ones(batch_size), atol=1e-6), "Diagonal should be 1"
    # Off-diagonal for different peaks should be small (close to 0, far apart spectra)
    assert JSS[0, 1] < 0.5, "Different spectra should have lower similarity"
    assert JSS[1, 0] < 0.5, "Different spectra should have lower similarity"
    # Symmetry
    assert th.allclose(JSS[0, 1], JSS[1, 0], atol=1e-6), "JSS should be symmetric"


def test_pairwise_jss_sim_empty_input():
    """Test JSS similarity with empty spectra."""
    mzs = th.tensor([], dtype=th.float32)
    logprobs = th.tensor([], dtype=th.float32)
    batch_idxs = th.tensor([], dtype=th.long)
    batch_size = 4
    mz_max = 100.0
    mz_bin_res = 1.0

    JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    assert JSS.shape == (batch_size, batch_size)
    assert th.allclose(JSS, th.zeros_like(JSS))


def test_pairwise_jss_sim_symmetry():
    """Test that JSS similarity matrix is symmetric."""
    mzs = th.tensor([50.0, 75.0, 100.0, 50.0, 150.0])
    logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2])
    batch_idxs = th.tensor([0, 1, 0, 1, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Check symmetry
    assert th.allclose(JSS, JSS.t(), atol=1e-6), "Matrix should be symmetric"
    # Check diagonal is 1 (self-similarity)
    assert th.allclose(th.diag(JSS), th.ones(batch_size), atol=1e-6), "Diagonal should be 1"
    # Check bounds [0, 1]
    assert th.all(JSS >= -1e-6) and th.all(JSS <= 1 + 1e-6), "JSS should be in [0, 1]"


def test_pairwise_jss_sim_chunking():
    """Test that chunked computation matches non-chunked."""
    mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
    logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
    batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    # Compute without chunking
    JSS_no_chunk = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None)
    
    # Compute with chunking
    JSS_chunk = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2)
    
    assert th.allclose(JSS_no_chunk, JSS_chunk, atol=1e-5), "Chunked and non-chunked results should match"


def test_pairwise_jss_sim_output_shape():
    """Test output shape is correct."""
    mzs = th.tensor([50.0, 100.0, 150.0])
    logprobs = th.tensor([0.0, 1.0, 0.5])
    batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    assert JSS.shape == (batch_size, batch_size), f"Expected shape {(batch_size, batch_size)}, got {JSS.shape}"


def test_pairwise_jss_sim_orthogonal_spectra():
    """Test orthogonal spectra have lower similarity."""
    # Spectrum 1: peak at 50 Da
    # Spectrum 2: peak at 150 Da (non-overlapping)
    mzs = th.tensor([50.0, 150.0])
    logprobs = th.tensor([0.0, 0.0])
    batch_idxs = th.tensor([0, 1], dtype=th.long)
    batch_size = 2
    mz_max = 200.0
    mz_bin_res = 1.0

    JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Off-diagonal should be 0 (no overlap means JSD is log(2), JSS is 1-1=0)
    assert th.allclose(JSS[0, 1], th.tensor(0.0), atol=1e-6), "Non-overlapping spectra should have 0 similarity"
    assert th.allclose(JSS[1, 0], th.tensor(0.0), atol=1e-6), "Non-overlapping spectra should have 0 similarity"


def test_pairwise_jss_sim_multiple_chunks():
    """Test with different chunk sizes."""
    mzs = th.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8, 0.3, 1.2])
    batch_idxs = th.tensor([0, 1, 0, 1, 2, 2, 3, 3], dtype=th.long)
    batch_size = 4
    mz_max = 100.0
    mz_bin_res = 1.0

    # Test with various chunk sizes
    JSS_ref = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None)
    
    for chunk_size in [1, 2, 3, 4]:
        JSS_chunk = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=chunk_size)
        assert th.allclose(JSS_ref, JSS_chunk, atol=1e-5), f"chunk_size={chunk_size} should match non-chunked"
