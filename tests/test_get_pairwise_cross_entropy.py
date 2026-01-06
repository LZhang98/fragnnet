import sys
from pathlib import Path
import torch as th
import pytest

# ensure src is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fragnnet.model.loss import get_pairwise_cross_entropy


def test_pairwise_cross_entropy_disjoint_spectra():
    """Test cross-entropy for disjoint (non-overlapping) spectra."""
    # Create two spectra with no overlapping peaks
    # Spectrum 0: mz=[50], logprob=[0]
    # Spectrum 1: mz=[150], logprob=[0]
    mzs = th.tensor([50.0, 150.0])
    logprobs = th.tensor([0.0, 0.0])
    batch_idxs = th.tensor([0, 1], dtype=th.long)
    batch_size = 2
    mz_max = 200.0
    mz_bin_res = 1.0

    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    # Diagonal should be 0 (CE between identical distributions)
    assert th.allclose(th.diag(CE), th.zeros(batch_size), atol=1e-6), "Diagonal should be 0"
    # Off-diagonal for completely disjoint spectra should be high (each has peaks the other doesn't)
    assert CE[0, 1] > 5.0, "CE for disjoint spectra should be high"
    assert CE[1, 0] > 5.0, "CE for disjoint spectra should be high"
    # CE is not symmetric: CE(P,Q) != CE(Q,P)
    # So we just check that values are reasonable


def test_pairwise_cross_entropy_empty_input():
    """Test cross-entropy with empty spectra."""
    mzs = th.tensor([], dtype=th.float32)
    logprobs = th.tensor([], dtype=th.float32)
    batch_idxs = th.tensor([], dtype=th.long)
    batch_size = 3
    mz_max = 100.0
    mz_bin_res = 1.0

    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    assert CE.shape == (batch_size, batch_size)
    assert th.allclose(CE, th.zeros_like(CE))


def test_pairwise_cross_entropy_non_symmetric():
    """Test that cross-entropy is not necessarily symmetric."""
    mzs = th.tensor([50.0, 100.0, 150.0])
    logprobs = th.tensor([0.0, 1.0, 0.5])
    batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Cross-entropy is not symmetric: CE(P,Q) != CE(Q,P) in general
    # But diagonal should be 0 (CE(P,P) = 0)
    assert th.allclose(th.diag(CE), th.zeros(batch_size), atol=1e-6), "Diagonal should be 0"


def test_pairwise_cross_entropy_chunking():
    """Test that chunked computation matches non-chunked."""
    mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
    logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
    batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    # Compute without chunking
    CE_no_chunk = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None)
    
    # Compute with chunking
    CE_chunk = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2)
    
    assert th.allclose(CE_no_chunk, CE_chunk, atol=1e-5), "Chunked and non-chunked results should match"


def test_pairwise_cross_entropy_output_shape():
    """Test output shape is correct."""
    mzs = th.tensor([50.0, 100.0, 150.0])
    logprobs = th.tensor([0.0, 1.0, 0.5])
    batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    assert CE.shape == (batch_size, batch_size), f"Expected shape {(batch_size, batch_size)}, got {CE.shape}"


def test_pairwise_cross_entropy_non_negative():
    """Test that cross-entropy is non-negative."""
    mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0])
    logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2])
    batch_idxs = th.tensor([0, 1, 0, 1, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Cross-entropy should be non-negative
    assert th.all(CE >= -1e-6), "Cross-entropy should be non-negative"


def test_pairwise_cross_entropy_multiple_chunks():
    """Test with different chunk sizes."""
    mzs = th.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8, 0.3, 1.2])
    batch_idxs = th.tensor([0, 1, 0, 1, 2, 2, 3, 3], dtype=th.long)
    batch_size = 4
    mz_max = 100.0
    mz_bin_res = 1.0

    # Test with various chunk sizes
    CE_ref = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None)
    
    for chunk_size in [1, 2, 3, 4]:
        CE_chunk = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=chunk_size)
        assert th.allclose(CE_ref, CE_chunk, atol=1e-5), f"chunk_size={chunk_size} should match non-chunked"


def test_pairwise_cross_entropy_disjoint_spectra():
    """Test cross-entropy with disjoint (non-overlapping) spectra."""
    # Spectrum 1: peak at 50 Da (with high intensity)
    # Spectrum 2: peak at 150 Da (non-overlapping)
    mzs = th.tensor([50.0, 150.0])
    logprobs = th.tensor([0.0, 0.0])
    batch_idxs = th.tensor([0, 1], dtype=th.long)
    batch_size = 2
    mz_max = 200.0
    mz_bin_res = 1.0

    CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # CE(P,Q) should be high when P has intensity but Q is empty
    assert CE[0, 1] > 0, "Cross-entropy from spec1 to spec2 should be positive (spec1 has peak, spec2 doesn't)"
    assert CE[1, 0] > 0, "Cross-entropy from spec2 to spec1 should be positive (spec2 has peak, spec1 doesn't)"
