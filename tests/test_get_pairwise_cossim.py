import sys
from pathlib import Path
import torch as th
import pytest

# ensure src is importable when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fragnnet.model.loss import get_pairwise_cossim


def test_pairwise_cossim_orthogonal_spectra():
    """Test cosine similarity for orthogonal (non-overlapping) spectra."""
    # Create two spectra with no overlapping peaks
    # Spectrum 0: mz=[50], logprob=[0]
    # Spectrum 1: mz=[150], logprob=[0]
    mzs = th.tensor([50.0, 150.0])
    logprobs = th.tensor([0.0, 0.0])
    batch_idxs = th.tensor([0, 1], dtype=th.long)
    batch_size = 2
    mz_max = 200.0
    mz_bin_res = 1.0

    S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

    # Diagonal should be 1 (self-similarity)
    assert th.allclose(S[0, 0], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
    assert th.allclose(S[1, 1], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
    # Off-diagonal: orthogonal vectors have zero cosine similarity (NaN becomes 0)
    assert S[0, 1] == 0.0 or th.isnan(S[0, 1]), "Orthogonal spectra should have similarity ~0"
    assert S[1, 0] == 0.0 or th.isnan(S[1, 0]), "Orthogonal spectra should have similarity ~0"


def test_pairwise_cossim_empty_input():
    """Test cosine similarity with empty spectra."""
    mzs = th.tensor([], dtype=th.float32)
    logprobs = th.tensor([], dtype=th.float32)
    batch_idxs = th.tensor([], dtype=th.long)
    batch_size = 3
    mz_max = 100.0
    mz_bin_res = 1.0

    S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    assert S.shape == (batch_size, batch_size)
    assert th.allclose(S, th.zeros_like(S))


def test_pairwise_cossim_symmetry():
    """Test that cosine similarity matrix is symmetric."""
    mzs = th.tensor([50.0, 75.0, 100.0, 50.0, 150.0])
    logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2])
    batch_idxs = th.tensor([0, 1, 0, 1, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Check symmetry
    assert th.allclose(S, S.t(), atol=1e-6), "Matrix should be symmetric"
    # Check diagonal is 1 (self-similarity)
    assert th.allclose(th.diag(S), th.ones(batch_size), atol=1e-6), "Diagonal should be 1"
    # Check bounds [0, 1]
    assert th.all(S >= -1e-6) and th.all(S <= 1 + 1e-6), "Similarity should be in [0, 1]"


def test_pairwise_cossim_chunking():
    """Test that chunked computation matches non-chunked."""
    mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
    logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
    batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    # Compute without chunking
    S_no_chunk = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None)
    
    # Compute with chunking
    S_chunk = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2)
    
    assert th.allclose(S_no_chunk, S_chunk, atol=1e-5), "Chunked and non-chunked results should match"


def test_pairwise_cossim_output_shape():
    """Test output shape is correct."""
    mzs = th.tensor([50.0, 100.0, 150.0])
    logprobs = th.tensor([0.0, 1.0, 0.5])
    batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
    batch_size = 3
    mz_max = 200.0
    mz_bin_res = 1.0

    S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    assert S.shape == (batch_size, batch_size), f"Expected shape {(batch_size, batch_size)}, got {S.shape}"


def test_pairwise_cossim_orthogonal_spectra():
    """Test orthogonal spectra have zero similarity."""
    # Spectrum 1: peak at 50 Da
    # Spectrum 2: peak at 150 Da (non-overlapping)
    mzs = th.tensor([50.0, 150.0])
    logprobs = th.tensor([0.0, 0.0])
    batch_idxs = th.tensor([0, 1], dtype=th.long)
    batch_size = 2
    mz_max = 200.0
    mz_bin_res = 1.0

    S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
    
    # Off-diagonal should be 0 (no overlap)
    assert th.allclose(S[0, 1], th.tensor(0.0), atol=1e-6), "Non-overlapping spectra should have 0 similarity"
    assert th.allclose(S[1, 0], th.tensor(0.0), atol=1e-6), "Non-overlapping spectra should have 0 similarity"
