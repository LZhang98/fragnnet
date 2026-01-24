"""Unit tests for fragnnet.model.loss module.

Tests for pairwise similarity and distance metrics:
- Cosine similarity
- Cross-entropy
- Jensen-Shannon divergence
"""

import math

import pytest
import torch as th

from fragnnet.model.loss import (
    _compute_pair_ce_hun,
    _compute_pair_jss_hun,
    get_pairwise_cossim,
    get_pairwise_cross_entropy,
    get_pairwise_jss_sim,
    get_pairwise_jss_sim_hun,
)


class TestPairwiseCossim:
    """Tests for cosine similarity computation."""

    def test_pairwise_cossim_orthogonal_spectra(self):
        """Test cosine similarity for orthogonal (non-overlapping) spectra."""
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
        # Off-diagonal: orthogonal vectors have zero cosine similarity
        assert S[0, 1] == 0.0 or th.isnan(S[0, 1]), "Orthogonal spectra should have similarity ~0"
        assert S[1, 0] == 0.0 or th.isnan(S[1, 0]), "Orthogonal spectra should have similarity ~0"

    def test_pairwise_cossim_empty_input(self):
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

    def test_pairwise_cossim_symmetry(self):
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

    def test_pairwise_cossim_chunking(self):
        """Test that chunked computation matches non-chunked."""
        mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        # Compute without chunking
        S_no_chunk = get_pairwise_cossim(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None
        )

        # Compute with chunking
        S_chunk = get_pairwise_cossim(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2
        )

        assert th.allclose(S_no_chunk, S_chunk, atol=1e-5), (
            "Chunked and non-chunked results should match"
        )

    def test_pairwise_cossim_output_shape(self):
        """Test output shape is correct."""
        mzs = th.tensor([50.0, 100.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5])
        batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert S.shape == (batch_size, batch_size), (
            f"Expected shape {(batch_size, batch_size)}, got {S.shape}"
        )


class TestPairwiseCrossEntropy:
    """Tests for cross-entropy computation."""

    def test_pairwise_cross_entropy_disjoint_spectra(self):
        """Test cross-entropy for disjoint (non-overlapping) spectra."""
        mzs = th.tensor([50.0, 150.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        # Diagonal should be 0 (CE between identical distributions)
        assert th.allclose(th.diag(CE), th.zeros(batch_size), atol=1e-6), "Diagonal should be 0"
        # Off-diagonal for completely disjoint spectra should be high
        assert CE[0, 1] > 5.0, "CE for disjoint spectra should be high"
        assert CE[1, 0] > 5.0, "CE for disjoint spectra should be high"

    def test_pairwise_cross_entropy_empty_input(self):
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

    def test_pairwise_cross_entropy_non_symmetric(self):
        """Test that cross-entropy is not necessarily symmetric."""
        mzs = th.tensor([50.0, 100.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5])
        batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        # Diagonal should be 0
        assert th.allclose(th.diag(CE), th.zeros(batch_size), atol=1e-6), "Diagonal should be 0"

    def test_pairwise_cross_entropy_chunking(self):
        """Test that chunked computation matches non-chunked."""
        mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        # Compute without chunking
        CE_no_chunk = get_pairwise_cross_entropy(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None
        )

        # Compute with chunking
        CE_chunk = get_pairwise_cross_entropy(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2
        )

        assert th.allclose(CE_no_chunk, CE_chunk, atol=1e-5), (
            "Chunked and non-chunked results should match"
        )

    def test_pairwise_cross_entropy_output_shape(self):
        """Test output shape is correct."""
        mzs = th.tensor([50.0, 100.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5])
        batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        CE = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert CE.shape == (batch_size, batch_size), (
            f"Expected shape {(batch_size, batch_size)}, got {CE.shape}"
        )

    def test_pairwise_cross_entropy_non_negative(self):
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

    def test_pairwise_cross_entropy_multiple_chunks(self):
        """Test with different chunk sizes."""
        mzs = th.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8, 0.3, 1.2])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2, 3, 3], dtype=th.long)
        batch_size = 4
        mz_max = 100.0
        mz_bin_res = 1.0

        # Test with various chunk sizes
        for chunk_size in [1, 2, 4, None]:
            CE = get_pairwise_cross_entropy(
                mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=chunk_size
            )
            assert CE.shape == (batch_size, batch_size)


class TestPairwiseJssSim:
    """Tests for Jensen-Shannon divergence based similarity."""

    def test_pairwise_jss_sim_single_peak(self):
        """Test JSS similarity for single-peak spectra."""
        mzs = th.tensor([50.0, 100.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        # Diagonal should be 1 (self-similarity)
        assert th.allclose(th.diag(JSS), th.ones(batch_size), atol=1e-6), "Diagonal should be 1"
        # Off-diagonal for different peaks should be small
        assert JSS[0, 1] < 0.5, "Different spectra should have lower similarity"
        assert JSS[1, 0] < 0.5, "Different spectra should have lower similarity"
        # Symmetry
        assert th.allclose(JSS[0, 1], JSS[1, 0], atol=1e-6), "JSS should be symmetric"

    def test_pairwise_jss_sim_empty_input(self):
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

    def test_pairwise_jss_sim_symmetry(self):
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

    def test_pairwise_jss_sim_chunking(self):
        """Test that chunked computation matches non-chunked."""
        mzs = th.tensor([50.0, 75.0, 100.0, 125.0, 150.0, 175.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        # Compute without chunking
        JSS_no_chunk = get_pairwise_jss_sim(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=None
        )

        # Compute with chunking
        JSS_chunk = get_pairwise_jss_sim(
            mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=2
        )

        assert th.allclose(JSS_no_chunk, JSS_chunk, atol=1e-5), (
            "Chunked and non-chunked results should match"
        )

    def test_pairwise_jss_sim_output_shape(self):
        """Test output shape is correct."""
        mzs = th.tensor([50.0, 100.0, 150.0])
        logprobs = th.tensor([0.0, 1.0, 0.5])
        batch_idxs = th.tensor([0, 1, 2], dtype=th.long)
        batch_size = 3
        mz_max = 200.0
        mz_bin_res = 1.0

        JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        assert JSS.shape == (batch_size, batch_size), (
            f"Expected shape {(batch_size, batch_size)}, got {JSS.shape}"
        )

    def test_pairwise_jss_sim_orthogonal_spectra(self):
        """Test orthogonal spectra have lower similarity."""
        mzs = th.tensor([50.0, 150.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        JSS = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        # Off-diagonal should be 0 (no overlap)
        assert th.allclose(JSS[0, 1], th.tensor(0.0), atol=1e-6), (
            "Non-overlapping spectra should have 0 similarity"
        )
        assert th.allclose(JSS[1, 0], th.tensor(0.0), atol=1e-6), (
            "Non-overlapping spectra should have 0 similarity"
        )

    def test_pairwise_jss_sim_multiple_chunks(self):
        """Test with different chunk sizes."""
        mzs = th.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        logprobs = th.tensor([0.0, 1.0, 0.5, 1.5, 0.2, 0.8, 0.3, 1.2])
        batch_idxs = th.tensor([0, 1, 0, 1, 2, 2, 3, 3], dtype=th.long)
        batch_size = 4
        mz_max = 100.0
        mz_bin_res = 1.0

        # Test with various chunk sizes
        for chunk_size in [1, 2, 4, None]:
            JSS = get_pairwise_jss_sim(
                mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res, chunk_size=chunk_size
            )
            assert JSS.shape == (batch_size, batch_size)


class TestHungarianJSS:
    """Tests for JSS computation using Hungarian matching."""

    def test_compute_pair_jss_hun_identical(self):
        """Test JSS for identical spectra is 1.0."""
        mzs = th.tensor([100.0, 200.0])
        logprobs = th.tensor([0.0, 0.0])  # Equal intensities

        sim = _compute_pair_jss_hun(
            mzs, logprobs, mzs, logprobs, tolerance=1e-5, relative=True, tolerance_min_mz=1.0
        )
        assert th.allclose(sim, th.tensor(1.0), atol=1e-6)

    def test_compute_pair_jss_hun_orthogonal(self):
        """Test JSS for orthogonal spectra is 0.0."""
        mzs1 = th.tensor([100.0])
        logprobs1 = th.tensor([0.0])
        mzs2 = th.tensor([200.0])
        logprobs2 = th.tensor([0.0])

        sim = _compute_pair_jss_hun(
            mzs1, logprobs1, mzs2, logprobs2, tolerance=1e-5, relative=True, tolerance_min_mz=1.0
        )
        assert th.allclose(sim, th.tensor(0.0), atol=1e-6)

    def test_compute_pair_jss_hun_tolerance(self):
        """Test spectra matching within tolerance."""
        mzs1 = th.tensor([100.0])
        logprobs1 = th.tensor([0.0])
        mzs2 = th.tensor([100.0001])
        logprobs2 = th.tensor([0.0])

        # Within tolerance
        sim = _compute_pair_jss_hun(
            mzs1, logprobs1, mzs2, logprobs2, tolerance=0.01, relative=False, tolerance_min_mz=1.0
        )
        assert th.allclose(sim, th.tensor(1.0), atol=1e-6)

        # Outside tolerance
        sim_out = _compute_pair_jss_hun(
            mzs1, logprobs1, mzs2, logprobs2, tolerance=1e-6, relative=False, tolerance_min_mz=1.0
        )
        assert th.allclose(sim_out, th.tensor(0.0), atol=1e-6)

    def test_get_pairwise_jss_sim_hun(self):
        """Test the pairwise wrapper for Hungarian JSS."""
        mzs = th.tensor([100.0, 200.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2

        S = get_pairwise_jss_sim_hun(
            mzs, logprobs, batch_idxs, batch_size, tolerance=1e-5, relative=True
        )

        assert S.shape == (2, 2)
        assert th.allclose(S[0, 0], th.tensor(1.0))
        assert th.allclose(S[1, 1], th.tensor(1.0))
        assert th.allclose(S[0, 1], th.tensor(0.0), atol=1e-6)
        assert th.allclose(S[1, 0], th.tensor(0.0), atol=1e-6)


class TestHungarianCE:
    """Tests for Cross-Entropy computation using Hungarian matching."""

    def test_compute_pair_ce_hun_identical(self):
        """Test CE for identical spectra is low (minimizing CE)."""
        mzs = th.tensor([100.0])
        logprobs = th.tensor([0.0])

        ce = _compute_pair_ce_hun(
            mzs, logprobs, mzs, logprobs, tolerance=1e-5, relative=True, tolerance_min_mz=1.0
        )
        # For P=Q=[1], CE = -sum(1 * log(1)) = 0.
        # However, we use log(1+eps) for stability, so it should be close to 0.
        assert ce < 1e-3

    def test_compute_pair_ce_hun_orthogonal(self):
        """Test CE for orthogonal spectra is high."""
        mzs1 = th.tensor([100.0])
        logprobs1 = th.tensor([0.0])
        mzs2 = th.tensor([200.0])
        logprobs2 = th.tensor([0.0])

        ce = _compute_pair_ce_hun(
            mzs1, logprobs1, mzs2, logprobs2, tolerance=1e-5, relative=True, tolerance_min_mz=1.0
        )
        # For orthogonal, it should mismatch and use log_eps.
        # CE = -log(eps)

        from fragnnet.model.loss import EPS

        expected = -math.log(EPS)
        assert th.allclose(ce, th.tensor(expected), atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
