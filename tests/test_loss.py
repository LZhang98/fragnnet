"""Unit tests for the fragnnet.loss modules.

Tests for pairwise similarity and distance metrics:
- Cosine similarity
- Cross-entropy
- Jensen-Shannon divergence
"""

import math

import pytest
import torch as th

from fragnnet.loss.cross_entropy import conditional_in_support_cross_entropy
from fragnnet.loss.pairwise import (
    get_pairwise_cossim,
    get_pairwise_cross_entropy,
    get_pairwise_jss_sim,
)
from fragnnet.loss.peak_list import (
    sparse_outside_window_intensity_fraction,
    sparse_peak_list_distance,
)
from fragnnet.loss.spectrum_distance import (
    sparse_hellinger_distance,
    sparse_jensen_shannon_divergence,
    sparse_jensen_shannon_divergence_hungarian,
    sparse_jensen_shannon_divergence_hungarian_vec,
)
from fragnnet.utils.spec_similarity import calculate_match_mzs
from fragnnet.utils.spec_utils import batched_bin_func


def test_sparse_hellinger_distance_matches_known_value_and_backward() -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_logprobs = th.tensor([0.75, 0.25]).log()
    pred_mzs = th.tensor([100.0, 200.0])
    pred_logprobs = th.tensor([0.25, 0.75]).log().requires_grad_()
    batch_idxs = th.zeros(2, dtype=th.long)

    distance = sparse_hellinger_distance(
        true_mzs,
        true_logprobs,
        batch_idxs,
        pred_mzs,
        pred_logprobs,
        batch_idxs,
    )
    distance.sum().backward()

    th.testing.assert_close(distance, th.tensor([1.0 - math.sqrt(0.75 * 0.25)]))
    assert pred_logprobs.grad is not None
    assert th.isfinite(pred_logprobs.grad).all()
    assert pred_logprobs.grad[0] < 0.0


def test_outside_window_fraction_ignores_hungarian_assignment_competition() -> None:
    pred_intensities = th.tensor([1.0, 2.0, 1.0], requires_grad=True)
    fraction = sparse_outside_window_intensity_fraction(
        th.tensor([100.0]),
        th.zeros(1, dtype=th.long),
        th.tensor([99.999, 100.001, 150.0]),
        pred_intensities,
        th.zeros(3, dtype=th.long),
        batch_size=1,
        tolerance=0.01,
        relative=False,
    )
    fraction.sum().backward()

    th.testing.assert_close(fraction, th.tensor([0.25]))
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()
    assert pred_intensities.grad[2] > 0.0
    assert pred_intensities.grad[:2].lt(0.0).all()


def test_conditional_in_support_cross_entropy_removes_coverage_weight() -> None:
    coverage_weighted_ce = th.tensor([1.78, 1.0, 0.0])
    true_oos_logprobs = th.tensor([0.11, 0.5, 1.0]).log()

    actual = conditional_in_support_cross_entropy(
        coverage_weighted_ce,
        true_oos_logprobs,
    )

    th.testing.assert_close(actual, th.tensor([2.0, 2.0, 0.0]))


def test_conditional_in_support_cross_entropy_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must have matching shapes"):
        conditional_in_support_cross_entropy(th.zeros(2), th.zeros(1))


def test_vectorized_hungarian_jss_matches_scalar_on_ambiguous_pairs() -> None:
    true_mzs = th.tensor([100.0, 100.0005, 100.001])
    true_logprobs = th.tensor([0.8, 0.15, 0.05]).log()
    pred_mzs = th.tensor([100.0002, 100.0007, 100.0012])
    pred_logprobs = th.tensor([0.8, 0.15, 0.05]).log()
    batch_idxs = th.zeros(3, dtype=th.long)

    scalar = sparse_jensen_shannon_divergence_hungarian(
        true_mzs,
        true_logprobs,
        batch_idxs,
        pred_mzs,
        pred_logprobs,
        batch_idxs,
    )
    vectorized = sparse_jensen_shannon_divergence_hungarian_vec(
        true_mzs,
        true_logprobs,
        batch_idxs,
        pred_mzs,
        pred_logprobs,
        batch_idxs,
    )

    th.testing.assert_close(vectorized, scalar, atol=1e-6, rtol=1e-6)


def test_vectorized_hungarian_jss_backward_uses_selected_original_values() -> None:
    true_mzs = th.tensor([100.0])
    true_logprobs = th.tensor([0.0])
    pred_mzs = th.tensor([100.0, 100.001])
    pred_logprobs = th.tensor([0.01, 0.99]).log().requires_grad_()

    loss = sparse_jensen_shannon_divergence_hungarian_vec(
        true_mzs,
        true_logprobs,
        th.zeros(1, dtype=th.long),
        pred_mzs,
        pred_logprobs,
        th.zeros(2, dtype=th.long),
    ).sum()
    loss.backward()

    assert pred_logprobs.grad is not None
    assert th.isfinite(pred_logprobs.grad).all()
    assert pred_logprobs.grad.abs().sum() > 0.0


def test_vectorized_hungarian_jss_matches_scalar_for_ragged_chunks() -> None:
    true_mzs = th.tensor([100.0, 200.0, 250.0])
    true_logprobs = th.tensor([0.7, 0.2, 0.8]).log()
    true_batch_idxs = th.tensor([0, 0, 1], dtype=th.long)
    pred_mzs = th.tensor([100.0005, 150.0, 250.0005, 300.0])
    pred_logprobs = th.tensor([0.6, 0.4, 0.9, 0.1]).log().requires_grad_()
    pred_batch_idxs = th.tensor([0, 0, 1, 1], dtype=th.long)

    scalar = sparse_jensen_shannon_divergence_hungarian(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
    )
    vectorized = sparse_jensen_shannon_divergence_hungarian_vec(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        loss_batch_size=1,
    )

    th.testing.assert_close(vectorized, scalar, atol=1e-6, rtol=1e-6)
    vectorized.sum().backward()
    assert pred_logprobs.grad is not None
    assert th.isfinite(pred_logprobs.grad).all()


@pytest.mark.parametrize("objective", ["cosine_distance", "jss_distance", "kl_distance"])
def test_peak_list_distances_normalize_intensity_scale(objective: str) -> None:
    common = {
        "true_mzs": th.tensor([100.0, 150.0]),
        "true_intensities": th.tensor([1.0, 1.0]),
        "true_batch_idxs": th.tensor([0, 0]),
        "pred_mzs": th.tensor([100.0, 150.0]),
        "pred_batch_idxs": th.tensor([0, 0]),
        "objective": objective,
        "matching": "hungarian",
        "tolerance": 0.01,
        "relative": False,
    }

    distance, _ = sparse_peak_list_distance(pred_intensities=th.tensor([1.0, 3.0]), **common)
    scaled_distance, _ = sparse_peak_list_distance(
        pred_intensities=th.tensor([100.0, 300.0]), **common
    )

    assert th.allclose(distance, scaled_distance)


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

    def test_pairwise_cossim_identical_spectra(self):
        """Regression: identical spectra in different batch slots must have cosine sim = 1.

        Before the batch-offset fix, batched_bin_func returned batch-offset bin indices.
        Each batch element occupied exclusively different columns in the V matrix, making
        all off-diagonal dot products zero. This test catches that regression: two
        identical single-peak spectra must produce off-diagonal similarity = 1, not 0.
        """
        # Two batch elements, each with one peak at m/z=100, equal intensity.
        mzs = th.tensor([100.0, 100.0])
        logprobs = th.tensor([0.0, 0.0])  # log(1) — single peak per spectrum
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        assert th.allclose(S[0, 0], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[1, 1], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[0, 1], th.tensor(1.0), atol=1e-6), (
            "Identical spectra must have cosine similarity 1, not 0 (batch-offset bug)"
        )
        assert th.allclose(S[1, 0], th.tensor(1.0), atol=1e-6), (
            "Identical spectra must have cosine similarity 1, not 0 (batch-offset bug)"
        )

    def test_pairwise_cossim_known_partial_overlap(self):
        """Spectra sharing half their peaks should have cosine similarity = 0.5.

        Spectrum 0: equal-intensity peaks at m/z=50 and m/z=100.
        Spectrum 1: equal-intensity peaks at m/z=50 and m/z=150.
        After L2 normalization both vectors are [1/√2, 1/√2, 0] and [1/√2, 0, 1/√2]
        in the three-bin space {50, 100, 150}. Their dot product = 1/2.
        """
        import math

        log_half = math.log(0.5)
        mzs = th.tensor([50.0, 100.0, 50.0, 150.0])
        logprobs = th.tensor([log_half, log_half, log_half, log_half])
        batch_idxs = th.tensor([0, 0, 1, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        S = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        assert th.allclose(S[0, 0], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[1, 1], th.tensor(1.0), atol=1e-6), "Self-similarity should be 1"
        assert th.allclose(S[0, 1], th.tensor(0.5), atol=1e-5), (
            f"Expected cosine similarity 0.5 for 50% peak overlap, got {S[0, 1].item():.6f}"
        )
        assert th.allclose(S[1, 0], th.tensor(0.5), atol=1e-5), "Matrix must be symmetric"

    def test_pairwise_cossim_mse_nonzero_when_spectra_differ(self):
        """MSE between two different pairwise sim matrices must be nonzero.

        This is the end-to-end regression for the pairwise cosine loss in fragnnet_pl.py:
        if pred_sim ≠ true_sim, F.mse_loss(pred_sim, true_sim) must be > 0.
        Before the fix, both matrices were always the identity, giving MSE ≈ 0 regardless
        of predictions.
        """
        import torch.nn.functional as F

        # true: two identical spectra at m/z=100 → sim matrix all-ones
        mzs_true = th.tensor([100.0, 100.0])
        logprobs_true = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        mz_max = 200.0
        mz_bin_res = 1.0

        # pred: two orthogonal spectra → off-diagonal sim = 0
        mzs_pred = th.tensor([50.0, 150.0])
        logprobs_pred = th.tensor([0.0, 0.0])

        true_sim = get_pairwise_cossim(
            mzs_true, logprobs_true, batch_idxs, batch_size, mz_max, mz_bin_res
        )
        pred_sim = get_pairwise_cossim(
            mzs_pred, logprobs_pred, batch_idxs, batch_size, mz_max, mz_bin_res
        )

        mse = F.mse_loss(pred_sim, true_sim)
        assert mse.item() > 0.1, (
            f"MSE between different pairwise sim matrices must be > 0.1, got {mse.item():.2e}. "
            "If this is ~1e-9, the batch-offset bug is present."
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
    """Tests for the vectorized JSS loss with Hungarian matching."""

    def test_sparse_jss_hun_vec_matches_sequential(self):
        """Vectorized training JSS-Hungarian should match the sequential loss."""
        true_mzs = th.tensor([100.0, 200.0, 100.0, 300.0])
        pred_mzs = th.tensor([100.0, 250.0, 100.0, 305.0])
        true_logprobs = th.log(th.tensor([0.6, 0.4, 0.7, 0.3]))
        pred_logprobs = th.log(th.tensor([0.5, 0.5, 0.8, 0.2]))
        batch_idxs = th.tensor([0, 0, 1, 1], dtype=th.long)

        seq = sparse_jensen_shannon_divergence_hungarian(
            true_mzs,
            true_logprobs,
            batch_idxs,
            pred_mzs,
            pred_logprobs,
            batch_idxs,
            tolerance=0.01,
            relative=False,
        )
        vec = sparse_jensen_shannon_divergence_hungarian_vec(
            true_mzs,
            true_logprobs,
            batch_idxs,
            pred_mzs,
            pred_logprobs,
            batch_idxs,
            tolerance=0.01,
            relative=False,
            loss_batch_size=2,
        )

        assert th.allclose(vec, seq, atol=1e-6), f"vec={vec.tolist()} seq={seq.tolist()}"

    def test_sparse_jss_hun_vec_per_sample_tolerance(self):
        """Vectorized JSS-Hungarian should honor per-sample tolerance overrides."""
        true_mzs = th.tensor([500.000, 500.000])
        pred_mzs = th.tensor([500.007, 500.007])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)

        vec = sparse_jensen_shannon_divergence_hungarian_vec(
            true_mzs,
            logprobs,
            batch_idxs,
            pred_mzs,
            logprobs,
            batch_idxs,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
            tol_per_sample=th.tensor([1e-5, 2e-5]),
            min_mz_per_sample=th.tensor([200.0, 750.0]),
            loss_batch_size=2,
        )

        assert vec[0].item() > vec[1].item()
        assert vec[1].item() == pytest.approx(0.0, abs=1e-5)


class TestSparseJensenShannonDivergence:
    """Tests for sparse_jensen_shannon_divergence (wraps jss_helper from spec_utils)."""

    def test_identical_single_peak(self):
        """Identical single-peak spectra should have zero divergence."""
        mzs = th.tensor([100.0])
        logprobs = th.tensor([0.0])
        batch_idxs = th.tensor([0], dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs,
            logprobs,
            batch_idxs,
            mzs,
            logprobs,
            batch_idxs,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        assert th.allclose(jsd, th.zeros(1), atol=1e-6), (
            "Identical spectra should have zero divergence"
        )

    def test_orthogonal_spectra(self):
        """Completely non-overlapping spectra should have normalized divergence = 1."""
        mzs1 = th.tensor([50.0])
        logprobs1 = th.tensor([0.0])
        batch_idxs1 = th.tensor([0], dtype=th.long)

        mzs2 = th.tensor([150.0])
        logprobs2 = th.tensor([0.0])
        batch_idxs2 = th.tensor([0], dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs1,
            logprobs1,
            batch_idxs1,
            mzs2,
            logprobs2,
            batch_idxs2,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        assert th.allclose(jsd, th.ones(1), atol=1e-6), (
            "Orthogonal spectra should have divergence = 1"
        )

    def test_known_partial_overlap(self):
        """Spectra sharing one of two equal-intensity peaks should have divergence = 0.5.

        P = [0.5, 0.5] at mz=50,100; Q = [0.5, 0.5] at mz=50,150.
        M = [0.5, 0.25, 0.25]; KL(P||M) = KL(Q||M) = 0.5*log2.
        JSD = 0.5*log2; normalized by log2 → 0.5.
        """
        mzs_true = th.tensor([50.0, 100.0])
        logprobs_true = th.tensor([0.0, 0.0])
        batch_idxs_true = th.tensor([0, 0], dtype=th.long)

        mzs_pred = th.tensor([50.0, 150.0])
        logprobs_pred = th.tensor([0.0, 0.0])
        batch_idxs_pred = th.tensor([0, 0], dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs_true,
            logprobs_true,
            batch_idxs_true,
            mzs_pred,
            logprobs_pred,
            batch_idxs_pred,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        assert th.allclose(jsd, th.tensor([0.5]), atol=1e-5), (
            f"Expected divergence 0.5, got {jsd.item():.6f}"
        )

    def test_non_negative_many_overlapping_bins(self):
        """Regression: JSS must stay non-negative with many overlapping bins.

        Before the renormalization fix in jss_helper, floating-point accumulation
        in scatter_reduce could cause union_bin_ints to sum slightly below 1,
        inflating the KL terms beyond log(2) and yielding a negative JSS.
        """
        N = 1000
        mzs = th.arange(1.0, N + 1.0)
        logprobs = th.full((N,), float(-math.log(N)))
        batch_idxs = th.zeros(N, dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs,
            logprobs,
            batch_idxs,
            mzs,
            logprobs,
            batch_idxs,
            mz_max=float(N + 1),
            mz_bin_res=1.0,
        )
        assert jsd.item() >= -1e-6, f"JSD must be non-negative, got {jsd.item()}"
        assert jsd.item() <= 1e-5, f"Identical spectra should have JSD ≈ 0, got {jsd.item()}"

    def test_symmetry(self):
        """Divergence should be symmetric: JSD(P, Q) == JSD(Q, P)."""
        mzs1 = th.tensor([50.0, 75.0, 100.0])
        logprobs1 = th.tensor([0.0, 1.0, 0.5])
        batch_idxs1 = th.tensor([0, 0, 0], dtype=th.long)

        mzs2 = th.tensor([100.0, 150.0])
        logprobs2 = th.tensor([0.5, 1.0])
        batch_idxs2 = th.tensor([0, 0], dtype=th.long)

        jsd_pq = sparse_jensen_shannon_divergence(
            mzs1,
            logprobs1,
            batch_idxs1,
            mzs2,
            logprobs2,
            batch_idxs2,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        jsd_qp = sparse_jensen_shannon_divergence(
            mzs2,
            logprobs2,
            batch_idxs2,
            mzs1,
            logprobs1,
            batch_idxs1,
            mz_max=200.0,
            mz_bin_res=1.0,
        )
        assert th.allclose(jsd_pq, jsd_qp, atol=1e-6), (
            f"JSD should be symmetric, got {jsd_pq.item():.6f} vs {jsd_qp.item():.6f}"
        )

    def test_batch_multiple_pairs(self):
        """Test JSD computation for a batch with multiple spectrum pairs.

        Pair 0: identical single-peak spectra → JSD = 0.
        Pair 1: orthogonal single-peak spectra → JSD = 1.
        """
        mzs_true = th.tensor([50.0, 100.0])
        logprobs_true = th.tensor([0.0, 0.0])
        batch_idxs_true = th.tensor([0, 1], dtype=th.long)

        mzs_pred = th.tensor([50.0, 200.0])
        logprobs_pred = th.tensor([0.0, 0.0])
        batch_idxs_pred = th.tensor([0, 1], dtype=th.long)

        jsd = sparse_jensen_shannon_divergence(
            mzs_true,
            logprobs_true,
            batch_idxs_true,
            mzs_pred,
            logprobs_pred,
            batch_idxs_pred,
            mz_max=300.0,
            mz_bin_res=1.0,
        )
        assert jsd.shape == (2,), f"Expected shape (2,), got {jsd.shape}"
        assert th.allclose(jsd[0], th.tensor(0.0), atol=1e-6), (
            f"Pair 0 (identical): expected JSD=0, got {jsd[0].item()}"
        )
        assert th.allclose(jsd[1], th.tensor(1.0), atol=1e-6), (
            f"Pair 1 (orthogonal): expected JSD=1, got {jsd[1].item()}"
        )


class TestPairwiseBinningGeometry:
    """Tests for configurable m/z grid geometry in pairwise losses."""

    @staticmethod
    def _simple_two_spectra() -> tuple[th.Tensor, th.Tensor, th.Tensor, int]:
        """Create two identical single-peak spectra in different batch slots."""
        mzs = th.tensor([100.0, 100.0])
        logprobs = th.tensor([0.0, 0.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)
        batch_size = 2
        return mzs, logprobs, batch_idxs, batch_size

    def test_pairwise_losses_support_non_default_resolution(self):
        """Pairwise helpers should work with a non-default valid bin geometry."""
        mzs, logprobs, batch_idxs, batch_size = self._simple_two_spectra()
        mz_max = 200.0
        mz_bin_res = 0.5  # 400 bins, valid integer grid

        cossim = get_pairwise_cossim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        jss = get_pairwise_jss_sim(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)
        ce = get_pairwise_cross_entropy(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

        assert th.allclose(cossim[0, 1], th.tensor(1.0), atol=1e-6)
        assert th.allclose(jss[0, 1], th.tensor(1.0), atol=1e-6)
        assert th.allclose(ce[0, 1], th.tensor(0.0), atol=1e-6)

    @pytest.mark.parametrize(
        "fn",
        [get_pairwise_cossim, get_pairwise_jss_sim, get_pairwise_cross_entropy],
    )
    def test_pairwise_losses_raise_on_non_integer_bin_geometry(self, fn):
        """Fail fast when mz_max / mz_bin_res does not define an integer bin count."""
        mzs, logprobs, batch_idxs, batch_size = self._simple_two_spectra()
        mz_max = 200.0
        mz_bin_res = 0.3  # 666.666..., invalid grid

        with pytest.raises(ValueError, match="mz_max / mz_bin_res"):
            fn(mzs, logprobs, batch_idxs, batch_size, mz_max, mz_bin_res)

    def test_batched_bin_func_validates_geometry(self):
        """Verify batched_bin_func validates bin geometry on invalid config."""
        from fragnnet.utils.spec_utils import batched_bin_func

        mzs = th.tensor([100.0, 150.0])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 1], dtype=th.long)

        # Valid geometry (400 bins) should work
        result = batched_bin_func(
            mzs, ints, batch_idxs, mz_max=250.0, mz_bin_res=0.5, agg="sum", sparse=True
        )
        assert result[0].numel() > 0  # Should return non-empty result

        # Invalid geometry (666.666... bins) should raise ValueError
        with pytest.raises(ValueError, match="mz_max / mz_bin_res"):
            batched_bin_func(
                mzs, ints, batch_idxs, mz_max=250.0, mz_bin_res=0.3, agg="sum", sparse=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _reference_outside_window_fraction(
    true_mzs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_intensities: th.Tensor,
    pred_batch_idxs: th.Tensor,
    batch_size: int,
    tolerance: float,
) -> th.Tensor:
    """Straightforward per-spectrum reference for the batched leak term."""
    fractions = []
    for batch_index in range(batch_size):
        batch_true = true_mzs[true_batch_idxs == batch_index]
        batch_pred = pred_mzs[pred_batch_idxs == batch_index]
        batch_ints = pred_intensities[pred_batch_idxs == batch_index]
        total = float(batch_ints.sum())
        if total == 0.0:
            fractions.append(0.0)
            continue
        outside = 0.0
        for mz, intensity in zip(batch_pred.tolist(), batch_ints.tolist(), strict=True):
            if not any(abs(mz - true_mz) < tolerance for true_mz in batch_true.tolist()):
                outside += intensity
        fractions.append(outside / total)
    return th.tensor(fractions)


def test_outside_window_fraction_matches_per_spectrum_reference() -> None:
    generator = th.Generator().manual_seed(17)
    batch_size = 6
    true_mzs, true_batch_idxs, pred_mzs, pred_intensities, pred_batch_idxs = [], [], [], [], []
    for batch_index in range(batch_size):
        observed = 100.0 + th.rand(4, generator=generator) * 50.0
        true_mzs.append(observed)
        true_batch_idxs.append(th.full((4,), batch_index))
        # Two predictions land on observed peaks and three are spurious.
        predicted = th.cat([observed[:2], 300.0 + th.rand(3, generator=generator) * 50.0])
        pred_mzs.append(predicted)
        pred_intensities.append(th.rand(5, generator=generator) + 0.1)
        pred_batch_idxs.append(th.full((5,), batch_index))
    true_mzs = th.cat(true_mzs)
    true_batch_idxs = th.cat(true_batch_idxs).long()
    pred_mzs = th.cat(pred_mzs)
    pred_intensities = th.cat(pred_intensities)
    pred_batch_idxs = th.cat(pred_batch_idxs).long()

    actual = sparse_outside_window_intensity_fraction(
        true_mzs,
        true_batch_idxs,
        pred_mzs,
        pred_intensities,
        pred_batch_idxs,
        batch_size=batch_size,
        tolerance=0.001,
        relative=False,
        loss_batch_size=4,
    )

    expected = _reference_outside_window_fraction(
        true_mzs,
        true_batch_idxs,
        pred_mzs,
        pred_intensities,
        pred_batch_idxs,
        batch_size,
        0.001,
    )
    assert actual.shape == (batch_size,)
    th.testing.assert_close(actual, expected)


def test_outside_window_fraction_is_invariant_to_loss_batch_size() -> None:
    generator = th.Generator().manual_seed(23)
    batch_size = 5
    true_mzs = th.cat([100.0 + th.rand(3, generator=generator) * 20.0 for _ in range(batch_size)])
    true_batch_idxs = th.arange(batch_size).repeat_interleave(3)
    pred_mzs = th.cat([100.0 + th.rand(4, generator=generator) * 20.0 for _ in range(batch_size)])
    pred_batch_idxs = th.arange(batch_size).repeat_interleave(4)
    pred_intensities = th.rand(4 * batch_size, generator=generator) + 0.1

    results = [
        sparse_outside_window_intensity_fraction(
            true_mzs,
            true_batch_idxs,
            pred_mzs,
            pred_intensities,
            pred_batch_idxs,
            batch_size=batch_size,
            tolerance=0.05,
            relative=False,
            loss_batch_size=chunk,
        )
        for chunk in (1, 2, 5, 64)
    ]

    for chunked in results[1:]:
        th.testing.assert_close(chunked, results[0])


def test_outside_window_fraction_tolerates_unsorted_batch_indices() -> None:
    true_mzs = th.tensor([200.0, 100.0])
    true_batch_idxs = th.tensor([1, 0])
    pred_mzs = th.tensor([300.0, 100.0, 200.0])
    pred_intensities = th.tensor([1.0, 3.0, 1.0])
    pred_batch_idxs = th.tensor([1, 0, 1])

    actual = sparse_outside_window_intensity_fraction(
        true_mzs,
        true_batch_idxs,
        pred_mzs,
        pred_intensities,
        pred_batch_idxs,
        batch_size=2,
        tolerance=0.01,
        relative=False,
    )

    th.testing.assert_close(actual, th.tensor([0.0, 0.5]))


def test_outside_window_fraction_keeps_one_entry_per_spectrum_when_observations_are_missing() -> (
    None
):
    """A trailing spectrum without observed peaks must not shorten the result.

    A short result broadcasts silently against a per-spectrum loss instead of
    raising, so the length is the contract being locked here.
    """
    pred_intensities = th.tensor([1.0, 2.0], requires_grad=True)

    actual = sparse_outside_window_intensity_fraction(
        th.tensor([100.0]),
        th.tensor([0]),
        th.tensor([100.0, 500.0]),
        pred_intensities,
        th.tensor([0, 1]),
        batch_size=2,
        tolerance=0.01,
        relative=False,
    )
    actual.sum().backward()

    # Spectrum 0 is fully explained; spectrum 1 has no window to fall inside.
    assert actual.shape == (2,)
    th.testing.assert_close(actual, th.tensor([0.0, 1.0]))
    assert th.isfinite(pred_intensities.grad).all()


def test_outside_window_fraction_is_zero_without_predicted_intensity() -> None:
    actual = sparse_outside_window_intensity_fraction(
        th.tensor([100.0, 200.0]),
        th.tensor([0, 1]),
        th.tensor([100.0]),
        th.tensor([1.0]),
        th.tensor([0]),
        batch_size=2,
        tolerance=0.01,
        relative=False,
    )

    assert actual.shape == (2,)
    th.testing.assert_close(actual, th.zeros(2))


def test_outside_window_fraction_rejects_batch_indices_outside_batch_size() -> None:
    with pytest.raises(ValueError, match="predicted batch indices must be below batch_size"):
        sparse_outside_window_intensity_fraction(
            th.tensor([100.0]),
            th.tensor([0]),
            th.tensor([100.0, 200.0]),
            th.tensor([1.0, 1.0]),
            th.tensor([0, 3]),
            batch_size=2,
            tolerance=0.01,
            relative=False,
        )


def test_outside_window_fraction_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        sparse_outside_window_intensity_fraction(
            th.tensor([100.0]),
            th.tensor([0]),
            th.tensor([100.0]),
            th.tensor([1.0]),
            th.tensor([0]),
            batch_size=0,
        )


def test_outside_window_fraction_applies_per_sample_tolerance() -> None:
    """The per-spectrum tolerance must follow the spectrum, not the chunk."""
    true_mzs = th.tensor([100.0, 100.0])
    true_batch_idxs = th.tensor([0, 1])
    pred_mzs = th.tensor([100.04, 100.04])
    pred_intensities = th.tensor([1.0, 1.0])
    pred_batch_idxs = th.tensor([0, 1])

    actual = sparse_outside_window_intensity_fraction(
        true_mzs,
        true_batch_idxs,
        pred_mzs,
        pred_intensities,
        pred_batch_idxs,
        batch_size=2,
        relative=False,
        tol_per_sample=th.tensor([0.1, 0.01]),
        min_mz_per_sample=th.tensor([200.0, 200.0]),
    )

    # 0.04 Da is inside the loose spectrum's window and outside the tight one's.
    th.testing.assert_close(actual, th.tensor([0.0, 1.0]))


def test_vectorized_hungarian_jss_scores_unpredicted_spectrum_as_maximum_distance() -> None:
    """A spectrum with observations but no predictions must not poison the chunk.

    The emptiness guard is per chunk, so this spectrum reaches the normalizer
    with zero predicted mass; without a per-spectrum guard it returns NaN.
    """
    true_mzs = th.tensor([100.0, 150.0, 200.0])
    true_logprobs = th.tensor([0.5, 0.5, 1.0]).log()
    true_batch_idxs = th.tensor([0, 0, 1])
    pred_mzs = th.tensor([100.0, 150.0])
    pred_logprobs = th.tensor([0.5, 0.5]).log().requires_grad_()
    pred_batch_idxs = th.tensor([0, 0])

    vectorized = sparse_jensen_shannon_divergence_hungarian_vec(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
    )
    vectorized.sum().backward()
    scalar = sparse_jensen_shannon_divergence_hungarian(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs.detach(),
        pred_batch_idxs,
    )

    assert th.isfinite(vectorized).all()
    assert vectorized[1].item() == 1.0
    th.testing.assert_close(vectorized, scalar, atol=1e-6, rtol=1e-6)
    assert th.isfinite(pred_logprobs.grad).all()


def test_vectorized_hungarian_jss_keeps_one_entry_per_unobserved_spectrum() -> None:
    """The mirror case: a trailing spectrum with predictions but no observations.

    Inferring the batch size from the observed peaks alone drops that spectrum, and a
    short result broadcasts silently against a per-spectrum loss instead of raising,
    so the length is the contract being locked here.
    """
    true_mzs = th.tensor([100.0, 150.0])
    true_logprobs = th.tensor([0.5, 0.5]).log()
    true_batch_idxs = th.tensor([0, 0])
    pred_mzs = th.tensor([100.0, 150.0, 300.0])
    pred_logprobs = th.tensor([0.5, 0.5, 1.0]).log().requires_grad_()
    pred_batch_idxs = th.tensor([0, 0, 1])

    vectorized = sparse_jensen_shannon_divergence_hungarian_vec(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
    )
    vectorized.sum().backward()
    scalar = sparse_jensen_shannon_divergence_hungarian(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs.detach(),
        pred_batch_idxs,
    )

    assert vectorized.shape == (2,)
    assert th.isfinite(vectorized).all()
    assert vectorized[1].item() == 1.0
    th.testing.assert_close(vectorized, scalar, atol=1e-6, rtol=1e-6)
    assert th.isfinite(pred_logprobs.grad).all()


def test_vectorized_hungarian_jss_rejects_two_empty_peak_lists() -> None:
    empty_mzs = th.zeros(0)
    empty_idxs = th.zeros(0, dtype=th.long)
    with pytest.raises(ValueError, match="empty peak lists"):
        sparse_jensen_shannon_divergence_hungarian_vec(
            empty_mzs,
            empty_mzs,
            empty_idxs,
            empty_mzs,
            empty_mzs,
            empty_idxs,
        )


def test_vectorized_hungarian_jss_matches_scalar_across_chunk_boundaries() -> None:
    generator = th.Generator().manual_seed(29)
    batch_size = 7
    true_mzs, true_logprobs, true_batch_idxs = [], [], []
    pred_mzs, pred_logprobs, pred_batch_idxs = [], [], []
    for batch_index in range(batch_size):
        num_true = int(th.randint(1, 6, (1,), generator=generator))
        num_pred = int(th.randint(1, 6, (1,), generator=generator))
        observed = 100.0 + th.rand(num_true, generator=generator) * 5.0
        true_mzs.append(observed)
        true_batch_idxs.append(th.full((num_true,), batch_index))
        observed_ints = th.rand(num_true, generator=generator) + 0.05
        true_logprobs.append((observed_ints / observed_ints.sum()).log())

        shared = min(num_true, num_pred)
        predicted = th.cat(
            [
                observed[:shared] + (th.rand(shared, generator=generator) - 0.5) * 2e-3,
                100.0 + th.rand(num_pred - shared, generator=generator) * 5.0,
            ]
        )
        pred_mzs.append(predicted)
        pred_batch_idxs.append(th.full((num_pred,), batch_index))
        predicted_ints = th.rand(num_pred, generator=generator) + 0.05
        pred_logprobs.append((predicted_ints / predicted_ints.sum()).log())

    args = (
        th.cat(true_mzs),
        th.cat(true_logprobs),
        th.cat(true_batch_idxs).long(),
        th.cat(pred_mzs),
        th.cat(pred_logprobs),
        th.cat(pred_batch_idxs).long(),
    )
    scalar = sparse_jensen_shannon_divergence_hungarian(*args)

    for chunk in (1, 3, 7, 64):
        vectorized = sparse_jensen_shannon_divergence_hungarian_vec(*args, loss_batch_size=chunk)
        assert th.isfinite(vectorized).all()
        th.testing.assert_close(vectorized, scalar, atol=1e-5, rtol=1e-5)


def test_sparse_hellinger_distance_scores_each_spectrum_independently() -> None:
    """Shared-bin pairing must stay inside a spectrum across a multi-spectrum batch."""
    true_mzs = th.tensor([100.0, 150.0, 100.0, 200.0])
    true_logprobs = th.tensor([0.75, 0.25, 0.5, 0.5]).log()
    true_batch_idxs = th.tensor([0, 0, 1, 1])
    # Spectrum 1 predicts only m/z 100, which spectrum 0 also occupies.
    pred_mzs = th.tensor([100.0, 200.0, 100.0])
    pred_logprobs = th.tensor([0.25, 0.75, 1.0]).log()
    pred_batch_idxs = th.tensor([0, 0, 1])

    actual = sparse_hellinger_distance(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
    )

    expected = th.tensor(
        [
            1.0 - math.sqrt(0.75 * 0.25),
            1.0 - math.sqrt(0.5 * 1.0),
        ]
    )
    th.testing.assert_close(actual, expected)


def test_outside_window_fraction_matches_reference_on_relative_ragged_batch() -> None:
    """The production path: relative ppm tolerance, ragged spectra, per-instrument tolerances.

    Padding is the risk here. Padded slots hold m/z 0 on both sides, and a relative
    divisor floor would otherwise let them match each other.
    """
    generator = th.Generator().manual_seed(31)
    batch_size = 7
    true_mzs, true_batch_idxs, pred_mzs, pred_intensities, pred_batch_idxs = [], [], [], [], []
    for batch_index in range(batch_size):
        # Spectrum 3 has no observed peaks and spectrum 5 has no predictions.
        num_true = 0 if batch_index == 3 else int(th.randint(1, 6, (1,), generator=generator))
        num_pred = 0 if batch_index == 5 else int(th.randint(1, 8, (1,), generator=generator))
        observed = 150.0 + th.rand(num_true, generator=generator) * 600.0
        true_mzs.append(observed)
        true_batch_idxs.append(th.full((num_true,), batch_index))
        shared = min(num_true, num_pred)
        predicted = th.cat(
            [
                observed[:shared] * (1 + (th.rand(shared, generator=generator) - 0.5) * 3e-5),
                150.0 + th.rand(num_pred - shared, generator=generator) * 600.0,
            ]
        )
        pred_mzs.append(predicted)
        pred_batch_idxs.append(th.full((num_pred,), batch_index))
        pred_intensities.append(th.rand(num_pred, generator=generator) + 0.01)
    true_mzs = th.cat(true_mzs)
    true_batch_idxs = th.cat(true_batch_idxs).long()
    pred_mzs = th.cat(pred_mzs)
    pred_intensities = th.cat(pred_intensities)
    pred_batch_idxs = th.cat(pred_batch_idxs).long()
    # FT and QTOF rows of a realistic inst_type_loss_tol table.
    tol_per_sample = th.tensor([1e-5, 2e-5, 1e-5, 1e-5, 2e-5, 1e-5, 1e-5])
    min_mz_per_sample = th.tensor([200.0, 750.0, 200.0, 200.0, 750.0, 200.0, 200.0])

    expected = []
    for batch_index in range(batch_size):
        batch_true = true_mzs[true_batch_idxs == batch_index]
        batch_pred_mask = pred_batch_idxs == batch_index
        batch_ints = pred_intensities[batch_pred_mask]
        if batch_ints.numel() == 0:
            expected.append(0.0)
            continue
        if batch_true.numel() == 0:
            expected.append(1.0)
            continue
        match_mask = th.as_tensor(
            calculate_match_mzs(
                batch_true,
                pred_mzs[batch_pred_mask],
                tolerance=float(tol_per_sample[batch_index]),
                relative=True,
                tolerance_min_mz=float(min_mz_per_sample[batch_index]),
            ),
            dtype=th.bool,
        )
        outside = ~match_mask.any(dim=0)
        expected.append((batch_ints[outside].sum() / batch_ints.sum()).item())
    expected = th.tensor(expected)

    for chunk in (1, 3, 7, 64):
        actual = sparse_outside_window_intensity_fraction(
            true_mzs,
            true_batch_idxs,
            pred_mzs,
            pred_intensities,
            pred_batch_idxs,
            batch_size=batch_size,
            relative=True,
            tol_per_sample=tol_per_sample,
            min_mz_per_sample=min_mz_per_sample,
            loss_batch_size=chunk,
        )
        assert actual.shape == (batch_size,)
        th.testing.assert_close(actual, expected)


def _dense_binned_hellinger(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    mz_max: float = 1500.0,
    mz_bin_res: float = 0.01,
) -> th.Tensor:
    """Dense reference: bin both spectra onto the full grid and reduce there."""
    binned_true = batched_bin_func(
        true_mzs, true_logprobs, true_batch_idxs, mz_max=mz_max, mz_bin_res=mz_bin_res, agg="lse"
    ).exp()
    binned_pred = batched_bin_func(
        pred_mzs, pred_logprobs, pred_batch_idxs, mz_max=mz_max, mz_bin_res=mz_bin_res, agg="lse"
    ).exp()
    binned_true = binned_true / binned_true.sum(-1, keepdim=True)
    binned_pred = binned_pred / binned_pred.sum(-1, keepdim=True)
    return 1.0 - (binned_true.sqrt() * binned_pred.sqrt()).sum(-1)


def _random_binned_spectra(seed: int, batch_size: int):
    generator = th.Generator().manual_seed(seed)
    grid = th.round(100.0 + th.arange(24) * 7.3, decimals=3)
    true_mzs, true_logprobs, true_batch_idxs = [], [], []
    pred_mzs, pred_logprobs, pred_batch_idxs = [], [], []
    for batch_index in range(batch_size):
        num_true = int(th.randint(1, 9, (1,), generator=generator))
        num_pred = int(th.randint(1, 12, (1,), generator=generator))
        # A small shared grid, so spectra collide in the same m/z bins on purpose.
        observed = grid[th.randperm(grid.numel(), generator=generator)[:num_true]]
        true_mzs.append(observed)
        true_batch_idxs.append(th.full((num_true,), batch_index))
        observed_ints = th.rand(num_true, generator=generator) + 0.02
        true_logprobs.append((observed_ints / observed_ints.sum()).log())
        # Overlap on the first few bins, then diverge. Every third spectrum shares nothing.
        shared = 0 if batch_index % 3 == 2 else min(num_true, num_pred)
        spurious = grid[th.randperm(grid.numel(), generator=generator)[: num_pred - shared]] + 400.0
        predicted = th.cat([observed[:shared], spurious])
        pred_mzs.append(predicted)
        pred_batch_idxs.append(th.full((num_pred,), batch_index))
        predicted_ints = th.rand(num_pred, generator=generator) + 0.02
        pred_logprobs.append((predicted_ints / predicted_ints.sum()).log())
    return (
        th.cat(true_mzs),
        th.cat(true_logprobs),
        th.cat(true_batch_idxs).long(),
        th.cat(pred_mzs),
        th.cat(pred_logprobs),
        th.cat(pred_batch_idxs).long(),
    )


@pytest.mark.parametrize("seed", range(4))
def test_sparse_hellinger_distance_matches_dense_reference_over_a_batch(seed: int) -> None:
    """The sparse shared-bin form must equal the dense whole-grid form per spectrum."""
    args = _random_binned_spectra(seed, batch_size=9)

    actual = sparse_hellinger_distance(*args)

    assert actual.shape == (9,)
    th.testing.assert_close(actual, _dense_binned_hellinger(*args), atol=1e-6, rtol=1e-5)


def test_sparse_hellinger_distance_is_independent_of_batch_composition() -> None:
    """Shared bins are matched on globally offset ids, so neighbours must not leak in.

    Two spectra here occupy the same m/z bins. If the pairing used bin position
    within a spectrum instead of the global id, their coefficients would mix.
    """
    args = _random_binned_spectra(7, batch_size=6)
    true_mzs, true_logprobs, true_batch_idxs, pred_mzs, pred_logprobs, pred_batch_idxs = args

    batched = sparse_hellinger_distance(*args)

    alone = []
    for batch_index in range(6):
        true_mask = true_batch_idxs == batch_index
        pred_mask = pred_batch_idxs == batch_index
        alone.append(
            sparse_hellinger_distance(
                true_mzs[true_mask],
                true_logprobs[true_mask],
                th.zeros(int(true_mask.sum()), dtype=th.long),
                pred_mzs[pred_mask],
                pred_logprobs[pred_mask],
                th.zeros(int(pred_mask.sum()), dtype=th.long),
            )
        )

    th.testing.assert_close(batched, th.cat(alone), atol=1e-6, rtol=1e-5)


def test_sparse_hellinger_distance_keeps_gradients_per_spectrum() -> None:
    """Only the spectrum that owns a predicted peak may receive its gradient."""
    true_mzs = th.tensor([100.0, 200.0])
    true_logprobs = th.tensor([0.0, 0.0])
    true_batch_idxs = th.tensor([0, 1])
    # Both spectra predict m/z 100; only spectrum 0 observes it.
    pred_mzs = th.tensor([100.0, 400.0, 100.0, 200.0])
    pred_logprobs = th.tensor([0.5, 0.5, 0.5, 0.5]).log().requires_grad_()
    pred_batch_idxs = th.tensor([0, 0, 1, 1])

    distances = sparse_hellinger_distance(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
    )
    # Differentiate spectrum 1 alone: spectrum 0's peaks must stay untouched.
    distances[1].backward()

    assert distances.shape == (2,)
    assert th.isfinite(pred_logprobs.grad).all()
    assert pred_logprobs.grad[:2].eq(0.0).all()
    assert pred_logprobs.grad[2:].abs().sum() > 0.0


def test_sparse_hellinger_distance_keeps_one_entry_per_unobserved_spectrum() -> None:
    """A trailing spectrum with predictions but no observed peaks must still score.

    Sizing the batch from the observed peaks alone leaves the out-of-range spectrum
    index to trip a scatter assertion, so this locks both the length and the value:
    sharing no bin means a zero coefficient and maximum distance.
    """
    true_mzs = th.tensor([100.0])
    true_logprobs = th.tensor([0.0])
    true_batch_idxs = th.tensor([0])
    pred_mzs = th.tensor([100.0, 300.0])
    pred_logprobs = th.tensor([0.0, 0.0]).requires_grad_()
    pred_batch_idxs = th.tensor([0, 1])

    actual = sparse_hellinger_distance(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
    )
    actual.sum().backward()

    assert actual.shape == (2,)
    th.testing.assert_close(actual, th.tensor([0.0, 1.0]), atol=1e-6, rtol=1e-6)
    assert th.isfinite(pred_logprobs.grad).all()


def test_sparse_hellinger_distance_separates_spectra_that_share_bins() -> None:
    """Three spectra reusing the same m/z values must not mix coefficients.

    Every spectrum here occupies m/z 100, so pairing shared bins by position
    inside a spectrum, or normalizing across the batch, would leak between them.
    The expected values are closed form.
    """
    true_mzs = th.tensor([100.0, 200.0, 100.0, 200.0, 300.0])
    true_logprobs = th.tensor([0.9, 0.1, 0.5, 0.5, 1.0]).log()
    true_batch_idxs = th.tensor([0, 0, 1, 1, 2])
    pred_mzs = th.tensor([100.0, 200.0, 100.0, 300.0, 100.0])
    pred_logprobs = th.tensor([0.1, 0.9, 0.5, 0.5, 1.0]).log()
    pred_batch_idxs = th.tensor([0, 0, 1, 1, 2])

    actual = sparse_hellinger_distance(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
    )

    expected = th.tensor(
        [
            # Both bins shared, intensities swapped.
            1.0 - 2.0 * math.sqrt(0.9 * 0.1),
            # Only m/z 100 is shared.
            1.0 - math.sqrt(0.5 * 0.5),
            # Nothing shared, even though m/z 100 is occupied by the other two.
            1.0,
        ]
    )
    th.testing.assert_close(actual, expected)
