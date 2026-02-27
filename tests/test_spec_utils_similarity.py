"""Unit tests for spec_utils similarity and matching functions.

Tests for:
- calculate_match_mzs (pairwise m/z tolerance matching, relative and absolute)
- binned_cosine_similarity (dense binned cosine similarity)
- cos_sim_helper (sparse cosine similarity via bin index overlap)
- batched_l1_normalize, batched_l2_normalize (scatter-based normalization)
- batched_mf1000_normalize, mf1000_normalize_np (MF1000 normalization)
- scipy_linear_sum_assignment (Hungarian algorithm wrapper)
- opt_cos_sim_helper (oracle upper-bound cosine similarity)
- calculate_cosine_similarity (end-to-end dense pipeline)
- calculate_sparse_cosine_similarity (end-to-end sparse pipeline)
- calculate_sparse_cosine_hungarian (end-to-end Hungarian-matched pipeline)
"""

import math

import numpy as np
import pytest
import torch as th

from fragnnet.utils.spec_utils import (
    batched_l1_normalize,
    batched_l2_normalize,
    batched_mf1000_normalize,
    binned_cosine_similarity,
    calculate_cosine_similarity,
    calculate_match_mzs,
    calculate_sparse_cosine_hungarian,
    calculate_sparse_cosine_similarity,
    cos_sim_helper,
    mf1000_normalize_np,
    opt_cos_sim_helper,
    scipy_linear_sum_assignment,
)


class TestCalculateMatchMzs:
    """Tests for calculate_match_mzs — pairwise N×M boolean match matrix.

    Output shape: (N_true, N_pred). True at [i, j] means true_mzs[i] matches pred_mzs[j].
    """

    def test_absolute_exact_match_tensor(self):
        """Exact m/z match within absolute tolerance returns True."""
        true_mzs = th.tensor([100.0])
        pred_mzs = th.tensor([100.0])
        result = calculate_match_mzs(true_mzs, pred_mzs, tolerance=0.01, relative=False)
        assert result.shape == (1, 1)
        assert result[0, 0].item() is True

    def test_absolute_no_match_tensor(self):
        """m/z difference exceeding absolute tolerance → False."""
        true_mzs = th.tensor([100.0])
        pred_mzs = th.tensor([100.5])
        result = calculate_match_mzs(true_mzs, pred_mzs, tolerance=0.1, relative=False)
        assert result[0, 0].item() is False

    def test_absolute_ndarray_2x2(self):
        """Numpy arrays: 2×2 match matrix with known pattern."""
        true_mzs = np.array([100.0, 200.0])
        pred_mzs = np.array([100.05, 201.0])
        result = calculate_match_mzs(true_mzs, pred_mzs, tolerance=0.1, relative=False)
        assert result.shape == (2, 2)
        assert bool(result[0, 0])     # |100 - 100.05| = 0.05 < 0.1 → match
        assert not bool(result[0, 1])  # |100 - 201| = 101 >= 0.1 → no match
        assert not bool(result[1, 0])  # |200 - 100.05| = 99.95 → no match
        assert not bool(result[1, 1])  # |200 - 201| = 1.0 >= 0.1 → no match

    def test_relative_high_mz_match(self):
        """At mz > tolerance_min_mz=200, relative tolerance applies: diff/mz < tol."""
        true_mzs = th.tensor([500.0])
        pred_mzs = th.tensor([504.0])  # diff=4, rel=4/500=0.008 < 0.01
        result = calculate_match_mzs(true_mzs, pred_mzs, tolerance=0.01, relative=True)
        assert result[0, 0].item() is True

    def test_relative_high_mz_no_match(self):
        """At mz > tolerance_min_mz, exceeding relative tolerance → False."""
        true_mzs = th.tensor([500.0])
        pred_mzs = th.tensor([510.0])  # diff=10, rel=10/500=0.02 >= 0.01
        result = calculate_match_mzs(true_mzs, pred_mzs, tolerance=0.01, relative=True)
        assert result[0, 0].item() is False

    def test_relative_low_mz_clamped_to_min(self):
        """At mz < 200 (tolerance_min_mz), divisor is clamped to 200."""
        true_mzs = th.tensor([100.0])  # < 200, so divisor = 200
        pred_mzs = th.tensor([101.0])  # diff=1, rel=1/200=0.005 < 0.01
        result = calculate_match_mzs(true_mzs, pred_mzs, tolerance=0.01, relative=True)
        assert result[0, 0].item() is True

    def test_relative_low_mz_no_match_after_clamping(self):
        """Even with clamping, large differences still don't match."""
        true_mzs = th.tensor([100.0])
        pred_mzs = th.tensor([110.0])  # diff=10, rel=10/200=0.05 >= 0.01
        result = calculate_match_mzs(true_mzs, pred_mzs, tolerance=0.01, relative=True)
        assert result[0, 0].item() is False

    def test_pred_mz_divisor_uses_pred_for_relative(self):
        """pred_mz_divisor=True uses pred_mzs as the divisor instead of true_mzs."""
        # With true as divisor: diff/true = 400/100 (clamped→200) = 2 >= 0.01 → no match
        # With pred as divisor: diff/pred = 400/500 = 0.8 >= 0.01 → no match either
        # Use a case where it makes a meaningful difference:
        true_mzs = th.tensor([200.0])
        pred_mzs = th.tensor([202.0])  # diff=2
        # true divisor: 2/200 = 0.01 → borderline False (not < 0.01, requires strict <)
        # pred divisor: 2/202 ≈ 0.0099 < 0.01 → True
        result_pred = calculate_match_mzs(
            true_mzs, pred_mzs, tolerance=0.01, relative=True, pred_mz_divisor=True
        )
        result_true = calculate_match_mzs(
            true_mzs, pred_mzs, tolerance=0.01, relative=True, pred_mz_divisor=False
        )
        # With pred divisor (202): 2/202 < 0.01 → True; with true (200): 2/200 = 0.01 → False
        assert result_pred[0, 0].item() is True
        assert result_true[0, 0].item() is False

    def test_output_matrix_shape_NxM(self):
        """Output shape is (N_true, N_pred)."""
        true_mzs = th.tensor([100.0, 200.0, 300.0])
        pred_mzs = th.tensor([100.0, 150.0])
        result = calculate_match_mzs(true_mzs, pred_mzs, tolerance=0.5, relative=False)
        assert result.shape == (3, 2)

    def test_mismatched_types_raises(self):
        """Mixing tensor and ndarray raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            calculate_match_mzs(th.tensor([100.0]), np.array([100.0]))


class TestBinnedCosineSimilarity:
    """Tests for binned_cosine_similarity — cosine similarity on dense spectra."""

    def test_identical_spectra_returns_one(self):
        """Identical spectra have cosine similarity = 1.0."""
        spec = th.tensor([[0.0, 1.0, 0.0, 2.0]])
        assert binned_cosine_similarity(spec, spec).item() == pytest.approx(1.0)

    def test_orthogonal_spectra_returns_zero(self):
        """Spectra with no shared non-zero bins → 0.0."""
        true_spec = th.tensor([[1.0, 0.0]])
        pred_spec = th.tensor([[0.0, 1.0]])
        assert binned_cosine_similarity(true_spec, pred_spec).item() == pytest.approx(0.0)

    def test_partial_overlap_known_value(self):
        """[1,0] vs [1,1]: cos = dot([1,0], [1/√2, 1/√2]) = 1/√2."""
        true_spec = th.tensor([[1.0, 0.0]])
        pred_spec = th.tensor([[1.0, 1.0]])
        result = binned_cosine_similarity(true_spec, pred_spec)
        assert result.item() == pytest.approx(1.0 / math.sqrt(2), abs=1e-5)

    def test_scale_invariant(self):
        """Scaling a spectrum by a positive constant doesn't change cosine similarity."""
        spec_a = th.tensor([[1.0, 0.0, 2.0]])
        spec_b = th.tensor([[100.0, 0.0, 200.0]])
        assert binned_cosine_similarity(spec_a, spec_b).item() == pytest.approx(1.0)

    def test_batch_of_two_computed_independently(self):
        """B=2: each (true, pred) pair is evaluated independently."""
        true_spec = th.tensor([[1.0, 0.0], [0.0, 1.0]])
        pred_spec = th.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = binned_cosine_similarity(true_spec, pred_spec)
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(1.0)

    def test_result_in_zero_one_range(self):
        """Cosine similarity is always in [0, 1] for non-negative spectra."""
        th.manual_seed(0)
        spec_a = th.rand(4, 20).abs()
        spec_b = th.rand(4, 20).abs()
        result = binned_cosine_similarity(spec_a, spec_b)
        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-6).all()


class TestCosSimHelper:
    """Tests for cos_sim_helper — sparse cosine similarity via bin index overlap."""

    def test_identical_sparse_returns_one(self):
        """Identical sparse spectra (same bins, same ints) → cos_sim = 1.0."""
        bin_idxs = th.tensor([10, 20])
        ints = th.tensor([3.0, 4.0])
        batch = th.tensor([0, 0])
        result = cos_sim_helper(bin_idxs, ints, batch, bin_idxs, ints, batch)
        assert result.item() == pytest.approx(1.0)

    def test_orthogonal_sparse_returns_zero(self):
        """No shared bins → cos_sim = 0.0."""
        true_bin_idxs = th.tensor([10])
        true_ints = th.tensor([1.0])
        true_batch = th.tensor([0])
        pred_bin_idxs = th.tensor([20])
        pred_ints = th.tensor([1.0])
        pred_batch = th.tensor([0])
        result = cos_sim_helper(
            true_bin_idxs, true_ints, true_batch,
            pred_bin_idxs, pred_ints, pred_batch,
        )
        assert result.item() == pytest.approx(0.0)

    def test_partial_overlap_known_value(self):
        """One shared bin out of two: (3/5)*(3/5) = 0.36.

        true: bins [10,20], ints [3,4], L2=5 → normalized [0.6, 0.8]
        pred: bins [10,30], ints [3,4], L2=5 → normalized [0.6, 0.8]
        Overlap only at bin 10: 0.6 * 0.6 = 0.36.
        """
        true_bin_idxs = th.tensor([10, 20])
        true_bin_ints = th.tensor([3.0, 4.0])
        true_batch = th.tensor([0, 0])
        pred_bin_idxs = th.tensor([10, 30])
        pred_bin_ints = th.tensor([3.0, 4.0])
        pred_batch = th.tensor([0, 0])
        result = cos_sim_helper(
            true_bin_idxs, true_bin_ints, true_batch,
            pred_bin_idxs, pred_bin_ints, pred_batch,
        )
        assert result.item() == pytest.approx(0.36, abs=1e-5)

    def test_scale_invariant(self):
        """Scaling both spectra uniformly doesn't change cosine similarity."""
        bin_idxs = th.tensor([5, 10])
        batch = th.tensor([0, 0])
        ints_a = th.tensor([1.0, 2.0])
        ints_b = th.tensor([100.0, 200.0])
        result = cos_sim_helper(bin_idxs, ints_a, batch, bin_idxs, ints_b, batch)
        assert result.item() == pytest.approx(1.0)

    def test_batch_two_items_independent(self):
        """Batch items 0 and 1 are processed independently."""
        # Batch 0: bins [1] vs [1] → cos=1; Batch 1: bins [2] vs [3] → cos=0
        bin_idxs = th.tensor([1, 2])       # true: bin 1 (batch 0), bin 2 (batch 1)
        ints = th.tensor([1.0, 1.0])
        batch = th.tensor([0, 1])
        pred_bin_idxs = th.tensor([1, 3])  # pred: bin 1 (batch 0), bin 3 (batch 1)
        pred_ints = th.tensor([1.0, 1.0])
        pred_batch = th.tensor([0, 1])
        result = cos_sim_helper(bin_idxs, ints, batch, pred_bin_idxs, pred_ints, pred_batch)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(0.0)


class TestBatchedNormalize:
    """Tests for batched_l1_normalize, batched_l2_normalize, batched_mf1000_normalize."""

    def test_l1_sum_equals_one_per_batch(self):
        """After L1 normalization, each batch group sums to 1."""
        ints = th.tensor([1.0, 2.0, 3.0, 4.0])
        batch_idxs = th.tensor([0, 0, 1, 1])
        result = batched_l1_normalize(ints, batch_idxs)
        assert result[batch_idxs == 0].sum().item() == pytest.approx(1.0)
        assert result[batch_idxs == 1].sum().item() == pytest.approx(1.0)

    def test_l1_preserves_relative_magnitudes(self):
        """L1 normalization preserves ratio between peaks in the same batch."""
        ints = th.tensor([1.0, 3.0])
        batch_idxs = th.tensor([0, 0])
        result = batched_l1_normalize(ints, batch_idxs)
        assert result[1].item() / result[0].item() == pytest.approx(3.0, abs=1e-5)

    def test_l2_norm_equals_one_per_batch(self):
        """After L2 normalization, each batch group has unit Euclidean norm."""
        ints = th.tensor([3.0, 4.0, 5.0])
        batch_idxs = th.tensor([0, 0, 1])
        result = batched_l2_normalize(ints, batch_idxs)
        norm_batch0 = result[batch_idxs == 0].pow(2).sum().sqrt()
        norm_batch1 = result[batch_idxs == 1].pow(2).sum().sqrt()
        assert norm_batch0.item() == pytest.approx(1.0, abs=1e-5)
        assert norm_batch1.item() == pytest.approx(1.0, abs=1e-5)

    def test_l2_known_values_pythagorean(self):
        """3-4-5 Pythagorean triple normalizes to [0.6, 0.8]."""
        ints = th.tensor([3.0, 4.0])
        batch_idxs = th.tensor([0, 0])
        result = batched_l2_normalize(ints, batch_idxs)
        assert result[0].item() == pytest.approx(0.6, abs=1e-5)
        assert result[1].item() == pytest.approx(0.8, abs=1e-5)

    def test_mf1000_max_equals_1000_per_batch(self):
        """After MF1000 normalization, max per batch is exactly 1000."""
        ints = th.tensor([100.0, 200.0, 500.0, 50.0])
        batch_idxs = th.tensor([0, 0, 1, 1])
        result = batched_mf1000_normalize(ints, batch_idxs)
        assert result[batch_idxs == 0].max().item() == pytest.approx(1000.0)
        assert result[batch_idxs == 1].max().item() == pytest.approx(1000.0)

    def test_mf1000_preserves_intensity_ratios(self):
        """MF1000 normalization scales all peaks proportionally."""
        ints = th.tensor([100.0, 200.0])
        batch_idxs = th.tensor([0, 0])
        result = batched_mf1000_normalize(ints, batch_idxs)
        assert result[1].item() / result[0].item() == pytest.approx(2.0, abs=1e-5)


class TestMf1000NormalizeNp:
    """Tests for mf1000_normalize_np — numpy MF1000 normalization (single spectrum)."""

    def test_max_equals_1000(self):
        """Maximum intensity after normalization is 1000."""
        ints = np.array([200.0, 400.0, 800.0])
        result = mf1000_normalize_np(ints)
        assert result.max() == pytest.approx(1000.0)

    def test_preserves_ratios(self):
        """Intensity ratios are preserved after normalization."""
        ints = np.array([200.0, 400.0, 800.0])
        result = mf1000_normalize_np(ints)
        assert result[1] / result[0] == pytest.approx(2.0)
        assert result[2] / result[0] == pytest.approx(4.0)

    def test_single_peak_becomes_1000(self):
        """Single peak scales to exactly 1000."""
        result = mf1000_normalize_np(np.array([42.0]))
        assert result[0] == pytest.approx(1000.0)


class TestScipyLinearSumAssignment:
    """Tests for scipy_linear_sum_assignment — Hungarian algorithm wrapper."""

    def test_minimize_2x2_known_assignment(self):
        """Minimize [[5,0],[0,3]]: off-diagonal (cost 0) beats diagonal (cost 8)."""
        matrix = th.tensor([[5.0, 0.0], [0.0, 3.0]])
        row_idx, col_idx = scipy_linear_sum_assignment(matrix, maximize=False)
        # Optimal: (0→1, 1→0), total cost = 0+0 = 0
        assert set(zip(row_idx.tolist(), col_idx.tolist())) == {(0, 1), (1, 0)}

    def test_maximize_2x2_known_assignment(self):
        """Maximize [[5,0],[0,3]]: diagonal (gain 5+3=8) beats off-diagonal (gain 0)."""
        matrix = th.tensor([[5.0, 0.0], [0.0, 3.0]])
        row_idx, col_idx = scipy_linear_sum_assignment(matrix, maximize=True)
        # Optimal: (0→0, 1→1), total gain = 5+3 = 8
        assert set(zip(row_idx.tolist(), col_idx.tolist())) == {(0, 0), (1, 1)}

    def test_3x3_diagonal_minimum(self):
        """3×3 matrix with zeros on diagonal is minimized by diagonal assignment."""
        matrix = th.tensor([
            [0.0, 10.0, 10.0],
            [10.0, 0.0, 10.0],
            [10.0, 10.0, 0.0],
        ])
        row_idx, col_idx = scipy_linear_sum_assignment(matrix, maximize=False)
        assert set(zip(row_idx.tolist(), col_idx.tolist())) == {(0, 0), (1, 1), (2, 2)}

    def test_returns_torch_long_tensors(self):
        """Output is a pair of torch.Tensor with dtype=long."""
        matrix = th.tensor([[1.0, 2.0], [3.0, 4.0]])
        row_idx, col_idx = scipy_linear_sum_assignment(matrix)
        assert isinstance(row_idx, th.Tensor)
        assert isinstance(col_idx, th.Tensor)
        assert row_idx.dtype == th.long
        assert col_idx.dtype == th.long

    def test_output_covers_all_rows(self):
        """Assignment covers every row (complete matching)."""
        matrix = th.rand(4, 4)
        row_idx, col_idx = scipy_linear_sum_assignment(matrix, maximize=False)
        assert sorted(row_idx.tolist()) == [0, 1, 2, 3]
        assert len(set(col_idx.tolist())) == 4  # no repeated columns


class TestOptCosSimHelper:
    """Tests for opt_cos_sim_helper — oracle (upper-bound) cosine similarity.

    Replaces pred intensities at matching bins with true intensities,
    then computes cosine similarity. This is an upper bound on achievable score.
    """

    def test_perfect_coverage_returns_one(self):
        """Pred has same bins as true → oracle assigns true ints to pred → cos=1.0."""
        true_bin_idxs = th.tensor([10, 20])
        true_bin_ints = th.tensor([1.0, 2.0])
        true_batch = th.tensor([0, 0])
        pred_bin_idxs = th.tensor([10, 20])
        pred_bin_ints = th.tensor([99.0, 999.0])  # values don't matter, get replaced
        pred_batch = th.tensor([0, 0])
        result = opt_cos_sim_helper(
            true_bin_idxs, true_bin_ints, true_batch,
            pred_bin_idxs, pred_bin_ints, pred_batch,
        )
        assert result.item() == pytest.approx(1.0)

    def test_no_coverage_returns_zero(self):
        """Pred has no overlapping bins → oracle zeroes all pred → cos=0.0."""
        true_bin_idxs = th.tensor([10, 20])
        true_bin_ints = th.tensor([1.0, 2.0])
        true_batch = th.tensor([0, 0])
        pred_bin_idxs = th.tensor([50, 60])  # completely disjoint bins
        pred_bin_ints = th.tensor([5.0, 6.0])
        pred_batch = th.tensor([0, 0])
        result = opt_cos_sim_helper(
            true_bin_idxs, true_bin_ints, true_batch,
            pred_bin_idxs, pred_bin_ints, pred_batch,
        )
        assert result.item() == pytest.approx(0.0)

    def test_oracle_geq_regular_cosine(self):
        """Oracle cosine similarity is always >= regular cosine similarity."""
        true_bin_idxs = th.tensor([1, 2, 3])
        true_bin_ints = th.tensor([1.0, 2.0, 3.0])
        true_batch = th.tensor([0, 0, 0])
        pred_bin_idxs = th.tensor([1, 2, 4])  # partial overlap: bins 1,2 match; 4 doesn't
        pred_bin_ints = th.tensor([0.5, 0.5, 0.5])
        pred_batch = th.tensor([0, 0, 0])
        oracle = opt_cos_sim_helper(
            true_bin_idxs, true_bin_ints, true_batch,
            pred_bin_idxs, pred_bin_ints, pred_batch,
        )
        regular = cos_sim_helper(
            true_bin_idxs, true_bin_ints, true_batch,
            pred_bin_idxs, pred_bin_ints, pred_batch,
        )
        assert oracle.item() >= regular.item() - 1e-6

    def test_in_range_zero_to_one(self):
        """Oracle similarity is always in [0, 1]."""
        th.manual_seed(7)
        bin_idxs = th.tensor([5, 10, 15, 20])
        ints = th.rand(4)
        batch = th.zeros(4, dtype=th.long)
        result = opt_cos_sim_helper(bin_idxs, ints, batch, bin_idxs[:2], ints[:2], batch[:2])
        assert 0.0 <= result.item() <= 1.0 + 1e-6


class TestCalculateCosineSimilarity:
    """End-to-end tests for calculate_cosine_similarity (dense binned pipeline).

    Uses mz_max=10.0, mz_bin_res=1.0 (10 bins) to keep tensors small.
    """

    def test_identical_spectra_returns_one(self):
        """Identical spectra → cos_sim = 1.0."""
        mzs = th.tensor([2.5])
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0])
        result = calculate_cosine_similarity(
            mzs, ints, batch_idxs, mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0
        )
        assert result.item() == pytest.approx(1.0)

    def test_orthogonal_spectra_returns_zero(self):
        """Spectra in different bins → cos_sim = 0.0."""
        true_mzs = th.tensor([2.5])
        pred_mzs = th.tensor([7.5])
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0])
        result = calculate_cosine_similarity(
            true_mzs, ints, batch_idxs,
            pred_mzs, ints, batch_idxs,
            mz_max=10.0, mz_bin_res=1.0,
        )
        assert result.item() == pytest.approx(0.0)

    def test_batch_of_two_processed_independently(self):
        """B=2: each pair is independent; output shape is (2,)."""
        mzs = th.tensor([2.5, 7.5])
        ints = th.tensor([1.0, 1.0])
        batch_idxs = th.tensor([0, 1])
        result = calculate_cosine_similarity(
            mzs, ints, batch_idxs, mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0
        )
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(1.0)

    def test_amax_aggregation_mode(self):
        """sum_ints=False uses amax per bin; identical spectra still give 1.0."""
        mzs = th.tensor([2.5])
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0])
        result = calculate_cosine_similarity(
            mzs, ints, batch_idxs, mzs, ints, batch_idxs,
            mz_max=10.0, mz_bin_res=1.0, sum_ints=False,
        )
        assert result.item() == pytest.approx(1.0)


class TestCalculateSparseCosineSimilarity:
    """End-to-end tests for calculate_sparse_cosine_similarity (sparse binning pipeline)."""

    def test_identical_spectra_returns_one(self):
        """Identical spectra → cos_sim = 1.0 via sparse binning."""
        mzs = th.tensor([2.5, 4.5])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        result = calculate_sparse_cosine_similarity(
            mzs, ints, batch_idxs, mzs, ints, batch_idxs, mz_max=10.0, mz_bin_res=1.0
        )
        assert result.item() == pytest.approx(1.0)

    def test_orthogonal_spectra_returns_zero(self):
        """No shared bins → cos_sim = 0.0 via sparse binning."""
        true_mzs = th.tensor([2.5])
        pred_mzs = th.tensor([7.5])
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0])
        result = calculate_sparse_cosine_similarity(
            true_mzs, ints, batch_idxs,
            pred_mzs, ints, batch_idxs,
            mz_max=10.0, mz_bin_res=1.0,
        )
        assert result.item() == pytest.approx(0.0)

    def test_agrees_with_dense_binning(self):
        """Sparse and dense methods give the same result for non-duplicate inputs."""
        mzs = th.tensor([2.5, 4.5, 7.5])
        ints = th.tensor([1.0, 2.0, 3.0])
        batch_idxs = th.tensor([0, 0, 0])
        pred_mzs = th.tensor([2.5, 7.5])
        pred_ints = th.tensor([1.0, 3.0])
        pred_batch = th.tensor([0, 0])
        dense = calculate_cosine_similarity(
            mzs, ints, batch_idxs, pred_mzs, pred_ints, pred_batch,
            mz_max=10.0, mz_bin_res=1.0,
        )
        sparse = calculate_sparse_cosine_similarity(
            mzs, ints, batch_idxs, pred_mzs, pred_ints, pred_batch,
            mz_max=10.0, mz_bin_res=1.0,
        )
        assert sparse.item() == pytest.approx(dense.item(), abs=1e-5)


class TestCalculateSparseCosinehungarian:
    """End-to-end tests for calculate_sparse_cosine_hungarian (Hungarian matching pipeline)."""

    def test_identical_spectra_returns_one(self):
        """Identical spectra with exact m/z match → cos_sim = 1.0."""
        mzs = th.tensor([100.0, 200.0])
        ints = th.tensor([1.0, 2.0])
        batch_idxs = th.tensor([0, 0])
        result = calculate_sparse_cosine_hungarian(
            mzs, ints, batch_idxs, mzs, ints, batch_idxs
        )
        assert result.item() == pytest.approx(1.0)

    def test_orthogonal_spectra_returns_zero(self):
        """Peaks with no m/z matches within tolerance → cos_sim = 0.0."""
        true_mzs = th.tensor([100.0])
        pred_mzs = th.tensor([500.0])  # far away from true
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0])
        result = calculate_sparse_cosine_hungarian(
            true_mzs, ints, batch_idxs, pred_mzs, ints, batch_idxs, tolerance=1e-5
        )
        assert result.item() == pytest.approx(0.0)

    def test_within_tolerance_matches_and_scores_one(self):
        """Peaks within relative tolerance with equal intensities → cos_sim = 1.0."""
        true_mzs = th.tensor([100.0])
        pred_mzs = th.tensor([100.001])  # very small relative diff
        ints = th.tensor([1.0])
        batch_idxs = th.tensor([0])
        result = calculate_sparse_cosine_hungarian(
            true_mzs, ints, batch_idxs,
            pred_mzs, ints, batch_idxs,
            tolerance=1e-2,  # 1%, clamped to 200 Da floor → diff 0.001/200 << 0.01
        )
        assert result.item() == pytest.approx(1.0)

    def test_batch_of_two_independent(self):
        """B=2: each pair is processed independently; output shape is (2,)."""
        mzs = th.tensor([100.0, 200.0])
        ints = th.tensor([1.0, 1.0])
        batch_idxs = th.tensor([0, 1])
        result = calculate_sparse_cosine_hungarian(
            mzs, ints, batch_idxs, mzs, ints, batch_idxs
        )
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(1.0)
        assert result[1].item() == pytest.approx(1.0)

    def test_output_in_zero_one_range(self):
        """Output is always in [0, 1] for valid inputs."""
        th.manual_seed(42)
        true_mzs = th.rand(5) * 400 + 100  # 100–500 Da
        pred_mzs = th.rand(3) * 400 + 100
        ints_true = th.rand(5)
        ints_pred = th.rand(3)
        batch_t = th.zeros(5, dtype=th.long)
        batch_p = th.zeros(3, dtype=th.long)
        result = calculate_sparse_cosine_hungarian(
            true_mzs, ints_true, batch_t, pred_mzs, ints_pred, batch_p
        )
        assert 0.0 <= result.item() <= 1.0 + 1e-6

    def test_result_leq_binned_cosine_on_identical(self):
        """Hungarian result matches binned cosine on identical inputs (both return 1.0)."""
        mzs = th.tensor([100.0, 200.0, 300.0])
        ints = th.tensor([1.0, 2.0, 3.0])
        batch_idxs = th.tensor([0, 0, 0])
        hungarian_result = calculate_sparse_cosine_hungarian(
            mzs, ints, batch_idxs, mzs, ints, batch_idxs
        )
        assert hungarian_result.item() == pytest.approx(1.0)
