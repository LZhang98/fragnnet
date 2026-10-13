"""Distances between sparse spectra: cosine, Jensen-Shannon and Hellinger, binned or with
Hungarian peak matching; the peak-list distance on unbinned spectra; and the intensity
fraction predicted outside every observed tolerance window.
"""

import math
from typing import Literal

import torch as th

from fragnnet.loss.padding import (
    build_loss_chunks,
    build_padded_batch_layout,
    pad_flat_by_batch,
    pad_flat_with_layout,
    paired_batch_offsets,
)
from fragnnet.utils.misc_utils import EPS, LOG_ZERO, TOLERANCE_MIN_MZ
from fragnnet.utils.scatter_utils import (
    scatter_logl2normalize,
    scatter_logsumexp,
    scatter_reduce,
    x_log_x,
)
from fragnnet.utils.spec_similarity import (
    align_sparse_peak_support,
    calculate_match_mzs,
    jss_helper,
    linear_sum_assignment,
)
from fragnnet.utils.spec_utils import batched_bin_func, round_aggregate_peaks


def _spectra_in_batch(true_batch_idxs: th.Tensor, pred_batch_idxs: th.Tensor) -> int:
    """Return the number of spectra covered by an observed/predicted peak index pair.

    Inferring the count from the observed peaks alone drops a trailing spectrum that
    holds predictions but no observations. Depending on the caller that either crashes
    a scatter or silently shortens a per-spectrum loss vector, which then broadcasts
    against the rest of the batch instead of raising.

    Args:
        true_batch_idxs: Spectrum index for each observed peak.
        pred_batch_idxs: Spectrum index for each predicted peak.

    Returns:
        One more than the largest spectrum index present in either tensor.

    Raises:
        ValueError: If both peak lists are empty, which leaves the count undefined.
    """
    if true_batch_idxs.numel() == 0 and pred_batch_idxs.numel() == 0:
        raise ValueError("cannot infer batch size from two empty peak lists")
    largest = max(
        int(true_batch_idxs.max().item()) if true_batch_idxs.numel() > 0 else -1,
        int(pred_batch_idxs.max().item()) if pred_batch_idxs.numel() > 0 else -1,
    )
    return largest + 1


def sparse_jensen_shannon_divergence_hungarian_vec(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    log_min: float = EPS,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    loss_batch_size: int = 32,
) -> th.Tensor:
    """Batched Jensen-Shannon distance with Hungarian peak matching.

    The batch spans every spectrum referenced by either peak list. A spectrum with
    no observed or no predicted mass scores one, matching the scalar
    ``sparse_jensen_shannon_divergence_hungarian``. ``log_min`` is retained for
    call compatibility but does not affect either Hungarian implementation.
    """
    true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
        true_mzs, true_logprobs, true_batch_idxs, agg="lse"
    )
    pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
        pred_mzs, pred_logprobs, pred_batch_idxs, agg="lse"
    )
    true_sort_idx = th.argsort(true_batch_idxs)
    true_mzs = true_mzs[true_sort_idx]
    true_logprobs = true_logprobs[true_sort_idx]
    true_batch_idxs = true_batch_idxs[true_sort_idx]
    pred_sort_idx = th.argsort(pred_batch_idxs)
    pred_mzs = pred_mzs[pred_sort_idx]
    pred_logprobs = pred_logprobs[pred_sort_idx]
    pred_batch_idxs = pred_batch_idxs[pred_sort_idx]

    batch_size = _spectra_in_batch(true_batch_idxs, pred_batch_idxs)
    if loss_batch_size is None or loss_batch_size <= 0:
        loss_batch_size = batch_size
    loss_num_batches = int((batch_size // loss_batch_size) + int(batch_size % loss_batch_size > 0))

    true_counts, pred_counts, true_cumsum, pred_cumsum = paired_batch_offsets(
        true_batch_idxs, pred_batch_idxs, batch_size
    )

    jsd_hun = th.zeros(batch_size, device=true_logprobs.device, dtype=true_logprobs.dtype)
    log2 = math.log(2.0)

    for bl in range(loss_num_batches):
        bl_lower = bl * loss_batch_size
        bl_upper = min((bl + 1) * loss_batch_size, batch_size)
        bl_true_lower = true_cumsum[bl_lower]
        bl_true_upper = true_cumsum[bl_upper]
        bl_pred_lower = pred_cumsum[bl_lower]
        bl_pred_upper = pred_cumsum[bl_upper]
        bl_batch_size = bl_upper - bl_lower

        bl_true_batch_idxs = true_batch_idxs[bl_true_lower:bl_true_upper] - bl_lower
        bl_pred_batch_idxs = pred_batch_idxs[bl_pred_lower:bl_pred_upper] - bl_lower
        bl_true_mzs = true_mzs[bl_true_lower:bl_true_upper]
        bl_pred_mzs = pred_mzs[bl_pred_lower:bl_pred_upper]
        bl_true_ints = th.exp(true_logprobs[bl_true_lower:bl_true_upper])
        bl_pred_ints = th.exp(pred_logprobs[bl_pred_lower:bl_pred_upper])

        true_sort, true_indices, true_valid, true_max = build_padded_batch_layout(
            bl_true_batch_idxs,
            bl_batch_size,
            assume_sorted=False,
            counts=true_counts[bl_lower:bl_upper],
            max_count=max(true_cumsum[i + 1] - true_cumsum[i] for i in range(bl_lower, bl_upper)),
        )
        pred_sort, pred_indices, pred_valid, pred_max = build_padded_batch_layout(
            bl_pred_batch_idxs,
            bl_batch_size,
            assume_sorted=False,
            counts=pred_counts[bl_lower:bl_upper],
            max_count=max(pred_cumsum[i + 1] - pred_cumsum[i] for i in range(bl_lower, bl_upper)),
        )
        true_layout = (true_sort, true_indices, bl_batch_size, true_max)
        pred_layout = (pred_sort, pred_indices, bl_batch_size, pred_max)
        batch_true_mzs = pad_flat_with_layout(bl_true_mzs, *true_layout, pad_value=0.0)
        batch_true_ints = pad_flat_with_layout(bl_true_ints, *true_layout, pad_value=0.0)
        batch_pred_mzs = pad_flat_with_layout(bl_pred_mzs, *pred_layout, pad_value=0.0)
        batch_pred_ints = pad_flat_with_layout(bl_pred_ints, *pred_layout, pad_value=0.0)

        if batch_true_mzs.shape[1] == 0 or batch_pred_mzs.shape[1] == 0:
            graph_zero = bl_pred_ints.sum() * 0.0
            jsd_hun[bl_lower:bl_upper] = 1.0 + graph_zero
            continue

        if tol_per_sample is not None:
            bl_tol_per_true = tol_per_sample[bl_lower:bl_upper].to(batch_true_mzs.dtype)
            bl_min_mz_per_true = min_mz_per_sample[bl_lower:bl_upper].to(batch_true_mzs.dtype)
            tol_per_true = bl_tol_per_true[:, None].expand_as(batch_true_mzs)
            min_mz_per_true = bl_min_mz_per_true[:, None].expand_as(batch_true_mzs)
        else:
            tol_per_true = None
            min_mz_per_true = None

        batch_match_mask = calculate_match_mzs(
            batch_true_mzs,
            batch_pred_mzs,
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            tol_per_true=tol_per_true,
            min_mz_per_true=min_mz_per_true,
        )
        batch_match_mask = th.as_tensor(batch_match_mask, device=true_logprobs.device)
        batch_match_mask = batch_match_mask & true_valid[:, :, None] & pred_valid[:, None, :]

        if not bool(batch_match_mask.any().item()):
            graph_zero = bl_pred_ints.sum() * 0.0
            jsd_hun[bl_lower:bl_upper] = 1.0 + graph_zero
            continue

        aligned_true, aligned_pred = align_sparse_peak_support(
            batch_true_mzs,
            batch_true_ints,
            true_valid,
            batch_pred_mzs,
            batch_pred_ints,
            pred_valid,
            batch_match_mask,
            matching="js_gain_hungarian",
        )
        # A chunk can mix comparable spectra with ones that hold no observed or no
        # predicted mass. Replacing zero normalizers keeps those rows finite instead of
        # 0/0, because a NaN branch of th.where would still poison the backward pass.
        true_total = aligned_true.sum(dim=1, keepdim=True)
        pred_total = aligned_pred.sum(dim=1, keepdim=True)
        true_distribution = aligned_true / th.where(true_total > 0, true_total, 1.0)
        pred_distribution = aligned_pred / th.where(pred_total > 0, pred_total, 1.0)
        mixture = 0.5 * (true_distribution + pred_distribution)
        divergence = (
            0.5 * x_log_x(true_distribution).sum(dim=1)
            + 0.5 * x_log_x(pred_distribution).sum(dim=1)
            - x_log_x(mixture).sum(dim=1)
        )
        comparable = ((true_total > 0.0) & (pred_total > 0.0)).squeeze(1)
        jsd_hun[bl_lower:bl_upper] = th.where(
            comparable,
            th.clamp(divergence / log2, min=0.0, max=1.0),
            th.ones_like(divergence),
        )

    return jsd_hun


def sparse_cosine_distance_binned(
    true_bin_idxs: th.Tensor,
    true_bin_logprobs: th.Tensor,
    true_bin_batch_idxs: th.Tensor,
    pred_bin_idxs: th.Tensor,
    pred_bin_logprobs: th.Tensor,
    pred_bin_batch_idxs: th.Tensor,
    log_distance: bool = False,
) -> th.Tensor:
    # l2 normalize
    true_bin_logprobs = scatter_logl2normalize(true_bin_logprobs, true_bin_batch_idxs)
    pred_bin_logprobs = scatter_logl2normalize(pred_bin_logprobs, pred_bin_batch_idxs)
    # dot product
    pred_mask = th.isin(pred_bin_idxs, true_bin_idxs)
    true_mask = th.isin(true_bin_idxs, pred_bin_idxs)
    batch_size = int(th.max(true_bin_batch_idxs).item() + 1)
    if th.any(pred_mask):
        both_bin_logprobs = pred_bin_logprobs[pred_mask] + true_bin_logprobs[true_mask]
        assert th.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
        log_cos_sim = scatter_logsumexp(
            both_bin_logprobs, pred_bin_batch_idxs[pred_mask], dim_size=batch_size
        )
    else:
        # cosine similarities are all zero
        log_cos_sim = th.full(
            size=(batch_size,),
            fill_value=LOG_ZERO(pred_bin_logprobs.dtype),
            dtype=pred_bin_logprobs.dtype,
            device=pred_bin_logprobs.device,
        )
        # involve pred_logprobs to keep gradient
        log_cos_sim = log_cos_sim + 0.0 * th.mean(pred_bin_logprobs, dim=0)
    if log_distance:
        cos_dist = th.log1p(-th.exp(log_cos_sim))
    else:
        cos_dist = 1.0 - th.exp(log_cos_sim)
    return cos_dist


def sparse_cosine_distance_hungarian(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    log_distance: bool = False,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> th.Tensor:
    """
    Compute the sparse cosine distance between spectra using the Hungarian algorithm for optimal matching.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        tolerance: m/z matching tolerance.
        relative: Whether tolerance is relative to m/z.
        tolerance_min_mz: Minimum m/z for relative tolerance.
        log_distance: If True, return log(1-cosine); else return 1-cosine.

    Returns:
        Tensor of cosine distances per batch.
    """

    # Aggregate and round peaks for both true and predicted spectra
    true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
        true_mzs, true_logprobs, true_batch_idxs, agg="lse"
    )
    pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
        pred_mzs, pred_logprobs, pred_batch_idxs, agg="lse"
    )
    # Get number of batches
    batch_size = int(th.max(true_batch_idxs).item() + 1)
    cos_dist_hun = th.zeros(batch_size, device=true_logprobs.device, dtype=true_logprobs.dtype)
    for batch_idx in range(batch_size):
        # Select peaks for current batch
        b_true_mask = true_batch_idxs == batch_idx
        b_pred_mask = pred_batch_idxs == batch_idx
        b_true_mzs = true_mzs[b_true_mask]
        b_pred_mzs = pred_mzs[b_pred_mask]
        b_true_logprobs = true_logprobs[b_true_mask]
        b_pred_logprobs = pred_logprobs[b_pred_mask]
        # L2 normalize log-probabilities for cosine similarity
        b_true_logprobs = b_true_logprobs - 0.5 * th.logsumexp(2 * b_true_logprobs, dim=0)
        b_pred_logprobs = b_pred_logprobs - 0.5 * th.logsumexp(2 * b_pred_logprobs, dim=0)
        # per-sample tolerance (falls back to global scalars when not provided)
        b_tolerance = tol_per_sample[batch_idx].item() if tol_per_sample is not None else tolerance
        b_min_mz = (
            min_mz_per_sample[batch_idx].item()
            if min_mz_per_sample is not None
            else tolerance_min_mz
        )
        # Compute m/z matching mask within tolerance
        b_match_mzs = calculate_match_mzs(
            b_true_mzs,
            b_pred_mzs,
            tolerance=b_tolerance,
            relative=relative,
            tolerance_min_mz=b_min_mz,
        )
        b_match_mzs = th.as_tensor(b_match_mzs, device=b_true_mzs.device)
        # Identify which true and predicted peaks have matches
        b_true_match_mzs = th.any(b_match_mzs, dim=1)
        b_pred_match_mzs = th.any(b_match_mzs, dim=0)
        # Compute score matrix for matched peaks (outer sum of logprobs, exponentiated)
        b_score = th.exp(
            b_true_logprobs[b_true_match_mzs].detach().unsqueeze(1)
            + b_pred_logprobs[b_pred_match_mzs].detach().unsqueeze(0)
        )
        valid_matches = b_match_mzs[b_true_match_mzs][:, b_pred_match_mzs]
        # Infeasible pairs contribute no cosine, including when the solver must
        # complete an assignment through them. Zero also preserves feasible score differences.
        b_score[~valid_matches] = 0.0
        # Hungarian algorithm for optimal matching (maximizing score)
        b_true_idxs, b_pred_idxs = linear_sum_assignment(b_score, maximize=True)
        feasible = valid_matches[b_true_idxs, b_pred_idxs]
        b_true_idxs = b_true_idxs[feasible]
        b_pred_idxs = b_pred_idxs[feasible]
        # Compute log-cosine similarity for matched pairs
        b_log_cos_hun = th.logsumexp(
            b_true_logprobs[b_true_match_mzs][b_true_idxs]
            + b_pred_logprobs[b_pred_match_mzs][b_pred_idxs],
            dim=0,
        )
        # Convert to cosine distance
        if log_distance:
            b_cos_dist_hun = th.log1p(-th.exp(b_log_cos_hun))
        else:
            b_cos_dist_hun = 1.0 - th.exp(b_log_cos_hun)
        cos_dist_hun[batch_idx] = b_cos_dist_hun
    return cos_dist_hun


def sparse_jensen_shannon_divergence_hungarian(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    log_min: float = EPS,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> th.Tensor:
    """
    Compute the sparse Jensen-Shannon divergence between spectra using the Hungarian algorithm for optimal matching.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        tolerance: m/z matching tolerance.
        relative: Whether tolerance is relative to m/z.
        tolerance_min_mz: Minimum m/z for relative tolerance.
        log_min: Retained for call compatibility; unused by this Hungarian objective.

    Returns:
        Tensor of Jensen-Shannon divergences per batch (normalized to [0,1]).
    """

    # Aggregate and round peaks for both true and predicted spectra
    true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
        true_mzs, true_logprobs, true_batch_idxs, agg="lse"
    )
    pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
        pred_mzs, pred_logprobs, pred_batch_idxs, agg="lse"
    )
    # Get number of batches
    batch_size = _spectra_in_batch(true_batch_idxs, pred_batch_idxs)
    jsd_hun = th.zeros(batch_size, device=true_logprobs.device, dtype=true_logprobs.dtype)
    LOG2 = math.log(2.0)  # precompute: avoids th.log(th.tensor(2.0)) inside the loop

    # Split peaks by batch element once (O(N)) instead of masking per element (O(B×N)).
    true_counts = th.bincount(true_batch_idxs, minlength=batch_size).tolist()
    pred_counts = th.bincount(pred_batch_idxs, minlength=batch_size).tolist()
    true_mzs_list = th.split(true_mzs, true_counts)
    true_logprobs_list = th.split(true_logprobs, true_counts)
    pred_mzs_list = th.split(pred_mzs, pred_counts)
    pred_logprobs_list = th.split(pred_logprobs, pred_counts)

    for batch_idx in range(batch_size):
        b_true_mzs = true_mzs_list[batch_idx]
        b_pred_mzs = pred_mzs_list[batch_idx]
        b_true_logprobs = true_logprobs_list[batch_idx]
        b_pred_logprobs = pred_logprobs_list[batch_idx]

        # Normalize probabilities (sum to 1) without GPU sync.
        # Only replace zero totals: an EPS floor distorts tiny positive spectra,
        # while dividing zero mass by dtype.tiny can overflow its backward derivative.
        b_true_probs = th.exp(b_true_logprobs)
        true_total = b_true_probs.sum()
        b_true_probs = b_true_probs / th.where(true_total > 0, true_total, 1.0)

        b_pred_probs = th.exp(b_pred_logprobs)
        pred_total = b_pred_probs.sum()
        b_pred_probs = b_pred_probs / th.where(pred_total > 0, pred_total, 1.0)

        # per-sample tolerance (falls back to global scalars when not provided)
        b_tolerance = tol_per_sample[batch_idx].item() if tol_per_sample is not None else tolerance
        b_min_mz = (
            min_mz_per_sample[batch_idx].item()
            if min_mz_per_sample is not None
            else tolerance_min_mz
        )
        # Compute m/z matching mask within tolerance
        b_match_mzs = calculate_match_mzs(
            b_true_mzs,
            b_pred_mzs,
            tolerance=b_tolerance,
            relative=relative,
            tolerance_min_mz=b_min_mz,
        )
        b_match_mzs = th.as_tensor(b_match_mzs, device=b_true_mzs.device)

        # Identify which true and predicted peaks have matches
        b_true_match_mzs = th.any(b_match_mzs, dim=1)
        b_pred_match_mzs = th.any(b_match_mzs, dim=0)

        # Filter to only matched peaks to reduce matrix size
        b_true_probs_m = b_true_probs[b_true_match_mzs]
        b_pred_probs_m = b_pred_probs[b_pred_match_mzs]

        # Compute score matrix (Gain)
        # Gain = 0.5 * [ (p+q) log(p+q) - p log p - q log q ]
        p_expand = b_true_probs_m.unsqueeze(1)
        q_expand = b_pred_probs_m.unsqueeze(0)
        sum_pq = p_expand + q_expand

        term1 = sum_pq * th.log(sum_pq + EPS)
        term2 = p_expand * th.log(p_expand + EPS)
        term3 = q_expand * th.log(q_expand + EPS)

        b_score = 0.5 * (term1 - term2 - term3)

        # Mask invalid matches
        valid_matches = b_match_mzs[b_true_match_mzs][:, b_pred_match_mzs]
        # Set invalid matches to a large negative number
        b_score[~valid_matches] = -1e9

        # Hungarian algorithm
        if b_score.numel() == 0:
            # No matched peaks => assignment matrix is empty.
            # Keep `total_gain` connected to the computation graph so downstream
            # loss still has `requires_grad=True` (manual backward would otherwise crash).
            if b_pred_probs.numel() > 0:
                total_gain = b_pred_probs.sum() * 0.0
            elif b_true_probs.numel() > 0:
                total_gain = b_true_probs.sum() * 0.0
            else:
                total_gain = th.tensor(0.0, device=b_score.device, dtype=b_score.dtype)
        else:
            # We detach b_score for the assignment step because the Hungarian algorithm
            # is non-differentiable (calculates indices).
            # Gradients will flow through the values selected from the original b_score below.
            b_true_idxs, b_pred_idxs = linear_sum_assignment(b_score.detach(), maximize=True)

            # Filter invalid matches (where score is -1e9)
            matched_gains = b_score[b_true_idxs, b_pred_idxs]
            valid_mask = matched_gains > -1e8
            if th.any(valid_mask):
                total_gain = th.sum(matched_gains[valid_mask])
            else:
                # All matches invalid; return a graph-connected zero.
                total_gain = b_score.sum() * 0.0

        # Normalized JSD = 1 - Total Gain / log 2
        b_jsd = 1.0 - total_gain / LOG2
        b_jsd = th.clamp(b_jsd, min=0.0, max=1.0)
        jsd_hun[batch_idx] = b_jsd

    return jsd_hun


def sparse_jensen_shannon_divergence(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    mz_max: float = 1500.0,
    mz_bin_res: float = 0.01,
    log_min: float = EPS,
) -> th.Tensor:
    """
    Compute the sparse Jensen-Shannon divergence between two sets of spectra.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        mz_max: Maximum m/z value for binning.
        mz_bin_res: Resolution of m/z bins.
        log_min: Minimum log value to avoid log(0).

    Note:
        The implicit spectrum support size is
        ``num_bins = round(mz_max / mz_bin_res)``. With defaults,
        this is 150,000 bins.

    Returns:
        Jensen-Shannon divergence (distance) between the binned spectra, per batch.
    """

    # Bin the ground truth peaks into sparse bins using batched_bin_func
    true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",  # Use log-sum-exp aggregation for intensities
        sparse=True,  # Use sparse representation for efficiency
    )
    # Bin the predicted peaks into sparse bins
    pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True,
    )
    # Calculate Jensen-Shannon similarity using the helper function
    jss = jss_helper(
        true_bin_idxs,
        true_bin_logprobs.exp(),  # Convert log-probabilities to probabilities
        true_bin_batch_idxs,
        pred_bin_idxs,
        pred_bin_logprobs.exp(),
        pred_bin_batch_idxs,
        log_min=log_min,
    )
    # Return 1 - similarity to get the divergence (distance)
    return 1.0 - jss


def sparse_hellinger_distance(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    mz_max: float = 1500.0,
    mz_bin_res: float = 0.01,
) -> th.Tensor:
    """Compute the squared Hellinger distance from sparse binned spectra.

    This uses the Bhattacharyya form ``1 - sum(sqrt(p_i * t_i))`` after
    independently L1-normalizing each spectrum. That is ``H^2``, not the
    literature's ``H = sqrt(1 - BC)``: the squared form is deliberate because its
    gradient stays bounded as the spectra agree, but its magnitude is not
    comparable with published Hellinger numbers. Only shared occupied bins need
    to be materialized; prediction-only and target-only bins contribute zero to
    the coefficient while still affecting normalization.

    Degenerate spectra are defined rather than rejected: a spectrum that holds
    predictions but no observed peaks shares no bin, so its coefficient is zero and
    its distance is one. The batch therefore spans every spectrum referenced by
    either peak list, not only the observed one.

    Args:
        true_mzs: Ground-truth m/z values.
        true_logprobs: Ground-truth log intensities.
        true_batch_idxs: Batch indices for ground-truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log intensities.
        pred_batch_idxs: Batch indices for predicted peaks.
        mz_max: Maximum m/z represented by the binning geometry.
        mz_bin_res: Width of each m/z bin.

    Returns:
        Hellinger distance per spectrum, in ``[0, 1]``.
    """
    true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
        true_mzs,
        true_logprobs,
        true_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True,
    )
    pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
        pred_mzs,
        pred_logprobs,
        pred_batch_idxs,
        mz_max=mz_max,
        mz_bin_res=mz_bin_res,
        agg="lse",
        sparse=True,
    )
    batch_size = _spectra_in_batch(true_batch_idxs, pred_batch_idxs)
    true_bin_logprobs = true_bin_logprobs - scatter_logsumexp(
        true_bin_logprobs,
        true_bin_batch_idxs,
        dim_size=batch_size,
    )[true_bin_batch_idxs]
    pred_bin_logprobs = pred_bin_logprobs - scatter_logsumexp(
        pred_bin_logprobs,
        pred_bin_batch_idxs,
        dim_size=batch_size,
    )[pred_bin_batch_idxs]

    # ``batched_bin_func`` offsets bin ids by ``batch_idx * num_bins`` and ``th.unique``
    # returns them sorted, so the two selections below are the same bin set in the same
    # order. Compare the bin ids themselves rather than only their spectra, because
    # equal spectra would not prove equal bins.
    pred_shared = th.isin(pred_bin_idxs, true_bin_idxs)
    true_shared = th.isin(true_bin_idxs, pred_bin_idxs)
    if th.any(pred_shared):
        if not th.equal(pred_bin_idxs[pred_shared], true_bin_idxs[true_shared]):
            raise RuntimeError("shared Hellinger bins must align elementwise")
        terms = th.exp(0.5 * (pred_bin_logprobs[pred_shared] + true_bin_logprobs[true_shared]))
        coefficient = scatter_reduce(
            src=terms,
            index=pred_bin_batch_idxs[pred_shared],
            reduce="sum",
            dim_size=batch_size,
            default=0.0,
        )
    else:
        graph_zero = pred_bin_logprobs.sum() * 0.0
        coefficient = pred_bin_logprobs.new_zeros(batch_size) + graph_zero
    return 1.0 - coefficient.clamp(min=0.0, max=1.0)


def _bhattacharyya_coefficient(
    pred_distribution: th.Tensor,
    true_distribution: th.Tensor,
    *,
    dim: int,
) -> th.Tensor:
    """Return a zero-safe Bhattacharyya coefficient along ``dim``.

    ``1 - BC`` is the squared Hellinger distance ``H^2``, which is what the
    ``hellinger_distance`` objective reports.

    Clamping is used only inside square roots. Structural-zero coordinates are
    restored to exactly zero, which avoids undefined square-root gradients while
    preserving the coefficient and gradients at every positive coordinate.
    """
    tiny = th.finfo(th.promote_types(pred_distribution.dtype, true_distribution.dtype)).tiny
    positive_support = (pred_distribution > 0.0) & (true_distribution > 0.0)
    terms = pred_distribution.clamp_min(tiny).sqrt() * true_distribution.clamp_min(tiny).sqrt()
    return th.where(positive_support, terms, th.zeros_like(terms)).sum(dim=dim)


def sparse_outside_window_intensity_fraction(
    true_mzs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_intensities: th.Tensor,
    pred_batch_idxs: th.Tensor,
    *,
    batch_size: int,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    loss_batch_size: int = 32,
) -> th.Tensor:
    """Return predicted intensity outside every observed tolerance window.

    The result is normalized by total predicted intensity in each spectrum. It
    depends only on m/z feasibility, not on closest or Hungarian assignment, so
    an in-window prediction is never labeled leakage merely because another peak
    won a one-to-one assignment.

    Spectra are padded per chunk rather than looped over one at a time, so the
    cost follows each chunk's peak-count tails instead of the batch size, and no
    per-spectrum device synchronization is needed. ``batch_size`` is required
    because the observed peaks alone do not determine it: a spectrum with no
    observed peaks would otherwise shorten the result and broadcast silently
    against a per-spectrum loss.

    Degenerate spectra are defined rather than rejected. A spectrum with no
    predicted intensity returns zero because it leaks nothing, and a spectrum
    with no observed peaks returns one because no prediction can fall inside a
    window. Both carry zero gradient.

    Args:
        true_mzs: Observed m/z values in flat batch form.
        true_batch_idxs: Spectrum index for each observed peak.
        pred_mzs: Predicted m/z values in flat batch form.
        pred_intensities: Nonnegative predicted intensities.
        pred_batch_idxs: Spectrum index for each predicted peak.
        batch_size: Number of spectra in the batch, including empty ones.
        tolerance: Relative or absolute matching tolerance.
        relative: Interpret ``tolerance`` relative to observed m/z when true.
        tolerance_min_mz: Divisor floor used by relative tolerance.
        tol_per_sample: Optional tolerance per spectrum, shape ``(batch_size,)``.
        min_mz_per_sample: Optional relative-tolerance floor per spectrum.
        loss_batch_size: Spectra padded together per chunk; values below one use
            a single chunk.

    Returns:
        Per-spectrum outside-window intensity fractions in ``[0, 1]``, shape
        ``(batch_size,)``.

    Raises:
        ValueError: If tensor shapes disagree, ``batch_size`` is not positive, or
            a batch index falls outside ``batch_size``.
    """
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    if true_mzs.shape != true_batch_idxs.shape:
        raise ValueError("observed m/z and batch indices must have matching shapes")
    if pred_mzs.shape != pred_intensities.shape or pred_mzs.shape != pred_batch_idxs.shape:
        raise ValueError("predicted peak tensors must have matching shapes")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    for name, batch_idxs in (("observed", true_batch_idxs), ("predicted", pred_batch_idxs)):
        if batch_idxs.numel() > 0 and int(batch_idxs.max().item()) >= batch_size:
            raise ValueError(f"{name} batch indices must be below batch_size {batch_size}")
    if loss_batch_size <= 0:
        loss_batch_size = batch_size

    # 1. Group both peak lists by spectrum so every chunk is a contiguous slice.
    true_sort_idx = th.argsort(true_batch_idxs)
    true_mzs = true_mzs[true_sort_idx]
    true_batch_idxs = true_batch_idxs[true_sort_idx]
    pred_sort_idx = th.argsort(pred_batch_idxs)
    pred_mzs = pred_mzs[pred_sort_idx]
    pred_intensities = pred_intensities[pred_sort_idx]
    pred_batch_idxs = pred_batch_idxs[pred_sort_idx]

    true_counts, pred_counts, true_cumsum, pred_cumsum = paired_batch_offsets(
        true_batch_idxs, pred_batch_idxs, batch_size
    )

    fractions = pred_intensities.new_zeros(batch_size)
    tiny = th.finfo(pred_intensities.dtype).tiny

    # 2. Pad each chunk, mark predictions covered by any observed window, and reduce.
    for bl_lower in range(0, batch_size, loss_batch_size):
        bl_upper = min(bl_lower + loss_batch_size, batch_size)
        bl_batch_size = bl_upper - bl_lower
        bl_true = slice(true_cumsum[bl_lower], true_cumsum[bl_upper])
        bl_pred = slice(pred_cumsum[bl_lower], pred_cumsum[bl_upper])
        bl_true_batch_idxs = true_batch_idxs[bl_true] - bl_lower
        bl_pred_batch_idxs = pred_batch_idxs[bl_pred] - bl_lower

        pred_sort, pred_indices, pred_valid, pred_max = build_padded_batch_layout(
            bl_pred_batch_idxs,
            bl_batch_size,
            assume_sorted=False,
            counts=pred_counts[bl_lower:bl_upper],
            max_count=max(pred_cumsum[i + 1] - pred_cumsum[i] for i in range(bl_lower, bl_upper)),
        )
        pred_layout = (pred_sort, pred_indices, bl_batch_size, pred_max)
        padded_pred_ints = pad_flat_with_layout(
            pred_intensities[bl_pred], *pred_layout, pad_value=0.0
        )
        padded_pred_mzs = pad_flat_with_layout(pred_mzs[bl_pred], *pred_layout, pad_value=0.0)
        padded_true_mzs, true_valid = pad_flat_by_batch(
            true_mzs[bl_true], bl_true_batch_idxs, bl_batch_size, 0.0
        )
        total_intensity = padded_pred_ints.sum(dim=1)

        if padded_true_mzs.shape[1] == 0 or padded_pred_mzs.shape[1] == 0:
            # Nothing observed to cover a prediction, or nothing predicted at all.
            inside_window = pred_valid.new_zeros(pred_valid.shape)
        else:
            tol_per_true = None
            min_mz_per_true = None
            if tol_per_sample is not None:
                tol_per_true = (
                    tol_per_sample[bl_lower:bl_upper]
                    .to(padded_true_mzs.dtype)[:, None]
                    .expand_as(padded_true_mzs)
                )
                min_mz_per_true = (
                    min_mz_per_sample[bl_lower:bl_upper]
                    .to(padded_true_mzs.dtype)[:, None]
                    .expand_as(padded_true_mzs)
                )
            match_mask = calculate_match_mzs(
                padded_true_mzs,
                padded_pred_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
                tol_per_true=tol_per_true,
                min_mz_per_true=min_mz_per_true,
            )
            match_mask = th.as_tensor(match_mask, device=pred_mzs.device, dtype=th.bool)
            # Padded slots hold m/z 0 on both sides and would otherwise match.
            match_mask = match_mask & true_valid[:, :, None] & pred_valid[:, None, :]
            inside_window = match_mask.any(dim=1)

        outside_intensity = th.where(
            inside_window, th.zeros_like(padded_pred_ints), padded_pred_ints
        ).sum(dim=1)
        # Clamping keeps a spectrum with no predicted intensity at zero instead of 0/0.
        fractions[bl_lower:bl_upper] = outside_intensity / total_intensity.clamp_min(tiny)
    return fractions


def sparse_peak_list_distance(
    true_mzs: th.Tensor,
    true_intensities: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_intensities: th.Tensor,
    pred_batch_idxs: th.Tensor,
    *,
    objective: Literal[
        "cosine_distance",
        "hellinger_distance",
        "jss_distance",
        "kl_distance",
    ],
    matching: Literal["closest", "hungarian"],
    pred_dump_intensities: th.Tensor | None = None,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    decimals: int = 4,
    kl_epsilon: float = 1e-4,
    loss_batch_size: int = 32,
) -> tuple[th.Tensor, th.Tensor]:
    """Compare sparse peak lists without an m/z probability density.

    Peaks are first coalesced at numerical exact mass by rounding to ``decimals``.
    Correspondence then depends only on m/z. ``closest`` assigns every predicted
    peak to its nearest in-tolerance observed peak and permits many-to-one matches;
    ``hungarian`` finds a minimum-distance one-to-one assignment. Unmatched peaks
    from either spectrum remain in the loss support.

    ``pred_dump_intensities`` adds an escape coordinate. Without it the predicted
    intensity must normalize over the candidate m/z values alone, while part of the
    observed intensity sits at m/z values no candidate reaches -- so the model can
    never place that mass and is forced to over-allocate everything it can reach. The
    escape collapses every unreachable observed peak into one coordinate and lets the
    model predict its mass, which removes a per-spectrum irreducible loss floor that
    varies with coverage and therefore weights spectra by how well the enumeration
    happened to do. Reachability is read from the tolerance ``match_mask``, not from
    the predicted values, so a peak a candidate *could* explain stays in the support
    even when the model assigns it nothing and a Hungarian assignment leaves it
    unmatched. For ``kl_distance`` the escape also makes the divergence finite on
    coverage gaps rather than relying on ``kl_epsilon`` smoothing. The escape coordinate
    is part of the distribution, so the returned entropy mismatch scores it too; two
    spectra with the same shape over reachable candidates but different escape shares
    have different entropy mismatches, which is intended.

    Args:
        true_mzs: Observed m/z values in flat batch form.
        true_intensities: Nonnegative observed intensities.
        true_batch_idxs: Spectrum index for each observed peak.
        pred_mzs: Predicted m/z values in flat batch form.
        pred_intensities: Nonnegative predicted intensities. Individual zeros are
            valid, but every complete spectrum must have positive total intensity.
        pred_batch_idxs: Spectrum index for each predicted peak.
        objective: Direct spectral objective.
        matching: M/z-only correspondence rule.
        pred_dump_intensities: Optional nonnegative predicted escape mass per spectrum,
            shape ``(batch_size,)``, in the same units as ``pred_intensities``. ``None``
            keeps the historical behaviour exactly.
        tolerance: Relative or absolute matching tolerance.
        relative: Interpret ``tolerance`` relative to observed m/z when true.
        tolerance_min_mz: Floor used by relative tolerance.
        tol_per_sample: Optional relative tolerance per spectrum.
        min_mz_per_sample: Optional relative-tolerance floor per spectrum.
        decimals: Decimal precision for numerical exact-mass coalescing.
        kl_epsilon: Fraction of the predicted distribution replaced by the
            observed distribution before Kullback-Leibler divergence.
        loss_batch_size: Ceiling on the number of spectra padded together. Chunks are
            narrowed further whenever padding would exceed the flat Cartesian pair count.

    Returns:
        Two per-spectrum vectors: primary distance and squared entropy mismatch.
    """
    if objective not in {
        "cosine_distance",
        "hellinger_distance",
        "jss_distance",
        "kl_distance",
    }:
        raise ValueError(f"unknown peak-list objective: {objective}")
    if matching not in {"closest", "hungarian"}:
        raise ValueError(f"unknown peak-list matching mode: {matching}")
    if not math.isfinite(kl_epsilon) or not 0.0 < kl_epsilon < 1.0:
        raise ValueError("kl_epsilon must be finite and in (0, 1)")
    if loss_batch_size <= 0:
        raise ValueError(f"loss_batch_size must be > 0, got {loss_batch_size}")
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    if pred_dump_intensities is not None:
        if pred_dump_intensities.ndim != 1:
            raise ValueError("pred_dump_intensities must be one-dimensional")
        if not bool(th.isfinite(pred_dump_intensities).all().item()):
            raise ValueError("pred_dump_intensities must be finite")
        if bool((pred_dump_intensities < 0).any().item()):
            raise ValueError("pred_dump_intensities must be nonnegative")
    tensors = {
        "true_mzs": true_mzs,
        "true_intensities": true_intensities,
        "true_batch_idxs": true_batch_idxs,
        "pred_mzs": pred_mzs,
        "pred_intensities": pred_intensities,
        "pred_batch_idxs": pred_batch_idxs,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if true_mzs.shape != true_intensities.shape or true_mzs.shape != true_batch_idxs.shape:
        raise ValueError("observed peak tensors must have matching shapes")
    if pred_mzs.shape != pred_intensities.shape or pred_mzs.shape != pred_batch_idxs.shape:
        raise ValueError("predicted peak tensors must have matching shapes")
    if true_batch_idxs.dtype != th.long or pred_batch_idxs.dtype != th.long:
        raise ValueError("peak batch indices must have dtype torch.long")
    if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
        raise ValueError("observed and predicted peak lists must not be empty")
    if not bool(th.isfinite(true_intensities).all().item()) or not bool(
        th.isfinite(pred_intensities).all().item()
    ):
        raise ValueError("peak intensities must be finite")
    if bool((true_intensities < 0).any().item()) or bool((pred_intensities < 0).any().item()):
        raise ValueError("peak intensities must be nonnegative")

    true_mzs, true_intensities, true_batch_idxs = round_aggregate_peaks(
        true_mzs,
        true_intensities,
        true_batch_idxs,
        decimals=decimals,
        agg="sum",
    )
    pred_mzs, pred_intensities, pred_batch_idxs = round_aggregate_peaks(
        pred_mzs,
        pred_intensities,
        pred_batch_idxs,
        decimals=decimals,
        agg="sum",
    )
    # ``round_aggregate_peaks`` returns groups in (batch, m/z) order, so the flat
    # tensors are already batch-major and chunk slicing below is contiguous.
    batch_size = int(true_batch_idxs.max().item()) + 1
    if pred_batch_idxs.min().item() < 0 or pred_batch_idxs.max().item() >= batch_size:
        raise ValueError("predicted batch indices reference a missing observed spectrum")
    if pred_dump_intensities is not None and pred_dump_intensities.shape[0] != batch_size:
        raise ValueError(
            "pred_dump_intensities must hold one value per spectrum, got "
            f"{pred_dump_intensities.shape[0]} for {batch_size} spectra"
        )

    true_batch_counts, pred_batch_counts, true_batch_cumsum, pred_batch_cumsum = (
        paired_batch_offsets(true_batch_idxs, pred_batch_idxs, batch_size)
    )
    empty_spectra = (true_batch_counts == 0) | (pred_batch_counts == 0)
    if bool(empty_spectra.any().item()):
        first_empty = int(empty_spectra.nonzero()[0, 0].item())
        raise ValueError(f"spectrum {first_empty} must contain observed and predicted peaks")
    true_batch_totals = true_intensities.new_zeros(batch_size).index_add(
        0, true_batch_idxs, true_intensities
    )
    if bool((true_batch_totals <= 0).any().item()):
        first_zero = int((true_batch_totals <= 0).nonzero()[0, 0].item())
        raise ValueError(f"observed spectrum {first_zero} has zero total intensity")
    pred_batch_totals = pred_intensities.new_zeros(batch_size).index_add(
        0, pred_batch_idxs, pred_intensities
    )
    if bool((pred_batch_totals <= 0).any().item()):
        first_zero = int((pred_batch_totals <= 0).nonzero()[0, 0].item())
        raise ValueError(f"predicted spectrum {first_zero} has zero total intensity")

    loss_chunks = build_loss_chunks(true_batch_cumsum, pred_batch_cumsum, loss_batch_size)

    primary_losses = true_intensities.new_empty(batch_size)
    entropy_losses = true_intensities.new_empty(batch_size)
    log_two = math.log(2.0)
    device = true_intensities.device

    for bl_lower, bl_upper in loss_chunks:
        bl_true_lower = true_batch_cumsum[bl_lower]
        bl_true_upper = true_batch_cumsum[bl_upper]
        bl_pred_lower = pred_batch_cumsum[bl_lower]
        bl_pred_upper = pred_batch_cumsum[bl_upper]
        bl_batch_size = bl_upper - bl_lower

        true_sort_idx, true_padded_indices, true_valid, max_true_peaks = build_padded_batch_layout(
            true_batch_idxs[bl_true_lower:bl_true_upper] - bl_lower,
            bl_batch_size,
            assume_sorted=True,
            counts=true_batch_counts[bl_lower:bl_upper],
        )
        pred_sort_idx, pred_padded_indices, pred_valid, max_pred_peaks = build_padded_batch_layout(
            pred_batch_idxs[bl_pred_lower:bl_pred_upper] - bl_lower,
            bl_batch_size,
            assume_sorted=True,
            counts=pred_batch_counts[bl_lower:bl_upper],
        )
        true_layout = (true_sort_idx, true_padded_indices, bl_batch_size, max_true_peaks)
        pred_layout = (pred_sort_idx, pred_padded_indices, bl_batch_size, max_pred_peaks)

        padded_true_mzs = pad_flat_with_layout(
            true_mzs[bl_true_lower:bl_true_upper], *true_layout, pad_value=0.0
        )
        padded_true_ints = pad_flat_with_layout(
            true_intensities[bl_true_lower:bl_true_upper], *true_layout, pad_value=0.0
        )
        padded_pred_mzs = pad_flat_with_layout(
            pred_mzs[bl_pred_lower:bl_pred_upper], *pred_layout, pad_value=0.0
        )
        padded_pred_ints = pad_flat_with_layout(
            pred_intensities[bl_pred_lower:bl_pred_upper], *pred_layout, pad_value=0.0
        )

        if tol_per_sample is None:
            match_mask = calculate_match_mzs(
                padded_true_mzs,
                padded_pred_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
        else:
            bl_tol = tol_per_sample[bl_lower:bl_upper].to(padded_true_mzs.dtype)
            bl_min_mz = min_mz_per_sample[bl_lower:bl_upper].to(padded_true_mzs.dtype)
            match_mask = calculate_match_mzs(
                padded_true_mzs,
                padded_pred_mzs,
                relative=True,
                tol_per_true=bl_tol[:, None].expand_as(padded_true_mzs),
                min_mz_per_true=bl_min_mz[:, None].expand_as(padded_true_mzs),
            )
        match_mask = th.as_tensor(match_mask, dtype=th.bool, device=device)
        match_mask = match_mask & true_valid[:, :, None] & pred_valid[:, None, :]
        aligned_true, aligned_pred = align_sparse_peak_support(
            padded_true_mzs,
            padded_true_ints,
            true_valid,
            padded_pred_mzs,
            padded_pred_ints,
            pred_valid,
            match_mask,
            matching="closest" if matching == "closest" else "mass_hungarian",
        )

        if pred_dump_intensities is not None:
            # A true peak is reachable when some candidate falls inside its tolerance
            # window, which is the match mask, never the predicted value: a candidate the
            # model scored zero, or one a Hungarian assignment gave to a neighbour, is still
            # reachable and must keep facing the loss.
            reachable = match_mask.any(dim=2)
            observed = aligned_true[:, :max_true_peaks]
            dump_true = (observed * (~reachable)).sum(dim=1, keepdim=True)
            # The tail holds unmatched predictions, whose observed side is structurally zero,
            # so only the observed block needs masking and no full-width buffer is allocated.
            aligned_true = th.cat(
                (
                    observed * reachable,
                    aligned_true[:, max_true_peaks:],
                    dump_true,
                ),
                dim=1,
            )
            dump_pred = (
                pred_dump_intensities[bl_lower:bl_upper].reshape(-1, 1).to(aligned_pred.dtype)
            )
            aligned_pred = th.cat((aligned_pred, dump_pred), dim=1)

        distance_dtype = th.promote_types(aligned_pred.dtype, aligned_true.dtype)
        if distance_dtype in {th.float16, th.bfloat16}:
            distance_dtype = th.float32
        distance_pred = aligned_pred.to(dtype=distance_dtype)
        distance_true = aligned_true.to(dtype=distance_dtype)

        pred_distribution = distance_pred / distance_pred.sum(dim=1, keepdim=True)
        true_distribution = distance_true / distance_true.sum(dim=1, keepdim=True)

        if objective == "cosine_distance":
            similarity = (distance_pred * distance_true).sum(dim=1) / (
                th.linalg.vector_norm(distance_pred, dim=1)
                * th.linalg.vector_norm(distance_true, dim=1)
            )
            primary_losses[bl_lower:bl_upper] = 1.0 - similarity
        else:
            if objective == "hellinger_distance":
                coefficient = _bhattacharyya_coefficient(
                    pred_distribution,
                    true_distribution,
                    dim=1,
                )
                primary_losses[bl_lower:bl_upper] = 1.0 - coefficient
            elif objective == "jss_distance":
                mixture = 0.5 * (pred_distribution + true_distribution)
                divergence = (
                    0.5 * x_log_x(pred_distribution).sum(dim=1)
                    + 0.5 * x_log_x(true_distribution).sum(dim=1)
                    - x_log_x(mixture).sum(dim=1)
                )
                primary_losses[bl_lower:bl_upper] = divergence / log_two
            else:
                # Reserve a fixed total epsilon mass for observed support. Unlike adding
                # epsilon per aligned entry, this remains invariant when the same unmatched
                # intensity is split across more predicted peaks.
                smoothed_pred_distribution = (
                    1.0 - kl_epsilon
                ) * pred_distribution + kl_epsilon * true_distribution
                positive_true = true_distribution > 0.0
                tiny = th.finfo(true_distribution.dtype).tiny
                divergence_terms = true_distribution * (
                    true_distribution.clamp_min(tiny).log()
                    - smoothed_pred_distribution.clamp_min(tiny).log()
                )
                primary_losses[bl_lower:bl_upper] = th.where(
                    positive_true, divergence_terms, th.zeros_like(divergence_terms)
                ).sum(dim=1)

        pred_entropy = -x_log_x(pred_distribution).sum(dim=1)
        true_entropy = -x_log_x(true_distribution).sum(dim=1)
        entropy_losses[bl_lower:bl_upper] = (pred_entropy - true_entropy).square()

    return primary_losses, entropy_losses
