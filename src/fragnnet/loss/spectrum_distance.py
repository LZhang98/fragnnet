"""Distances between sparse spectra: cosine, Jensen-Shannon and Hellinger, binned or with
Hungarian peak matching.
"""

import math

import torch as th

from fragnnet.loss.padding import pad_flat_by_batch
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
    ``sparse_jensen_shannon_divergence_hungarian``.
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

    true_counts = th.bincount(true_batch_idxs, minlength=batch_size)
    pred_counts = th.bincount(pred_batch_idxs, minlength=batch_size)
    true_cumsum = th.cat([true_batch_idxs.new_zeros(1), true_counts.cumsum(0)]).tolist()
    pred_cumsum = th.cat([pred_batch_idxs.new_zeros(1), pred_counts.cumsum(0)]).tolist()

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

        batch_true_mzs, true_valid = pad_flat_by_batch(
            bl_true_mzs, bl_true_batch_idxs, bl_batch_size, 0.0
        )
        batch_pred_mzs, pred_valid = pad_flat_by_batch(
            bl_pred_mzs, bl_pred_batch_idxs, bl_batch_size, 0.0
        )
        batch_true_ints, _ = pad_flat_by_batch(
            bl_true_ints, bl_true_batch_idxs, bl_batch_size, 0.0
        )
        batch_pred_ints, _ = pad_flat_by_batch(
            bl_pred_ints, bl_pred_batch_idxs, bl_batch_size, 0.0
        )

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
        # predicted mass. Clamping the normalizer keeps those rows finite instead of
        # 0/0, because a NaN branch of th.where would still poison the backward pass.
        true_total = aligned_true.sum(dim=1, keepdim=True)
        pred_total = aligned_pred.sum(dim=1, keepdim=True)
        distribution_tiny = th.finfo(aligned_pred.dtype).tiny
        true_distribution = aligned_true / true_total.clamp_min(distribution_tiny)
        pred_distribution = aligned_pred / pred_total.clamp_min(distribution_tiny)
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
        log_min: Minimum log value to avoid log(0).

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
        # clamp(min=EPS) avoids division-by-zero; when all probs are ~0, sum≈0/EPS=0 anyway.
        b_true_probs = th.exp(b_true_logprobs)
        b_true_probs = b_true_probs / b_true_probs.sum().clamp(min=EPS)

        b_pred_probs = th.exp(b_pred_logprobs)
        b_pred_probs = b_pred_probs / b_pred_probs.sum().clamp(min=EPS)

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
