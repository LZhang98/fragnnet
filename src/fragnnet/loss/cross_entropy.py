"""Sparse cross-entropy between observed and predicted spectra, with its mixture and
peak-matching variants, and the ``loss_type`` dispatcher ``get_sparse_cross_entropy_fn``.
"""

import logging
import math
from collections.abc import Callable

import torch as th

from fragnnet.loss.padding import build_loss_chunks, build_padded_batch_layout, pad_flat_with_layout
from fragnnet.utils.misc_utils import LOG_ZERO
from fragnnet.utils.spec_utils import calculate_match_mzs

logger = logging.getLogger(__name__)


def conditional_in_support_cross_entropy(
    coverage_weighted_ce: th.Tensor,
    true_oos_logprobs: th.Tensor,
) -> th.Tensor:
    """Remove target-coverage weighting from an in-support cross-entropy.

    Sparse CE sums ``-t_i log(q_i)`` only over observed peaks represented by the
    predicted support. Its magnitude and gradient are therefore multiplied by the
    represented target mass. This function returns the conditional CE over that
    represented mass. Spectra with no represented target mass return zero because
    they contain no in-support intensity ratios to supervise.

    Args:
        coverage_weighted_ce: In-support CE per spectrum, shape ``(batch_size,)``.
        true_oos_logprobs: Log target mass outside predicted support per spectrum,
            shape ``(batch_size,)``.

    Returns:
        Conditional in-support CE per spectrum, shape ``(batch_size,)``.
    """
    if coverage_weighted_ce.shape != true_oos_logprobs.shape:
        raise ValueError("in-support CE and target OOS mass must have matching shapes")
    represented_mass = -th.expm1(true_oos_logprobs).clamp(max=0.0)
    has_support = represented_mass > 0.0
    safe_mass = represented_mass.clamp_min(th.finfo(coverage_weighted_ce.dtype).tiny)
    return th.where(
        has_support,
        coverage_weighted_ce / safe_mass,
        th.zeros_like(coverage_weighted_ce),
    )


def mog_log_prob(
    samples: th.Tensor,
    means: th.Tensor,
    variances: th.Tensor,
    log_weights: th.Tensor,
    log_trunc_factors: th.Tensor,
    match_mask: th.Tensor | None = None,
) -> th.Tensor:
    """Mixture of Gaussian
        We are modeling each peaks are mixture of guassian for given set of fragments
    Args:
        samples (_type_): b_true_mz,
        means (_type_): b_pred_mzs
        variances (_type_): b_variances
        log_weights (_type_): b_pred_logprobs
        log_trunc_factors (_type_): b_log_trunc_factors
        match_mask: Optional [T, P] boolean mask. When given, only candidates inside a
            true peak's tolerance window contribute to its mixture, matching
            ``sparse_cross_entropy_vec``. ``None`` leaves the mixture untruncated.

    Returns:
        _type_: _description_
    """

    samples = samples.reshape(-1, 1)
    means = means.reshape(1, -1)
    variances = variances.reshape(1, -1)
    log_weights = log_weights.reshape(1, -1)
    log_trunc_factors = log_trunc_factors.reshape(1, -1)
    # [B,K]
    normal_log_probs = -0.5 * (samples - means) ** 2 / variances - 0.5 * th.log(
        2 * th.pi * variances
    )
    normal_log_probs = normal_log_probs - log_trunc_factors
    if match_mask is not None:
        normal_log_probs = normal_log_probs + LOG_ZERO(normal_log_probs.dtype) * (~match_mask).to(
            normal_log_probs.dtype
        )
    # [B]
    log_probs = th.logsumexp(normal_log_probs + log_weights, dim=1)
    return log_probs


def mog_ce_fn(
    b_true_mzs: th.Tensor,
    b_pred_mzs: th.Tensor,
    b_pred_logprobs: th.Tensor,
    relative: bool,
    tolerance: float,
    tolerance_min_mz: float,
    tolerance_multiple: float,
    gaussian_renormalize: bool,
    **kwargs,
) -> th.Tensor:
    if relative:
        b_stds = th.clamp(b_pred_mzs, min=tolerance_min_mz) * tolerance
    else:
        b_stds = tolerance * th.ones_like(b_pred_mzs)
    b_vars = b_stds**2
    if gaussian_renormalize:
        assert tolerance_multiple > 0, tolerance_multiple
        # log_trunc_factor = LOG_TRUNC_FACTOR.get(tolerance_multiple,1.)
        # b_log_trunc_factors = log_trunc_factor*th.ones_like(b_pred_mzs)
        sigma = math.sqrt(2)
        trunc_factor = math.erf(tolerance_multiple / sigma)
        b_log_trunc_factors = th.log(
            th.tensor(trunc_factor, device=b_pred_mzs.device) * th.ones_like(b_pred_mzs)
        )
    else:
        b_log_trunc_factors = th.zeros_like(b_pred_mzs)
    # A negative multiple means "no truncation": every true peak counts as in-support and
    # the mixture spans all candidates. A positive multiple defines a real tolerance window,
    # so the mixture must be truncated to it -- otherwise ``gaussian_renormalize`` corrects
    # for a truncation that never happened, and this path disagrees with the vectorized one.
    if tolerance_multiple > 0:
        b_match_mask = calculate_match_mzs(
            b_true_mzs,
            b_pred_mzs,
            tolerance=tolerance * tolerance_multiple,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        )
        b_match_mask = th.as_tensor(b_match_mask, device=b_pred_logprobs.device)
    else:
        b_match_mask = None
    b_logprobs = mog_log_prob(
        b_true_mzs,
        b_pred_mzs,
        b_vars,
        b_pred_logprobs,
        b_log_trunc_factors,
        match_mask=b_match_mask,
    )

    return b_logprobs


def pm_ce_fn(
    b_pred_mzs: th.Tensor,
    b_true_mzs: th.Tensor,
    b_pred_logprobs: th.Tensor,
    relative: bool,
    tolerance: float,
    tolerance_multiple: float,
    tolerance_min_mz: float,
    **kwargs,
) -> th.Tensor:
    """
    Compute per-true-peak log-probability via peak-marginal aggregation.

    For each true peak in `b_true_mzs`, this function aggregates the
    log-probability mass assigned by the predicted peaks `b_pred_mzs`
    within an m/z tolerance window. Concretely, it performs a
    log-sum-exp over the predicted log-probabilities `b_pred_logprobs`
    that match each true peak, yielding a length-`T` tensor (one value
    per true peak).

    Args:
        b_pred_mzs: [P] Predicted m/z values for the current spectrum.
        b_true_mzs: [T] Ground-truth m/z values for the current spectrum.
        b_pred_logprobs: [P] Predicted log-probabilities (intensities) per predicted peak.
        relative: If True, the tolerance is relative to m/z; otherwise, absolute.
        tolerance: Base tolerance value (absolute or the factor used for relative).
        tolerance_multiple: Multiplicative factor applied to `tolerance` when matching.
        tolerance_min_mz: Minimum m/z used when applying relative tolerance.

    Returns:
        [T] Log-probabilities per true peak obtained by aggregating matching
        predicted peak log-probabilities via log-sum-exp.
    """

    # Compute boolean match mask [T, P] indicating which predicted peaks
    # fall within the m/z tolerance window of each true peak.
    b_match_mask = calculate_match_mzs(
        b_true_mzs,
        b_pred_mzs,
        tolerance=tolerance * tolerance_multiple,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
    )
    b_match_mask = th.as_tensor(b_match_mask, device=b_pred_logprobs.device)
    # Aggregate matched predicted log-probs for each true peak via log-sum-exp.
    # Unmatched (false mask) entries are suppressed by adding a large negative
    # value (`LOG_ZERO`) so they do not contribute to the sum.
    b_logprobs = th.logsumexp(
        b_pred_logprobs.reshape(1, -1) * b_match_mask.to(b_pred_logprobs.dtype)
        + LOG_ZERO(b_pred_logprobs.dtype) * (~b_match_mask).to(b_pred_logprobs.dtype),
        dim=1,
    )
    return b_logprobs


def sparse_cross_entropy_seq(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    pred_oos_logprobs: th.Tensor,
    ce_fn: Callable,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    tolerance_multiple: float,
    gaussian_renormalize: bool,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """compute sparse cross entropy in squence

    Args:
        true_mzs (_type_): groundtruth mzs
        true_logprobs (_type_): groundtruth log prob of each mz aka intensities
        true_batch_idxs (_type_): spec idx signed to each groundtruth
        pred_mzs (_type_): predicted mzs
        pred_logprobs (_type_): predicted log prob of each mz aka intensities
        pred_batch_idxs (_type_): pec idx signed to each prediction
        ce_fn (_type_, optional): _description_. Defaults to mog_log_prob.
        tolerance (_type_, optional): _description_. Defaults to 1e-3.
        relative (bool, optional): _description_. Defaults to False.
        oos_tolerance_multiple (int, optional): _description_.
        gaussian_renormalize (bool, optional): _description_.
        pm_tolerance_multiple (int, optional): _description_.

    Returns:
        tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]: ios_ces, oos_ces, true_oos_logprobs, true_oos_entropies
    """

    mask_value = LOG_ZERO(true_logprobs.dtype)
    batch_size = pred_oos_logprobs.shape[0]
    # ces = th.zeros([batch_size],dtype=true_logprobs.dtype,device=true_logprobs.device)
    ios_ces = th.zeros(batch_size, dtype=true_logprobs.dtype, device=true_logprobs.device)
    oos_ces = th.zeros(batch_size, dtype=true_logprobs.dtype, device=true_logprobs.device)
    true_oos_logprobs = th.zeros(batch_size, dtype=true_logprobs.dtype, device=true_logprobs.device)
    true_oos_entropies = th.zeros_like(true_oos_logprobs)
    for batch_idx in range(batch_size):
        # get groundtruth belongs to this batch idx
        b_true_mask = true_batch_idxs == batch_idx
        # get predictions belongs to this batch idx
        b_pred_mask = pred_batch_idxs == batch_idx
        # get mzs and logprobs
        b_true_logprobs = true_logprobs[b_true_mask]
        b_true_mzs = true_mzs[b_true_mask]
        b_pred_logprobs = pred_logprobs[b_pred_mask]
        b_pred_mzs = pred_mzs[b_pred_mask]
        # per-sample tolerance (falls back to global scalars when not provided)
        b_tolerance = tol_per_sample[batch_idx].item() if tol_per_sample is not None else tolerance
        b_min_mz = (
            min_mz_per_sample[batch_idx].item()
            if min_mz_per_sample is not None
            else tolerance_min_mz
        )
        # get matches
        b_match_mask = calculate_match_mzs(
            b_true_mzs,
            b_pred_mzs,
            tolerance=tolerance_multiple * b_tolerance,
            relative=relative,
            tolerance_min_mz=b_min_mz,
        )
        b_match_mask = th.as_tensor(b_match_mask, device=b_true_mzs.device)
        if tolerance_multiple == -1:
            b_true_match_mask = th.ones_like(b_true_mzs, dtype=th.bool)
        else:
            assert tolerance_multiple > 0, tolerance_multiple
            b_true_match_mask = th.any(b_match_mask, dim=1)
        if not th.any(b_true_match_mask):
            logger.warning("Everything is OOS!")
        # calculate ios logprobs
        b_ios_logprobs = ce_fn(
            b_true_mzs=b_true_mzs,
            b_pred_mzs=b_pred_mzs,
            b_pred_logprobs=b_pred_logprobs,
            relative=relative,
            tolerance=b_tolerance,
            tolerance_min_mz=b_min_mz,
            tolerance_multiple=tolerance_multiple,
            gaussian_renormalize=gaussian_renormalize,
        )
        b_ios_ce = th.sum(
            b_true_match_mask.to(b_true_logprobs.dtype) * -th.exp(b_true_logprobs) * b_ios_logprobs
        )
        # calculate oos logprobs
        b_true_oos_logprobs = th.logsumexp(
            mask_value * b_true_match_mask.to(b_true_logprobs.dtype)
            + b_true_logprobs * (~b_true_match_mask).to(b_true_logprobs.dtype),
            dim=0,
        )
        b_true_oos_entropy = -th.sum(
            th.exp(b_true_logprobs[~b_true_match_mask] - b_true_oos_logprobs)
            * (b_true_logprobs[~b_true_match_mask] - b_true_oos_logprobs),
            dim=0,
        )
        b_pred_oos_logprobs = pred_oos_logprobs[batch_idx]
        b_oos_ce = -th.exp(b_true_oos_logprobs) * b_pred_oos_logprobs
        # combine
        ios_ces[batch_idx] = b_ios_ce
        oos_ces[batch_idx] = b_oos_ce
        true_oos_logprobs[batch_idx] = b_true_oos_logprobs
        true_oos_entropies[batch_idx] = b_true_oos_entropy
    return ios_ces, oos_ces, true_oos_logprobs, true_oos_entropies


def sparse_cross_entropy_vec(
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    pred_oos_logprobs: th.Tensor,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    oos_tolerance_multiple: float,
    gaussian_renormalize: bool,
    loss_batch_size: int,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    dist: str = "gaussian",
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    Vectorized computation of sparse cross-entropy loss for batched spectra.

    Each loss chunk is padded per spectrum before peak matching. Chunk boundaries
    ensure the resulting ``[chunk, max_true_peaks, max_pred_peaks]`` pairwise tensors
    are no larger than the Cartesian product of all flattened peaks in that chunk.

    Args:
        true_mzs: Ground truth m/z values.
        true_logprobs: Ground truth log-probabilities (intensities).
        true_batch_idxs: Batch indices for ground truth peaks.
        pred_mzs: Predicted m/z values.
        pred_logprobs: Predicted log-probabilities (intensities).
        pred_batch_idxs: Batch indices for predicted peaks.
        pred_oos_logprobs: Predicted log-probabilities for out-of-sample (OOS) peaks.
        tolerance: m/z matching tolerance.
        relative: Whether tolerance is relative to m/z.
        tolerance_min_mz: Minimum m/z for relative tolerance.
        oos_tolerance_multiple: Multiplier for OOS tolerance.
        gaussian_renormalize: Whether to renormalize Gaussian.
        loss_batch_size: Batch size for loss computation.
        dist: Output distribution. ``"gaussian"`` weights each matched candidate by a
            Gaussian m/z density; ``"peak_marginal"`` aggregates matched candidate
            log-probabilities with a hard tolerance window and no density term.

    Returns:
        ios_ce: In-sample cross-entropy per batch.
        oos_ce: Out-of-sample cross-entropy per batch.
        0., 1.: Placeholders for compatibility.
    """

    # 1. Set up types and constants.
    float_dtype = true_logprobs.dtype
    device = true_logprobs.device
    mask_value = LOG_ZERO(float_dtype)
    batch_size = pred_oos_logprobs.shape[0]

    # 2. Build spectrum-aligned slices for memory-bounded loss chunks.
    # Compute cumulative sum of counts for batch slicing.
    # .tolist() transfers batch_size+1 ints once, turning loop-body indexing into Python-int
    # slicing (no GPU→CPU sync per iteration). th.bincount is O(N) vs th.unique's O(N log N).
    true_batch_counts = th.bincount(true_batch_idxs, minlength=batch_size)
    pred_batch_counts = th.bincount(pred_batch_idxs, minlength=batch_size)
    packed_batch_cumsums = th.cat(
        (
            true_batch_idxs.new_zeros(1),
            true_batch_counts.cumsum(0),
            pred_batch_idxs.new_zeros(1),
            pred_batch_counts.cumsum(0),
        )
    ).tolist()
    cumsum_size = batch_size + 1
    true_batch_cumsum: list[int] = packed_batch_cumsums[:cumsum_size]
    pred_batch_cumsum: list[int] = packed_batch_cumsums[cumsum_size:]
    loss_chunks = build_loss_chunks(
        true_batch_cumsum,
        pred_batch_cumsum,
        loss_batch_size,
    )

    # Precompute loop-invariant constants as Python scalars (avoids per-iteration tensor creation).
    _LOG_2PI = math.log(2 * math.pi)
    if dist not in ("gaussian", "peak_marginal"):
        raise ValueError(
            f"Distribution '{dist}' not implemented. "
            "Supported distributions: 'gaussian', 'peak_marginal'."
        )
    use_gaussian = dist == "gaussian"
    if gaussian_renormalize and use_gaussian:
        if oos_tolerance_multiple <= 0:
            raise ValueError(f"oos_tolerance_multiple must be > 0, got {oos_tolerance_multiple}")
        _log_trunc_factor: float = math.log(math.erf(oos_tolerance_multiple / math.sqrt(2)))
    else:
        _log_trunc_factor = 0.0

    # 3. Initialize outputs and per-peak tolerances.
    ios_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    oos_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    true_oos_logprobs = th.zeros(batch_size, dtype=float_dtype, device=device)

    # Expand per-sample tolerances to per-peak if provided
    if tol_per_sample is not None:
        tol_per_true_peak = tol_per_sample[true_batch_idxs].to(float_dtype)
        min_mz_per_true_peak = min_mz_per_sample[true_batch_idxs].to(float_dtype)
        tol_per_pred_peak = tol_per_sample[pred_batch_idxs].to(float_dtype)
        min_mz_per_pred_peak = min_mz_per_sample[pred_batch_idxs].to(float_dtype)
    else:
        tol_per_true_peak = None
        min_mz_per_true_peak = None
        tol_per_pred_peak = None
        min_mz_per_pred_peak = None

    # 4. Evaluate padded, block-diagonal loss chunks.
    for bl_lower, bl_upper in loss_chunks:
        # Compute batch slice indices (Python ints from .tolist() — no GPU sync)
        bl_true_lower = true_batch_cumsum[bl_lower]
        bl_true_upper = true_batch_cumsum[bl_upper]
        bl_pred_lower = pred_batch_cumsum[bl_lower]
        bl_pred_upper = pred_batch_cumsum[bl_upper]
        bl_batch_size = bl_upper - bl_lower

        # Slice batch data and adjust indices to local batch
        bl_true_batch_idxs = true_batch_idxs[bl_true_lower:bl_true_upper] - bl_lower
        bl_pred_batch_idxs = pred_batch_idxs[bl_pred_lower:bl_pred_upper] - bl_lower
        bl_true_logprobs = true_logprobs[bl_true_lower:bl_true_upper]
        bl_pred_logprobs = pred_logprobs[bl_pred_lower:bl_pred_upper]
        bl_true_mzs = true_mzs[bl_true_lower:bl_true_upper]
        bl_pred_mzs = pred_mzs[bl_pred_lower:bl_pred_upper]
        bl_pred_oos_logprobs = pred_oos_logprobs[bl_lower:bl_upper]

        # Slice per-peak tolerance arrays for this loss batch
        bl_tol_per_true = (
            tol_per_true_peak[bl_true_lower:bl_true_upper]
            if tol_per_true_peak is not None
            else None
        )
        bl_min_mz_per_true = (
            min_mz_per_true_peak[bl_true_lower:bl_true_upper]
            if min_mz_per_true_peak is not None
            else None
        )
        bl_tol_per_pred = (
            tol_per_pred_peak[bl_pred_lower:bl_pred_upper]
            if tol_per_pred_peak is not None
            else None
        )
        bl_min_mz_per_pred = (
            min_mz_per_pred_peak[bl_pred_lower:bl_pred_upper]
            if min_mz_per_pred_peak is not None
            else None
        )

        # Pad spectra independently so pairwise tensors have shape
        # [batch, max_true_peaks, max_pred_peaks], without cross-spectrum pairs.
        max_true_peaks = max(
            true_batch_cumsum[index + 1] - true_batch_cumsum[index]
            for index in range(bl_lower, bl_upper)
        )
        max_pred_peaks = max(
            pred_batch_cumsum[index + 1] - pred_batch_cumsum[index]
            for index in range(bl_lower, bl_upper)
        )
        true_sort_idx, true_padded_indices, bl_true_valid, max_true_peaks = (
            build_padded_batch_layout(
                bl_true_batch_idxs,
                bl_batch_size,
                assume_sorted=True,
                counts=true_batch_counts[bl_lower:bl_upper],
                max_count=max_true_peaks,
            )
        )
        pred_sort_idx, pred_padded_indices, bl_pred_valid, max_pred_peaks = (
            build_padded_batch_layout(
                bl_pred_batch_idxs,
                bl_batch_size,
                assume_sorted=True,
                counts=pred_batch_counts[bl_lower:bl_upper],
                max_count=max_pred_peaks,
            )
        )
        true_layout = (
            true_sort_idx,
            true_padded_indices,
            bl_batch_size,
            max_true_peaks,
        )
        pred_layout = (
            pred_sort_idx,
            pred_padded_indices,
            bl_batch_size,
            max_pred_peaks,
        )
        bl_true_mzs_padded = pad_flat_with_layout(bl_true_mzs, *true_layout, pad_value=0.0)
        bl_pred_mzs_padded = pad_flat_with_layout(bl_pred_mzs, *pred_layout, pad_value=0.0)
        bl_true_logprobs_padded = pad_flat_with_layout(
            bl_true_logprobs, *true_layout, pad_value=0.0
        )
        bl_pred_logprobs_padded = pad_flat_with_layout(
            bl_pred_logprobs, *pred_layout, pad_value=0.0
        )

        bl_tol_per_true_padded = None
        bl_min_mz_per_true_padded = None
        bl_tol_per_pred_padded = None
        bl_min_mz_per_pred_padded = None
        if bl_tol_per_true is not None:
            bl_tol_per_true_padded = pad_flat_with_layout(
                bl_tol_per_true,
                *true_layout,
                pad_value=tolerance,
            )
            bl_min_mz_per_true_padded = pad_flat_with_layout(
                bl_min_mz_per_true,
                *true_layout,
                pad_value=tolerance_min_mz,
            )
            bl_tol_per_pred_padded = pad_flat_with_layout(
                bl_tol_per_pred,
                *pred_layout,
                pad_value=tolerance,
            )
            bl_min_mz_per_pred_padded = pad_flat_with_layout(
                bl_min_mz_per_pred,
                *pred_layout,
                pad_value=tolerance_min_mz,
            )

        # 5. Match peaks and compute Gaussian-mixture probabilities within each spectrum.
        bl_match_mask = calculate_match_mzs(
            bl_true_mzs_padded,
            bl_pred_mzs_padded,
            tolerance=oos_tolerance_multiple * tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            tol_per_true=(
                oos_tolerance_multiple * bl_tol_per_true_padded
                if bl_tol_per_true_padded is not None
                else None
            ),
            min_mz_per_true=bl_min_mz_per_true_padded,
        )
        bl_match_mask = th.as_tensor(bl_match_mask, device=device)
        bl_valid_pairs = bl_true_valid.unsqueeze(2) & bl_pred_valid.unsqueeze(1)
        bl_both_mask = bl_valid_pairs & bl_match_mask
        del bl_valid_pairs, bl_match_mask

        bl_true_both_mask = th.any(bl_both_mask, dim=2)

        if use_gaussian:
            if relative:
                if bl_tol_per_pred_padded is not None:
                    bl_stds = (
                        th.maximum(bl_pred_mzs_padded, bl_min_mz_per_pred_padded)
                        * bl_tol_per_pred_padded
                    )
                else:
                    bl_stds = th.clamp(bl_pred_mzs_padded, min=tolerance_min_mz) * tolerance
            else:
                bl_stds = tolerance * th.ones_like(bl_pred_mzs_padded)
            bl_vars = bl_stds**2

            bl_ios_log_probs = (
                -0.5
                * (bl_true_mzs_padded.unsqueeze(2) - bl_pred_mzs_padded.unsqueeze(1)) ** 2
                / bl_vars.unsqueeze(1)
                - 0.5 * (_LOG_2PI + th.log(bl_vars.unsqueeze(1)))
                + bl_pred_logprobs_padded.unsqueeze(1)
                - _log_trunc_factor
            )
        else:
            # Peak-marginal: every candidate inside the tolerance window contributes its
            # full log-probability, with no m/z density weighting.
            bl_ios_log_probs = bl_pred_logprobs_padded.unsqueeze(1).expand(
                -1, bl_true_mzs_padded.shape[1], -1
            )
        bl_ios_log_probs = bl_ios_log_probs + (~bl_both_mask).to(float_dtype) * mask_value
        bl_ios_log_probs = th.logsumexp(bl_ios_log_probs, dim=2)

        # 6. Reduce padded peak losses back to one value per spectrum.
        bl_ios_ce = th.sum(
            th.where(
                bl_true_both_mask,
                -th.exp(bl_true_logprobs_padded) * bl_ios_log_probs,
                th.zeros_like(bl_ios_log_probs),
            ),
            dim=1,
        )
        ios_ce[bl_lower:bl_upper] = bl_ios_ce

        bl_unmatched_mask = bl_true_valid & ~bl_true_both_mask
        bl_true_oos_logprobs = th.logsumexp(
            bl_true_logprobs_padded + (~bl_unmatched_mask).to(float_dtype) * mask_value,
            dim=1,
        )
        bl_true_oos_logprobs = th.where(
            th.any(bl_unmatched_mask, dim=1),
            bl_true_oos_logprobs,
            th.full_like(bl_true_oos_logprobs, mask_value),
        )
        # Compute out-of-sample cross-entropy
        bl_oos_ce = -th.exp(bl_true_oos_logprobs) * bl_pred_oos_logprobs
        oos_ce[bl_lower:bl_upper] = bl_oos_ce
        true_oos_logprobs[bl_lower:bl_upper] = bl_true_oos_logprobs

    return ios_ce, oos_ce, true_oos_logprobs, th.zeros_like(ios_ce)


def get_sparse_cross_entropy_fn(
    dist: str,
    vectorized: bool,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    oos_tolerance_multiple: float,
    gaussian_renormalize: bool,
    pm_tolerance_multiple: float,
    loss_batch_size: int,
) -> Callable:
    if dist == "gaussian":
        ce_fn = mog_ce_fn
    elif dist == "peak_marginal":
        ce_fn = pm_ce_fn
    else:
        msg = (
            f"Distribution '{dist}' not implemented. "
            "Supported distributions: 'gaussian', 'peak_marginal'."
        )
        raise NotImplementedError(msg)

    if vectorized:
        # The window multiple is distribution-specific, matching the sequential path.
        vec_tolerance_multiple = (
            oos_tolerance_multiple if dist == "gaussian" else pm_tolerance_multiple
        )

        def sce_fn(
            *args: th.Tensor,
            tol_per_sample: th.Tensor | None = None,
            min_mz_per_sample: th.Tensor | None = None,
        ) -> th.Tensor:
            return sparse_cross_entropy_vec(
                *args,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
                oos_tolerance_multiple=vec_tolerance_multiple,
                gaussian_renormalize=gaussian_renormalize,
                loss_batch_size=loss_batch_size,
                tol_per_sample=tol_per_sample,
                min_mz_per_sample=min_mz_per_sample,
                dist=dist,
            )

    else:
        if dist == "gaussian":
            tolerance_multiple = oos_tolerance_multiple
        else:
            tolerance_multiple = pm_tolerance_multiple

        def sce_fn(
            *args: th.Tensor,
            tol_per_sample: th.Tensor | None = None,
            min_mz_per_sample: th.Tensor | None = None,
        ) -> th.Tensor:
            return sparse_cross_entropy_seq(
                *args,
                ce_fn=ce_fn,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
                tolerance_multiple=tolerance_multiple,
                gaussian_renormalize=gaussian_renormalize,
                tol_per_sample=tol_per_sample,
                min_mz_per_sample=min_mz_per_sample,
            )

    return sce_fn
