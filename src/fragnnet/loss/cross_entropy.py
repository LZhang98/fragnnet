"""Entropy and cross-entropy of sparse spectra: the entropy of a predicted spectrum and its
conditional variant, the sparse cross-entropy between observed and predicted spectra with its
mixture and peak-matching variants, and the ``loss_type`` dispatcher
``get_sparse_cross_entropy_fn``.
"""

import logging
import math
from collections.abc import Callable

import torch as th

from fragnnet.loss.padding import (
    build_loss_chunks,
    build_padded_batch_layout,
    pad_flat_with_layout,
    paired_batch_offsets,
)
from fragnnet.utils.misc_utils import EPS, LOG_ZERO
from fragnnet.utils.scatter_utils import safelog, scatter_logsumexp, scatter_reduce
from fragnnet.utils.spec_similarity import calculate_match_mzs

logger = logging.getLogger(__name__)


def _plogp(logprobs: th.Tensor) -> th.Tensor:
    # Replace log(0) before multiplying: masking a NaN product poisons backward.
    finite_logs = th.where(th.isneginf(logprobs), th.zeros_like(logprobs), logprobs)
    return logprobs.exp() * finite_logs


def _rescale_entropy(
    entropy: th.Tensor, max_entropy: th.Tensor, label: str
) -> tuple[th.Tensor, th.Tensor]:
    max_entropy = th.clamp(max_entropy, min=EPS)
    entropy = th.clamp(entropy, min=th.zeros_like(max_entropy), max=max_entropy)
    norm_entropy = entropy / max_entropy
    if th.any(norm_entropy < 0.0) or th.any(norm_entropy > 1.0):
        raise ValueError(f"Normalized {label} out of bounds [0, 1]")
    return entropy, norm_entropy


def sparse_entropy(
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    oos_logprobs: th.Tensor | None = None,
    renormalize: bool = False,
    support_size_delta: float = 0.0,
) -> tuple[th.Tensor, th.Tensor]:
    if renormalize:
        logpartition = scatter_logsumexp(logprobs, batch_idxs)
        if oos_logprobs is not None:
            logpartition = th.logsumexp(th.stack([logpartition, oos_logprobs], dim=1), dim=1)
            oos_logprobs = oos_logprobs - logpartition
        logprobs = logprobs - th.gather(input=logpartition, index=batch_idxs, dim=0)
    logprobs = th.clamp(logprobs, max=0.0)
    plogp = _plogp(logprobs)
    batch_size = int(th.max(batch_idxs).item() + 1)
    entropy = -scatter_reduce(
        src=plogp,
        index=batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    if th.min(entropy) < -0.001:
        raise ValueError(f"Negative entropy: {th.min(entropy)}")

    support_size = scatter_reduce(
        src=th.ones_like(batch_idxs, dtype=logprobs.dtype),
        index=batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    support_size += support_size_delta
    if oos_logprobs is not None:
        assert oos_logprobs.shape[0] == batch_size, (oos_logprobs.shape, batch_size)
        entropy = entropy - _plogp(oos_logprobs)
        support_size = support_size + 1.0
    max_entropy = safelog(support_size)
    return _rescale_entropy(entropy, max_entropy, "entropy")


def sparse_conditional_entropy(
    marginal_logprobs: th.Tensor,
    marginal_batch_idxs: th.Tensor,
    conditional_logprobs: th.Tensor,
    conditional_idxs: th.Tensor,
    conditional_batch_idxs: th.Tensor,
    conditional_support_size_delta: float = 0.0,
) -> tuple[th.Tensor, th.Tensor]:
    batch_size = int(th.max(conditional_batch_idxs).item() + 1)
    marginal_support_size = marginal_logprobs.shape[0]
    marginal_logprobs = th.clamp(marginal_logprobs, max=0.0)
    conditional_logprobs = th.clamp(conditional_logprobs, max=0.0)
    marginal_probs = th.exp(marginal_logprobs)
    conditional_plogp = _plogp(conditional_logprobs)
    conditional_entropy = -scatter_reduce(
        conditional_plogp, index=conditional_idxs, reduce="sum", dim_size=marginal_support_size
    )
    entropy = scatter_reduce(
        marginal_probs * conditional_entropy,
        index=marginal_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    if th.min(entropy) < -0.001:
        raise ValueError(f"Negative conditional entropy: {th.min(entropy)}")

    conditional_support_size = scatter_reduce(
        th.ones_like(conditional_logprobs, dtype=marginal_logprobs.dtype),
        index=conditional_idxs,
        reduce="sum",
        dim_size=marginal_support_size,
    )
    conditional_support_size += conditional_support_size_delta
    max_entropy = scatter_reduce(
        marginal_probs * th.clamp(safelog(conditional_support_size), min=0.0),
        index=marginal_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    return _rescale_entropy(entropy, max_entropy, "conditional entropy")


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


def gaussian_ce(
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


def laplace_ce(
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
    """Per-true-peak log density under a mixture of Laplace kernels.

    The kernel standard deviation equals the Gaussian one, ``sigma = max(mz, min_mz) * tol``
    (relative) or ``tol`` (absolute), so the Laplace scale is ``b = sigma / sqrt(2)`` and

        log p(x | mu) = -|x - mu| / b - log(2 b).

    Truncating at ``k * sigma`` keeps mass ``1 - exp(-sqrt(2) k)``; ``gaussian_renormalize``
    divides it out, as for the Gaussian.
    """
    if relative:
        b_scales = th.clamp(b_pred_mzs, min=tolerance_min_mz) * tolerance / math.sqrt(2)
    else:
        b_scales = tolerance / math.sqrt(2) * th.ones_like(b_pred_mzs)
    log_trunc_factor = 0.0
    if gaussian_renormalize:
        assert tolerance_multiple > 0, tolerance_multiple
        log_trunc_factor = math.log(-math.expm1(-math.sqrt(2) * tolerance_multiple))
    # [T, P]
    b_log_probs = (
        -th.abs(b_true_mzs.reshape(-1, 1) - b_pred_mzs.reshape(1, -1)) / b_scales.reshape(1, -1)
        - th.log(2 * b_scales.reshape(1, -1))
        + b_pred_logprobs.reshape(1, -1)
        - log_trunc_factor
    )
    # Same windowing as ``gaussian_ce``: a positive multiple truncates, -1 does not.
    if tolerance_multiple > 0:
        b_match_mask = calculate_match_mzs(
            b_true_mzs,
            b_pred_mzs,
            tolerance=tolerance * tolerance_multiple,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
        )
        b_match_mask = th.as_tensor(b_match_mask, device=b_pred_logprobs.device)
        b_log_probs = b_log_probs + LOG_ZERO(b_log_probs.dtype) * (~b_match_mask).to(
            b_log_probs.dtype
        )
    return th.logsumexp(b_log_probs, dim=1)


def peak_marginal_ce(
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
    """Score each spectrum independently, including conditional observed OOS entropy.

    Batch indices may be unsorted. Empty target subsets contribute zero CE;
    empty predictions with observed mass require truncated matching. Tolerances
    are scalar defaults overridden by each supplied per-spectrum vector.

    Returns:
        In-support CE, OOS CE, observed OOS log mass, and conditional OOS entropy,
        each a vector with one entry per predicted OOS log probability.
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
            b_true_logprobs.masked_fill(
                b_true_match_mask | th.isneginf(b_true_logprobs), mask_value
            ),
            dim=0,
        )
        b_true_oos_entropy = -_plogp(
            b_true_logprobs[~b_true_match_mask] - b_true_oos_logprobs
        ).sum()
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

    Inputs must be sorted by batch index; empty rows are retained.
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
        oos_tolerance_multiple: Multiplier for OOS tolerance. For Gaussian mixtures,
            ``-1`` disables truncation and requires ``gaussian_renormalize=False``.
        gaussian_renormalize: Whether to renormalize Gaussian.
        loss_batch_size: Batch size for loss computation.
        dist: Output distribution. ``"gaussian"`` weights each matched candidate by a
            Gaussian m/z density; ``"laplace"`` by a Laplace density with the same standard
            deviation (scale ``sigma / sqrt(2)``); ``"peak_marginal"`` aggregates matched candidate
            log-probabilities with a hard tolerance window and no density term.

    Returns:
        ios_ce: In-sample cross-entropy per batch.
        oos_ce: Out-of-sample cross-entropy per batch.
        true_oos_logprobs: Observed mass outside the prediction windows, in log space.
        true_oos_entropies: Zeros; conditional OOS entropy is only computed by the sequential path.
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
    true_batch_counts, pred_batch_counts, true_batch_cumsum, pred_batch_cumsum = (
        paired_batch_offsets(true_batch_idxs, pred_batch_idxs, batch_size)
    )
    loss_chunks = build_loss_chunks(
        true_batch_cumsum,
        pred_batch_cumsum,
        loss_batch_size,
    )

    # Precompute loop-invariant constants as Python scalars (avoids per-iteration tensor creation).
    _LOG_2PI = math.log(2 * math.pi)
    if dist not in ("gaussian", "laplace", "peak_marginal"):
        raise ValueError(
            f"Distribution '{dist}' not implemented. "
            "Supported distributions: 'gaussian', 'laplace', 'peak_marginal'."
        )
    use_density = dist in ("gaussian", "laplace")
    if gaussian_renormalize and use_density:
        if oos_tolerance_multiple <= 0:
            raise ValueError(f"oos_tolerance_multiple must be > 0, got {oos_tolerance_multiple}")
        if dist == "gaussian":
            _log_trunc_factor: float = math.log(math.erf(oos_tolerance_multiple / math.sqrt(2)))
        else:
            _log_trunc_factor = math.log(-math.expm1(-math.sqrt(2) * oos_tolerance_multiple))
    else:
        _log_trunc_factor = 0.0

    # 3. Initialize per-spectrum outputs.
    ios_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    oos_ce = th.zeros(batch_size, dtype=float_dtype, device=device)
    true_oos_logprobs = th.zeros(batch_size, dtype=float_dtype, device=device)

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
        if tol_per_sample is not None:
            row_tol = tol_per_sample[bl_lower:bl_upper].to(float_dtype)[:, None]
            row_min_mz = min_mz_per_sample[bl_lower:bl_upper].to(float_dtype)[:, None]
            # Preserve padding fill values as well as the real-peak tolerances.
            bl_tol_per_true_padded = th.where(bl_true_valid, row_tol, tolerance)
            bl_tol_per_pred_padded = th.where(bl_pred_valid, row_tol, tolerance)
            bl_min_mz_per_true_padded = th.where(bl_true_valid, row_min_mz, tolerance_min_mz)
            bl_min_mz_per_pred_padded = th.where(bl_pred_valid, row_min_mz, tolerance_min_mz)

        # 5. Match peaks and compute Gaussian-mixture probabilities within each spectrum.
        bl_valid_pairs = bl_true_valid.unsqueeze(2) & bl_pred_valid.unsqueeze(1)
        if use_density and oos_tolerance_multiple == -1:
            # Untruncated mixtures include every real candidate of the same spectrum.
            bl_both_mask = bl_valid_pairs
        else:
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
            bl_both_mask = bl_valid_pairs & th.as_tensor(bl_match_mask, device=device)
            del bl_match_mask
        del bl_valid_pairs

        bl_true_both_mask = th.any(bl_both_mask, dim=2)

        if use_density:
            if relative:
                if bl_tol_per_pred_padded is not None:
                    bl_stds = (
                        th.maximum(bl_pred_mzs_padded, bl_min_mz_per_pred_padded)
                        * bl_tol_per_pred_padded
                    )
                else:
                    bl_stds = th.clamp(bl_pred_mzs_padded, min=tolerance_min_mz) * tolerance
            else:
                bl_stds = (
                    bl_tol_per_pred_padded
                    if bl_tol_per_pred_padded is not None
                    else tolerance * th.ones_like(bl_pred_mzs_padded)
                )
            bl_diffs = bl_true_mzs_padded.unsqueeze(2) - bl_pred_mzs_padded.unsqueeze(1)
            if dist == "gaussian":
                bl_vars = bl_stds**2
                bl_kernel_log_probs = -0.5 * bl_diffs**2 / bl_vars.unsqueeze(1) - 0.5 * (
                    _LOG_2PI + th.log(bl_vars.unsqueeze(1))
                )
            else:
                # Laplace with the Gaussian's standard deviation: scale b = sigma / sqrt(2).
                bl_scales = bl_stds.unsqueeze(1) / math.sqrt(2)
                bl_kernel_log_probs = -th.abs(bl_diffs) / bl_scales - th.log(2 * bl_scales)
            bl_ios_log_probs = (
                bl_kernel_log_probs + bl_pred_logprobs_padded.unsqueeze(1) - _log_trunc_factor
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
        ce_fn = gaussian_ce
    elif dist == "laplace":
        ce_fn = laplace_ce
    elif dist == "peak_marginal":
        ce_fn = peak_marginal_ce
    else:
        msg = (
            f"Distribution '{dist}' not implemented. "
            "Supported distributions: 'gaussian', 'laplace', 'peak_marginal'."
        )
        raise NotImplementedError(msg)

    tolerance_multiple = (
        pm_tolerance_multiple if dist == "peak_marginal" else oos_tolerance_multiple
    )
    common = {
        "tolerance": tolerance,
        "relative": relative,
        "tolerance_min_mz": tolerance_min_mz,
        "gaussian_renormalize": gaussian_renormalize,
    }
    if vectorized:
        implementation = sparse_cross_entropy_vec
        fixed = dict(
            common,
            oos_tolerance_multiple=tolerance_multiple,
            loss_batch_size=loss_batch_size,
            dist=dist,
        )
    else:
        implementation = sparse_cross_entropy_seq
        fixed = dict(common, ce_fn=ce_fn, tolerance_multiple=tolerance_multiple)

    def sce_fn(
        *args: th.Tensor,
        tol_per_sample: th.Tensor | None = None,
        min_mz_per_sample: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        return implementation(
            *args, **fixed, tol_per_sample=tol_per_sample, min_mz_per_sample=min_mz_per_sample
        )

    return sce_fn
