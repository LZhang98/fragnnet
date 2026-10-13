"""Observational targets and losses for candidate-support heads (moved from
``fragmentation/candidate_support.py``): match observed peaks to candidate channels and score
the support predictions.
"""

from typing import Literal

import torch as th
import torch.nn.functional as F

from fragnnet.utils.misc_utils import _validate_tensor_condition
from fragnnet.utils.scatter_utils import scatter_reduce
from fragnnet.utils.spec_similarity import linear_sum_assignment

# The candidate window is searched with a slightly widened absolute bound so that it is a
# strict superset of the tolerance predicate; the exact predicate is then re-evaluated on
# the surviving pairs, so the widening cannot change which pairs match.
_WINDOW_SLACK = 1e-4


def _flatten_valid_channels(
    channel_mzs: th.Tensor,
    channel_log_weights: th.Tensor,
    chemical_valid_channels: th.Tensor,
    fragment_batch_idxs: th.Tensor,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Gather the chemically valid channels of every fragment into flat arrays.

    Returns ``(flat_index, mzs, log_weights, batch_idxs)`` where ``flat_index`` addresses
    the flattened ``[num_fragments, num_ion_modes, num_h_shifts]`` grid.
    """
    channels_per_fragment = channel_mzs.shape[1] * channel_mzs.shape[2]
    flat_index = th.nonzero(chemical_valid_channels.reshape(-1), as_tuple=False).squeeze(1)
    fragment_index = th.div(flat_index, channels_per_fragment, rounding_mode="floor")
    return (
        flat_index,
        channel_mzs.reshape(-1)[flat_index],
        channel_log_weights.reshape(-1)[flat_index].detach(),
        fragment_batch_idxs[fragment_index],
    )


def _matching_scales(
    *,
    true_mzs: th.Tensor,
    true_batch_idxs: th.Tensor,
    channel_mzs: th.Tensor,
    channel_batch_idxs: th.Tensor,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float | None,
    tol_per_sample: th.Tensor | None,
    min_mz_per_sample: th.Tensor | None,
) -> tuple[th.Tensor, th.Tensor | None, th.Tensor]:
    """Return ``(peak_tolerance, peak_divisor, channel_std)`` for the match predicate.

    ``peak_divisor`` is only set for the scalar relative mode, where the original predicate
    divides the mass error rather than scaling the tolerance; keeping that form makes the
    batched match bit-identical to the per-spectrum one.
    """
    if tol_per_sample is not None:
        peak_floor = min_mz_per_sample[true_batch_idxs]
        peak_tolerance = tol_per_sample[true_batch_idxs] * th.maximum(true_mzs, peak_floor)
        channel_std = tol_per_sample[channel_batch_idxs] * th.maximum(
            channel_mzs,
            min_mz_per_sample[channel_batch_idxs],
        )
        return peak_tolerance, None, channel_std
    if relative:
        peak_divisor = th.clamp(true_mzs, min=tolerance_min_mz)
        channel_std = tolerance * th.clamp(channel_mzs, min=tolerance_min_mz)
        return tolerance * peak_divisor, peak_divisor, channel_std
    return (
        th.full_like(true_mzs, tolerance),
        None,
        th.full_like(channel_mzs, tolerance),
    )


def _candidate_pairs(
    *,
    true_mzs: th.Tensor,
    true_batch_idxs: th.Tensor,
    channel_mzs: th.Tensor,
    channel_batch_idxs: th.Tensor,
    peak_tolerance: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Return ``(peak_idxs, channel_idxs)`` for every peak/channel pair worth scoring.

    Channels are ordered by ``(batch, global m/z rank)`` so that the tolerance window of a
    peak is one contiguous run of its own spectrum's channels, found with ``searchsorted``.
    Ranks are integers, so the composite sort key is exact in int64.
    """
    num_channels = channel_mzs.shape[0]
    num_peaks = true_mzs.shape[0]
    if num_channels == 0 or num_peaks == 0:
        empty = th.zeros(0, dtype=th.long, device=channel_mzs.device)
        return empty, empty

    mz_order = th.argsort(channel_mzs)
    sorted_mzs = channel_mzs[mz_order].to(th.float64)
    channel_rank = th.empty_like(mz_order)
    channel_rank[mz_order] = th.arange(num_channels, device=mz_order.device)
    channel_key = channel_batch_idxs.to(th.int64) * num_channels + channel_rank
    key_order = th.argsort(channel_key)
    sorted_keys = channel_key[key_order]

    peak_mzs = true_mzs.to(th.float64)
    window = peak_tolerance.to(th.float64) * (1.0 + _WINDOW_SLACK)
    # ``right=True``/``right=False`` make both window edges strict, matching ``diff < tol``.
    rank_lo = th.searchsorted(sorted_mzs, peak_mzs - window, right=True)
    rank_hi = th.searchsorted(sorted_mzs, peak_mzs + window, right=False)
    batch_offset = true_batch_idxs.to(th.int64) * num_channels
    start = th.searchsorted(sorted_keys, batch_offset + rank_lo)
    end = th.searchsorted(sorted_keys, batch_offset + rank_hi)

    counts = (end - start).clamp_min(0)
    peak_idxs = th.repeat_interleave(counts)
    within_peak = (
        th.arange(peak_idxs.shape[0], device=counts.device) - (counts.cumsum(0) - counts)[peak_idxs]
    )
    return peak_idxs, key_order[start[peak_idxs] + within_peak]


def _pair_match(
    *,
    mass_errors: th.Tensor,
    peak_idxs: th.Tensor,
    peak_tolerance: th.Tensor,
    peak_divisor: th.Tensor | None,
    tolerance: float,
) -> th.Tensor:
    """Evaluate the tolerance predicate on candidate pairs, in its original algebraic form."""
    absolute_errors = mass_errors.abs()
    if peak_divisor is not None:
        return absolute_errors / peak_divisor[peak_idxs] < tolerance
    return absolute_errors < peak_tolerance[peak_idxs]


def matched_peak_candidate_pairs(
    *,
    true_mzs: th.Tensor,
    true_batch_idxs: th.Tensor,
    candidate_mzs: th.Tensor,
    candidate_batch_idxs: th.Tensor,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float | None,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor]:
    """Return every peak/candidate pair of one spectrum that is inside the m/z tolerance.

    This is the whole-batch form of the per-spectrum ``calculate_match_mzs`` grid: only the
    pairs inside a peak's tolerance window are materialized, and the predicate is then
    evaluated in its original algebraic form, so the matched set is the one the dense grid
    selects.

    Args:
        true_mzs: Observed m/z values of every spectrum, shape ``(num_peaks,)``.
        true_batch_idxs: Spectrum index per observed peak, shape ``(num_peaks,)``.
        candidate_mzs: Candidate m/z values, shape ``(num_candidates,)``.
        candidate_batch_idxs: Spectrum index per candidate, shape ``(num_candidates,)``.
        tolerance: Matching tolerance; a ratio when ``relative``, otherwise Da.
        relative: Whether the scalar tolerance is relative to the observed m/z.
        tolerance_min_mz: Divisor floor for the relative scalar tolerance.
        tol_per_sample: Optional per-spectrum relative tolerance, shape ``(batch_size,)``.
        min_mz_per_sample: Optional per-spectrum divisor floor, shape ``(batch_size,)``.

    Returns:
        ``(peak_idxs, candidate_idxs)``: two aligned ``(num_matched_pairs,)`` long tensors
        indexing ``true_mzs`` and ``candidate_mzs``. Pairs are not deduplicated; one peak
        can match several candidates and one candidate several peaks.
    """
    peak_tolerance, peak_divisor, _ = _matching_scales(
        true_mzs=true_mzs,
        true_batch_idxs=true_batch_idxs,
        channel_mzs=candidate_mzs,
        channel_batch_idxs=candidate_batch_idxs,
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
        tol_per_sample=tol_per_sample,
        min_mz_per_sample=min_mz_per_sample,
    )
    peak_idxs, candidate_idxs = _candidate_pairs(
        true_mzs=true_mzs,
        true_batch_idxs=true_batch_idxs,
        channel_mzs=candidate_mzs,
        channel_batch_idxs=candidate_batch_idxs,
        peak_tolerance=peak_tolerance,
    )
    matched = _pair_match(
        mass_errors=true_mzs[peak_idxs] - candidate_mzs[candidate_idxs],
        peak_idxs=peak_idxs,
        peak_tolerance=peak_tolerance,
        peak_divisor=peak_divisor,
        tolerance=tolerance,
    )
    return peak_idxs[matched], candidate_idxs[matched]


def _pair_gaussian_log_likelihood(
    true_mzs: th.Tensor,
    peak_idxs: th.Tensor,
    channel_idxs: th.Tensor,
    channel_mzs: th.Tensor,
    channel_log_priors: th.Tensor,
    channel_stds: th.Tensor,
) -> th.Tensor:
    """Log prior plus Gaussian log density, omitting the common log(sqrt(2*pi))."""
    pair_stds = channel_stds[channel_idxs]
    mass_errors = true_mzs[peak_idxs] - channel_mzs[channel_idxs]
    mass_log_likelihood = (-0.5 * (mass_errors / pair_stds) ** 2 - th.log(pair_stds)).to(
        channel_log_priors
    )
    return channel_log_priors[channel_idxs] + mass_log_likelihood


def _gaussian_posterior_targets(
    *,
    true_mzs: th.Tensor,
    peak_idxs: th.Tensor,
    channel_idxs: th.Tensor,
    channel_mzs: th.Tensor,
    channel_log_priors: th.Tensor,
    channel_stds: th.Tensor,
    num_peaks: int,
) -> th.Tensor:
    """Per-channel mixture responsibility, maximized over the peaks that matched it.

    The dense form softmaxes over all channels with non-matching entries at ``-inf``; those
    contribute neither to the denominator nor to the per-channel maximum, so restricting the
    computation to the matching pairs is exact.
    """
    num_channels = channel_stds.shape[0]
    if peak_idxs.numel() == 0:
        return channel_log_priors.new_zeros(num_channels)
    pair_log_likelihood = _pair_gaussian_log_likelihood(
        true_mzs, peak_idxs, channel_idxs, channel_mzs, channel_log_priors, channel_stds
    )
    peak_maxima = scatter_reduce(
        pair_log_likelihood,
        peak_idxs,
        reduce="amax",
        dim_size=num_peaks,
        default=0.0,
        include_self=False,
    )
    pair_weights = th.exp(pair_log_likelihood - peak_maxima[peak_idxs])
    peak_mass = scatter_reduce(
        pair_weights,
        peak_idxs,
        reduce="sum",
        dim_size=num_peaks,
        default=0.0,
    )
    responsibilities = pair_weights / peak_mass[peak_idxs]
    return scatter_reduce(
        responsibilities,
        channel_idxs,
        reduce="amax",
        dim_size=num_channels,
        default=0.0,
    )


def _hungarian_targets(
    *,
    true_mzs: th.Tensor,
    peak_idxs: th.Tensor,
    channel_idxs: th.Tensor,
    channel_mzs: th.Tensor,
    channel_batch_idxs: th.Tensor,
    channel_log_priors: th.Tensor,
    channel_stds: th.Tensor,
    batch_size: int,
) -> th.Tensor:
    """Hard one-to-one peak/channel assignment, solved one spectrum at a time.

    Unlike the posterior, the assignment couples every feasible pair of a spectrum, and the
    solver is sequential (scipy unless ``torch_linear_assignment`` is installed), so this
    path keeps a Python loop. It is restricted to the feasible pairs, which is far smaller
    than the dense peak-by-channel grid the loop used to build.
    """
    targets = channel_log_priors.new_zeros(channel_stds.shape[0])
    if peak_idxs.numel() == 0:
        return targets
    pair_log_likelihood = _pair_gaussian_log_likelihood(
        true_mzs, peak_idxs, channel_idxs, channel_mzs, channel_log_priors, channel_stds
    )
    pair_batch_idxs = channel_batch_idxs[channel_idxs]
    pair_order = th.argsort(pair_batch_idxs, stable=True)
    pair_bounds = th.zeros(batch_size + 1, dtype=th.long, device=targets.device)
    pair_bounds[1:] = th.bincount(pair_batch_idxs, minlength=batch_size).cumsum(0)
    bounds = pair_bounds.tolist()

    global_cost = pair_log_likelihood[pair_order].max() - pair_log_likelihood[pair_order]
    for batch_index in range(batch_size):
        start, end = bounds[batch_index], bounds[batch_index + 1]
        if start == end:
            continue
        spectrum_pairs = pair_order[start:end]
        spectrum_peaks, local_peaks = th.unique(peak_idxs[spectrum_pairs], return_inverse=True)
        spectrum_channels, local_channels = th.unique(
            channel_idxs[spectrum_pairs], return_inverse=True
        )
        # Infeasible cells must never be chosen: price them above any full feasible
        # assignment, exactly as the dense implementation did.
        feasible_cost = global_cost[start:end]
        infeasible_cost = (feasible_cost.max() + 1.0) * (
            min(spectrum_peaks.shape[0], spectrum_channels.shape[0]) + 1
        )
        assignment_cost = th.full(
            (spectrum_peaks.shape[0], spectrum_channels.shape[0]),
            float(infeasible_cost),
            dtype=feasible_cost.dtype,
            device=feasible_cost.device,
        )
        assignment_cost[local_peaks, local_channels] = feasible_cost
        row_idxs, col_idxs = linear_sum_assignment(assignment_cost.detach(), maximize=False)
        feasible = assignment_cost[row_idxs, col_idxs] < infeasible_cost
        targets[spectrum_channels[col_idxs[feasible]]] = 1.0
    return targets


def observational_channel_targets(
    *,
    true_mzs: th.Tensor,
    true_batch_idxs: th.Tensor,
    channel_mzs: th.Tensor,
    channel_log_weights: th.Tensor,
    chemical_valid_channels: th.Tensor,
    fragment_batch_idxs: th.Tensor,
    batch_size: int,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float | None,
    attribution: Literal["gaussian_posterior", "hungarian"],
    use_intensity_prior: bool = True,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> th.Tensor:
    """Attribute observed peaks to chemically valid candidate channels.

    ``gaussian_posterior`` returns mixture responsibilities. For each observed
    peak, an optional detached channel prior is multiplied by a Gaussian
    mass-error likelihood and normalized over matching channels. ``hungarian``
    converts the same pairwise likelihood into a hard one-to-one assignment.

    The whole batch is handled at once: peak/channel pairs outside the mass tolerance
    contribute nothing to either attribution, so only the pairs inside it are materialized.
    """
    if channel_mzs.shape != chemical_valid_channels.shape or channel_mzs.ndim != 3:
        raise ValueError("channel masses and validity must be aligned rank-3 tensors")
    if channel_log_weights.shape != channel_mzs.shape:
        raise ValueError("channel log weights must align with channel masses")
    if fragment_batch_idxs.shape != (channel_mzs.shape[0],):
        raise ValueError("fragment_batch_idxs must contain one index per fragment")
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    if relative and tolerance_min_mz is None:
        raise ValueError("relative support matching requires tolerance_min_mz")
    if attribution not in {"gaussian_posterior", "hungarian"}:
        raise ValueError(f"unknown channel support attribution: {attribution}")
    if not isinstance(use_intensity_prior, bool):
        raise ValueError("use_intensity_prior must be boolean")
    if not bool(th.isfinite(channel_log_weights[chemical_valid_channels]).all().item()):
        raise ValueError("valid channels must have finite log weights")

    peak_counts = th.zeros(batch_size, dtype=th.long, device=true_mzs.device).index_add(
        0,
        true_batch_idxs.to(th.int64),
        th.ones_like(true_batch_idxs, dtype=th.long),
    )
    empty_spectra = (peak_counts == 0).nonzero().flatten()
    if empty_spectra.numel() > 0:
        raise ValueError(f"spectrum {int(empty_spectra[0])} has no true peaks")

    flat_index, valid_mzs, valid_log_weights, valid_batch_idxs = _flatten_valid_channels(
        channel_mzs,
        channel_log_weights,
        chemical_valid_channels,
        fragment_batch_idxs,
    )
    peak_tolerance, peak_divisor, channel_stds = _matching_scales(
        true_mzs=true_mzs,
        true_batch_idxs=true_batch_idxs,
        channel_mzs=valid_mzs,
        channel_batch_idxs=valid_batch_idxs,
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
        tol_per_sample=tol_per_sample,
        min_mz_per_sample=min_mz_per_sample,
    )
    if bool((channel_stds <= 0).any().item()):
        raise ValueError("channel Gaussian standard deviations must be positive")

    peak_idxs, channel_idxs = _candidate_pairs(
        true_mzs=true_mzs,
        true_batch_idxs=true_batch_idxs,
        channel_mzs=valid_mzs,
        channel_batch_idxs=valid_batch_idxs,
        peak_tolerance=peak_tolerance,
    )
    matched = _pair_match(
        mass_errors=true_mzs[peak_idxs] - valid_mzs[channel_idxs],
        peak_idxs=peak_idxs,
        peak_tolerance=peak_tolerance,
        peak_divisor=peak_divisor,
        tolerance=tolerance,
    )
    peak_idxs, channel_idxs = peak_idxs[matched], channel_idxs[matched]

    channel_log_priors = (
        valid_log_weights if use_intensity_prior else th.zeros_like(valid_log_weights)
    )
    if attribution == "gaussian_posterior":
        valid_targets = _gaussian_posterior_targets(
            true_mzs=true_mzs,
            peak_idxs=peak_idxs,
            channel_idxs=channel_idxs,
            channel_mzs=valid_mzs,
            channel_log_priors=channel_log_priors,
            channel_stds=channel_stds,
            num_peaks=true_mzs.shape[0],
        )
    else:
        valid_targets = _hungarian_targets(
            true_mzs=true_mzs,
            peak_idxs=peak_idxs,
            channel_idxs=channel_idxs,
            channel_mzs=valid_mzs,
            channel_batch_idxs=valid_batch_idxs,
            channel_log_priors=channel_log_priors,
            channel_stds=channel_stds,
            batch_size=batch_size,
        )

    targets = channel_log_weights.new_zeros(channel_log_weights.shape)
    targets.view(-1)[flat_index] = valid_targets
    return targets.detach()


def balanced_binary_support_loss(
    logits: th.Tensor,
    targets: th.Tensor,
    valid_mask: th.Tensor,
    item_batch_idxs: th.Tensor,
    batch_size: int,
    *,
    allow_empty_batches: bool = False,
) -> th.Tensor:
    """Return class-balanced binary support loss for each spectrum."""
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("support logits, targets, and validity must have matching shapes")
    if logits.shape[0] != item_batch_idxs.numel():
        raise ValueError("item_batch_idxs must contain one index per first-axis item")
    if valid_mask.dtype != th.bool:
        raise ValueError("support validity must be boolean")
    target_values = targets.to(logits)
    if not bool(th.isfinite(target_values).all().item()) or bool(
        ((target_values < 0.0) | (target_values > 1.0)).any().item()
    ):
        raise ValueError("support targets must be finite values in [0, 1]")

    valid_weights = valid_mask.to(logits)
    positive_weights = (target_values * valid_weights).flatten()
    negative_weights = ((1.0 - target_values) * valid_weights).flatten()
    positive_element_loss = F.softplus(-logits).flatten() * positive_weights
    negative_element_loss = F.softplus(logits).flatten() * negative_weights
    flat_batch_idxs = (
        item_batch_idxs.reshape(item_batch_idxs.shape[0], *([1] * (logits.ndim - 1)))
        .expand_as(logits)
        .flatten()
        .long()
    )

    def _spectrum_sum(values: th.Tensor) -> th.Tensor:
        return logits.new_zeros(batch_size).scatter_add(0, flat_batch_idxs, values)

    positive_mass = _spectrum_sum(positive_weights)
    negative_mass = _spectrum_sum(negative_weights)
    has_positive = positive_mass > 0.0
    has_negative = negative_mass > 0.0
    if not allow_empty_batches:
        _validate_tensor_condition(
            (has_positive | has_negative).all(),
            "every spectrum must have at least one valid support state",
        )
    tiny = th.finfo(logits.dtype).tiny
    positive_mean = _spectrum_sum(positive_element_loss) / positive_mass.clamp_min(tiny)
    negative_mean = _spectrum_sum(negative_element_loss) / negative_mass.clamp_min(tiny)
    class_count = has_positive.to(logits) + has_negative.to(logits)
    return (
        th.where(has_positive, positive_mean, th.zeros_like(positive_mean))
        + th.where(has_negative, negative_mean, th.zeros_like(negative_mean))
    ) / class_count.clamp_min(1.0)
