"""Shared observational targets and losses for candidate-support heads."""

from typing import Literal

import torch as th
import torch.nn.functional as F

from fragnnet.model.bond_fragment_heads import _validate_tensor_condition
from fragnnet.utils.spec_utils import calculate_match_mzs, linear_sum_assignment


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

    targets = channel_log_weights.new_zeros(channel_log_weights.shape)
    for batch_index in range(batch_size):
        batch_true_mzs = true_mzs[true_batch_idxs == batch_index]
        fragment_mask = fragment_batch_idxs == batch_index
        valid_mask = chemical_valid_channels[fragment_mask]
        if batch_true_mzs.numel() == 0:
            raise ValueError(f"spectrum {batch_index} has no true peaks")
        if valid_mask.numel() == 0:
            # No chemically valid candidates were proposed for this spectrum (e.g. every
            # fragment collapsed to the NULL formula); its targets stay zero.
            continue
        valid_mzs = channel_mzs[fragment_mask][valid_mask]
        valid_log_weights = channel_log_weights[fragment_mask][valid_mask].detach()
        channel_log_priors = (
            valid_log_weights if use_intensity_prior else th.zeros_like(valid_log_weights)
        )
        if tol_per_sample is None:
            channel_match = calculate_match_mzs(
                batch_true_mzs,
                valid_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
            channel_stds = (
                tolerance * th.maximum(valid_mzs, valid_mzs.new_tensor(tolerance_min_mz))
                if relative
                else th.full_like(valid_mzs, tolerance)
            )
        else:
            per_peak_tolerance = tol_per_sample[batch_index] * th.maximum(
                batch_true_mzs,
                min_mz_per_sample[batch_index],
            )
            channel_match = calculate_match_mzs(
                batch_true_mzs,
                valid_mzs,
                tolerance=0.0,
                relative=False,
                tol_per_true=per_peak_tolerance,
            )
            channel_stds = tol_per_sample[batch_index] * th.maximum(
                valid_mzs,
                min_mz_per_sample[batch_index],
            )
        channel_match = th.as_tensor(
            channel_match,
            dtype=th.bool,
            device=valid_mzs.device,
        )
        if bool((channel_stds <= 0).any().item()):
            raise ValueError("channel Gaussian standard deviations must be positive")

        mass_errors = batch_true_mzs[:, None] - valid_mzs[None, :]
        mass_log_likelihood = (
            -0.5 * (mass_errors / channel_stds[None, :]) ** 2 - th.log(channel_stds)[None, :]
        ).to(valid_log_weights)
        pair_log_likelihood = channel_log_priors[None, :] + mass_log_likelihood
        valid_targets = valid_log_weights.new_zeros(valid_log_weights.shape)
        if attribution == "gaussian_posterior":
            matched_peaks = channel_match.any(dim=1)
            if matched_peaks.any():
                responsibilities = th.softmax(
                    pair_log_likelihood[matched_peaks].masked_fill(
                        ~channel_match[matched_peaks],
                        -th.inf,
                    ),
                    dim=1,
                )
                valid_targets = responsibilities.amax(dim=0)
        elif channel_match.any():
            feasible_peaks = channel_match.any(dim=1)
            feasible_channels = channel_match.any(dim=0)
            feasible_channel_indices = th.where(feasible_channels)[0]
            feasible_match = channel_match[feasible_peaks][:, feasible_channels]
            feasible_pair_log_likelihood = pair_log_likelihood[feasible_peaks][:, feasible_channels]
            valid_pair_scores = feasible_pair_log_likelihood[feasible_match]
            pair_cost = valid_pair_scores.max() - feasible_pair_log_likelihood
            max_valid_cost = pair_cost[feasible_match].max()
            invalid_cost = (max_valid_cost + 1.0) * (min(feasible_match.shape) + 1)
            assignment_cost = pair_cost.masked_fill(~feasible_match, invalid_cost)
            peak_indices, channel_indices = linear_sum_assignment(
                assignment_cost.detach(),
                maximize=False,
            )
            assigned_valid = feasible_match[peak_indices, channel_indices]
            matched_channel_indices = feasible_channel_indices[channel_indices[assigned_valid]]
            valid_targets[matched_channel_indices] = 1.0

        batch_targets = targets.new_zeros(valid_mask.shape)
        batch_targets[valid_mask] = valid_targets
        targets[fragment_mask] = batch_targets
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
