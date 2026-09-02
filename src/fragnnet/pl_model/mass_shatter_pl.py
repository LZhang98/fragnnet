"""PyTorch Lightning training wrapper for MASShatter.

MASShatter emits either normalized sparse log-probabilities or independent sparse
normal-space intensities. SpectrumPL's preprocessing, metric, and reduction machinery is
otherwise reused unchanged. This module only supplies the three model-specific pieces:

- :meth:`MassShatterPL._setup_model` maps flat hparams onto ``MassShatterModel``.
- :meth:`MassShatterPL._setup_loss_fn` wraps the sparse cross entropy and the
  adjacent-CE switch objective, and forwards MASShatter's candidate-generation
  diagnostics into the logged metrics.
- :meth:`MassShatterPL.transfer_batch_to_device` keeps a CPU copy of the graph
  indices, because mask selection runs in Python on the host.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Literal

import torch as th
from rdkit import Chem

from fragnnet.frag.mass_shatter_candidates import MassShatterStaticGraph
from fragnnet.frag.mass_shatter_templates import (
    MassShatterTemplate,
    build_mass_shatter_templates,
)
from fragnnet.model import MassShatterModel
from fragnnet.model.loss import (
    ce_switch_pair_bce,
    get_sparse_cross_entropy_fn,
    sparse_peak_list_distance,
)
from fragnnet.pl_model.spectrum_pl import SpectrumPL, select_context_ce_values
from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.utils.spec_utils import (
    calculate_match_mzs,
    get_ints_transform_func,
    linear_sum_assignment,
)

_CUT_COUNT_BUCKETS = {
    "0": lambda cut_counts: cut_counts == 0,
    "1": lambda cut_counts: cut_counts == 1,
    "2": lambda cut_counts: cut_counts == 2,
    "3": lambda cut_counts: cut_counts == 3,
    "4": lambda cut_counts: cut_counts == 4,
    "5": lambda cut_counts: cut_counts == 5,
    "6plus": lambda cut_counts: cut_counts >= 6,
    "4plus": lambda cut_counts: cut_counts >= 4,
}

logger = logging.getLogger(__name__)

_ION_H_REQUIRED_K_METRIC_NAMES = (
    "valid_ion_h_per_fragment_mean",
    "attributed_ion_h_per_fragment_mean",
    "ion_h_supported_fragment_fraction",
    "required_ion_h_top_k_95_mean",
    "required_ion_h_top_k_95_p90",
    "required_ion_h_top_k_95_max",
    "required_ion_h_top_k_99_mean",
    "required_ion_h_top_k_99_p90",
    "required_ion_h_top_k_99_max",
)


@lru_cache(maxsize=100_000)
def _cached_mass_shatter_templates(
    smiles: str,
) -> tuple[MassShatterTemplate, ...]:
    """Match SMARTS once per unique molecule, outside the model hot path."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecule SMILES for SMARTS matching: {smiles}")
    return build_mass_shatter_templates(mol)


def _candidate_coverage_by_cut_count(
    *,
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    true_batch_idxs: th.Tensor,
    candidate_mzs: th.Tensor,
    candidate_batch_idxs: th.Tensor,
    candidate_cut_counts: th.Tensor,
    batch_size: int,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> dict[str, th.Tensor]:
    """Measure overall and cut-bucket true-peak coverage from candidate masses.

    A true peak may be covered by multiple buckets, so the resulting recall values
    describe each bucket's standalone support and are not additive.
    """
    metrics = {
        f"candidate_{metric}_cut_{bucket}": true_ints.new_zeros(batch_size)
        for bucket in _CUT_COUNT_BUCKETS
        for metric in ("recall", "wrecall", "precision")
    }
    metrics["candidate_recall"] = true_ints.new_zeros(batch_size)
    metrics["candidate_wrecall"] = true_ints.new_zeros(batch_size)
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    for batch_index in range(batch_size):
        true_mask = true_batch_idxs == batch_index
        batch_true_mzs = true_mzs[true_mask]
        batch_true_ints = true_ints[true_mask]
        if batch_true_mzs.numel() == 0:
            raise ValueError(f"spectrum {batch_index} has no true peaks")
        true_intensity = batch_true_ints.sum()
        batch_candidate_mzs = candidate_mzs[candidate_batch_idxs == batch_index]
        if tol_per_sample is None:
            all_match_mask = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
        else:
            tol_per_true = tol_per_sample[batch_index] * th.maximum(
                batch_true_mzs,
                min_mz_per_sample[batch_index],
            )
            all_match_mask = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=0.0,
                relative=False,
                tol_per_true=tol_per_true,
            )
        metrics["candidate_recall"][batch_index] = (
            all_match_mask.any(dim=1).to(batch_true_ints).mean()
        )
        metrics["candidate_wrecall"][batch_index] = (
            batch_true_ints * all_match_mask.any(dim=1)
        ).sum() / true_intensity
        for bucket, select_bucket in _CUT_COUNT_BUCKETS.items():
            candidate_mask = (candidate_batch_idxs == batch_index) & select_bucket(
                candidate_cut_counts
            )
            if tol_per_sample is None:
                match_mask = calculate_match_mzs(
                    batch_true_mzs,
                    candidate_mzs[candidate_mask],
                    tolerance=tolerance,
                    relative=relative,
                    tolerance_min_mz=tolerance_min_mz,
                )
            else:
                tol_per_true = tol_per_sample[batch_index] * th.maximum(
                    batch_true_mzs,
                    min_mz_per_sample[batch_index],
                )
                match_mask = calculate_match_mzs(
                    batch_true_mzs,
                    candidate_mzs[candidate_mask],
                    tolerance=0.0,
                    relative=False,
                    tol_per_true=tol_per_true,
                )
            true_match_mask = match_mask.any(dim=1)
            metrics[f"candidate_recall_cut_{bucket}"][batch_index] = true_match_mask.to(
                batch_true_ints
            ).mean()
            metrics[f"candidate_wrecall_cut_{bucket}"][batch_index] = (
                batch_true_ints * true_match_mask
            ).sum() / true_intensity
            candidate_match_mask = match_mask.any(dim=0)
            if candidate_match_mask.numel() > 0:
                metrics[f"candidate_precision_cut_{bucket}"][batch_index] = candidate_match_mask.to(
                    batch_true_ints
                ).mean()
    return metrics


def _candidate_weighted_recall(
    *,
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    true_batch_idxs: th.Tensor,
    candidate_mzs: th.Tensor,
    candidate_batch_idxs: th.Tensor,
    batch_size: int,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> th.Tensor:
    """Measure weighted true-peak coverage from the full chemical candidate set."""
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    weighted_recall = true_ints.new_zeros(batch_size)
    for batch_index in range(batch_size):
        true_mask = true_batch_idxs == batch_index
        batch_true_mzs = true_mzs[true_mask]
        batch_true_ints = true_ints[true_mask]
        if batch_true_mzs.numel() == 0:
            raise ValueError(f"spectrum {batch_index} has no true peaks")
        batch_candidate_mzs = candidate_mzs[candidate_batch_idxs == batch_index]
        if tol_per_sample is None:
            match_mask = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
        else:
            tol_per_true = tol_per_sample[batch_index] * th.maximum(
                batch_true_mzs,
                min_mz_per_sample[batch_index],
            )
            match_mask = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=0.0,
                relative=False,
                tol_per_true=tol_per_true,
            )
        weighted_recall[batch_index] = (
            batch_true_ints * match_mask.any(dim=1)
        ).sum() / batch_true_ints.sum()
    return weighted_recall


def _proposal_coverage_loss(
    *,
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    candidate_mzs: th.Tensor,
    candidate_fragment_idxs: th.Tensor,
    fragment_logprobs: th.Tensor,
    fragment_batch_idxs: th.Tensor,
    batch_size: int,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    barcode_target: bool = False,
) -> th.Tensor:
    """Train proposal coverage without assigning probability to ion/H channels.

    Every unique fragment is counted at most once for a true peak, even when several
    of its valid CRF/CMF or hydrogen-transfer channels match that peak. This makes the
    proposal objective depend on reachable support rather than the number or learned
    probability of ion/H channels. When ``barcode_target`` is enabled, every retained
    true peak contributes equally instead of being weighted by observed intensity.
    """
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    if candidate_mzs.shape != candidate_fragment_idxs.shape:
        raise ValueError("proposal candidate masses and fragment indices must align")
    if fragment_logprobs.shape != fragment_batch_idxs.shape:
        raise ValueError("proposal fragment probabilities and batch indices must align")
    coverage_loss = fragment_logprobs.new_zeros(batch_size)
    for batch_index in range(batch_size):
        true_mask = true_batch_idxs == batch_index
        batch_true_mzs = true_mzs[true_mask]
        batch_fragment_indices = th.where(fragment_batch_idxs == batch_index)[0]
        if batch_true_mzs.numel() == 0:
            raise ValueError(f"spectrum {batch_index} has no true peaks")
        if batch_fragment_indices.numel() == 0:
            raise ValueError(f"spectrum {batch_index} has no proposed fragments")
        batch_true_probabilities = (
            true_logprobs.new_full(
                batch_true_mzs.shape,
                1.0 / batch_true_mzs.numel(),
            )
            if barcode_target
            else true_logprobs[true_mask].exp()
        )

        first_fragment_index = batch_fragment_indices[0]
        if not th.equal(
            batch_fragment_indices,
            th.arange(
                first_fragment_index,
                first_fragment_index + batch_fragment_indices.numel(),
                device=batch_fragment_indices.device,
            ),
        ):
            raise ValueError("proposal fragments must be contiguous within each spectrum")
        candidate_mask = fragment_batch_idxs[candidate_fragment_idxs] == batch_index
        batch_candidate_mzs = candidate_mzs[candidate_mask]
        local_fragment_indices = candidate_fragment_idxs[candidate_mask] - first_fragment_index
        if tol_per_sample is None:
            channel_match = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
        else:
            per_peak_tolerance = tol_per_sample[batch_index] * th.maximum(
                batch_true_mzs,
                min_mz_per_sample[batch_index],
            )
            channel_match = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=0.0,
                relative=False,
                tol_per_true=per_peak_tolerance,
            )

        fragment_match_counts = fragment_logprobs.new_zeros(
            (batch_true_mzs.numel(), batch_fragment_indices.numel())
        )
        fragment_match_counts.scatter_add_(
            1,
            local_fragment_indices.reshape(1, -1).expand(batch_true_mzs.numel(), -1),
            channel_match.to(fragment_match_counts),
        )
        fragment_match = fragment_match_counts > 0
        true_peak_covered = fragment_match.any(dim=1)
        batch_fragment_logprobs = fragment_logprobs[batch_fragment_indices]
        peak_log_coverage = th.logsumexp(
            batch_fragment_logprobs.reshape(1, -1).masked_fill(~fragment_match, -th.inf),
            dim=1,
        )
        coverage_loss[batch_index] = th.where(
            true_peak_covered,
            -batch_true_probabilities * peak_log_coverage,
            th.zeros_like(peak_log_coverage),
        ).sum()
    return coverage_loss


def _observational_channel_targets(
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
    tolerance_min_mz: float,
    attribution: Literal["gaussian_posterior", "hungarian"],
    use_intensity_prior: bool = True,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> th.Tensor:
    """Attribute observed peaks to chemically valid ion/H channels.

    ``gaussian_posterior`` returns FraGNNet-style mixture responsibilities. For each
    observed peak, an optional detached channel prior is multiplied by a Gaussian
    mass-error likelihood and normalized over matching channels. ``hungarian`` converts
    the same pairwise likelihood into a hard one-to-one peak/channel assignment.
    """
    if channel_mzs.shape != chemical_valid_channels.shape or channel_mzs.ndim != 3:
        raise ValueError("channel masses and validity must be aligned rank-3 tensors")
    if channel_log_weights.shape != channel_mzs.shape:
        raise ValueError("channel log weights must align with channel masses")
    if fragment_batch_idxs.shape != (channel_mzs.shape[0],):
        raise ValueError("fragment_batch_idxs must contain one index per fragment")
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
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
            raise ValueError(f"spectrum {batch_index} has no proposed fragments")
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
                # Channel support is existential, so a channel keeps its largest
                # responsibility if overlapping observed tolerance windows occur.
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


def _balanced_binary_support_loss(
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
    positive_element_loss = th.nn.functional.softplus(-logits)
    negative_element_loss = th.nn.functional.softplus(logits)
    losses = logits.new_zeros(batch_size)
    for batch_index in range(batch_size):
        batch_items = item_batch_idxs == batch_index
        batch_valid = valid_mask[batch_items]
        batch_targets = target_values[batch_items]
        class_losses = []
        positive_weights = batch_targets * batch_valid
        negative_weights = (1.0 - batch_targets) * batch_valid
        positive_mass = positive_weights.sum()
        negative_mass = negative_weights.sum()
        if positive_mass.item() > 0.0:
            class_losses.append(
                (positive_element_loss[batch_items] * positive_weights).sum() / positive_mass
            )
        if negative_mass.item() > 0.0:
            class_losses.append(
                (negative_element_loss[batch_items] * negative_weights).sum() / negative_mass
            )
        if not class_losses:
            if allow_empty_batches:
                continue
            raise ValueError(f"spectrum {batch_index} has no valid support states")
        losses[batch_index] = th.stack(class_losses).mean()
    return losses


def _ion_mode_support_loss(
    logits: th.Tensor,
    targets: th.Tensor,
    valid_mask: th.Tensor,
    item_batch_idxs: th.Tensor,
    batch_size: int,
) -> th.Tensor:
    """Supervise CRF/CMF ranking only where both ion modes are valid."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("ion-mode support loss requires exactly two ion modes")
    two_mode_items = valid_mask.all(dim=1, keepdim=True)
    return _balanced_binary_support_loss(
        logits,
        targets,
        valid_mask & two_mode_items,
        item_batch_idxs,
        batch_size,
        allow_empty_batches=True,
    )


def _channel_support_metrics(
    *,
    channel_targets: th.Tensor,
    chemical_valid_channels: th.Tensor,
    fragment_batch_idxs: th.Tensor,
    batch_size: int,
    fragment_logprobs: th.Tensor | None,
) -> dict[str, th.Tensor]:
    """Measure responsibility-weighted fragment and channel support prevalence."""
    valid_modes = chemical_valid_channels.any(dim=2)
    mode_targets = channel_targets.sum(dim=2).clamp(max=1.0)
    fragment_targets = channel_targets.flatten(start_dim=1).sum(dim=1).clamp(max=1.0)
    reference = channel_targets.new_zeros((), dtype=th.float32)
    metrics = {
        "fragment_precision": reference.new_zeros(batch_size),
        "fragment_weighted_precision": reference.new_zeros(batch_size),
        "ion_mode_support_positive_rate": reference.new_zeros(batch_size),
        "ion_h_support_positive_rate": reference.new_zeros(batch_size),
    }
    for batch_index in range(batch_size):
        fragment_mask = fragment_batch_idxs == batch_index
        batch_fragments = fragment_targets[fragment_mask]
        batch_valid_modes = valid_modes[fragment_mask]
        batch_valid_channels = chemical_valid_channels[fragment_mask]
        metrics["fragment_precision"][batch_index] = batch_fragments.mean()
        metrics["ion_mode_support_positive_rate"][batch_index] = mode_targets[fragment_mask][
            batch_valid_modes
        ].mean()
        metrics["ion_h_support_positive_rate"][batch_index] = channel_targets[fragment_mask][
            batch_valid_channels
        ].mean()
        if fragment_logprobs is None:
            metrics["fragment_weighted_precision"][batch_index] = batch_fragments.mean()
        else:
            metrics["fragment_weighted_precision"][batch_index] = (
                fragment_logprobs[fragment_mask].exp() * batch_fragments
            ).sum()
    return metrics


def _ion_h_required_k_metrics(
    *,
    channel_targets: th.Tensor,
    support_logits: th.Tensor,
    chemical_valid_channels: th.Tensor,
    fragment_batch_idxs: th.Tensor,
    batch_size: int,
) -> dict[str, th.Tensor]:
    """Summarize per-fragment H support and model-ranked coverage requirements."""
    if (
        channel_targets.shape != support_logits.shape
        or channel_targets.shape != chemical_valid_channels.shape
        or channel_targets.ndim != 3
    ):
        raise ValueError("channel targets, logits, and validity must be aligned rank-3 tensors")
    if chemical_valid_channels.dtype != th.bool:
        raise ValueError("chemical channel validity must be boolean")
    if fragment_batch_idxs.shape != (channel_targets.shape[0],):
        raise ValueError("fragment_batch_idxs must contain one index per fragment")

    flat_valid = chemical_valid_channels.flatten(start_dim=1)
    flat_targets = channel_targets.flatten(start_dim=1).masked_fill(~flat_valid, 0.0)
    flat_logits = support_logits.detach().flatten(start_dim=1).masked_fill(~flat_valid, -th.inf)
    ranked_indices = th.sort(flat_logits, dim=1, descending=True, stable=True).indices
    ranked_targets = flat_targets.gather(1, ranked_indices)
    target_mass = flat_targets.sum(dim=1)
    supported_fragments = target_mass > 0.0
    cumulative_target_mass = ranked_targets.cumsum(dim=1)

    required_ks = {}
    for label, coverage in (("95", 0.95), ("99", 0.99)):
        reaches_coverage = cumulative_target_mass >= target_mass.unsqueeze(1) * coverage
        first_reaching_rank = reaches_coverage.to(th.int64).argmax(dim=1) + 1
        required_ks[label] = th.where(
            supported_fragments,
            first_reaching_rank,
            th.zeros_like(first_reaching_rank),
        ).to(device=channel_targets.device, dtype=th.float32)

    reference = channel_targets.new_zeros((), dtype=th.float32)
    metrics = {
        "valid_ion_h_per_fragment_mean": reference.new_zeros(batch_size),
        "attributed_ion_h_per_fragment_mean": reference.new_zeros(batch_size),
        "ion_h_supported_fragment_fraction": reference.new_zeros(batch_size),
    }
    for label in required_ks:
        for statistic in ("mean", "p90", "max"):
            metrics[f"required_ion_h_top_k_{label}_{statistic}"] = reference.new_zeros(batch_size)

    valid_counts = flat_valid.sum(dim=1).to(device=channel_targets.device, dtype=th.float32)
    attributed_counts = (
        ((flat_targets > 0.0) & flat_valid)
        .sum(dim=1)
        .to(
            device=channel_targets.device,
            dtype=th.float32,
        )
    )
    for batch_index in range(batch_size):
        fragment_mask = fragment_batch_idxs == batch_index
        batch_supported = supported_fragments[fragment_mask]
        metrics["valid_ion_h_per_fragment_mean"][batch_index] = valid_counts[fragment_mask].mean()
        metrics["attributed_ion_h_per_fragment_mean"][batch_index] = attributed_counts[
            fragment_mask
        ].mean()
        metrics["ion_h_supported_fragment_fraction"][batch_index] = batch_supported.to(
            channel_targets
        ).mean()
        if not batch_supported.any():
            continue
        for label, values in required_ks.items():
            batch_values = values[fragment_mask][batch_supported]
            metrics[f"required_ion_h_top_k_{label}_mean"][batch_index] = batch_values.mean()
            metrics[f"required_ion_h_top_k_{label}_p90"][batch_index] = th.quantile(
                batch_values,
                0.9,
            )
            metrics[f"required_ion_h_top_k_{label}_max"][batch_index] = batch_values.max()
    return metrics


def _shared_group_proposal_coverage_loss(
    *,
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    candidate_mzs: th.Tensor,
    candidate_fragment_idxs: th.Tensor,
    fragment_logprobs: th.Tensor,
    fragment_batch_idxs: th.Tensor,
    group_ids: th.Tensor,
    batch_size: int,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    barcode_target: bool = False,
) -> th.Tensor:
    """Supervise one shared proposal against one merged target per group.

    Each preprocessed spectrum contributes unit total intensity to its group. Exact
    duplicate m/z values are summed, matching dataset-level spectrum merging, and the
    result is normalized before proposal coverage is evaluated. Candidate tensors are
    compacted to the first spectrum in each group because shared proposals repeat the
    same fragment set for every group member.
    """
    group_ids = group_ids.reshape(-1)
    if group_ids.shape != (batch_size,):
        raise ValueError("proposal_group_ids must contain one id per spectrum")
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")

    # 1. Map arbitrary group ids to stable first-occurrence indices.
    _, sorted_inverse = th.unique(group_ids, sorted=True, return_inverse=True)
    num_groups = int(sorted_inverse.max().item()) + 1
    spectrum_indices = th.arange(batch_size, device=group_ids.device)
    first_by_sorted_group = spectrum_indices.new_full((num_groups,), batch_size).scatter_reduce(
        0,
        sorted_inverse,
        spectrum_indices,
        reduce="amin",
        include_self=True,
    )
    sorted_group_order = th.argsort(first_by_sorted_group)
    sorted_to_stable = th.empty_like(sorted_group_order)
    sorted_to_stable[sorted_group_order] = th.arange(num_groups, device=group_ids.device)
    spectrum_to_group = sorted_to_stable[sorted_inverse]
    representatives = first_by_sorted_group[sorted_group_order]

    # 2. Merge exact-mass peaks after assigning every spectrum to its group.
    true_group_idxs = spectrum_to_group[true_batch_idxs]
    merged_mzs = []
    merged_probabilities = []
    merged_batch_idxs = []
    for group_index in range(num_groups):
        group_mask = true_group_idxs == group_index
        group_mzs, inverse = th.unique(
            true_mzs[group_mask],
            sorted=True,
            return_inverse=True,
        )
        group_probabilities = true_logprobs.new_zeros(group_mzs.shape)
        group_probabilities.scatter_add_(0, inverse, true_logprobs[group_mask].exp())
        group_probabilities = group_probabilities / group_probabilities.sum()
        merged_mzs.append(group_mzs)
        merged_probabilities.append(group_probabilities)
        merged_batch_idxs.append(
            th.full(
                group_mzs.shape,
                group_index,
                dtype=th.long,
                device=true_batch_idxs.device,
            )
        )

    # 3. Keep one copy of the repeated shared candidates per group.
    compact_candidate_mzs = []
    compact_candidate_fragment_idxs = []
    compact_fragment_logprobs = []
    compact_fragment_batch_idxs = []
    fragment_offset = 0
    for group_index, representative in enumerate(representatives):
        fragment_indices = th.where(fragment_batch_idxs == representative)[0]
        if fragment_indices.numel() == 0:
            raise ValueError(f"shared proposal group {group_index} has no fragments")
        first_fragment_index = fragment_indices[0]
        if not th.equal(
            fragment_indices,
            th.arange(
                first_fragment_index,
                first_fragment_index + fragment_indices.numel(),
                device=fragment_indices.device,
            ),
        ):
            raise ValueError("proposal fragments must be contiguous within each spectrum")
        candidate_mask = fragment_batch_idxs[candidate_fragment_idxs] == representative
        compact_candidate_mzs.append(candidate_mzs[candidate_mask])
        compact_candidate_fragment_idxs.append(
            candidate_fragment_idxs[candidate_mask] - first_fragment_index + fragment_offset
        )
        compact_fragment_logprobs.append(fragment_logprobs[fragment_indices])
        compact_fragment_batch_idxs.append(
            th.full(
                fragment_indices.shape,
                group_index,
                dtype=th.long,
                device=fragment_batch_idxs.device,
            )
        )
        fragment_offset += fragment_indices.numel()

    group_losses = _proposal_coverage_loss(
        true_mzs=th.cat(merged_mzs),
        true_logprobs=th.cat(merged_probabilities).log(),
        true_batch_idxs=th.cat(merged_batch_idxs),
        candidate_mzs=th.cat(compact_candidate_mzs),
        candidate_fragment_idxs=th.cat(compact_candidate_fragment_idxs),
        fragment_logprobs=th.cat(compact_fragment_logprobs),
        fragment_batch_idxs=th.cat(compact_fragment_batch_idxs),
        batch_size=num_groups,
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
        tol_per_sample=(None if tol_per_sample is None else tol_per_sample[representatives]),
        min_mz_per_sample=(
            None if min_mz_per_sample is None else min_mz_per_sample[representatives]
        ),
        barcode_target=barcode_target,
    )
    return group_losses[spectrum_to_group]


def _cross_ev_fragment_delta_loss(
    *,
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    candidate_mzs: th.Tensor,
    candidate_fragment_idxs: th.Tensor,
    fragment_logprobs: th.Tensor,
    fragment_batch_idxs: th.Tensor,
    group_ids: th.Tensor,
    ce_values: th.Tensor,
    batch_size: int,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
) -> tuple[th.Tensor, th.Tensor]:
    """Match the predicted and observed probability change of shared fragments."""
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    group_ids = group_ids.reshape(-1)
    ce_values = ce_values.reshape(-1)
    if group_ids.shape != (batch_size,) or ce_values.shape != (batch_size,):
        raise ValueError("group_ids and ce_values must contain one value per spectrum")

    # 1. Convert observed peak probabilities into atom-fragment targets. When one
    # peak aliases several fragments, split its mass evenly to keep targets normalized.
    targets: list[th.Tensor] = []
    target_available: list[th.Tensor] = []
    predictions: list[th.Tensor] = []
    for batch_index in range(batch_size):
        fragment_indices = th.where(fragment_batch_idxs == batch_index)[0]
        if fragment_indices.numel() == 0:
            raise ValueError(f"spectrum {batch_index} has no proposed fragments")
        first_fragment_index = fragment_indices[0]
        if not th.equal(
            fragment_indices,
            th.arange(
                first_fragment_index,
                first_fragment_index + fragment_indices.numel(),
                device=fragment_indices.device,
            ),
        ):
            raise ValueError("shared fragments must be contiguous within each spectrum")

        true_mask = true_batch_idxs == batch_index
        batch_true_mzs = true_mzs[true_mask]
        batch_true_probabilities = true_logprobs[true_mask].exp()
        candidate_mask = fragment_batch_idxs[candidate_fragment_idxs] == batch_index
        batch_candidate_mzs = candidate_mzs[candidate_mask]
        local_fragment_indices = candidate_fragment_idxs[candidate_mask] - first_fragment_index
        if tol_per_sample is None:
            channel_match = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
        else:
            per_peak_tolerance = tol_per_sample[batch_index] * th.maximum(
                batch_true_mzs,
                min_mz_per_sample[batch_index],
            )
            channel_match = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=0.0,
                relative=False,
                tol_per_true=per_peak_tolerance,
            )
        fragment_match_counts = fragment_logprobs.new_zeros(
            (batch_true_mzs.numel(), fragment_indices.numel())
        )
        fragment_match_counts.scatter_add_(
            1,
            local_fragment_indices.reshape(1, -1).expand(batch_true_mzs.numel(), -1),
            channel_match.to(fragment_match_counts),
        )
        fragment_match = fragment_match_counts > 0
        aliases_per_peak = fragment_match.sum(dim=1, keepdim=True).clamp_min(1)
        assignments = fragment_match.to(fragment_logprobs) / aliases_per_peak
        target = (batch_true_probabilities[:, None] * assignments).sum(dim=0)
        # Compare the distribution among covered fragments, not the amount of the
        # spectrum covered by the proposal set. Otherwise changing coverage across CE
        # introduces a uniform bias into every observed fragment delta.
        target_total = target.sum()
        target_available.append(target_total > 0)
        target = target / target_total.clamp_min(th.finfo(target.dtype).tiny)
        targets.append(target)
        predictions.append(fragment_logprobs[fragment_indices].exp())

    # 2. Compare adjacent-energy deltas for the same local fragment identity.
    losses = fragment_logprobs.new_zeros(batch_size)
    eligible_endpoints = fragment_logprobs.new_zeros(batch_size)
    endpoint_counts = fragment_logprobs.new_zeros(batch_size)
    for group_id in th.unique(group_ids).tolist():
        sample_indices = th.where(group_ids == group_id)[0]
        if sample_indices.numel() < 2:
            continue
        sample_indices = sample_indices[th.argsort(ce_values[sample_indices])]
        for low_tensor, high_tensor in zip(sample_indices[:-1], sample_indices[1:]):
            low = int(low_tensor.item())
            high = int(high_tensor.item())
            if targets[low].shape != targets[high].shape:
                raise ValueError("shared proposal spectra must contain the same fragment count")
            if not bool((target_available[low] & target_available[high]).item()):
                continue
            active = (targets[low] + targets[high]) > 0
            if not active.any():
                continue
            predicted_delta = predictions[high][active] - predictions[low][active]
            observed_delta = targets[high][active] - targets[low][active]
            pair_loss = (predicted_delta - observed_delta).square().mean()
            losses[low] = losses[low] + 0.5 * pair_loss
            losses[high] = losses[high] + 0.5 * pair_loss
            endpoint_counts[low] += 1
            endpoint_counts[high] += 1
            eligible_endpoints[low] = 1
            eligible_endpoints[high] = 1
    losses = losses / endpoint_counts.clamp_min(1)
    return losses, eligible_endpoints


class MassShatterPL(SpectrumPL):
    """Train MASShatter with SpectrumPL and FraGNNet's CE-pair objective.

    Attributes:
        model: The wrapped :class:`MassShatterModel`.
        loss_fn: Closure built in :meth:`_setup_loss_fn`; called with the batch merged
            with the model's output dictionary, returning per-spectrum vectors that
            SpectrumPL reduces.
        loss_names: Keys of ``loss_fn``'s output that get logged every step.
    """

    hparams: Any

    def _setup_model(self) -> None:
        """Instantiate ``MassShatterModel`` from the flat hparam namespace.

        Raises:
            ValueError: If ``compile`` is on, or the spectrum config does not provide
                the precursor type MASShatter needs to compute adduct-dependent m/z.
        """
        legacy_oos_enabled = (
            self.hparams.predict_oos
            or self.hparams.oos_loss
            or (
                "mass_shatter_learned_oos" in self.hparams
                and self.hparams["mass_shatter_learned_oos"]
            )
        )
        if legacy_oos_enabled:
            logger.warning(
                "Ignoring legacy OOS settings for MASShatter; spectra are normalized "
                "over generated candidates"
            )
        self.hparams.predict_oos = False
        self.hparams.oos_loss = False
        if "mass_shatter_learned_oos" in self.hparams:
            self.hparams["mass_shatter_learned_oos"] = False
        self.hparams.auxiliary_scores = [
            score
            for score in self.hparams.auxiliary_scores
            if score not in {"true_oos_prob", "true_oos_e"}
        ]

        # Candidate selection runs Python/CPU code inside forward, which dynamo cannot
        # trace; compiling would graph-break every batch.
        if self.hparams.compile:
            raise ValueError(
                "MASShatter candidate selection is CPU/Python code; set compile: False"
            )
        if not self.hparams.spec_params["prec_type"]:
            raise ValueError("MASShatter requires spec_params.prec_type: True")
        node_feature_dim, edge_feature_dim = get_mol_feats_sizes(
            self.hparams.mol_params["pyg_node_feats"],
            self.hparams.mol_params["pyg_edge_feats"],
            self.hparams.mol_params["pyg_pe_embed_k"],
        )
        # An empty vocabulary tells the context encoder to skip that embedding entirely,
        # so a disabled metadata field costs no parameters.
        instrument_types = (
            self.hparams.spec_params["inst_types"] if self.hparams.spec_params["inst_type"] else []
        )
        fragmentation_modes = (
            self.hparams.spec_params["frag_modes"] if self.hparams.spec_params["frag_mode"] else []
        )
        self.model = MassShatterModel(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            precursor_hidden_dim=self.hparams.mol_hidden_size,
            precursor_num_layers=self.hparams.mol_num_layers,
            precursor_num_heads=self.hparams.mol_num_heads,
            precursor_gnn_type=self.hparams.mol_gnn_type,
            precursor_normalization=self.hparams.mol_normalization,
            precursor_dropout=self.hparams.mol_dropout,
            context_dim=self.hparams.mass_shatter_context_dim,
            metadata_embedding_dim=self.hparams.mass_shatter_metadata_embedding_dim,
            head_hidden_dim=self.hparams.mass_shatter_head_hidden_dim,
            head_dropout=self.hparams.mass_shatter_head_dropout,
            bond_num_layers=self.hparams.mass_shatter_bond_num_layers,
            proposal_num_mixtures=self.hparams.mass_shatter_proposal_num_mixtures,
            intensity_transformer_dim=(self.hparams.mass_shatter_intensity_transformer_dim),
            intensity_transformer_num_layers=(
                self.hparams.mass_shatter_intensity_transformer_num_layers
            ),
            intensity_transformer_num_heads=(
                self.hparams.mass_shatter_intensity_transformer_num_heads
            ),
            intensity_transformer_feedforward_dim=(
                self.hparams.mass_shatter_intensity_transformer_feedforward_dim
            ),
            intensity_transformer_num_inducing_points=(
                self.hparams.mass_shatter_intensity_transformer_num_inducing_points
            ),
            intensity_atom_decoder_num_layers=(
                self.hparams.mass_shatter_intensity_atom_decoder_num_layers
            ),
            intensity_transformer_dropout=(self.hparams.mass_shatter_intensity_transformer_dropout),
            intensity_output=self.hparams.mass_shatter_intensity_output,
            use_mask_residuals=self.hparams.mass_shatter_use_mask_residuals,
            use_base_probability_feature=(not self.hparams.mass_shatter_proposal_anchored),
            return_base_spectrum=(
                self.hparams.mass_shatter_log_base_ce
                or self.hparams.mass_shatter_base_ce_weight > 0.0
            ),
            detach_base_ion_h=self.hparams.mass_shatter_detach_base_ion_h,
            proposal_selection_only=self.hparams.mass_shatter_proposal_selection_only,
            separate_proposal_encoder=(self.hparams.mass_shatter_separate_proposal_encoder),
            share_proposal_across_ev=self.hparams.mass_shatter_share_proposal_across_ev,
            separate_ion_mode_proposal_head=(
                self.hparams.mass_shatter_separate_ion_mode_proposal_head
            ),
            ion_mode_proposal_binary_logits=(
                self.hparams.mass_shatter_ion_mode_proposal_binary_logits
            ),
            min_ion_mode_relative_probability=(
                self.hparams.mass_shatter_min_ion_mode_relative_probability
            ),
            separate_ion_h_support_head=(self.hparams.mass_shatter_separate_ion_h_support_head),
            ion_h_support_binary_logits=(
                self.hparams.mass_shatter_ion_h_support_binary_logits
            ),
            min_ion_h_relative_probability=(
                self.hparams.mass_shatter_min_ion_h_relative_probability
            ),
            apply_channel_selection_in_training=(
                self.hparams.mass_shatter_apply_channel_selection_in_training
            ),
            precursor_types=self.hparams.spec_params["prec_types"],
            instrument_types=instrument_types,
            fragmentation_modes=fragmentation_modes,
            int_embedder=self.hparams.int_embedder,
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_size=self.hparams.ce_insert_size,
            use_nce=self.hparams.spec_params["nce"],
            use_ace=self.hparams.spec_params["ace"],
            nce_max=self.hparams.nce_max,
            nce_mean=self.hparams.nce_mean,
            nce_std=self.hparams.nce_std,
            ace_max=self.hparams.ace_max,
            ace_mean=self.hparams.ace_mean,
            ace_std=self.hparams.ace_std,
            max_masks=(
                self.hparams.mass_shatter_unique_fragment_target
                if self.hparams.mass_shatter_unique_fragment_mode
                else self.hparams.mass_shatter_selected_mask_limit
            ),
            unique_fragment_mode=self.hparams.mass_shatter_unique_fragment_mode,
            max_enumerated_masks=self.hparams.mass_shatter_ranked_mask_budget,
            mask_selection_threads=self.hparams.mass_shatter_mask_selection_threads,
            min_mask_relative_probability=(self.hparams.mass_shatter_min_mask_relative_probability),
            min_fragment_relative_probability=(
                self.hparams.mass_shatter_min_fragment_relative_probability
            ),
            include_single_bond_cuts=(self.hparams.mass_shatter_include_single_bond_cuts),
            max_h_transfer=self.hparams.mass_shatter_max_h_transfer,
            include_cmf=self.hparams.mass_shatter_include_cmf,
            charge_pooling=self.hparams.mass_shatter_charge_pooling,
            use_smarts_templates=self.hparams.mass_shatter_smarts_templates,
        )
        self._check_ce_params()

    def _setup_loss_fn(self) -> None:
        """Configure legacy CE or a direct clustered peak-list objective.

        Direct spectral losses use m/z-only closest or Hungarian correspondence and
        never evaluate a Gaussian density. The legacy cross-entropy branch is retained
        unchanged as the control. This method also registers candidate-generation
        diagnostics and compatible auxiliary losses.

        Raises:
            ValueError: If the configured loss/spectrum options are outside the subset
                this training path supports.
        """
        accepted_loss_types = {
            "cross_entropy",
            "cosine_distance",
            "jss_distance",
            "kl_distance",
        }
        if self.hparams.loss_type not in accepted_loss_types:
            raise ValueError(f"MASShatter loss_type must be one of {sorted(accepted_loss_types)}")
        if self.hparams.mass_shatter_intensity_output not in {
            "probability",
            "independent",
        }:
            raise ValueError("mass_shatter_intensity_output must be probability or independent")
        independent_output = self.hparams.mass_shatter_intensity_output == "independent"
        if independent_output and self.hparams.loss_type == "cross_entropy":
            raise ValueError("independent intensity output requires a direct spectral loss")
        if self.hparams.loss_type != "cross_entropy" and (
            self.hparams.mass_shatter_base_ce_weight != 0.0 or self.hparams.mass_shatter_log_base_ce
        ):
            raise ValueError("direct spectral losses do not support the base Gaussian CE")
        if independent_output and (
            self.hparams.mass_shatter_base_ce_weight != 0.0
            or self.hparams.mass_shatter_log_base_ce
            or self.hparams.ce_pair_loss_weight != 0.0
            or self.hparams.precursor_pair_loss_weight != 0.0
            or self.hparams.cross_ev_fragment_loss_weight != 0.0
        ):
            raise ValueError(
                "independent intensity output does not support probability-based CE auxiliaries"
            )
        if self.hparams.mass_shatter_peak_matching not in {"closest", "hungarian"}:
            raise ValueError("mass_shatter_peak_matching must be closest or hungarian")
        if self.hparams.mass_shatter_channel_support_attribution not in {
            "gaussian_posterior",
            "hungarian",
        }:
            raise ValueError(
                "mass_shatter_channel_support_attribution must be gaussian_posterior or hungarian"
            )
        if not isinstance(
            self.hparams.mass_shatter_channel_support_use_intensity_prior,
            bool,
        ):
            raise ValueError("mass_shatter_channel_support_use_intensity_prior must be boolean")
        proposal_ints_transform = self.hparams.mass_shatter_proposal_ints_transform
        proposal_ints_transform_func = None
        if proposal_ints_transform != "same":
            proposal_ints_transform_func = get_ints_transform_func(proposal_ints_transform)
        if (
            not math.isfinite(self.hparams.mass_shatter_intensity_entropy_match_weight)
            or self.hparams.mass_shatter_intensity_entropy_match_weight < 0.0
        ):
            raise ValueError(
                "mass_shatter_intensity_entropy_match_weight must be finite and nonnegative"
            )
        if (
            not math.isfinite(self.hparams.mass_shatter_kl_epsilon)
            or not 0.0 < self.hparams.mass_shatter_kl_epsilon < 1.0
        ):
            raise ValueError("mass_shatter_kl_epsilon must be finite and in (0, 1)")
        if self.hparams.loss_type == "kl_distance" and self.hparams.precision in {
            16,
            "16",
            "16-mixed",
        }:
            raise ValueError(
                "kl_distance does not support FP16 mixed precision; use precision 32 or bf16"
            )
        if not self.hparams.sparse_cosine_similarity or not self.hparams.sum_ints:
            raise ValueError("MASShatter requires sparse_cosine_similarity and sum_ints")
        if self.hparams.ce_pair_min_endpoint_reliability is not None:
            raise ValueError("ce_pair_min_endpoint_reliability is not implemented")
        # Predictions are exact monoisotopic m/z values, so targets stay unbinned and
        # are matched against predictions by instrument tolerance instead.
        self.binned_loss = False
        if (
            not math.isfinite(self.hparams.mass_shatter_base_ce_weight)
            or self.hparams.mass_shatter_base_ce_weight < 0.0
        ):
            raise ValueError("mass_shatter_base_ce_weight must be finite and nonnegative")
        if (
            not math.isfinite(self.hparams.mass_shatter_proposal_loss_weight)
            or self.hparams.mass_shatter_proposal_loss_weight < 0.0
        ):
            raise ValueError("mass_shatter_proposal_loss_weight must be finite and nonnegative")
        support_head_configs = (
            (
                "ion_mode",
                self.hparams.mass_shatter_separate_ion_mode_proposal_head,
                self.hparams.mass_shatter_ion_mode_proposal_binary_logits,
                self.hparams.mass_shatter_ion_mode_proposal_loss_weight,
                self.hparams.mass_shatter_min_ion_mode_relative_probability,
            ),
            (
                "ion_h",
                self.hparams.mass_shatter_separate_ion_h_support_head,
                self.hparams.mass_shatter_ion_h_support_binary_logits,
                self.hparams.mass_shatter_ion_h_support_loss_weight,
                self.hparams.mass_shatter_min_ion_h_relative_probability,
            ),
        )
        for name, enabled, binary_logits, loss_weight, probability_floor in support_head_configs:
            if not isinstance(enabled, bool):
                raise ValueError(f"the separate {name} head switch must be boolean")
            if not isinstance(binary_logits, bool):
                raise ValueError(f"the {name} binary-logit switch must be boolean")
            if not math.isfinite(loss_weight) or loss_weight < 0.0:
                raise ValueError(f"the {name} support loss weight must be nonnegative")
            if enabled != (loss_weight > 0.0):
                raise ValueError(
                    f"the separate {name} support head and its positive loss weight "
                    "must be enabled together"
                )
            if binary_logits != (loss_weight > 0.0):
                raise ValueError(
                    f"the {name} binary-logit semantics and its positive BCE loss "
                    "weight must be enabled together"
                )
            if not math.isfinite(probability_floor) or not 0.0 <= probability_floor <= 1.0:
                raise ValueError(f"mass_shatter_min_{name}_relative_probability must be in [0, 1]")
        if not isinstance(self.hparams.mass_shatter_apply_channel_selection_in_training, bool):
            raise ValueError("mass_shatter_apply_channel_selection_in_training must be boolean")
        if not isinstance(self.hparams.mass_shatter_channel_diagnostics, bool):
            raise ValueError("mass_shatter_channel_diagnostics must be boolean")
        if self.hparams.mass_shatter_channel_diagnostics and not (
            self.hparams.mass_shatter_unique_fragment_mode
            and self.hparams.mass_shatter_proposal_selection_only
        ):
            raise ValueError(
                "mass_shatter_channel_diagnostics requires unique-fragment "
                "proposal-selection-only mode"
            )
        if self.hparams.mass_shatter_apply_channel_selection_in_training and not any(
            enabled and loss_weight > 0.0 and probability_floor > 0.0
            for _, enabled, _, loss_weight, probability_floor in support_head_configs
        ):
            raise ValueError(
                "training-time channel selection requires a supervised support head "
                "with a positive selection floor"
            )
        if not isinstance(self.hparams.mass_shatter_proposal_barcode_target, bool):
            raise ValueError("mass_shatter_proposal_barcode_target must be boolean")
        if (
            self.hparams.mass_shatter_proposal_barcode_target
            and self.hparams.mass_shatter_proposal_loss_weight == 0.0
        ):
            raise ValueError(
                "mass_shatter_proposal_barcode_target requires a positive "
                "mass_shatter_proposal_loss_weight"
            )
        if (
            self.hparams.mass_shatter_proposal_selection_only
            and self.hparams.mass_shatter_proposal_loss_weight == 0.0
        ):
            raise ValueError(
                "mass_shatter_proposal_selection_only requires a positive "
                "mass_shatter_proposal_loss_weight"
            )
        if (
            not self.hparams.mass_shatter_proposal_selection_only
            and self.hparams.mass_shatter_proposal_loss_weight != 0.0
        ):
            raise ValueError(
                "mass_shatter_proposal_loss_weight is only supported when "
                "mass_shatter_proposal_selection_only is true"
            )
        if (
            self.hparams.mass_shatter_proposal_selection_only
            and self.hparams.mass_shatter_base_ce_weight != 0.0
        ):
            raise ValueError(
                "mass_shatter_proposal_selection_only uses the H-independent proposal loss; "
                "set mass_shatter_base_ce_weight to 0"
            )
        if (
            self.hparams.mass_shatter_proposal_anchored
            and self.hparams.mass_shatter_base_ce_weight == 0.0
            and not (
                self.hparams.mass_shatter_proposal_selection_only
                and self.hparams.mass_shatter_proposal_loss_weight > 0.0
            )
        ):
            raise ValueError(
                "mass_shatter_proposal_anchored requires a positive proposal supervision loss"
            )
        proposal_loss_enabled = self.hparams.mass_shatter_proposal_loss_weight > 0.0
        ion_mode_proposal_loss_enabled = (
            self.hparams.mass_shatter_ion_mode_proposal_loss_weight > 0.0
        )
        ion_h_support_loss_enabled = self.hparams.mass_shatter_ion_h_support_loss_weight > 0.0
        channel_targets_enabled = (
            self.hparams.mass_shatter_channel_diagnostics
            or ion_mode_proposal_loss_enabled
            or ion_h_support_loss_enabled
        )
        cross_ev_fragment_loss_enabled = self.hparams.cross_ev_fragment_loss_weight > 0.0
        if (
            not math.isfinite(self.hparams.cross_ev_fragment_loss_weight)
            or self.hparams.cross_ev_fragment_loss_weight < 0.0
        ):
            raise ValueError("cross_ev_fragment_loss_weight must be finite and nonnegative")
        if cross_ev_fragment_loss_enabled and not (
            self.hparams.mass_shatter_unique_fragment_mode
            and self.hparams.mass_shatter_share_proposal_across_ev
        ):
            raise ValueError(
                "cross_ev_fragment_loss_weight requires unique-fragment mode and "
                "mass_shatter_share_proposal_across_ev"
            )
        base_ce_enabled = (
            self.hparams.mass_shatter_log_base_ce or self.hparams.mass_shatter_base_ce_weight > 0.0
        )
        sparse_ce_fn = None
        if self.hparams.loss_type == "cross_entropy":
            sparse_ce_fn = get_sparse_cross_entropy_fn(
                dist=self.hparams.output_distribution,
                vectorized=self.hparams.loss_vectorized,
                tolerance=self.tolerance,
                relative=self.relative,
                tolerance_min_mz=self.tolerance_min_mz,
                oos_tolerance_multiple=1.0,
                gaussian_renormalize=self.hparams.gaussian_renormalize,
                pm_tolerance_multiple=self.hparams.pm_tolerance_multiple,
                loss_batch_size=self.hparams.loss_batch_size,
            )

        # The CE-pair term only applies to configured instrument/adduct combinations.
        # Translate those names into the sorted label indices the batch carries (the
        # same sort order the model uses), once at setup rather than per batch.
        instrument_types = sorted(self.hparams.spec_params["inst_types"])
        precursor_types = sorted(self.hparams.spec_params["prec_types"])
        eligible_instrument_indices = th.empty(0, dtype=th.long)
        eligible_precursor_indices = th.empty(0, dtype=th.long)
        if self.hparams.ce_pair_loss_weight != 0.0:
            eligible_instrument_indices = th.tensor(
                [
                    instrument_types.index(instrument_type)
                    for instrument_type in self.hparams.ce_pair_eligible_inst_types
                ],
                dtype=th.long,
            )
            eligible_precursor_indices = th.tensor(
                [
                    precursor_types.index(precursor_type)
                    for precursor_type in self.hparams.ce_pair_eligible_prec_types
                ],
                dtype=th.long,
            )

        # SpectrumPL calls this with the training dict merged with the model's output
        # dict, so every key MassShatterModel returns has to be named here or absorbed
        # by **kwargs. The diagnostics are named explicitly because they are re-emitted
        # as metrics below.
        def loss_fn(
            true_mzs: th.Tensor,
            true_logprobs: th.Tensor,
            true_batch_idxs: th.Tensor,
            pred_mzs: th.Tensor,
            pred_batch_idxs: th.Tensor,
            retained_mask_probability: th.Tensor,
            selected_mask_count: th.Tensor,
            component_pathway_count: th.Tensor,
            mean_mask_cut_count: th.Tensor,
            proposal_single_cut_fraction: th.Tensor,
            proposal_mixture_entropy: th.Tensor,
            proposal_mixture_effective_count: th.Tensor,
            proposal_mixture_top1_weight: th.Tensor,
            unique_component_count: th.Tensor,
            fragment_count_before_cutoff: th.Tensor,
            fragment_count_after_cutoff: th.Tensor,
            retained_fragment_probability: th.Tensor,
            fragment_cutoff_pruned_fraction: th.Tensor,
            components_per_mask: th.Tensor,
            unique_components_per_mask: th.Tensor,
            latent_pathways_per_unique_fragment: th.Tensor,
            unique_fragment_target_fill_fraction: th.Tensor,
            unique_fragment_target_reached: th.Tensor,
            unique_fragment_effective_count: th.Tensor,
            unique_fragment_top1_mass_fraction: th.Tensor,
            duplicate_pathway_probability_fraction: th.Tensor,
            productive_mask_fraction: th.Tensor,
            duplicate_component_pathway_fraction: th.Tensor,
            mask_cut_fraction_0: th.Tensor,
            mask_cut_fraction_1: th.Tensor,
            mask_cut_fraction_2: th.Tensor,
            mask_cut_fraction_3: th.Tensor,
            mask_cut_fraction_4: th.Tensor,
            mask_cut_fraction_5: th.Tensor,
            mask_cut_fraction_6plus: th.Tensor,
            mask_cut_fraction_4plus: th.Tensor,
            candidate_input_transfer_batch_ms: th.Tensor,
            candidate_mask_selection_batch_ms: th.Tensor,
            candidate_topology_batch_ms: th.Tensor,
            candidate_assembly_batch_ms: th.Tensor,
            candidate_output_transfer_batch_ms: th.Tensor,
            candidate_total_batch_ms: th.Tensor,
            candidate_total_molecule_ms: th.Tensor,
            **kwargs,
        ) -> dict[str, th.Tensor]:
            """Return per-spectrum loss terms and metrics, each of shape ``(batch_size,)``.

            Every value stays a per-spectrum vector; SpectrumPL owns the weighted batch
            reduction, so reducing here would double-count spectra from merged groups.
            """
            tolerance_per_sample = kwargs.get("tol_per_sample")
            minimum_mz_per_sample = kwargs.get("min_mz_per_sample")
            batch_size = int(kwargs["batch_size"])
            pred_logprobs = kwargs.get("pred_logprobs")
            pred_intensities = kwargs.get("pred_intensities")
            if pred_intensities is None:
                if pred_logprobs is None:
                    raise ValueError("loss input is missing predicted intensity values")
                pred_intensities = pred_logprobs.exp()
            true_intensities = kwargs.get("true_intensities", true_logprobs.exp())
            proposal_true_intensities = true_intensities
            proposal_true_logprobs = true_logprobs
            if proposal_ints_transform_func is not None:
                proposal_true_intensities = self.ints_untransform_func(
                    true_intensities,
                    true_batch_idxs,
                )
                proposal_true_intensities = proposal_ints_transform_func(proposal_true_intensities)
                proposal_true_intensities = self.ints_normalize_func(
                    proposal_true_intensities,
                    true_batch_idxs,
                )
                proposal_true_logprobs = proposal_true_intensities.clamp_min(
                    th.finfo(proposal_true_intensities.dtype).tiny
                ).log()
            prediction_reference = pred_logprobs if pred_logprobs is not None else pred_intensities
            spectrum_ce = None
            entropy_match = None
            if self.hparams.loss_type == "cross_entropy":
                if pred_logprobs is None:
                    raise ValueError("cross_entropy requires probability output")
                if sparse_ce_fn is None:
                    raise RuntimeError("sparse CE was not configured")
                # The legacy control retains its Gaussian/peak-marginal behavior.
                spectrum_ce, _, _, _ = sparse_ce_fn(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_logprobs,
                    pred_batch_idxs,
                    pred_logprobs.new_zeros(batch_size),
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                )
                primary_loss = spectrum_ce
                candidate_wrecall = _candidate_weighted_recall(
                    true_mzs=true_mzs,
                    true_ints=proposal_true_intensities,
                    true_batch_idxs=true_batch_idxs,
                    candidate_mzs=kwargs["candidate_mzs_by_cut"],
                    candidate_batch_idxs=kwargs["candidate_batch_idxs_by_cut"],
                    batch_size=batch_size,
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                )
            else:
                primary_loss, entropy_match = sparse_peak_list_distance(
                    true_mzs,
                    true_intensities,
                    true_batch_idxs,
                    pred_mzs,
                    pred_intensities,
                    pred_batch_idxs,
                    objective=self.hparams.loss_type,
                    matching=self.hparams.mass_shatter_peak_matching,
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                    decimals=4,
                    kl_epsilon=self.hparams.mass_shatter_kl_epsilon,
                )
                coverage = _candidate_coverage_by_cut_count(
                    true_mzs=true_mzs,
                    true_ints=proposal_true_intensities,
                    true_batch_idxs=true_batch_idxs,
                    candidate_mzs=kwargs["candidate_mzs_by_cut"],
                    candidate_batch_idxs=kwargs["candidate_batch_idxs_by_cut"],
                    candidate_cut_counts=kwargs["candidate_cut_counts"],
                    batch_size=batch_size,
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                )
                candidate_wrecall = coverage["candidate_wrecall"]
            loss = self.hparams.primary_loss_weight * primary_loss
            if entropy_match is not None:
                loss = (
                    loss + self.hparams.mass_shatter_intensity_entropy_match_weight * entropy_match
                )
            support_metric_names = (
                "fragment_precision",
                "fragment_weighted_precision",
                "ion_mode_support_positive_rate",
                "ion_h_support_positive_rate",
                *_ION_H_REQUIRED_K_METRIC_NAMES,
            )
            support_metrics = {
                name: prediction_reference.new_zeros(batch_size) for name in support_metric_names
            }
            channel_targets = None
            if channel_targets_enabled:
                channel_targets = _observational_channel_targets(
                    true_mzs=true_mzs,
                    true_batch_idxs=true_batch_idxs,
                    channel_mzs=kwargs["ion_h_dense_mzs"],
                    channel_log_weights=kwargs["ion_h_dense_log_weights"],
                    chemical_valid_channels=kwargs["ion_h_chemical_valid_channels"],
                    fragment_batch_idxs=kwargs["ion_h_fragment_batch_idxs"],
                    batch_size=batch_size,
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                    attribution=(self.hparams.mass_shatter_channel_support_attribution),
                    use_intensity_prior=(
                        self.hparams.mass_shatter_channel_support_use_intensity_prior
                    ),
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                )
                support_metrics = _channel_support_metrics(
                    channel_targets=channel_targets,
                    chemical_valid_channels=kwargs["ion_h_chemical_valid_channels"],
                    fragment_batch_idxs=kwargs["ion_h_fragment_batch_idxs"],
                    batch_size=batch_size,
                    fragment_logprobs=kwargs["proposal_fragment_logprobs"],
                )
                support_metrics.update(
                    _ion_h_required_k_metrics(
                        channel_targets=channel_targets,
                        support_logits=kwargs["ion_h_support_logits"],
                        chemical_valid_channels=kwargs["ion_h_chemical_valid_channels"],
                        fragment_batch_idxs=kwargs["ion_h_fragment_batch_idxs"],
                        batch_size=batch_size,
                    )
                )
            if ion_h_support_loss_enabled:
                if channel_targets is None:
                    raise ValueError("ion/H support loss requires dense channel tensors")
                ion_h_support_loss = _balanced_binary_support_loss(
                    kwargs["ion_h_support_logits"],
                    channel_targets,
                    kwargs["ion_h_chemical_valid_channels"],
                    kwargs["ion_h_fragment_batch_idxs"],
                    batch_size,
                )
                loss = loss + (
                    self.hparams.mass_shatter_ion_h_support_loss_weight * ion_h_support_loss
                )
            if ion_mode_proposal_loss_enabled:
                if channel_targets is None:
                    raise ValueError("ion-mode proposal loss requires dense channel tensors")
                ion_mode_proposal_loss = _ion_mode_support_loss(
                    kwargs["ion_mode_proposal_logits"],
                    channel_targets.sum(dim=2).clamp(max=1.0),
                    kwargs["ion_h_chemical_valid_channels"].any(dim=2),
                    kwargs["ion_h_fragment_batch_idxs"],
                    batch_size,
                )
                loss = loss + (
                    self.hparams.mass_shatter_ion_mode_proposal_loss_weight * ion_mode_proposal_loss
                )
            if proposal_loss_enabled:
                proposal_loss_kwargs = {
                    "true_mzs": true_mzs,
                    "true_logprobs": proposal_true_logprobs,
                    "true_batch_idxs": true_batch_idxs,
                    "candidate_mzs": kwargs["proposal_candidate_mzs"],
                    "candidate_fragment_idxs": kwargs["proposal_candidate_fragment_idxs"],
                    "fragment_logprobs": kwargs["proposal_fragment_logprobs"],
                    "fragment_batch_idxs": kwargs["proposal_fragment_batch_idxs"],
                    "batch_size": batch_size,
                    "tolerance": self.tolerance,
                    "relative": self.relative,
                    "tolerance_min_mz": self.tolerance_min_mz,
                    "tol_per_sample": tolerance_per_sample,
                    "min_mz_per_sample": minimum_mz_per_sample,
                    "barcode_target": self.hparams.mass_shatter_proposal_barcode_target,
                }
                if self.hparams.mass_shatter_share_proposal_across_ev:
                    proposal_loss = _shared_group_proposal_coverage_loss(
                        **proposal_loss_kwargs,
                        group_ids=kwargs["proposal_group_ids"],
                    )
                else:
                    proposal_loss = _proposal_coverage_loss(**proposal_loss_kwargs)
                loss = loss + self.hparams.mass_shatter_proposal_loss_weight * proposal_loss
            if cross_ev_fragment_loss_enabled:
                cross_ev_fragment_loss = pred_logprobs.new_zeros(batch_size)
                cross_ev_fragment_endpoints = pred_logprobs.new_zeros(batch_size)
                if "cross_ev_group_ids" in kwargs:
                    cross_ev_fragment_loss, cross_ev_fragment_endpoints = (
                        _cross_ev_fragment_delta_loss(
                            true_mzs=true_mzs,
                            true_logprobs=true_logprobs,
                            true_batch_idxs=true_batch_idxs,
                            candidate_mzs=kwargs["cross_ev_candidate_mzs"],
                            candidate_fragment_idxs=kwargs["cross_ev_candidate_fragment_idxs"],
                            fragment_logprobs=kwargs["cross_ev_fragment_logprobs"],
                            fragment_batch_idxs=kwargs["cross_ev_fragment_batch_idxs"],
                            group_ids=kwargs["cross_ev_group_ids"],
                            ce_values=kwargs["cross_ev_ce_values"],
                            batch_size=batch_size,
                            tolerance=self.tolerance,
                            relative=self.relative,
                            tolerance_min_mz=self.tolerance_min_mz,
                            tol_per_sample=tolerance_per_sample,
                            min_mz_per_sample=minimum_mz_per_sample,
                        )
                    )
                loss = loss + self.hparams.cross_ev_fragment_loss_weight * cross_ev_fragment_loss
            if base_ce_enabled:
                base_pred_logprobs = kwargs["base_pred_logprobs"]
                base_spec_ce, _, _, _ = sparse_ce_fn(
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    base_pred_logprobs,
                    pred_batch_idxs,
                    base_pred_logprobs.new_zeros(batch_size),
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                )
                base_primary_loss = base_spec_ce
                loss = loss + self.hparams.mass_shatter_base_ce_weight * base_primary_loss
            # The candidate-generation values below are monitoring only and detached:
            # candidate_wrecall is how much true intensity the enumerated masks could
            # explain at all (the ceiling on this batch), retained_mask_probability is
            # how much of the Bernoulli mask distribution survived truncation, and the
            # counts/timings track the cost of candidate generation.
            loss_values = {
                "primary_loss": primary_loss,
                "candidate_wrecall": candidate_wrecall,
                "retained_mask_probability": retained_mask_probability.detach(),
                "selected_mask_count": selected_mask_count.detach().to(prediction_reference),
                "component_pathway_count": component_pathway_count.detach().to(
                    prediction_reference
                ),
                "mean_mask_cut_count": mean_mask_cut_count.detach().to(prediction_reference),
                "proposal_single_cut_fraction": (
                    proposal_single_cut_fraction.detach().to(prediction_reference)
                ),
                "proposal_mixture_entropy": (
                    proposal_mixture_entropy.detach().to(prediction_reference)
                ),
                "proposal_mixture_effective_count": (
                    proposal_mixture_effective_count.detach().to(prediction_reference)
                ),
                "proposal_mixture_top1_weight": (
                    proposal_mixture_top1_weight.detach().to(prediction_reference)
                ),
                "unique_component_count": unique_component_count.detach().to(prediction_reference),
                "fragment_count_before_cutoff": (
                    fragment_count_before_cutoff.detach().to(prediction_reference)
                ),
                "fragment_count_after_cutoff": (
                    fragment_count_after_cutoff.detach().to(prediction_reference)
                ),
                "retained_fragment_probability": (
                    retained_fragment_probability.detach().to(prediction_reference)
                ),
                "fragment_cutoff_pruned_fraction": (
                    fragment_cutoff_pruned_fraction.detach().to(prediction_reference)
                ),
                "components_per_mask": components_per_mask.detach().to(prediction_reference),
                "unique_components_per_mask": (
                    unique_components_per_mask.detach().to(prediction_reference)
                ),
                "latent_pathways_per_unique_fragment": (
                    latent_pathways_per_unique_fragment.detach().to(prediction_reference)
                ),
                "unique_fragment_target_fill_fraction": (
                    unique_fragment_target_fill_fraction.detach().to(prediction_reference)
                ),
                "unique_fragment_target_reached": (
                    unique_fragment_target_reached.detach().to(prediction_reference)
                ),
                "unique_fragment_effective_count": (
                    unique_fragment_effective_count.detach().to(prediction_reference)
                ),
                "unique_fragment_top1_mass_fraction": (
                    unique_fragment_top1_mass_fraction.detach().to(prediction_reference)
                ),
                "duplicate_pathway_probability_fraction": (
                    duplicate_pathway_probability_fraction.detach().to(prediction_reference)
                ),
                "productive_mask_fraction": (
                    productive_mask_fraction.detach().to(prediction_reference)
                ),
                "duplicate_component_pathway_fraction": (
                    duplicate_component_pathway_fraction.detach().to(prediction_reference)
                ),
                "mask_cut_fraction_0": mask_cut_fraction_0.detach().to(prediction_reference),
                "mask_cut_fraction_1": mask_cut_fraction_1.detach().to(prediction_reference),
                "mask_cut_fraction_2": mask_cut_fraction_2.detach().to(prediction_reference),
                "mask_cut_fraction_3": mask_cut_fraction_3.detach().to(prediction_reference),
                "mask_cut_fraction_4": mask_cut_fraction_4.detach().to(prediction_reference),
                "mask_cut_fraction_5": mask_cut_fraction_5.detach().to(prediction_reference),
                "mask_cut_fraction_6plus": (
                    mask_cut_fraction_6plus.detach().to(prediction_reference)
                ),
                "mask_cut_fraction_4plus": (
                    mask_cut_fraction_4plus.detach().to(prediction_reference)
                ),
                "candidate_input_transfer_batch_ms": (candidate_input_transfer_batch_ms.detach()),
                "candidate_mask_selection_batch_ms": (candidate_mask_selection_batch_ms.detach()),
                "candidate_topology_batch_ms": candidate_topology_batch_ms.detach(),
                "candidate_assembly_batch_ms": candidate_assembly_batch_ms.detach(),
                "candidate_output_transfer_batch_ms": (candidate_output_transfer_batch_ms.detach()),
                "candidate_total_batch_ms": candidate_total_batch_ms.detach(),
                "candidate_total_molecule_ms": candidate_total_molecule_ms.detach(),
            }
            loss_values.update(
                {
                    name: value.detach().to(prediction_reference)
                    for name, value in support_metrics.items()
                }
            )
            for diagnostic_name in (
                "valid_ion_h_count",
                "retained_ion_h_count",
                "retained_ion_h_probability",
                "retained_joint_channel_intensity_fraction",
                "ion_h_cutoff_pruned_fraction",
                "valid_ion_mode_count",
                "retained_ion_mode_count",
                "retained_ion_mode_probability",
                "ion_mode_cutoff_pruned_fraction",
            ):
                diagnostic = kwargs[diagnostic_name]
                loss_values[diagnostic_name] = diagnostic.detach().to(prediction_reference)
            if spectrum_ce is not None:
                loss_values["spec_ce"] = spectrum_ce
            else:
                loss_values["spec_peak_distance"] = primary_loss
                loss_values["spec_entropy_match"] = entropy_match
            if proposal_loss_enabled:
                loss_values["proposal_loss"] = proposal_loss
            if ion_h_support_loss_enabled:
                loss_values["ion_h_support_loss"] = ion_h_support_loss
            if ion_mode_proposal_loss_enabled:
                loss_values["ion_mode_proposal_loss"] = ion_mode_proposal_loss
            if cross_ev_fragment_loss_enabled:
                loss_values["cross_ev_fragment_loss"] = cross_ev_fragment_loss
                loss_values["cross_ev_fragment_endpoints"] = cross_ev_fragment_endpoints
            if base_ce_enabled:
                loss_values.update(
                    {
                        "base_spec_ce": base_spec_ce,
                        "base_primary_loss": base_primary_loss,
                        "base_ce_gain": base_spec_ce.detach() - spectrum_ce.detach(),
                    }
                )

            if self.hparams.ce_pair_loss_weight != 0.0:
                # The term is train-only (SpectrumPL sets ``ce_pair_enabled``), but the
                # keys must exist on every split so the logged metrics stay aligned;
                # hence the zero vectors when it is skipped.
                batch_size = int(kwargs["batch_size"])
                pair_loss = pred_logprobs.new_zeros(batch_size)
                eligible_endpoints = pred_logprobs.new_zeros(batch_size)
                if kwargs["ce_pair_enabled"]:
                    # Only spectra from a configured instrument *and* adduct take part;
                    # elsewhere the observed winner switch is too noisy to supervise.
                    instrument_indices = kwargs["spec_inst_type"].reshape(-1)
                    precursor_indices = kwargs["spec_prec_type"].reshape(-1)
                    eligible_samples = th.isin(
                        instrument_indices,
                        eligible_instrument_indices.to(instrument_indices.device),
                    ) & th.isin(
                        precursor_indices,
                        eligible_precursor_indices.to(precursor_indices.device),
                    )
                    # Supervises which of two adjacent-CE spectra ranks the observed
                    # winner higher. It reads the ``ce_pair_*`` peak lists, which
                    # SpectrumPL passes through without the filtering and intensity
                    # transform applied to the tensors the primary CE consumes.
                    pair_loss, eligible_endpoints = ce_switch_pair_bce(
                        true_mzs=kwargs["ce_pair_true_mzs"],
                        true_ints=kwargs["ce_pair_true_ints"],
                        true_batch_idxs=kwargs["ce_pair_true_batch_idxs"],
                        pred_mzs=kwargs["ce_pair_pred_mzs"],
                        pred_logprobs=kwargs["ce_pair_pred_logprobs"],
                        pred_batch_idxs=kwargs["ce_pair_pred_batch_idxs"],
                        group_ids=kwargs["group_id"].reshape(-1),
                        ce_values=kwargs["ce_pair_ce_values"],
                        eligible_samples=eligible_samples,
                        batch_size=batch_size,
                        min_endpoint_confidence=(self.hparams.ce_pair_min_endpoint_confidence),
                        min_contender_relative_intensity=(self.hparams.target_rel_ints_thresh),
                        tol_per_sample=tolerance_per_sample,
                        min_mz_per_sample=minimum_mz_per_sample,
                    )
                loss = loss + self.hparams.ce_pair_loss_weight * pair_loss
                loss_values["ce_pair_loss"] = pair_loss
                loss_values["ce_pair_eligible_endpoints"] = eligible_endpoints
            loss_values["loss"] = loss
            return loss_values

        self.loss_fn = loss_fn
        # Every name here must be a key of loss_values above: SpectrumPL reduces and logs
        # exactly these, so a missing key fails at the first logging step.
        loss_names = [
            "loss",
            "primary_loss",
            "candidate_wrecall",
            "retained_mask_probability",
            "selected_mask_count",
            "component_pathway_count",
            "mean_mask_cut_count",
            "proposal_single_cut_fraction",
            "proposal_mixture_entropy",
            "proposal_mixture_effective_count",
            "proposal_mixture_top1_weight",
            "unique_component_count",
            "fragment_count_before_cutoff",
            "fragment_count_after_cutoff",
            "retained_fragment_probability",
            "fragment_cutoff_pruned_fraction",
            "components_per_mask",
            "unique_components_per_mask",
            "latent_pathways_per_unique_fragment",
            "unique_fragment_target_fill_fraction",
            "unique_fragment_target_reached",
            "unique_fragment_effective_count",
            "unique_fragment_top1_mass_fraction",
            "duplicate_pathway_probability_fraction",
            "productive_mask_fraction",
            "duplicate_component_pathway_fraction",
            "mask_cut_fraction_0",
            "mask_cut_fraction_1",
            "mask_cut_fraction_2",
            "mask_cut_fraction_3",
            "mask_cut_fraction_4",
            "mask_cut_fraction_5",
            "mask_cut_fraction_6plus",
            "mask_cut_fraction_4plus",
            "candidate_input_transfer_batch_ms",
            "candidate_mask_selection_batch_ms",
            "candidate_topology_batch_ms",
            "candidate_assembly_batch_ms",
            "candidate_output_transfer_batch_ms",
            "candidate_total_batch_ms",
            "candidate_total_molecule_ms",
            "valid_ion_h_count",
            "retained_ion_h_count",
            "retained_ion_h_probability",
            "retained_joint_channel_intensity_fraction",
            "ion_h_cutoff_pruned_fraction",
            "valid_ion_mode_count",
            "retained_ion_mode_count",
            "retained_ion_mode_probability",
            "ion_mode_cutoff_pruned_fraction",
        ]
        if channel_targets_enabled:
            loss_names.extend(
                [
                    "fragment_precision",
                    "fragment_weighted_precision",
                    "ion_mode_support_positive_rate",
                    "ion_h_support_positive_rate",
                    *_ION_H_REQUIRED_K_METRIC_NAMES,
                ]
            )
        if self.hparams.loss_type == "cross_entropy":
            loss_names.append("spec_ce")
        else:
            loss_names.extend(["spec_peak_distance", "spec_entropy_match"])
        if self.hparams.ce_pair_loss_weight != 0.0:
            loss_names.extend(["ce_pair_loss", "ce_pair_eligible_endpoints"])
        if proposal_loss_enabled:
            loss_names.append("proposal_loss")
        if ion_h_support_loss_enabled:
            loss_names.append("ion_h_support_loss")
        if ion_mode_proposal_loss_enabled:
            loss_names.append("ion_mode_proposal_loss")
        if cross_ev_fragment_loss_enabled:
            loss_names.extend(["cross_ev_fragment_loss", "cross_ev_fragment_endpoints"])
        if base_ce_enabled:
            loss_names.extend(
                [
                    "base_spec_ce",
                    "base_primary_loss",
                    "base_ce_gain",
                ]
            )
        self.loss_names = loss_names
        self.metric_names.update(loss_names)

    def _setup_metric_fns(self) -> None:
        """Add overall and cut-bucket validation candidate coverage."""
        super()._setup_metric_fns()
        coverage_metric_names = {
            f"candidate_{metric}_cut_{bucket}"
            for bucket in _CUT_COUNT_BUCKETS
            for metric in ("recall", "wrecall", "precision")
        }
        coverage_metric_names.add("candidate_recall")
        for metric_name in coverage_metric_names:
            self.metric_min_max[metric_name] = None
        self.auxiliary_metric_names.update(coverage_metric_names)
        self.metric_names.update(coverage_metric_names)
        self.extra_metric_args.extend(
            [
                "candidate_mzs_by_cut",
                "candidate_batch_idxs_by_cut",
                "candidate_cut_counts",
            ]
        )
        if self.inst_tol_lookup is not None:
            self.batch_metric_args.add("spec_inst_type")
        base_metric_fn = self.metric_fn

        def metric_fn(**kwargs: Any) -> dict[str, th.Tensor]:
            metrics = base_metric_fn(**kwargs)
            tol_per_sample = None
            min_mz_per_sample = None
            spec_inst_type = kwargs.get("spec_inst_type")
            if spec_inst_type is not None and self.inst_tol_lookup is not None:
                inst_type_idxs = spec_inst_type.reshape(-1)
                dtype = kwargs["true_mzs"].dtype
                tol_per_sample = self.inst_tol_lookup[inst_type_idxs].to(dtype=dtype)
                min_mz_per_sample = self.inst_min_mz_lookup[inst_type_idxs].to(dtype=dtype)
            coverage = _candidate_coverage_by_cut_count(
                true_mzs=kwargs["true_mzs"],
                true_ints=kwargs["true_ints"],
                true_batch_idxs=kwargs["true_batch_idxs"],
                candidate_mzs=kwargs["candidate_mzs_by_cut"],
                candidate_batch_idxs=kwargs["candidate_batch_idxs_by_cut"],
                candidate_cut_counts=kwargs["candidate_cut_counts"],
                batch_size=kwargs["true_prec_mzs"].shape[0],
                tolerance=self.tolerance,
                relative=self.relative,
                tolerance_min_mz=self.tolerance_min_mz,
                tol_per_sample=tol_per_sample,
                min_mz_per_sample=min_mz_per_sample,
            )
            # The loss path already emits this key; returning it here would collide
            # when SpectrumPL merges validation/test auxiliary metrics.
            coverage.pop("candidate_wrecall")
            metrics.update(coverage)
            return metrics

        self.metric_fn = metric_fn

    def forward(self, **batch_kwargs) -> dict[str, th.Tensor]:
        """Select the MASShatter inputs from SpectrumPL's collated batch.

        The collated batch carries fields for every model in the repo, so only the ones
        MASShatter uses are forwarded. Optional metadata is read with ``get`` because
        the dataset omits those keys entirely when the corresponding spec_param is off,
        and the model then also has no embedding expecting them.

        Args:
            **batch_kwargs: The collated batch produced by SpectrumPL's dataloader.

        Returns:
            The model's sparse prediction dictionary, forwarded unchanged.
        """
        output = self.model(
            mol_pyg=batch_kwargs["mol_pyg"],
            spec_prec_type=batch_kwargs["spec_prec_type"],
            spec_inst_type=batch_kwargs.get("spec_inst_type"),
            spec_frag_mode=batch_kwargs.get("spec_frag_mode"),
            spec_nce=batch_kwargs.get("spec_nce"),
            spec_nce_batch_idxs=batch_kwargs.get("spec_nce_batch_idxs"),
            spec_ace=batch_kwargs.get("spec_ace"),
            spec_ace_batch_idxs=batch_kwargs.get("spec_ace_batch_idxs"),
            static_graph=batch_kwargs.get("mass_shatter_static_graph"),
            proposal_group_ids=(
                batch_kwargs.get("group_id")
                if self.hparams.mass_shatter_share_proposal_across_ev
                else None
            ),
        )
        if (
            self.hparams.mass_shatter_share_proposal_across_ev
            and self.hparams.mass_shatter_proposal_loss_weight > 0.0
        ):
            output["proposal_group_ids"] = batch_kwargs["group_id"]
        cross_ev_enabled = self.hparams.cross_ev_fragment_loss_weight != 0.0 and self.training
        if cross_ev_enabled:
            output.update(
                {
                    "cross_ev_group_ids": batch_kwargs["group_id"],
                    "cross_ev_ce_values": select_context_ce_values(
                        batch=batch_kwargs,
                        batch_size=int(batch_kwargs["batch_size"]),
                        inst_types=self.hparams.spec_params["inst_types"],
                        ce_key_by_inst=self.hparams.ce_context_ce_key_by_inst,
                    ),
                }
            )
        else:
            for key in (
                "cross_ev_candidate_mzs",
                "cross_ev_candidate_fragment_idxs",
                "cross_ev_fragment_logprobs",
                "cross_ev_fragment_batch_idxs",
            ):
                output.pop(key, None)
        return output

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: th.device,
        dataloader_idx: int,
    ) -> Any:
        """Keep static candidate-generation graph indices on CPU.

        Mask selection and connected-component search run in Python on the host, so the
        graph topology is needed on CPU every step. Snapshotting it here — before the
        base implementation moves the batch to the accelerator — means only the bond
        logits have to come back per batch, instead of the whole index structure.

        Args:
            batch: The collated batch, still on CPU.
            device: Target accelerator device.
            dataloader_idx: Index of the dataloader that produced the batch.

        Returns:
            The transferred batch, with the CPU-resident graph indices attached under
            ``mass_shatter_static_graph``.
        """
        mol_pyg = batch["mol_pyg"]
        static_graph = MassShatterStaticGraph(
            chemical_bond_index=mol_pyg.chemical_bond_index,
            atom_batch_indices=mol_pyg.batch,
            atom_ptr=mol_pyg.ptr,
            smarts_templates=(
                tuple(_cached_mass_shatter_templates(str(smiles)) for smiles in batch["mol_smiles"])
                if self.hparams.mass_shatter_smarts_templates
                else None
            ),
        )
        transferred = super().transfer_batch_to_device(batch, device, dataloader_idx)
        transferred["mass_shatter_static_graph"] = static_graph
        return transferred
