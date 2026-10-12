"""PyTorch Lightning training wrapper for MASSter.

MASSter emits either normalized sparse log-probabilities or independent sparse
normal-space intensities. SpectrumPL's preprocessing, metric, and reduction machinery is
otherwise reused unchanged. This module only supplies the three model-specific pieces:

- :meth:`MassterPL._setup_model` maps flat hparams onto ``MassterModel``.
- :meth:`MassterPL._setup_loss_fn` wraps the sparse cross entropy and the
  adjacent-CE switch objective, and forwards MASSter's candidate-generation
  diagnostics into the logged metrics.
- :meth:`MassterPL.transfer_batch_to_device` keeps static graph metadata on CPU
  for both candidate backends.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import torch as th
from rdkit import Chem

from fragnnet.frag.masster_candidates import MassterStaticGraph
from fragnnet.frag.masster_templates import (
    MassterTemplate,
    build_masster_templates,
)
from fragnnet.model import MassterModel
from fragnnet.model.candidate_features import describe_candidate_features
from fragnnet.model.candidate_support import (
    balanced_binary_support_loss as _balanced_binary_support_loss,
)
from fragnnet.model.candidate_support import (
    matched_peak_candidate_pairs,
)
from fragnnet.model.candidate_support import (
    observational_channel_targets as _observational_channel_targets,
)
from fragnnet.model.loss import (
    conditional_in_support_cross_entropy,
    get_sparse_cross_entropy_fn,
    sparse_peak_list_distance,
)
from fragnnet.model.output_schema import RETURN_EXPLAINABILITY_TENSORS_DEFAULT
from fragnnet.model.rt_ccs import property_regression_terms
from fragnnet.pl_model.spectrum_pl import SpectrumPL, select_context_ce_values
from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.utils.misc_utils import scatter_reduce
from fragnnet.utils.pl_utils import (
    migrate_legacy_masster_name,
    migrate_masster_proposal_method,
    migrate_masster_rt_ccs_hparams,
)
from fragnnet.utils.spec_utils import (
    calculate_match_mzs,
    get_ints_transform_func,
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
def _cached_masster_templates(
    smiles: str,
) -> tuple[MassterTemplate, ...]:
    """Match SMARTS once per unique molecule, outside the model hot path."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecule SMILES for SMARTS matching: {smiles}")
    return build_masster_templates(mol)


def _candidate_coverage_by_cut_count(
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
    candidate_cut_counts: th.Tensor | None = None,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    include_cut_buckets: bool = True,
) -> dict[str, th.Tensor]:
    """Measure overall and cut-bucket true-peak coverage from candidate masses.

    A true peak may be covered by multiple buckets, so the resulting recall values
    describe each bucket's standalone support and are not additive. The buckets cost one
    extra match-mask pass per bucket per spectrum, so ``include_cut_buckets=False`` keeps
    only the overall coverage.
    """
    metrics = (
        {
            f"candidate_{metric}_cut_{bucket}": true_ints.new_zeros(batch_size)
            for bucket in _CUT_COUNT_BUCKETS
            for metric in ("recall", "wrecall", "precision")
        }
        if include_cut_buckets
        else {}
    )
    metrics["candidate_recall"] = true_ints.new_zeros(batch_size)
    metrics["candidate_wrecall"] = true_ints.new_zeros(batch_size)
    if include_cut_buckets and candidate_cut_counts is None:
        raise ValueError("candidate_cut_counts is required for cut-bucket coverage")
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
        if not include_cut_buckets:
            continue
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
    """Measure weighted true-peak coverage from the full chemical candidate set.

    The whole batch is matched at once: a peak only ever matches candidates of its own
    spectrum, and pairs outside the mass tolerance are never materialized, so this is the
    per-spectrum ``calculate_match_mzs`` grid without the Python loop over spectra. The
    loop form cost one host-device synchronization per spectrum and dominated its own
    arithmetic; see ``_proposal_coverage_loss`` for the same rewrite.

    Args:
        true_mzs: Observed m/z values of every spectrum, shape ``(num_peaks,)``.
        true_ints: Observed intensity per peak, shape ``(num_peaks,)``.
        true_batch_idxs: Spectrum index per observed peak, shape ``(num_peaks,)``.
        candidate_mzs: Candidate m/z values, shape ``(num_candidates,)``.
        candidate_batch_idxs: Spectrum index per candidate, shape ``(num_candidates,)``.
        batch_size: Number of spectra in the batch.
        tolerance: Matching tolerance; a ratio when ``relative``, otherwise Da.
        relative: Whether the scalar tolerance is relative to the observed m/z.
        tolerance_min_mz: Divisor floor for the relative scalar tolerance.
        tol_per_sample: Optional per-spectrum relative tolerance, shape ``(batch_size,)``.
        min_mz_per_sample: Optional per-spectrum divisor floor, shape ``(batch_size,)``.

    Returns:
        Covered fraction of observed intensity per spectrum, shape ``(batch_size,)``. A
        spectrum whose candidates match nothing scores 0.

    Raises:
        ValueError: If only one of the per-sample tolerance tensors is given, or if a
            spectrum contributes no observed peaks.
    """
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    # 1. Every spectrum must contribute observed peaks; an empty one has no defined recall.
    peak_batch_idxs = true_batch_idxs.to(th.int64)
    peak_counts = th.zeros(batch_size, dtype=th.long, device=true_mzs.device).index_add(
        0,
        peak_batch_idxs,
        th.ones_like(peak_batch_idxs),
    )
    empty_spectra = (peak_counts == 0).nonzero().flatten()
    if empty_spectra.numel() > 0:
        raise ValueError(f"spectrum {int(empty_spectra[0])} has no true peaks")

    # 2. Match every observed peak against the candidates of its own spectrum.
    peak_idxs, _ = matched_peak_candidate_pairs(
        true_mzs=true_mzs,
        true_batch_idxs=true_batch_idxs,
        candidate_mzs=candidate_mzs,
        candidate_batch_idxs=candidate_batch_idxs,
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
        tol_per_sample=tol_per_sample,
        min_mz_per_sample=min_mz_per_sample,
    )

    # 3. A peak counts once however many candidates matched it, as ``any`` did per row.
    covered = th.zeros_like(true_ints, dtype=th.bool).index_fill_(
        0,
        peak_idxs,
        True,
    )

    # 4. Reduce covered and total observed intensity back to one value per spectrum.
    covered_intensity = true_ints.new_zeros(batch_size).index_add(
        0,
        peak_batch_idxs,
        true_ints * covered,
    )
    total_intensity = true_ints.new_zeros(batch_size).index_add(
        0,
        peak_batch_idxs,
        true_ints,
    )
    return covered_intensity / total_intensity


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

    The whole batch is matched at once: peak/channel pairs outside the mass tolerance
    contribute nothing, so only the pairs inside it are materialized, and fragments are
    addressed by their global index instead of a per-spectrum local one.
    """
    if (tol_per_sample is None) != (min_mz_per_sample is None):
        raise ValueError("tol_per_sample and min_mz_per_sample must be provided together")
    if candidate_mzs.shape != candidate_fragment_idxs.shape:
        raise ValueError("proposal candidate masses and fragment indices must align")
    if fragment_logprobs.shape != fragment_batch_idxs.shape:
        raise ValueError("proposal fragment probabilities and batch indices must align")

    # 1. Every spectrum must contribute both observed peaks and proposed fragments.
    peak_counts = th.zeros(batch_size, dtype=th.long, device=true_mzs.device).index_add(
        0,
        true_batch_idxs.to(th.int64),
        th.ones_like(true_batch_idxs, dtype=th.long),
    )
    fragment_counts = th.zeros(
        batch_size,
        dtype=th.long,
        device=fragment_batch_idxs.device,
    ).index_add(
        0,
        fragment_batch_idxs.to(th.int64),
        th.ones_like(fragment_batch_idxs, dtype=th.long),
    )
    empty_spectra = (peak_counts == 0).nonzero().flatten()
    if empty_spectra.numel() > 0:
        raise ValueError(f"spectrum {int(empty_spectra[0])} has no true peaks")
    empty_proposals = (fragment_counts == 0).nonzero().flatten()
    if empty_proposals.numel() > 0:
        raise ValueError(f"spectrum {int(empty_proposals[0])} has no proposed fragments")

    # 2. Match every observed peak against the channels of its own spectrum.
    peak_idxs, candidate_idxs = matched_peak_candidate_pairs(
        true_mzs=true_mzs,
        true_batch_idxs=true_batch_idxs,
        candidate_mzs=candidate_mzs,
        candidate_batch_idxs=fragment_batch_idxs[candidate_fragment_idxs],
        tolerance=tolerance,
        relative=relative,
        tolerance_min_mz=tolerance_min_mz,
        tol_per_sample=tol_per_sample,
        min_mz_per_sample=min_mz_per_sample,
    )

    # 3. Count each fragment once per peak, however many of its channels matched it.
    num_fragments = fragment_logprobs.shape[0]
    num_peaks = true_mzs.shape[0]
    pair_keys = th.unique(peak_idxs * num_fragments + candidate_fragment_idxs[candidate_idxs])
    pair_peak_idxs = th.div(pair_keys, num_fragments, rounding_mode="floor")
    pair_fragment_idxs = pair_keys - pair_peak_idxs * num_fragments

    # 4. Sum the probability of the distinct fragments covering each peak, in log space.
    # The shift is detached, so this is the gradient of the plain log-sum-exp.
    pair_logprobs = fragment_logprobs[pair_fragment_idxs]
    peak_maxima = scatter_reduce(
        pair_logprobs.detach(),
        pair_peak_idxs,
        reduce="amax",
        dim_size=num_peaks,
        default=0.0,
        include_self=False,
    )
    peak_coverage = scatter_reduce(
        th.exp(pair_logprobs - peak_maxima[pair_peak_idxs]),
        pair_peak_idxs,
        reduce="sum",
        dim_size=num_peaks,
        default=0.0,
        include_self=False,
    )
    covered = peak_coverage > 0.0
    peak_log_coverage = (
        peak_coverage.clamp_min(th.finfo(peak_coverage.dtype).tiny).log() + peak_maxima
    )

    # 5. Weight every covered peak and reduce back to one value per spectrum.
    peak_probabilities = (
        true_logprobs.new_ones(true_logprobs.shape) / peak_counts[true_batch_idxs]
        if barcode_target
        else true_logprobs.exp()
    )
    peak_losses = th.where(
        covered,
        -peak_probabilities * peak_log_coverage,
        th.zeros_like(peak_log_coverage),
    )
    coverage_loss = fragment_logprobs.new_zeros(batch_size).index_add(
        0,
        true_batch_idxs.to(th.int64),
        peak_losses.to(fragment_logprobs),
    )
    # A batch where nothing matched references no fragment probability at all. Tie the
    # result back to them with zero weight so the loss stays differentiable either way.
    return coverage_loss + 0.0 * fragment_logprobs.sum()


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
    if fragment_logprobs is not None and fragment_logprobs.shape != fragment_batch_idxs.shape:
        raise ValueError(
            "fragment_logprobs must hold one value per scored fragment, "
            f"got {tuple(fragment_logprobs.shape)} for {tuple(fragment_batch_idxs.shape)}"
        )
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


class MassterPL(SpectrumPL):
    """Train MASSter with SpectrumPL and FraGNNet's CE-pair objective.

    Attributes:
        model: The wrapped :class:`MassterModel`.
        loss_fn: Closure built in :meth:`_setup_loss_fn`; called with the batch merged
            with the model's output dictionary, returning per-spectrum vectors that
            SpectrumPL reduces.
        loss_names: Keys of ``loss_fn``'s output that get logged every step.
    """

    hparams: Any

    def __init__(self, **kwargs: Any) -> None:
        migrate_legacy_masster_name(kwargs)
        migrate_masster_proposal_method(kwargs)
        legacy_candidate_backend = kwargs.pop("_legacy_candidate_backend", False)
        if legacy_candidate_backend and "masster_candidate_backend" not in kwargs:
            kwargs["masster_candidate_backend"] = "cpu"
        if legacy_candidate_backend:
            migrate_masster_rt_ccs_hparams(kwargs)
            if "masster_proposal_method" not in kwargs:
                kwargs["masster_proposal_method"] = "cut_masks"
            if "masster_fast_gnn_beam_width" not in kwargs:
                kwargs["masster_fast_gnn_beam_width"] = 256
            if "masster_fast_gnn_max_boundary_cuts" not in kwargs:
                kwargs["masster_fast_gnn_max_boundary_cuts"] = 3
            if "masster_fragment_selection" not in kwargs:
                kwargs["masster_fragment_selection"] = "mask_order"
            if "masster_selected_fragment_target" not in kwargs:
                kwargs["masster_selected_fragment_target"] = None
        super().__init__(**kwargs)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Any,
        map_location: Any = None,
        hparams_file: Any = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> MassterPL:
        """Load Lightning checkpoints, migrating missing candidate/proposal settings.

        Lightning merges saved hyperparameters and explicit overrides before the
        constructor runs. The private marker is removed before hyperparameters are
        captured, so saved or overridden backend values retain their precedence.
        Positional arguments have the same meaning as in Lightning's loader.
        """
        kwargs["_legacy_candidate_backend"] = True
        return super().load_from_checkpoint(
            checkpoint_path, map_location, hparams_file, strict, **kwargs
        )

    def _setup_model(self) -> None:
        """Instantiate ``MassterModel`` from the flat hparam namespace.

        Raises:
            ValueError: If ``compile`` is on, or the spectrum config does not provide
                the precursor type MASSter needs to compute adduct-dependent m/z.
        """
        for name in ("rt", "ccs", "rt_ccs"):
            if not isinstance(self.hparams[f"masster_predict_{name}"], bool):
                raise ValueError(f"masster_predict_{name} must be boolean")
        self.property_names = tuple(
            name for name in ("rt", "ccs")
            if self.hparams[f"masster_predict_{name}"] or self.hparams.masster_predict_rt_ccs
        )
        if self.property_names:
            if self.hparams.spec_params["merge"] or self.hparams.spec_params["merge_train_only"]:
                raise ValueError("RT/CCS heads require unmerged spectra")
            if self.hparams.masster_proposal_only_until_epoch is not None:
                raise ValueError("RT/CCS heads require joint training, without proposal-only warmup")
            for name in self.property_names:
                weight = self.hparams[f"masster_{name}_loss_weight"]
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(f"masster_{name}_loss_weight must be finite and positive")
        proposal_only_until_epoch = self.hparams.masster_proposal_only_until_epoch
        if proposal_only_until_epoch is not None and (
            isinstance(proposal_only_until_epoch, bool)
            or not isinstance(proposal_only_until_epoch, int)
            or proposal_only_until_epoch < 0
        ):
            raise ValueError(
                "masster_proposal_only_until_epoch must be null or a nonnegative integer"
            )
        if proposal_only_until_epoch is not None:
            if not self.hparams.masster_proposal_selection_only:
                raise ValueError(
                    "masster_proposal_only_until_epoch requires "
                    "masster_proposal_selection_only=True"
                )
            if self.hparams.masster_proposal_loss_weight <= 0.0:
                raise ValueError(
                    "masster_proposal_only_until_epoch requires a positive "
                    "masster_proposal_loss_weight"
                )
        self._check_proposal_early_stop_params(proposal_only_until_epoch)
        legacy_oos_enabled = (
            self.hparams.predict_oos
            or self.hparams.oos_loss
            or (
                "masster_learned_oos" in self.hparams
                and self.hparams["masster_learned_oos"]
            )
        )
        if legacy_oos_enabled:
            logger.warning(
                "Ignoring legacy OOS settings for MASSter; the shared predict_oos/oos_loss "
                "path is FraGNNet's. Use masster_intensity_escape for MASSter's "
                "escape coordinate (currently %s).",
                self.hparams.masster_intensity_escape,
            )
        self.hparams.predict_oos = False
        self.hparams.oos_loss = False
        if "masster_learned_oos" in self.hparams:
            self.hparams["masster_learned_oos"] = False
        self.hparams.auxiliary_scores = [
            score
            for score in self.hparams.auxiliary_scores
            if score not in {"true_oos_prob", "true_oos_e"}
        ]

        # Candidate construction uses native extension calls and host metadata that
        # dynamo cannot trace; compiling would graph-break every batch.
        if self.hparams.compile:
            raise ValueError(
                "MASSter candidate construction uses native kernels; set compile: False"
            )
        if (
            self.hparams.masster_proposal_method == "cut_masks"
            and self.hparams.masster_candidate_backend == "cuda"
            and self.hparams.precision
            not in (
                32,
                "32",
                "32-true",
            )
        ):
            raise ValueError("CUDA candidates require precision: 32-true (FP32)")
        if not self.hparams.spec_params["prec_type"]:
            raise ValueError("MASSter requires spec_params.prec_type: True")
        if self.hparams.masster_intensity_atom_decoder_num_layers != 0:
            raise ValueError(
                "masster_intensity_atom_decoder_num_layers must be 0: the "
                "fragment-to-atom decoder was removed on 2026-09-19 after it showed no "
                "epoch-matched cosine gain at 1.8x the wall clock; set it to 0"
            )
        if (
            self.hparams.ints_transform != "none"
            and self.hparams.masster_mass_mode == "exact"
            and not self.hparams.masster_merge_same_mz_rows
        ):
            # Training learns the transformed intensity of each m/z (pathways summed in the
            # Gaussian mixture); per-pathway untransform at eval would square each pathway
            # instead of their sum and under-predict peaks formed by several pathways.
            raise ValueError(
                f"ints_transform: {self.hparams.ints_transform} with masster_mass_mode: exact "
                "requires masster_merge_same_mz_rows: true"
            )
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
        self.model = MassterModel(
            predict_rt="rt" in self.property_names,
            predict_ccs="ccs" in self.property_names,
            rt_scale=self.hparams.masster_rt_scale,
            ccs_scale=self.hparams.masster_ccs_scale,
            candidate_backend=self.hparams.masster_candidate_backend,
            proposal_method=self.hparams.masster_proposal_method,
            fast_gnn_beam_width=self.hparams.masster_fast_gnn_beam_width,
            fast_gnn_max_boundary_cuts=self.hparams.masster_fast_gnn_max_boundary_cuts,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            precursor_hidden_dim=self.hparams.mol_hidden_size,
            precursor_num_layers=self.hparams.mol_num_layers,
            precursor_num_heads=self.hparams.mol_num_heads,
            precursor_gnn_type=self.hparams.mol_gnn_type,
            precursor_normalization=self.hparams.mol_normalization,
            context_normalization=self.hparams.masster_context_normalization,
            context_dropout=self.hparams.masster_context_dropout,
            precursor_dropout=self.hparams.mol_dropout,
            precursor_max_degree=self.hparams["mol_max_degree"],
            precursor_max_spatial_distance=self.hparams["mol_max_spatial_distance"],
            precursor_multi_hop_max_distance=self.hparams["mol_multi_hop_max_distance"],
            context_dim=self.hparams.masster_context_dim,
            context_in_gnn=self.hparams.masster_context_in_gnn,
            encoder_context_token=self.hparams.masster_encoder_context_token,
            metadata_embedding_dim=self.hparams.masster_metadata_embedding_dim,
            head_hidden_dim=self.hparams.masster_head_hidden_dim,
            head_dropout=self.hparams.masster_head_dropout,
            bond_num_layers=self.hparams.masster_bond_num_layers,
            proposal_num_mixtures=self.hparams.masster_proposal_num_mixtures,
            intensity_transformer_dim=(self.hparams.masster_intensity_transformer_dim),
            intensity_transformer_num_layers=(
                self.hparams.masster_intensity_transformer_num_layers
            ),
            intensity_transformer_num_heads=(
                self.hparams.masster_intensity_transformer_num_heads
            ),
            intensity_transformer_feedforward_dim=(
                self.hparams.masster_intensity_transformer_feedforward_dim
            ),
            intensity_transformer_num_inducing_points=(
                self.hparams.masster_intensity_transformer_num_inducing_points
            ),
            intensity_transformer_attention=(
                self.hparams.masster_intensity_transformer_attention
            ),
            intensity_transformer_dropout=(self.hparams.masster_intensity_transformer_dropout),
            intensity_norm_placement=(self.hparams.masster_intensity_norm_placement),
            intensity_qk_norm=self.hparams.masster_intensity_qk_norm,
            intensity_mass_pair_bias=self.hparams.masster_intensity_mass_pair_bias,
            intensity_mz_positional_encoding=(
                self.hparams.masster_intensity_mz_positional_encoding
            ),
            intensity_nl_positional_encoding=(
                self.hparams.masster_intensity_nl_positional_encoding
            ),
            intensity_mass_window=self.hparams.masster_intensity_mass_window,
            intensity_candidate_graph=self.hparams.masster_intensity_candidate_graph,
            intensity_ce_pair_bias=self.hparams.masster_intensity_ce_pair_bias,
            intensity_ce_film=self.hparams.masster_intensity_ce_film,
            bond_endpoint_aggregation=(self.hparams.masster_bond_endpoint_aggregation),
            gnn_residual_activation=self.hparams.gnn_residual_activation,
            intensity_output=self.hparams.masster_intensity_output,
            use_mask_residuals=self.hparams.masster_use_mask_residuals,
            use_base_probability_feature=(not self.hparams.masster_proposal_anchored),
            return_base_spectrum=(
                self.hparams.masster_log_base_ce
                or self.hparams.masster_base_ce_weight > 0.0
            ),
            detach_base_ion_h=self.hparams.masster_detach_base_ion_h,
            proposal_selection_only=self.hparams.masster_proposal_selection_only,
            separate_proposal_encoder=(self.hparams.masster_separate_proposal_encoder),
            share_proposal_across_ev=self.hparams.masster_share_proposal_across_ev,
            separate_ion_mode_proposal_head=(
                self.hparams.masster_separate_ion_mode_proposal_head
            ),
            ion_mode_proposal_binary_logits=(
                self.hparams.masster_ion_mode_proposal_binary_logits
            ),
            min_ion_mode_relative_probability=(
                self.hparams.masster_min_ion_mode_relative_probability
            ),
            separate_ion_h_support_head=(self.hparams.masster_separate_ion_h_support_head),
            ion_h_support_binary_logits=(self.hparams.masster_ion_h_support_binary_logits),
            gate_independent_intensity_with_support=(
                self.hparams.masster_gate_independent_intensity_with_support
            ),
            min_ion_h_relative_probability=(
                self.hparams.masster_min_ion_h_relative_probability
            ),
            min_ion_h_absolute_probability=(
                self.hparams.masster_min_ion_h_absolute_probability
            ),
            apply_channel_selection_in_training=(
                self.hparams.masster_apply_channel_selection_in_training
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
                self.hparams.masster_unique_fragment_target
                if self.hparams.masster_unique_fragment_mode
                else self.hparams.masster_selected_mask_limit
            ),
            unique_fragment_mode=self.hparams.masster_unique_fragment_mode,
            max_enumerated_masks=self.hparams.masster_ranked_mask_budget,
            mask_selection_threads=self.hparams.masster_mask_selection_threads,
            min_mask_relative_probability=(self.hparams.masster_min_mask_relative_probability),
            min_fragment_relative_probability=(
                self.hparams.masster_min_fragment_relative_probability
            ),
            fragment_selection=self.hparams.masster_fragment_selection,
            selected_fragment_target=self.hparams.masster_selected_fragment_target,
            include_single_bond_cuts=(self.hparams.masster_include_single_bond_cuts),
            max_h_transfer=self.hparams.masster_max_h_transfer,
            h_transfer_bound=self.hparams.masster_h_transfer_bound,
            include_cmf=self.hparams.masster_include_cmf,
            charge_pooling=self.hparams.masster_charge_pooling,
            use_smarts_templates=self.hparams.masster_smarts_templates,
            boundary_bond_encoder=self.hparams.masster_boundary_bond_encoder,
            intensity_escape=self.hparams.masster_intensity_escape,
            candidate_features=self.hparams.masster_candidate_features,
            candidate_atom_dropout=self.hparams.masster_candidate_atom_dropout,
            candidate_boundary_dropout=self.hparams.masster_candidate_boundary_dropout,
            candidate_boundary_block=self.hparams.masster_candidate_boundary_block,
            boundary_local_layers=self.hparams.masster_boundary_local_layers,
            boundary_encoder_dropout=self.hparams.masster_boundary_encoder_dropout,
            fragment_connectivity_layers=self.hparams.masster_fragment_connectivity_layers,
            fragment_connectivity_hidden_dim=(
                self.hparams.masster_fragment_connectivity_hidden_dim
            ),
            fragment_connectivity_messages=self.hparams.masster_fragment_connectivity_messages,
            intensity_input_dropout=self.hparams.masster_intensity_input_dropout,
            intensity_set_message_substitution=self.hparams.masster_set_message_substitution,
            intensity_set_key_dropout=self.hparams.masster_set_key_dropout,
            memorization_sink_dim=self.hparams.masster_memorization_sink_dim,
            memorization_sink_num_molecules=(
                self.hparams.masster_memorization_sink_num_molecules
            ),
            memorization_sink_dropout=self.hparams.masster_memorization_sink_dropout,
            global_message_substitution=self.hparams.masster_global_message_substitution,
            global_message_noise_std=self.hparams.masster_global_message_noise_std,
            global_message_swap=self.hparams.masster_global_message_swap,
            global_message_swap_unit=self.hparams.masster_global_message_swap_unit,
            encoder_atom_mask=self.hparams.masster_encoder_atom_mask,
            encoder_bond_dropout=self.hparams.masster_encoder_bond_dropout,
            light_side_token=self.hparams.masster_light_side_token,
            rule_pathway_token=self.hparams.masster_rule_pathway_token,
            mass_mode=self.hparams.masster_mass_mode,
            merge_same_mz_rows=self.hparams.masster_merge_same_mz_rows,
            isotope_envelope=self.hparams.masster_isotope_envelope,
            isotope_max_order=self.hparams.masster_isotope_max_order,
            isotope_min_ratio=self.hparams.masster_isotope_min_ratio,
        )
        logger.info("Resolved MASSter proposal method: %s", self.model.proposal_method)
        logger.info(
            "Resolved MASSter candidate token: %s",
            describe_candidate_features(
                self.model.candidate_feature_specs,
                edge_feature_dim,
            ),
        )
        self._proposal_only_training = False
        self._check_ce_params()

    def _check_proposal_early_stop_params(self, proposal_only_until_epoch: int | None) -> None:
        """Validate the proposal early-stop epoch and the proposal-only weight decay.

        Args:
            proposal_only_until_epoch: The already-validated proposal warm-up boundary, used
                only to reject combining the warm-up with the early stop.

        Raises:
            ValueError: If either hyperparameter is outside its domain, or the early stop is
                requested for a configuration where freezing the proposal branch would also
                change what the intensity branch optimizes.
        """
        frozen_from_epoch = self.hparams.masster_proposal_frozen_from_epoch
        if frozen_from_epoch is not None and (
            isinstance(frozen_from_epoch, bool)
            or not isinstance(frozen_from_epoch, int)
            or frozen_from_epoch < 0
        ):
            raise ValueError(
                "masster_proposal_frozen_from_epoch must be null or a nonnegative integer"
            )
        if frozen_from_epoch is not None:
            if not self.hparams.masster_proposal_selection_only:
                # Outside selection-only mode the proposal probabilities multiply into the
                # spectrum, so the primary loss keeps pulling on the proposal heads and
                # freezing them would change the intensity path rather than early-stop one
                # objective.
                raise ValueError(
                    "masster_proposal_frozen_from_epoch requires "
                    "masster_proposal_selection_only=True"
                )
            if self.hparams.masster_proposal_loss_weight <= 0.0:
                raise ValueError(
                    "masster_proposal_frozen_from_epoch requires a positive "
                    "masster_proposal_loss_weight"
                )
            if proposal_only_until_epoch is not None:
                raise ValueError(
                    "masster_proposal_frozen_from_epoch and "
                    "masster_proposal_only_until_epoch cannot be combined"
                )
        proposal_weight_decay = self.hparams.masster_proposal_weight_decay
        if proposal_weight_decay is not None and (
            isinstance(proposal_weight_decay, bool)
            or not isinstance(proposal_weight_decay, (int, float))
            or not math.isfinite(proposal_weight_decay)
            or proposal_weight_decay < 0.0
        ):
            raise ValueError(
                "masster_proposal_weight_decay must be null or a finite nonnegative float"
            )

    def _proposal_modules(self) -> tuple[th.nn.Module, ...]:
        """Return modules whose outputs determine proposal candidate selection."""
        proposal_encoder = (
            self.model.proposal_encoder
            if self.model.proposal_encoder is not None
            else self.model.precursor_encoder
        )
        modules = (
            self.model.context_encoder,
            proposal_encoder,
            self.model.bond_break_head,
            self.model.proposal_mixture_head,
            self.model.charge_head if self.model.proposal_method == "cut_masks" else None,
            self.model.ion_mode_proposal_head,
            self.model.mechanism_head,
        )
        return tuple(module for module in modules if module is not None)

    def _proposal_branch_modules(self) -> tuple[th.nn.Module, ...]:
        """Return the modules that only the proposal branch uses.

        This is ``_proposal_modules`` without the two encoders the intensity branch shares.
        Freezing or re-decaying ``precursor_encoder`` and ``context_encoder`` would change
        what the spectrum objective optimizes, which neither the proposal early stop nor the
        proposal weight decay intends. ``proposal_encoder`` appears only when
        ``separate_proposal_encoder`` gives the proposal its own GNN, where it is
        proposal-only by construction.
        """
        modules = (
            self.model.proposal_encoder,
            self.model.bond_break_head,
            self.model.proposal_mixture_head,
            self.model.charge_head if self.model.proposal_method == "cut_masks" else None,
            self.model.ion_mode_proposal_head,
            self.model.mechanism_head,
        )
        return tuple(module for module in modules if module is not None)

    def _weight_decay_overrides(self) -> dict[int, float]:
        """Give proposal-only parameters their own weight decay when one is configured."""
        proposal_weight_decay = self.hparams.masster_proposal_weight_decay
        if proposal_weight_decay is None:
            return {}
        return {
            id(parameter): float(proposal_weight_decay)
            for module in self._proposal_branch_modules()
            for parameter in module.parameters()
        }

    def _intensity_modules(self) -> tuple[th.nn.Module, ...]:
        """Return modules trained by the final spectrum-intensity objective."""
        return (self.model.precursor_encoder, self.model.intensity_head)

    @staticmethod
    def _set_modules_trainable(
        modules: tuple[th.nn.Module, ...],
        *,
        trainable: bool,
    ) -> None:
        """Set gradient and train/evaluation state for one training branch."""
        for module in modules:
            module.requires_grad_(trainable)
            if trainable:
                module.train()
            else:
                for parameter in module.parameters():
                    parameter.grad = None
                module.eval()

    def _apply_staged_training(self, epoch: int) -> None:
        """Train proposal only through the boundary, then train both branches."""
        proposal_only_until_epoch = self.hparams.masster_proposal_only_until_epoch
        if proposal_only_until_epoch is None:
            return

        # 1. Select the phase using the inclusive proposal-only boundary.
        proposal_only = epoch <= proposal_only_until_epoch
        self._proposal_only_training = proposal_only
        self.model.set_proposal_input_gradients_enabled(
            proposal_only or self.model.proposal_encoder is None
        )

        # 2. Keep proposal training active and enable intensity after warm-up.
        self._set_modules_trainable(self._proposal_modules(), trainable=True)
        if proposal_only:
            intensity_only_modules = (self.model.intensity_head,)
            if self.model.proposal_encoder is not None:
                intensity_only_modules = (
                    self.model.precursor_encoder,
                    *intensity_only_modules,
                )
            self._set_modules_trainable(intensity_only_modules, trainable=False)
        else:
            self._set_modules_trainable(self._intensity_modules(), trainable=True)

    def _apply_proposal_early_stop(self, epoch: int) -> None:
        """Stop updating the proposal branch from its configured epoch onward.

        The validation proposal term turns at epochs 26-47 while the spectrum term keeps
        improving to epoch 52-120, so the branches want different horizons; this holds the
        proposal at the epoch it stops generalizing and lets the intensity head train on.
        Only proposal-only modules freeze. ``masster_proposal_selection_only`` is
        required, so the proposal reads detached inputs and its term contributes no gradient
        anywhere once these parameters are frozen -- the loss is left unchanged, and the
        logged ``proposal_loss`` stays comparable across the boundary. The shared encoders
        keep training on the spectrum objective, so the proposal *function* still drifts
        unless ``separate_proposal_encoder`` gives it its own GNN.
        """
        frozen_from_epoch = self.hparams.masster_proposal_frozen_from_epoch
        if frozen_from_epoch is None:
            return
        self._set_modules_trainable(
            self._proposal_branch_modules(),
            trainable=epoch < frozen_from_epoch,
        )

    def on_train_epoch_start(self) -> None:
        """Seed the loader and apply the configured staged-training phase."""
        super().on_train_epoch_start()
        self._apply_staged_training(self.current_epoch)
        self._apply_proposal_early_stop(self.current_epoch)

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
            "conditional_cross_entropy",
            "cross_entropy",
            "cosine_distance",
            "hellinger_distance",
            "jss_distance",
            "kl_distance",
        }
        if self.hparams.loss_type not in accepted_loss_types:
            raise ValueError(f"MASSter loss_type must be one of {sorted(accepted_loss_types)}")
        if self.hparams.masster_intensity_output not in {
            "probability",
            "independent",
        }:
            raise ValueError("masster_intensity_output must be probability or independent")
        if self.hparams.masster_intensity_transformer_attention not in {
            "induced",
            "full",
            "none",
        }:
            raise ValueError(
                "masster_intensity_transformer_attention must be induced, full or none"
            )
        independent_output = self.hparams.masster_intensity_output == "independent"
        cross_entropy_loss = self.hparams.loss_type in {
            "conditional_cross_entropy",
            "cross_entropy",
        }
        if self.hparams.precursor_pair_loss_weight != 0.0:
            raise ValueError(
                "MASSter does not support precursor_pair_loss_weight; set it to 0.0"
            )
        if independent_output and cross_entropy_loss:
            raise ValueError("independent intensity output requires a direct spectral loss")
        if self.hparams.masster_intensity_escape and not independent_output:
            # Fail loudly rather than silently ignore: the escape only reaches the objective
            # through sparse_peak_list_distance. A cross-entropy loss_type needs no separate
            # check here, because independent output already rejects it directly above and
            # the escape already requires independent output.
            raise ValueError(
                "masster_intensity_escape requires "
                "masster_intensity_output: independent"
            )
        if not cross_entropy_loss and (
            self.hparams.masster_base_ce_weight != 0.0 or self.hparams.masster_log_base_ce
        ):
            raise ValueError("direct spectral losses do not support the base Gaussian CE")
        if independent_output and (
            self.hparams.masster_base_ce_weight != 0.0
            or self.hparams.masster_log_base_ce
            or self.hparams.ce_pair_loss_weight != 0.0
            or self.hparams.cross_ev_fragment_loss_weight != 0.0
        ):
            raise ValueError(
                "independent intensity output does not support probability-based CE auxiliaries"
            )
        if self.hparams.masster_peak_matching not in {"closest", "hungarian"}:
            raise ValueError("masster_peak_matching must be closest or hungarian")
        if self.hparams.masster_channel_support_attribution not in {
            "gaussian_posterior",
            "hungarian",
        }:
            raise ValueError(
                "masster_channel_support_attribution must be gaussian_posterior or hungarian"
            )
        if not isinstance(
            self.hparams.masster_channel_support_use_intensity_prior,
            bool,
        ):
            raise ValueError("masster_channel_support_use_intensity_prior must be boolean")
        proposal_ints_transform = self.hparams.masster_proposal_ints_transform
        proposal_ints_transform_func = None
        if proposal_ints_transform != "same":
            proposal_ints_transform_func = get_ints_transform_func(proposal_ints_transform)
        if (
            not math.isfinite(self.hparams.masster_intensity_entropy_match_weight)
            or self.hparams.masster_intensity_entropy_match_weight < 0.0
        ):
            raise ValueError(
                "masster_intensity_entropy_match_weight must be finite and nonnegative"
            )
        self._validate_spectral_auxiliary_weights()
        if (
            not math.isfinite(self.hparams.masster_kl_epsilon)
            or not 0.0 < self.hparams.masster_kl_epsilon < 1.0
        ):
            raise ValueError("masster_kl_epsilon must be finite and in (0, 1)")
        if self.hparams.loss_type == "kl_distance" and self.hparams.precision in {
            16,
            "16",
            "16-mixed",
        }:
            raise ValueError(
                "kl_distance does not support FP16 mixed precision; use precision 32 or bf16"
            )
        if not self.hparams.sparse_cosine_similarity or not self.hparams.sum_ints:
            raise ValueError("MASSter requires sparse_cosine_similarity and sum_ints")
        if self.hparams.ce_pair_min_endpoint_reliability is not None:
            raise ValueError("ce_pair_min_endpoint_reliability is not implemented")
        # Predictions are exact monoisotopic m/z values, so targets stay unbinned and
        # are matched against predictions by instrument tolerance instead.
        self.binned_loss = False
        if (
            not math.isfinite(self.hparams.masster_base_ce_weight)
            or self.hparams.masster_base_ce_weight < 0.0
        ):
            raise ValueError("masster_base_ce_weight must be finite and nonnegative")
        if (
            not math.isfinite(self.hparams.masster_proposal_loss_weight)
            or self.hparams.masster_proposal_loss_weight < 0.0
        ):
            raise ValueError("masster_proposal_loss_weight must be finite and nonnegative")
        support_head_configs = (
            (
                "ion_mode",
                self.hparams.masster_separate_ion_mode_proposal_head,
                self.hparams.masster_ion_mode_proposal_binary_logits,
                self.hparams.masster_ion_mode_proposal_loss_weight,
                self.hparams.masster_min_ion_mode_relative_probability,
            ),
            (
                "ion_h",
                self.hparams.masster_separate_ion_h_support_head,
                self.hparams.masster_ion_h_support_binary_logits,
                self.hparams.masster_ion_h_support_loss_weight,
                self.hparams.masster_min_ion_h_relative_probability,
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
                raise ValueError(f"masster_min_{name}_relative_probability must be in [0, 1]")
        if not isinstance(self.hparams.masster_apply_channel_selection_in_training, bool):
            raise ValueError("masster_apply_channel_selection_in_training must be boolean")
        absolute_support_floor = self.hparams.masster_min_ion_h_absolute_probability
        if not math.isfinite(absolute_support_floor) or not 0.0 <= absolute_support_floor <= 1.0:
            raise ValueError("masster_min_ion_h_absolute_probability must be in [0, 1]")
        if absolute_support_floor > 0.0 and not (
            independent_output
            and self.hparams.masster_separate_ion_h_support_head
            and self.hparams.masster_ion_h_support_binary_logits
        ):
            raise ValueError(
                "an absolute ion/H support floor requires independent intensity and a "
                "separate binary support head"
            )
        if not isinstance(self.hparams.masster_candidate_coverage_metrics, bool):
            raise ValueError("masster_candidate_coverage_metrics must be boolean")
        if not isinstance(self.hparams.masster_candidate_timing_metrics, bool):
            raise ValueError("masster_candidate_timing_metrics must be boolean")
        if not isinstance(self.hparams.masster_candidate_cut_diagnostics, bool):
            raise ValueError("masster_candidate_cut_diagnostics must be boolean")
        if self.hparams.masster_candidate_cut_diagnostics and not (
            self.hparams.masster_candidate_coverage_metrics
        ):
            raise ValueError(
                "masster_candidate_cut_diagnostics requires "
                "masster_candidate_coverage_metrics"
            )
        if not isinstance(self.hparams.masster_channel_diagnostics, bool):
            raise ValueError("masster_channel_diagnostics must be boolean")
        if not isinstance(self.hparams.masster_channel_support_metrics, bool):
            raise ValueError("masster_channel_support_metrics must be boolean")
        if self.hparams.masster_channel_diagnostics and not (
            self.hparams.masster_unique_fragment_mode
            and self.hparams.masster_proposal_selection_only
        ):
            raise ValueError(
                "masster_channel_diagnostics requires unique-fragment "
                "proposal-selection-only mode"
            )
        if self.hparams.masster_apply_channel_selection_in_training and not any(
            enabled and loss_weight > 0.0 and probability_floor > 0.0
            for _, enabled, _, loss_weight, probability_floor in support_head_configs
        ):
            raise ValueError(
                "training-time channel selection requires a supervised support head "
                "with a positive selection floor"
            )
        if not isinstance(self.hparams.masster_proposal_barcode_target, bool):
            raise ValueError("masster_proposal_barcode_target must be boolean")
        if (
            self.hparams.masster_proposal_barcode_target
            and self.hparams.masster_proposal_loss_weight == 0.0
        ):
            raise ValueError(
                "masster_proposal_barcode_target requires a positive "
                "masster_proposal_loss_weight"
            )
        if (
            self.hparams.masster_proposal_selection_only
            and self.hparams.masster_proposal_loss_weight == 0.0
        ):
            raise ValueError(
                "masster_proposal_selection_only requires a positive "
                "masster_proposal_loss_weight"
            )
        if (
            not self.hparams.masster_proposal_selection_only
            and self.hparams.masster_proposal_loss_weight != 0.0
        ):
            raise ValueError(
                "masster_proposal_loss_weight is only supported when "
                "masster_proposal_selection_only is true"
            )
        if (
            self.hparams.masster_proposal_selection_only
            and self.hparams.masster_base_ce_weight != 0.0
        ):
            raise ValueError(
                "masster_proposal_selection_only uses the H-independent proposal loss; "
                "set masster_base_ce_weight to 0"
            )
        if (
            self.hparams.masster_proposal_anchored
            and self.hparams.masster_base_ce_weight == 0.0
            and not (
                self.hparams.masster_proposal_selection_only
                and self.hparams.masster_proposal_loss_weight > 0.0
            )
        ):
            raise ValueError(
                "masster_proposal_anchored requires a positive proposal supervision loss"
            )
        candidate_coverage_enabled = self.hparams.masster_candidate_coverage_metrics
        candidate_timing_enabled = self.hparams.masster_candidate_timing_metrics
        proposal_loss_enabled = self.hparams.masster_proposal_loss_weight > 0.0
        ion_mode_proposal_loss_enabled = self.hparams.masster_ion_mode_proposal_loss_weight > 0.0
        ion_h_support_loss_enabled = self.hparams.masster_ion_h_support_loss_weight > 0.0
        # masster_channel_diagnostics exists to produce exactly these summaries, so it
        # implies them; without that the dense targets it builds would go unused.
        channel_support_metrics_enabled = (
            self.hparams.masster_channel_support_metrics
            or self.hparams.masster_channel_diagnostics
        )
        channel_targets_enabled = (
            channel_support_metrics_enabled
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
            self.hparams.masster_unique_fragment_mode
            and self.hparams.masster_share_proposal_across_ev
        ):
            raise ValueError(
                "cross_ev_fragment_loss_weight requires unique-fragment mode and "
                "masster_share_proposal_across_ev"
            )
        base_ce_enabled = (
            self.hparams.masster_log_base_ce or self.hparams.masster_base_ce_weight > 0.0
        )
        sparse_ce_fn = None
        if cross_entropy_loss or self.hparams.spectral_ce_auxiliary_weight > 0.0:
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

        ce_pair_loss_fn = self._build_ce_pair_loss_fn()

        # SpectrumPL calls this with the training dict merged with the model's output
        # dict, so every key MassterModel returns has to be named here or absorbed
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
            candidate_topology_assembly_batch_ms: th.Tensor,
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
            escape_intensities = kwargs.get("pred_escape_intensities")
            prediction_reference = pred_logprobs if pred_logprobs is not None else pred_intensities
            coverage_weighted_spec_ce = None
            conditional_spec_ce = None
            spectral_ce_auxiliary = None
            spectral_leak_loss = None
            entropy_match = None
            if cross_entropy_loss:
                if pred_logprobs is None:
                    raise ValueError("cross_entropy requires probability output")
                if sparse_ce_fn is None:
                    raise RuntimeError("sparse CE was not configured")
                # The legacy control retains its Gaussian/peak-marginal behavior.
                coverage_weighted_spec_ce, _, true_oos_logprobs, _ = sparse_ce_fn(
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
                conditional_spec_ce = conditional_in_support_cross_entropy(
                    coverage_weighted_spec_ce,
                    true_oos_logprobs,
                )
                primary_loss = (
                    conditional_spec_ce
                    if self.hparams.loss_type == "conditional_cross_entropy"
                    else coverage_weighted_spec_ce
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
                    matching=self.hparams.masster_peak_matching,
                    pred_dump_intensities=escape_intensities,
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                    decimals=4,
                    kl_epsilon=self.hparams.masster_kl_epsilon,
                    loss_batch_size=self.hparams.loss_batch_size,
                )
            candidate_wrecall = None
            if candidate_coverage_enabled:
                # Monitoring only, and the overall coverage alone: the per-cut-count
                # buckets belong to the validation metric path.
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
            loss = self.hparams.primary_loss_weight * primary_loss
            if entropy_match is not None:
                loss = loss + self.hparams.masster_intensity_entropy_match_weight * entropy_match
            if self.hparams.spectral_leak_loss_weight > 0.0:
                spectral_leak_loss = self._spectral_leak_loss(
                    true_mzs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_intensities,
                    pred_batch_idxs,
                    batch_size=batch_size,
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                )
                loss = loss + self.hparams.spectral_leak_loss_weight * spectral_leak_loss
            if self.hparams.spectral_ce_auxiliary_weight > 0.0:
                if sparse_ce_fn is None:
                    raise RuntimeError("sparse CE auxiliary was not configured")
                pred_totals = scatter_reduce(
                    src=pred_intensities,
                    index=pred_batch_idxs,
                    reduce="sum",
                    dim_size=batch_size,
                    default=0.0,
                )
                # Clamp the divisor, not the quotient. A spectrum whose predicted
                # intensities sum to zero would otherwise yield NaN here, and the
                # clamp below cannot repair a NaN.
                intensity_tiny = th.finfo(pred_intensities.dtype).tiny
                pred_ce_probs = pred_intensities / pred_totals[pred_batch_idxs].clamp_min(
                    intensity_tiny
                )
                pred_ce_logprobs = pred_ce_probs.clamp_min(intensity_tiny).log()
                spectral_ce_auxiliary = self._spectral_ce_auxiliary(
                    sparse_ce_fn,
                    true_mzs,
                    true_logprobs,
                    true_batch_idxs,
                    pred_mzs,
                    pred_ce_logprobs,
                    pred_batch_idxs,
                    pred_ce_logprobs.new_zeros(batch_size),
                    include_oos=False,
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                )
                loss = loss + (
                    self.hparams.spectral_ce_auxiliary_weight * spectral_ce_auxiliary
                )
            support_metrics: dict[str, th.Tensor] = {}
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
                    attribution=(self.hparams.masster_channel_support_attribution),
                    use_intensity_prior=(
                        self.hparams.masster_channel_support_use_intensity_prior
                    ),
                    tol_per_sample=tolerance_per_sample,
                    min_mz_per_sample=minimum_mz_per_sample,
                )
            if channel_support_metrics_enabled:
                if channel_targets is None:
                    raise ValueError("channel support metrics require dense channel tensors")
                # Monitoring only, and the dominant cost of the channel path: both summaries
                # iterate over the batch in Python.
                support_metrics = _channel_support_metrics(
                    channel_targets=channel_targets,
                    chemical_valid_channels=kwargs["ion_h_chemical_valid_channels"],
                    fragment_batch_idxs=kwargs["ion_h_fragment_batch_idxs"],
                    batch_size=batch_size,
                    fragment_logprobs=kwargs["ion_h_fragment_proposal_logprobs"],
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
                    self.hparams.masster_ion_h_support_loss_weight * ion_h_support_loss
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
                    self.hparams.masster_ion_mode_proposal_loss_weight * ion_mode_proposal_loss
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
                    "barcode_target": self.hparams.masster_proposal_barcode_target,
                }
                if self.hparams.masster_share_proposal_across_ev:
                    proposal_loss = _shared_group_proposal_coverage_loss(
                        **proposal_loss_kwargs,
                        group_ids=kwargs["proposal_group_ids"],
                    )
                else:
                    proposal_loss = _proposal_coverage_loss(**proposal_loss_kwargs)
                loss = loss + self.hparams.masster_proposal_loss_weight * proposal_loss
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
                base_coverage_weighted_spec_ce, _, base_true_oos_logprobs, _ = sparse_ce_fn(
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
                base_conditional_spec_ce = conditional_in_support_cross_entropy(
                    base_coverage_weighted_spec_ce,
                    base_true_oos_logprobs,
                )
                base_primary_loss = (
                    base_conditional_spec_ce
                    if self.hparams.loss_type == "conditional_cross_entropy"
                    else base_coverage_weighted_spec_ce
                )
                loss = loss + self.hparams.masster_base_ce_weight * base_primary_loss
            # The candidate-generation values below are monitoring only and detached:
            # retained_mask_probability is how much of the Bernoulli mask distribution
            # survived truncation, and the counts track the cost of candidate generation.
            # The two optional blocks that follow add candidate_wrecall (how much true
            # intensity the enumerated masks could explain at all, the ceiling on this
            # batch) and the candidate-generation stage timings.
            loss_values = {
                "primary_loss": primary_loss,
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
            }
            if candidate_coverage_enabled:
                loss_values["candidate_wrecall"] = candidate_wrecall
            if candidate_timing_enabled:
                loss_values.update(
                    {
                        "candidate_input_transfer_batch_ms": (
                            candidate_input_transfer_batch_ms.detach()
                        ),
                        "candidate_mask_selection_batch_ms": (
                            candidate_mask_selection_batch_ms.detach()
                        ),
                        "candidate_topology_assembly_batch_ms": (
                            candidate_topology_assembly_batch_ms.detach()
                        ),
                        "candidate_output_transfer_batch_ms": (
                            candidate_output_transfer_batch_ms.detach()
                        ),
                        "candidate_total_batch_ms": candidate_total_batch_ms.detach(),
                        "candidate_total_molecule_ms": candidate_total_molecule_ms.detach(),
                    }
                )
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
            if cross_entropy_loss:
                # Compatibility alias for existing dashboards and checkpoints. New
                # analyses should use one of the two explicit names below.
                loss_values["spec_ce"] = primary_loss
                loss_values["coverage_weighted_spec_ce"] = coverage_weighted_spec_ce
                loss_values["conditional_spec_ce"] = conditional_spec_ce
            else:
                loss_values["spec_peak_distance"] = primary_loss
                loss_values["spec_entropy_match"] = entropy_match
                if escape_intensities is not None:
                    # Without this the escape is invisible: the loss alone cannot say whether
                    # the head learned the coverage deficit or parked at its initial 0.1.
                    escape_total = escape_intensities + pred_intensities.new_zeros(
                        batch_size
                    ).index_add(0, pred_batch_idxs, pred_intensities)
                    loss_values["escape_share"] = (
                        (
                            escape_intensities
                            / escape_total.clamp_min(th.finfo(escape_total.dtype).tiny)
                        )
                        .detach()
                        .to(prediction_reference)
                    )
                if spectral_leak_loss is not None:
                    loss_values["spectral_leak_loss"] = spectral_leak_loss
                if spectral_ce_auxiliary is not None:
                    loss_values["spectral_ce_auxiliary"] = spectral_ce_auxiliary
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
                        "base_coverage_weighted_spec_ce": base_coverage_weighted_spec_ce,
                        "base_conditional_spec_ce": base_conditional_spec_ce,
                        "base_spec_ce": base_primary_loss,
                        "base_primary_loss": base_primary_loss,
                        "base_ce_gain": base_primary_loss.detach() - primary_loss.detach(),
                    }
                )

            if self.hparams.ce_pair_loss_weight != 0.0:
                # The term is train-only (SpectrumPL sets ``ce_pair_enabled``), but the
                # keys must exist on every split so the logged metrics stay aligned;
                # hence the zero vectors when it is skipped.
                batch_size = int(kwargs["batch_size"])
                pair_loss, eligible_endpoints = ce_pair_loss_fn(
                    kwargs, batch_size, pred_logprobs, tolerance_per_sample, minimum_mz_per_sample
                )
                loss = loss + self.hparams.ce_pair_loss_weight * pair_loss
                loss_values["ce_pair_loss"] = pair_loss
                loss_values["ce_pair_eligible_endpoints"] = eligible_endpoints
            if self.training and self._proposal_only_training:
                # Phase 1 trains proposal coverage and any dedicated proposal heads.
                # Intensity-head support and cross-EV losses begin with joint training.
                loss = self.hparams.masster_proposal_loss_weight * proposal_loss
                if ion_mode_proposal_loss_enabled:
                    loss = loss + (
                        self.hparams.masster_ion_mode_proposal_loss_weight
                        * ion_mode_proposal_loss
                    )
            if self.property_names:
                for name in self.property_names:
                    if f"true_{name}" not in kwargs:
                        raise ValueError(f"{name.upper()} training requires spec_params.{name} labels")
                    terms = property_regression_terms(
                        kwargs[f"pred_{name}"],
                        kwargs[f"true_{name}"],
                        self.hparams[f"masster_{name}_scale"],
                    )
                    aux_loss, abs_error, pct_error, valid = terms
                    loss = loss + self.hparams[f"masster_{name}_loss_weight"] * aux_loss
                    loss_values[f"{name}_loss"] = aux_loss
                    loss_values[f"{name}_mae"] = abs_error
                    loss_values[f"{name}_label_fraction"] = valid.to(abs_error)
                    if name == "ccs":
                        loss_values["ccs_mape"] = pct_error
            loss_values["loss"] = loss
            if self.hparams.masster_proposal_method == "connected_fragment_beam":
                loss_values = {
                    fast_metric_names.get(name, name): value
                    for name, value in loss_values.items()
                    if fast_metric_names.get(name, name) is not None
                }
            return loss_values

        self.loss_fn = loss_fn
        # Every name here must be a key of loss_values above: SpectrumPL reduces and logs
        # exactly these, so a missing key fails at the first logging step.
        loss_names = [
            "loss",
            "primary_loss",
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
        if candidate_coverage_enabled:
            loss_names.append("candidate_wrecall")
        if candidate_timing_enabled:
            loss_names.extend(
                [
                    "candidate_input_transfer_batch_ms",
                    "candidate_mask_selection_batch_ms",
                    "candidate_topology_assembly_batch_ms",
                    "candidate_output_transfer_batch_ms",
                    "candidate_total_batch_ms",
                    "candidate_total_molecule_ms",
                ]
            )
        if channel_support_metrics_enabled:
            loss_names.extend(
                [
                    "fragment_precision",
                    "fragment_weighted_precision",
                    "ion_mode_support_positive_rate",
                    "ion_h_support_positive_rate",
                    *_ION_H_REQUIRED_K_METRIC_NAMES,
                ]
            )
        if self.hparams.loss_type in {"conditional_cross_entropy", "cross_entropy"}:
            loss_names.extend(
                ["spec_ce", "coverage_weighted_spec_ce", "conditional_spec_ce"]
            )
        else:
            loss_names.extend(["spec_peak_distance", "spec_entropy_match"])
            if self.hparams.masster_intensity_escape:
                loss_names.append("escape_share")
            if self.hparams.spectral_leak_loss_weight > 0.0:
                loss_names.append("spectral_leak_loss")
            if self.hparams.spectral_ce_auxiliary_weight > 0.0:
                loss_names.append("spectral_ce_auxiliary")
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
                    "base_coverage_weighted_spec_ce",
                    "base_conditional_spec_ce",
                    "base_spec_ce",
                    "base_primary_loss",
                    "base_ce_gain",
                ]
            )
        fast_metric_names = {
            "retained_mask_probability": None,
            "selected_mask_count": "fast_gnn_fragment_count",
            "component_pathway_count": None,
            "mean_mask_cut_count": "fast_gnn_mean_boundary_cut_count",
            "proposal_single_cut_fraction": "fast_gnn_single_boundary_cut_fraction",
            "components_per_mask": None,
            "unique_components_per_mask": None,
            "latent_pathways_per_unique_fragment": None,
            "duplicate_pathway_probability_fraction": None,
            "productive_mask_fraction": None,
            "duplicate_component_pathway_fraction": None,
            "candidate_mask_selection_batch_ms": "fast_gnn_search_batch_ms",
            **{
                f"mask_cut_fraction_{bucket}": f"fast_gnn_boundary_cut_fraction_{bucket}"
                for bucket in ("0", "1", "2", "3", "4", "5", "6plus", "4plus")
            },
        }
        if self.hparams.masster_proposal_method == "connected_fragment_beam":
            loss_names = [
                fast_metric_names.get(name, name)
                for name in loss_names
                if fast_metric_names.get(name, name) is not None
            ]
        for name in self.property_names:
            loss_names.extend([f"{name}_loss", f"{name}_mae", f"{name}_label_fraction"])
            if name == "ccs":
                loss_names.append("ccs_mape")
        self.loss_names = loss_names
        self.metric_names.update(loss_names)

    def _setup_metric_fns(self) -> None:
        """Add validation candidate coverage when the coverage metrics are enabled."""
        super()._setup_metric_fns()
        if not self.hparams.masster_candidate_coverage_metrics:
            # Without the coverage metrics nothing here needs the candidate m/z values,
            # so the base metric function is left untouched.
            return
        cut_diagnostics = self.hparams.masster_candidate_cut_diagnostics
        coverage_metric_names = (
            {
                f"candidate_{metric}_cut_{bucket}"
                for bucket in _CUT_COUNT_BUCKETS
                for metric in ("recall", "wrecall", "precision")
            }
            if cut_diagnostics
            else set()
        )
        coverage_metric_names.add("candidate_recall")
        # Marginal selection keeps only part of the enumerated pool; the pool's own
        # coverage is the ceiling for any selection from it.
        pool_coverage = self.hparams.masster_fragment_selection == "marginal"
        if pool_coverage:
            coverage_metric_names.add("pool_candidate_wrecall")
        for metric_name in coverage_metric_names:
            self.metric_min_max[metric_name] = None
        self.auxiliary_metric_names.update(coverage_metric_names)
        self.metric_names.update(coverage_metric_names)
        self.extra_metric_args.extend(
            [
                "candidate_mzs_by_cut",
                "candidate_batch_idxs_by_cut",
            ]
        )
        if cut_diagnostics:
            self.extra_metric_args.append("candidate_cut_counts")
        if pool_coverage:
            self.extra_metric_args.extend(["pool_candidate_mzs", "pool_candidate_batch_idxs"])
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
                candidate_cut_counts=kwargs.get("candidate_cut_counts"),
                include_cut_buckets=cut_diagnostics,
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
            if pool_coverage:
                coverage["pool_candidate_wrecall"] = _candidate_weighted_recall(
                    true_mzs=kwargs["true_mzs"],
                    true_ints=kwargs["true_ints"],
                    true_batch_idxs=kwargs["true_batch_idxs"],
                    candidate_mzs=kwargs["pool_candidate_mzs"],
                    candidate_batch_idxs=kwargs["pool_candidate_batch_idxs"],
                    batch_size=kwargs["true_prec_mzs"].shape[0],
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                    tol_per_sample=tol_per_sample,
                    min_mz_per_sample=min_mz_per_sample,
                )
            metrics.update(coverage)
            return metrics

        self.metric_fn = metric_fn

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Preserve optimizer parameter alignment for combined-wrapper checkpoints."""
        state = checkpoint["state_dict"]
        legacy_prefix = "model.rt_ccs_heads."
        if any(key.startswith(legacy_prefix) for key in state):
            renames = {
                legacy_prefix + "rt_head.": "model.rt_head.regressor.",
                legacy_prefix + "ccs_head.": "model.ccs_head.regressor.",
                legacy_prefix + "adduct_embedding.": "model.ccs_head.adduct_embedding.",
            }
            old_order = {}
            for index, key in enumerate(state):
                for old, new in renames.items():
                    if key.startswith(old):
                        key = new + key[len(old):]
                        break
                old_order[key] = index
            parameter_names = {id(parameter): name for name, parameter in self.named_parameters()}
            groups = self._weight_decay_parameter_groups()
            if groups is None:
                groups = [{"params": list(self.parameters())}]
            for optimizer_state in checkpoint.get("optimizer_states", []):
                saved_groups = optimizer_state["param_groups"]
                if len(saved_groups) != len(groups):
                    raise ValueError("Cannot migrate RT/CCS optimizer with changed parameter groups")
                for saved_group, group in zip(saved_groups, groups):
                    names = [parameter_names[id(parameter)] for parameter in group["params"]]
                    if len(names) != len(saved_group["params"]):
                        raise ValueError("Cannot migrate RT/CCS optimizer with changed parameters")
                    ids_by_name = dict(zip(sorted(names, key=old_order.__getitem__), saved_group["params"]))
                    saved_group["params"] = [ids_by_name[name] for name in names]
        super().on_load_checkpoint(checkpoint)

    def _accumulate_running_metrics(self, batch_results: dict, split: str) -> None:
        """Exclude missing property labels from epoch metric denominators."""
        super()._accumulate_running_metrics(batch_results, split)
        if not self.property_names:
            return
        weights = self._get_sample_weights(
            split,
            batch_results["spec_per_group"],
            batch_results["spec_per_mol"],
            batch_results["group_per_mol"],
        )
        sum_w = getattr(self, f"{split}_metric_sum_w")
        for name in self.property_names:
            invalid = 1.0 - batch_results[f"{name}_label_fraction"]
            missing_weight = (weights.to(invalid) * invalid).sum().detach()
            metrics = [f"{name}_loss", f"{name}_mae"]
            if name == "ccs":
                metrics.append("ccs_mape")
            for metric in metrics:
                sum_w[metric] = sum_w[metric] - missing_weight

    def _memorization_sink_indices(self, batch_kwargs: dict) -> th.Tensor | None:
        """Return the memorization-sink row of each spectrum in a training batch, else None.

        The sink is a training-only random effect indexed by ``mol_id``. Evaluation, inference,
        and the epochs before ``masster_memorization_sink_start_epoch`` get ``None``, which
        runs the shared model.

        Args:
            batch_kwargs: The collated batch; ``mol_id`` holds one raw id per spectrum.

        Returns:
            Long tensor of shape ``(batch_size,)`` on the batch's device, or ``None``.
        """
        if (
            self.hparams.masster_memorization_sink_dim == 0
            or not self.training
            or self.current_epoch < self.hparams.masster_memorization_sink_start_epoch
        ):
            return None
        return th.tensor(
            [int(mol_id) for mol_id in batch_kwargs["mol_id"]],
            dtype=th.long,
            device=batch_kwargs["spec_prec_type"].device,
        )

    def forward(self, **batch_kwargs) -> dict[str, th.Tensor]:
        """Select the MASSter inputs from SpectrumPL's collated batch.

        The collated batch carries fields for every model in the repo, so only the ones
        MASSter uses are forwarded. Optional metadata is read with ``get`` because
        the dataset omits those keys entirely when the corresponding spec_param is off,
        and the model then also has no embedding expecting them.

        Args:
            **batch_kwargs: The collated batch produced by SpectrumPL's dataloader.

        Returns:
            The model's sparse prediction dictionary, forwarded unchanged.
        """
        # Standalone inference bypasses Lightning's transfer hook. Match the same
        # templates here when that path has not supplied the CPU graph sidecar.
        static_graph = batch_kwargs.get("masster_static_graph")
        if self.hparams.masster_smarts_templates and (
            static_graph is None or static_graph.smarts_templates is None
        ):
            static_graph = self._build_static_graph(batch_kwargs)
        output = self.model(
            mol_pyg=batch_kwargs["mol_pyg"],
            spec_prec_type=batch_kwargs["spec_prec_type"],
            spec_inst_type=batch_kwargs.get("spec_inst_type"),
            spec_frag_mode=batch_kwargs.get("spec_frag_mode"),
            spec_nce=batch_kwargs.get("spec_nce"),
            spec_nce_batch_idxs=batch_kwargs.get("spec_nce_batch_idxs"),
            spec_ace=batch_kwargs.get("spec_ace"),
            spec_ace_batch_idxs=batch_kwargs.get("spec_ace_batch_idxs"),
            spec_nce_weights=batch_kwargs.get("spec_nce_weights"),
            spec_ace_weights=batch_kwargs.get("spec_ace_weights"),
            static_graph=static_graph,
            proposal_group_ids=(
                batch_kwargs.get("group_id")
                if self.hparams.masster_share_proposal_across_ev
                else None
            ),
            memorization_sink_indices=self._memorization_sink_indices(batch_kwargs),
            # The exporter's switch for the per-peak annotation; absent in training.
            return_annotation=bool(
                batch_kwargs.get(
                    "return_explainability_tensors",
                    RETURN_EXPLAINABILITY_TENSORS_DEFAULT["masster"],
                )
            ),
        )
        if (
            self.hparams.masster_share_proposal_across_ev
            and self.hparams.masster_proposal_loss_weight > 0.0
        ):
            output["proposal_group_ids"] = batch_kwargs["group_id"]
        if self.property_names:
            # Targets travel only to the loss, never into the model. Inference can omit them.
            for name in self.property_names:
                if f"spec_{name}" in batch_kwargs:
                    output[f"true_{name}"] = batch_kwargs[f"spec_{name}"]
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

    def _build_static_graph(self, batch: dict[str, Any]) -> MassterStaticGraph:
        """Build CPU topology and optional templates for training or inference."""
        mol_pyg = batch["mol_pyg"]
        return MassterStaticGraph(
            chemical_bond_index=mol_pyg.chemical_bond_index.detach().cpu(),
            atom_batch_indices=mol_pyg.batch.detach().cpu(),
            atom_ptr=mol_pyg.ptr.detach().cpu(),
            smarts_templates=(
                tuple(_cached_masster_templates(str(smiles)) for smiles in batch["mol_smiles"])
                if self.hparams.masster_smarts_templates
                else None
            ),
        )

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: th.device,
        dataloader_idx: int,
    ) -> Any:
        """Keep static candidate-generation graph indices on CPU.

        Both backends need host graph counts and endpoints. Snapshotting them before
        device transfer avoids downloading graph indices during candidate construction.
        The CUDA backend keeps proposal logits and selected masks on the device.

        Args:
            batch: The collated batch, still on CPU.
            device: Target accelerator device.
            dataloader_idx: Index of the dataloader that produced the batch.

        Returns:
            The transferred batch, with the CPU-resident graph indices attached under
            ``masster_static_graph``.
        """
        static_graph = self._build_static_graph(batch)
        transferred = super().transfer_batch_to_device(batch, device, dataloader_idx)
        transferred["masster_static_graph"] = static_graph
        return transferred
