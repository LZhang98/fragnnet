"""Peak-list distance on unbinned spectra, and the intensity fraction predicted outside
every observed tolerance window.
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
from fragnnet.utils.misc_utils import TOLERANCE_MIN_MZ
from fragnnet.utils.scatter_utils import x_log_x
from fragnnet.utils.spec_similarity import align_sparse_peak_support, calculate_match_mzs
from fragnnet.utils.spec_utils import round_aggregate_peaks


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
