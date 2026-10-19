"""Padded layouts for flat, batch-indexed peak lists, and the chunking that bounds their
memory.
"""

import torch as th

from fragnnet.utils.misc_utils import bincount_in_range


def paired_batch_offsets(
    true_batch_idxs: th.Tensor, pred_batch_idxs: th.Tensor, batch_size: int
) -> tuple[th.Tensor, th.Tensor, list[int], list[int]]:
    """Count both peak lists and transfer cumulative offsets together.

    Offsets describe contiguous slices only when each input is sorted by batch index.
    Empty rows are retained. Callers own sorting, empty-input policy and chunk selection.
    """
    true_counts = bincount_in_range(true_batch_idxs, batch_size)
    pred_counts = bincount_in_range(pred_batch_idxs, batch_size)
    offsets = th.cat(
        (
            true_counts.new_zeros(1),
            true_counts.cumsum(0),
            pred_counts.new_zeros(1),
            pred_counts.cumsum(0),
        )
    ).tolist()
    return true_counts, pred_counts, offsets[: batch_size + 1], offsets[batch_size + 1 :]


def build_padded_batch_layout(
    batch_idxs: th.Tensor,
    batch_size: int,
    assume_sorted: bool = False,
    counts: th.Tensor | None = None,
    max_count: int | None = None,
) -> tuple[th.Tensor | None, th.Tensor, th.Tensor, int]:
    """Build reusable indices for mapping flat ragged values to padded batch rows."""
    if counts is None:
        counts = th.bincount(batch_idxs, minlength=batch_size)
    if max_count is None:
        max_count = int(counts.max().item()) if counts.numel() > 0 else 0
    valid = th.arange(max_count, device=batch_idxs.device).unsqueeze(0) < counts.unsqueeze(1)
    if batch_idxs.numel() == 0 or max_count == 0:
        return None, batch_idxs.new_empty(0), valid, max_count

    sort_idx = None if assume_sorted else th.argsort(batch_idxs)
    sorted_batch_idxs = batch_idxs if sort_idx is None else batch_idxs[sort_idx]
    row_starts = th.cumsum(counts, dim=0) - counts
    positions = th.arange(batch_idxs.numel(), device=batch_idxs.device) - th.repeat_interleave(
        row_starts, counts, output_size=batch_idxs.numel()
    )
    padded_indices = sorted_batch_idxs * max_count + positions
    return sort_idx, padded_indices, valid, max_count


def pad_flat_with_layout(
    values: th.Tensor,
    sort_idx: th.Tensor | None,
    padded_indices: th.Tensor,
    batch_size: int,
    max_count: int,
    pad_value: float,
) -> th.Tensor:
    """Apply a reusable padded-batch layout while preserving gradients to ``values``."""
    padded = values.new_full((batch_size, max_count), pad_value)
    if values.numel() == 0 or max_count == 0:
        return padded
    sorted_values = values if sort_idx is None else values[sort_idx]
    padded.reshape(-1)[padded_indices] = sorted_values
    return padded


def build_loss_chunks(
    true_batch_cumsum: list[int],
    pred_batch_cumsum: list[int],
    loss_batch_size: int,
) -> list[tuple[int, int]]:
    """Build contiguous chunks whose padded pairs do not exceed flat Cartesian pairs."""
    if loss_batch_size <= 0:
        raise ValueError(f"loss_batch_size must be > 0, got {loss_batch_size}")

    batch_size = len(true_batch_cumsum) - 1
    chunks: list[tuple[int, int]] = []

    for window_lower in range(0, batch_size, loss_batch_size):
        window_upper = min(window_lower + loss_batch_size, batch_size)
        lower = window_lower

        while lower < window_upper:
            upper = lower
            max_true = 0
            max_pred = 0
            sum_true = 0
            sum_pred = 0

            while upper < window_upper:
                true_count = true_batch_cumsum[upper + 1] - true_batch_cumsum[upper]
                pred_count = pred_batch_cumsum[upper + 1] - pred_batch_cumsum[upper]
                candidate_size = upper - lower + 1
                candidate_max_true = max(max_true, true_count)
                candidate_max_pred = max(max_pred, pred_count)
                candidate_sum_true = sum_true + true_count
                candidate_sum_pred = sum_pred + pred_count
                padded_pairs = candidate_size * candidate_max_true * candidate_max_pred
                flat_pairs = candidate_sum_true * candidate_sum_pred

                if upper > lower and padded_pairs > flat_pairs:
                    break

                upper += 1
                max_true = candidate_max_true
                max_pred = candidate_max_pred
                sum_true = candidate_sum_true
                sum_pred = candidate_sum_pred

            chunks.append((lower, upper))
            lower = upper

    return chunks


def pad_flat_by_batch(
    values: th.Tensor,
    batch_idxs: th.Tensor,
    batch_size: int,
    pad_value: float,
) -> tuple[th.Tensor, th.Tensor]:
    """Convert flat ragged values plus batch indices into a padded [B, N] tensor."""
    sort_idx, padded_indices, valid, max_count = build_padded_batch_layout(batch_idxs, batch_size)
    padded = pad_flat_with_layout(
        values,
        sort_idx,
        padded_indices,
        batch_size,
        max_count,
        pad_value,
    )
    return padded, valid
