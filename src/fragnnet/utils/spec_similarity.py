"""Spectrum similarity: m/z peak matching within a tolerance, Hungarian peak assignment
(torch-linear-assignment on GPU when installed, SciPy otherwise), and the cosine, Jensen-Shannon
and NDCG similarity helpers (split out of ``utils/spec_utils.py``).
"""

import logging
from typing import Literal

import numpy as np
import torch as th
import torch.nn.functional as F

from fragnnet.utils.misc_utils import EPS, TOLERANCE_MIN_MZ, th_setdiff1d
from fragnnet.utils.scatter_utils import (
    safelog,
    scatter_l1normalize,
    scatter_l2normalize,
    scatter_reduce,
)

logger = logging.getLogger(__name__)

try:
    from torch_linear_assignment import (  # pyright: ignore[reportMissingImports]
        assignment_to_indices,
        batch_linear_assignment,
    )

    _torch_linear_assignment = True
    logger.info("torch_linear_assignment found; using GPU-accelerated batch linear assignment")
except ImportError:
    _torch_linear_assignment = False
    logger.info(
        "torch_linear_assignment not found; falling back to "
        "scipy.optimize.linear_sum_assignment (slower, CPU-only)"
    )


_assignment_device_logged = False


def _log_assignment_device_once(device) -> None:
    global _assignment_device_logged
    if not _assignment_device_logged:
        logger.info("torch_linear_assignment running on device: %s", device)
        _assignment_device_logged = True


def _as_scoring_dtype(tensor: th.Tensor) -> th.Tensor:
    """Upcast half and bfloat16 tensors to float32 and leave float32/float64 alone.

    Low-precision inputs arrive under mixed-precision training and break the peak
    matching in three ways: SciPy cannot convert bfloat16, the CUDA assignment backend
    has no half kernel, and the infeasible-pair sentinel plus the Jensen-Shannon log
    terms want more headroom than half provides. Keeping the policy in one place means
    a change to it cannot be applied to only some of the call sites.
    """
    if tensor.dtype not in (th.float32, th.float64):
        return tensor.float()
    return tensor


def _solve_batch_assignment(batch_score: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Solve a batched linear assignment problem, maximizing total score.

    Args:
        batch_score: Score matrix of shape (batch, rows, cols).

    Returns:
        Tuple of (row_ind, col_ind) tensors of shape (batch, K) where
        K = min(rows, cols), giving the optimal assignment indices.
    """
    batch_score = _as_scoring_dtype(batch_score)
    if _torch_linear_assignment:
        assignment = batch_linear_assignment(-batch_score)
        _log_assignment_device_once(batch_score.device)
        return assignment_to_indices(assignment)

    device = batch_score.device
    if batch_score.shape[0] == 0:
        row_ind = th.empty((0, 0), device=device, dtype=th.long)
        col_ind = th.empty((0, 0), device=device, dtype=th.long)
        return row_ind, col_ind

    row_inds, col_inds = [], []
    for b in range(batch_score.shape[0]):
        b_row_ind, b_col_ind = scipy_linear_sum_assignment(batch_score[b], maximize=True)
        row_inds.append(b_row_ind.detach().cpu().numpy())
        col_inds.append(b_col_ind.detach().cpu().numpy())
    row_ind = th.tensor(np.stack(row_inds), device=device, dtype=th.long)
    col_ind = th.tensor(np.stack(col_inds), device=device, dtype=th.long)
    return row_ind, col_ind


def align_sparse_peak_support(
    true_mzs: th.Tensor,
    true_intensities: th.Tensor,
    true_valid: th.Tensor,
    pred_mzs: th.Tensor,
    pred_intensities: th.Tensor,
    pred_valid: th.Tensor,
    match_mask: th.Tensor,
    *,
    matching: Literal["closest", "mass_hungarian", "js_gain_hungarian"],
) -> tuple[th.Tensor, th.Tensor]:
    """Align padded sparse spectra onto a shared true-plus-unmatched support.

    The matching policy is explicit because m/z-nearest correspondence and
    Jensen-Shannon-gain correspondence are different objectives. Returned vectors
    contain one position for every true peak plus one position for every predicted
    peak; unmatched predictions occupy the latter positions.

    Args:
        true_mzs: Padded observed m/z values, shape ``(batch, num_true)``.
        true_intensities: Padded observed intensities with the same shape.
        true_valid: Valid observed-position mask, shape ``(batch, num_true)``.
        pred_mzs: Padded predicted m/z values, shape ``(batch, num_pred)``.
        pred_intensities: Padded predicted intensities with the same shape.
        pred_valid: Valid predicted-position mask, shape ``(batch, num_pred)``.
        match_mask: Feasible m/z pairs, shape ``(batch, num_true, num_pred)``.
        matching: ``closest`` permits many predictions per true peak;
            ``mass_hungarian`` maximizes cardinality then minimizes m/z distance;
            ``js_gain_hungarian`` maximizes Jensen-Shannon gain.

    Returns:
        Observed and predicted aligned intensities, each of shape
        ``(batch, num_true + num_pred)``.
    """
    if true_mzs.shape != true_intensities.shape or true_mzs.shape != true_valid.shape:
        raise ValueError("true m/z, intensities, and validity must have matching shapes")
    if pred_mzs.shape != pred_intensities.shape or pred_mzs.shape != pred_valid.shape:
        raise ValueError("predicted m/z, intensities, and validity must have matching shapes")
    expected_match_shape = (*true_mzs.shape, pred_mzs.shape[1])
    if match_mask.shape != expected_match_shape:
        raise ValueError(f"match_mask must have shape {expected_match_shape}")
    if matching not in {"closest", "mass_hungarian", "js_gain_hungarian"}:
        raise ValueError(f"unknown sparse peak matching policy: {matching}")

    batch_size, max_true = true_mzs.shape
    max_pred = pred_mzs.shape[1]
    valid_pairs = match_mask & true_valid[:, :, None] & pred_valid[:, None, :]
    mass_distance = (true_mzs[:, :, None] - pred_mzs[:, None, :]).abs()

    if matching == "closest":
        matched_pred = valid_pairs.any(dim=1)
        matched_true_of_pred = mass_distance.masked_fill(~valid_pairs, th.inf).argmin(dim=1)
    else:
        if matching == "mass_hungarian":
            feasible_distance = mass_distance.masked_fill(~valid_pairs, 0.0)
            invalid_cost = (feasible_distance.flatten(1).amax(dim=1).detach() + 1.0) * (
                min(max_true, max_pred) + 1
            )
            assignment_score = -th.where(
                valid_pairs,
                mass_distance,
                invalid_cost[:, None, None].expand_as(mass_distance),
            )
        else:
            true_probs = F.normalize(true_intensities, p=1, dim=1)
            pred_probs = F.normalize(pred_intensities, p=1, dim=1)
            pair_mass = true_probs[:, :, None] + pred_probs[:, None, :]
            assignment_score = 0.5 * (
                pair_mass * th.log(pair_mass + EPS)
                - true_probs[:, :, None] * th.log(true_probs[:, :, None] + EPS)
                - pred_probs[:, None, :] * th.log(pred_probs[:, None, :] + EPS)
            )
            # float() first: the infeasible sentinel overflows float16.
            assignment_score = assignment_score.float().masked_fill(~valid_pairs, -1e9)

        row_indices, column_indices = _solve_batch_assignment(assignment_score.detach())
        expected_shape = (batch_size, min(max_true, max_pred))
        if tuple(row_indices.shape) != expected_shape:
            raise RuntimeError(
                "batched linear assignment returned indices of shape "
                f"{tuple(row_indices.shape)}, expected {expected_shape}"
            )
        batch_indices = th.arange(batch_size, device=true_mzs.device)[:, None]
        valid_assignments = valid_pairs[batch_indices, row_indices, column_indices]
        matched_pred = pred_valid.new_zeros(pred_valid.shape).scatter(
            1, column_indices, valid_assignments
        )
        matched_true_of_pred = th.zeros(
            (batch_size, max_pred), dtype=th.long, device=true_mzs.device
        ).scatter(1, column_indices, row_indices)

    matched_contribution = th.where(
        matched_pred,
        pred_intensities,
        th.zeros_like(pred_intensities),
    )
    pred_at_true = pred_intensities.new_zeros((batch_size, max_true)).scatter_add(
        1,
        matched_true_of_pred,
        matched_contribution,
    )
    unmatched_pred = th.where(
        pred_valid & ~matched_pred,
        pred_intensities,
        th.zeros_like(pred_intensities),
    )
    return (
        th.cat([true_intensities, true_intensities.new_zeros((batch_size, max_pred))], dim=1),
        th.cat([pred_at_true, unmatched_pred], dim=1),
    )


def calculate_match_mzs(
    true_mzs: th.Tensor | np.ndarray,
    pred_mzs: th.Tensor | np.ndarray,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    pred_mz_divisor: bool = False,
    tol_per_true: th.Tensor | None = None,
    min_mz_per_true: th.Tensor | None = None,
) -> th.Tensor | np.ndarray:
    """
    Method to match two spectra based on m/z, return a N x M matrix, where N is number of mz in true_mzs, M is number of mz in pred_mzs
    Each cell i,j means if true_mzs[i] matches pred_mzs[j]
    works with numpy arrays or torch tensors inspired by the function in ms-pred
    The torch path broadcasts over leading dimensions, so padded ``[batch, N]`` and
    ``[batch, M]`` inputs return a ``[batch, N, M]`` mask.
    Args:
        true_mzs (th.Tensor): 1d flat tensor of true m/zs; multiple m/z lists are concatenated.
        pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs; multiple m/z lists are concatenated.
        tolerance (float, optional): Tolerance; Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
        relative (bool, optional): Flag to use a relative measure. Defaults to True.
        tolerance_min_mz (float, optional): Divisor floor used in relative measure. Defaults to 200 Da.
        pred_mz_divisor (bool, optional): If the measure is relative, whether to use the pred_mzs as divisors (instead of the true_mzs). Defaults to False.
        tol_per_true (th.Tensor | None): Per-true-peak tolerance of shape [N_true]. When provided,
            overrides the scalar ``tolerance`` for each true peak (torch path only).
        min_mz_per_true (th.Tensor | None): Per-true-peak divisor floor of shape [N_true]. When
            provided, overrides the scalar ``tolerance_min_mz`` for each true peak (torch path only).
    Returns:
        th.Tensor: return a N x M matrix of True or False, dim 0 for ground truth, dim 1 for predicted
    """

    if isinstance(true_mzs, th.Tensor) and isinstance(pred_mzs, th.Tensor):
        with th.autocast(device_type="cuda", dtype=th.bfloat16):  # speed up and save mem
            diff_mzs = th.abs(true_mzs[..., None] - pred_mzs[..., None, :])
            if tol_per_true is not None:
                # Per-true-peak tolerance: shape [N_true, 1] broadcasts over [N_true, N_pred]
                min_diff_mzs = tol_per_true[..., None]
                if relative:
                    if pred_mz_divisor:
                        divisor_mzs = pred_mzs.clone()
                        divisor_mzs[divisor_mzs < tolerance_min_mz] = tolerance_min_mz
                        diff_mzs = diff_mzs / divisor_mzs[..., None]
                    else:
                        divisor_mzs = true_mzs.clone()
                        if min_mz_per_true is not None:
                            divisor_mzs = th.maximum(divisor_mzs, min_mz_per_true)
                        else:
                            divisor_mzs[divisor_mzs < tolerance_min_mz] = tolerance_min_mz
                        diff_mzs = diff_mzs / divisor_mzs[..., None]
            else:
                min_diff_mzs = tolerance
                if relative:
                    if pred_mz_divisor:
                        divisor_mzs = pred_mzs.clone()
                    else:
                        divisor_mzs = true_mzs.clone()
                    divisor_mzs[divisor_mzs < tolerance_min_mz] = tolerance_min_mz
                    diff_mzs = diff_mzs / divisor_mzs[..., None]
        return diff_mzs < min_diff_mzs

    if isinstance(true_mzs, np.ndarray) and isinstance(pred_mzs, np.ndarray):
        diff_mzs = np.abs(true_mzs[..., None] - pred_mzs[..., None, :])
        min_diff_mzs = tolerance
        if relative:
            if pred_mz_divisor:
                divisor_mzs = np.copy(pred_mzs)
            else:
                divisor_mzs = np.copy(true_mzs)
            divisor_mzs[divisor_mzs < tolerance_min_mz] = tolerance_min_mz
            diff_mzs = diff_mzs / divisor_mzs[..., None]
        return diff_mzs < min_diff_mzs

    raise ValueError("true_mzs and pred_mzs should be both torch tensors or numpy arrays")


def calculate_match_mz_reductions(
    true_mzs: th.Tensor,
    pred_mzs: th.Tensor,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    pred_mz_divisor: bool = False,
    tol_per_true: th.Tensor | None = None,
    min_mz_per_true: th.Tensor | None = None,
    chunk_size: int = 1024,
) -> tuple[th.Tensor, th.Tensor]:
    """Return whether each true and predicted m/z has any match.

    Predicted peaks are processed in chunks so recall and precision metrics do
    not materialize the full floating-point true-by-predicted difference tensor.
    The returned tensors have the same shapes as ``true_mzs`` and ``pred_mzs``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    true_match_mask = th.zeros_like(true_mzs, dtype=th.bool)
    pred_match_mask = th.zeros_like(pred_mzs, dtype=th.bool)
    for start in range(0, pred_mzs.shape[-1], chunk_size):
        stop = min(start + chunk_size, pred_mzs.shape[-1])
        match_mask = calculate_match_mzs(
            true_mzs,
            pred_mzs[..., start:stop],
            tolerance=tolerance,
            relative=relative,
            tolerance_min_mz=tolerance_min_mz,
            pred_mz_divisor=pred_mz_divisor,
            tol_per_true=tol_per_true,
            min_mz_per_true=min_mz_per_true,
        )
        true_match_mask |= match_mask.any(dim=-1)
        pred_match_mask[..., start:stop] = match_mask.any(dim=-2)

    return true_match_mask, pred_match_mask


def cos_sim_helper(
    true_bin_idxs,
    true_bin_ints,
    true_bin_batch_idxs,
    pred_bin_idxs,
    pred_bin_ints,
    pred_bin_batch_idxs,
):
    """_summary_

    Args:
        true_bin_idxs (_type_): _description_
        true_bin_ints (_type_): _description_
        true_bin_batch_idxs (_type_): _description_
        pred_bin_idxs (_type_): _description_
        pred_bin_ints (_type_): _description_
        pred_bin_batch_idxs (_type_): _description_

    Returns:
        _type_: _description_
    """
    # l2 normalize
    true_bin_ints = scatter_l2normalize(true_bin_ints, true_bin_batch_idxs)
    pred_bin_ints = scatter_l2normalize(pred_bin_ints, pred_bin_batch_idxs)

    pred_mask = th.isin(pred_bin_idxs, true_bin_idxs)
    true_mask = th.isin(true_bin_idxs, pred_bin_idxs)
    both_bin_ints = pred_bin_ints[pred_mask] * true_bin_ints[true_mask]
    assert th.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
    dim_size = int(th.max(true_bin_batch_idxs).item()) + 1
    if pred_mask.sum() == 0:
        cos_sims = scatter_reduce(
            src=0.0 * pred_bin_ints,
            index=pred_bin_batch_idxs,
            reduce="sum",
            dim_size=dim_size,
        )
    else:
        cos_sims = scatter_reduce(
            src=both_bin_ints,
            index=pred_bin_batch_idxs[pred_mask],
            reduce="sum",
            dim_size=dim_size,
        )
    return cos_sims


def scipy_linear_sum_assignment(
    matrix: th.Tensor, maximize: bool = False
) -> tuple[th.Tensor, th.Tensor]:
    """Solve linear assignment via scipy (CPU fallback).

    Args:
        matrix: Cost (or score) matrix of shape (N, M). Caller must detach from graph.
        maximize: If True, maximize total score instead of minimizing cost.

    Returns:
        Tuple of (row_ind, col_ind) long tensors on the original device.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("scipy is required for scipy_linear_sum_assignment") from exc

    device = matrix.device
    np_matrix = matrix.cpu().numpy()
    x_idx, y_idx = linear_sum_assignment(np_matrix, maximize=maximize)
    return th.tensor(x_idx, device=device).long(), th.tensor(y_idx, device=device).long()


def linear_sum_assignment(matrix: th.Tensor, maximize: bool = False) -> tuple[th.Tensor, th.Tensor]:
    """Solve linear assignment, using GPU via torch_linear_assignment when available.

    Uses ``batch_linear_assignment`` (GPU) when ``torch_linear_assignment`` is installed and
    the matrix is on a CUDA device.  Falls back to scipy otherwise.

    Caller is responsible for detaching ``matrix`` from the autograd graph before calling.

    Args:
        matrix: Cost (or score) matrix of shape (N, M).
        maximize: If True, maximize total score instead of minimizing cost.

    Returns:
        Tuple of (row_ind, col_ind) long tensors on the same device as ``matrix``.
    """
    if _torch_linear_assignment and matrix.is_cuda:
        cost = -matrix if maximize else matrix
        # batch_linear_assignment expects (B, N, M) and minimises; returns (B, N) col-index per row
        assignment = batch_linear_assignment(cost.unsqueeze(0)).squeeze(0)  # (N,)
        row_ind = th.where(assignment >= 0)[0]
        col_ind = assignment[row_ind]
        return row_ind, col_ind
    return scipy_linear_sum_assignment(matrix, maximize=maximize)


def opt_cos_sim_helper(
    true_bin_idxs,
    true_bin_ints,
    true_bin_batch_idxs,
    pred_bin_idxs,
    pred_bin_ints,
    pred_bin_batch_idxs,
):
    """Compute the oracle (upper-bound) cosine similarity for a batch of spectrum pairs.

    Replaces predicted intensities at matching bin positions with the true intensities
    and zeroes out all non-matching predicted bins.  The result is the cosine similarity
    between the true spectrum and this oracle prediction — an upper bound on what the
    model could achieve given perfect intensity prediction at the correct peak positions.

    Args:
        true_bin_idxs: Bin indices for true spectra of shape (M_true,).
        true_bin_ints: True intensities of shape (M_true,).
        true_bin_batch_idxs: Batch indices for true peaks of shape (M_true,).
        pred_bin_idxs: Bin indices for predicted spectra of shape (M_pred,).
        pred_bin_ints: Predicted intensities of shape (M_pred,). Used only for
            structural matching; values at non-overlapping bins are zeroed.
        pred_bin_batch_idxs: Batch indices for predicted peaks of shape (M_pred,).

    Returns:
        Tensor of shape (B,) with oracle cosine similarity in [0, 1] per batch item.
    """
    pred_opt_mask = th.isin(pred_bin_idxs, true_bin_idxs)
    true_opt_mask = th.isin(true_bin_idxs, pred_bin_idxs[pred_opt_mask])
    pred_opt_bin_ints = pred_bin_ints.clone()
    pred_opt_bin_ints[~pred_opt_mask] = 0.0
    pred_opt_bin_ints[pred_opt_mask] = true_bin_ints[true_opt_mask]
    opt_cos_sim = cos_sim_helper(
        true_bin_idxs,
        true_bin_ints,
        true_bin_batch_idxs,
        pred_bin_idxs,
        pred_opt_bin_ints,
        pred_bin_batch_idxs,
    )
    return opt_cos_sim


def batch_cos_hun_helper(
    batch_true_ints,
    batch_pred_ints,
    batch_match_mask,
    batch_true_match_mask,
    batch_pred_match_mask,
    remove_prec_peak,
    batch_true_prec_mask,
    batch_pred_prec_mask,
):
    """Compute cosine-Hungarian similarity for a padded batch of spectrum pairs.

    Batched analogue of ``cos_hun_helper``.  All spectra are zero-padded to the
    same peak-count along their respective dimensions.  The function optionally
    zeroes precursor peaks, L2-normalises each spectrum, restricts the score
    matrix to mutually-matchable peaks, solves the linear assignment problem for
    all batch items simultaneously, and returns the dot-product sum of matched
    normalised intensities per item.

    Args:
        batch_true_ints: True intensities of shape (B, N_true).
        batch_pred_ints: Predicted intensities of shape (B, N_pred).
        batch_match_mask: Boolean match tensor of shape (B, N_true, N_pred);
            True where true peak i matches predicted peak j within tolerance.
        batch_true_match_mask: Boolean mask of shape (B, N_true); True where a
            true peak has at least one matching predicted peak.
        batch_pred_match_mask: Boolean mask of shape (B, N_pred); True where a
            predicted peak has at least one matching true peak.
        remove_prec_peak: If True, zero precursor peaks before scoring.
        batch_true_prec_mask: Boolean mask of shape (B, N_true). Ignored when
            remove_prec_peak is False.
        batch_pred_prec_mask: Boolean mask of shape (B, N_pred). Ignored when
            remove_prec_peak is False.

    Returns:
        Tensor of shape (B,) with cosine-Hungarian similarity in [0, 1] per
        batch item.
    """
    if remove_prec_peak:
        assert (
            batch_true_ints.shape == batch_true_prec_mask.shape
        ), f"{batch_true_ints.shape} {batch_true_prec_mask.shape}"
        assert (
            batch_pred_ints.shape == batch_pred_prec_mask.shape
        ), f"{batch_pred_ints.shape} {batch_pred_prec_mask.shape}"
        batch_true_ints = batch_true_ints * (1 - batch_true_prec_mask.float())
        batch_pred_ints = batch_pred_ints * (1 - batch_pred_prec_mask.float())
    assert len(batch_true_ints.shape) == 2
    assert len(batch_pred_ints.shape) == 2
    batch_true_ints = F.normalize(batch_true_ints, p=2, dim=-1)
    batch_pred_ints = F.normalize(batch_pred_ints, p=2, dim=-1)
    true_batch_pos_idx = th.argsort(batch_true_match_mask.to(th.int64), descending=True)
    pred_batch_pos_idx = th.argsort(batch_pred_match_mask.to(th.int64), descending=True)
    max_true_match_pos = batch_true_match_mask.float().sum(-1).long().max()
    max_pred_match_pos = batch_pred_match_mask.float().sum(-1).long().max()
    true_batch_pos_idx = true_batch_pos_idx[:, :max_true_match_pos]
    pred_batch_pos_idx = pred_batch_pos_idx[:, :max_pred_match_pos]
    batch_idx = th.arange(true_batch_pos_idx.shape[0], device=true_batch_pos_idx.device)[:, None]
    sub_mask = batch_match_mask[
        batch_idx[..., None], true_batch_pos_idx[:, :, None], pred_batch_pos_idx[:, None, :]
    ]
    sub_score = batch_true_ints[batch_idx, true_batch_pos_idx][..., None] * (
        batch_pred_ints[batch_idx, pred_batch_pos_idx][..., None, :]
    )
    assert sub_mask.shape == sub_score.shape, f"{sub_mask.shape} {sub_score.shape}"
    batch_score = sub_mask * sub_score
    # b_match_mask[b_true_match_mask][:, b_pred_match_mask] * (
    #    b_true_ints[b_true_match_mask].unsqueeze(1) * b_pred_ints[b_pred_match_mask].unsqueeze(0)
    # )
    row_ind, col_ind = _solve_batch_assignment(batch_score)
    return (batch_score[batch_idx, row_ind, col_ind]).sum(-1)  # sum for cosine


def batch_jss_hun_helper(
    batch_true_ints,
    batch_pred_ints,
    batch_match_mask,
    batch_true_match_mask,
    batch_pred_match_mask,
    remove_prec_peak,
    batch_true_prec_mask,
    batch_pred_prec_mask,
    log_min,
):
    """Compute Jensen-Shannon similarity via Hungarian matching for a padded batch.

    Batched analogue of ``jss_hun_helper``.  All spectra are zero-padded to the
    same peak-count.  The function optionally zeroes precursor peaks, L1-normalises
    each spectrum, solves the linear assignment problem for all batch items
    simultaneously, and computes the Jensen-Shannon divergence through the union
    mixture distribution.

    The return value per batch item is ``log(2) - 0.5 * (KL(P||M) + KL(Q||M))``
    where M is the union mixture of P and Q.  Divide by ``log(2)`` to obtain the
    conventional JSS score in [0, 1].

    Args:
        batch_true_ints: True intensities of shape (B, N_true).
        batch_pred_ints: Predicted intensities of shape (B, N_pred).
        batch_match_mask: Boolean match tensor of shape (B, N_true, N_pred);
            True where true peak i matches predicted peak j within tolerance.
        batch_true_match_mask: Boolean mask of shape (B, N_true); True where a
            true peak has at least one matching predicted peak.
        batch_pred_match_mask: Boolean mask of shape (B, N_pred); True where a
            predicted peak has at least one matching true peak.
        remove_prec_peak: If True, zero precursor peaks before scoring.
        batch_true_prec_mask: Boolean mask of shape (B, N_true). Ignored when
            remove_prec_peak is False.
        batch_pred_prec_mask: Boolean mask of shape (B, N_pred). Ignored when
            remove_prec_peak is False.
        log_min: Floor value passed to ``safelog`` to avoid ``log(0)``.

    Returns:
        Tensor of shape (B,) with JSS * log(2) values in [0, log(2)] per
        batch item.
    """
    if remove_prec_peak:
        assert (
            batch_true_ints.shape == batch_true_prec_mask.shape
        ), f"{batch_true_ints.shape} {batch_true_prec_mask.shape}"
        assert (
            batch_pred_ints.shape == batch_pred_prec_mask.shape
        ), f"{batch_pred_ints.shape} {batch_pred_prec_mask.shape}"
        batch_true_ints = batch_true_ints * (1 - batch_true_prec_mask.float())
        batch_pred_ints = batch_pred_ints * (1 - batch_pred_prec_mask.float())
    # The union-mixture bookkeeping below mixes bool-derived float32 masks into the
    # intensity tensors, so score in float32 under mixed-precision training.
    batch_true_ints = _as_scoring_dtype(batch_true_ints)
    batch_pred_ints = _as_scoring_dtype(batch_pred_ints)
    # Convert to probability vectors per spectrum:
    #   P_i <- P_i / ||P_i||_1, Q_i <- Q_i / ||Q_i||_1
    batch_true_ints = F.normalize(batch_true_ints, p=1, dim=-1)
    batch_pred_ints = F.normalize(batch_pred_ints, p=1, dim=-1)

    # select subset to find idx
    true_batch_pos_idx = th.argsort(batch_true_match_mask.to(th.int64), descending=True)
    pred_batch_pos_idx = th.argsort(batch_pred_match_mask.to(th.int64), descending=True)
    max_true_match_pos = batch_true_match_mask.float().sum(-1).long().max()
    max_pred_match_pos = batch_pred_match_mask.float().sum(-1).long().max()
    true_batch_pos_idx = true_batch_pos_idx[:, :max_true_match_pos]
    pred_batch_pos_idx = pred_batch_pos_idx[:, :max_pred_match_pos]
    batch_idx = th.arange(true_batch_pos_idx.shape[0], device=true_batch_pos_idx.device)[:, None]
    sub_mask = batch_match_mask[
        batch_idx[..., None], true_batch_pos_idx[:, :, None], pred_batch_pos_idx[:, None, :]
    ]
    sub_true_ints = batch_true_ints[batch_idx, true_batch_pos_idx][..., None]
    sub_pred_ints = batch_pred_ints[batch_idx, pred_batch_pos_idx][..., None, :]
    pair_mass = sub_true_ints + sub_pred_ints
    pair_gain = 0.5 * (
        pair_mass * th.log(pair_mass + EPS)
        - sub_true_ints * th.log(sub_true_ints + EPS)
        - sub_pred_ints * th.log(sub_pred_ints + EPS)
    )
    assert sub_mask.shape == pair_gain.shape, f"{sub_mask.shape} {pair_gain.shape}"
    # Match the scalar sparse-JSD objective: maximize the reduction in JSD from
    # merging each feasible true/predicted pair onto one union-support position.
    assignment_score = pair_gain.float().masked_fill(~sub_mask, -1e9)
    row_ind, col_ind = _solve_batch_assignment(assignment_score.detach())
    # map ind to ori ind
    raw_row_ind = true_batch_pos_idx[batch_idx, row_ind]
    raw_col_ind = pred_batch_pos_idx[batch_idx, col_ind]

    match_mask = sub_mask[batch_idx, row_ind, col_ind]  # False means wrong match
    # The union mixture uses P+Q mass after assignment, not the assignment gain.
    match_score = pair_mass[batch_idx, row_ind, col_ind] * match_mask.float()
    unmatch_true_ints_mask = th.zeros_like(batch_true_match_mask)
    unmatch_pred_ints_mask = th.zeros_like(batch_pred_match_mask)
    unmatch_true_ints_mask[batch_idx, raw_row_ind] = (
        match_mask  # True means match, False means unmatch
    )
    unmatch_pred_ints_mask[batch_idx, raw_col_ind] = match_mask
    unmatch_true_ints_mask = ~unmatch_true_ints_mask
    unmatch_pred_ints_mask = ~unmatch_pred_ints_mask
    # Union total mass before mixture scaling:
    #   Z = sum(match_score) + sum(unmatched P) + sum(unmatched Q)
    batch_score_sum = match_score.sum(-1)
    unmatch_true_ints_sum = (batch_true_ints * unmatch_true_ints_mask.float()).sum(-1)
    unmatch_pred_ints_sum = (batch_pred_ints * unmatch_pred_ints_mask.float()).sum(-1)
    # Mixture normalization factor for M = 0.5 * (P + Q on union support).
    union_sum = (batch_score_sum + unmatch_true_ints_sum + unmatch_pred_ints_sum).clamp(1e-8) * 0.5

    batch_true_kl1_union_probs = th.zeros_like(batch_true_ints)
    # fill match pos with union match score
    batch_true_kl1_union_probs[batch_idx, raw_row_ind] = match_score * 0.5 / union_sum[..., None]
    # fill unmatch pos with true intensity/union sum # order matters
    batch_true_kl1_union_probs[unmatch_true_ints_mask] = (
        0.5 * batch_true_ints / union_sum[..., None]
    )[unmatch_true_ints_mask]
    # KL(P||M) = Σ_u P_u * (log P_u - log M_u)
    batch_true_kl1 = th.sum(
        batch_true_ints
        * (
            safelog(batch_true_ints, eps=log_min) - safelog(batch_true_kl1_union_probs, eps=log_min)
        ),
        dim=-1,
    )

    batch_pred_kl1_union_probs = th.zeros_like(batch_pred_ints)
    # fill match pos with union match score, assign order matters here
    batch_pred_kl1_union_probs[batch_idx, raw_col_ind] = match_score * 0.5 / union_sum[..., None]
    batch_pred_kl1_union_probs[unmatch_pred_ints_mask] = (
        0.5 * batch_pred_ints / union_sum[..., None]
    )[unmatch_pred_ints_mask]
    # KL(Q||M) = Σ_v Q_v * (log Q_v - log M_v)
    batch_pred_kl1 = th.sum(
        batch_pred_ints
        * (
            safelog(batch_pred_ints, eps=log_min) - safelog(batch_pred_kl1_union_probs, eps=log_min)
        ),
        dim=-1,
    )
    # JSS (scaled by ln 2): ln(2) - 0.5 * (KL(P||M) + KL(Q||M))
    return np.log(2.0) - 0.5 * (batch_true_kl1 + batch_pred_kl1)


def cos_hun_helper(
    b_true_ints,
    b_pred_ints,
    b_match_mask,
    b_true_match_mask,
    b_pred_match_mask,
    remove_prec_peak,
    b_true_prec_mask,
    b_pred_prec_mask,
):
    """Compute cosine-Hungarian similarity for a single spectrum pair.

    Optionally zeroes the precursor peak, L2-normalises both intensity vectors,
    builds a score matrix restricted to mutually-matchable peaks, solves the
    linear assignment problem to find the optimal one-to-one peak pairing, then
    returns the dot product of the matched normalised intensities.

    Args:
        b_true_ints: True intensities of shape (N_true,).
        b_pred_ints: Predicted intensities of shape (N_pred,).
        b_match_mask: Boolean match matrix of shape (N_true, N_pred); True where
            true peak i is within m/z tolerance of predicted peak j.
        b_true_match_mask: Boolean mask of shape (N_true,); True for true peaks
            that match at least one predicted peak.
        b_pred_match_mask: Boolean mask of shape (N_pred,); True for predicted
            peaks that match at least one true peak.
        remove_prec_peak: If True, zero the precursor peak in both spectra using
            the provided masks before scoring.
        b_true_prec_mask: Boolean mask of shape (N_true,); True where the true
            peak is the precursor peak. Ignored when remove_prec_peak is False.
        b_pred_prec_mask: Boolean mask of shape (N_pred,); True where the
            predicted peak is the precursor peak. Ignored when
            remove_prec_peak is False.

    Returns:
        Scalar tensor with the cosine-Hungarian similarity in [0, 1].
    """
    if remove_prec_peak:
        b_true_ints = b_true_ints * (1 - b_true_prec_mask.float())
        b_pred_ints = b_pred_ints * (1 - b_pred_prec_mask.float())
    # L2-normalize intensities so matched dot products form cosine terms.
    #   p <- p / ||p||_2, q <- q / ||q||_2
    b_true_ints = F.normalize(b_true_ints, p=2, dim=0)
    b_pred_ints = F.normalize(b_pred_ints, p=2, dim=0)
    # Feasible pair score matrix:
    #   S_{uv} = 1[match(u,v)] * p_u * q_v
    b_score = b_match_mask[b_true_match_mask][:, b_pred_match_mask] * (
        b_true_ints[b_true_match_mask].unsqueeze(1) * b_pred_ints[b_pred_match_mask].unsqueeze(0)
    )
    b_true_idxs, b_pred_idxs = _solve_batch_assignment(b_score.unsqueeze(0))
    b_true_idxs, b_pred_idxs = b_true_idxs[0], b_pred_idxs[0]
    # Cosine-Hungarian value is the sum of assigned pairwise cosine products.
    b_cos_hun = th.dot(
        b_true_ints[b_true_match_mask][b_true_idxs],
        b_pred_ints[b_pred_match_mask][b_pred_idxs],
    )
    return b_cos_hun


def ndcg_helper(
    b_true_ints,
    b_pred_ints,
    b_match_mask,
    b_true_match_mask,
    b_pred_match_mask,
    optimistic,
    union,
):
    """Compute normalised discounted cumulative gain (NDCG) for a single spectrum pair.

    Hungarian matching finds the optimal one-to-one pairing between matchable peaks.
    Two scoring modes are supported:

    - **Intersection** (``union=False``): only matched peaks contribute to the DCG
      sum.  Returns 0.0 when there are no matched peaks.
    - **Union** (``union=True``): all peaks (matched and unmatched) are included.
      Unmatched predicted peaks receive a true-intensity gain of zero; unmatched
      true peaks are ranked by the ``optimistic`` heuristic (descending if True,
      ascending otherwise).

    Args:
        b_true_ints: True intensities of shape (N_true,).
        b_pred_ints: Predicted intensities of shape (N_pred,).
        b_match_mask: Boolean match matrix of shape (N_true, N_pred); True where
            true peak i is within m/z tolerance of predicted peak j.
        b_true_match_mask: Boolean mask of shape (N_true,); True for true peaks
            that have at least one match in the predicted spectrum.
        b_pred_match_mask: Boolean mask of shape (N_pred,); True for predicted
            peaks that have at least one match in the true spectrum.
        optimistic: When union=True, controls the heuristic ranking of unmatched
            true peaks.  True → sorted descending (optimistic upper bound);
            False → sorted ascending (pessimistic lower bound).
        union: If True, use union NDCG; if False, use intersection NDCG.

    Returns:
        Scalar float or tensor with the NDCG value in [0, 1].  Returns the
        Python float 0.0 when union=False and no peaks are matched.
    """
    th_device = b_true_ints.device
    # Matching score used to derive one-to-one correspondence.
    #   S_{uv} = 1[match(u,v)] * rel_u * score_v
    b_score = b_match_mask[b_true_match_mask][:, b_pred_match_mask] * (
        b_true_ints[b_true_match_mask].unsqueeze(1) * b_pred_ints[b_pred_match_mask].unsqueeze(0)
    )
    b_true_match_idxs, b_pred_match_idxs = _solve_batch_assignment(b_score.unsqueeze(0))
    b_true_match_idxs, b_pred_match_idxs = b_true_match_idxs[0], b_pred_match_idxs[0]
    b_pred_match_ints = b_pred_ints[b_pred_match_mask][b_pred_match_idxs]
    b_true_match_ints = b_true_ints[b_true_match_mask][b_true_match_idxs]
    if union:
        b_pred_unmatch_idxs = th_setdiff1d(
            th.arange(b_pred_match_mask.sum(), device=th_device), b_pred_match_idxs
        )
        b_pred_unmatch_ints = th.cat(
            [
                b_pred_ints[~b_pred_match_mask],
                b_pred_ints[b_pred_match_mask][b_pred_unmatch_idxs],
            ],
            dim=0,
        )
        b_true_unmatch_idxs = th_setdiff1d(
            th.arange(b_true_match_mask.sum(), device=th_device), b_true_match_idxs
        )
        b_true_unmatch_ints = th.cat(
            [
                b_true_ints[~b_true_match_mask],
                b_true_ints[b_true_match_mask][b_true_unmatch_idxs],
            ],
            dim=0,
        )
        b_pred_all_ints = th.cat(
            [
                b_pred_match_ints,
                b_pred_unmatch_ints,
                # heuristic: rank the unmatched true peaks by their intensity
                -(th.argsort(b_true_unmatch_ints, descending=optimistic) + 1).type(
                    b_true_unmatch_ints.dtype
                ),
            ],
            dim=0,
        )
        b_true_all_ints = th.cat(
            [
                b_true_match_ints,
                # score of zero for the unmatched predicted peaks
                th.zeros_like(b_pred_unmatch_ints),
                b_true_unmatch_ints,
            ],
            dim=0,
        )
        b_pred_ranking = th.argsort(b_pred_all_ints, descending=True)
        b_true_ranking = th.argsort(b_true_all_ints, descending=True)
        b_denom = th.log2(
            2 + th.arange(b_true_all_ints.shape[0], dtype=b_true_all_ints.dtype, device=th_device)
        )
        # DCG  = Σ_k gain_pred(k) / log2(k+1)
        # IDCG = Σ_k gain_ideal(k) / log2(k+1)
        b_dcg = th.sum(b_true_all_ints[b_pred_ranking] / b_denom)
        b_idcg = th.sum(b_true_all_ints[b_true_ranking] / b_denom)
        b_ndcg = b_dcg / b_idcg
    else:  # intersection
        if b_true_match_idxs.shape[0] == 0:
            b_ndcg = 0.0
        else:
            b_pred_ranking = th.argsort(b_pred_match_ints, descending=True)
            b_true_ranking = th.argsort(b_true_match_ints, descending=True)
            b_denom = th.log2(
                2
                + th.arange(
                    b_true_match_ints.shape[0],
                    dtype=b_true_match_ints.dtype,
                    device=th_device,
                )
            )
            # Intersection mode uses matched peaks only.
            b_dcg = th.sum(b_true_match_ints[b_pred_ranking] / b_denom)
            b_idcg = th.sum(b_true_match_ints[b_true_ranking] / b_denom)
            b_ndcg = b_dcg / b_idcg
    return b_ndcg


def jss_hun_helper(
    b_true_ints,
    b_pred_ints,
    b_match_mask,
    b_true_match_mask,
    b_pred_match_mask,
    remove_prec_peak,
    b_true_prec_mask,
    b_pred_prec_mask,
    log_min,
):
    """Compute Jensen-Shannon similarity via Hungarian matching for a single spectrum pair.

    Optionally zeroes the precursor peak, L1-normalises both intensity vectors,
    solves the linear assignment problem to find the optimal one-to-one peak pairing
    among matchable peaks, then computes the Jensen-Shannon divergence between the
    true and predicted distributions through their union mixture.

    The return value is ``log(2) - 0.5 * (KL(P||M) + KL(Q||M))`` where M is
    the union mixture distribution.  This lies in ``[0, log(2)]``; divide by
    ``log(2)`` to obtain the conventional JSS score in [0, 1].

    Args:
        b_true_ints: True intensities of shape (N_true,).
        b_pred_ints: Predicted intensities of shape (N_pred,).
        b_match_mask: Boolean match matrix of shape (N_true, N_pred); True where
            true peak i is within m/z tolerance of predicted peak j.
        b_true_match_mask: Boolean mask of shape (N_true,); True for true peaks
            that match at least one predicted peak.
        b_pred_match_mask: Boolean mask of shape (N_pred,); True for predicted
            peaks that match at least one true peak.
        remove_prec_peak: If True, zero the precursor peak in both spectra using
            the provided masks before scoring.
        b_true_prec_mask: Boolean mask of shape (N_true,); True where the true
            peak is the precursor peak. Ignored when remove_prec_peak is False.
        b_pred_prec_mask: Boolean mask of shape (N_pred,); True where the
            predicted peak is the precursor peak. Ignored when
            remove_prec_peak is False.
        log_min: Floor value passed to ``safelog`` to avoid ``log(0)``.

    Returns:
        Scalar tensor with the JSS value scaled by log(2), lying in [0, log(2)].

    Note:
        This implementation constructs KL supports using explicit matched/unmatched
        masks for each side (instead of concatenation with implicit index ranges).
        That guarantees probability-mass conservation for P, Q and their union
        mixture M even when Hungarian assignment covers only a subset of matchable
        peaks. Without this, dropped/misaligned mass can inflate KL and push JSS
        below 0 due to indexing artifacts.
    """
    if remove_prec_peak:
        b_true_ints = b_true_ints * (1 - b_true_prec_mask.float())
        b_pred_ints = b_pred_ints * (1 - b_pred_prec_mask.float())
    # Score in float32 for the same reason as ``batch_jss_hun_helper``: the union
    # bookkeeping mixes bool-derived float32 masks into the intensity tensors.
    b_true_ints = _as_scoring_dtype(b_true_ints)
    b_pred_ints = _as_scoring_dtype(b_pred_ints)
    # Convert to probability vectors:
    #   P <- P / ||P||_1, Q <- Q / ||Q||_1
    b_true_ints = F.normalize(b_true_ints, p=1, dim=0)
    b_pred_ints = F.normalize(b_pred_ints, p=1, dim=0)
    # if th.all(b_pred_ints) == 0.0:
    #    # heuristic to prevent nan
    #    b_pred_ints = th.ones_like(b_pred_ints) / b_pred_ints.shape[0]
    b_matchable_true_idxs = th.where(b_true_match_mask)[0]
    b_matchable_pred_idxs = th.where(b_pred_match_mask)[0]
    # Restrict assignment to matchable peaks; Hungarian returns one-to-one pairs.
    if b_matchable_true_idxs.numel() > 0 and b_matchable_pred_idxs.numel() > 0:
        b_sub_match_mask = b_match_mask[b_matchable_true_idxs][:, b_matchable_pred_idxs]
        sub_true_ints = b_true_ints[b_matchable_true_idxs].unsqueeze(1)
        sub_pred_ints = b_pred_ints[b_matchable_pred_idxs].unsqueeze(0)
        b_pair_mass = sub_true_ints + sub_pred_ints
        b_pair_gain = 0.5 * (
            b_pair_mass * th.log(b_pair_mass + EPS)
            - sub_true_ints * th.log(sub_true_ints + EPS)
            - sub_pred_ints * th.log(sub_pred_ints + EPS)
        )
        b_assignment_score = b_pair_gain.float().masked_fill(~b_sub_match_mask, -1e9)
        b_true_idxs, b_pred_idxs = _solve_batch_assignment(
            b_assignment_score.detach().unsqueeze(0)
        )
        b_true_idxs, b_pred_idxs = b_true_idxs[0], b_pred_idxs[0]

        b_raw_true_idxs = b_matchable_true_idxs[b_true_idxs]
        b_raw_pred_idxs = b_matchable_pred_idxs[b_pred_idxs]
        b_pair_match_mask = b_sub_match_mask[b_true_idxs, b_pred_idxs]
        # Keep only valid matched pairs (guard against padded/invalid indices).
        b_match_score = b_pair_mass[b_true_idxs, b_pred_idxs] * b_pair_match_mask.float()
    else:
        b_raw_true_idxs = th.empty(0, dtype=th.long, device=b_true_ints.device)
        b_raw_pred_idxs = th.empty(0, dtype=th.long, device=b_pred_ints.device)
        b_pair_match_mask = th.empty(0, dtype=th.bool, device=b_true_ints.device)
        b_match_score = th.empty(0, dtype=b_true_ints.dtype, device=b_true_ints.device)

    b_unmatch_true_mask = th.ones_like(b_true_match_mask)
    b_unmatch_pred_mask = th.ones_like(b_pred_match_mask)
    if b_raw_true_idxs.numel() > 0:
        b_unmatch_true_mask[b_raw_true_idxs] = ~b_pair_match_mask
    if b_raw_pred_idxs.numel() > 0:
        b_unmatch_pred_mask[b_raw_pred_idxs] = ~b_pair_match_mask

    # Union support mass:
    #   Z = sum(match_score) + sum(unmatched P) + sum(unmatched Q)
    b_match_score_sum = b_match_score.sum()
    b_unmatch_true_ints_sum = (b_true_ints * b_unmatch_true_mask.float()).sum()
    b_unmatch_pred_ints_sum = (b_pred_ints * b_unmatch_pred_mask.float()).sum()
    # Mixture normalization factor for M = 0.5*(P + Q) on union support.
    b_union_sum = (b_match_score_sum + b_unmatch_true_ints_sum + b_unmatch_pred_ints_sum).clamp(
        1e-8
    ) * 0.5

    b_kl1_union_probs = th.zeros_like(b_true_ints)
    if b_raw_true_idxs.numel() > 0:
        b_kl1_union_probs[b_raw_true_idxs] = b_match_score * 0.5 / b_union_sum
    b_kl1_union_probs[b_unmatch_true_mask] = (0.5 * b_true_ints / b_union_sum)[b_unmatch_true_mask]
    # KL(P||M) = Σ_u P_u * (log P_u - log M_u)
    b_kl1 = th.sum(
        b_true_ints * (safelog(b_true_ints, eps=log_min) - safelog(b_kl1_union_probs, eps=log_min)),
        dim=0,
    )

    b_kl2_union_probs = th.zeros_like(b_pred_ints)
    if b_raw_pred_idxs.numel() > 0:
        b_kl2_union_probs[b_raw_pred_idxs] = b_match_score * 0.5 / b_union_sum
    b_kl2_union_probs[b_unmatch_pred_mask] = (0.5 * b_pred_ints / b_union_sum)[b_unmatch_pred_mask]
    # KL(Q||M) = Σ_v Q_v * (log Q_v - log M_v)
    b_kl2 = th.sum(
        b_pred_ints * (safelog(b_pred_ints, eps=log_min) - safelog(b_kl2_union_probs, eps=log_min)),
        dim=0,
    )

    # JSS (scaled by ln 2): ln(2) - 0.5 * (KL(P||M) + KL(Q||M))
    b_jss_hun = np.log(2.0) - 0.5 * (b_kl1 + b_kl2)
    # Numerical guardrail: theoretical range is [0, ln(2)], but tiny floating-point
    # drift may produce values slightly outside this interval.
    b_jss_hun = th.clamp(b_jss_hun, min=0.0, max=np.log(2.0))
    return b_jss_hun


def jss_helper(
    true_bin_idxs,
    true_bin_ints,
    true_bin_batch_idxs,
    pred_bin_idxs,
    pred_bin_ints,
    pred_bin_batch_idxs,
    log_min,
):
    batch_size = int(th.max(true_bin_batch_idxs).item()) + 1
    # l1 normalize
    true_bin_ints = scatter_l1normalize(true_bin_ints, true_bin_batch_idxs)
    pred_bin_ints = scatter_l1normalize(pred_bin_ints, pred_bin_batch_idxs)
    # union distribution
    union_bin_idxs, union_bin_idxs_rev = th.unique(
        th.cat([true_bin_idxs, pred_bin_idxs], dim=0), return_inverse=True
    )
    union_bin_ints = scatter_reduce(
        src=0.5 * th.cat([true_bin_ints, pred_bin_ints], dim=0),
        index=union_bin_idxs_rev,
        reduce="sum",
        dim_size=union_bin_idxs.shape[0],
    )

    # Normalize the mixture distribution per batch to sum exactly to 1.
    # Without this, floating-point errors in scatter_reduce can cause
    # union_bin_ints to sum slightly below 1, which inflates the KL terms
    # beyond their theoretical maximum (log 2) and makes JSS go negative.
    # union_bin_batch_idxs = scatter_reduce(
    #    src=th.cat([true_bin_batch_idxs, pred_bin_batch_idxs], dim=0),
    #    index=union_bin_idxs_rev,
    #    reduce="amax",
    #    dim_size=union_bin_idxs.shape[0],
    # )
    # union_bin_ints = scatter_l1normalize(union_bin_ints, union_bin_batch_idxs)
    # kl1
    kl1_union_bin_ints = union_bin_ints[union_bin_idxs_rev[: true_bin_idxs.shape[0]]]
    kl1 = scatter_reduce(
        true_bin_ints
        * (safelog(true_bin_ints, eps=log_min) - safelog(kl1_union_bin_ints, eps=log_min)),
        true_bin_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )

    # kl2
    kl2_union_bin_ints = union_bin_ints[union_bin_idxs_rev[true_bin_idxs.shape[0] :]]
    kl2 = scatter_reduce(
        pred_bin_ints
        * (safelog(pred_bin_ints, eps=log_min) - safelog(kl2_union_bin_ints, eps=log_min)),
        pred_bin_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    # jss
    jsd_e = 0.5 * (kl1 + kl2)
    # (ln(2) - jsd_e) / ln(2); clamp to [0,1] to guard against FP drift in union mixture
    jss = th.clamp(1.0 - jsd_e / np.log(2.0), min=0.0, max=1.0)
    return jss
