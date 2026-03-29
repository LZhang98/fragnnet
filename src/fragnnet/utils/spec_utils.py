import logging
from collections.abc import Callable
from typing import cast

import numpy as np
import torch as th
import torch.nn.functional as F

from fragnnet.utils.formula_utils import (
    C13_SPACING,
    MASS,
    NEUTRON_MASS,
    PREC_TYPE_TO_FORMULA_DIFF,
    isotope_offsets_from_formula,
    parse_formula,
)
from fragnnet.utils.misc_utils import (
    EPS,
    TOLERANCE_MIN_MZ,
    safelog,
    scatter_l1normalize,
    scatter_l2normalize,
    scatter_logsoftmax,
    scatter_logsumexp,
    scatter_reduce,
    th_setdiff1d,
)

try:
    from torch_linear_assignment import (  # pyright: ignore[reportMissingImports]
        assignment_to_indices,
        batch_linear_assignment,
    )

    _torch_linear_assignment = True
except ImportError:
    _torch_linear_assignment = False

    logging.getLogger(__name__).warning(
        "torch_linear_assignment not found; falling back to scipy.optimize.linear_sum_assignment (slower, CPU-only)"
    )


MZ_MAX = 1500.0
MZ_BIN_RES = 0.01
INTS_THRESH = 0.0


def _solve_batch_assignment(batch_score: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """Solve a batched linear assignment problem, maximizing total score.

    Args:
        batch_score: Score matrix of shape (batch, rows, cols).

    Returns:
        Tuple of (row_ind, col_ind) tensors of shape (batch, K) where
        K = min(rows, cols), giving the optimal assignment indices.
    """
    if _torch_linear_assignment:
        assignment = batch_linear_assignment(-batch_score)
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


def get_ints_transform_func(ints_transform: str) -> Callable[[th.Tensor], th.Tensor]:
    """method to get int transferom func

    Args:
        ints_transform (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if ints_transform == "log10":

        def _func(ints: th.Tensor) -> th.Tensor:
            return th.log10(ints + 1.0)
    elif ints_transform == "log10t3":

        def _func(ints: th.Tensor) -> th.Tensor:
            return th.log10(ints / 3.0 + 1.0)
    elif ints_transform == "loge":

        def _func(ints: th.Tensor) -> th.Tensor:
            return th.log(ints + 1.0)
    elif ints_transform == "sqrt":

        def _func(ints: th.Tensor) -> th.Tensor:
            return th.sqrt(ints)
    elif ints_transform == "none":

        def _func(ints: th.Tensor) -> th.Tensor:
            return ints
    else:
        raise ValueError(f"Invalid ints_transform: {ints_transform}")
    return _func


def get_ints_untransform_func(ints_transform: str):
    """

    Args:
        ints_transform (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if ints_transform == "log10":
        max_ints = float(np.log10(1000.0 + 1.0))

        def _untransform_fn_log10(x: th.Tensor) -> th.Tensor:
            return 10**x - 1.0

        _untransform_fn = _untransform_fn_log10
    elif ints_transform == "log10t3":
        max_ints = float(np.log10(1000.0 / 3.0 + 1.0))

        def _untransform_fn_log10t3(x: th.Tensor) -> th.Tensor:
            return 3.0 * (10**x - 1.0)

        _untransform_fn = _untransform_fn_log10t3
    elif ints_transform == "loge":
        max_ints = float(np.log(1000.0 + 1.0))

        def _untransform_fn_loge(x: th.Tensor) -> th.Tensor:
            return th.exp(x) - 1.0

        _untransform_fn = _untransform_fn_loge
    elif ints_transform == "sqrt":
        max_ints = float(np.sqrt(1000.0))

        def _untransform_fn_sqrt(x: th.Tensor) -> th.Tensor:
            return x**2

        _untransform_fn = _untransform_fn_sqrt
    elif ints_transform == "none":
        max_ints = 1000.0

        def _untransform_fn_none(x: th.Tensor) -> th.Tensor:
            return x

        _untransform_fn = _untransform_fn_none
    else:
        raise ValueError("invalid transform")

    def _func(ints, batch_idxs):
        old_max_ints = scatter_reduce(ints, batch_idxs, "amax", default=0.0, include_self=False)
        ints = ints / (old_max_ints[batch_idxs] + EPS) * max_ints
        ints = _untransform_fn(ints)
        ints = th.clamp(ints, min=0.0)
        assert not th.isnan(ints).any()
        return ints

    return _func


def transform_ce(ce: th.Tensor | float, ce_mean: float, ce_std: float) -> th.Tensor:
    """Transform collision energy (CE) values.
        if ce < -1, it is considered missing/unknown and returned as is.
    Args:
        ce (torch.Tensor): The CE values to transform.
        ce_mean (float): The mean CE value.
        ce_std (float): The standard deviation of CE values.

    Returns:
        torch.Tensor: The transformed CE values.
    """
    if isinstance(ce, th.Tensor):
        # Handle tensor input
        mask = ce >= 0
        transformed_ce = th.where(mask, (ce - ce_mean) / ce_std, ce)
        return transformed_ce
    else:
        # Handle scalar input
        if ce >= 0:
            return th.tensor((ce - ce_mean) / ce_std)
        else:
            return th.tensor(ce)


def transform_nce_to_ace(nce: float, mw: float, charge_factor: int = 1) -> float:
    """get ace from given nce and mw
       Absolute energy (eV) = (settling NCE) x (Isolation center) / (500 m/z) x (charge factor)
    Args:
        nce (float): normalized collision energy
        mw (float): Isolation center mw, most time it is the precursor ion mass

    Returns:
        float: absolute energy
    """
    return nce * mw / 500 * charge_factor


def filter_func(
    mzs: th.Tensor | np.ndarray,
    ints: th.Tensor | np.ndarray,
    ints_thresh: float,
    mz_max: float,
) -> tuple[th.Tensor | np.ndarray, th.Tensor | np.ndarray]:
    """filter spectrum by intesnity value and max mz

    Args:
        mzs (Union[th.Tensor,np.ndarray]): m/z s
        ints (Union[th.Tensor,np.ndarray]): intesnities
        ints_thresh (float): intesnity thresh hold
        mz_max (float): max mz, if mz_max <= 0, mz_max filter will be ignored

    Returns:
        Tuple[Union[th.Tensor,np.ndarray], Union[th.Tensor,np.ndarray]]: mzs, ints
    """
    is_tensor_pair = isinstance(mzs, th.Tensor) and isinstance(ints, th.Tensor)
    is_ndarray_pair = isinstance(mzs, np.ndarray) and isinstance(ints, np.ndarray)
    if not (is_tensor_pair or is_ndarray_pair):
        raise TypeError("mzs and ints must both be torch tensors or both numpy arrays")

    thresh_mask = ints > ints_thresh
    if mz_max > 0:
        both_mask = thresh_mask & (mzs < mz_max)
    else:
        both_mask = thresh_mask

    if is_tensor_pair:
        mzs_t = cast(th.Tensor, mzs)
        ints_t = cast(th.Tensor, ints)
        mask_t = cast(th.Tensor, both_mask)
        return mzs_t[mask_t], ints_t[mask_t]

    mzs_n = cast(np.ndarray, mzs)
    ints_n = cast(np.ndarray, ints)
    mask_n = cast(np.ndarray, both_mask)
    return mzs_n[mask_n], ints_n[mask_n]


def bin_func(
    mzs: th.Tensor,
    ints: th.Tensor,
    mz_max: float,
    mz_bin_res: float,
    return_index: bool,
    sum_ints: bool,
):
    """
    return binned spectra
    Note: if return_index is True, returns the (possibly non-unique) bin index for each mz
    Note: intensities may not be normalized due to peak merging
    Args:
        mzs (th.Tensor): 1d flat tensor of m/zs; multiple m/z lists are concatenated.
        ints (th.Tensor): 1d flat tensor of intensities; multiple intensities lists are concatenated.
        mz_max (float): max mz value allowed
        mz_bin_res (float): bin size
        return_index (bool): if return_index is True, returns the (possibly non-unique) bin index for each mz
        sum_ints (bool): flag for sum intensities within the bin, else take max
    Returns:
        _type_: _description_
    """

    assert th.max(mzs) < mz_max, (th.max(mzs), mz_max)
    # bin
    bins = th.arange(
        mz_bin_res,
        mz_max + mz_bin_res,
        step=mz_bin_res,
        device=mzs.device,
        dtype=mzs.dtype,
    )
    bin_idx = th.searchsorted(bins, mzs, right=True)
    if return_index:
        return bin_idx
    else:
        bin_spec = scatter_reduce(
            src=ints,
            index=bin_idx,
            reduce="sum" if sum_ints else "amax",
            dim_size=bins.shape[0],
        )
        if th.all(bin_spec == 0.0):
            print("> warning: bin_spec is all zeros!")
            bin_spec[-1] = 1.0
        return bin_spec


def batch_func(*lists, offset_flags: list[bool] | None = None) -> tuple[th.Tensor]:
    """ """

    if offset_flags is None:
        offset_flags = [False] * len(lists)
    batch_size = len(lists[0])
    batch_idxs = th.arange(batch_size)
    repeat_sizes = th.tensor([th.numel(item) for item in lists[0]])
    batch_idxs = th.repeat_interleave(batch_idxs, repeat_sizes)
    if any(offset_flags):
        offsets = th.cat([th.zeros([1], dtype=th.long), th.cumsum(repeat_sizes, dim=0)[:-1]], dim=0)
    b_lists = []
    for l_idx, seq in enumerate(lists):
        b_list = th.cat(seq, dim=0)
        if offset_flags[l_idx]:
            repeat_sizes = th.tensor([th.numel(item) for item in seq])
            repeat_offsets = th.repeat_interleave(offsets, repeat_sizes)
            b_list = b_list + repeat_offsets
        b_lists.append(b_list)
    return tuple(b_lists) + (batch_idxs,)


def batched_filter_func(
    mzs: th.Tensor,
    ints: th.Tensor,
    batch_idxs: th.Tensor,
    ints_thresh: float,
    mz_max: float,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """bacthed filter func for filter spectra

    Args:
        mzs (th.Tensor): 1d flat tensor of m/zs; multiple m/z lists are concatenated.
        ints (th.Tensor): 1d flat tensor of intensities; multiple intensities lists are concatenated.
        batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.
        ints_thresh (float): intesnity thresh hold
        mz_max (float): max m/z, if mz_max <= 0, mz_max filter will be ignored

    Returns:
        Tuple[th.Tensor,th.Tensor,th.Tensor]: mz tensoer, intensity tensor, batch_idxs tensor
    """

    thresh_mask = ints > ints_thresh
    if mz_max > 0:
        max_mask = mzs < mz_max
        both_mask = thresh_mask & max_mask
    else:
        both_mask = thresh_mask
    mzs = mzs[both_mask]
    ints = ints[both_mask]
    batch_idxs = batch_idxs[both_mask]
    return mzs, ints, batch_idxs


def batched_bin_func(
    mzs: th.Tensor,
    ints: th.Tensor,
    batch_idxs: th.Tensor,
    mz_max: float,
    mz_bin_res: float,
    agg: str,
    sparse: bool = False,
    remove_prec_peaks: bool = False,
    prec_mzs: th.Tensor | None = None,
    return_mzs: bool = False,
) -> th.Tensor | tuple[th.Tensor, th.Tensor, th.Tensor]:
    """method to get binned spectra for batch

    Args:
        mzs (th.Tensor): 1d flat tensor of m/zs; multiple m/z lists are concatenated.
        ints (th.Tensor): 1d flat tensor of intensities; multiple intensities lists are concatenated.
        batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.
        mz_max (float): max mz value allowed
        mz_bin_res (float): bin size
        sum_ints (bool): flag for sum intensities within the bin, else take max
        sparse (bool, optional): flag to use sparse  method. Defaults to False.

    Returns:
        _type_: binned spectra
    """

    if mzs.shape[0] == 0:
        raise ValueError("batched_bin_func received empty mzs tensor")
    assert th.max(mzs) < mz_max, (th.max(mzs), mz_max)
    batch_size = int(th.max(batch_idxs).item()) + 1
    bins = th.arange(
        mz_bin_res,
        mz_max + mz_bin_res,
        step=mz_bin_res,
        device=mzs.device,
        dtype=mzs.dtype,
    )
    num_bins = int(bins.shape[0])
    bin_idxs = th.searchsorted(bins, mzs, right=True)
    bin_offsets = (th.arange(batch_size, device=mzs.device) * num_bins)[batch_idxs]
    bin_idxs = bin_idxs + bin_offsets
    if remove_prec_peaks:
        assert prec_mzs is not None
        assert th.max(prec_mzs) < mz_max, (th.max(prec_mzs), mz_max)
        prec_mz_bin_idxs = th.searchsorted(bins, prec_mzs, right=True)
        prec_mz_bin_offsets = th.arange(batch_size, device=mzs.device) * num_bins
        prec_mz_bin_idxs = prec_mz_bin_idxs + prec_mz_bin_offsets
        prec_ints_mask = th.isin(bin_idxs, prec_mz_bin_idxs)
        ints = ints * (1 - prec_ints_mask.float())
    if sparse:
        un_bin_idxs, un_bin_idxs_rev = th.unique(bin_idxs, return_inverse=True)
        new_bin_idxs = th.arange(un_bin_idxs.shape[0], device=un_bin_idxs.device)
        if agg in ["sum", "amax"]:
            un_bin_ints = scatter_reduce(
                src=ints,
                index=new_bin_idxs[un_bin_idxs_rev],
                reduce=agg,
                dim_size=new_bin_idxs.shape[0],
            )
        else:
            assert agg == "lse", agg
            un_bin_ints = scatter_logsumexp(
                ints, new_bin_idxs[un_bin_idxs_rev], dim_size=new_bin_idxs.shape[0]
            )
        un_bin_batch_idxs = un_bin_idxs // num_bins
        if return_mzs:
            un_bin_mzs = bins[un_bin_idxs % num_bins] - mz_bin_res / 2.0
            return un_bin_mzs, un_bin_ints, un_bin_batch_idxs
        else:
            return un_bin_idxs, un_bin_ints, un_bin_batch_idxs
    else:
        if agg in ["sum", "amax"]:
            bin_spec = scatter_reduce(
                src=ints, index=bin_idxs, reduce=agg, dim_size=num_bins * batch_size
            )
        else:
            assert agg == "lse", agg
            bin_spec = scatter_logsumexp(ints, bin_idxs, dim_size=num_bins * batch_size)
        bin_spec = bin_spec.reshape(batch_size, num_bins)
        if agg in ["sum", "amax"] and th.any(th.all(bin_spec == 0.0, dim=1)):
            print("> warning: bin_spec is all zeros!")
            mask = th.zeros_like(bin_spec, dtype=th.bool)
            mask[:, 0] = 1.0
            mask = mask * th.all(bin_spec == 0.0, dim=1, keepdim=True)
            bin_spec = bin_spec + mask.float()
        return bin_spec


def merge_sparse_specs(
    *peakses, renormalize: bool = False, sum_ints: bool = True
) -> list[tuple[float, float]]:
    """
    this will result in peaks that are really close in mass (<5ppm)
    they are probably the same peak, but our model can handle this type of ambiguity
    for now, let's keep them unmerged

    Args:
        renormalize (bool, optional): flag to get renormalize spectra. Defaults to False.
        sum_ints (bool, optional): flag for sum intensities within the bin, else take max

    Returns:
        _type_: _description_
    """
    merged_peaks = {}
    # total_intensity = 0.
    for peaks in peakses:
        for mz, intensity in peaks:
            if mz in merged_peaks:
                if sum_ints:
                    merged_peaks[mz] += intensity
                else:
                    merged_peaks[mz] = max(merged_peaks[mz], intensity)
            else:
                merged_peaks[mz] = intensity
    merged_peaks = sorted(merged_peaks.items(), key=lambda x: x[0])
    if renormalize:
        total_intensity = sum([intensity for mz, intensity in merged_peaks])
        merged_peaks = [(mz, intensity / total_intensity) for mz, intensity in merged_peaks]
    return merged_peaks


def calculate_spectrum_entropy(log_ints: th.Tensor, batch_idxs: th.Tensor) -> th.Tensor:
    """method to compute entropy.
        NOTE: this is NOT same spectra entropy in this https://www.nature.com/articles/s41592-023-02012-9 a
        nd https://www.nature.com/articles/s41592-021-01331-z
    Args:
        log_ints (th.Tensor): intensity in log scale
        batch_idxs (th.Tensor): batch idx

    Returns:
        th.Tensor: spectrum entropy
    """

    k = int(th.max(batch_idxs).item()) + 1
    log_norm_ints = scatter_logsoftmax(log_ints, batch_idxs)
    entropy = -scatter_reduce(
        src=th.exp(log_norm_ints) * log_norm_ints,
        index=batch_idxs,
        reduce="sum",
        dim_size=k,
    )
    return entropy


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
    NOT BATCHED
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


def batched_l1_normalize(ints, batch_idxs):
    ints = scatter_l1normalize(ints, batch_idxs)
    return ints


def batched_mf1000_normalize(ints, batch_idxs):
    """mf 1000 normalize implemention in pytorch, batched

    Args:
        ints (_type_): _description_
        batch_idxs (_type_): _description_

    Returns:
        _type_: _description_
    """

    dim_size = int(th.max(batch_idxs).item()) + 1
    max_ints = scatter_reduce(ints, batch_idxs, reduce="max", dim_size=dim_size)
    ints = (ints / max_ints[batch_idxs]) * 1000.0
    return ints


def round_aggregate_peaks(
    mzs: th.Tensor, ints: th.Tensor, batch_idxs: th.Tensor, decimals: int = 4, agg="sum"
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """methds to round and aggreate peaks to give decimals points

    Args:
        mzs (th.Tensor): 1d flat tensor of m/zs, each row is an m/z array
        ints (th.Tensor): 1d flat tensor of intensities, each row is an intensities array, should be the same size as mz tensor
        batch_idxs (th.Tensor): 1d flat tensor of batch indices, each cell, batch_idxs[i] indicates batch index for mzs[i] and ints[i], should be the same size as mz tensor
        decimals (int, optional): Decimals. Defaults to 4.
        sum_ints (bool, optional): Flag to sum intensities if True. Defaults to True.
    Returns:
        Tuple[th.Tensor,th.Tensor,th.Tensor]: round_mzs, round_ints, round_batch_idxs
    """

    batch_size = th.max(batch_idxs) + 1
    round_mzs, round_ints, round_batch_idxs = [], [], []
    for b in range(batch_size):
        b_mask = batch_idxs == b
        b_round_mzs = th.round(mzs[b_mask], decimals=decimals)
        b_ints = ints[b_mask]
        b_round_mzs_un, b_round_mzs_inv = th.unique(b_round_mzs, return_inverse=True)
        if agg in ["sum", "amax"]:
            b_round_ints = scatter_reduce(
                src=b_ints,
                index=b_round_mzs_inv,
                reduce=agg,
                dim_size=b_round_mzs_un.shape[0],
            )
        else:
            assert agg == "lse", agg
            b_round_ints = scatter_logsumexp(
                logits=b_ints,
                subset_idxs=b_round_mzs_inv,
                dim_size=b_round_mzs_un.shape[0],
            )
        round_mzs.append(b_round_mzs_un)
        round_ints.append(b_round_ints)
        round_batch_idxs.append(th.full_like(b_round_mzs_un, b, dtype=batch_idxs.dtype))
    round_mzs = th.cat(round_mzs, dim=0)
    round_ints = th.cat(round_ints, dim=0)
    round_batch_idxs = th.cat(round_batch_idxs, dim=0)
    return round_mzs, round_ints, round_batch_idxs


def scipy_linear_sum_assignment(matrix, maximize=False):
    """
    Solve the linear sum assignment problem using Scipy's implementation (Hungarian algorithm).
    This is a wrapper to handle PyTorch tensors,
    It is caller's response to to detaching from the graph.

    Args:
        matrix (th.Tensor): The cost matrix (or score matrix if maximize=True).
        maximize (bool, optional): Calculates a maximum weight matching if true. Defaults to False.

    Returns:
        Tuple[th.Tensor, th.Tensor]: Row indices and column indices of the optimal assignment.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("scipy is required for scipy_linear_sum_assignment") from exc

    device = matrix.device
    matrix = matrix.cpu().numpy()
    x_idx, y_idx = linear_sum_assignment(matrix, maximize=maximize)
    return th.tensor(x_idx, device=device).long(), th.tensor(y_idx, device=device).long()


### helpers


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
        assert batch_true_ints.shape == batch_true_prec_mask.shape, (
            f"{batch_true_ints.shape} {batch_true_prec_mask.shape}"
        )
        assert batch_pred_ints.shape == batch_pred_prec_mask.shape, (
            f"{batch_pred_ints.shape} {batch_pred_prec_mask.shape}"
        )
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
    sub_score = (
        batch_true_ints[batch_idx, true_batch_pos_idx][..., None]
        * (batch_pred_ints[batch_idx, pred_batch_pos_idx][..., None, :])
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
        assert batch_true_ints.shape == batch_true_prec_mask.shape, (
            f"{batch_true_ints.shape} {batch_true_prec_mask.shape}"
        )
        assert batch_pred_ints.shape == batch_pred_prec_mask.shape, (
            f"{batch_pred_ints.shape} {batch_pred_prec_mask.shape}"
        )
        batch_true_ints = batch_true_ints * (1 - batch_true_prec_mask.float())
        batch_pred_ints = batch_pred_ints * (1 - batch_pred_prec_mask.float())
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
    sub_score = (
        batch_true_ints[batch_idx, true_batch_pos_idx][..., None]
        + (batch_pred_ints[batch_idx, pred_batch_pos_idx][..., None, :])
    )
    assert sub_mask.shape == sub_score.shape, f"{sub_mask.shape} {sub_score.shape}"
    # Hungarian score uses additive mass on feasible matches:
    #   S_{uv} = 1[match(u,v)] * (P_u + Q_v)
    batch_score = sub_mask.float() * sub_score
    row_ind, col_ind = _solve_batch_assignment(batch_score)
    # map ind to ori ind
    raw_row_ind = true_batch_pos_idx[batch_idx, row_ind]
    raw_col_ind = pred_batch_pos_idx[batch_idx, col_ind]

    # Matched pair mass contribution from Hungarian assignment.
    match_score = batch_score[batch_idx, row_ind, col_ind]
    match_mask = sub_mask[batch_idx, row_ind, col_ind]  # False means wrong match
    unmatch_true_ints_mask = th.zeros_like(batch_true_match_mask)
    unmatch_pred_ints_mask = th.zeros_like(batch_pred_match_mask)
    unmatch_true_ints_mask[batch_idx, raw_row_ind] = (
        match_mask  # True means match, False means unmatch
    )
    unmatch_pred_ints_mask[batch_idx, raw_col_ind] = match_mask
    unmatch_true_ints_mask = ~unmatch_true_ints_mask
    unmatch_pred_ints_mask = ~unmatch_pred_ints_mask
    match_score = match_score * match_mask.float()

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
        # Pair score for JSS matching:
        #   S_{uv} = 1[match(u,v)] * (P_u + Q_v)
        b_score = b_sub_match_mask * (
            b_true_ints[b_matchable_true_idxs].unsqueeze(1)
            + b_pred_ints[b_matchable_pred_idxs].unsqueeze(0)
        )
        b_true_idxs, b_pred_idxs = _solve_batch_assignment(b_score.unsqueeze(0))
        b_true_idxs, b_pred_idxs = b_true_idxs[0], b_pred_idxs[0]

        b_raw_true_idxs = b_matchable_true_idxs[b_true_idxs]
        b_raw_pred_idxs = b_matchable_pred_idxs[b_pred_idxs]
        b_pair_match_mask = b_sub_match_mask[b_true_idxs, b_pred_idxs]
        # Keep only valid matched pairs (guard against padded/invalid indices).
        b_match_score = b_score[b_true_idxs, b_pred_idxs] * b_pair_match_mask.float()
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
    # (ln(2) - jsd_e) / ln(2)
    jss = 1.0 - jsd_e / np.log(2.0)
    return jss


def detect_isotope_peaks(
    mzs: np.ndarray,
    ints: np.ndarray,
    mz_tol: float = 0.02,
    max_isotope: int = 3,
    check_intensity: bool = True,
) -> np.ndarray:
    """Detect isotope peaks in a reference spectrum.

    A peak is considered an isotope peak (M+k, k >= 1) if there exists a
    lower-m/z peak separated by approximately k * NEUTRON_MASS within
    ``mz_tol``.  Optionally, that lower-m/z peak must also be more intense
    than the candidate isotope peak.

    The function works with raw (non-normalised) m/z and intensity arrays as
    produced by standard reference libraries such as NIST.

    Args:
        mzs: Peak m/z values of shape (N,).  Need not be sorted.
        ints: Peak intensities of shape (N,).  Same order as ``mzs``.
        mz_tol: Absolute m/z tolerance in Da for matching isotope spacing.
            The default of 0.02 Da covers both the exact 13C–12C spacing
            (~1.0034 Da) and the NEUTRON_MASS approximation (1.0087 Da) used
            in the rest of the codebase.  Defaults to 0.02.
        max_isotope: Maximum isotope order to consider (1 = M+1 only,
            2 = up to M+2, etc.). Defaults to 3.
        check_intensity: If True, require that the candidate monoisotopic
            peak (the one at mz - k * NEUTRON_MASS) is at least as intense
            as the current peak.  This suppresses false positives from
            noise but may miss isotope peaks of low-intensity fragments.
            Defaults to True.

    Returns:
        Boolean array of shape (N,) where True means the peak is an
        isotope peak.

    Example:
        >>> mzs = np.array([100.0, 101.008, 200.0, 201.012])
        >>> ints = np.array([1000.0, 100.0, 500.0, 50.0])
        >>> mask = detect_isotope_peaks(mzs, ints)
        >>> mask  # peaks at 101.008 and 201.012 flagged as isotopes
        array([False,  True, False,  True])
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.ndim != 1:
        raise ValueError(f"mzs and ints must be 1-D arrays of the same length, got {mzs.shape}")

    n = len(mzs)
    is_isotope = np.zeros(n, dtype=bool)
    if n == 0:
        return is_isotope

    sort_idx = np.argsort(mzs)
    sorted_mzs = mzs[sort_idx]
    sorted_ints = ints[sort_idx]

    for i in range(n):
        for k in range(1, max_isotope + 1):
            target_mz = sorted_mzs[i] - k * NEUTRON_MASS
            lo = np.searchsorted(sorted_mzs, target_mz - mz_tol, side="left")
            hi = np.searchsorted(sorted_mzs, target_mz + mz_tol, side="right")
            if lo >= hi:
                continue
            # At least one candidate monoisotopic peak exists in the window.
            if check_intensity:
                if np.any(sorted_ints[lo:hi] >= sorted_ints[i]):
                    is_isotope[sort_idx[i]] = True
                    break
            else:
                is_isotope[sort_idx[i]] = True
                break

    return is_isotope


def detect_isotope_peaks_formula_aware(
    mzs: np.ndarray,
    ints: np.ndarray,
    element_counts: dict[str, int],
    mz_tol: float = 0.02,
    max_isotope: int = 3,
    check_intensity: bool = True,
    intensity_ratio_lo: float = 0.1,
    intensity_ratio_hi: float = 10.0,
) -> np.ndarray:
    """Detect isotope peaks using exact pyopenms isotopologue mass offsets.

    Unlike :func:`detect_isotope_peaks`, which uses a fixed ``k * NEUTRON_MASS``
    step, this function queries pyopenms ``CoarseIsotopePatternGenerator`` for
    the actual M+k mass offsets predicted for the given molecular formula.  This
    improves accuracy for molecules containing Cl, Br, S, or K where even-mass
    isotopes shift the M+2/M+3 positions away from the simple neutron-mass
    approximation.

    An optional intensity ratio gate rejects peaks whose observed/expected ratio
    falls outside ``[intensity_ratio_lo * exp, intensity_ratio_hi * exp]``.

    Requires pyopenms to be installed.

    Args:
        mzs: Peak m/z values of shape (N,).  Need not be sorted.
        ints: Peak intensities of shape (N,).  Same order as ``mzs``.
        element_counts: Molecular formula as element count dict from
            :func:`~fragnnet.utils.formula_utils.parse_formula`.
        mz_tol: Absolute m/z tolerance in Da.  Defaults to 0.02.
        max_isotope: Maximum isotope order to query from pyopenms.
            Defaults to 3.
        check_intensity: If True, require that the candidate monoisotopic peak
            is at least as intense as the candidate isotope peak.  Defaults to
            True.
        intensity_ratio_lo: Lower multiplier on the pyopenms-predicted
            relative intensity for the ratio gate.  Set to 0.0 to disable.
            Defaults to 0.1.
        intensity_ratio_hi: Upper multiplier on the pyopenms-predicted
            relative intensity for the ratio gate.  Set to ``np.inf`` to
            disable.  Defaults to 10.0.

    Returns:
        Boolean array of shape (N,) where True means the peak is an isotope
        peak.

    Example:
        >>> import numpy as np
        >>> from fragnnet.utils.formula_utils import parse_formula
        >>> ec = parse_formula("C10H8Br2")
        >>> mzs = np.array([285.899, 286.902, 287.902, 288.902, 289.902])
        >>> ints = np.array([100.0, 5.0, 97.0, 2.0, 24.0])
        >>> mask = detect_isotope_peaks_formula_aware(mzs, ints, ec)
        >>> mask[2]  # M+2 (large Br peak) is correctly flagged
        True
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.ndim != 1:
        raise ValueError(f"mzs and ints must be 1-D arrays of the same length, got {mzs.shape}")

    n = len(mzs)
    is_isotope = np.zeros(n, dtype=bool)
    if n == 0:
        return is_isotope

    offsets = isotope_offsets_from_formula(element_counts, max_isotope=max_isotope)
    if not offsets:
        return is_isotope

    sort_idx = np.argsort(mzs)
    sorted_mzs = mzs[sort_idx]
    sorted_ints = ints[sort_idx]

    for i in range(n):
        for offset, exp_ratio in offsets:
            target_mz = sorted_mzs[i] - offset
            if target_mz < 0:
                continue
            lo = np.searchsorted(sorted_mzs, target_mz - mz_tol, side="left")
            hi = np.searchsorted(sorted_mzs, target_mz + mz_tol, side="right")
            if lo >= hi:
                continue
            mono_int = sorted_ints[lo:hi].max()
            if check_intensity and mono_int < sorted_ints[i]:
                continue
            if mono_int > 0:
                obs_ratio = sorted_ints[i] / mono_int
                if not (
                    intensity_ratio_lo * exp_ratio <= obs_ratio <= intensity_ratio_hi * exp_ratio
                ):
                    continue
            is_isotope[sort_idx[i]] = True
            break

    return is_isotope


# ---------------------------------------------------------------------------
# Training-data isotope cleaning
# ---------------------------------------------------------------------------

# Natural abundance M+1/M+0 contribution per atom (Poisson/binomial approx)
_M1_CONTRIB: dict[str, float] = {
    "C": 0.01103,  # 13C
    "H": 0.000149,  # 2H
    "N": 0.00368,  # 15N
    "O": 0.000404,  # 17O
    "S": 0.00421,  # 33S
    "Si": 0.04683,  # 29Si
}

# Additive M+2/M+0 contributions from heavy isotopes (Cl, Br dominate)
_M2_ADDITIVE: dict[str, float] = {
    "Cl": 0.3220,  # 37Cl
    "Br": 0.9735,  # 81Br
    "K": 0.0673,  # 41K
    "S": 0.0425,  # 34S
    "Si": 0.0337,  # 30Si
    "O": 0.00204,  # 18O
}

# mz_tol per instrument type for isotope detection.
# The M+0 search is centred on k × C13_SPACING (1.003355 Da per step) not NEUTRON_MASS.
# H-gain (+1H = 1.007825 Da) sits 4.47 mDa from C13_SPACING, so 4 mDa excludes it
# while retaining a 0.47 mDa safety margin for FT mass accuracy (~1–3 ppm).
# For k ≥ 2, the tolerance is doubled (8 mDa) to cover heavy-isotope spacings
# (³⁷Cl 1.9970 Da, ⁴¹K 1.9981 Da, ³⁴S 1.9985 Da) that differ from 2 × C13_SPACING
# (2.0067 Da) by up to ~9.7 mDa.
# All other instruments: None → never filter (H-gain and isotope indistinguishable).
_INST_MZ_TOL: dict[str, float | None] = {
    "FT": 0.004,
}


def _expected_mk_ratio(
    mol_counts: dict[str, int],
    mol_scale: float,
    k: int,
    adduct_counts: dict[str, int] | None = None,
) -> float:
    """Expected M+k/M+0 intensity ratio for a fragment, adduct-aware.

    Molecule atoms are scaled by ``mol_scale`` (≈ neutral fragment mass / neutral
    precursor mass).  Adduct atoms (e.g. K from ``[M+K]+``, Cl from ``[M+Cl]-``)
    are applied at **full weight** because the adduct charge carrier either stays
    with a fragment entirely or is absent—it does not distribute proportionally
    across all fragments the way carbon does.

    Args:
        mol_counts: Element counts for the neutral molecule from ``parse_formula``.
        mol_scale: Fraction of the neutral molecule mass present in this neutral
            fragment, i.e. ``(frag_mz - adduct_delta_mass) / (prec_mz - adduct_delta_mass)``.
        k: Isotope order (1 = M+1, 2 = M+2).
        adduct_counts: Element counts for atoms *added* by the adduct (positive
            atoms only, e.g. ``{"K": 1}`` for ``[M+K]+``).  None or empty means
            no adduct correction is applied.

    Returns:
        Expected M+k/M+0 intensity ratio, floored at 1e-4.
    """
    import math

    if adduct_counts is None:
        adduct_counts = {}

    m1_mol = sum(mol_counts.get(elem, 0) * c for elem, c in _M1_CONTRIB.items())
    m1_add = sum(adduct_counts.get(elem, 0) * c for elem, c in _M1_CONTRIB.items())
    total_m1 = m1_mol * mol_scale + m1_add
    poisson = total_m1**k / math.factorial(k)
    if k == 2:
        m2_mol = sum(mol_counts.get(elem, 0) * c for elem, c in _M2_ADDITIVE.items())
        m2_add = sum(adduct_counts.get(elem, 0) * c for elem, c in _M2_ADDITIVE.items())
        return max(poisson + m2_mol * mol_scale + m2_add, 1e-4)
    return max(poisson, 1e-4)


def _parse_adduct_counts(prec_type: str | None) -> tuple[dict[str, int], float]:
    """Return positive adduct atom counts and their total monoisotopic mass.

    Only atoms that are *added* by the adduct are returned (positive counts).
    Atoms removed by the adduct (e.g. the ``-H`` in ``[M-H]-``) are ignored
    because they reduce the precursor mass and have negligible isotope impact.

    Args:
        prec_type: Adduct/precursor type string (e.g. ``"[M+K]+"``, ``"[M+Cl]-"``).
            If None or not found in ``PREC_TYPE_TO_FORMULA_DIFF``, returns empty
            counts and zero mass.

    Returns:
        Tuple of (adduct_counts, adduct_delta_mass) where adduct_counts maps
        element symbol to integer count for atoms added by the adduct, and
        adduct_delta_mass is the total monoisotopic mass of those atoms in Da.
    """
    if prec_type is None:
        return {}, 0.0
    formula_diff = PREC_TYPE_TO_FORMULA_DIFF.get(prec_type, "")
    if not formula_diff:
        return {}, 0.0
    try:
        diff_counts = parse_formula(formula_diff)
    except (ValueError, AssertionError):
        return {}, 0.0
    adduct_counts = {elem: cnt for elem, cnt in diff_counts.items() if cnt > 0}
    adduct_delta_mass = sum(MASS(elem) * cnt for elem, cnt in adduct_counts.items())
    return adduct_counts, adduct_delta_mass


def _estimate_leak_ratio_from_spectrum(
    smzs: np.ndarray,
    sints: np.ndarray,
    k: int,
    mz_tol: float,
    min_pairs: int,
    consistency_cv: float,
    mol_counts: dict[str, int] | None = None,
    adduct_counts: dict[str, int] | None = None,
    adduct_delta_mass: float = 0.0,
    prec_neutral_mass: float = 0.0,
) -> float | None:
    """Estimate co-isolation ratio from cross-pair consistency of M+k peaks.

    When ``mol_counts`` is provided, each raw ratio ``r_i = I(M+k_i) / I(M_i)``
    is normalised by the formula-predicted natural M+k/M+0 for that fragment:

        r_normalised_i = r_i / R_exp_frag_i ≈ R_leak   (co-isolation amplification)

    Because ``R_exp_frag_i`` varies with fragment composition while ``R_leak`` is
    a spectrum-level constant, normalising collapses the distribution so Coefficient of Variation (CV) is
    small only when genuine co-isolation is present.  Without ``mol_counts`` the
    raw ``r_i`` values are used and the function returns the median as ``R_k``
    (the old behaviour, appropriate when the formula is unavailable).

    Args:
        smzs: Sorted m/z array.
        sints: Sorted intensity array (same order as smzs).
        k: Isotope order (1 = M+1, 2 = M+2).
        mz_tol: Per-step m/z tolerance in Da (doubled for k ≥ 2 by caller).
        min_pairs: Minimum number of (M, M+k) pairs required to accept consensus.
        consistency_cv: Maximum coefficient of variation (std/mean) of values to
            accept the consensus as a genuine co-isolation signal.
        mol_counts: Element counts of the neutral precursor molecule.  When given,
            enables formula-normalised R_leak estimation.
        adduct_counts: Element counts added by the adduct (positive atoms only).
        adduct_delta_mass: Monoisotopic mass of adduct atoms (Da).
        prec_neutral_mass: Neutral precursor mass ``prec_mz - adduct_delta_mass``
            used as denominator for the per-fragment mol_scale estimate.

    Returns:
        Consensus R_leak (when ``mol_counts`` given) or R_k (when not), or None
        if the signal is inconsistent or there are too few pairs.
    """
    k_tol = mz_tol if k == 1 else mz_tol * 2
    n = len(smzs)
    r_vals: list[float] = []
    use_formula = mol_counts is not None and prec_neutral_mass > 0

    for i in range(n):
        m0_int = sints[i]
        if m0_int <= 0:
            continue
        target = smzs[i] + k * C13_SPACING
        lo = np.searchsorted(smzs, target - k_tol, side="left")
        hi = np.searchsorted(smzs, target + k_tol, side="right")
        if lo >= hi:
            continue
        mk_int = sints[lo + int(np.argmax(sints[lo:hi]))]
        # Monotonicity: M+0 must be more intense (R_k < 1 for typical co-isolation)
        if mk_int >= m0_int:
            continue
        r_raw = mk_int / m0_int
        if use_formula:
            neutral_frag = smzs[i] - adduct_delta_mass
            mol_scale = max(0.0, min(1.0, neutral_frag / prec_neutral_mass))
            r_exp = _expected_mk_ratio(mol_counts, mol_scale, k, adduct_counts or {})
            if r_exp <= 0:
                continue
            r_vals.append(r_raw / r_exp)  # normalised → R_leak
        else:
            r_vals.append(r_raw)  # unnormalised → R_k

    if len(r_vals) < min_pairs:
        return None

    r_arr = np.array(r_vals)
    r_median = float(np.median(r_arr))
    if r_median <= 0:
        return None

    cv = float(np.std(r_arr) / r_median)
    if cv > consistency_cv:
        return None  # values too scattered → not a coherent co-isolation signal

    return r_median


def detect_isotope_peaks_for_training(
    mzs: np.ndarray,
    ints: np.ndarray,
    prec_mz: float,
    inst_type: str,
    max_isotope: int = 2,
    require_monotone: bool = True,
    require_envelope: bool = True,
    ratio_lo: float = 0.7,
    ratio_hi: float = 2.0,
    aggressive: bool = True,
    aggressive_min_pairs: int = 3,
    aggressive_cv: float = 0.3,
    formula: str | None = None,
    prec_type: str | None = None,
    min_coiso_fraction: float = 0.05,
    max_leak_factor: float = 1.5,
) -> np.ndarray:
    """Detect isotope leak peaks for training-data cleaning.

    Designed for high precision (minimal false positives) rather than high recall,
    because false positives delete real H-transfer peaks that the model must learn.

    **Physics model** (``formula`` provided):

    The isolation window co-selects a fraction ``f`` of the M+k precursor isotopologue
    alongside M+0.  Both fragment identically, so every fragment ion gains a satellite
    at ``+k × C13_SPACING`` with intensity proportional to ``f``:

        observed ratio at fragment i  =  R_exp_frag_i × f

    where ``R_exp_frag_i`` is the natural M+k/M+0 ratio for that fragment's formula
    (proportional to carbon/heteroatom count) and ``f`` is the co-isolation fraction
    (0 < f ≤ 1).  ``f`` is estimated from the precursor residual peaks above
    ``prec_mz`` (primary path) or from cross-pair consistency of in-spectrum
    fragment M+k pairs (aggressive fallback):

        f  =  R_k_prec / R_exp_prec          (primary: precursor residuals)
        f  ≈  median(r_raw_i / R_exp_frag_i) (aggressive: formula-normalised pairs)

    Only spectra where ``f ≥ min_coiso_fraction`` are flagged.  Because ``f ≤ 1``
    always, the old ``min_leak_factor ≥ 1.5`` threshold was physically impossible
    for genuine co-isolation — this is the corrected formulation.

    Per-fragment expected ratio is estimated as
    ``R_exp_frag_i ≈ _expected_mk_ratio(mol_counts, frag_mz/prec_neutral, k)``
    using the fragment's m/z as a proxy for its formula size.  This correctly
    scales the detection threshold: small fragments require a much smaller
    absolute M+k/M+0 ratio to be considered a leak than large ones.

    **Without formula** (``formula=None``): falls back to comparing ``obs ≈ R_k``
    uniformly across all fragments (previous behaviour).  No ``R_leak`` gating.

    **Default mode** (``aggressive=False``): ``R_k`` / ``R_leak`` is read from
    the precursor's own M+k peaks above ``prec_mz``.  Returns all-False when
    those peaks are absent.

    **Aggressive mode** (``aggressive=True``): when no above-prec reference is
    found, estimates ``R_leak`` from cross-pair consistency of in-spectrum M+k
    candidates, formula-normalised when ``mol_counts`` is available.

    Key design choices:

    * **Instrument gating**: Only FT spectra are filtered.
    * **Tight mz_tol**: 4 mDa (k=1), 8 mDa (k≥2), centred on C13_SPACING.
    * **Monotonicity**: Requires I(M+0_frag) > I(M+k_frag).
    * **Envelope coherence**: M+k requires M+1…M+(k-1) confirmed first.

    Args:
        mzs: Peak m/z values of shape (N,).  Need not be sorted.
        ints: Peak intensities of shape (N,).  Same order as ``mzs``.
        prec_mz: Precursor m/z.
        inst_type: Instrument type string (``"FT"``, ``"QTOF"``, ``"IT"``).
            Only ``"FT"`` is currently filtered.
        max_isotope: Maximum isotope order to consider.  Defaults to 2.
        require_monotone: If True, require I(M+0_frag) > I(M+k_frag).
        require_envelope: If True, require all intermediate M+j confirmed first.
        ratio_lo: Lower bound multiplier on the per-fragment expected ratio.
        ratio_hi: Upper bound multiplier on the per-fragment expected ratio.
        aggressive: If True and no above-prec reference found, estimate R_leak
            from cross-pair consistency.  Defaults to False.
        aggressive_min_pairs: Min pairs for aggressive consensus.  Defaults to 3.
        aggressive_cv: Max CV of (normalised) r_i for aggressive mode.  Defaults to 0.3.
        formula: Neutral molecular formula string (e.g. ``"C10H14N2"``).  When
            given, enables R_leak gating and per-fragment expected-ratio correction.
        prec_type: Adduct/precursor type (e.g. ``"[M+H]+"``, ``"[M-H]-"``).
            Used together with ``formula`` to account for adduct atoms.
        min_coiso_fraction: Minimum co-isolation fraction ``f = R_k / R_exp_prec``
            required to trigger flagging.  For genuine co-isolation ``f ≤ 1``; below
            this threshold the signal is too weak to distinguish from noise.
            Only active when ``formula`` is provided.  Defaults to 0.05.
        max_leak_factor: Maximum ``f = R_k / R_exp_prec`` accepted for the above-prec
            reference peak.  Genuine co-isolation gives ``f ≤ 1``; values above this
            indicate a stray fragment coincidentally at the isotope position.
            Only active when ``formula`` is provided.  Defaults to 1.5.

    Returns:
        Boolean array of shape (N,) where True means co-isolation isotope leak.
        All-False for non-FT instruments or when co-isolation cannot be confirmed.

    Example:
        >>> mzs = np.array([100.0, 101.003, 300.0, 301.003])
        >>> ints = np.array([1000.0, 50.0, 2000.0, 100.0])  # R_1 = 100/2000 = 0.05
        >>> mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=300.0, inst_type="FT")
        >>> mask[1]  # fragment M+1: ratio 50/1000=0.05 == R_1 → True
        True
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.shape != ints.shape or mzs.ndim != 1:
        raise ValueError(f"mzs and ints must be 1-D arrays of the same length, got {mzs.shape}")

    n = len(mzs)
    is_isotope = np.zeros(n, dtype=bool)

    # Gate: only FT spectra are filtered
    mz_tol = _INST_MZ_TOL.get(inst_type)
    if mz_tol is None:
        return is_isotope

    if n == 0 or prec_mz <= 0:
        return is_isotope

    sort_idx = np.argsort(mzs)
    smzs = mzs[sort_idx]
    sints = ints[sort_idx]

    def _build_tol_with_floor(tol: float, tol_floor: float = 1e-6) -> float:
        """Return a non-zero tolerance window using a small floor."""
        return max(float(tol), tol_floor)

    def _find_peak(target: float, tol: float) -> tuple[float, float] | None:
        """Return (mz, intensity) of the most intense peak within tol, or None."""
        tol = _build_tol_with_floor(tol)
        lo = np.searchsorted(smzs, target - tol, side="left")
        hi = np.searchsorted(smzs, target + tol, side="right")
        if lo >= hi:
            return None
        best = lo + int(np.argmax(sints[lo:hi]))
        return smzs[best], sints[best]

    # Precompute precursor formula-based expected ratios when formula is known.
    # R_exp_prec[k] = natural M+k/M+0 for the intact precursor (adduct-inclusive).
    # R_leak[k]     = R_k / R_exp_prec[k] — pure co-isolation amplification factor.
    # Only genuine co-isolation (R_leak >= min_leak_factor) triggers flagging.
    mol_counts: dict[str, int] | None = None
    adduct_counts_parsed: dict[str, int] = {}
    adduct_delta_mass: float = 0.0
    r_exp_prec: dict[int, float] = {}

    if formula is not None:
        try:
            mol_counts = parse_formula(formula)
            adduct_counts_parsed, adduct_delta_mass = _parse_adduct_counts(prec_type)
            for _k in range(1, max_isotope + 1):
                r_exp_prec[_k] = _expected_mk_ratio(mol_counts, 1.0, _k, adduct_counts_parsed)
        except Exception:
            mol_counts = None
            r_exp_prec = {}

    prec_neutral_mass = prec_mz - adduct_delta_mass if mol_counts is not None else 0.0

    # Step 1: Estimate R_k from precursor M+k peaks above prec_mz (primary path).
    # When formula is known, gate on R_leak = R_k/R_exp_prec >= min_leak_factor to
    # avoid false positives from natural precursor M+1 peaks in high-S/N spectra.
    M0_TOL = 0.05
    # Require prec M+0 to be at least 0.5% of the base peak; noise peaks at
    # prec_mz (int ≈ 5-10 in a spectrum with base 10000) produce bogus R_k values.
    MIN_PREC_M0_FRAC = 0.01
    base_int = float(sints.max()) if len(sints) > 0 else 0.0
    prec_m0 = _find_peak(prec_mz, M0_TOL)
    prec_m0_int: float | None = None
    if prec_m0 is not None and prec_m0[1] > 0 and (base_int <= 0 or prec_m0[1] / base_int >= MIN_PREC_M0_FRAC):
        prec_m0_int = prec_m0[1]

    leak_ratios: dict[int, float] = {}  # k -> R_k  (= R_leak * R_exp_prec[k])
    r_leak: dict[int, float] = {}  # k -> R_leak (only when formula known)

    if prec_m0_int is not None:
        for k in range(1, max_isotope + 1):
            k_tol = mz_tol if k == 1 else mz_tol * 2
            peak = _find_peak(prec_mz + k * C13_SPACING, k_tol)
            if peak is None:
                continue
            pk_mz, pk_int = peak
            if pk_mz <= prec_mz + k_tol:
                continue  # not strictly above prec_mz
            R_k = pk_int / prec_m0_int
            if r_exp_prec:
                r_exp_k = r_exp_prec.get(k, 0.0)
                if r_exp_k > 0:
                    factor = R_k / r_exp_k
                    if factor < min_coiso_fraction:
                        continue  # co-isolation fraction too small — skip
                    if factor > max_leak_factor:
                        continue  # too intense — stray fragment, not co-isolation
                    r_leak[k] = factor
            leak_ratios[k] = R_k

    # Step 1b: R_leak consistency check across k values (formula mode only).
    # Both k=1 and k=2 come from the same isolation window, so their co-isolation
    # fractions must be equal: r_leak[1] ≈ r_leak[2] ≈ f.
    # If k=2 is present but its ratio disagrees with k=1, both are stray fragments.
    # If k=2 is absent (e.g. small molecule where M+2 residual is below noise),
    # keep k=1 — absence of k=2 does not indicate a stray fragment.
    if r_exp_prec and max_isotope >= 2 and 1 in r_leak and 2 in r_leak:
        ratio = r_leak[2] / r_leak[1]
        if not (ratio_lo <= ratio <= ratio_hi):
            # r_leak[2] inconsistent with r_leak[1] — both are stray fragments
            leak_ratios.clear()
            r_leak.clear()

    # Step 2 (aggressive fallback): when above-prec reference is unavailable,
    # estimate R_leak (formula-normalised) or R_k (no formula) from cross-pair
    # consistency of in-spectrum M+k candidates.
    if aggressive:
        for k in range(1, max_isotope + 1):
            if k in leak_ratios:
                continue
            r_consensus = _estimate_leak_ratio_from_spectrum(
                smzs,
                sints,
                k,
                mz_tol=mz_tol,
                min_pairs=aggressive_min_pairs,
                consistency_cv=aggressive_cv,
                mol_counts=mol_counts,
                adduct_counts=adduct_counts_parsed if mol_counts is not None else None,
                adduct_delta_mass=adduct_delta_mass,
                prec_neutral_mass=prec_neutral_mass,
            )
            if r_consensus is None:
                continue
            if mol_counts is not None:
                # r_consensus ≈ f (co-isolation fraction); apply min_coiso_fraction gate
                if r_consensus < min_coiso_fraction:
                    continue
                r_leak[k] = r_consensus
                r_exp_k = r_exp_prec.get(k, 0.0)
                leak_ratios[k] = r_consensus * r_exp_k if r_exp_k > 0 else r_consensus
            else:
                leak_ratios[k] = r_consensus

    if not leak_ratios:
        return is_isotope

    # Step 3: Flag each peak that is the M+k satellite of a lower-m/z fragment.
    # When formula is known, per-fragment expected ratio = R_exp_frag_i × R_leak[k]
    # where R_exp_frag_i is estimated via m/z-based mol_scale.  This correctly
    # scales the window for each fragment's composition instead of applying the
    # precursor-level R_k uniformly.
    def _frag_expected(m0_mz: float, k: int) -> float:
        """Return per-fragment expected M+k/M+0 ratio given its M+0 m/z."""
        if mol_counts is None or k not in r_leak or prec_neutral_mass <= 0:
            return leak_ratios[k]  # fallback: use R_k directly
        neutral_frag = m0_mz - adduct_delta_mass
        mol_scale = max(0.0, min(1.0, neutral_frag / prec_neutral_mass))
        r_exp_frag = _expected_mk_ratio(mol_counts, mol_scale, k, adduct_counts_parsed)
        return r_exp_frag * r_leak[k]

    for i in range(n):
        frag_mz = smzs[i]
        frag_int = sints[i]

        for k in leak_ratios:
            k_tol = mz_tol if k == 1 else mz_tol * 2
            m0 = _find_peak(frag_mz - k * C13_SPACING, k_tol)
            if m0 is None:
                continue
            m0_mz, m0_int = m0

            if require_monotone and m0_int <= frag_int:
                continue
            if m0_int <= 0:
                continue

            obs = frag_int / m0_int
            R_expected = _frag_expected(m0_mz, k)
            if not (ratio_lo * R_expected <= obs <= ratio_hi * R_expected):
                continue

            # Envelope coherence: all intermediate M+j peaks must be present
            # and match their per-fragment expected ratio before accepting M+k.
            if require_envelope and k > 1:
                envelope_ok = True
                for j in range(1, k):
                    if j not in leak_ratios:
                        envelope_ok = False
                        break
                    j_tol = mz_tol if j == 1 else mz_tol * 2
                    mj = _find_peak(frag_mz - (k - j) * C13_SPACING, j_tol)
                    if mj is None:
                        envelope_ok = False
                        break
                    _, mj_int = mj
                    if require_monotone and mj_int >= m0_int:
                        envelope_ok = False
                        break
                    if m0_int > 0:
                        obs_j = mj_int / m0_int
                        R_exp_j = _frag_expected(m0_mz, j)
                        if not (ratio_lo * R_exp_j <= obs_j <= ratio_hi * R_exp_j):
                            envelope_ok = False
                            break
                if not envelope_ok:
                    continue

            is_isotope[sort_idx[i]] = True
            break

    return is_isotope
