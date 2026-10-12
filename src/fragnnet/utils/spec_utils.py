import logging
from collections.abc import Callable

import numpy as np
import torch as th

from fragnnet.utils.misc_utils import EPS, validate_bin_geometry
from fragnnet.utils.scatter_utils import scatter_l1normalize, scatter_logsumexp, scatter_reduce

logger = logging.getLogger(__name__)


def weighted_mean_by_batch(
    values: th.Tensor,
    weights: th.Tensor,
    batch_indices: th.Tensor | None,
    batch_size: int,
) -> th.Tensor:
    """Compute sum(w*x)/sum(w) for each spectrum's CE features.

    Args:
        values: Reading features with shape (N, D).
        weights: Finite, nonnegative reading weights with shape (N,).
        batch_indices: Long spectrum indices (N,), or None for one reading per spectrum.
        batch_size: Number of spectra; each must have positive total weight.

    Returns:
        Weighted feature means with shape (batch_size, D).

    Raises:
        ValueError: If shapes, weights, or spectrum indices are invalid.
    """
    if values.ndim != 2 or weights.shape != (values.shape[0],):
        raise ValueError("CE weights must match the number of reading features")
    if batch_indices is None:
        if values.shape[0] != batch_size:
            raise ValueError("One CE reading per spectrum is required without batch indices")
        batch_indices = th.arange(batch_size, device=values.device)
    if batch_indices.shape != weights.shape or batch_indices.dtype != th.long:
        raise ValueError("CE batch indices must be a matching long vector")
    weights = weights.to(values)
    valid_weights = (th.isfinite(weights) & (weights >= 0)).all()
    valid_indices = ((batch_indices >= 0) & (batch_indices < batch_size)).all()
    if values.is_cuda or th.compiler.is_compiling():
        th._assert_async(valid_weights, "CE weights must be finite and nonnegative")
        th._assert_async(valid_indices, "CE batch indices are out of range")
    else:
        if not bool(valid_weights):
            raise ValueError("CE weights must be finite and nonnegative")
        if not bool(valid_indices):
            raise ValueError("CE batch indices are out of range")
    totals = values.new_zeros(batch_size).index_add(0, batch_indices, weights)
    if values.is_cuda or th.compiler.is_compiling():
        th._assert_async((totals > 0).all(), "Each spectrum must have positive total CE weight")
    elif not bool((totals > 0).all()):
        raise ValueError("Each spectrum must have positive total CE weight")
    sums = values.new_zeros(batch_size, values.shape[1]).index_add(
        0, batch_indices, values * weights.unsqueeze(1)
    )
    return sums / totals.unsqueeze(1)


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

    elif ints_transform == "pow0.75":

        def _func(ints: th.Tensor) -> th.Tensor:
            return ints**0.75

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
    elif ints_transform == "pow0.75":
        max_ints = float(1000.0**0.75)

        def _untransform_fn_pow075(x: th.Tensor) -> th.Tensor:
            return x ** (4.0 / 3.0)

        _untransform_fn = _untransform_fn_pow075
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
    """Transform collision energy (CE) values via z-score normalization.

    Values below 0 are treated as missing/unknown and returned as-is.

    Args:
        ce: The CE values to transform.
        ce_mean: Mean CE value for normalization.
        ce_std: Standard deviation of CE values for normalization.

    Returns:
        Transformed CE values of the same type as input.
    """
    if isinstance(ce, th.Tensor):
        mask = ce >= 0
        transformed_ce = th.where(mask, (ce - ce_mean) / ce_std, ce)
        return transformed_ce
    else:
        if ce >= 0:
            return th.tensor((ce - ce_mean) / ce_std)
        else:
            return th.tensor(ce)


def get_split_fn(
    bs_idx: th.Tensor, batch_size: int | None = None
) -> Callable[[th.Tensor], tuple[th.Tensor, ...]]:
    """Build a splitter for tensors grouped by sorted batch indices.

    Args:
        bs_idx: Sorted batch index tensor aligned with the tensor to split.
        batch_size: Number of batch entries. Indices below it without rows (trailing ones
            included) get zero-length splits, which keeps per-spectrum outputs aligned after
            filtering removes every entry of a spectrum. Without it, the splits run to the
            largest index in ``bs_idx``.

    Returns:
        A function that splits any tensor aligned with ``bs_idx`` into per-batch chunks.
    """
    if len(bs_idx) > 1:
        assert (bs_idx[1:] - bs_idx[:-1]).min() >= 0
    batch_counts = th.bincount(bs_idx.long(), minlength=batch_size or 0).detach().cpu().tolist()

    def split_fn(inp_tensor: th.Tensor) -> tuple[th.Tensor, ...]:
        return th.split(inp_tensor, batch_counts)

    return split_fn


def ragged_to_batch(
    feat: th.Tensor, batch_idx: th.Tensor, pad_value: float = 0
) -> tuple[th.Tensor, th.Tensor]:
    assert batch_idx.ndim == 1
    assert feat.size(0) == batch_idx.size(0)

    device = feat.device
    N = batch_idx.size(0)

    lengths = th.bincount(batch_idx)
    B = lengths.size(0)
    max_len = lengths.max()

    offsets = th.cumsum(lengths, dim=0)
    offsets = th.cat([th.zeros(1, device=device, dtype=offsets.dtype), offsets[:-1]])
    local_idx = th.arange(N, device=device) - offsets[batch_idx]
    assert feat.ndim == 1

    out = th.full((B, max_len), pad_value, device=device, dtype=feat.dtype)
    out[batch_idx, local_idx] = feat
    return out


def primary_ce_to_batch(
    spec_ce: th.Tensor, spec_ce_batch_idxs: th.Tensor | None, batch_size: int
) -> th.Tensor:
    """Extract one collision energy value per batch element.

    Args:
        spec_ce: Collision energy tensor. This is typically flattened by collation and has
            shape (num_ce_values,) or (batch_size, num_ce_values_per_sample).
        spec_ce_batch_idxs: Batch indices for flattened CE values of shape (num_ce_values,).
        batch_size: Number of samples in the batch.

    Returns:
        Collision energy tensor of shape (batch_size,).
    """
    if spec_ce.ndim == 2:
        if spec_ce.size(1) == 1:
            # batch_func produced (total_ces, 1) — squeeze trailing dim and
            # fall through to the sparse batch_idxs path below.
            spec_ce = spec_ce.squeeze(-1)
        elif spec_ce.size(0) == batch_size:
            # Dense layout (batch_size, num_ces) — take first CE per sample.
            return spec_ce[:, 0].detach().cpu()
        else:
            raise ValueError(
                "Expected spec_ce to have batch_size rows when it is 2D, "
                f"got shape {tuple(spec_ce.shape)} and batch_size={batch_size}"
            )

    flat_spec_ce = spec_ce.reshape(-1).detach().cpu()
    if spec_ce_batch_idxs is None:
        if flat_spec_ce.numel() != batch_size:
            raise ValueError(
                "spec_ce_batch_idxs is required when spec_ce is flattened and does not "
                f"already have one value per batch item; got {flat_spec_ce.numel()} values "
                f"for batch_size={batch_size}"
            )
        return flat_spec_ce

    flat_batch_idxs = spec_ce_batch_idxs.reshape(-1).detach().cpu().long()
    if flat_batch_idxs.numel() != flat_spec_ce.numel():
        raise ValueError(
            "spec_ce and spec_ce_batch_idxs must have the same number of elements; "
            f"got {flat_spec_ce.numel()} and {flat_batch_idxs.numel()}"
        )

    ce_primary = th.full((batch_size,), float("nan"), dtype=flat_spec_ce.dtype)
    for batch_idx in range(batch_size):
        batch_mask = flat_batch_idxs == batch_idx
        if batch_mask.any():
            ce_primary[batch_idx] = flat_spec_ce[batch_mask][0]
    return ce_primary


def select_context_ce_values(
    batch: dict,
    batch_size: int,
    inst_types: list[str],
    ce_key_by_inst: dict[str, str],
) -> th.Tensor:
    """Select the sampler's configured CE axis for every spectrum."""
    ce_by_key = {}
    for ce_key in set(ce_key_by_inst.values()):
        values = batch[f"spec_{ce_key}"]
        batch_idxs = batch[f"spec_{ce_key}_batch_idxs"]
        valid = th.isfinite(values) & (values >= 0)
        value_sums = scatter_reduce(
            th.where(valid, values, 0.0),
            batch_idxs,
            reduce="sum",
            dim_size=batch_size,
            include_self=False,
        )
        valid_counts = scatter_reduce(
            valid.to(values.dtype),
            batch_idxs,
            reduce="sum",
            dim_size=batch_size,
            include_self=False,
        )
        mean_values = value_sums / valid_counts.clamp(min=1)
        mean_values = th.where(
            valid_counts > 0,
            th.round(mean_values * 1.0e6) / 1.0e6,
            th.full_like(mean_values, float("nan")),
        )
        ce_by_key[ce_key] = mean_values
    inst_idxs = batch["spec_inst_type"].reshape(-1)
    ce_values = ce_by_key[next(iter(ce_by_key))].new_full((batch_size,), float("nan"))
    for inst_idx, inst_type in enumerate(sorted(inst_types)):
        ce_key = ce_key_by_inst[inst_type]
        ce_values[inst_idxs == inst_idx] = ce_by_key[ce_key][inst_idxs == inst_idx]
    return ce_values


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
    top_k_peaks: int | None = None,
    drop_min_int_peak: bool = False,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """bacthed filter func for filter spectra

    Args:
        mzs (th.Tensor): 1d flat tensor of m/zs; multiple m/z lists are concatenated.
        ints (th.Tensor): 1d flat tensor of intensities; multiple intensities lists are concatenated.
        batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.
        ints_thresh (float): intesnity thresh hold
        mz_max (float): max m/z, if mz_max <= 0, mz_max filter will be ignored
        top_k_peaks: If positive, keep only the k highest-intensity peaks per spectrum.
            None or a negative value disables top-k filtering.
        drop_min_int_peak: If True, drop the weakest peak per spectrum when more than one peak remains.

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

    if top_k_peaks is not None and top_k_peaks == 0:
        raise ValueError(f"top_k_peaks must be positive or None, got {top_k_peaks}")
    if top_k_peaks is not None and top_k_peaks < 0:
        top_k_peaks = None

    if (top_k_peaks is not None or drop_min_int_peak) and ints.numel() > 0:
        keep_mask = th.ones(ints.shape[0], dtype=th.bool, device=ints.device)
        for batch_idx in th.unique(batch_idxs, sorted=True):
            idxs = th.nonzero(batch_idxs == batch_idx, as_tuple=False).flatten()
            if idxs.numel() == 0:
                continue

            local_keep = th.ones(idxs.shape[0], dtype=th.bool, device=ints.device)
            if drop_min_int_peak and idxs.numel() > 1:
                local_keep[th.argmin(ints[idxs])] = False

            kept_local = th.nonzero(local_keep, as_tuple=False).flatten()
            if top_k_peaks is not None and kept_local.numel() > top_k_peaks:
                local_ints = ints[idxs[kept_local]]
                _, top_local = th.topk(local_ints, k=top_k_peaks)
                top_keep = th.zeros_like(local_keep)
                top_keep[kept_local[top_local]] = True
                local_keep = top_keep

            keep_mask[idxs] = local_keep

        mzs = mzs[keep_mask]
        ints = ints[keep_mask]
        batch_idxs = batch_idxs[keep_mask]
    return mzs, ints, batch_idxs


def batched_rel_filter_func(
    mzs: th.Tensor,
    ints: th.Tensor,
    batch_idxs: th.Tensor,
    rel_thresh: float,
    mz_max: float = 0.0,
    dim_size: int | None = None,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Filter peaks whose intensity is below a per-spectrum relative threshold.

    Args:
        mzs: 1-D tensor of m/z values, concatenated across batch.
        ints: 1-D tensor of raw intensities, concatenated across batch.
        batch_idxs: 1-D tensor of batch indices, same length as mzs/ints.
        rel_thresh: Fraction of per-spectrum maximum intensity below which peaks
            are dropped (e.g. 0.01 keeps only peaks ≥ 1 % of that spectrum's max).
            A value of 0.0 is a no-op.
        mz_max: If positive, also drop peaks at or above this m/z. Defaults to 0.0 (disabled).
        dim_size: Number of spectra in the batch, if already known on the host. Passing this
            avoids a `th.max(batch_idxs).item()` device sync; omit only when the caller
            genuinely does not know the batch size in advance.

    Returns:
        Filtered (mzs, ints, batch_idxs) tensors.
    """
    if rel_thresh <= 0.0 and mz_max <= 0.0:
        return mzs, ints, batch_idxs

    if rel_thresh > 0.0:
        # Per-spectrum maxima are only needed for a positive relative threshold. With
        # rel_thresh == 0.0 the comparison below reduces to `ints > 0`, so computing them
        # would be a scatter plus a host sync whose result is multiplied by zero.
        if dim_size is None:
            dim_size = int(th.max(batch_idxs).item()) + 1
        max_ints = scatter_reduce(ints, batch_idxs, reduce="max", dim_size=dim_size)
        mask = ints > rel_thresh * max_ints[batch_idxs]
    else:
        mask = ints > 0
    if mz_max > 0.0:
        mask = mask & (mzs < mz_max)
    return mzs[mask], ints[mask], batch_idxs[mask]


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
    # Validate bin geometry
    num_bins = validate_bin_geometry(mz_max, mz_bin_res)
    batch_size = int(th.max(batch_idxs).item()) + 1
    bins = th.arange(
        mz_bin_res,
        mz_max + mz_bin_res,
        step=mz_bin_res,
        device=mzs.device,
        dtype=mzs.dtype,
    )
    # num_bins already validated above
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
            logger.warning("bin_spec is all zeros!")
            mask = th.zeros_like(bin_spec, dtype=th.bool)
            mask[:, 0] = 1.0
            mask = mask * th.all(bin_spec == 0.0, dim=1, keepdim=True)
            bin_spec = bin_spec + mask.float()
        return bin_spec


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
        Tuple[th.Tensor,th.Tensor,th.Tensor]: round_mzs, round_ints, round_batch_idxs,
            ordered by batch index and then by ascending m/z. Callers that slice the
            result by batch (``sparse_peak_list_distance``) rely on that ordering.
    """

    # Group across the whole batch at once: two stable sorts put the peaks in
    # lexicographic (batch, rounded m/z) order, which is the order a per-spectrum
    # th.unique produces, and equal neighbours in that order are exactly one group.
    round_mzs = th.round(mzs, decimals=decimals)
    order = th.argsort(round_mzs, stable=True)
    order = order[th.argsort(batch_idxs[order], stable=True)]
    sorted_mzs = round_mzs[order]
    sorted_ints = ints[order]
    sorted_batch_idxs = batch_idxs[order]

    group_starts = th.ones_like(sorted_batch_idxs, dtype=th.bool)
    group_starts[1:] = (sorted_batch_idxs[1:] != sorted_batch_idxs[:-1]) | (
        sorted_mzs[1:] != sorted_mzs[:-1]
    )
    subset_idxs = th.cumsum(group_starts, dim=0) - 1
    start_idxs = th.nonzero(group_starts, as_tuple=True)[0]
    dim_size = int(start_idxs.numel())

    if agg in ["sum", "amax"]:
        round_ints = scatter_reduce(
            src=sorted_ints,
            index=subset_idxs,
            reduce=agg,
            dim_size=dim_size,
        )
    else:
        assert agg == "lse", agg
        round_ints = scatter_logsumexp(
            logits=sorted_ints,
            subset_idxs=subset_idxs,
            dim_size=dim_size,
        )
    return sorted_mzs[start_idxs], round_ints, sorted_batch_idxs[start_idxs]


def aggregate_peaks_exact_mz(
    mzs: th.Tensor,
    ints: th.Tensor,
    batch_idxs: th.Tensor,
    decimals: int = 4,
    agg: str = "sum",
    *,
    return_inverse: bool = False,
) -> tuple[th.Tensor, ...]:
    """Aggregate peaks as ``round_aggregate_peaks`` does, but keep each peak's exact m/z.

    Groups are the same: (batch index, m/z rounded to ``decimals``), with intensities reduced by
    ``agg`` ("sum" or "amax"), so the peaks and intensities equal what the Hungarian scores see.
    The m/z returned for a group is its members' shared exact m/z when they all have one (members
    that share a formula do), and otherwise the intensity-weighted mean
    ``sum_i(w_i * mz_i) / sum_i(w_i)``, falling back to the first member when every weight is 0.
    So a 0.01 Da binning of the output sees the bins a binning of the unaggregated peaks would.

    Args:
        mzs: Exact m/z, shape (N,).
        ints: Intensities (linear, nonnegative), shape (N,).
        batch_idxs: Spectrum index of each peak, shape (N,).
        decimals: Rounding that defines one peak. Defaults to 4, as in ``round_aggregate_peaks``.
        agg: Intensity reduction within a peak, "sum" or "amax".
        return_inverse: Also return the aggregated peak each input entry landed in, which is
            what lets a caller carry per-entry data (e.g. fragment annotation) onto the peaks.

    Returns:
        (mzs, ints, batch_idxs) of the aggregated peaks, each shape (P,), ordered by batch index
        and then by ascending rounded m/z. With ``return_inverse``, a fourth tensor of shape
        (N,) holds each input entry's index into those P peaks.
    """
    if agg not in ("sum", "amax"):
        raise ValueError(f"agg must be 'sum' or 'amax', got {agg!r}")
    round_mzs = th.round(mzs, decimals=decimals)
    order = th.argsort(round_mzs, stable=True)
    order = order[th.argsort(batch_idxs[order], stable=True)]
    sorted_round_mzs = round_mzs[order]
    sorted_mzs = mzs[order]
    sorted_ints = ints[order]
    sorted_batch_idxs = batch_idxs[order]

    group_starts = th.ones_like(sorted_batch_idxs, dtype=th.bool)
    group_starts[1:] = (sorted_batch_idxs[1:] != sorted_batch_idxs[:-1]) | (
        sorted_round_mzs[1:] != sorted_round_mzs[:-1]
    )
    subset_idxs = th.cumsum(group_starts, dim=0) - 1
    start_idxs = th.nonzero(group_starts, as_tuple=True)[0]
    dim_size = int(start_idxs.numel())

    group_ints = scatter_reduce(src=sorted_ints, index=subset_idxs, reduce=agg, dim_size=dim_size)
    weight_sums = scatter_reduce(
        src=sorted_ints.to(sorted_mzs.dtype), index=subset_idxs, reduce="sum", dim_size=dim_size
    )
    weighted_mzs = scatter_reduce(
        src=sorted_mzs * sorted_ints.to(sorted_mzs.dtype),
        index=subset_idxs,
        reduce="sum",
        dim_size=dim_size,
    )
    first_mzs = sorted_mzs[start_idxs]
    mean_mzs = weighted_mzs / weight_sums.clamp_min(th.finfo(sorted_mzs.dtype).tiny)
    # A group whose members share one exact m/z returns it bit for bit, not a mean that can
    # land an ulp away (and across a bin edge).
    shared = scatter_reduce(
        src=sorted_mzs, index=subset_idxs, reduce="amin", dim_size=dim_size, include_self=False
    ) == scatter_reduce(
        src=sorted_mzs, index=subset_idxs, reduce="amax", dim_size=dim_size, include_self=False
    )
    group_mzs = th.where(shared | (weight_sums <= 0), first_mzs, mean_mzs)
    if not return_inverse:
        return group_mzs, group_ints, sorted_batch_idxs[start_idxs]
    inverse = th.empty_like(subset_idxs)
    inverse[order] = subset_idxs
    return group_mzs, group_ints, sorted_batch_idxs[start_idxs], inverse


### helpers


