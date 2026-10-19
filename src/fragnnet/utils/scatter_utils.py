"""Grouped (scatter) tensor reductions over a batch index: sum/mean/max via scatter_reduce,
log-sum-exp, log-softmax, L1/L2 normalization, arg-max and top-k, plus safelog and x_log_x
(split out of ``utils/misc_utils.py``).
"""

import torch as th

from fragnnet.utils.misc_utils import EPS, LOG_ZERO


def scatter_logsumexp(
    logits: th.Tensor, subset_idxs: th.Tensor, eps: float = EPS, dim_size: int | None = None
) -> th.Tensor:
    """Numerically stable logsumexp grouped by ``subset_idxs``."""

    # numel() is metadata — no GPU sync. Return LOG_ZERO for every group when input is empty.
    if logits.numel() == 0:
        k = dim_size if dim_size is not None else 0
        return th.full((k,), LOG_ZERO(logits.dtype), device=logits.device, dtype=logits.dtype)

    if dim_size is None:
        # An int, not a tensor: a tensor dim_size would sync again in every scatter_reduce.
        k = int(th.max(subset_idxs).item()) + 1
    else:
        # On CUDA the scatter_reduce calls below bound-check subset_idxs against dim_size
        # asynchronously, so a blocking assert here would only add a host sync.
        if subset_idxs.device.type != "cuda":
            assert dim_size >= th.max(subset_idxs) + 1
        k = dim_size

    sm = scatter_reduce(
        src=logits,
        index=subset_idxs,
        reduce="amax",
        dim_size=k,
        default=LOG_ZERO(logits.dtype),
    )

    lm = th.gather(input=sm, index=subset_idxs, dim=0)
    logits = logits - lm

    se = scatter_reduce(
        src=th.exp(logits), index=subset_idxs, reduce="sum", dim_size=k, default=0.0
    )
    return sm + th.log(se + eps)


def scatter_logmeanexp(
    logits: th.Tensor, subset_idxs: th.Tensor, eps: float = EPS, dim_size: int | None = None
) -> th.Tensor:
    """Compute log-mean-exp across groups defined by ``subset_idxs``."""

    den = scatter_reduce(
        src=th.ones_like(logits),
        index=subset_idxs,
        reduce="sum",
        dim_size=dim_size,
        default=0.0,
    )
    log_num = scatter_logsumexp(logits, subset_idxs, eps=eps, dim_size=dim_size)
    return log_num - safelog(den)


def scatter_logsoftmax(logits: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Compute a log-softmax over groups defined by ``subset_idxs``."""

    c = scatter_logsumexp(logits, subset_idxs)
    logits = logits - c[subset_idxs]
    return logits


def scatter_l1normalize(vals: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Apply L1 normalization within each group of ``subset_idxs``."""

    c = scatter_reduce(src=vals, index=subset_idxs, reduce="sum", dim_size=th.max(subset_idxs) + 1)
    c = th.clamp(c, min=EPS)
    vals = vals / c[subset_idxs]
    return vals


def scatter_l2normalize(vals: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Apply L2 normalization within each group of ``subset_idxs``."""

    c = scatter_reduce(
        src=vals**2, index=subset_idxs, reduce="sum", dim_size=th.max(subset_idxs) + 1
    )
    c = th.clamp(th.sqrt(c), min=EPS)
    vals = vals / c[subset_idxs]
    return vals


def scatter_logl2normalize(logits: th.Tensor, subset_idxs: th.Tensor) -> th.Tensor:
    """Apply log-space L2 normalization grouped by ``subset_idxs``."""

    c = scatter_logsumexp(2 * logits, subset_idxs)
    logits = logits - 0.5 * c[subset_idxs]
    return logits


def scatter_argmax(
    src: th.Tensor,
    index: th.Tensor,
    other_index: th.Tensor,
    dim_size: int | None = None,
    return_max: bool = False,
) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
    """Argmax grouped by ``index`` returning positions from ``other_index``."""

    if dim_size is None:
        dim_size = th.max(index) + 1
    else:
        assert dim_size >= th.max(index) + 1
    mx = scatter_reduce(src=src, index=index, reduce="amax", dim_size=dim_size, include_self=False)
    ma = src == mx[index]
    ma_idx = other_index * ma + (-1) * (~ma)
    amx = scatter_reduce(
        src=ma_idx,
        index=index,
        reduce="amax",
        dim_size=dim_size,
        include_self=True,
        default=-1,
    )
    if return_max:
        return amx, mx
    return amx


def scatter_argtopk(
    src: th.Tensor,
    index: th.Tensor,
    other_index: th.Tensor,
    k: int,
    dim_size: int | None = None,
    return_max: bool = False,
) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
    """Return top-k indices per group defined by ``index``."""

    assert k > 0
    assert th.is_floating_point(src)
    src = src.detach().clone()
    counts = scatter_reduce(
        src=th.ones_like(src, dtype=th.long),
        index=index,
        reduce="sum",
        dim_size=th.max(index) + 1,
    )
    amxs, mxs = [], []
    for i in range(k):
        amx, mx = scatter_argmax(src, index, other_index, dim_size=dim_size, return_max=True)
        mask = (th.arange(amx.shape[0], device=index.device)[index] == index) & (
            amx[index] == other_index
        )
        src[mask] = -float("inf")
        amx[counts <= i] = -1
        amxs.append(amx)
        mxs.append(mx)
    amxs = th.stack(amxs, dim=1)
    mxs = th.stack(mxs, dim=1)
    if return_max:
        return amxs, mxs
    return amxs


def scatter_reduce(
    src: th.Tensor,
    index: th.Tensor,
    reduce: str,
    dim: int = 0,
    dim_size: int | None = None,
    default: float = 0.0,
    include_self: bool = True,
) -> th.Tensor:
    """Thin wrapper over torch.scatter_reduce_ with validation.

    Pass `dim_size` whenever it is already known on the host (e.g. the batch size):
    leaving it `None` forces a blocking `int(th.max(index))` host sync to derive it.
    With `dim_size` given, bounds are checked with a non-blocking `torch._assert_async`
    on CUDA tensors instead, so no sync is introduced.
    """

    if reduce == "mean" and include_self:
        print("scatter_reduce: mean reduce with include_self=True is not recommended")
    index = index.to(th.int64)
    if index.numel() == 0:
        if dim_size is None:
            dim_size = 0
        result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim + 1 :]
        return th.full(result_shape, default, dtype=src.dtype, device=src.device)
    if dim_size is None:
        max_index = int(th.max(index))
        if max_index < 0:
            raise ValueError("scatter_reduce: index contains negative entries")
        dim_size = max_index + 1
    elif index.device.type == "cuda":
        th._assert_async(
            ((index >= 0) & (index < dim_size)).all(),
            "scatter_reduce: index contains a negative entry or exceeds dim_size",
        )
    else:
        max_index = int(th.max(index))
        if max_index < 0:
            raise ValueError("scatter_reduce: index contains negative entries")
        if dim_size <= max_index:
            raise ValueError(
                f"scatter_reduce: dim_size {dim_size} is too small for max index {max_index}"
            )
    result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim + 1 :]
    results = th.full(result_shape, default, dtype=src.dtype, device=src.device)
    results.scatter_reduce_(dim=dim, index=index, src=src, reduce=reduce, include_self=include_self)
    return results


def safelog(x: th.Tensor, eps: float = EPS) -> th.Tensor:
    """Safe log with epsilon clamp."""

    return th.log(th.clamp(x, min=eps))


def x_log_x(values: th.Tensor) -> th.Tensor:
    """Return ``x log x`` with the ``x = 0`` limit, so padded entries contribute nothing."""
    return values * values.clamp_min(th.finfo(values.dtype).tiny).log()
