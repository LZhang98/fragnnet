"""Utility helpers shared across FraGNNet training and inference.

This module bundles lightweight helpers for seeding, type conversions,
progress/reporting, scatter reductions, memory checks, SLURM info, and
W&B checkpoint management. Functions are designed to be side-effect light
and torch/pandas friendly for use in data loading, model code, and scripts.
"""

import glob
import logging
import multiprocessing
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
)
from contextlib import contextmanager
from distutils.util import strtobool
from functools import wraps
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import torch as th
import torch_geometric as pyg

T = TypeVar("T")
R = TypeVar("R")

PPM = 1 / 1000000
EPS = 1e-7
LOG_TWO = float(np.log(2.0))
LOG_ZERO_FP32 = float(th.finfo(th.float32).min)
LOG_ZERO_FP16 = float(th.finfo(th.float16).min)
LOG_ZERO_BP16 = float(th.finfo(th.bfloat16).min)
TOLERANCE_MIN_MZ = 200.0


def LOG_ZERO(dtype: th.dtype) -> float:
    """Return a tiny log value for the given torch dtype."""

    if dtype == th.float32:
        return LOG_ZERO_FP32
    if dtype == th.float16:
        return LOG_ZERO_FP16
    if dtype == th.bfloat16:
        return LOG_ZERO_BP16
    raise ValueError(dtype)


def timeit(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to print execution time for the wrapped function."""

    # adapted from https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    @wraps(func)
    def timeit_wrapper(*args: Any, **kwargs: Any) -> R:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def booltype(x: str) -> bool:
    """Convert a string to boolean using distutils.strtobool."""

    return bool(strtobool(x))


def none_or_nan(thing: Any) -> bool:
    """Return True when the value is effectively missing.

    Treats common null-like cases as missing: None, empty string, scalar NaN,
    empty containers, pandas objects that are all null, and NumPy arrays that
    are empty or all-NaN. Falls back to ``pd.isnull`` for other scalar-ish
    inputs.
    """

    # Check for None first
    if thing is None:
        return True

    # Check for empty string
    if isinstance(thing, str) and thing == "":
        return True

    # Check for NaN in floats
    if isinstance(thing, float) and np.isnan(thing):
        return True

    # Handle different data types safely
    try:
        # For scalar values, use pd.isnull which handles NaN properly
        if np.isscalar(thing):
            return bool(pd.isnull(thing))

        # For empty containers (lists, tuples, arrays)
        if hasattr(thing, "__len__") and len(thing) == 0:
            return True

        # For pandas Series/DataFrame
        if isinstance(thing, (pd.Series, pd.DataFrame)):
            return (
                thing.isnull().all().all()
                if isinstance(thing, pd.DataFrame)
                else bool(thing.isnull().all())
            )

        # For numpy arrays
        if isinstance(thing, np.ndarray):
            return thing.size == 0 or (thing.dtype.kind in "fc" and bool(np.isnan(thing).all()))

        # For other iterables, check if they're effectively empty
        if hasattr(thing, "__iter__") and not isinstance(thing, str):
            try:
                return len(thing) == 0
            except TypeError:
                # Some iterables don't support len()
                return False

        # Try pd.isnull as fallback for other types
        return bool(pd.isnull(thing))

    except (TypeError, ValueError, AttributeError):
        # If all checks fail, assume it's not null/nan
        return False


@contextmanager
def np_temp_seed(seed: int) -> Generator[None, None, None]:
    """Temporarily set the NumPy random seed within the context."""

    # Deliberately the legacy global RNG: this context manager exists to seed and restore
    # the process-wide state that library code already uses. np.random.Generator is a
    # separate stream and would not affect those callers.
    state = np.random.get_state()  # noqa: NPY002
    np.random.seed(seed)  # noqa: NPY002
    try:
        yield
    finally:
        np.random.set_state(state)  # noqa: NPY002


def flatten_lol(lol: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a list of iterables into a single list."""

    return [item for sublist in lol for item in sublist]


# some utils for function timeout
# adapted from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call


# @contextmanager
# def time_limit(seconds):
#   if seconds is None:
#       yield
#   def signal_handler(signum, frame):
#       raise TimeoutException(f"Timed out! ({seconds} seconds)")
#   signal.signal(signal.SIGALRM, signal_handler)
#   signal.alarm(seconds)
#   try:
#       yield
#   finally:
#       signal.alarm(0)


def get_tensor_memory_usage(tensor: th.Tensor) -> int:
    """Calculate the byte size of a tensor."""

    return tensor.nelement() * tensor.element_size()


def get_tensor_dict_memory_usage(**tensor_dict: Any) -> int:
    """Compute total memory usage for tensor values in a dictionary."""

    total_memory = 0
    for _k, v in tensor_dict.items():
        if isinstance(v, th.Tensor):
            total_memory += get_tensor_memory_usage(v)
    return total_memory


def get_pyg_memory_usage(pyg_graph: pyg.data.Data) -> int:
    """Return the approximate memory usage of a PyG graph."""

    return pyg.profile.get_data_size(pyg_graph)


def get_mol_graph_size(mol_data: Mapping[str, Any], mol_params: Mapping[str, Any]) -> int:
    """Compute memory usage for molecular graph data in bytes."""

    if mol_params["pyg"]:
        mol_pyg = mol_data["mol_pyg"]
        mol_graph_size = get_pyg_memory_usage(mol_pyg)
    else:
        mol_graph_size = 0
    return mol_graph_size


def scatter_masked_softmax(
    logits: th.Tensor,
    mask: th.Tensor,
    subset_idxs: th.Tensor,
    mask_logprob: float | None = None,
    log: bool = True,
) -> th.Tensor:
    """Apply log-softmax over grouped logits with masking support."""

    if mask_logprob is None:
        mask_logprob = LOG_ZERO(logits.dtype)
    # calculate appropriate mask value
    with th.no_grad():
        c = scatter_masked_logsumexp(logits, mask, subset_idxs)
        lm = th.gather(input=c, index=subset_idxs, dim=0)
        mask_value = mask_logprob + lm
    # apply mask
    masked_logits = mask * logits + (1 - mask) * mask_value
    # normalize
    masked_logits = scatter_logsoftmax(masked_logits, subset_idxs)
    if not log:
        # exponentiate
        return th.exp(masked_logits)
    return masked_logits


def scatter_masked_logsumexp(
    logits: th.Tensor, mask: th.Tensor, subset_idxs: th.Tensor, mask_value: float | None = None
) -> th.Tensor:
    """Compute logsumexp over groups after applying a mask."""

    if mask_value is None:
        mask_value = LOG_ZERO(logits.dtype)
    masked_logits = mask * logits + (1 - mask) * mask_value
    masked_logsumexp = scatter_logsumexp(masked_logits, subset_idxs)
    return masked_logsumexp


def scatter_logsumexp(
    logits: th.Tensor, subset_idxs: th.Tensor, eps: float = EPS, dim_size: int | None = None
) -> th.Tensor:
    """Numerically stable logsumexp grouped by ``subset_idxs``."""

    # numel() is metadata — no GPU sync. Return LOG_ZERO for every group when input is empty.
    if logits.numel() == 0:
        k = dim_size if dim_size is not None else 0
        return th.full((k,), LOG_ZERO(logits.dtype), device=logits.device, dtype=logits.dtype)

    if dim_size is None:
        k = th.max(subset_idxs) + 1
    else:
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


def to_cpu(
    data_d: dict[str, Any], non_blocking: bool = True, detach: bool = False
) -> dict[str, Any]:
    """Move tensor-like entries in a dict to CPU."""

    for k in data_d:
        if isinstance(data_d[k], th.Tensor):
            data = data_d[k]
            if detach:
                data = data.detach()
            data = data.to("cpu", non_blocking=non_blocking)
            data_d[k] = data
    return data_d


def to_device(
    data_d: dict[str, Any], device: str | th.device, non_blocking: bool = True
) -> dict[str, Any]:
    """Move tensor or graph entries in a dict to the specified device."""

    for k in data_d:
        v = data_d[k]
        if isinstance(v, (th.Tensor, pyg.data.Data)):
            v = v.to(device, non_blocking=non_blocking)
            data_d[k] = v
    return data_d


def deep_update(
    mapping: MutableMapping[str, Any], *updating_mappings: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merge dictionaries, giving precedence to later mappings."""

    updated_mapping: MutableMapping[str, Any] = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def print_shapes(input_dict: Mapping[str, Any]) -> None:
    """Print shapes/types for entries in ``input_dict`` (debug helper)."""

    for k, v in input_dict.items():
        if isinstance(v, (th.Tensor, np.ndarray)):
            print(k, "/", tuple(v.shape), "/", type(v))
        elif isinstance(v, (list, tuple)):
            print(k, "/", len(v), "/", type(v))
        elif isinstance(v, pyg.data.Data):
            print(k, "/", (v.num_nodes, v.num_edges), "/", type(v))
        else:
            print(k, "/", None, "/", type(v))


def _validate_tensor_condition(condition: th.Tensor, message: str) -> None:
    """Validate tensor data without synchronizing CUDA after every check."""
    if th.compiler.is_compiling() or condition.device.type == "cuda":
        th._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def th_setdiff1d(t1: th.Tensor, t2: th.Tensor) -> th.Tensor:
    """Set difference of two 1D torch tensors."""

    t1 = th.unique(t1)
    t2 = th.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]


def get_package_version(package: Any) -> tuple[int, int, int]:
    """Return a (major, minor, patch) tuple for a package."""

    version = package.__version__.split("+")[0]
    major, minor, patch = version.split(".")
    return (int(major), int(minor), int(patch))


def check_pyg_compile() -> bool:
    """Verify PyTorch Geometric is compiled against torch >=2.1."""

    th_major_version, th_minor_version = get_package_version(th)[:2]
    pyg_major_version, pyg_minor_version = get_package_version(pyg)[:2]
    assert th_major_version >= 2, th_major_version
    assert pyg_major_version >= 2, pyg_major_version
    return th_minor_version >= 1 and pyg_minor_version >= 4


def check_pyg_full_compile() -> bool:
    """Check for full PyG compatibility (torch>=2.x, pyg>=2.5)."""

    th_major_version, th_minor_version = get_package_version(th)[:2]
    pyg_major_version, pyg_minor_version = get_package_version(pyg)[:2]
    assert th_major_version >= 2, th_major_version
    return pyg_major_version >= 2 and pyg_minor_version >= 5


def get_best_ckpt_from_wandb(
    saved_dp: str,
    run_id: str,
    entity: str = "frag-gnn",
    project: str = "frag-gnn",
    use_cached: bool = False,
) -> str:
    """Fetch the best (latest) checkpoint from a W&B run and cache it."""

    cached_ckpts = glob.glob(os.path.join(saved_dp, f"*_{run_id}.ckpt"))
    if use_cached and len(cached_ckpts) == 1:
        print(f"> found cached ckpt {cached_ckpts[0]} for {run_id}")
        return cached_ckpts[0]

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"> Processing model files for run {run.id} {run.name}")
    model_tag = f"{run.name}_{run_id}"
    os.makedirs(saved_dp, exist_ok=True)
    ckpt_file, ckpt_epoch_num = None, None
    for file in run.files():
        if file.name.startswith("ckpt/model-epoch="):
            epoch_num = int(file.name.removeprefix("ckpt/model-epoch=").removesuffix(".ckpt"))
            if ckpt_file is None or epoch_num > ckpt_epoch_num:
                ckpt_epoch_num = epoch_num
                ckpt_file = file.name

    if ckpt_file is None or ckpt_epoch_num is None:
        print("> Skip, ckpt_file is None or ckpt_epoch_num is None")

    ckpt_fp = f"{saved_dp}/{model_tag}.ckpt"
    if os.path.isfile(ckpt_fp) and use_cached:
        return ckpt_fp

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"> Downloading ckpt {ckpt_file}")
        run.file(ckpt_file).download(root=tmp_dir, replace=False)
        print(f"> Save ckpt {ckpt_fp}")
        shutil.copy(f"{tmp_dir}/{ckpt_file}", ckpt_fp)
    return ckpt_fp


class NestedDefaultDict(defaultdict):
    """Nested defaultdict that auto-creates deeper levels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self) -> str:
        return repr(dict(self))


def kl_div(x_logprobs: th.Tensor, y_logprobs: th.Tensor) -> th.Tensor:
    """Compute Kullback-Leibler divergence between two log-prob vectors."""

    return th.sum(th.exp(x_logprobs) * (x_logprobs - y_logprobs), dim=0)


def js_div(
    x_ids: th.Tensor, x_logprobs: th.Tensor, y_ids: th.Tensor, y_logprobs: th.Tensor
) -> th.Tensor:
    """Compute Jensen-Shannon divergence for sparse log-prob vectors."""

    assert th.unique(x_ids).shape[0] == x_ids.shape[0]
    assert th.unique(y_ids).shape[0] == y_ids.shape[0]

    z_ids, z_inv_idxs = th.unique(th.cat([x_ids, y_ids], dim=0), return_inverse=True)

    x_logprobs_p = th.full_like(
        z_ids, fill_value=LOG_ZERO(x_logprobs.dtype), dtype=x_logprobs.dtype
    )
    y_logprobs_p = th.full_like(
        z_ids, fill_value=LOG_ZERO(y_logprobs.dtype), dtype=y_logprobs.dtype
    )

    x_logprobs_p[z_inv_idxs[: x_ids.shape[0]]] = x_logprobs
    y_logprobs_p[z_inv_idxs[x_ids.shape[0] :]] = y_logprobs

    z_logprobs = th.logsumexp(th.stack([x_logprobs_p, y_logprobs_p], dim=0), dim=0) - LOG_TWO
    jsd = 0.5 * kl_div(x_logprobs_p, z_logprobs) + 0.5 * kl_div(y_logprobs_p, z_logprobs)
    jsd_n = jsd / LOG_TWO

    return jsd_n


def get_slurm_job_id() -> str | None:
    """Return SLURM job id from environment if available."""

    return os.getenv("SLURM_JOB_ID", default=None)


def get_slurm_allocated_cores(job_id: str) -> int | None:
    """Query SLURM for number of CPUs allocated to a job.

    Prefers the ``SLURM_CPUS_PER_TASK`` environment variable (set by SLURM
    for every task, including array tasks) and falls back to ``scontrol``
    when that variable is absent.  The ``scontrol`` path takes the first
    match so it works correctly for array jobs, which emit one line per task.
    """
    cpus_env = os.getenv("SLURM_CPUS_PER_TASK")
    if cpus_env is not None:
        try:
            return int(cpus_env)
        except ValueError:
            pass  # fall back to scontrol

    # fallback: use scontrol and take only the first line
    try:
        command = f"scontrol show job {job_id} | grep -oP 'NumCPUs=\\K\\d+'"
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,  # get str directly
        )
        lines = result.stdout.strip().splitlines()
        if not lines:
            return None
        return int(lines[0])  # first line only; array jobs may have multiple
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error querying SLURM for job {job_id}: {e}")
        return None


def get_core_count() -> int:
    """Return CPU core count, preferring SLURM allocation when present."""

    slurm_id = get_slurm_job_id()
    if slurm_id is not None:
        num_core = get_slurm_allocated_cores(slurm_id)
    else:
        num_core = multiprocessing.cpu_count()
    return int(num_core) if num_core is not None else multiprocessing.cpu_count()


def validate_bin_geometry(mz_max: float, mz_bin_res: float) -> int:
    """Validate and return the bin count for a given m/z grid configuration.

    Enforces that mz_max and mz_bin_res define a consistent, integer bin grid:
    ``num_bins = mz_max / mz_bin_res`` must be an integer.

    Args:
        mz_max: Maximum m/z value covered by the grid.
        mz_bin_res: m/z bin resolution.

    Returns:
        Number of bins in the grid.

    Raises:
        ValueError: If ``mz_max`` or ``mz_bin_res`` is non-positive, or if
            ``mz_max / mz_bin_res`` is not an integer (within floating-point tolerance).

    Example:
        >>> validate_bin_geometry(1500.0, 0.01)  # Default: 150,000 bins
        150000
        >>> validate_bin_geometry(200.0, 0.5)    # Non-default: 400 bins
        400
        >>> validate_bin_geometry(200.0, 0.3)    # ValueError: 666.666... not integer
        Traceback: ValueError: Incompatible bin geometry...
    """
    import math

    if mz_max <= 0:
        raise ValueError(f"mz_max must be > 0, got {mz_max}")
    if mz_bin_res <= 0:
        raise ValueError(f"mz_bin_res must be > 0, got {mz_bin_res}")

    num_bins_float = mz_max / mz_bin_res
    num_bins = int(round(num_bins_float))
    if not math.isclose(num_bins_float, num_bins, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Incompatible bin geometry: mz_max / mz_bin_res must be an integer. "
            f"Got mz_max={mz_max}, mz_bin_res={mz_bin_res}, ratio={num_bins_float}."
        )
    return num_bins


def progress_wrapper(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str | None = None,
    disable_tqdm: bool = False,
) -> Iterator[T]:
    """Wrap an iterable with tqdm or fallback logging-based progress."""

    import logging

    if disable_tqdm:
        log_interval = max(1, total // 20) if total else 100
        for idx, item in enumerate(iterable):
            if idx % log_interval == 0:
                progress = f"{idx}/{total}" if total else f"{idx}"
                logging.info(f"{desc}: {progress}")
            yield item
        if total:
            logging.info(f"{desc}: {total}/{total} - Complete")
    else:
        from tqdm import tqdm

        for item in tqdm(iterable, total=total, desc=desc):
            yield item


class ProgressLogger:
    """Interval-based progress logger with rate, ETA, and error tracking.

    A line is emitted every ``log_every`` items OR every ``log_seconds`` seconds,
    whichever comes first, so long-running jobs stay legible in SLURM logs without
    depending on tqdm.

    Example:
        progress = ProgressLogger(total=len(rows), desc="Fragment generation")
        for row in rows:
            ...
            progress.update(is_error=failed)
        progress.finish()
    """

    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        log_every: int = 100,
        log_seconds: float = 30.0,
        logger: "logging.Logger | None" = None,
    ) -> None:
        self.total = total
        self.desc = desc
        self.log_every = max(1, log_every)
        self.log_seconds = log_seconds
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.n_done = 0
        self.n_errors = 0
        self.start_time = time.time()
        self._last_log_time = self.start_time

    @property
    def elapsed(self) -> float:
        """Seconds since the logger was created."""
        return time.time() - self.start_time

    @property
    def rate(self) -> float:
        """Completed items per second, 0.0 before any time has passed."""
        elapsed = self.elapsed
        return self.n_done / elapsed if elapsed > 0 else 0.0

    def update(self, is_error: bool = False, n: int = 1) -> None:
        """Record ``n`` completed items and log if an interval boundary was crossed."""
        self.n_done += n
        if is_error:
            self.n_errors += n

        now = time.time()
        if self.n_done % self.log_every == 0 or (now - self._last_log_time) >= self.log_seconds:
            self._log_progress(now)

    def _log_progress(self, now: float) -> None:
        elapsed = now - self.start_time
        rate = self.n_done / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.n_done)
        eta = remaining / rate if rate > 0 else 0.0
        pct = (self.n_done / self.total * 100) if self.total > 0 else 0.0

        self.logger.info(f"{self.desc}: {self.n_done}/{self.total} ({pct:.1f}%)")
        self.logger.info(f"  Rate: {rate:.2f} items/sec")
        self.logger.info(f"  Elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s")
        if self.n_errors > 0:
            self.logger.warning(f"  Errors encountered: {self.n_errors}")
        self._last_log_time = now

    def finish(self) -> None:
        """Log the completion summary: totals, wall time, average rate, success rate."""
        elapsed = self.elapsed
        avg_rate = self.n_done / elapsed if elapsed > 0 else 0.0

        self.logger.info(f"{self.desc} completed")
        self.logger.info(f"Total items processed: {self.n_done}")
        self.logger.info(f"Total time elapsed: {elapsed:.2f} seconds")
        self.logger.info(f"Average processing rate: {avg_rate:.2f} items/second")
        if self.n_errors > 0:
            self.logger.warning(f"Total errors encountered: {self.n_errors}")
            if self.n_done > 0:
                success = (self.n_done - self.n_errors) / self.n_done * 100
                self.logger.warning(f"Success rate: {success:.1f}%")
        else:
            self.logger.info("No errors encountered - 100% success rate")
