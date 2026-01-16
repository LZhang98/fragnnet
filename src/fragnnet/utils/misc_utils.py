import glob
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from distutils.util import strtobool
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import torch as th
import torch_geometric as pyg
import tqdm

from fragnnet.utils.dgl_compat_utils import DGLGraph

TQDM_DISABLE = False
PPM = 1 / 1000000
EPS = 1e-7
LOG_HALF = float(np.log(0.5))
LOG_TWO = float(np.log(2.0))
LOG_ZERO_FP32 = float(th.finfo(th.float32).min)
LOG_ZERO_FP16 = float(th.finfo(th.float16).min)
LOG_ZERO_BP16 = float(th.finfo(th.bfloat16).min)
MAX_CROSS_ENTROPY = 1e19
TOLERANCE_MIN_MZ = 200.0


def LOG_ZERO(dtype):
    if dtype == th.float32:
        return LOG_ZERO_FP32
    elif dtype == th.float16:
        return LOG_ZERO_FP16
    elif dtype == th.bfloat16:
        return LOG_ZERO_BP16
    else:
        raise ValueError(dtype)


def timeit(func):
    # adapted from https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def booltype(x):
    return bool(strtobool(x))


def none_or_nan(thing):
    """
    Check if a value is None, NaN, or empty, handling various data types safely.

    Args:
        thing: Value to check for None, NaN, or emptiness

    Returns:
        bool: True if the value is None, NaN, empty, or invalid, False otherwise
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
            return pd.isnull(thing)

        # For empty containers (lists, tuples, arrays)
        if hasattr(thing, "__len__") and len(thing) == 0:
            return True

        # For pandas Series/DataFrame
        if isinstance(thing, (pd.Series, pd.DataFrame)):
            return (
                thing.isnull().all().all()
                if isinstance(thing, pd.DataFrame)
                else thing.isnull().all()
            )

        # For numpy arrays
        if isinstance(thing, np.ndarray):
            return thing.size == 0 or (
                thing.dtype.kind in "fc" and np.isnan(thing).all()
            )

        # For other iterables, check if they're effectively empty
        if hasattr(thing, "__iter__") and not isinstance(thing, str):
            try:
                return len(thing) == 0
            except TypeError:
                # Some iterables don't support len()
                return False

        # Try pd.isnull as fallback for other types
        return pd.isnull(thing)

    except (TypeError, ValueError, AttributeError):
        # If all checks fail, assume it's not null/nan
        return False


@contextmanager
def np_temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextmanager
def th_temp_seed(seed):
    state = th.get_rng_state()
    th.manual_seed(seed)
    try:
        yield
    finally:
        th.set_rng_state(state)


def flatten_lol(lol):
    return [item for sublist in lol for item in sublist]


def wandb_symlink(run_dir, wandb_symlink_dp, job_id):
    symlink_dst = os.path.join(wandb_symlink_dp, str(job_id))
    symlink_src = os.path.split(os.path.abspath(run_dir))[0]
    if os.path.islink(symlink_dst):
        os.unlink(symlink_dst)
    os.symlink(symlink_src, symlink_dst)


def list_str2float(str_list):
    return [float(str_item) for str_item in str_list]


# https://stackoverflow.com/a/58936697/6937913
@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# some utils for function timeout
# adapted from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call


class TimeoutException(Exception):
    pass


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


def timeout_func(func, args=None, kwargs=None, timeout=30, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout is exceeded.
    http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
    """

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default
            self.exc_info = (None, None, None)

        def run(self):
            try:
                self.result = func(*(args or ()), **(kwargs or {}))
            except Exception:
                self.exc_info = sys.exc_info()

        def suicide(self):
            raise TimeoutException(
                "{0} timeout (taking more than {1} sec)".format(func.__name__, timeout)
            )

    it = InterruptableThread()
    it.start()
    it.join(timeout)
    if it.exc_info[0] is not None:
        a, b, c = it.exc_info
        raise Exception(a, b, c)  # communicate that to caller
    if it.is_alive():
        it.suicide()
        raise RuntimeError
    else:
        return it.result


def my_tqdm(*args, **kwargs):
    return tqdm.tqdm(*args, **kwargs, disable=TQDM_DISABLE)


def get_tensor_memory_usage(tensor):
    return tensor.nelement() * tensor.element_size()


def get_tensor_dict_memory_usage(**tensor_dict):
    total_memory = 0
    for k, v in tensor_dict.items():
        if isinstance(v, th.Tensor):
            total_memory += get_tensor_memory_usage(v)
    return total_memory


def get_pyg_memory_usage(pyg_graph):
    return pyg.profile.get_data_size(pyg_graph)


def get_mol_graph_size(mol_data, mol_params):
    """
    Calculate the memory usage of molecular graph data.

    Args:
        mol_data: Dictionary containing molecular data
        mol_params: Dictionary containing molecular parameters

    Returns:
        Memory usage in bytes
    """
    if mol_params["pyg"]:
        mol_pyg = mol_data["mol_pyg"]
        mol_graph_size = get_pyg_memory_usage(mol_pyg)
    else:
        mol_graph_size = 0
    return mol_graph_size


def scatter_masked_softmax(logits, mask, subset_idxs, mask_logprob=None, log=True):
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
    else:
        return masked_logits


def scatter_masked_logsumexp(logits, mask, subset_idxs, mask_value=None):
    if mask_value is None:
        mask_value = LOG_ZERO(logits.dtype)
    # apply mask
    masked_logits = mask * logits + (1 - mask) * mask_value
    # normalize
    masked_logsumexp = scatter_logsumexp(masked_logits, subset_idxs)
    return masked_logsumexp


def scatter_logsumexp(logits, subset_idxs, eps=EPS, dim_size=None):
    """
    Computes a numerically stable logsumexp over groups defined by subset_idxs.

    Args:
        logits (Tensor): Input tensor of values (e.g., log-probabilities), shape [N].
        subset_idxs (Tensor): 1D tensor of group indices for each element in logits, shape [N].
        eps (float, optional): Small value to avoid log(0). Defaults to EPS.
        dim_size (int, optional): Number of groups. If None, inferred from subset_idxs.

    Returns:
        Tensor: logsumexp for each group, shape [num_groups]. If used as c[subset_idxs], broadcasts to [N].
    """
    # Determine the number of groups (k)
    if dim_size is None:
        k = th.max(subset_idxs) + 1
    else:
        assert dim_size >= th.max(subset_idxs) + 1
        k = dim_size

    # Step 1: For each group, find the maximum logit (for numerical stability)
    sm = scatter_reduce(
        src=logits,
        index=subset_idxs,
        reduce="amax",
        dim_size=k,
        default=LOG_ZERO(logits.dtype),
    )

    # Step 2: Gather the group max for each element
    lm = th.gather(input=sm, index=subset_idxs, dim=0)

    # Step 3: Subtract the group max from each logit (for stability)
    logits = logits - lm

    # Step 4: For each group, sum the exponentiated shifted logits
    se = scatter_reduce(
        src=th.exp(logits), index=subset_idxs, reduce="sum", dim_size=k, default=0.0
    )

    # Step 5: Compute logsumexp for each group: max + log(sum(exp(x - max)))
    return sm + th.log(se + eps)


def scatter_logmeanexp(logits, subset_idxs, eps=EPS, dim_size=None):
    """
    Computes a numerically stable log-mean-exp over groups defined by subset_idxs.

    Args:
        logits (Tensor): Input tensor of values (e.g., log-probabilities), shape [N].
        subset_idxs (Tensor): 1D tensor of group indices for each element in logits, shape [N].
        eps (float, optional): Small value to avoid log(0). Defaults to EPS.
        dim_size (int, optional): Number of groups. If None, inferred from subset_idxs.

    Returns:
        Tensor: log-mean-exp for each group, shape [num_groups].
    """
    # Compute the denominator: number of elements in each group
    den = scatter_reduce(
        src=th.ones_like(logits),
        index=subset_idxs,
        reduce="sum",
        dim_size=dim_size,
        default=0.0,
    )
    # Compute the numerator: logsumexp for each group
    log_num = scatter_logsumexp(logits, subset_idxs, eps=eps, dim_size=dim_size)
    # log-mean-exp = logsumexp - log(denominator)
    return log_num - safelog(den)


def scatter_logsoftmax(logits, subset_idxs):
    """
    Computes a log-softmax over groups defined by subset_idxs.

    For each group (as defined by subset_idxs), this function normalizes the logits
    so that the exponentiated values sum to 1 within each group, but returns the result in log-space.

    Args:
        logits (Tensor): Input tensor of values (e.g., logits or log-probabilities), shape [N].
        subset_idxs (Tensor): 1D tensor of group indices for each element in logits, shape [N].

    Returns:
        Tensor: Log-softmax-normalized values for each element, shape [N].
    """
    # Calculate the logsumexp for each group (normalizing constant)
    c = scatter_logsumexp(logits, subset_idxs)
    # Subtract the group logsumexp from each element's logit to get log-softmax
    logits = logits - c[subset_idxs]
    return logits


def scatter_softmax(logits, subset_idxs):
    return th.exp(scatter_logsoftmax(logits, subset_idxs))


def scatter_l1normalize(vals, subset_idxs):
    # calculate normalizing constant
    c = scatter_reduce(
        src=vals, index=subset_idxs, reduce="sum", dim_size=th.max(subset_idxs) + 1
    )
    c = th.clamp(c, min=EPS)
    # apply normalizing constant
    vals = vals / c[subset_idxs]
    return vals


def scatter_l2normalize(vals, subset_idxs):
    # calculate normalizing constant
    c = scatter_reduce(
        src=vals**2, index=subset_idxs, reduce="sum", dim_size=th.max(subset_idxs) + 1
    )
    c = th.clamp(th.sqrt(c), min=EPS)
    # apply normalizing constant
    vals = vals / c[subset_idxs]
    return vals


def scatter_logl2normalize(logits, subset_idxs):
    # calculate normalizing constant
    c = scatter_logsumexp(2 * logits, subset_idxs)
    # apply normalizing constant
    logits = logits - 0.5 * c[subset_idxs]
    return logits


def scatter_var(src, index, dim_size=None, correction=1, sqrt=False):
    # calculate dim_size
    if dim_size is None:
        dim_size = th.max(index) + 1
    else:
        assert dim_size >= th.max(index) + 1
    # calculate mean
    m = scatter_reduce(
        src=src, index=index, reduce="mean", dim_size=dim_size, include_self=False
    )
    # calculate variance
    v_num = scatter_reduce(
        src=(src - m[index]) ** 2, index=index, reduce="sum", dim_size=dim_size
    )
    v_den = scatter_reduce(
        src=th.ones_like(src), index=index, reduce="sum", dim_size=dim_size
    )
    v = v_num / th.clamp(v_den - correction, min=EPS)
    if sqrt:
        v = th.sqrt(v)
    return v


def scatter_argmax(src, index, other_index, dim_size=None, return_max=False):
    # calculate dim_size
    if dim_size is None:
        dim_size = th.max(index) + 1
    else:
        assert dim_size >= th.max(index) + 1
    # calculate max
    mx = scatter_reduce(
        src=src, index=index, reduce="amax", dim_size=dim_size, include_self=False
    )
    # calculate mask
    ma = src == mx[index]
    # calculate argmax
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
    else:
        return amx


def scatter_argtopk(src, index, other_index, k, dim_size=None, return_max=False):
    assert k > 0
    assert th.is_floating_point(src)
    src = src.detach().clone()
    counts = scatter_reduce(
        src=th.ones_like(src, dtype=th.long),
        index=index,
        reduce="sum",
        dim_size=th.max(index) + 1,
    )
    # print(counts)
    amxs, mxs = [], []
    for i in range(k):
        # print(src, index)
        amx, mx = scatter_argmax(
            src, index, other_index, dim_size=dim_size, return_max=True
        )
        mask = (th.arange(amx.shape[0], device=index.device)[index] == index) & (
            amx[index] == other_index
        )
        src[mask] = -float("inf")  # LOG_ZERO(src.dtype)
        # don't double count
        # amx[th.isinf(mx)] = -1
        amx[counts <= i] = -1
        amxs.append(amx)
        mxs.append(mx)
    amxs = th.stack(amxs, dim=1)
    mxs = th.stack(mxs, dim=1)
    if return_max:
        return amxs, mxs
    else:
        return amxs


def scatter_reduce(
    src, index, reduce, dim=0, dim_size=None, default=0.0, include_self=True
):
    if reduce == "mean" and include_self:
        print("scatter_reduce: mean reduce with include_self=True is not recommended")
    index = index.to(th.int64)
    if index.numel() == 0:
        if dim_size is None:
            dim_size = 0
        result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim + 1 :]
        return th.full(result_shape, default, dtype=src.dtype, device=src.device)
    max_index = int(th.max(index))
    if max_index < 0:
        raise ValueError("scatter_reduce: index contains negative entries")
    if dim_size is None:
        dim_size = max_index + 1
    elif dim_size <= max_index:
        raise ValueError(
            f"scatter_reduce: dim_size {dim_size} is too small for max index {max_index}"
        )
    result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim + 1 :]
    results = th.full(result_shape, default, dtype=src.dtype, device=src.device)
    results.scatter_reduce_(
        dim=dim, index=index, src=src, reduce=reduce, include_self=include_self
    )
    return results


def safelog(x, eps=EPS):
    return th.log(th.clamp(x, min=eps))


def batchwise_max(xs, batch_idxs):
    """for debugging"""

    batch_size = th.max(batch_idxs) + 1
    maxs = th.zeros([batch_size], device=xs.device, dtype=xs.dtype)
    for b in range(batch_size):
        maxs[b] = th.max(xs[batch_idxs == b])
    return maxs


def batchwise_lse(xs, batch_idxs):
    """for debugging"""

    batch_size = th.max(batch_idxs) + 1
    lses = th.zeros([batch_size], device=xs.device, dtype=xs.dtype)
    for b in range(batch_size):
        lses[b] = th.logsumexp(xs[batch_idxs == b], 0)
    return lses


def dedup_peaks(mzs, logprobs, batch_idxs):
    b_mzs = th.stack([batch_idxs.type(mzs.dtype), mzs], dim=1)
    dd_b_mzs, dd_logprobs, dd_batch_idxs = dedup(
        b_mzs, *[("lse", logprobs), ("amax", batch_idxs)]
    )
    dd_b_mzs = dd_b_mzs[:, 1]
    return dd_b_mzs, dd_logprobs, dd_batch_idxs


def dedup(keys, *agg_vals_tups, dim=0):
    un_keys, inv_keys = th.unique(keys, dim=dim, return_inverse=True)
    res = [un_keys]
    for agg_vals_tup in agg_vals_tups:
        agg, vals = agg_vals_tup
        assert agg in ["lse", "sum", "min", "mean", "amax"]
        assert vals.shape[0] == keys.shape[0]
        if agg == "lse":
            un_vals = scatter_logsumexp(vals, inv_keys, dim_size=un_keys.shape[0])
        else:
            un_vals = scatter_reduce(
                vals,
                inv_keys,
                reduce=agg,
                dim_size=un_keys.shape[0],
                include_self=False,
            )
        res.append(un_vals)
    res = tuple(res)
    return res


def to_cpu(data_d, non_blocking=True, detach=False):
    for k in data_d.keys():
        if isinstance(data_d[k], th.Tensor):
            data = data_d[k]
            if detach:
                data = data.detach()
            data = data.to("cpu", non_blocking=non_blocking)
            data_d[k] = data
    return data_d


def to_device(data_d, device, non_blocking=True):
    for k in data_d.keys():
        v = data_d[k]
        # Support optional DGL: use DGLGraph placeholder for isinstance checks
        if (
            isinstance(v, th.Tensor)
            or isinstance(v, DGLGraph)
            or isinstance(v, pyg.data.Data)
        ):
            v = v.to(device, non_blocking=non_blocking)
            data_d[k] = v
    return data_d


def deep_update(mapping, *updating_mappings):
    """
    adapted from pydantic
    https://github.com/pydantic/pydantic/blob/fd2991fe6a73819b48c906e3c3274e8e47d0f761/pydantic/utils.py#L200-L208
    """
    updated_mapping = mapping.copy()
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


def print_shapes(input_dict):
    for k, v in input_dict.items():
        if isinstance(v, th.Tensor) or isinstance(v, np.ndarray):
            print(k, "/", tuple(v.shape), "/", type(v))
        elif isinstance(v, list) or isinstance(v, tuple):
            print(k, "/", len(v), "/", type(v))
        elif isinstance(v, pyg.data.Data):
            print(k, "/", (v.num_nodes, v.num_edges), "/", type(v))
        elif isinstance(v, DGLGraph):
            print(k, "/", (v.number_of_nodes(), v.number_of_edges()), "/", type(v))
        else:
            print(k, "/", None, "/", type(v))


def th_setdiff1d(t1, t2):
    t1 = th.unique(t1)
    t2 = th.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]


def get_package_version(package):
    version = package.__version__.split("+")[0]
    major, minor, patch = version.split(".")
    return (int(major), int(minor), int(patch))


def check_pyg_compile():
    th_major_version, th_minor_version = get_package_version(th)[:2]
    pyg_major_version, pyg_minor_version = get_package_version(pyg)[:2]
    assert th_major_version >= 2, th_major_version
    assert pyg_major_version >= 2, pyg_major_version
    return th_minor_version >= 1 and pyg_minor_version >= 4


def check_pyg_full_compile():
    th_major_version, th_minor_version = get_package_version(th)[:2]
    pyg_major_version, pyg_minor_version = get_package_version(pyg)[:2]
    assert th_major_version >= 2, th_major_version
    return pyg_major_version >= 2 and pyg_minor_version >= 5


# wandb stuff
def check_import_wandb():
    try:
        import wandb
    except ImportError:
        print(
            "wandb is not installed. Please install it using 'pip install wandb' if needed."
        )
    return False


def get_best_ckpt_from_wandb(
    saved_dp, run_id, entity="frag-gnn", project="frag-gnn", use_cached=False
):
    # return ckpt
    cached_ckpts = glob.glob(os.path.join(saved_dp, f"*_{run_id}.ckpt"))
    if use_cached and len(cached_ckpts) == 1:
        print(f"> found cached ckpt {cached_ckpts[0]} for {run_id}")
        return cached_ckpts[0]

    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    # print(run.id, run.name, run.tags)
    print(f"> Processing model files for run {run.id} {run.name}")
    model_tag = f"{run.name}_{run_id}"
    # model_save_dp = f"{saved_dp}/{model_tag}"
    os.makedirs(saved_dp, exist_ok=True)
    ckpt_file, ckpt_epoch_num = None, None
    for file in run.files():
        if file.name.startswith("ckpt/model-epoch="):
            epoch_num = int(
                file.name.removeprefix("ckpt/model-epoch=").removesuffix(".ckpt")
            )
            # print(epoch_num)
            if ckpt_file is None or epoch_num > ckpt_epoch_num:
                ckpt_epoch_num = epoch_num
                ckpt_file = file.name

    if ckpt_file is None or ckpt_epoch_num is None:
        print("> Skip, ckpt_file is None or ckpt_epoch_num is None")

    # assert not ckpt_file is None
    # assert not ckpt_epoch_num is None
    ckpt_fp = f"{saved_dp}/{model_tag}.ckpt"
    if os.path.isfile(ckpt_fp) and use_cached:
        return ckpt_fp

    with tempfile.TemporaryDirectory() as tmp_dir:
        # if not os.path.exists(ckpt_fp):
        print(f"> Downloading ckpt {ckpt_file}")
        run.file(ckpt_file).download(root=tmp_dir, replace=False)
        print(f"> Save ckpt {ckpt_fp}")
        shutil.copy(f"{tmp_dir}/{ckpt_file}", ckpt_fp)
    return ckpt_fp


def get_wandb_runs_by_grp(group_name, entity="frag-gnn", project="frag-gnn"):
    import wandb

    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}", include_sweeps=False, filters={"group": group_name}
    )
    return runs


def get_wandb_runids_by_grp(group_name, entity="frag-gnn", project="frag-gnn"):
    runs = get_wandb_runs_by_grp("nist20v3_inchikey_fraggnn_d3")
    run_ids = [run.id for run in runs]
    return run_ids


def delete_ckpt_from_wandb(run_id, entity="frag-gnn", project="frag-gnn"):
    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    for file in run.files():
        if file.name.startswith("ckpt/model-epoch="):
            file.delete()


class NestedDefaultDict(defaultdict):
    """https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict"""

    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


def kl_div(x_logprobs, y_logprobs) -> th.Tensor:
    """method to compute Kullback-Leibler divergence
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    Args:
        x_logprobs (_type_): _description_
        y_logprobs (_type_): _description_

    Returns:
        th.Tensor: Kullback-Leibler divergence
    """

    return th.sum(th.exp(x_logprobs) * (x_logprobs - y_logprobs), dim=0)


def js_div(x_ids, x_logprobs, y_ids, y_logprobs) -> th.Tensor:
    """method to compute Jensen-Shannon Divergence

    Args:
        x_ids (_type_): _description_
        x_logprobs (_type_): _description_
        y_ids (_type_): _description_
        y_logprobs (_type_): _description_

    Returns:
        th.Tensor: Jensen-Shannon divergence
    """

    # Ensure that each element in x_ids and y_ids is unique (i.e., no duplicates)
    assert th.unique(x_ids).shape[0] == x_ids.shape[0]
    assert th.unique(y_ids).shape[0] == y_ids.shape[0]

    # Combine the ids from x and y, creating a union of the two sets of ids
    z_ids, z_inv_idxs = th.unique(th.cat([x_ids, y_ids], dim=0), return_inverse=True)

    # Initialize probability vectors for x and y, filled with a small value to avoid -inf (log(0))
    # LOG_ZERO is assumed to return the log of a very small value (close to 0)
    x_logprobs_p = th.full_like(
        z_ids, fill_value=LOG_ZERO(x_logprobs.dtype), dtype=x_logprobs.dtype
    )
    y_logprobs_p = th.full_like(
        z_ids, fill_value=LOG_ZERO(y_logprobs.dtype), dtype=y_logprobs.dtype
    )

    # Fill the initialized vectors with the actual log-probabilities from x and y, respectively
    x_logprobs_p[z_inv_idxs[: x_ids.shape[0]]] = x_logprobs
    y_logprobs_p[z_inv_idxs[x_ids.shape[0] :]] = y_logprobs

    # Compute the combined log-probabilities (z_logprobs) using logsumexp, which handles the sum of logs
    z_logprobs = (
        th.logsumexp(th.stack([x_logprobs_p, y_logprobs_p], dim=0), dim=0) - LOG_TWO
    )

    # Compute the Jensen-Shannon Divergence (JSD) as the average of two Kullback-Leibler (KL) divergences
    jsd = 0.5 * kl_div(x_logprobs_p, z_logprobs) + 0.5 * kl_div(
        y_logprobs_p, z_logprobs
    )

    # Normalize the JSD by dividing by LOG_TWO to ensure it is in a standard scale (between 0 and 1)
    jsd_n = jsd / LOG_TWO

    return jsd_n


def get_slurm_job_id():
    return os.getenv("SLURM_JOB_ID", default=None)


def get_slurm_allocated_cores(job_id):
    # cpus_on_node = os.getenv('SLURM_CPUS_ON_NODE', default=None)
    # ComputeCanada does not set this on some systems
    # use scontrol to get the number of CPUs allocated to the job
    # int(os.getenv('SLURM_CPUS_PER_TASK', default=1))
    try:
        # Construct the command to get allocated CPUs
        command = f"scontrol show job {job_id} | grep -oP 'NumCPUs=\\K\\d+'"

        # Run the command using subprocess
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Decode and strip the output to get the number of CPUs
        cpus_allocated = result.stdout.decode("utf-8").strip()
        return int(cpus_allocated)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")


def get_core_count():
    """
        get cpu core count, in case slurm env get one from scontrol
    Returns:
        _type_: _description_
    """
    slurm_id = get_slurm_job_id()
    if slurm_id is not None:
        num_core = get_slurm_allocated_cores(slurm_id)
    else:
        num_core = multiprocessing.cpu_count()
    return num_core


def progress_wrapper(iterable, total=None, desc=None, disable_tqdm=False):
    """Wrapper that uses tqdm or logging based on disable_tqdm flag

    Args:
        iterable: The iterable to wrap
        total: Total number of items (optional)
        desc: Description for progress display
        disable_tqdm: If True, use logging instead of tqdm progress bars

    Yields:
        Items from the iterable
    """
    import logging

    if disable_tqdm:
        # Use logging instead of tqdm
        log_interval = max(1, total // 20) if total else 100  # Log ~20 times
        for idx, item in enumerate(iterable):
            if idx % log_interval == 0:
                progress = f"{idx}/{total}" if total else f"{idx}"
                logging.info(f"{desc}: {progress}")
            yield item
        if total:
            logging.info(f"{desc}: {total}/{total} - Complete")
    else:
        # Use tqdm as normal
        from tqdm import tqdm

        for item in tqdm(iterable, total=total, desc=desc):
            yield item
