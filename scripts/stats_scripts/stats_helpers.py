"""Helpers shared by the statistics scripts in this directory (moved from
``fragnnet.inference.runners``); import as a sibling module: ``from stats_helpers import ...``.
"""

import numpy as np
import torch as th


def select_model_vals(
    model_to_vals: dict[str, dict[str, th.Tensor]],
    keys: list[str],
    stack_dim: int = 0,
) -> dict[str, th.Tensor]:
    """Select and stack specific keys from model value dictionary.

    Args:
        model_to_vals: Dictionary mapping model names to value dictionaries
        keys: List of keys to extract from each value dictionary
        stack_dim: Dimension along which to stack values (default: 0)

    Returns:
        Dictionary with selected keys stacked into tensors
    """
    select_d = {k: [] for k in keys}
    for _, vals in model_to_vals.items():
        for k in keys:
            select_d[k].append(vals[k])
    select_d = {k: th.stack(v, dim=stack_dim) for k, v in select_d.items()}
    return select_d


def log_mean(
    log_p: th.Tensor,
    dim: int = 0,
) -> th.Tensor:
    """Compute logarithm of mean in log-probability space.

    Computes: log(mean(exp(log_p))) = logsumexp(log_p) - log(size).

    Args:
        log_p: Log-probability tensor
        dim: Dimension along which to compute mean (default: 0)

    Returns:
        Log of mean along specified dimension
    """
    return th.logsumexp(log_p, dim=dim) - np.log(log_p.shape[dim])


def pearson_r(
    x: th.Tensor,
    y: th.Tensor,
    dim: int = 0,
) -> th.Tensor:
    """Compute Pearson correlation coefficient.

    Args:
        x: First tensor
        y: Second tensor
        dim: Dimension along which to compute correlation (default: 0)

    Returns:
        Pearson correlation coefficient
    """
    x_mean = th.mean(x, dim=dim)
    y_mean = th.mean(y, dim=dim)
    x_std = th.std(x, dim=dim)
    y_std = th.std(y, dim=dim)
    r = th.mean((x - x_mean) * (y - y_mean), dim=dim) / (x_std * y_std)
    return r
