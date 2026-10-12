"""Entropy of a predicted sparse spectrum, and its conditional variant."""

import torch as th

from fragnnet.utils.misc_utils import EPS, safelog, scatter_logsumexp, scatter_reduce


def sparse_entropy_fn(
    logprobs: th.Tensor,
    batch_idxs: th.Tensor,
    oos_logprobs: th.Tensor | None = None,
    renormalize: bool = False,
    support_size_delta: float = 0.0,
) -> tuple[th.Tensor, th.Tensor]:
    if renormalize:
        logpartition = scatter_logsumexp(logprobs, batch_idxs)
        if oos_logprobs is not None:
            logpartition = th.logsumexp(th.stack([logpartition, oos_logprobs], dim=1), dim=1)
            oos_logprobs = oos_logprobs - logpartition
        logprobs = logprobs - th.gather(input=logpartition, index=batch_idxs, dim=0)
    logprobs = th.clamp(logprobs, max=0.0)
    probs = th.exp(logprobs)
    plogp = probs * logprobs
    batch_size = int(th.max(batch_idxs).item() + 1)
    entropy = -scatter_reduce(
        src=plogp,
        index=batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    if th.min(entropy) < -0.001:
        raise ValueError(f"Negative entropy: {th.min(entropy)}")

    support_size = scatter_reduce(
        src=th.ones_like(batch_idxs, dtype=logprobs.dtype),
        index=batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    support_size += support_size_delta
    if oos_logprobs is not None:
        assert oos_logprobs.shape[0] == batch_size, (oos_logprobs.shape, batch_size)
        entropy = entropy - th.exp(oos_logprobs) * oos_logprobs
        support_size = support_size + 1.0
    max_entropy = safelog(support_size)
    # rescale entropy
    max_entropy = th.clamp(max_entropy, min=EPS)
    entropy = th.clamp(entropy, min=th.zeros_like(max_entropy), max=max_entropy)
    norm_entropy = entropy / max_entropy
    if th.any(norm_entropy < 0.0) or th.any(norm_entropy > 1.0):
        raise ValueError("Normalized entropy out of bounds [0, 1]")
    return entropy, norm_entropy


def sparse_conditional_entropy_fn(
    marginal_logprobs: th.Tensor,
    marginal_batch_idxs: th.Tensor,
    conditional_logprobs: th.Tensor,
    conditional_idxs: th.Tensor,
    conditional_batch_idxs: th.Tensor,
    conditional_support_size_delta: float = 0.0,
) -> tuple[th.Tensor, th.Tensor]:
    batch_size = int(th.max(conditional_batch_idxs).item() + 1)
    marginal_support_size = marginal_logprobs.shape[0]
    marginal_logprobs = th.clamp(marginal_logprobs, max=0.0)
    conditional_logprobs = th.clamp(conditional_logprobs, max=0.0)
    marginal_probs = th.exp(marginal_logprobs)
    conditional_plogp = th.exp(conditional_logprobs) * conditional_logprobs
    conditional_entropy = -scatter_reduce(
        conditional_plogp, index=conditional_idxs, reduce="sum", dim_size=marginal_support_size
    )
    entropy = scatter_reduce(
        marginal_probs * conditional_entropy,
        index=marginal_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    if th.min(entropy) < -0.001:
        raise ValueError(f"Negative conditional entropy: {th.min(entropy)}")

    conditional_support_size = scatter_reduce(
        th.ones_like(conditional_logprobs, dtype=marginal_logprobs.dtype),
        index=conditional_idxs,
        reduce="sum",
        dim_size=marginal_support_size,
    )
    conditional_support_size += conditional_support_size_delta
    max_entropy = scatter_reduce(
        marginal_probs * th.clamp(safelog(conditional_support_size), min=0.0),
        index=marginal_batch_idxs,
        reduce="sum",
        dim_size=batch_size,
    )
    # rescale entropy
    max_entropy = th.clamp(max_entropy, min=EPS)
    entropy = th.clamp(entropy, min=th.zeros_like(max_entropy), max=max_entropy)
    norm_entropy = entropy / max_entropy
    if th.any(norm_entropy < 0.0) or th.any(norm_entropy > 1.0):
        raise ValueError("Normalized conditional entropy out of bounds [0, 1]")
    return entropy, norm_entropy
