"""Graph normalization and the residual-branch activation shared by the GNN layers."""

from collections.abc import Callable

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg

_RESIDUAL_ACTIVATIONS = ("relu", "identity", "gelu")


def validate_residual_activation(activation: str) -> str:
    """Validate the activation applied to a GNN residual branch before the skip add."""
    if activation not in _RESIDUAL_ACTIVATIONS:
        raise ValueError(
            f"residual_activation must be one of {_RESIDUAL_ACTIVATIONS}, got {activation!r}"
        )
    return activation


def residual_branch(activation: str, value: th.Tensor) -> th.Tensor:
    """Apply the configured branch activation.

    ``relu`` reproduces the historical behaviour, in which a layer can only ADD
    per-coordinate mass to the residual stream and can never attenuate it. ``identity``
    removes that restriction; ``gelu`` keeps a nonlinearity while allowing negative
    contributions.
    """
    if activation == "relu":
        return F.relu(value)
    if activation == "gelu":
        return F.gelu(value)
    return value


def get_norm(normalization: str, hidden_size: int) -> tuple[nn.Module, Callable]:
    # norm_fn may or may not depend on the batch index
    if normalization == "batch":
        norm_mod = nn.BatchNorm1d(hidden_size)

        def norm_fn(x: th.Tensor, b: th.Tensor, n: nn.Module) -> th.Tensor:
            return n(x)

    elif normalization == "layer":
        norm_mod = nn.LayerNorm(hidden_size)

        def norm_fn(x: th.Tensor, b: th.Tensor, n: nn.Module) -> th.Tensor:
            return n(x)

    elif normalization == "graph":
        norm_mod = pyg.nn.GraphNorm(hidden_size)

        def norm_fn(x: th.Tensor, b: th.Tensor, n: nn.Module) -> th.Tensor:
            return n(x, b)

    else:
        assert normalization == "none", normalization
        norm_mod = nn.Identity()

        def norm_fn(x: th.Tensor, b: th.Tensor, n: nn.Module) -> th.Tensor:
            return n(x)

    return norm_mod, norm_fn
