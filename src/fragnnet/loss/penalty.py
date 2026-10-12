"""Scalar penalty functions on edge outputs."""

from collections.abc import Callable

import torch as th


def get_edge_loss_fn(edge_loss_fn_type: str, constant: float) -> Callable:
    if edge_loss_fn_type == "quadratic":

        def quad_loss(x: th.Tensor) -> th.Tensor:
            return constant * x**2

        return quad_loss
    elif edge_loss_fn_type == "linear":

        def linear_loss(x: th.Tensor) -> th.Tensor:
            return constant * th.abs(x)

        return linear_loss
    elif edge_loss_fn_type == "exponential":

        def exp_loss(x: th.Tensor) -> th.Tensor:
            return constant * th.exp(th.abs(x))

        return exp_loss
    else:
        raise NotImplementedError(f"Edge loss function type {edge_loss_fn_type} not implemented")
