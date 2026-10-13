"""Graph pooling: the PyG aggregations selected by ``*_pool_type`` and a zero pool."""

import torch as th
import torch.nn as nn
from torch_geometric.nn import aggr


class PoolWithDimSizeAdapter(nn.Module):
    """Normalize pooling modules to a common interface with optional ``dim_size``."""

    def __init__(self, pool_module: nn.Module) -> None:
        super().__init__()
        self.pool_module = pool_module

    def forward(
        self,
        x: th.Tensor,
        index: th.Tensor | None = None,
        ptr: th.Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
        **kwargs,
    ) -> th.Tensor:
        try:
            pooled = self.pool_module(
                x,
                index=index,
                ptr=ptr,
                dim_size=dim_size,
                dim=dim,
                **kwargs,
            )
        except TypeError as exc:
            if "dim_size" not in str(exc):
                raise
            pooled = self.pool_module(x, index)

        if dim_size is None or pooled.shape[0] == dim_size:
            return pooled

        out = x.new_zeros((dim_size, pooled.shape[1]))
        if pooled.shape[0] > 0:
            out[: pooled.shape[0]] = pooled
        return out


class ZerosAggregation(nn.Module):
    """
    Zero Aggregation module.
    This module performs a dummy aggregation by summing up features across all elements
    and then multiplying the result by zero. Effectively, it outputs a tensor of zeros
    with the same shape as the aggregated result.

    Args:
        nn (torch.nn.Module): Base class for PyTorch modules.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy_agg = aggr.SumAggregation()

    def forward(
        self,
        x: th.Tensor,
        index: th.Tensor | None = None,
        ptr: th.Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
        **kwargs,
    ) -> th.Tensor:
        """
        Forward pass for zero aggregation.

        Args:
            x (torch.Tensor): Input features.
            index (torch.Tensor | None): Group indices for aggregation.
            ptr (torch.Tensor | None): CSR-style segment pointers.
            dim_size (int | None): Explicit number of output groups.
            dim (int): Aggregation dimension.

        Returns:
            torch.Tensor: Zero tensor with the same shape as the aggregated result.
        """
        agg = self.dummy_agg(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, **kwargs)
        return th.zeros_like(agg)


def build_pool_module(pool_type: str, node_dim: int) -> nn.Module:
    """Method to build pooling layer for given pool_type and dim
        node dim is required for attention layer.
    Args:
        pool_type (str): pooling type in ['sum', 'mean', 'max', 'attention', 'softmax',
            'mean_max_sum', 'mean_std_softmax', 'none']
        node_dim (int): node_dim, used for attention
    Returns:
        nn.Module: pooling module
    """

    if pool_type == "sum":
        pool_module = aggr.SumAggregation()
    elif pool_type == "mean":
        pool_module = aggr.MeanAggregation()
    elif pool_type == "max":
        pool_module = aggr.MaxAggregation()
    elif pool_type == "attention":
        pool_module = aggr.AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, 1))
        )
    elif pool_type == "softmax":
        pool_module = aggr.SoftmaxAggregation(learn=True)
    elif pool_type == "mean_max_sum":
        # raise NotImplementedError
        pool_module = aggr.MultiAggregation(["mean", "max", "sum"])
    elif pool_type == "mean_std_softmax":
        # raise NotImplementedError
        pool_module = aggr.MultiAggregation(["mean", "std", aggr.SoftmaxAggregation(learn=True)])
    elif pool_type == "none":
        # just return 0s
        pool_module = ZerosAggregation()
    else:
        raise ValueError(f"Unknown pool_type: {pool_type!r}")
    return PoolWithDimSizeAdapter(pool_module)
