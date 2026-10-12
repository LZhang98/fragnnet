"""The binned spectrum head shared by NEIMS, GNN and MassFormer: SpecFFN with its
NeimsBlock and LowRankDense layers, and the NEIMS reverse-prediction and precursor-mass
masking helpers.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fragnnet.utils.misc_utils import safelog


class NeimsBlock(nn.Module):
    """from the NEIMS paper (uses LeakyReLU instead of ReLU)"""

    def __init__(
        self, in_dim: int, out_dim: int, dropout: float, bottleneck_factor: float = 0.5
    ) -> None:
        super().__init__()
        bottleneck_size = int(round(bottleneck_factor * out_dim))
        self.in_batch_norm = nn.BatchNorm1d(in_dim)
        self.in_activation = nn.LeakyReLU()
        self.in_linear = nn.Linear(in_dim, bottleneck_size)
        self.out_batch_norm = nn.BatchNorm1d(bottleneck_size)
        self.out_linear = nn.Linear(bottleneck_size, out_dim)
        self.out_activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: th.Tensor) -> th.Tensor:
        h = x
        h = self.in_batch_norm(h)
        h = self.in_activation(h)
        h = self.dropout(h)
        h = self.in_linear(h)
        h = self.out_batch_norm(h)
        h = self.out_activation(h)
        h = self.out_linear(h)
        return h


class LowRankDense(nn.Module):
    """
    Low-Rank Dense Layer. Ref: https://arxiv.org/pdf/2001.08885
    This class implements a low-rank approximation of a dense (fully connected) layer.
    Instead of a single weight matrix, it factorizes the weight matrix into two smaller
    matrices, reducing the number of parameters and computational cost.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        rank (int): Rank of the factorization (controls the size of the intermediate layer).
        bias (bool): Whether to include a bias term in the layer.
    """

    def __init__(self, input_dim: int, output_dim: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # First projection: input_dim -> rank
        self.proj1 = nn.Linear(input_dim, rank, bias=False)

        # Second projection: rank -> output_dim
        self.proj2 = nn.Linear(rank, output_dim, bias=bias)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass for the Low-Rank Dense layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        # First projection: Reduce input dimension to rank
        x = self.proj1(x)

        # Second projection: Map rank dimension to output dimension
        x = self.proj2(x)

        return x


class SpecFFN(nn.Module):
    """Spectral Feed-Forward Network.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        mz_max,
        mz_bin_res,
        num_layers,
        dropout,
        prec_mz_offset,
        bidirectional,
        use_residuals,
        output_map_size,
        output_activation,
        log_min,
        bottleneck_factor=0.5,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.mz_max = mz_max
        self.mz_bin_res = mz_bin_res
        self.prec_mz_offset = prec_mz_offset
        self.bidirectional = bidirectional
        self.use_residuals = use_residuals
        self._compute_output_size()

        self.in_layer = nn.Linear(input_size, hidden_size)
        self.ff_layers = nn.ModuleList([])
        # self.ff_layers.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
        for _i in range(num_layers):
            self.ff_layers.append(NeimsBlock(hidden_size, hidden_size, dropout, bottleneck_factor))
        if output_map_size == -1:
            if self.bidirectional:
                # assumes gating, mass masking
                self.forw_out_layer = nn.Linear(hidden_size, self.output_size)
                self.rev_out_layer = nn.Linear(hidden_size, self.output_size)
                self.out_gate = nn.Sequential(
                    *[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()]
                )
            else:
                self.out_layer = nn.Linear(hidden_size, self.output_size)
                self.out_gate = nn.Sequential(
                    *[nn.Linear(hidden_size, self.output_size), nn.Sigmoid()]
                )
        else:
            if self.bidirectional:
                # assumes gating, mass masking
                self.forw_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.rev_out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.out_gate = nn.Sequential(
                    *[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()]
                )
            else:
                self.out_layer = LowRankDense(hidden_size, self.output_size, output_map_size)
                self.out_gate = nn.Sequential(
                    *[LowRankDense(hidden_size, self.output_size, output_map_size), nn.Sigmoid()]
                )
        if output_activation not in ["relu", "sigmoid"]:
            raise ValueError(
                f"unknown output_activation {output_activation!r}; expected 'relu' or 'sigmoid'"
            )
        if output_activation == "relu":
            self.out_activation = nn.ReLU()
        else:
            self.out_activation = nn.Sigmoid()
        self.out_normalization = lambda x: F.normalize(x, p=1, dim=1)
        self.log_min = log_min

    def _compute_output_size(self):
        mz_bins = th.arange(
            self.mz_bin_res, self.mz_max + self.mz_bin_res, self.mz_bin_res, dtype=th.float32
        )
        self.register_buffer("mz_bins", mz_bins, persistent=False)
        self.register_buffer("mzs", mz_bins - 0.5 * self.mz_bin_res, persistent=False)
        self.output_size = mz_bins.shape[0]

    def _prec_mz_to_idx(self, prec_mz):
        prec_mz_idx = th.bucketize(prec_mz, self.mz_bins.to(prec_mz.device), right=True)
        assert th.max(prec_mz_idx) < self.output_size, (prec_mz_idx, self.output_size)
        return prec_mz_idx

    def forward(
        self, input_h: th.Tensor, prec_mz: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # get prec_mz_idxs
        prec_mz_idxs = self._prec_mz_to_idx(prec_mz)
        # process inputs
        fh = self.in_layer(input_h)
        # big MLP
        for layer in self.ff_layers:
            if self.use_residuals:
                fh = fh + layer(fh)
            else:
                fh = layer(fh)
        # bidirectional prediction
        if self.bidirectional:
            ff = self.forw_out_layer(fh)
            fr = reverse_prediction(self.rev_out_layer(fh), prec_mz_idxs, self.prec_mz_offset)
            fg = self.out_gate(fh)
            fo = ff * fg + fr * (1.0 - fg)
        else:
            # apply output layer
            fo = self.out_layer(fh)
            # apply gating
            fg = self.out_gate(fh)
            fo = fg * fo
        fo = self.out_activation(fo)
        fo = mask_prediction_by_mass(fo, prec_mz_idxs, self.prec_mz_offset)
        spec = self.out_normalization(fo)
        # handle all zeroes (set to first bin by default)
        all_zero_mask = th.max(spec, dim=1)[0] <= 0.0
        all_zero_bonus = th.zeros_like(spec)
        all_zero_bonus[all_zero_mask, 0] = 1.0
        spec = spec + all_zero_bonus
        # convert dense spectrum to sparse
        mask = spec > 0.0
        pred_mzs = (self.mzs.unsqueeze(0).expand(spec.shape[0], -1))[mask]
        pred_logprobs = safelog(spec, eps=self.log_min)[mask]
        pred_batch_idxs = (
            th.arange(spec.shape[0], device=spec.device)
            .unsqueeze(1)
            .expand(-1, spec.shape[1])[mask]
        )
        pred_specs = spec
        return pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs


def reverse_prediction(
    raw_prediction: th.Tensor, prec_mass_idx: th.Tensor, prec_mass_offset: int
) -> th.Tensor:
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    batch_size = raw_prediction.shape[0]
    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    rev_prediction = th.flip(raw_prediction, dims=(1,))
    # convention is to shift right, so we express as negative to go left
    offset_idx = th.minimum(
        max_idx * th.ones_like(prec_mass_idx), prec_mass_idx + prec_mass_offset + 1
    )
    shifts = -(max_idx - offset_idx)
    gather_idx = (
        th.arange(max_idx, device=raw_prediction.device).unsqueeze(0).expand(batch_size, max_idx)
    )
    gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
    offset_rev_prediction = th.gather(rev_prediction, 1, gather_idx)
    # you could mask_prediction_by_mass here but it's unnecessary
    return offset_rev_prediction


def mask_prediction_by_mass(
    raw_prediction: th.Tensor, prec_mass_idx: th.Tensor, prec_mass_offset: int
) -> th.Tensor:
    # adapted from NEIMS
    # raw_prediction is [B,D], prec_mass_idx is [B]

    max_idx = raw_prediction.shape[1]
    assert th.all(prec_mass_idx < max_idx)
    idx = th.arange(max_idx, device=prec_mass_idx.device)
    mask = (idx.unsqueeze(0) <= (prec_mass_idx.unsqueeze(1) + prec_mass_offset)).float()
    return mask * raw_prediction
