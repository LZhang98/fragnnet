"""Optional intact-molecule RT and adduct-conditioned CCS regression heads."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def pool_property_atoms(
    atom_embeddings: th.Tensor, atom_batch: th.Tensor, batch_size: int
) -> th.Tensor:
    """Mean-pool atom embeddings into one structure embedding per spectrum."""
    pooled = atom_embeddings.new_zeros(batch_size, atom_embeddings.shape[-1])
    pooled = pooled.index_add(0, atom_batch, atom_embeddings)
    counts = th.bincount(atom_batch, minlength=batch_size).clamp_min(1)
    return pooled / counts.to(pooled).unsqueeze(1)


class RTHead(nn.Module):
    """Predict positive retention time in seconds from pooled structure embeddings."""

    def __init__(self, atom_dim: int, hidden_dim: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.regressor = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, pooled: th.Tensor) -> th.Tensor:
        """Map ``(batch, atom_dim)`` embeddings to ``(batch,)`` RT predictions."""
        return F.softplus(self.regressor(pooled).squeeze(-1)) * self.scale


class CCSHead(nn.Module):
    """Predict positive CCS in source units from structure and precursor adduct."""

    def __init__(
        self, atom_dim: int, hidden_dim: int, num_adducts: int, adduct_dim: int, scale: float
    ) -> None:
        super().__init__()
        self.scale = scale
        self.adduct_embedding = nn.Embedding(num_adducts, adduct_dim)
        self.regressor = nn.Sequential(
            nn.Linear(atom_dim + adduct_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, pooled: th.Tensor, adducts: th.Tensor) -> th.Tensor:
        """Return ``(batch,)`` CCS predictions conditioned on integer adduct IDs."""
        inputs = th.cat((pooled, self.adduct_embedding(adducts.reshape(-1))), dim=1)
        return F.softplus(self.regressor(inputs).squeeze(-1)) * self.scale


def property_regression_terms(
    prediction: th.Tensor,
    target: th.Tensor,
    scale: float,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Return masked Smooth-L1, absolute error, percentage error, and validity.

    Finite positive labels are valid. Missing labels contribute a graph-connected
    zero, including wholly unlabelled batches. Loss is Smooth-L1 of the residual
    divided by ``scale`` (beta=1), returned per spectrum for the existing reducer.
    Metric callers must exclude invalid rows from their denominators.
    """
    prediction, target = prediction.float(), target.float()
    if prediction.shape != target.shape:
        raise ValueError("RT/CCS predictions and labels must have matching shapes")
    valid = th.isfinite(target) & (target > 0)
    safe_target = th.where(valid, target, prediction.detach())
    residual = (prediction - safe_target) / scale
    loss = F.smooth_l1_loss(residual, th.zeros_like(residual), reduction="none")
    absolute_error = (prediction.detach() - safe_target).abs()
    percentage_error = 100.0 * absolute_error / th.where(valid, target, th.ones_like(target))
    return loss, absolute_error, percentage_error, valid
