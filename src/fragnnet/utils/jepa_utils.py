"""Frozen-embedding targets and predictor for MASSter's JEPA auxiliary loss.

``docs/plans/masster/masster_jepa_embedding_plan.md`` (Step 1) adds one training-only term: a small
MLP predicts a frozen DreaMS embedding of each spectrum's measured spectrum from MASSter's own
pooled representation, and ``weight * (1 - cosine)`` joins the loss. The targets are computed once,
offline, and stored in a side table keyed by ``spec_id``. This module holds what the Lightning
wrapper needs and nothing else: the table, the predictor and the masked cosine.

Side-table ``.npz`` layout (no pickled objects). One row per spectrum of the train and val splits,
sorted by ``spec_id``, including the spectra that have no target:

- ``schema``: ``int64`` scalar, ``1``.
- ``spec_id``: ``int64 (n,)``, strictly ascending.
- ``has_target``: ``bool (n,)``. False when DreaMS cannot embed the spectrum: fewer than 3 peaks
  after the 1% floor, or a precursor above m/z 1000. Such spectra are masked out of the loss.
- ``emb``: ``float16 (n, 1024)``. Row ``i`` is the target of ``spec_id[i]``. All zero exactly where
  ``has_target`` is False, finite everywhere.
- ``n_peaks``: ``int32 (n,)``. Raw peak count of the spectrum, which is what the dataset delivers.
- ``mz_sum``: ``float64 (n,)``. Sum of the raw m/z values, each rounded to float32 as the dataset
  does (:func:`raw_peak_signature`). With ``n_peaks`` it ties a batch spectrum to its table row.
- provenance scalars (checkpoint hash, DreaMS commit, floor rule, md5 of spec_df and splits). They
  are written by the builder and not read here.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

JEPA_SOURCES = ("encoder", "allocator")
JEPA_EMBEDDING_DIM = 1024
JEPA_TABLE_SCHEMA = 1
# Float64 sums of at most ~1,000 float32 values below m/z 1,500 agree to ~1e-8 in any order.
MZ_SUM_ATOL = 1e-6
_TABLE_KEYS = ("schema", "spec_id", "has_target", "emb", "n_peaks", "mz_sum")
_CHECK_BLOCK_ROWS = 32768


def raw_peak_signature(mzs: np.ndarray | list[float]) -> tuple[int, float]:
    """Return the count and m/z sum a side-table row stores for one spectrum.

    The dataset rounds every m/z to float32 (``th.tensor(..., dtype=th.float)``), so the sum is
    taken over the float32-rounded values, in float64.

    Args:
        mzs: Raw m/z of every peak of the spec_df row, ``(num_peaks,)``.

    Returns:
        ``(num_peaks, sum of float32-rounded m/z)``.
    """
    rounded = np.asarray(mzs, dtype=np.float64).astype(np.float32).astype(np.float64)
    return int(rounded.size), float(rounded.sum())


def write_jepa_table(
    path: str,
    *,
    spec_id: np.ndarray,
    has_target: np.ndarray,
    emb: np.ndarray,
    n_peaks: np.ndarray,
    mz_sum: np.ndarray,
    **provenance: object,
) -> None:
    """Write a side table in the layout of the module docstring, sorted by ``spec_id``.

    Args:
        path: Output ``.npz`` file.
        spec_id: Spectrum ids, ``(n,)``, any order.
        has_target: Whether each spectrum has a target, ``(n,)``.
        emb: Target embeddings, ``(n, 1024)``, all zero where ``has_target`` is False.
        n_peaks: Raw peak counts, ``(n,)``.
        mz_sum: Raw m/z sums from :func:`raw_peak_signature`, ``(n,)``.
        **provenance: Scalars stored for provenance (checkpoint hash, commit, floor rule, ...).
    """
    order = np.argsort(spec_id, kind="stable")
    np.savez(
        path,
        schema=np.array(JEPA_TABLE_SCHEMA),
        spec_id=np.asarray(spec_id, dtype=np.int64)[order],
        has_target=np.asarray(has_target, dtype=bool)[order],
        emb=np.asarray(emb, dtype=np.float16)[order],
        n_peaks=np.asarray(n_peaks, dtype=np.int32)[order],
        mz_sum=np.asarray(mz_sum, dtype=np.float64)[order],
        **{key: np.array(value) for key, value in provenance.items()},
    )


class JepaTargetTable:
    """Target embeddings of the train and val spectra, held in host memory.

    Numpy arrays, not tensors: a table first opened inside an inference-mode evaluation loop
    cannot leave inference tensors behind.

    Args:
        path: Side-table ``.npz`` (layout in the module docstring).

    Raises:
        ValueError: If the file does not follow the layout. ``searchsorted`` on unsorted ids, or a
            nonzero row without a target, would otherwise give wrong targets silently.
    """

    def __init__(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as table:
            missing = [key for key in _TABLE_KEYS if key not in table.files]
            if missing:
                raise ValueError(f"{path}: missing arrays {missing}")
            if int(table["schema"]) != JEPA_TABLE_SCHEMA:
                raise ValueError(f"{path}: expected JEPA table schema {JEPA_TABLE_SCHEMA}")
            self.spec_ids: np.ndarray = table["spec_id"]
            self.has_target: np.ndarray = table["has_target"]
            self.embeddings: np.ndarray = table["emb"]
            self.n_peaks: np.ndarray = table["n_peaks"]
            self.mz_sum: np.ndarray = table["mz_sum"]
        rows = self.spec_ids.size
        if (
            self.spec_ids.dtype != np.int64
            or self.spec_ids.ndim != 1
            or self.has_target.dtype != np.bool_
            or self.has_target.shape != (rows,)
            or self.embeddings.dtype != np.float16
            or self.embeddings.shape != (rows, JEPA_EMBEDDING_DIM)
            or self.n_peaks.dtype != np.int32
            or self.n_peaks.shape != (rows,)
            or self.mz_sum.dtype != np.float64
            or self.mz_sum.shape != (rows,)
        ):
            raise ValueError(
                f"{path}: spec_id int64 (n,), has_target bool (n,), emb float16 (n, "
                f"{JEPA_EMBEDDING_DIM}), n_peaks int32 (n,) and mz_sum float64 (n,) are required"
            )
        if rows == 0 or (np.diff(self.spec_ids) <= 0).any():
            raise ValueError(f"{path}: spec_id must be non-empty and strictly ascending")
        for start in range(0, rows, _CHECK_BLOCK_ROWS):
            block = self.embeddings[start : start + _CHECK_BLOCK_ROWS]
            if not np.isfinite(block).all():
                raise ValueError(f"{path}: emb holds a non-finite value")
            nonzero = (block != 0).any(axis=1)
            if not np.array_equal(nonzero, self.has_target[start : start + _CHECK_BLOCK_ROWS]):
                raise ValueError(f"{path}: emb must be nonzero exactly where has_target is True")
        logger.info(
            "JEPA target table %s: %d spectra, %d with a target",
            path,
            rows,
            int(self.has_target.sum()),
        )

    def lookup(
        self,
        spec_ids: th.Tensor,
        peak_mzs: th.Tensor,
        peak_batch_idxs: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Return the target of each batch spectrum and whether it has one.

        Every spectrum must be in the table, and its raw peak count and m/z sum must equal the
        table's record for that ``spec_id``. A table built for other spectra, a regenerated
        ``spec_df``, or a shifted row therefore fails at the first batch instead of training on
        another spectrum's embedding.

        Args:
            spec_ids: ``spec_id`` of each spectrum, CPU ``int64`` ``(batch_size,)``.
            peak_mzs: m/z of every peak the dataset delivered, CPU ``(num_peaks,)``.
            peak_batch_idxs: Spectrum index of each peak, CPU ``int64`` ``(num_peaks,)``.

        Returns:
            ``(target, valid)``: ``float16 (batch_size, 1024)`` with all-zero rows where there is
            no target, and ``bool (batch_size,)``.

        Raises:
            ValueError: If a tensor is not on the CPU, an id is not in the table, or a spectrum's
                peaks do not match the table's record of it.
        """
        for name, tensor in (("spec_ids", spec_ids), ("peak_batch_idxs", peak_batch_idxs)):
            if tensor.device.type != "cpu" or tensor.dtype != th.long:
                raise ValueError(
                    f"{name} must be a CPU int64 tensor, got {tensor.dtype} on {tensor.device}"
                )
        if peak_mzs.device.type != "cpu":
            raise ValueError(f"peak_mzs must be a CPU tensor, got {peak_mzs.device}")
        ids = spec_ids.reshape(-1).numpy()
        batch_size = ids.size
        rows = np.minimum(np.searchsorted(self.spec_ids, ids), self.spec_ids.size - 1)
        covered = self.spec_ids[rows] == ids
        if not covered.all():
            raise ValueError(
                f"{int((~covered).sum())} of {batch_size} spec_ids are not in the JEPA target "
                f"table, for example {ids[~covered][:5].tolist()}; rebuild or audit the table "
                "for this spec_df and split"
            )
        batch_idxs = peak_batch_idxs.numpy()
        counts = np.bincount(batch_idxs, minlength=batch_size)
        if counts.size != batch_size:
            raise ValueError("peak_batch_idxs refers to a spectrum outside the batch")
        sums = np.bincount(
            batch_idxs, weights=peak_mzs.numpy().astype(np.float64), minlength=batch_size
        )
        mismatch = (counts != self.n_peaks[rows]) | (np.abs(sums - self.mz_sum[rows]) > MZ_SUM_ATOL)
        if mismatch.any():
            raise ValueError(
                f"the peaks of spec_ids {ids[mismatch][:5].tolist()} do not match the raw peaks "
                "the JEPA target table was built from; rebuild or audit the table, and note that "
                "the target describes the unmodified spec_df spectrum"
            )
        # Fancy indexing copies. A row is zero exactly where there is no target (checked on load).
        return th.from_numpy(self.embeddings[rows]), th.from_numpy(self.has_target[rows])


@lru_cache(maxsize=1)
def load_jepa_target_table(path: str) -> JepaTargetTable:
    """Open the side table once per process (about 0.43 GB for NIST20 train + val)."""
    return JepaTargetTable(path)


def build_jepa_predictor(in_dim: int, hidden_dim: int) -> nn.Sequential:
    """Two-layer MLP from a pooled representation to the 1,024-d target.

    The input LayerNorm makes the predictor indifferent to the scale of the encoder it reads,
    which drifts during training, and it adds no dependence between spectra (BatchNorm would).
    There is no dropout, so the module draws no random numbers in training. The output layer is
    not zero-initialised: a zero prediction has cosine 0 with a ~1e8 gradient on that layer and
    exactly zero gradient into the encoder.

    Args:
        in_dim: Width of the pooled representation.
        hidden_dim: Width of the hidden layer.

    Returns:
        ``LayerNorm(in_dim) -> Linear(in_dim, hidden_dim) -> SiLU -> Linear(hidden_dim, 1024)``.
    """
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, JEPA_EMBEDDING_DIM),
    )


def masked_cosine_distance(
    pred: th.Tensor, target: th.Tensor, valid: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    """Per-spectrum ``1 - cosine`` and cosine, exactly zero where a spectrum has no target.

    The target of a masked spectrum is replaced by zeros before the cosine, so a bad value in its
    row can neither reach the loss nor turn its gradient into NaN.

    Args:
        pred: Predicted embeddings, ``(batch_size, 1024)``.
        target: Target embeddings in any floating dtype, same shape.
        valid: Whether each spectrum has a target, ``bool (batch_size,)``.

    Returns:
        ``(distance, cosine)``, each of shape ``(batch_size,)``.

    Raises:
        ValueError: If the shapes disagree. A short vector would otherwise broadcast silently
            against the per-spectrum loss.
    """
    if pred.ndim != 2 or target.shape != pred.shape or valid.shape != pred.shape[:1]:
        raise ValueError(
            f"pred {tuple(pred.shape)}, target {tuple(target.shape)} and valid "
            f"{tuple(valid.shape)} must be (B, D), (B, D) and (B,)"
        )
    target = th.where(valid.unsqueeze(1), target.to(pred.dtype), th.zeros_like(pred))
    cosine = F.cosine_similarity(pred, target, dim=1)
    zeros = th.zeros_like(cosine)
    return th.where(valid, 1.0 - cosine, zeros), th.where(valid, cosine, zeros)
