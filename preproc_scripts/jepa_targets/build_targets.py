"""Build the DreaMS JEPA target side table (``docs/plans/masster/masster_jepa_embedding_plan.md``).

Embeds measured spectra with the self-supervised DreaMS backbone (``ssl_model.ckpt`` only; the
fine-tuned ``embedding_model.ckpt`` that ``dreams_embeddings()`` loads saw labelled MoNA spectra)
using the representation Gate 0 validated: the top 100 peaks, m/z <= 1000, peaks below 1% of the
base peak dropped, at least 3 peaks left. The 1% cut keeps a peak at exactly 1% (``>=``): every
NIST20 base peak is 999.0, so the cut is 9.99 and thousands of peaks sit on it. A spectrum failing
the rule, or with precursor m/z above 1000 (DreaMS format A), is still written, with
``has_target`` False and an all-zero row, so the table covers every requested spectrum.

Input: the peaks ``.npz`` of ``prep_inputs.py``. Output: the ``.npz`` layout of
``src/fragnnet/utils/jepa_utils.py`` (the schema is written by hand here because this script runs
in the ``dreams`` environment, which does not import fragnnet).

Run with the dreams environment from the repository root. ``DREAMS_SSL_CKPT`` names a local
``ssl_model.ckpt``; without it the checkpoint is downloaded:

    /home/feiw/miniconda3/envs/dreams/bin/python preproc_scripts/jepa_targets/build_targets.py \
        <peaks.npz> data/proc/nist20/jepa_dreams_ssl_1pct_floor.npz <dreams_commit> <source note>
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel
from dreams.utils import data as du
from dreams.utils import dformats
from dreams.utils.misc import download_pretrained_model

logger = logging.getLogger(__name__)

SCHEMA = 1
N_PEAKS = 100
MAX_MZ = 1000.0
REL_FLOOR = 0.01
MIN_PEAKS = 3
BATCH = 128


def top_n(mz: np.ndarray, it: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ``N_PEAKS`` most intense peaks with positive intensity, sorted by m/z."""
    keep = it > 0
    mz, it = mz[keep], it[keep]
    if mz.size > N_PEAKS:
        idx = np.argpartition(-it, N_PEAKS - 1)[:N_PEAKS]
        mz, it = mz[idx], it[idx]
    order = np.argsort(mz)
    return mz[order], it[order]


def floored(mz: np.ndarray, it: np.ndarray) -> np.ndarray | None:
    """m/z <= 1000, then the 1% floor on what is left; ``None`` below ``MIN_PEAKS`` peaks."""
    keep = mz <= MAX_MZ
    mz, it = mz[keep], it[keep].astype(np.float64)
    if it.size:
        keep = it >= REL_FLOOR * it.max()
        mz, it = mz[keep], it[keep]
    if mz.size < MIN_PEAKS:
        return None
    return np.stack([mz, it], axis=1)


def main(peaks_fp: str, out_fp: str, dreams_commit: str, source: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    local_ckpt = os.environ.get("DREAMS_SSL_CKPT")
    ckpt = Path(local_ckpt or download_pretrained_model("ssl_model.ckpt", verbose=False))
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    model = PreTrainedModel.from_ckpt(ckpt, DreaMSModel, n_highest_peaks=N_PEAKS).model
    prep = du.SpectrumPreprocessor(dformat=dformats.DataFormatA(), n_highest_peaks=N_PEAKS)

    # 1. Inputs, which must already be sorted by spec_id as the table is.
    arr = np.load(peaks_fp)
    spec_id, prec_mz = arr["spec_id"], arr["prec_mz"]
    off, mz_all, it_all = arr["meas_off"], arr["meas_mz"], arr["meas_int"]
    if not (np.diff(spec_id) > 0).all():
        raise ValueError("spec_id in the peaks file must be strictly ascending")

    # 2. Embed every spectrum the rule keeps; the rest keep a zero row.
    n = spec_id.size
    emb = np.zeros((n, 1024), dtype=np.float16)
    has_target = np.zeros(n, dtype=bool)
    rows: list[int] = []
    batch: list[np.ndarray] = []
    n_prec = 0

    def flush() -> None:
        x = torch.from_numpy(np.stack(batch)).to(model.device, model.dtype)
        with torch.inference_mode():
            emb[rows] = model(x)[:, 0, :].float().cpu().numpy().astype(np.float16)
        has_target[rows] = True
        rows.clear()
        batch.clear()

    for i in range(n):
        if prec_mz[i] > MAX_MZ:
            n_prec += 1
            continue
        s = floored(*top_n(mz_all[off[i] : off[i + 1]], it_all[off[i] : off[i + 1]]))
        if s is None:
            continue
        batch.append(prep(s, prec_mz=float(prec_mz[i]), high_form=True))
        rows.append(i)
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()

    # 3. Spot check: one-spectrum forward passes must reproduce 64 batched rows, which catches a
    #    row written next to the wrong spectrum. Batch composition does not change an embedding:
    #    every spectrum is padded to the same 100 peaks.
    rng = np.random.default_rng(0)
    for i in rng.choice(np.flatnonzero(has_target), size=min(64, int(has_target.sum())), replace=False):
        s = floored(*top_n(mz_all[off[i] : off[i + 1]], it_all[off[i] : off[i + 1]]))
        x = torch.from_numpy(prep(s, prec_mz=float(prec_mz[i]), high_form=True)[None])
        with torch.inference_mode():
            single = model(x.to(model.device, model.dtype))[0, 0, :].float().cpu().numpy()
        row = emb[i].astype(np.float32)
        if single @ row / np.linalg.norm(single) / np.linalg.norm(row) < 0.9999:
            raise ValueError(f"row {i} (spec_id {spec_id[i]}) differs from its one-spectrum embedding")

    # 4. The invariants the training-side loader re-checks.
    if not np.isfinite(emb.astype(np.float32)).all():
        raise ValueError("non-finite embedding")
    if not np.array_equal((emb != 0).any(axis=1), has_target):
        raise ValueError("a row is nonzero where has_target is False, or zero where it is True")
    np.savez(
        out_fp,
        schema=np.array(SCHEMA),
        spec_id=spec_id.astype(np.int64),
        has_target=has_target,
        emb=emb,
        n_peaks=arr["n_peaks"].astype(np.int32),
        mz_sum=arr["mz_sum"].astype(np.float64),
        ckpt_sha256=np.array(sha),
        dreams_commit=np.array(dreams_commit),
        rel_floor=np.array(REL_FLOOR),
        top_n_peaks=np.array(N_PEAKS),
        max_mz=np.array(MAX_MZ),
        min_peaks=np.array(MIN_PEAKS),
        tie_rule=np.array(">="),
        source=np.array(source),
        **{k: arr[k] for k in ("spec_df_md5", "train_ids_md5", "val_ids_md5") if k in arr.files},
    )
    logger.info(
        "input %d  with target %d  without: precursor above m/z %d %d, rule %d",
        n,
        int(has_target.sum()),
        int(MAX_MZ),
        n_prec,
        n - int(has_target.sum()) - n_prec,
    )
    logger.info("ckpt sha256 %s\nwrote %s", sha, out_fp)


if __name__ == "__main__":
    main(*sys.argv[1:5])
