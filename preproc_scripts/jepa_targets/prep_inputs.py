"""Raw measured peaks of the train and val spectra, for the JEPA target builder.

Step 1 of ``docs/plans/masster/masster_jepa_embedding_plan.md``. Reads ``spec_df.pkl`` and the two
split files, and writes one ``.npz`` the ``dreams`` environment can read without unpickling pandas
objects: the RAW float64 peaks and precursor m/z of every train and val spectrum, plus each
spectrum's peak signature (count and float32-rounded m/z sum), which the training hook checks
every batch against. Intensities stay float64: DreaMS is scale invariant, but the 1% floor has
heavily populated ties (every NIST20 base peak is 999.0, so the cut is 9.99), and float32 or
L1-normalised intensities classify 16 val spectra differently.

Run in FRAGNNET-GPU from the repository root (about 10 GB of RAM for the 1.5 GB spec_df):

    python preproc_scripts/jepa_targets/prep_inputs.py \
        data/proc/nist20/spec_df.pkl data/split/nist20_inchikey <peaks.npz>
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fragnnet.utils.jepa_utils import raw_peak_signature

logger = logging.getLogger(__name__)


def md5_of(path: Path) -> str:
    """Return the md5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec_fp", help="spec_df pickle (data/proc/nist20/spec_df.pkl)")
    parser.add_argument("split_dp", help="directory with train_ids.csv and val_ids.csv")
    parser.add_argument("out_fp", help="output peaks .npz")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Which spectra: every id of the train and val split files, each exactly once.
    split_files = {name: Path(args.split_dp) / f"{name}_ids.csv" for name in ("train", "val")}
    wanted = pd.concat([pd.read_csv(fp).spec_id for fp in split_files.values()]).astype(np.int64)
    if not wanted.is_unique:
        raise ValueError("a spec_id appears twice in the train and val split files")

    # 2. Their raw peaks, from the one spec_df of the dataset.
    spec = pd.read_pickle(args.spec_fp)
    spec["spec_id"] = spec["spec_id"].astype(np.int64)
    spec = spec[spec["spec_id"].isin(wanted)]
    if not spec["spec_id"].is_unique:
        raise ValueError("spec_df holds a spec_id twice")
    missing = np.setdiff1d(wanted.to_numpy(), spec["spec_id"].to_numpy())
    if missing.size:
        raise ValueError(f"{missing.size} split spec_ids are not in spec_df, e.g. {missing[:5]}")
    spec = spec.sort_values("spec_id")
    peaks = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in spec["peaks"]]
    counts = np.array([p.shape[0] for p in peaks], dtype=np.int64)
    signatures = np.array([raw_peak_signature(p[:, 0]) for p in peaks])

    # 3. One file, with the identity of every input.
    np.savez(
        args.out_fp,
        spec_id=spec["spec_id"].to_numpy(np.int64),
        prec_mz=spec["prec_mz"].to_numpy(np.float64),
        meas_off=np.concatenate([[0], np.cumsum(counts)]),
        meas_mz=np.concatenate([p[:, 0] for p in peaks]),
        meas_int=np.concatenate([p[:, 1] for p in peaks]),
        n_peaks=signatures[:, 0].astype(np.int32),
        mz_sum=signatures[:, 1],
        spec_df_md5=np.array(md5_of(Path(args.spec_fp))),
        train_ids_md5=np.array(md5_of(split_files["train"])),
        val_ids_md5=np.array(md5_of(split_files["val"])),
    )
    logger.info(
        "wrote %s: %d spectra, %d peaks, median %d per spectrum",
        args.out_fp,
        len(spec),
        counts.sum(),
        np.median(counts),
    )


if __name__ == "__main__":
    main()
