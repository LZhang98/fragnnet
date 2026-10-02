"""Audit a JEPA target side table against the spec_df and split files it must describe.

Step 1 of ``docs/plans/masster/masster_jepa_embedding_plan.md``. The training hook checks every
batch spectrum's peaks against the table, but it cannot see a wrong EMBEDDING inside a table whose
ids, counts and m/z sums are right (for example a builder that sorted the ids and not the rows).
This audit is the other half: it recomputes everything the table claims from the raw spec_df with
its own implementation of the target rule, and it can compare val rows with the Gate 0 embeddings,
which were made by a different pipeline. Run it before any run that reads the table.

Checks, each fatal on failure:
1. the table loads (layout, dtypes, sorted ids, finite rows, zero row exactly where no target);
2. its ids are exactly the train and val ids of the split files, each once;
3. ``n_peaks`` and ``mz_sum`` equal the raw peaks of the spec_df row;
4. ``has_target`` equals the rule recomputed here on the raw float64 peaks;
5. (optional, val) rows agree with ``gate0_embeddings.npz`` ``meas_rel01``: cosine 1 up to float16,
   except spectra that have a peak at exactly 1% of the base peak, which Gate 0's float32 pipeline
   dropped and this table keeps.

Run in FRAGNNET-GPU from the repository root (about 10 GB of RAM for the 1.5 GB spec_df):

    python preproc_scripts/jepa_targets/audit_targets.py <table.npz> \
        data/proc/nist20/spec_df.pkl data/split/nist20_inchikey \
        [--gate0 output/dreams_gate0_2026_10_01/gate0_embeddings.npz]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fragnnet.utils.jepa_utils import JepaTargetTable, raw_peak_signature

logger = logging.getLogger(__name__)

TOP_N, MAX_MZ, REL_FLOOR, MIN_PEAKS = 100, 1000.0, 0.01, 3


def expected_has_target(peaks: np.ndarray, prec_mz: float) -> tuple[bool, bool]:
    """The target rule on raw float64 peaks: whether the spectrum has a target, and whether a
    peak sits exactly on the 1% cut (where ``>=`` and ``>`` would disagree).

    Rule: precursor m/z <= 1000; intensity > 0; the 100 most intense peaks; m/z <= 1000; peaks
    with intensity >= 1% of what is then the base peak; at least 3 of them.
    """
    if prec_mz > MAX_MZ:
        return False, False
    peaks = peaks[peaks[:, 1] > 0]
    if peaks.shape[0] > TOP_N:
        peaks = peaks[np.argsort(-peaks[:, 1], kind="stable")[:TOP_N]]
    peaks = peaks[peaks[:, 0] <= MAX_MZ]
    if peaks.shape[0] == 0:
        return False, False
    cut = REL_FLOOR * peaks[:, 1].max()
    return int((peaks[:, 1] >= cut).sum()) >= MIN_PEAKS, bool(np.isclose(peaks[:, 1], cut).any())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table_fp")
    parser.add_argument("spec_fp")
    parser.add_argument("split_dp")
    parser.add_argument("--gate0", default=None, help="gate0_embeddings.npz for the val check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. The table, which validates its own layout on load.
    table = JepaTargetTable(args.table_fp)

    # 2. Its ids against the split files.
    splits = {
        name: pd.read_csv(Path(args.split_dp) / f"{name}_ids.csv").spec_id.to_numpy(np.int64)
        for name in ("train", "val")
    }
    wanted = np.sort(np.concatenate(list(splits.values())))
    if not np.array_equal(table.spec_ids, wanted):
        extra = np.setdiff1d(table.spec_ids, wanted)
        absent = np.setdiff1d(wanted, table.spec_ids)
        raise ValueError(f"table ids differ from the split ids: {extra.size} extra, {absent.size} absent")

    # 3. and 4. Peaks and rule, recomputed from the raw spec_df.
    spec = pd.read_pickle(args.spec_fp)
    spec["spec_id"] = spec["spec_id"].astype(np.int64)
    spec = spec[spec["spec_id"].isin(wanted)].sort_values("spec_id")
    if not np.array_equal(spec["spec_id"].to_numpy(), wanted):
        raise ValueError("spec_df does not hold every split spectrum exactly once")
    peaks = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in spec["peaks"]]
    signatures = np.array([raw_peak_signature(p[:, 0]) for p in peaks])
    rule = [expected_has_target(p, float(m)) for p, m in zip(peaks, spec["prec_mz"], strict=True)]
    expected = np.array([r[0] for r in rule])
    on_tie = np.array([r[1] for r in rule])
    if not np.array_equal(table.n_peaks, signatures[:, 0].astype(np.int32)):
        raise ValueError("n_peaks differs from the raw spec_df peak counts")
    if np.abs(table.mz_sum - signatures[:, 1]).max() > 1e-6:
        raise ValueError("mz_sum differs from the raw spec_df m/z sums")
    if not np.array_equal(table.has_target, expected):
        bad = table.spec_ids[table.has_target != expected]
        raise ValueError(f"has_target differs from the rule for {bad.size} spectra, e.g. {bad[:5]}")
    for name, ids in splits.items():
        rows = np.searchsorted(table.spec_ids, ids)
        logger.info(
            "%s: %d spectra, %d with a target (%.2f%%), %d with a peak on the 1%% cut",
            name,
            ids.size,
            int(table.has_target[rows].sum()),
            100.0 * table.has_target[rows].mean(),
            int(on_tie[rows].sum()),
        )

    # 5. Val rows against Gate 0, a different pipeline.
    if args.gate0:
        gate0 = np.load(args.gate0)
        row_of = {int(s): i for i, s in enumerate(gate0["spec_id"])}
        val_rows = np.searchsorted(table.spec_ids, splits["val"])
        pairs = [
            (r, row_of[int(s)])
            for r, s in zip(val_rows, splits["val"], strict=True)
            if table.has_target[r] and gate0["meas_rel01_valid"][row_of[int(s)]]
        ]
        ours = table.embeddings[[p[0] for p in pairs]].astype(np.float32)
        theirs = gate0["meas_rel01"][[p[1] for p in pairs]].astype(np.float32)
        cos = (ours * theirs).sum(1) / np.linalg.norm(ours, axis=1) / np.linalg.norm(theirs, axis=1)
        off = cos < 0.9999
        unexplained = off & ~on_tie[[p[0] for p in pairs]]
        logger.info(
            "val vs Gate 0: %d rows, median cosine %.7f, %d below 0.9999 (min %.3f), %d of them "
            "without a peak on the 1%% cut",
            len(pairs),
            np.median(cos),
            int(off.sum()),
            cos.min(),
            int(unexplained.sum()),
        )
        if unexplained.any():
            raise ValueError("val rows differ from Gate 0 for a reason other than 1%-cut ties")
    logger.info("AUDIT PASS")


if __name__ == "__main__":
    main()
