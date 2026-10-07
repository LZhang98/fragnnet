"""Is the noise in CASMI's RIKEN/BMDMS-NP block one shared foreign signal?

The RIKEN rows labelled plain ``QTOF``/``Orbitrap`` are BMDMS-NP records (exact m/z lists in
``MSMS-Pos-bmdms-np_20200811.msp``, Zenodo 10951913; see the CASMI finding, section 5.9). Many of
their scans are not explained by the labelled molecule. This script asks whether those noise scans
of DIFFERENT molecules share the same spectrum, which would make the foreign signal a duplicate that
can be found and removed:

1. Each scan is labelled ``good`` (tight-tier base peak explained and coverage > 0.3 above the
   shifted-peak control), ``noise`` (base peak unexplained and coverage within 0.1 of the control)
   or ``mid``, from ``casmi26_train_spec_stats.parquet``.
2. Scans with >= 3 peaks are binned (0.01 Da, sqrt intensity, L2-normalised). For each scan the
   most similar scan of a molecule with a DIFFERENT formula is found by sparse cosine.
3. A pair above ``--min_cosine`` is a cross-molecule twin. For each twin we record the partner's own
   label (a ``good`` partner means the foreign spectrum is an identifiable library compound), the
   precursor m/z difference (co-isolation needs a small one) and the BMDMS-NP ``spec_id`` distance
   (carry-over between consecutive injections gives a small one).

Outputs in ``--out_dp``: ``scans.parquet`` (one row per scan with its label, twin and twin
statistics) and ``summary.json``.

Usage:
    PYTHONPATH=src python scripts/debug/casmi26/riken_bmdms_shared_signal.py \
        --msp_fp <path to MSMS-Pos-bmdms-np_20200811.msp> \
        --out_dp output/riken_bmdms_shared_signal_2026_10_06
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

STATS_FP = Path("output/casmi26_train_check_2026_10_05/casmi26_train_spec_stats.parquet")


def parse_msp_ids(msp_fp: Path) -> pd.DataFrame:
    """BMDMS-NP records: spec_id, instrument, CE, adduct and the 3-decimal m/z key."""
    rows, rec, mzs = [], {}, []
    with open(msp_fp, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                if rec:
                    rec["key"] = tuple(np.round(mzs, 3))
                    rows.append(rec)
                rec, mzs = {}, []
                continue
            if line[0].isdigit():
                mzs.append(float(line.split()[0]))
            elif ":" in line:
                k, v = line.split(":", 1)
                rec[k.strip()] = v.strip()
    if rec:
        rec["key"] = tuple(np.round(mzs, 3))
        rows.append(rec)
    msp = pd.DataFrame(rows)
    msp["spec_id"] = msp["COMMENT"].str.extract(r"spec_id=(\d+)")[0].astype(int)
    return msp[["spec_id", "INSTRUMENTTYPE", "COLLISIONENERGY", "PRECURSORTYPE", "key"]]


def binned_matrix(mz_lists: list, int_lists: list, bin_da: float) -> csr_matrix:
    """sqrt-intensity, L2-normalised sparse vectors on a 0.01 Da grid."""
    indptr, cols, vals = [0], [], []
    for mzs, ints in zip(mz_lists, int_lists):
        b = np.floor(np.asarray(mzs) / bin_da).astype(np.int64)
        w = np.sqrt(np.asarray(ints, dtype=float))
        ub, inv = np.unique(b, return_inverse=True)
        acc = np.bincount(inv, weights=w)
        cols.extend(ub.tolist())
        vals.extend((acc / np.linalg.norm(acc)).tolist())
        indptr.append(len(cols))
    return csr_matrix((vals, cols, indptr), shape=(len(mz_lists), int(max(cols)) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    parser.add_argument("--msp_fp", type=Path, required=True)
    parser.add_argument("--out_dp", type=Path, required=True)
    parser.add_argument("--bin_da", type=float, default=0.01)
    parser.add_argument("--min_cosine", type=float, default=0.9)
    parser.add_argument("--min_peaks", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=2000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dp.mkdir(parents=True, exist_ok=True)

    # 1. RIKEN plain-label rows (global row index kept), with their stats.
    cols = ["ingest_lib", "inchikey14", "molecular_formula", "adduct", "instrument_type",
            "collision_energy_orig", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
    table = pq.read_table(args.train_fp, columns=cols)
    lib = table.column("ingest_lib").to_numpy(zero_copy_only=False)
    inst = table.column("instrument_type").to_numpy(zero_copy_only=False)
    rows = np.flatnonzero((lib == "riken") & np.isin(inst, ["QTOF", "Orbitrap"]))
    scans = table.take(rows).to_pandas()
    scans.insert(0, "row", rows)
    stats = pd.read_parquet(STATS_FP).iloc[rows].reset_index(drop=True)
    base = stats["sf_base_explained_tight"].astype("boolean").fillna(False).to_numpy()
    excess = (stats["sf_int_frac_tight"] - stats["sf_int_frac_null_tight"]).fillna(-1).to_numpy()
    scans["label"] = np.where(base & (excess > 0.3), "good", np.where(~base & (excess <= 0.1), "noise", "mid"))
    scans["num_peaks"] = scans["ms2_mzs"].map(len)
    logger.info("RIKEN plain-label scans: %d; labels %s", len(scans), scans["label"].value_counts().to_dict())

    # 2. BMDMS-NP spec_id per scan (exact 3-decimal m/z list + instrument + CE + adduct).
    msp = parse_msp_ids(args.msp_fp)
    msp["ce"] = msp["COLLISIONENERGY"].astype(str)
    lookup = msp.groupby(["key", "INSTRUMENTTYPE", "ce", "PRECURSORTYPE"])["spec_id"].apply(list)
    scans["key"] = scans["ms2_mzs"].map(lambda m: tuple(np.round(np.asarray(m, float), 3)))
    ids = [
        lookup.get((k, i, str(c), a)) for k, i, c, a in zip(
            scans["key"], scans["instrument_type"], scans["collision_energy_orig"], scans["adduct"]
        )
    ]
    scans["bmdms_spec_id"] = [x[0] if x and len(x) == 1 else np.nan for x in ids]
    logger.info("mapped to a unique BMDMS-NP spec_id: %.3f", scans["bmdms_spec_id"].notna().mean())

    # 3. Cross-formula nearest neighbour among scans with >= min_peaks peaks.
    sel = scans.index[scans["num_peaks"] >= args.min_peaks].to_numpy()
    mat = binned_matrix(scans.loc[sel, "ms2_mzs"].tolist(), scans.loc[sel, "ms2_normalized_intensities"].tolist(), args.bin_da)
    formula_id = pd.factorize(scans.loc[sel, "molecular_formula"])[0]
    best_cos = np.zeros(len(sel))
    best_j = np.full(len(sel), -1)
    matT = mat.T.tocsr()
    for start in range(0, len(sel), args.chunk):
        sim = (mat[start : start + args.chunk] @ matT).tocoo()
        r, c, v = sim.row + start, sim.col, sim.data
        keep = formula_id[r] != formula_id[c]
        r, c, v = r[keep], c[keep], v[keep]
        order = np.lexsort((-v, r))
        r, c, v = r[order], c[order], v[order]
        first = np.r_[True, r[1:] != r[:-1]]
        best_cos[r[first]] = v[first]
        best_j[r[first]] = c[first]
        logger.info("neighbours %d/%d", min(start + args.chunk, len(sel)), len(sel))
    scans["twin_cos"] = np.nan
    scans["twin_row"] = np.nan
    scans.loc[sel, "twin_cos"] = best_cos
    has = best_j >= 0
    scans.loc[sel[has], "twin_row"] = sel[best_j[has]]

    tw = scans.loc[sel[has]].copy()
    partner = scans.loc[tw["twin_row"].astype(int).to_numpy()].reset_index(drop=True)
    tw = tw.reset_index()
    tw["twin_label"] = partner["label"].to_numpy()
    tw["twin_dprec"] = partner["precursor_mz"].to_numpy() - tw["precursor_mz"].to_numpy()
    tw["twin_same_ce"] = partner["collision_energy_orig"].to_numpy() == tw["collision_energy_orig"].to_numpy()
    tw["twin_same_inst"] = partner["instrument_type"].to_numpy() == tw["instrument_type"].to_numpy()
    tw["twin_dspec"] = (partner["bmdms_spec_id"].to_numpy() - tw["bmdms_spec_id"].to_numpy())
    tw = tw.set_index("index")
    for col in ["twin_label", "twin_dprec", "twin_same_ce", "twin_same_inst", "twin_dspec"]:
        scans[col] = tw[col]
    scans.drop(columns=["ms2_mzs", "ms2_normalized_intensities", "key"]).to_parquet(
        args.out_dp / "scans.parquet", index=False
    )

    # 4. Summary by label.
    multi = scans[scans["num_peaks"] >= args.min_peaks]
    twin = multi["twin_cos"] >= args.min_cosine
    summary = {"min_cosine": args.min_cosine, "min_peaks": args.min_peaks, "by_label": {}}
    for label, grp in multi.groupby("label"):
        t = grp[grp["twin_cos"] >= args.min_cosine]
        summary["by_label"][label] = {
            "scans": int(len(grp)),
            "with_cross_formula_twin": int(len(t)),
            "share": round(float(len(t) / max(len(grp), 1)), 4),
            "twin_partner_label": t["twin_label"].value_counts().to_dict(),
            "abs_dprec_quantiles_Da": t["twin_dprec"].abs().quantile([0.25, 0.5, 0.75, 0.9]).round(4).to_dict(),
            "dprec_within_1p5Da": round(float((t["twin_dprec"].abs() <= 1.5).mean()), 4) if len(t) else None,
            "same_ce": round(float(t["twin_same_ce"].mean()), 4) if len(t) else None,
            "same_instrument": round(float(t["twin_same_inst"].mean()), 4) if len(t) else None,
            "abs_dspec_quantiles": t["twin_dspec"].abs().quantile([0.25, 0.5, 0.75]).to_dict(),
            "dspec_within_10": round(float((t["twin_dspec"].abs() <= 10).mean()), 4) if len(t) else None,
        }
    # Clusters of mutual twins: how many distinct foreign spectra explain the noise twins?
    noise_tw = multi[(multi["label"] == "noise") & twin]
    summary["noise_twin_molecules"] = int(noise_tw["inchikey14"].nunique())
    summary["noise_twin_partner_molecules"] = int(scans.loc[noise_tw["twin_row"].astype(int), "inchikey14"].nunique())
    summary["scans_lt_min_peaks"] = scans[scans["num_peaks"] < args.min_peaks]["label"].value_counts().to_dict()
    summary["dspec_random_pair_median"] = float(
        np.nanmedian(np.abs(np.random.default_rng(0).permutation(scans["bmdms_spec_id"].to_numpy()) - scans["bmdms_spec_id"].to_numpy()))
    )
    (args.out_dp / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("done: %s", json.dumps(summary["by_label"], default=str)[:2000])


if __name__ == "__main__":
    main()
