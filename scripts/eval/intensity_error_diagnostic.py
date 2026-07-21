"""Diagnose WHAT kind of intensity error the model makes on DAG-reachable peaks.

Today's dag_oracle_decomposition.py showed ~84% of the cos_hun gap is intensity error on
peaks the DAG already contains (not coverage). "Wrong intensities" is not one failure mode,
and each points at a different fix, so this script splits it three ways per spectrum, then
summarizes per instrument/frag-mode/adduct:

  1. RANK error -- are the peaks in the right relative order?
     Spearman correlation between true intensity and the model's matched predicted intensity,
     over the true peaks that have a predicted match. Low -> the model orders peaks wrong
     (points at objective/features, not the output layer).

  2. DYNAMIC-RANGE error -- is the predicted spectrum too flat or too peaky?
     Shannon spectral entropy (nats) of the full predicted vs. the full true spectrum, each
     renormalized to a probability distribution. pred_entropy - true_entropy > 0 means the
     model spreads mass too evenly (too flat); < 0 means over-concentrated (too peaky).

  3. DOMINANT-PEAK error -- does the model get the base peak right?
     base_peak_hit = the predicted base peak (max-intensity peak) falls within the same
     per-instrument m/z tolerance of the true base peak. cos_hun is dominated by the top
     peaks, so missing the base peak is expensive.

A true peak's predicted intensity is the sum of predicted intensities of all pred peaks within
tolerance (a peak's intensity is the sum of its contributing fragments), mirroring how the
peak-level metric aggregates. Only DAG-reachable true peaks enter the rank stat -- that is
exactly where the intensity gap lives.

Usage:
    python scripts/eval/intensity_error_diagnostic.py \\
        --pred_hdf5 output/inference_eval/4ews8j6f/predictions_test.hdf5 \\
        --split test
"""

from __future__ import annotations

import argparse
import logging
import os

import h5py
import numpy as np
import pandas as pd

from oracle_dataset_utils import resolve_dsets_for_spec_fp
import torch as th
from scipy.stats import entropy as scipy_entropy
from scipy.stats import spearmanr

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.spec_utils import calculate_match_mzs

# Reuse the per-instrument tolerance convention from the oracle script (FT 10ppm/200Da,
# QTOF 20ppm/750Da, else global 10ppm/200Da) so this diagnostic matches the cos_hun_inst_tol
# variant the decomposition used.
INST_TYPE_LOSS_TOL = {
    "FT": {"rel": 1.0e-5, "min_mz": 200.0},
    "QTOF": {"rel": 2.0e-5, "min_mz": 750.0},
}
TOLERANCE = 1.0e-5
TOLERANCE_MIN_MZ = 200.0
MIN_PEAKS_FOR_RANK = 3  # Spearman is meaningless below this many matched peaks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _tolerance_for_inst_type(inst_type: str) -> tuple[float, float]:
    cfg = INST_TYPE_LOSS_TOL.get(inst_type)
    return (cfg["rel"], cfg["min_mz"]) if cfg else (TOLERANCE, TOLERANCE_MIN_MZ)


def _diagnose_one(rec: dict) -> dict | None:
    """Compute rank/dynamic-range/dominant-peak diagnostics for one spectrum.

    Args:
        rec: Dict with pred_mzs, pred_ints, true_mzs, true_ints (np arrays) plus spec_id,
            mol_id, inst_type, frag_mode, prec_type.

    Returns:
        A result dict, or None if the spectrum has no usable peaks.
    """
    true_mzs = th.as_tensor(rec["true_mzs"], dtype=th.float32)
    true_ints = th.as_tensor(rec["true_ints"], dtype=th.float32)
    pred_mzs = th.as_tensor(rec["pred_mzs"], dtype=th.float32)
    pred_ints = th.as_tensor(rec["pred_ints"], dtype=th.float32)
    if true_mzs.numel() == 0 or pred_mzs.numel() == 0:
        return None

    tol, tol_min = _tolerance_for_inst_type(rec["inst_type"])
    match_mask = calculate_match_mzs(
        true_mzs, pred_mzs, tolerance=tol, relative=True, tolerance_min_mz=tol_min
    )
    match_mask = th.as_tensor(match_mask)  # [n_true, n_pred]

    # per true peak: predicted intensity = sum of matched pred peaks
    pred_agg = match_mask.to(th.float32) @ pred_ints
    matched_true = match_mask.any(dim=1)
    n_matched = int(matched_true.sum())

    # 1. RANK error -- Spearman over matched true peaks
    if n_matched >= MIN_PEAKS_FOR_RANK:
        t = true_ints[matched_true].numpy()
        p = pred_agg[matched_true].numpy()
        rho = spearmanr(t, p).correlation
        spearman = float(rho) if np.isfinite(rho) else np.nan
    else:
        spearman = np.nan

    # 2. DYNAMIC-RANGE error -- spectral entropy (nats), full spectra
    pred_entropy = float(scipy_entropy(pred_ints.numpy()))
    true_entropy = float(scipy_entropy(true_ints.numpy()))

    # 3. DOMINANT-PEAK error -- base peak within tolerance
    true_base_mz = float(true_mzs[int(th.argmax(true_ints))])
    pred_base_mz = float(pred_mzs[int(th.argmax(pred_ints))])
    allowed = tol * max(true_base_mz, tol_min)
    base_peak_hit = abs(true_base_mz - pred_base_mz) <= allowed

    # fraction of TRUE intensity mass sitting on DAG-reachable (matched) peaks -- context for
    # how much of the spectrum the rank stat even covers
    true_mass_matched = float(true_ints[matched_true].sum() / true_ints.sum())

    return {
        "spec_id": rec["spec_id"],
        "mol_id": rec["mol_id"],
        "inst_type": rec["inst_type"],
        "frag_mode": rec["frag_mode"],
        "prec_type": rec["prec_type"],
        "n_true_peaks": int(true_mzs.numel()),
        "n_matched_peaks": n_matched,
        "spearman": spearman,
        "pred_entropy": pred_entropy,
        "true_entropy": true_entropy,
        "entropy_diff": pred_entropy - true_entropy,
        "base_peak_hit": base_peak_hit,
        "true_mass_matched": true_mass_matched,
    }


def _diagnose_chunk(recs: list[dict]) -> list[dict]:
    out = []
    for rec in recs:
        r = _diagnose_one(rec)
        if r is not None:
            out.append(r)
    return out


def load_pred_spectra(pred_hdf5: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-spectrum predicted (m/z, intensity) arrays keyed by spec_id."""
    with h5py.File(pred_hdf5, "r") as f:
        spec_id = f["spec_id"][:]
        start = f["start_points"][:]
        end = f["end_points"][:]
        pred_mzs = f["pred_mzs"][:]
        pred_ints = f["pred_ints"][:]
    return {
        int(sid): (pred_mzs[s:e], pred_ints[s:e])
        for sid, s, e in zip(spec_id, start, end)
    }


def _fmt(df: pd.DataFrame, label: str) -> str:
    """One-line stratum summary of the three diagnostics."""
    rank = df["spearman"].dropna()
    return (
        f"{label:14s} n={len(df):>6d}  "
        f"spearman={rank.mean():.3f} (n={len(rank)})  "
        f"H_pred={df['pred_entropy'].mean():.3f} H_true={df['true_entropy'].mean():.3f} "
        f"dH={df['entropy_diff'].mean():+.3f}  "
        f"base_hit={df['base_peak_hit'].mean():.3f}  "
        f"true_mass_on_dag={df['true_mass_matched'].mean():.3f}"
    )


def summarize(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 108}")
    print("Intensity-error diagnostic (rank / dynamic-range / dominant-peak)")
    print(f"{'=' * 108}")
    print(f"spectra: {len(df)}  (mols: {df['mol_id'].nunique()})")
    print(
        "\nLegend: spearman = pred-vs-true intensity rank corr on matched peaks (1=perfect order); "
        "\n        dH = pred_entropy - true_entropy (>0 too flat, <0 too peaky); "
        "base_hit = base peak within tol; \n        true_mass_on_dag = fraction of true intensity on DAG-reachable peaks.\n"
    )
    print(_fmt(df, "OVERALL"))
    print("\nBy inst_type / frag_mode:")
    for (inst, mode), g in df.groupby(["inst_type", "frag_mode"], observed=True):
        if len(g) >= 50:
            print("  " + _fmt(g, f"{inst}/{mode}"))
    print("\nBy adduct:")
    for ptype, g in df.groupby("prec_type", observed=True):
        if len(g) >= 50:
            print("  " + _fmt(g, str(ptype)))

    # How often is each failure mode "bad"? Thresholds are descriptive, not tuned.
    print(f"\n{'-' * 108}")
    rank = df["spearman"].dropna()
    print(f"rank error:       {100 * (rank < 0.5).mean():.1f}% of spectra have spearman < 0.5")
    print(f"dynamic range:    {100 * (df['entropy_diff'] > 0).mean():.1f}% predicted TOO FLAT "
          f"(pred entropy > true); mean dH = {df['entropy_diff'].mean():+.3f} nats")
    print(f"dominant peak:    {100 * (~df['base_peak_hit']).mean():.1f}% MISS the base peak")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred_hdf5", required=True)
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument(
        "--min_int_frac", type=float, default=0.0,
        help="Drop peaks below this fraction of the base peak in BOTH pred and true before "
        "scoring (e.g. 0.01 = 1%% threshold). Default 0.0 = no threshold.",
    )
    parser.add_argument("--limit_mols", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=data_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", default=None, help="Optional per-spectrum CSV.")
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)

    log.info("Loading predicted spectra from %s", args.pred_hdf5)
    pred_by_sid = load_pred_spectra(args.pred_hdf5)

    log.info("Loading %s", args.spec_fp)
    spec_df = pd.read_pickle(args.spec_fp)
    if args.dsets:
        spec_df = spec_df[spec_df["dset"].isin(args.dsets)]
    spec_df = spec_df[spec_df["prec_type"].isin(PREC_TYPE_TO_MASS_DIFF.keys())]
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]
    if args.split != "all":
        split_fp = os.path.join(args.split_dp, f"{args.split}_ids.csv")
        split_ids = set(pd.read_csv(split_fp)["spec_id"])
        spec_df = spec_df[spec_df["spec_id"].isin(split_ids)]
    spec_df = spec_df[spec_df["spec_id"].isin(pred_by_sid.keys())]

    if args.limit_mols is not None:
        rng = np.random.default_rng(args.seed)
        mol_ids = spec_df["mol_id"].unique()
        keep = rng.choice(mol_ids, size=min(args.limit_mols, len(mol_ids)), replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(keep)]

    def _threshold(mzs: np.ndarray, ints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Drop peaks below min_int_frac of the spectrum's base peak (both pred and true)."""
        if args.min_int_frac <= 0.0 or ints.size == 0:
            return mzs, ints
        keep = ints >= args.min_int_frac * ints.max()
        return mzs[keep], ints[keep]

    recs = []
    for row in spec_df.itertuples(index=False):
        pred_mzs, pred_ints = pred_by_sid[int(row.spec_id)]
        true_mzs, true_ints = (np.array(a, dtype=np.float32) for a in zip(*row.peaks))
        pred_mzs, pred_ints = _threshold(pred_mzs, pred_ints)
        true_mzs, true_ints = _threshold(true_mzs, true_ints)
        recs.append(
            {
                "spec_id": int(row.spec_id),
                "mol_id": int(row.mol_id),
                "inst_type": row.inst_type,
                "frag_mode": row.frag_mode,
                "prec_type": row.prec_type,
                "pred_mzs": pred_mzs,
                "pred_ints": pred_ints,
                "true_mzs": true_mzs,
                "true_ints": true_ints,
            }
        )

    log.info("Diagnosing %d spectra", len(recs))
    n_chunks = max(1, args.n_jobs * 4)
    # par_apply calls par_func(*item), so each item is a 1-tuple wrapping one chunk (list).
    chunks = [(recs[i::n_chunks],) for i in range(n_chunks)]
    per_chunk = data_utils.par_apply(chunks, _diagnose_chunk, need_unpack=True, n_jobs=args.n_jobs)
    rows = [r for chunk in per_chunk for r in chunk]
    df = pd.DataFrame(rows)

    summarize(df)
    if args.out_fp:
        df.to_csv(args.out_fp, index=False)
        log.info("Saved per-spectrum diagnostics to %s", args.out_fp)


if __name__ == "__main__":
    main()
