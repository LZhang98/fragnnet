"""Decompose the model's cos_hun gap into DAG-coverage vs intensity-prediction error.

For each test spectrum there are three quantities on the same cos_hun scale:

  - ``cos_actual``  : the trained model's cos_hun_inst_tol (from predictions_test.hdf5).
  - ``cos_oracle``  : the DAG-achievable ceiling -- true intensities placed on exactly the
                      true peaks the DAG can reach, zero elsewhere (dag_oracle_scores.score_dag,
                      column ``cos_hun_inst_tol_opt``). Same per-instrument tolerance as the model.
  - ``1.0``         : a perfect prediction.

The oracle is an upper bound on what any intensity assignment over the current DAG can score,
so the total shortfall ``1 - cos_actual`` splits cleanly into two non-negative parts:

    coverage_gap  = 1.0        - cos_oracle   # peaks the DAG never contains -> fixable only by
                                              #   a bigger/deeper DAG (depth, bond-mask, fan-out).
    intensity_gap = cos_oracle - cos_actual   # peaks the DAG HAS but the model mis-weights ->
                                              #   fixable by a better intensity model, no new chem.

The script reports which component dominates overall, per inst_type/frag_mode, and specifically
for the worst spectra (cos_actual < a threshold) -- answering "are bad predictions bad because
the DAG can't reach the peaks, or because we put the wrong intensities on peaks we already have?"

Usage:
    python scripts/eval/dag_oracle_decomposition.py \\
        --pred_hdf5 output/inference_eval/4ews8j6f/predictions_test.hdf5 \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --split test
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import h5py
import numpy as np
import pandas as pd

from oracle_dataset_utils import resolve_dsets_for_spec_fp

from fragnnet.utils import parallel_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF

# Reuse the oracle machinery from the sibling script rather than reimplementing it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_oracle_scores import score_dag  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
ORACLE_TIER = "1_above_prec"


def load_actual_cos(pred_hdf5: str) -> pd.DataFrame:
    """Load per-spectrum model cos_hun_inst_tol from a predictions HDF5.

    Args:
        pred_hdf5: Path to a predictions_test.hdf5 written by run_save_inference_eval.py.

    Returns:
        DataFrame with columns ``spec_id`` and ``cos_actual``.
    """
    with h5py.File(pred_hdf5, "r") as f:
        if "cos_hun_inst_tol" not in f:
            raise KeyError(
                f"{pred_hdf5} has no 'cos_hun_inst_tol' dataset; keys={list(f.keys())}"
            )
        spec_id = f["spec_id"][:]
        cos_actual = f["cos_hun_inst_tol"][:]
    return pd.DataFrame({"spec_id": spec_id, "cos_actual": cos_actual})


def load_test_spec_df(spec_fp: str, split_dp: str, split: str, dsets: list[str] | None) -> pd.DataFrame:
    """Load and filter spec_df to a split, matching dag_oracle_scores.main's filtering."""
    log.info("Loading %s", spec_fp)
    spec_df = pd.read_pickle(spec_fp)
    dsets = resolve_dsets_for_spec_fp(spec_fp, dsets)
    if dsets:
        spec_df = spec_df[spec_df["dset"].isin(dsets)]
    spec_df = spec_df[spec_df["prec_type"].isin(PREC_TYPE_TO_MASS_DIFF.keys())]
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]
    if split != "all":
        split_fp = os.path.join(split_dp, f"{split}_ids.csv")
        split_spec_ids = set(pd.read_csv(split_fp)["spec_id"])
        spec_df = spec_df[spec_df["spec_id"].isin(split_spec_ids)]
    return spec_df[["spec_id", "mol_id", "peaks", "prec_mz", "prec_type", "inst_type", "frag_mode"]]


def _fmt_gap_split(df: pd.DataFrame) -> str:
    """One-line summary: mean actual/oracle and the coverage-vs-intensity split of the gap."""
    cov = df["coverage_gap"].mean()
    inten = df["intensity_gap"].mean()
    total = cov + inten
    if total <= 0:
        return (
            f"n={len(df):>6d}  actual={df['cos_actual'].mean():.4f}  "
            f"oracle={df['cos_oracle'].mean():.4f}  gap~0"
        )
    return (
        f"n={len(df):>6d}  actual={df['cos_actual'].mean():.4f}  "
        f"oracle={df['cos_oracle'].mean():.4f}  gap={total:.4f}  "
        f"[coverage {cov:.4f} ({100 * cov / total:4.1f}%) | "
        f"intensity {inten:.4f} ({100 * inten / total:4.1f}%)]"
    )


def summarize(df: pd.DataFrame, bad_threshold: float) -> None:
    """Print the overall decomposition, per-stratum breakdowns, and a worst-spectra focus."""
    print(f"\n{'=' * 100}")
    print("DAG-coverage vs intensity-prediction decomposition of the cos_hun gap")
    print(f"{'=' * 100}")
    print(f"joined spectra: {len(df)}  (mols: {df['mol_id'].nunique()})")
    print(f"\nOVERALL   {_fmt_gap_split(df)}")

    # Fraction of the *total* gap mass (summed, not per-spectrum-averaged) from each component.
    cov_sum = df["coverage_gap"].sum()
    inten_sum = df["intensity_gap"].sum()
    tot_sum = cov_sum + inten_sum
    print(
        f"\nTotal gap mass:  coverage {100 * cov_sum / tot_sum:.1f}%  |  "
        f"intensity {100 * inten_sum / tot_sum:.1f}%   "
        f"(the model would gain {inten_sum / len(df):.4f} mean cos from perfect "
        f"intensities on the CURRENT DAG)"
    )

    print("\nBy inst_type / frag_mode:")
    for (inst, mode), g in df.groupby(["inst_type", "frag_mode"], observed=True):
        if len(g) < 50:
            continue
        print(f"  {str(inst):6s} {str(mode):5s}  {_fmt_gap_split(g)}")

    print("\nBy adduct (prec_type):")
    for ptype, g in df.groupby("prec_type", observed=True):
        if len(g) < 50:
            continue
        print(f"  {str(ptype):8s}  {_fmt_gap_split(g)}")

    # Worst-spectra focus: of the spectra the model scores badly, how many are bad because the
    # DAG can't reach the peaks (oracle also low) vs. reachable-but-mis-weighted (oracle high)?
    bad = df[df["cos_actual"] < bad_threshold]
    print(f"\n{'-' * 100}")
    print(f"WORST SPECTRA  (cos_actual < {bad_threshold}):  {len(bad)}  "
          f"({100 * len(bad) / len(df):.1f}% of all)")
    if len(bad) > 0:
        reachable = bad[bad["cos_oracle"] >= bad_threshold]
        print(
            f"  of these, oracle >= {bad_threshold} (peaks ARE reachable, pure intensity failure): "
            f"{len(reachable)} ({100 * len(reachable) / len(bad):.1f}%)"
        )
        print(
            f"  of these, oracle <  {bad_threshold} (DAG genuinely can't reach the peaks): "
            f"{len(bad) - len(reachable)} ({100 * (len(bad) - len(reachable)) / len(bad):.1f}%)"
        )
        print(f"  {_fmt_gap_split(bad)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred_hdf5", required=True, help="predictions_test.hdf5 with cos_hun_inst_tol.")
    parser.add_argument("--frag_dp", required=True, help="DAG dir the run was trained/evaluated on.")
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--bad_threshold", type=float, default=0.3, help="cos_actual below this = 'worst'.")
    parser.add_argument("--limit_mols", type=int, default=None, help="Subsample this many mols for a quick check.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=parallel_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", default=None, help="Optional CSV of the per-spectrum joined table.")
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)

    actual_df = load_actual_cos(args.pred_hdf5)
    log.info("Loaded %d actual cos values from %s", len(actual_df), args.pred_hdf5)

    spec_df = load_test_spec_df(args.spec_fp, args.split_dp, args.split, args.dsets)
    # Only score spectra we have predictions for (and vice versa) -- the join below is the intersection.
    spec_df = spec_df[spec_df["spec_id"].isin(set(actual_df["spec_id"]))]

    if args.limit_mols is not None:
        rng = np.random.default_rng(args.seed)
        mol_ids = spec_df["mol_id"].unique()
        keep = rng.choice(mol_ids, size=min(args.limit_mols, len(mol_ids)), replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(keep)]

    log.info("Scoring oracle on %d spectra / %d mols", len(spec_df), spec_df["mol_id"].nunique())
    oracle_df = score_dag(spec_df, args.frag_dp, n_jobs=args.n_jobs)
    tier_cols = {
        f"cos_hun_inst_tol_opt__{ORACLE_TIER}": "cos_oracle",
        f"n_true_peaks__{ORACLE_TIER}": "n_true_peaks",
        f"n_achievable_peaks_inst_tol__{ORACLE_TIER}": "n_achievable_peaks_inst_tol",
    }
    oracle_df = oracle_df[
        [
            "spec_id",
            "mol_id",
            *tier_cols.keys(),
            "inst_type",
            "frag_mode",
            "prec_type",
        ]
    ].rename(columns=tier_cols)

    df = oracle_df.merge(actual_df, on="spec_id", how="inner")
    log.info("Joined %d spectra (oracle x actual)", len(df))

    # Coverage vs intensity split. The oracle is an upper bound on the model; tiny numerical
    # violations (model slightly above oracle, e.g. from H-transfer budget mismatches) are
    # clipped to 0 so both components stay non-negative and sum to the total gap.
    df["coverage_gap"] = (1.0 - df["cos_oracle"]).clip(lower=0.0)
    df["intensity_gap"] = (df["cos_oracle"] - df["cos_actual"]).clip(lower=0.0)
    n_viol = int((df["cos_actual"] > df["cos_oracle"] + 1e-6).sum())
    if n_viol > 0:
        log.warning("%d/%d spectra have cos_actual > cos_oracle (clipped to 0 intensity_gap)",
                    n_viol, len(df))

    summarize(df, args.bad_threshold)

    if args.out_fp:
        df.to_csv(args.out_fp, index=False)
        log.info("Saved per-spectrum joined table to %s", args.out_fp)


if __name__ == "__main__":
    main()
