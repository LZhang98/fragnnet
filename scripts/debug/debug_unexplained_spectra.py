"""Find NIST20 spectra where the fragmentation DAG explains nothing (wrecall ≈ 0).

A spectrum is "unexplained" when no DAG fragment m/z matches any true peak
within tolerance, giving wrecall ≈ 0. These are candidates for fragmentation
coverage improvements.

Usage:
    python scripts/debug/debug_unexplained_spectra.py
    python scripts/debug/debug_unexplained_spectra.py --wrecall_thresh 0.05 --top_n 100
    python scripts/debug/debug_unexplained_spectra.py --frag_dp data/frag/nist23_ma_mi_d3_h4_isoFalse
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find NIST20 spectra unexplained by the DAG")
    parser.add_argument(
        "--spec_fp",
        type=str,
        default="data/proc/nist20/spec_df.pkl",
        help="Path to spec_df.pkl",
    )
    parser.add_argument(
        "--mol_fp",
        type=str,
        default="data/proc/nist20/mol_df.pkl",
        help="Path to mol_df.pkl",
    )
    parser.add_argument(
        "--frag_dp",
        type=str,
        default="data/frag/nist20_ma_mi_d3_h4_isoFalse_cmf",
        help="Path to frag directory containing spec_stats_df.pkl",
    )
    parser.add_argument(
        "--wrecall_col",
        type=str,
        default="wrecall_0.01_h4",
        help="wrecall column to filter on (e.g. wrecall_0.01_h4, wrecall_10ppm_h4)",
    )
    parser.add_argument(
        "--wrecall_thresh",
        type=float,
        default=0.01,
        help="Spectra with wrecall <= this threshold are considered unexplained",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Number of worst spectra to print",
    )
    parser.add_argument(
        "--save_fp",
        type=str,
        default=None,
        help="If set, save unexplained spectra DataFrame as CSV to this path",
    )
    parser.add_argument(
        "--prec_types",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific precursor types (e.g. [M+H]+ [M-H]-). Default: all.",
    )
    parser.add_argument(
        "--inst_types",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific instrument types. Default: all.",
    )
    return parser.parse_args()


def load_data(
    spec_fp: str,
    mol_fp: str,
    frag_dp: str,
    wrecall_col: str,
) -> pd.DataFrame:
    """Load and merge spec, mol, and DAG stats DataFrames.

    Args:
        spec_fp: Path to spec_df.pkl.
        mol_fp: Path to mol_df.pkl.
        frag_dp: Path to frag directory.
        wrecall_col: wrecall column name to keep.

    Returns:
        Merged DataFrame with one row per spectrum.
    """
    log.info("Loading spec_df from %s", spec_fp)
    spec_df = pd.read_pickle(spec_fp)

    log.info("Loading mol_df from %s", mol_fp)
    mol_df = pd.read_pickle(mol_fp)[["mol_id", "smiles", "inchikey_s", "formula", "exact_mw"]]

    stats_fp = Path(frag_dp) / "spec_stats_df.pkl"
    log.info("Loading spec_stats_df from %s", stats_fp)
    stats_df = pd.read_pickle(stats_fp)

    if wrecall_col not in stats_df.columns:
        available = [c for c in stats_df.columns if c.startswith("wrecall")]
        raise ValueError(
            f"Column '{wrecall_col}' not found. Available wrecall columns:\n  "
            + "\n  ".join(available)
        )

    keep_cols = ["spec_id", "mol_id", wrecall_col, "dag_num_nodes", "dag_num_edges", "num_formulae"]
    stats_slim = stats_df[keep_cols].copy()

    merged = spec_df.merge(stats_slim, on=["spec_id", "mol_id"], how="inner")
    merged = merged.merge(mol_df, on="mol_id", how="left")

    log.info(
        "Merged: %d spectra (%.1f%% of spec_df covered by DAG stats)",
        len(merged),
        100 * len(merged) / len(spec_df),
    )
    return merged


def report(df: pd.DataFrame, wrecall_col: str, wrecall_thresh: float, top_n: int) -> pd.DataFrame:
    """Print summary statistics and return unexplained subset.

    Args:
        df: Merged DataFrame with wrecall column.
        wrecall_col: wrecall column name.
        wrecall_thresh: Threshold below which a spectrum is "unexplained".
        top_n: Number of worst spectra to print.

    Returns:
        DataFrame of unexplained spectra sorted by wrecall ascending.
    """
    col = df[wrecall_col]
    print(f"\n{'='*60}")
    print(f"  wrecall column : {wrecall_col}")
    print(f"  total spectra  : {len(df):,}")
    print(f"  mean           : {col.mean():.4f}")
    print(f"  median         : {col.median():.4f}")
    print(f"  p10            : {np.percentile(col, 10):.4f}")
    print(f"  p1             : {np.percentile(col, 1):.4f}")
    print(f"  wrecall == 0   : {(col == 0).sum():,} ({100*(col==0).mean():.2f}%)")
    print(f"  wrecall <= {wrecall_thresh:.3f} : {(col <= wrecall_thresh).sum():,} ({100*(col<=wrecall_thresh).mean():.2f}%)")
    print(f"{'='*60}\n")

    unexplained = df[col <= wrecall_thresh].sort_values(wrecall_col)

    # Breakdown by prec_type × inst_type
    if "prec_type" in unexplained.columns and "inst_type" in unexplained.columns:
        print("By prec_type × inst_type (count):")
        ct = pd.crosstab(unexplained["prec_type"], unexplained["inst_type"], margins=True)
        print(ct.to_string())
        print()
        print("By prec_type × inst_type (median wrecall among unexplained):")
        med = (
            unexplained.groupby(["prec_type", "inst_type"])[wrecall_col]
            .median()
            .unstack(fill_value=float("nan"))
            .round(4)
        )
        print(med.to_string())
        print()

    display_cols = [
        c
        for c in [
            "spec_id",
            "dset",
            "dset_spec_id",
            "mol_id",
            "smiles",
            "formula",
            "prec_type",
            "inst_type",
            "frag_mode",
            "prec_mz",
            wrecall_col,
            "dag_num_nodes",
            "num_formulae",
        ]
        if c in unexplained.columns
    ]
    print(f"Worst {min(top_n, len(unexplained))} spectra (lowest {wrecall_col}):")
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(unexplained[display_cols].head(top_n).to_string(index=False))

    return unexplained


def main() -> None:
    args = get_args()

    df = load_data(args.spec_fp, args.mol_fp, args.frag_dp, args.wrecall_col)

    if args.prec_types is not None:
        before = len(df)
        df = df[df["prec_type"].isin(args.prec_types)]
        log.info("prec_type filter: %d → %d spectra", before, len(df))

    if args.inst_types is not None:
        before = len(df)
        df = df[df["inst_type"].isin(args.inst_types)]
        log.info("inst_type filter: %d → %d spectra", before, len(df))

    unexplained = report(df, args.wrecall_col, args.wrecall_thresh, args.top_n)

    if args.save_fp:
        out_path = Path(args.save_fp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        unexplained.drop(columns=["peaks", "mol"], errors="ignore").to_csv(out_path, index=False)
        log.info("Saved %d unexplained spectra to %s", len(unexplained), out_path)


if __name__ == "__main__":
    main()
