"""Plot the per-spectrum DAG oracle cosine distribution and dump the low-ceiling tail.

Consumes the per-spectrum CSV produced by ``scripts/debug/oracle/dag_oracle_scores.py`` (which
holds ``cos_hun_opt`` -- the best cosine any model could reach given only the DAG's
fragment coverage). Produces a by-prec_type distribution figure and a CSV of the
"low-ceiling" spectra (oracle cos below a threshold), merged with SMILES/formula so the
tail can be fed to subclass analysis.

Usage:
    python scripts/debug/oracle/plot_oracle_cos_dist.py \\
        --oracle_csv data/eval/oracle_cos_nist20_mbfs/oracle_cos.csv \\
        --out_dp data/eval/oracle_cos_nist20_mbfs \\
        --prec_types "[M+H]+" "[M-H]-" "[M+Na]+" "[M+Cl]-" \\
        --tail_thresh 0.9
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oracle_dataset_utils import resolve_dsets_for_spec_fp, resolve_metric_col

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COL = "cos_hun_opt"
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DAG oracle cosine distribution")
    parser.add_argument("--oracle_csv", required=True, help="Per-spectrum CSV from dag_oracle_scores.py")
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter oracle rows to these dataset labels."
    )
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--out_dp", required=True, help="Output directory for figure + tail CSV")
    parser.add_argument(
        "--prec_types",
        nargs="+",
        default=["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"],
        help="Precursor types to include, in plot order.",
    )
    parser.add_argument(
        "--tail_thresh",
        type=float,
        default=0.9,
        help="Spectra with oracle cos below this go into the low-ceiling tail dump.",
    )
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)
    return args


def plot_distribution(df: pd.DataFrame, prec_types: list[str], out_fp: Path, col: str) -> None:
    """Plot histogram + CDF of oracle cosine, one series per precursor type.

    Args:
        df: Per-spectrum DataFrame with columns ``col`` and ``prec_type``.
        prec_types: Precursor types to plot, in order.
        out_fp: Path to save the PNG figure.
    """
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 51)

    for pt, color in zip(prec_types, COLORS):
        vals = df.loc[df["prec_type"] == pt, col].to_numpy()
        if len(vals) == 0:
            continue
        label = (
            f"{pt}  n={len(vals):,}\n"
            f"  mean={vals.mean():.3f} med={np.median(vals):.3f}  "
            f"<0.9:{(vals < 0.9).mean() * 100:.1f}%"
        )
        ax_h.hist(vals, bins=bins, histtype="step", lw=2, color=color, label=label)
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax_c.plot(xs, ys, color=color, lw=2, label=pt)

    allv = df[col].to_numpy()
    ax_h.set(
        xlabel=f"oracle cosine ({col}, 10 ppm)",
        ylabel="num spectra",
        yscale="log",
        title=f"DAG oracle cos ceiling by prec_type\noverall mean={allv.mean():.3f} med={np.median(allv):.4f}",
    )
    ax_h.legend(fontsize=8, loc="upper left")
    ax_c.set(
        xlabel=f"oracle cosine ({col})",
        ylabel="cumulative fraction",
        title="CDF (lower-left = poorly-covered tail)",
    )
    ax_c.legend(fontsize=9, loc="upper left")
    ax_c.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=130)
    log.info("Saved figure to %s", out_fp)


def main() -> None:
    args = get_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.oracle_csv)
    if args.dsets:
        spec_ids = pd.read_pickle(args.spec_fp)[["spec_id", "dset"]]
        spec_ids = spec_ids[spec_ids["dset"].isin(args.dsets)]
        df = df.merge(spec_ids, on="spec_id", how="inner")
    df = df[df["prec_type"].isin(args.prec_types)].reset_index(drop=True)
    col = resolve_metric_col(df, COL)

    mol_df = pd.read_pickle(args.mol_fp)[["mol_id", "smiles", "inchikey_s", "formula", "exact_mw"]]
    df = df.merge(mol_df, on="mol_id", how="left")

    plot_distribution(df, args.prec_types, out_dp / "oracle_cos_dist.png", col)

    # Per-prec_type summary
    summary = (
        df.groupby("prec_type")[col]
        .agg(count="count", mean="mean", median="median", frac_below_0p9=lambda s: (s < 0.9).mean())
        .reindex(args.prec_types)
        .round(4)
    )
    print("\nOracle cos by prec_type:")
    print(summary.to_string())

    # Low-ceiling tail dump (all requested adducts)
    tail = df[df[col] < args.tail_thresh].sort_values(col)
    tail_cols = [
        "spec_id", "mol_id", "prec_type", "inst_type", "frag_mode",
        "smiles", "formula", "inchikey_s", "exact_mw",
        col, "cos_hun_0p01da_opt", "n_true_peaks", "n_achievable_peaks", "dag_num_nodes",
    ]
    tail_cols = [c for c in tail_cols if c in tail.columns]
    tail_fp = out_dp / f"low_ceiling_tail_lt{args.tail_thresh}.csv"
    tail[tail_cols].to_csv(tail_fp, index=False)
    log.info(
        "Dumped %d low-ceiling spectra (cos < %.2f) to %s",
        len(tail), args.tail_thresh, tail_fp,
    )


if __name__ == "__main__":
    main()
