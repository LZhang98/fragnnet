"""Flag spectra that dip out of the monotone CE trend (per-group isotonic-residual detector).

Oracle cos declines with collision energy: within-molecule Spearman(ce, cos_hun_inst_tol_opt) is
negative for ~82% of groups (mean rho -0.53). The expected shape is therefore *monotone
non-increasing* in CE. This script flags spectra that sit well BELOW that trend -- a local dip a
level threshold cannot separate from the genuine gradual high-CE decline.

For each group ``(mol_id, prec_type, inst_type[, frag_mode])`` with at least ``--min_ce_bins``
distinct CE values, it fits a non-increasing isotonic regression to the per-CE metric envelope
(max metric at each CE) and scores every spectrum by its residual below the fit
(``iso_resid = fit - metric``). The residual is threshold-independent, so a sweep over
``--resid_thresholds`` is just a filter -- the per-spectrum table is written once and re-used.

A dip that ALSO has a well-explained sibling at another CE (``best_diff_ce >= --normal_thresh``) is
corroborated as a per-spectrum data-quality artifact (the DAG is CE-independent, so a good sibling
proves coverage is fine). The sweep reports both the flagged count and the corroborated subset.

Usage:
    python scripts/eval/oracle_ce_trend_anomaly.py \\
        --oracle_csv data/eval/oracle_cos_nist20_default/oracle_cos_default.csv \\
        --out_dp data/eval/oracle_cos_nist20_default \\
        --resid_thresholds 0.2 0.3 0.4 0.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from oracle_dataset_utils import resolve_dsets_for_spec_fp
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_ce_sibling_trend_audit import (  # noqa: E402  (needs sys.path tweak first)
    add_ce_columns,
    add_sibling_features,
    resolve_metric_col,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--oracle_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_cos_default.csv",
        help="Per-spectrum oracle or recall CSV.",
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--out_dp", required=True)
    parser.add_argument(
        "--metric",
        default="cos_hun_inst_tol_opt",
        help="Metric column to audit. Tier-suffixed columns are resolved automatically.",
    )
    parser.add_argument(
        "--resid_thresholds",
        nargs="+",
        type=float,
        default=[0.2, 0.3, 0.4, 0.5],
        help="Isotonic-residual cutoffs to sweep (flag if fit - metric >= threshold).",
    )
    parser.add_argument(
        "--normal_thresh",
        type=float,
        default=0.90,
        help="A different-CE sibling >= this corroborates a dip as a data-quality artifact.",
    )
    parser.add_argument(
        "--min_ce_bins",
        type=int,
        default=3,
        help="Minimum distinct CE values in a group to fit a trend.",
    )
    parser.add_argument("--inst_types", nargs="+", default=["FT", "QTOF"])
    parser.add_argument(
        "--prec_types", nargs="+", default=["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"]
    )
    parser.add_argument(
        "--ignore_frag_mode",
        action="store_true",
        help="Group only by mol_id/adduct/instrument. Default also controls for frag_mode.",
    )
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)
    return args


def add_isotonic_residual(
    df: pd.DataFrame, group_cols: list[str], metric_col: str, min_ce_bins: int
) -> pd.DataFrame:
    """Attach the per-spectrum residual below a non-increasing isotonic CE trend.

    The trend is fit to the per-CE metric envelope (max metric at each distinct CE) so a single bad
    duplicate at one CE cannot drag the expected trend down. ``iso_resid = fit(ce) - metric`` is
    positive for spectra below the trend (dips) and ~0 on-trend.

    Args:
        df: Per-spectrum DataFrame with ``ce_value``, ``metric_col`` and ``group_cols``.
        group_cols: Columns defining a shared-DAG CE series.
        metric_col: Metric to fit the trend on.
        min_ce_bins: Minimum distinct CE values required to fit a group's trend.

    Returns:
        Copy of ``df`` with added ``iso_fit``, ``iso_resid`` and ``n_ce_bins`` columns. Rows in
        groups with fewer than ``min_ce_bins`` distinct CEs get NaN residuals.
    """
    df = df.copy()
    iso_fit = np.full(len(df), np.nan)
    n_ce_bins = np.zeros(len(df), dtype=int)
    ce_all = df["ce_value"].to_numpy()
    val_all = df[metric_col].to_numpy()

    for _, gidx in df.groupby(group_cols, observed=True).indices.items():
        g_ce = ce_all[gidx]
        g_val = val_all[gidx]
        finite = np.isfinite(g_ce) & np.isfinite(g_val)
        n_bins = len(np.unique(g_ce[finite]))
        n_ce_bins[gidx] = n_bins
        if n_bins < min_ce_bins:
            continue
        # Fit on the per-CE envelope (best metric at each CE), then predict at every spectrum's CE.
        ce_u, inv = np.unique(g_ce[finite], return_inverse=True)
        env = np.full(len(ce_u), -np.inf)
        np.maximum.at(env, inv, g_val[finite])
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
        ir.fit(ce_u, env)
        iso_fit[gidx] = ir.predict(g_ce)

    df["n_ce_bins"] = n_ce_bins
    df["iso_fit"] = iso_fit
    df["iso_resid"] = df["iso_fit"] - df[metric_col]
    return df


def sweep_thresholds(
    scored: pd.DataFrame, thresholds: list[float], normal_thresh: float
) -> pd.DataFrame:
    """Count flagged and sibling-corroborated dips at each residual threshold."""
    fittable = scored[scored["iso_resid"].notna()]
    n_total = len(fittable)
    rows = []
    for t in thresholds:
        flagged = fittable[fittable["iso_resid"] >= t]
        corrob = flagged[flagged["best_diff_ce"] >= normal_thresh]
        rows.append(
            {
                "resid_thresh": t,
                "n_flagged": len(flagged),
                "pct_of_fittable": 100.0 * len(flagged) / n_total if n_total else 0.0,
                "n_mols": flagged["mol_id"].nunique(),
                "n_corroborated": len(corrob),
                "pct_corroborated": 100.0 * len(corrob) / len(flagged) if len(flagged) else 0.0,
                "median_metric": flagged[scored.attrs["metric_col"]].median(),
                "median_resid": flagged["iso_resid"].median(),
            }
        )
    return pd.DataFrame(rows)


def plot_sweep(sweep: pd.DataFrame, out_fp: Path) -> None:
    """Plot flagged / corroborated counts and corroboration rate vs residual threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(sweep["resid_thresh"], sweep["n_flagged"], "-o", color=COLORS[0], label="flagged")
    ax.plot(
        sweep["resid_thresh"],
        sweep["n_corroborated"],
        "-s",
        color=COLORS[2],
        label="corroborated (sibling >= normal)",
    )
    ax.set_xlabel("isotonic-residual threshold")
    ax.set_ylabel("# spectra")
    ax.set_title("Trend-dip flags vs threshold")
    ax.legend()

    ax = axes[1]
    ax.plot(sweep["resid_thresh"], sweep["pct_corroborated"], "-o", color=COLORS[4])
    ax.set_xlabel("isotonic-residual threshold")
    ax.set_ylabel("% flagged with good different-CE sibling")
    ax.set_ylim(0, 100)
    ax.set_title("Corroboration rate")

    fig.tight_layout()
    fig.savefig(out_fp, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Fit per-group CE trends, score residuals, and sweep the flag threshold."""
    args = get_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    oracle = pd.read_csv(args.oracle_csv)
    spec = pd.read_pickle(args.spec_fp)[["spec_id", "dset", "ace", "nce"]]
    if args.dsets:
        spec = spec[spec["dset"].isin(args.dsets)]
    df = oracle.merge(spec, on="spec_id", how="left")
    df = df[df["dset"].notna()].copy()
    df = df[df["prec_type"].isin(args.prec_types) & df["inst_type"].isin(args.inst_types)].copy()
    if df.empty:
        raise ValueError("Scope filter (inst_types / prec_types) matched no rows.")
    df = add_ce_columns(df)

    group_cols = ["mol_id", "prec_type", "inst_type"]
    if not args.ignore_frag_mode:
        group_cols.append("frag_mode")

    metric_col = resolve_metric_col(df, args.metric)
    # best_diff_ce (the corroborating good sibling) comes from the shared audit machinery.
    df = add_sibling_features(df, group_cols, metric_col)
    df = add_isotonic_residual(df, group_cols, metric_col, args.min_ce_bins)
    df.attrs["metric_col"] = metric_col

    fittable = df[df["iso_resid"].notna()]
    log.info(
        "Scope: %d spectra (%d in groups with >=%d CE bins), %d molecules; grouping by %s",
        len(df),
        len(fittable),
        args.min_ce_bins,
        df["mol_id"].nunique(),
        ", ".join(group_cols),
    )

    # Per-spectrum residual table (threshold-independent) -- apply any cutoff downstream.
    per_cols = [
        "spec_id",
        *group_cols,
        "ce_key",
        "ace",
        "nce",
        "ce_value",
        metric_col,
        "iso_fit",
        "iso_resid",
        "n_ce_bins",
        "best_lower_ce",
        "best_higher_ce",
        "best_diff_ce",
        "n_true_peaks",
        "n_achievable_peaks_inst_tol",
    ]
    per_cols = [c for c in per_cols if c in fittable.columns]
    per_fp = out_dp / "oracle_ce_trend_anomaly_scores.csv"
    fittable[per_cols].sort_values("iso_resid", ascending=False).to_csv(per_fp, index=False)

    sweep = sweep_thresholds(df, args.resid_thresholds, args.normal_thresh)
    sweep_fp = out_dp / "oracle_ce_trend_anomaly_sweep.csv"
    sweep.to_csv(sweep_fp, index=False)
    plot_fp = out_dp / "oracle_ce_trend_anomaly_sweep.png"
    plot_sweep(sweep, plot_fp)

    log.info("Wrote %s (%d scored spectra)", per_fp, len(fittable))
    log.info("Wrote %s", sweep_fp)
    log.info("Wrote %s", plot_fp)
    log.info("\n%s", sweep.to_string(index=False))


if __name__ == "__main__":
    main()
