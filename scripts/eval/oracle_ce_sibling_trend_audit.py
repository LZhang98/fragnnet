"""Find low-oracle spectra contradicted by comparable CE siblings.

Different-CE comparisons describe CE-dependent DAG coverage but do not by themselves prove bad
data: a shallow DAG can explain low-energy fragments while missing legitimate high-energy products.
The conservative human-review candidates are therefore limited to low spectra with either a normal
replicate at the same CE or normal siblings on both sides of an interior CE dip.

The script is metric-agnostic. It defaults to ``cos_hun_inst_tol_opt`` from
``dag_oracle_scores.py`` and also accepts any other per-spectrum metric columns, including
``wrecall_*`` columns when present in the input CSV.

Usage:
    python scripts/eval/oracle_ce_sibling_trend_audit.py \\
        --oracle_csv data/eval/oracle_cos_nist20_default/oracle_cos_default.csv \\
        --out_dp data/eval/oracle_cos_nist20_default
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from oracle_dataset_utils import resolve_dsets_for_spec_fp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_TIER = "1_above_prec"


def resolve_metric_col(df: pd.DataFrame, base: str, tier: str = DEFAULT_TIER) -> str:
    """Resolve a metric column from current tier-suffixed or legacy unsuffixed oracle CSVs."""
    suffixed = f"{base}__{tier}"
    if suffixed in df.columns:
        return suffixed
    if base in df.columns:
        return base
    raise KeyError(f"Missing column {base!r} or {suffixed!r}")


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
        "--metrics",
        nargs="+",
        default=["cos_hun_inst_tol_opt"],
        help="Metric columns to audit. Tier-suffixed columns are resolved automatically.",
    )
    parser.add_argument("--low_thresh", type=float, default=0.30)
    parser.add_argument("--normal_thresh", type=float, default=0.90)
    parser.add_argument(
        "--drop_thresh",
        type=float,
        default=0.50,
        help="Minimum best-different-CE minus current metric gap for the trend_drop flag.",
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
    if not 0.0 <= args.low_thresh < args.normal_thresh <= 1.0:
        raise ValueError("Require 0 <= low_thresh < normal_thresh <= 1")
    if args.drop_thresh < 0.0:
        raise ValueError("drop_thresh must be non-negative")
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)
    return args


def add_ce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add comparable CE value/key columns from ace/nce.

    FT spectra use NCE when available because NCE is the intended comparable scale there.
    Other instruments use ACE first, then fall back to NCE.
    """
    df = df.copy()
    use_nce = df["inst_type"].eq("FT") & df["nce"].notna()
    df["ce_value"] = df["ace"].where(df["ace"].notna(), df["nce"])
    df.loc[use_nce, "ce_value"] = df.loc[use_nce, "nce"]
    df["ce_source"] = np.where(
        use_nce,
        "nce",
        np.where(df["ace"].notna(), "ace", np.where(df["nce"].notna(), "nce", "none")),
    )
    df["ce_bin"] = df["ce_value"].round(6)
    df["ce_key"] = np.where(
        df["ce_bin"].notna(),
        df["ce_source"] + "=" + df["ce_bin"].astype(str),
        "ce=NA",
    )
    return df


def add_sibling_features(df: pd.DataFrame, group_cols: list[str], metric_col: str) -> pd.DataFrame:
    """Attach same-, lower-, higher-, and different-CE sibling scores for one metric."""
    # Work at CE-bin level first so duplicate spectra at the same CE do not become siblings.
    # Keep ACE and NCE series separate: their numeric values are not directly comparable.
    sibling_group_cols = group_cols + ["ce_source"]
    ce_stats = (
        df.dropna(subset=["ce_bin", metric_col])
        .groupby(sibling_group_cols + ["ce_bin"], observed=True)[metric_col]
        .max()
        .reset_index(name="ce_metric_max")
    )
    sibling_rows = []
    for _, g in ce_stats.groupby(sibling_group_cols, observed=True):
        g = g.sort_values("ce_bin")
        ce_scores = g["ce_metric_max"].to_numpy()
        ce_bins = g["ce_bin"].to_numpy()

        lower = [ce_scores[:i].max() if i > 0 else np.nan for i in range(len(g))]
        higher = [
            ce_scores[i + 1 :].max() if i + 1 < len(g) else np.nan for i in range(len(g))
        ]
        group_vals = {col: g[col].iloc[0] for col in sibling_group_cols}
        for ce_bin, lo, hi in zip(ce_bins, lower, higher):
            sibling_rows.append(
                {**group_vals, "ce_bin": ce_bin, "best_lower_ce": lo, "best_higher_ce": hi}
            )

    sibling_cols = sibling_group_cols + [
        "ce_bin",
        "best_lower_ce",
        "best_higher_ce",
    ]
    sibling_df = pd.DataFrame(sibling_rows, columns=sibling_cols)
    out = df.merge(sibling_df, on=sibling_group_cols + ["ce_bin"], how="left", validate="many_to_one")

    out["best_same_ce_other"] = np.nan
    same_ce_cols = sibling_group_cols + ["ce_bin"]
    finite = out["ce_bin"].notna() & out[metric_col].notna()
    for _, index_labels in out.loc[finite].groupby(same_ce_cols, observed=True).groups.items():
        positions = out.index.get_indexer(index_labels)
        if len(positions) < 2:
            continue
        values = out.iloc[positions][metric_col].to_numpy()
        order = np.argsort(values)
        best_pos = order[-1]
        second_best = values[order[-2]]
        best = values[best_pos]
        out.iloc[positions, out.columns.get_loc("best_same_ce_other")] = best
        out.iat[positions[best_pos], out.columns.get_loc("best_same_ce_other")] = second_best

    out["best_diff_ce"] = out[["best_lower_ce", "best_higher_ce"]].max(axis=1, skipna=True)
    return out


def audit_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
    low_thresh: float,
    normal_thresh: float,
    drop_thresh: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-spectrum flags and summary rows for one metric."""
    scored = add_sibling_features(df, group_cols, metric_col)
    metric = scored[metric_col]
    low = metric < low_thresh
    scored["flag_low_metric"] = low
    # Legacy directional context columns. These do not determine flag_review_candidate.
    scored["flag_low_ce_bad_high_ce_ok"] = low & scored["best_higher_ce"].ge(normal_thresh)
    scored["flag_high_ce_bad_low_ce_ok"] = low & scored["best_lower_ce"].ge(normal_thresh)
    scored["flag_any_diff_ce_ok"] = low & scored["best_diff_ce"].ge(normal_thresh)
    scored["flag_trend_drop"] = low & ((scored["best_diff_ce"] - metric) >= drop_thresh)
    scored["flag_same_ce_replicate_contradiction"] = low & scored["best_same_ce_other"].ge(
        normal_thresh
    )
    scored["flag_interior_ce_dip"] = (
        low
        & scored["best_lower_ce"].ge(normal_thresh)
        & scored["best_higher_ce"].ge(normal_thresh)
    )
    scored["flag_review_candidate"] = scored[
        ["flag_same_ce_replicate_contradiction", "flag_interior_ce_dip"]
    ].any(axis=1)

    rows = []
    low_df = scored[low]
    for label, mask in {
        "low_all": low,
        "low_ce_with_normal_higher": scored["flag_low_ce_bad_high_ce_ok"],
        "high_ce_with_normal_lower": scored["flag_high_ce_bad_low_ce_ok"],
        "any_diff_ce_ok": scored["flag_any_diff_ce_ok"],
        "trend_drop": scored["flag_trend_drop"],
        "same_ce_replicate_contradiction": scored["flag_same_ce_replicate_contradiction"],
        "interior_ce_dip": scored["flag_interior_ce_dip"],
        "review_candidate": scored["flag_review_candidate"],
    }.items():
        sub = scored[mask]
        rows.append(
            {
                "metric": metric_col,
                "flag": label,
                "n_spectra": len(sub),
                "n_mols": sub["mol_id"].nunique(),
                "pct_of_low": 100.0 * len(sub) / len(low_df) if len(low_df) else 0.0,
                "median_metric": sub[metric_col].median(),
                "median_best_diff_ce": sub["best_diff_ce"].median(),
            }
        )
    return scored, pd.DataFrame(rows)


def summarize_by_inst_adduct(scored_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Summarize low-metric sibling flags by instrument/adduct."""
    low = scored_df[scored_df["flag_low_metric"]].copy()
    if low.empty:
        return pd.DataFrame()
    return (
        low.groupby(["inst_type", "prec_type"], observed=True)
        .agg(
            metric=(metric_col, lambda _s: metric_col),
            n_low=("spec_id", "size"),
            n_mols=("mol_id", "nunique"),
            low_ce_with_normal_higher=("flag_low_ce_bad_high_ce_ok", "sum"),
            high_ce_with_normal_lower=("flag_high_ce_bad_low_ce_ok", "sum"),
            any_diff_ce_ok=("flag_any_diff_ce_ok", "sum"),
            trend_drop=("flag_trend_drop", "sum"),
            same_ce_replicate_contradiction=("flag_same_ce_replicate_contradiction", "sum"),
            interior_ce_dip=("flag_interior_ce_dip", "sum"),
            review_candidate=("flag_review_candidate", "sum"),
            median_metric=(metric_col, "median"),
            median_best_diff_ce=("best_diff_ce", "median"),
        )
        .reset_index()
        .sort_values("n_low", ascending=False)
    )


def plot_by_inst_adduct(by_group_df: pd.DataFrame, out_fp: Path) -> None:
    """Plot low-spectrum CE sibling flags by adduct and instrument."""
    if by_group_df.empty:
        log.warning("No grouped rows to plot; skipping %s", out_fp)
        return

    df = by_group_df.copy()
    df["label"] = df["inst_type"] + " " + df["prec_type"]
    df = df.sort_values(["inst_type", "n_low"], ascending=[True, False])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    x = np.arange(len(df))
    width = 0.28
    ax.bar(
        x - width / 2,
        df["same_ce_replicate_contradiction"],
        width=width,
        label="same-CE replicate contradiction",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        df["interior_ce_dip"],
        width=width,
        label="interior CE dip",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel("# flagged spectra")
    ax.set_title("CE-sibling flags by instrument/adduct")
    ax.legend(fontsize=8)

    ax = axes[1]
    pct = df["review_candidate"] / df["n_low"].replace(0, np.nan) * 100.0
    ax.barh(df["label"], pct, color="#8172B3", edgecolor="black", linewidth=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of low spectra meeting conservative review criteria")
    ax.set_title("Human-review candidates")
    for y, val in enumerate(pct):
        if np.isfinite(val):
            ax.text(min(val + 1.0, 98.0), y, f"{val:.0f}%", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_fp, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the CE sibling trend audit and write CSV outputs."""
    args = get_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    oracle = pd.read_csv(args.oracle_csv)
    spec = pd.read_pickle(args.spec_fp)[["spec_id", "dset", "ace", "nce", "ce_type"]]
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

    metric_cols = [resolve_metric_col(df, metric) for metric in args.metrics]
    log.info(
        "Scope: %d spectra, %d molecules, %d CE groups; grouping by %s",
        len(df),
        df["mol_id"].nunique(),
        df.groupby(group_cols, observed=True).ngroups,
        ", ".join(group_cols),
    )

    flagged_frames = []
    summary_frames = []
    by_group_frames = []
    for metric_col in metric_cols:
        scored, summary = audit_metric(
            df,
            group_cols,
            metric_col,
            args.low_thresh,
            args.normal_thresh,
            args.drop_thresh,
        )
        scored["audit_metric"] = metric_col
        by_group_frames.append(summarize_by_inst_adduct(scored, metric_col))
        flagged_frames.append(
            scored[scored["flag_review_candidate"]]
        )
        summary_frames.append(summary)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    by_group_df = pd.concat(by_group_frames, ignore_index=True)
    flagged_df = pd.concat(flagged_frames, ignore_index=True)

    summary_fp = out_dp / "oracle_ce_sibling_trend_summary.csv"
    by_group_fp = out_dp / "oracle_ce_sibling_trend_by_inst_adduct.csv"
    flagged_fp = out_dp / "oracle_ce_sibling_trend_flags.csv"
    plot_fp = out_dp / "oracle_ce_sibling_trend_by_inst_adduct.png"
    summary_df.to_csv(summary_fp, index=False)
    by_group_df.to_csv(by_group_fp, index=False)
    flagged_cols = [
        "audit_metric",
        "spec_id",
        *group_cols,
        "ce_key",
        "ace",
        "nce",
        *metric_cols,
        "best_lower_ce",
        "best_higher_ce",
        "best_diff_ce",
        "best_same_ce_other",
        "flag_low_ce_bad_high_ce_ok",
        "flag_high_ce_bad_low_ce_ok",
        "flag_any_diff_ce_ok",
        "flag_trend_drop",
        "flag_same_ce_replicate_contradiction",
        "flag_interior_ce_dip",
        "flag_review_candidate",
        "n_true_peaks",
        "n_achievable_peaks_inst_tol",
    ]
    flagged_cols = [c for c in flagged_cols if c in flagged_df.columns]
    flagged_df[flagged_cols].sort_values(
        ["audit_metric", "flag_same_ce_replicate_contradiction", "best_same_ce_other"],
        ascending=[True, False, False],
    ).to_csv(flagged_fp, index=False)
    plot_by_inst_adduct(by_group_df, plot_fp)

    log.info("Wrote %s", summary_fp)
    log.info("Wrote %s", by_group_fp)
    log.info("Wrote %s", flagged_fp)
    log.info("Wrote %s", plot_fp)
    log.info("\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
