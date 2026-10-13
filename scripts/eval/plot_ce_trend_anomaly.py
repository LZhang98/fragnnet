"""Render a summary PDF for the CE trend-anomaly detector from its output CSVs.

Reads ``oracle_ce_trend_anomaly_scores.csv`` and ``oracle_ce_trend_anomaly_sweep.csv`` (produced by
``oracle_ce_trend_anomaly.py``) and draws: example per-group CE curves with the isotonic trend and
flagged dip, the flag/corroboration threshold sweep, and the corroboration-rate curve.

Usage:
    python scripts/eval/plot_ce_trend_anomaly.py --out_dp data/eval/oracle_cos_nist20_default
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

GROUP_COLS = ["mol_id", "prec_type", "inst_type", "frag_mode"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out_dp", required=True)
    parser.add_argument("--metric", default="cos_hun_inst_tol_opt")
    parser.add_argument(
        "--example_thresh",
        type=float,
        default=0.4,
        help="Residual cutoff used to pick and mark example dips.",
    )
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--min_ce_for_example", type=int, default=5)
    return parser.parse_args()


def pick_examples(scores: pd.DataFrame, metric: str, thresh: float, n: int, min_ce: int) -> list:
    """Pick n distinct groups with the deepest dip and enough CE points to show a curve."""
    dips = scores[(scores["iso_resid"] >= thresh) & (scores["n_ce_bins"] >= min_ce)]
    dips = dips.sort_values("iso_resid", ascending=False)
    picked, seen = [], set()
    for _, row in dips.iterrows():
        key = tuple(row[c] for c in GROUP_COLS)
        if key in seen:
            continue
        seen.add(key)
        grp = scores[(scores[GROUP_COLS] == pd.Series(dict(zip(GROUP_COLS, key)))).all(axis=1)]
        picked.append((key, grp.sort_values("ce_value"), row["spec_id"]))
        if len(picked) == n:
            break
    return picked


def plot_example(ax, key, grp: pd.DataFrame, dip_spec_id, metric: str) -> None:
    """Plot one group's CE curve: points, isotonic trend line, and the flagged dip."""
    ax.plot(grp["ce_value"], grp["iso_fit"], "-", color=COLORS[2], lw=2, label="isotonic trend", zorder=1)
    ax.scatter(grp["ce_value"], grp[metric], color=COLORS[0], s=28, zorder=2, label="spectra")
    dip = grp[grp["spec_id"] == dip_spec_id]
    ax.scatter(dip["ce_value"], dip[metric], color=COLORS[3], s=90, marker="v", zorder=3,
               edgecolor="black", linewidth=0.6, label="flagged dip")
    mol_id, prec, inst, _ = key
    resid = float(dip["iso_resid"].iloc[0])
    ax.set_title(f"mol {mol_id}  {inst} {prec}\ndip residual = {resid:.2f}", fontsize=9)
    ax.set_xlabel("collision energy")
    ax.set_ylabel(metric.replace("_opt", ""))
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7, loc="best")


def main() -> None:
    """Build the trend-anomaly summary PDF."""
    args = get_args()
    out_dp = Path(args.out_dp)
    scores = pd.read_csv(out_dp / "oracle_ce_trend_anomaly_scores.csv")
    sweep = pd.read_csv(out_dp / "oracle_ce_trend_anomaly_sweep.csv")
    metric = args.metric if args.metric in scores.columns else "cos_hun_inst_tol_opt"

    examples = pick_examples(
        scores, metric, args.example_thresh, args.n_examples, args.min_ce_for_example
    )
    if len(examples) < args.n_examples:
        log.warning("Only found %d example groups (asked %d).", len(examples), args.n_examples)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # top row: example dips
    for ax, (key, grp, dip_id) in zip(axes[0], examples):
        plot_example(ax, key, grp, dip_id, metric)
    for ax in axes[0][len(examples):]:
        ax.axis("off")

    # bottom-left: sweep counts
    ax = axes[1][0]
    ax.plot(sweep["resid_thresh"], sweep["n_flagged"], "-o", color=COLORS[0], label="flagged")
    ax.plot(sweep["resid_thresh"], sweep["n_corroborated"], "-s", color=COLORS[2],
            label="corroborated by sibling")
    ax.set_xlabel("isotonic-residual threshold")
    ax.set_ylabel("# spectra")
    ax.set_title("Trend-dip flags vs threshold")
    ax.legend(fontsize=8)

    # bottom-middle: corroboration rate
    ax = axes[1][1]
    ax.plot(sweep["resid_thresh"], sweep["pct_corroborated"], "-o", color=COLORS[4])
    ax.set_ylim(90, 100.5)
    ax.set_xlabel("isotonic-residual threshold")
    ax.set_ylabel("% flagged with good different-CE sibling")
    ax.set_title("Corroboration rate")
    for _, r in sweep.iterrows():
        ax.annotate(f"{r['pct_corroborated']:.1f}%", (r["resid_thresh"], r["pct_corroborated"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)

    # bottom-right: residual distribution
    ax = axes[1][2]
    ax.hist(scores["iso_resid"].clip(lower=-0.1, upper=1.0), bins=60, color=COLORS[0],
            edgecolor="white", linewidth=0.3)
    ax.axvline(args.example_thresh, color=COLORS[3], ls="--", lw=1.5,
               label=f"example thresh {args.example_thresh}")
    ax.set_yscale("log")
    ax.set_xlabel("isotonic residual (fit - metric)")
    ax.set_ylabel("# spectra (log)")
    ax.set_title(f"Residual distribution (n={len(scores):,})")
    ax.legend(fontsize=8)

    fig.suptitle("CE trend-anomaly detection (oracle cos vs collision energy)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_fp = out_dp / "oracle_ce_trend_anomaly.pdf"
    fig.savefig(out_fp, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_fp)


if __name__ == "__main__":
    main()
