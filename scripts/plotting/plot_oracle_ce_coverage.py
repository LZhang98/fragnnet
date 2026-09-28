"""Plot DAG oracle recall / wrecall vs collision energy for NIST20.

Consumes the per-spectrum CSV produced alongside the oracle scoring
(``recall_wrecall_by_ce.csv`` with columns spec_id, inst_type, prec_type, ce, recall,
wrecall) and regenerates every figure in ``docs/findings/fragnnet/oracle_recall_wrecall_by_ce.md``:

  1. recall & wrecall means vs CE, one line per adduct (FT + QTOF panels)
  2. wrecall mean vs median vs CE (tail-broadening view)
  3. recall & wrecall distributions vs CE (violin), all adducts combined
  4. recall & wrecall distributions vs CE (violin), faceted by adduct
  5. recall & wrecall per-spectrum scatter vs CE, faceted by adduct

Colors follow the dataviz reference palette: adducts use the categorical slots (fixed
order); CE uses the sequential blue ramp (light -> dark = low -> high energy).

Usage:
    python scripts/plotting/plot_oracle_ce_coverage.py \\
        --csv data/eval/oracle_cos_nist20_default/recall_wrecall_by_ce.csv \\
        --out_dp docs/figures
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

INK, MUTED = "#0b0b0b", "#52514e"
# dataviz reference categorical slots 1-4 (fixed order), keyed by adduct
ADDUCTS = ["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"]
ADDUCT_COLORS = {"[M+H]+": "#2a78d6", "[M-H]-": "#008300", "[M+Na]+": "#e87ba4", "[M+Cl]-": "#eda100"}
# dataviz reference sequential blue ramp, light -> dark
CE_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]
INSTRUMENTS = [("OB", "NCE (%)"), ("OB_F", "NCE (%)"), ("QTOF", "ACE (eV)")]
MIN_N = 30
SCATTER_MAX_POINTS = 8000  # per panel, subsampled for legibility/rendering
SINA_HALFWIDTH = 3.5  # max +/- CE offset; each point's offset scales with local density

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": "#c9c8c4",
        "axes.linewidth": 0.8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def load_binned(csv: str) -> pd.DataFrame:
    """Load the per-spectrum CSV and add a 10-unit CE bin, dropping the boundary bins."""
    df = pd.read_csv(csv)
    df["ce_bin"] = (df["ce"] // 10 * 10).astype(int)
    return df[df["ce_bin"] < 70]


def _violin(ax, sub: pd.DataFrame, metric: str) -> None:
    """Draw one violin per CE bin (bins with < MIN_N spectra skipped)."""
    bins = [b for b in sorted(sub["ce_bin"].unique()) if (sub["ce_bin"] == b).sum() >= MIN_N]
    if not bins:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", color=MUTED,
                transform=ax.transAxes)
        return
    data = [sub.loc[sub["ce_bin"] == b, metric].to_numpy() for b in bins]
    parts = ax.violinplot(data, positions=[b + 5 for b in bins], widths=8,
                          showmedians=True, showextrema=False, points=200)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(CE_RAMP[i % len(CE_RAMP)])
        body.set_edgecolor(MUTED)
        body.set_alpha(0.9)
        body.set_linewidth(0.5)
    parts["cmedians"].set_color(INK)
    parts["cmedians"].set_linewidth(1.5)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax.set_ylim(0, 1.03)


def plot_lines(df: pd.DataFrame, out_fp: Path) -> None:
    """recall & wrecall means vs CE, one line per adduct."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for col, (inst, unit) in enumerate(INSTRUMENTS):
        for r_i, metric in enumerate(["recall", "wrecall"]):
            ax = axes[r_i][col]
            for pt in ADDUCTS:
                sub = df[(df["inst_type"] == inst) & (df["prec_type"] == pt)]
                g = sub.groupby("ce_bin").agg(n=(metric, "size"), val=(metric, "mean")).reset_index()
                g = g[g["n"] >= MIN_N]
                if len(g) < 2:
                    continue
                ax.plot(g["ce_bin"] + 5, g["val"], "-o", lw=2, ms=6, color=ADDUCT_COLORS[pt], label=pt)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.set_ylabel(f"mean {metric}")
            if r_i == 0:
                ax.set_title(inst, fontsize=12, color=INK)
            else:
                ax.set_xlabel(f"collision energy — {unit}")
        axes[0][col].set_ylim(0.55, 1.0)
        axes[1][col].set_ylim(0.80, 1.0)
    axes[0][0].legend(frameon=False, fontsize=9, title="adduct")
    fig.suptitle("DAG oracle coverage vs collision energy (NIST20, default DAGs)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)
    log.info("saved %s", out_fp)


def plot_mean_median(df: pd.DataFrame, out_fp: Path) -> None:
    """wrecall mean vs median vs CE (all adducts combined)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for col, (inst, unit) in enumerate(INSTRUMENTS):
        ax = axes[col]
        sub = df[df["inst_type"] == inst]
        g = sub.groupby("ce_bin").agg(mean=("wrecall", "mean"), median=("wrecall", "median"),
                                      n=("wrecall", "size")).reset_index()
        g = g[g["n"] >= MIN_N]
        ax.plot(g["ce_bin"] + 5, g["median"], "-o", lw=2, ms=6, color="#2a78d6", label="median")
        ax.plot(g["ce_bin"] + 5, g["mean"], "--o", lw=2, ms=6, color="#eb6834", label="mean")
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_title(inst, fontsize=12, color=INK)
        ax.set_xlabel(f"collision energy — {unit}")
        ax.set_ylim(0.80, 1.0)
    axes[0].set_ylabel("wrecall")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("wrecall: median holds, mean falls with CE (tail-broadening)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)
    log.info("saved %s", out_fp)


def plot_violin_combined(df: pd.DataFrame, out_fp: Path) -> None:
    """recall & wrecall distributions vs CE (violin), all adducts combined."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex="col")
    for col, (inst, unit) in enumerate(INSTRUMENTS):
        sub = df[df["inst_type"] == inst]
        for r_i, metric in enumerate(["recall", "wrecall"]):
            ax = axes[r_i][col]
            _violin(ax, sub, metric)
            ax.set_ylabel(metric)
            if r_i == 0:
                ax.set_title(inst, fontsize=12, color=INK)
            else:
                ax.set_xlabel(f"collision energy — {unit}")
    fig.suptitle(
        "DAG oracle coverage distribution by collision energy (NIST20, default DAGs)\n"
        "violin = full per-spectrum distribution; black bar = median",
        fontsize=13, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)
    log.info("saved %s", out_fp)


def plot_violin_by_adduct(df: pd.DataFrame, metric: str, out_fp: Path) -> None:
    """Violin of one metric vs CE, faceted adduct (rows) x instrument (cols)."""
    fig, axes = plt.subplots(len(ADDUCTS), 2, figsize=(12, 12), sharex="col")
    for r_i, pt in enumerate(ADDUCTS):
        for col, (inst, unit) in enumerate(INSTRUMENTS):
            ax = axes[r_i][col]
            _violin(ax, df[(df["inst_type"] == inst) & (df["prec_type"] == pt)], metric)
            if col == 0:
                ax.set_ylabel(f"{pt}\n{metric}")
            if r_i == 0:
                ax.set_title(inst, fontsize=12, color=INK)
            if r_i == len(ADDUCTS) - 1:
                ax.set_xlabel(f"collision energy — {unit}")
    fig.suptitle(
        f"DAG oracle {metric} distribution by collision energy and adduct (NIST20, default DAGs)\n"
        "violin = full per-spectrum distribution; black bar = median",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)
    log.info("saved %s", out_fp)


def _sina_x(vals: np.ndarray, center: float, rng: np.random.Generator) -> np.ndarray:
    """CE positions for a sina/violin-shaped strip: horizontal jitter scaled by local density.

    Each point in a CE bin is offset from ``center`` by a random amount whose maximum scales
    with the density of points at that metric value, so the cloud takes the violin's outline.
    """
    counts, edges = np.histogram(vals, bins=40, range=(0.0, 1.0))
    peak = counts.max() or 1
    idx = np.clip(np.digitize(vals, edges) - 1, 0, len(counts) - 1)
    width = counts[idx] / peak * SINA_HALFWIDTH
    return center + rng.uniform(-1.0, 1.0, len(vals)) * width


def plot_scatter_by_adduct(df: pd.DataFrame, metric: str, out_fp: Path) -> None:
    """Per-spectrum sina (violin-shaped) strip of one metric vs CE, faceted adduct x instrument.

    Same facet layout as plot_violin_by_adduct, but showing the raw per-spectrum points; within
    each CE bin the horizontal jitter is scaled by local density so the cloud mirrors the violin.
    """
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(ADDUCTS), 2, figsize=(12, 12), sharex="col", sharey=True)
    for r_i, pt in enumerate(ADDUCTS):
        for col, (inst, unit) in enumerate(INSTRUMENTS):
            ax = axes[r_i][col]
            sub = df[(df["inst_type"] == inst) & (df["prec_type"] == pt)]
            if len(sub) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", color=MUTED,
                        transform=ax.transAxes)
            else:
                if len(sub) > SCATTER_MAX_POINTS:
                    sub = sub.sample(SCATTER_MAX_POINTS, random_state=0)
                for b in sorted(sub["ce_bin"].unique()):
                    v = sub.loc[sub["ce_bin"] == b, metric].to_numpy()
                    ax.scatter(_sina_x(v, b + 5, rng), v, s=5, alpha=0.2,
                               color=ADDUCT_COLORS[pt], linewidths=0)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.set_ylim(-0.02, 1.03)
            if col == 0:
                ax.set_ylabel(f"{pt}\n{metric}")
            if r_i == 0:
                ax.set_title(inst, fontsize=12, color=INK)
            if r_i == len(ADDUCTS) - 1:
                ax.set_xlabel(f"collision energy — {unit}")
    fig.suptitle(
        f"Per-spectrum {metric} vs collision energy by adduct (NIST20, default DAGs)\n"
        "each point = one spectrum; horizontal spread scaled by density (sina / violin shape)",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fp, dpi=140)
    plt.close(fig)
    log.info("saved %s", out_fp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot oracle recall/wrecall vs CE for NIST20")
    parser.add_argument("--csv", default="data/eval/oracle_cos_nist20_default/recall_wrecall_by_ce.csv")
    parser.add_argument("--out_dp", default="docs/figures")
    args = parser.parse_args()

    df = load_binned(args.csv)
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    plot_lines(df, out_dp / "oracle_recall_wrecall_by_ce_adduct.png")
    plot_mean_median(df, out_dp / "oracle_wrecall_mean_vs_median_by_ce.png")
    plot_violin_combined(df, out_dp / "oracle_recall_wrecall_violin_by_ce.png")
    plot_violin_by_adduct(df, "recall", out_dp / "oracle_recall_violin_by_ce_adduct.png")
    plot_violin_by_adduct(df, "wrecall", out_dp / "oracle_wrecall_violin_by_ce_adduct.png")
    plot_scatter_by_adduct(df, "recall", out_dp / "oracle_recall_scatter_by_ce_adduct.png")
    plot_scatter_by_adduct(df, "wrecall", out_dp / "oracle_wrecall_scatter_by_ce_adduct.png")


if __name__ == "__main__":
    main()
