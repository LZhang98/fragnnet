"""Plot how much of each predicted EI peak comes from its largest fragment.

Input is the per-peak output of analyze_masster_ei_checkpoint.py. The intensity cutoff
is relative to each spectrum's predicted base peak.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dominance(
    input_path: Path, png_path: Path, pdf_path: Path, min_relative_intensity: float
) -> None:
    """Plot peak-level and base-peak dominance from an annotated inference table."""
    peaks = pd.read_csv(input_path)
    selected = peaks[peaks.relative_to_base >= min_relative_intensity]
    base = peaks.loc[peaks.groupby("spec_id").pred_share.idxmax()]
    if len(base) != peaks.spec_id.nunique():
        raise ValueError("expected one predicted base peak per spectrum")

    blue, orange = "#2563a6", "#dc643e"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.25), constrained_layout=True)
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(
        selected.top_fragment_share,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.3,
        color=blue,
        label=f"Predicted peaks (n={len(selected):,})",
    )
    ax.hist(
        base.top_fragment_share,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.3,
        color=orange,
        label=f"Predicted base peaks (n={len(base):,})",
    )
    ax.axvline(0.5, color="#52525b", linestyle="--", linewidth=1.3)
    ax.text(0.515, ax.get_ylim()[1] * 0.92, "majority", color="#52525b", fontsize=9)
    ax.set(xlim=(0, 1), xlabel="Largest fragment's share of a peak", ylabel="Density")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    cutoffs = (0.5, 0.8, 0.9)
    x = np.arange(len(cutoffs))
    width = 0.34
    ordinary = np.array([(selected.top_fragment_share >= t).mean() for t in cutoffs])
    base_rates = np.array([(base.top_fragment_share >= t).mean() for t in cutoffs])
    bars_a = ax.bar(x - width / 2, ordinary * 100, width, color=blue, label="Predicted peaks")
    bars_b = ax.bar(x + width / 2, base_rates * 100, width, color=orange, label="Base peaks")
    for bars in (bars_a, bars_b):
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9)
    ax.set(
        xticks=x,
        xticklabels=["≥50%", "≥80%", "≥90%"],
        ylim=(0, 100),
        ylabel="Peaks with one fragment above threshold (%)",
        xlabel="Largest fragment's share",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("EI predictions: does one fragment dominate a peak?", fontsize=13)
    fig.text(
        0.5, -0.03,
        "e1 checkpoint jeljdikj, epoch 37 · 512 validation molecules · "
        f"predicted peaks ≥{100 * min_relative_intensity:g}% of each spectrum's base peak",
        ha="center", fontsize=9, color="#52525b",
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--min-relative-intensity", type=float, default=0.001)
    args = parser.parse_args()
    if not 0 < args.min_relative_intensity <= 1:
        raise ValueError("--min-relative-intensity must be in (0, 1]")
    plot_dominance(args.input, args.png, args.pdf, args.min_relative_intensity)
    print(args.png)
    print(args.pdf)


if __name__ == "__main__":
    main()
