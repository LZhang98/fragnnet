"""Sina plot of per-spectrum HR-EI cosine by molecular-ion m/z bin, one column per source."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

BIN_EDGES = [0, 100, 150, 200, 250, 300, 400, 500, np.inf]
SOURCE_COLORS = {
    "hrei_msdb": "#2a78d6",
    "recetox_exposome": "#eb6834",
    "recetox_metabolome": "#1baf7a",
}


def sina_offsets(values: np.ndarray, half_width: float, rng: np.random.Generator) -> np.ndarray:
    """Return horizontal offsets whose spread follows the KDE of ``values``."""
    if len(values) < 3 or np.ptp(values) == 0:
        return rng.uniform(-0.3, 0.3, len(values)) * half_width
    density = gaussian_kde(values)(values)
    return rng.uniform(-1, 1, len(values)) * half_width * density / density.max()


def main() -> None:
    """Plot cosine against molecular-ion m/z bins for an HR-EI evaluation directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tolerance", default="10ppm", choices=("5ppm", "10ppm", "20ppm"))
    args = parser.parse_args()
    column = f"{args.tolerance}_cosine"
    spectra = pd.read_pickle(args.directory / "spec_df.pkl")[["spec_id", "prec_mz"]]
    scores = pd.read_csv(args.directory / "scores.csv").merge(spectra, on="spec_id")
    labels = [
        f"{int(lo)}–{int(hi)}" if np.isfinite(hi) else f"≥{int(lo)}"
        for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:])
    ]
    scores["mz_bin"] = pd.cut(scores.prec_mz, BIN_EDGES, right=False, labels=labels)

    rng = np.random.default_rng(0)
    sources = list(SOURCE_COLORS)
    slot = 0.8 / len(sources)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    rows = []
    for bin_index, label in enumerate(labels):
        for source_index, source in enumerate(sources):
            values = scores.loc[
                (scores.mz_bin == label) & (scores.source == source), column
            ].to_numpy()
            if len(values) == 0:
                continue
            center = bin_index - 0.4 + slot * (source_index + 0.5)
            x = center + sina_offsets(values, 0.45 * slot, rng)
            ax.scatter(
                x, values, s=12, color=SOURCE_COLORS[source], alpha=0.7,
                edgecolors="white", linewidths=0.4, zorder=2,
            )
            median = float(np.median(values))
            ax.hlines(median, center - 0.45 * slot, center + 0.45 * slot,
                      color="#0b0b0b", linewidth=2, zorder=3)
            ax.text(center, -0.07, f"{len(values)}", ha="center", va="top",
                    fontsize=7, color="#52514e")
            rows.append({"mz_bin": label, "source": source, "n": len(values),
                         "median": median, "mean": float(values.mean())})
    for boundary in range(1, len(labels)):
        ax.axvline(boundary - 0.5, color="#e4e3df", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(-0.12, 1.02)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Molecular ion m/z  (count per source under each column)")
    ax.set_ylabel(f"Cosine, {args.tolerance.replace('ppm', ' ppm')}")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=s)
               for s, c in SOURCE_COLORS.items()]
    handles.append(plt.Line2D([], [], color="#0b0b0b", linewidth=2, label="median"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8,
              frameon=False, ncol=4)
    ckpt = pd.read_json(args.directory / "provenance.json", typ="series")["checkpoint"]
    ax.set_title(f"{Path(ckpt).name}, natural exact isotopes: cosine by molecular-ion m/z",
                 loc="left", fontsize=11)
    out = args.directory / f"mz_bin_sina_{args.tolerance}.png"
    fig.savefig(out, dpi=180)
    pd.DataFrame(rows).to_csv(args.directory / f"mz_bin_sina_{args.tolerance}.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
