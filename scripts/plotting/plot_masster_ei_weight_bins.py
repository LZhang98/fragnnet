"""Plot EI prediction similarity distributions in molecular-weight bins.

Uses the full-test spectrum scores and molecule table produced by
``analyze_masster_ei_checkpoint.py``. Each spectrum contributes once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_predictions_hdf5 import _sina_ax, _violin_ax

plt.switch_backend("Agg")

WEIGHT_EDGES = [0, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, np.inf]
METRICS = (("cosine_full", "1 Da cosine"), ("stein_full", "Stein weighted similarity"))


def plot_weight_bins(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Save CE-style violin and sina plots with per-bin summaries."""
    scores = pd.read_csv(input_dir / "spectrum_rows.csv")
    specs = pd.read_pickle(input_dir / "input_subset/spec_df.pkl")
    molecules = pd.read_pickle(input_dir / "input_subset/mol_df.pkl")
    if scores.spec_id.duplicated().any() or specs.spec_id.duplicated().any():
        raise ValueError("expected one row per spectrum")
    if molecules.mol_id.duplicated().any():
        raise ValueError("expected one row per molecule")

    rows = scores[["spec_id", "cosine_full", "stein_full"]].merge(
        specs[["spec_id", "mol_id"]], on="spec_id", validate="one_to_one"
    ).merge(molecules[["mol_id", "mw"]], on="mol_id", validate="many_to_one")
    if len(rows) != len(scores) or rows[["mw", "cosine_full", "stein_full"]].isna().any().any():
        raise ValueError("missing molecular weight or similarity score")
    if not np.isfinite(rows[["mw", "cosine_full", "stein_full"]].to_numpy()).all():
        raise ValueError("non-finite molecular weight or similarity score")

    labels = [
        f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"≥{lo:g}"
        for lo, hi in zip(WEIGHT_EDGES[:-1], WEIGHT_EDGES[1:])
    ]
    rows["weight_bin"] = pd.cut(
        rows.mw, WEIGHT_EDGES, labels=labels, right=False, include_lowest=True
    )
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    summaries = []
    for ax, (metric, title) in zip(axes, METRICS):
        arrays = [rows.loc[rows.weight_bin == label, metric].to_numpy() for label in labels]
        _violin_ax(
            ax, labels, arrays, title=title, xlabel="Molecular weight (Da)",
            color="#5B9BD5" if metric == "cosine_full" else "#70AD47",
            score_col=title,
        )
        ax.set_ylim(0, 1.05)
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, [label.replace("IQR (tol range)", "IQR (25–75%)") for label in legend_labels],
            fontsize=8, loc="lower left", ncol=3, framealpha=0.95, edgecolor="grey",
        )
        for label, values in zip(labels, arrays):
            if len(values) == 0:
                continue
            summaries.append({
                "metric": metric,
                "weight_bin": label,
                "n_spectra": len(values),
                "mean": values.mean(),
                "median": np.median(values),
                "q25": np.quantile(values, 0.25),
                "q75": np.quantile(values, 0.75),
            })
    fig.suptitle(
        f"MASSter EI test predictions by molecular weight · {len(rows):,} spectra · "
        "e1 jeljdikj epoch 37",
        fontsize=14,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "ei_weight_bin_violin.png", dpi=180)
    fig.savefig(output_dir / "ei_weight_bin_violin.pdf")
    plt.close(fig)

    sina_fig, sina_axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    for ax, (metric, title) in zip(sina_axes, METRICS):
        arrays = [rows.loc[rows.weight_bin == label, metric].to_numpy() for label in labels]
        _sina_ax(
            ax, labels, arrays, title=title, xlabel="Molecular weight (Da)",
            color="#5B9BD5" if metric == "cosine_full" else "#70AD47",
            score_col=title,
        )
        ax.set_ylim(0, 1.05)
    sina_fig.suptitle(
        f"MASSter EI test predictions by molecular weight · {len(rows):,} spectra · "
        "e1 jeljdikj epoch 37",
        fontsize=14,
    )
    sina_fig.savefig(output_dir / "ei_weight_bin_sina.png", dpi=180)
    sina_fig.savefig(output_dir / "ei_weight_bin_sina.pdf")
    plt.close(sina_fig)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "ei_weight_bin_violin_summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_weight_bins(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
