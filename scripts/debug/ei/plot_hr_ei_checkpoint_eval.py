"""Plot HR-EI checkpoint scores and representative measured/predicted mirror spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.debug.ei.audit_hr_ei_raw import read_records
from scripts.debug.ei.evaluate_hr_ei_checkpoint import SOURCES


def main() -> None:
    """Plot paired nominal/HR scores and one median-scoring example per source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    scores = pd.read_csv(args.directory / "scores.csv")
    provenance = json.loads((args.directory / "provenance.json").read_text())
    natural = provenance["isotope_envelope"] == "natural"
    prediction_label = "Predicted, natural isotopes" if natural else "Predicted, monoisotopic"
    scores.groupby("source").agg(
        spectra=("spec_id", "size"),
        hr_5ppm_mean=("5ppm_cosine", "mean"),
        hr_5ppm_median=("5ppm_cosine", "median"),
        hr_10ppm_mean=("10ppm_cosine", "mean"),
        hr_20ppm_mean=("20ppm_cosine", "mean"),
        nominal_mean=("nominal_cosine", "mean"),
        target_intensity_recall=("5ppm_target_intensity_recall", "mean"),
    ).to_csv(args.directory / "metric_summary.csv")
    selected = pd.read_csv(args.directory / "selected_records.csv").set_index("spec_id")
    predictions = np.load(args.directory / "cache/exact_predictions.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for source, group in scores.groupby("source"):
        axes[0].scatter(group.nominal_cosine, group["5ppm_cosine"], s=10, alpha=0.5, label=source)
        axes[1].plot(
            [5, 10, 20],
            [group[f"{ppm}ppm_cosine"].mean() for ppm in [5, 10, 20]],
            marker="o",
            label=f"{source} (n={len(group)})",
        )
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0].set(
        xlabel="Nominal control cosine (same predictions)",
        ylabel="HR cosine (5 ppm, 0.5 mDa floor)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].set(
        xlabel="Mass tolerance (ppm; floors 0.5, 1, 2 mDa)",
        ylabel="Mean HR cosine",
        xticks=[5, 10, 20],
        ylim=(0, 1),
    )
    axes[1].legend(fontsize=8)
    fig.suptitle("Frozen e1 checkpoint: " + prediction_label.lower())
    fig.savefig(args.directory / "hr_vs_nominal.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    examples = []
    for ax, (source, group) in zip(axes, scores.groupby("source")):
        row = group.sort_values("5ppm_cosine").iloc[len(group) // 2]
        spec_id = int(row.spec_id)
        record = selected.loc[spec_id]
        target = np.array(
            read_records(Path("data/raw/hr_ei_ms") / SOURCES[source])[int(record.record)][1]
        )
        mz, intensity = predictions[f"{spec_id}_mz"], predictions[f"{spec_id}_int"]
        ax.vlines(
            target[:, 0],
            0,
            target[:, 1] / target[:, 1].max(),
            color="#2463a6",
            label="Measured HR-EI",
        )
        ax.vlines(mz, 0, -intensity / intensity.max(), color="#d56a24", label=prediction_label)
        ax.set(
            xlabel="m/z",
            ylabel="Relative intensity",
            ylim=(-1.08, 1.08),
            title=f"{source}: {record['name']}\n5 ppm cosine {row['5ppm_cosine']:.3f}; nominal {row.nominal_cosine:.3f}",
        )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.legend(fontsize=8, loc="upper right")
        examples.append({"source": source, "spec_id": spec_id, "selection": "median 5ppm cosine"})
    fig.savefig(args.directory / "median_examples.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(examples).to_csv(args.directory / "plotted_examples.csv", index=False)


if __name__ == "__main__":
    main()
