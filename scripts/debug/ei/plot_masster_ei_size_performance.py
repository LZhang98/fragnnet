"""Plot EI cosine or Stein similarity by molecular weight and total atom count.

Uses per-spectrum output of analyze_masster_ei_checkpoint.py.
All input spectra contribute equally. Total-atom bins isolate the 64-atom limit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem


def plot_size_performance(
    input_dir: Path, output_dir: Path, metric: str = "cosine_full"
) -> pd.DataFrame:
    """Save a two-panel plot and bin summaries from EI inference."""
    if metric not in ("cosine_full", "stein_full"):
        raise ValueError(f"unsupported metric: {metric}")
    spectra = pd.read_csv(input_dir / "spectrum_rows.csv")
    specs = pd.read_pickle(input_dir / "input_subset/spec_df.pkl")
    molecules = pd.read_pickle(input_dir / "input_subset/mol_df.pkl")
    if spectra.spec_id.duplicated().any() or specs.spec_id.duplicated().any():
        raise ValueError("expected one score per spectrum")
    molecules = molecules.assign(
        total_atoms=molecules.mol.map(lambda mol: Chem.AddHs(mol).GetNumAtoms())
    )
    rows = spectra[["spec_id", metric]].merge(
        specs[["spec_id", "mol_id"]], on="spec_id", validate="one_to_one"
    ).merge(
        molecules[["mol_id", "mw", "total_atoms"]], on="mol_id", validate="many_to_one"
    )
    if len(rows) != len(spectra) or rows.isna().any().any():
        raise ValueError("missing EI score or molecular descriptor")

    summaries = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, (column, title, unit) in zip(
        axes,
        (("mw", "Molecular weight", "Da"), ("total_atoms", "Total atoms including H", "atoms")),
    ):
        groups = (
            pd.qcut(rows[column], q=5, duplicates="drop")
            if column == "mw"
            else pd.cut(rows[column], bins=[0, 16, 32, 48, 64, np.inf])
        )
        summary = rows.groupby(groups, observed=True).agg(
            n=(metric, "size"),
            mean_similarity=(metric, "mean"),
            sd_similarity=(metric, "std"),
            lower=(column, "min"),
            upper=(column, "max"),
        ).reset_index(drop=True)
        summary.insert(0, "descriptor", column)
        summary.insert(0, "metric", metric)
        summary["ci95"] = 1.96 * summary.sd_similarity / np.sqrt(summary.n)
        summaries.append(summary)
        positions = np.arange(len(summary))
        ax.errorbar(
            positions, summary.mean_similarity, yerr=summary.ci95,
            fmt="o-", color="#2563a6", capsize=4, linewidth=2, markersize=6,
        )
        labels = [
            f"{lo:.0f}–{hi:.0f}\n(n={n})"
            for lo, hi, n in zip(summary.lower, summary.upper, summary.n)
        ]
        ax.axhline(rows[metric].mean(), color="#777777", linestyle="--", linewidth=1)
        y_limits = (0.82, 0.90) if metric == "cosine_full" else (
            max(0.0, rows[metric].mean() - 0.05), min(1.0, rows[metric].mean() + 0.05)
        )
        label = "1 Da cosine" if metric == "cosine_full" else "Stein weighted similarity"
        ax.set(
            xticks=positions, xticklabels=labels, xlabel=f"{title} ({unit})",
            ylabel=f"Mean {label}", ylim=y_limits, title=title,
        )
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"MASSter EI test {label} by molecule size")
    fig.supxlabel(f"e1 checkpoint jeljdikj, epoch 37 · {len(rows):,} test spectra · bars: 95% normal CI · dashed: overall mean")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if metric == "cosine_full" else "_stein"
    fig.savefig(output_dir / f"size_performance{suffix}.png", dpi=220)
    fig.savefig(output_dir / f"size_performance{suffix}.pdf")
    plt.close(fig)
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(output_dir / f"size_performance{suffix}_bins.csv", index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=("cosine_full", "stein_full"), default="cosine_full")
    args = parser.parse_args()
    plot_size_performance(args.input_dir, args.output_dir, args.metric)


if __name__ == "__main__":
    main()
