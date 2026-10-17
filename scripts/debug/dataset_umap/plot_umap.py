"""Joint chemical-space UMAP of the MS/MS datasets, one panel per dataset.

Fits ONE UMAP (Morgan radius 2, 2048 bits, Jaccard) on a molecule sample drawn so that every
dataset contributes up to ``--per_dataset`` molecules, so all panels share coordinates. Each
panel shows that dataset's sampled molecules over the grey union; a second figure marks the
Br, Cl and I carriers per dataset. Input is ``collect_mols.py`` output. Runs in the
``umap-viz`` conda env (umap-learn 0.5.7, RDKit 2024.09); ``FRAGNNET-GPU`` has no UMAP. Saves
``umap_model.pkl`` for ``project.py``.

Usage (from the repository root):
    conda run -n umap-viz python scripts/debug/dataset_umap/plot_umap.py \
        --mols_fp output/dataset_umap_2026_10_16/mols.parquet \
        --out_dp output/dataset_umap_2026_10_16
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASET_COLORS,
    DATASET_GREY,
    FP_BITS,
    HALOGEN_COLORS,
    INK,
    ORDER,
    base_map,
    fingerprints,
    in_view,
    note,
    style,
    view_limits,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols_fp", type=Path, required=True)
    ap.add_argument("--out_dp", type=Path, required=True)
    ap.add_argument("--per_dataset", type=int, default=25000)
    ap.add_argument("--n_bits", type=int, default=FP_BITS)
    ap.add_argument("--n_neighbors", type=int, default=30)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--view_pct", type=float, default=2.0, help="Zoom to this..100-this percentile")
    ap.add_argument("--reuse_coords", action="store_true", help="Re-plot from saved coordinates")
    args = ap.parse_args()
    args.out_dp.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.mols_fp)
    totals = df.groupby("dataset").agg(molecules=("ik14", "size"), Br=("Br", "sum"), Cl=("Cl", "sum"), I=("I", "sum"))
    sampled = pd.concat(
        g.sample(n=min(len(g), args.per_dataset), random_state=args.seed) for _, g in df.groupby("dataset")
    )
    keys = sampled.drop_duplicates("ik14")[["ik14", "smiles"]].reset_index(drop=True)
    coords_fp = args.out_dp / "umap_coords.parquet"
    if args.reuse_coords:
        keys = keys.merge(pd.read_parquet(coords_fp)[["ik14", "x", "y"]], on="ik14", validate="1:1")
    else:
        t0 = time.time()
        fps = fingerprints(keys["smiles"].tolist(), args.n_bits)
        print(f"{len(keys)} molecules fingerprinted in {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        # No random_state: a fixed seed makes umap-learn single-threaded. Coordinates are saved instead.
        reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric="jaccard", low_memory=True)
        xy = reducer.fit_transform(fps)
        print(f"UMAP fit in {time.time() - t0:.0f}s", flush=True)
        keys["x"], keys["y"] = xy[:, 0], xy[:, 1]
        keys.to_parquet(coords_fp, index=False)
        with open(args.out_dp / "umap_model.pkl", "wb") as fh:  # for reducer.transform of new datasets
            pickle.dump(reducer, fh)
    sampled = sampled.merge(keys[["ik14", "x", "y"]], on="ik14")
    sampled.to_parquet(args.out_dp / "umap_sampled_mols.parquet", index=False)

    datasets = [d for d in ORDER if d in totals.index]
    nrow = 2
    ncol = int(np.ceil((len(datasets) + 1) / nrow))
    lims = view_limits(keys, args.view_pct)
    off_view = f"{int((~in_view(keys, lims)).sum()):,} of {len(keys):,} sampled molecules lie outside the view"

    def grid():
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow), facecolor="white")
        axes = axes.ravel()
        for ax in axes:
            base_map(ax, keys, lims)
        return fig, axes

    def dot_size(n: int, small: float) -> float:
        return 6 if n < 1000 else small

    # Largest sample first so smaller datasets stay visible on top.
    overlay_order = sorted(datasets, key=lambda d: -int((sampled["dataset"] == d).sum()))
    overlap = df.groupby("ik14")["dataset"].nunique()
    overlay_sub = (f"{len(keys):,} unique sampled molecules; {int((overlap > 1).sum()):,} of "
                   f"{len(overlap):,} in 2+ datasets. Drawn largest first, so later datasets cover earlier ones")

    def overlay(ax, small: float, legend_size: float, legend: bool = True) -> None:
        for name in overlay_order:
            sub = sampled[sampled["dataset"] == name]
            ax.scatter(sub["x"], sub["y"], s=dot_size(len(sub), small), c=DATASET_COLORS[name], alpha=0.5,
                       linewidths=0, rasterized=True)
        for name in datasets:
            ax.scatter([], [], s=legend_size, c=DATASET_COLORS[name], label=name)
        if legend:
            ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)

    # Figure 1: where each dataset sits, one colour per dataset.
    fig, axes = grid()
    for ax, name in zip(axes, datasets):
        sub = sampled[sampled["dataset"] == name]
        ax.scatter(sub["x"], sub["y"], s=dot_size(len(sub), 0.25), c=DATASET_COLORS[name], alpha=0.5,
                   linewidths=0, rasterized=True)
        n = int(totals.loc[name, "molecules"])
        style(ax, name, f"{n:,} molecules" + (f", {len(sub):,} plotted" if len(sub) < n else ""))
    overlay(axes[len(datasets)], 0.25, 30, legend=False)
    style(axes[len(datasets)], "All datasets overlaid", "Drawn largest first; smaller datasets on top")
    note(axes[len(datasets)], off_view)
    fig.legend(*axes[len(datasets)].get_legend_handles_labels(), loc="lower center", ncol=len(datasets),
               frameon=False, fontsize=10, labelcolor=INK)
    for ax in axes[len(datasets) + 1:]:
        ax.set_visible(False)
    fig.suptitle("Chemical space by dataset: joint UMAP of Morgan fingerprints (r=2, 2048 bits, Jaccard)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(args.out_dp / "umap_by_dataset.png", dpi=150)
    plt.close(fig)

    # Figure 1b: the overlay alone, large.
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    overlay(ax, 0.6, 40)
    style(ax, "Chemical space of all datasets: joint UMAP of Morgan fingerprints (r=2, 2048 bits, Jaccard)",
          overlay_sub)
    ax.title.set_y(1.02)
    note(ax, off_view)
    fig.tight_layout()
    fig.savefig(args.out_dp / "umap_all_datasets.png", dpi=150)
    plt.close(fig)

    # Figure 2: halogen carriers per dataset (Cl first so rarer Br and I draw on top).
    fig, axes = grid()
    for ax, name in zip(axes, datasets):
        sub = sampled[sampled["dataset"] == name]
        parts = []
        ax.scatter(sub["x"], sub["y"], s=0.2, c=DATASET_GREY, linewidths=0, rasterized=True)
        for el in ("Cl", "Br", "I"):
            hal = sub[sub[el]]
            ax.scatter(hal["x"], hal["y"], s=1.0 if len(sub) >= 1000 else 8, c=HALOGEN_COLORS[el], alpha=0.8,
                       linewidths=0, rasterized=True)
            parts.append(f"{el} {int(totals.loc[name, el]):,}")
        style(ax, name, "  ·  ".join(parts) + " molecules")
    legend_ax = axes[len(datasets)]
    style(legend_ax, "All datasets (light grey)", "Dark grey in each panel: that dataset's own molecules")
    note(legend_ax, off_view)
    for el, color in HALOGEN_COLORS.items():
        legend_ax.scatter([], [], s=30, c=color, label={"Cl": "Cl carrier", "Br": "Br carrier", "I": "I carrier"}[el])
    legend_ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    for ax in axes[len(datasets) + 1:]:
        ax.set_visible(False)
    fig.suptitle("Halogen carriers by dataset (same joint UMAP; datasets over 25,000 molecules are subsampled)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_dp / "umap_halogens_by_dataset.png", dpi=150)
    plt.close(fig)

    (args.out_dp / "umap_run_info.json").write_text(json.dumps({
        **vars(args), "mols_fp": str(args.mols_fp), "out_dp": str(args.out_dp), "umap": umap.__version__,
        "n_umap_molecules": len(keys), "seed_note": "seed used for sampling only; UMAP itself unseeded (parallel)",
    }, indent=2))


if __name__ == "__main__":
    main()
