"""Project a structure-only set (HMDB, NP Atlas: ``--dataset``) onto a UMAP fitted on the MS/MS datasets only.

A joint fit with HMDB fails spectral initialisation (HMDB's lipid series form many disconnected
components) and scatters them as islands across the frame. Here the map is fitted without the set
(``plot_umap.py`` on the MS/MS molecules, which saves ``umap_model.pkl``) and a seeded sample is
placed with ``reducer.transform``, so the set is shown where it lands relative to the training
data. Each point is coloured by its nearest-molecule Tanimoto to all MS/MS datasets
(``<slug>_nn.parquet`` from ``nn_fit.py``) when available.

Usage (from the repository root, ``umap-viz`` env):
    python -I scripts/debug/dataset_umap/project.py \
        --map_dp output/dataset_umap_proj_2026_10_17 \
        --mols_fp output/dataset_umap_hmdb_2026_10_17/mols.parquet \
        --nn_fp output/dataset_umap_hmdb_2026_10_17/hmdb_nn.parquet
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASET_COLORS,
    FP_BITS,
    INK,
    MUTED,
    SIMILARITY_BINS,
    base_map,
    fingerprints,
    in_view,
    note,
    slug,
    style,
    view_limits,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_dp", type=Path, required=True)
    ap.add_argument("--mols_fp", type=Path, required=True, help="collect_mols.py output holding --dataset")
    ap.add_argument("--dataset", default="HMDB", help="Dataset name in --mols_fp")
    ap.add_argument("--nn_fp", type=Path, default=None, help="nn_fit.py output for --dataset")
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--view_pct", type=float, default=0.2)
    ap.add_argument("--n_label", type=int, default=5, help="Label this many densest landing spots")
    args = ap.parse_args()

    with open(args.map_dp / "umap_model.pkl", "rb") as fh:
        reducer = pickle.load(fh)
    coords = pd.read_parquet(args.map_dp / "umap_coords.parquet")
    sampled = pd.read_parquet(args.map_dp / "umap_sampled_mols.parquet")
    mols = pd.read_parquet(args.mols_fp)
    name = args.dataset
    proj = mols[mols["dataset"] == name]
    nn = None
    if args.nn_fp is not None:
        nn = pd.read_parquet(args.nn_fp)[["smiles", "nn_all", "long_chain"]]
    # Project the whole similarity sample, topped up with a seeded random sample.
    first = proj[proj["smiles"].isin(set(nn["smiles"]))] if nn is not None else proj.iloc[:0]
    rest = proj.drop(first.index)
    rest = rest.sample(n=max(0, min(args.n - len(first), len(rest))), random_state=args.seed)
    proj = pd.concat([first, rest]).reset_index(drop=True)
    xy = reducer.transform(fingerprints(proj["smiles"].tolist(), FP_BITS))
    proj["x"], proj["y"] = xy[:, 0], xy[:, 1]
    proj.to_parquet(args.map_dp / f"{slug(name)}_projected.parquet", index=False)

    proj = proj.dropna(subset=["x", "y"])
    # HMDB's lipid series land on small, far-out lipid islands of the map; keep them in view.
    lims = view_limits(pd.concat([coords[["x", "y"]], proj[["x", "y"]]]), args.view_pct)
    off = ~in_view(proj, lims)

    panels = [f"{name} projected", "Training datasets"] + ([f"{name} by nearest library molecule"] if nn is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 8), facecolor="white")
    for ax in axes:
        base_map(ax, coords, lims)
    axes[0].scatter(proj["x"], proj["y"], s=0.4, c=DATASET_COLORS.get(name, "#e34948"), alpha=0.5, linewidths=0,
                    rasterized=True)
    style(axes[0], f"{name}, projected onto the MS/MS-only map",
          f"{len(proj):,} of {int((mols['dataset'] == name).sum()):,} {name} molecules (seeded sample)")
    note(axes[0], f"{int(off.sum()):,} projected {name} molecules lie outside the view")
    # Label the densest landing spots with the set's and the training molecules' counts (radius 1.5).
    grid = proj[["x", "y"]].round(0).value_counts()
    labelled = []
    for (gx, gy), _ in grid.items():
        if len(labelled) == args.n_label:
            break
        if any(np.hypot(gx - lx, gy - ly) < 4 for lx, ly in labelled):
            continue
        labelled.append((gx, gy))
        n_h = int((np.hypot(proj["x"] - gx, proj["y"] - gy) < 1.5).sum())
        near = sampled[np.hypot(sampled["x"] - gx, sampled["y"] - gy) < 1.5].drop_duplicates("ik14")
        top = near["dataset"].value_counts()
        axes[0].annotate(f"{name} {n_h:,} / train {len(near):,}" + (f"\n({top.index[0]})" if len(top) else ""),
                         (gx, gy), xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK,
                         arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    order = sorted(sampled["dataset"].unique(), key=lambda d: -int((sampled["dataset"] == d).sum()))
    for ds in order:
        sub = sampled[sampled["dataset"] == ds]
        axes[1].scatter(sub["x"], sub["y"], s=6 if len(sub) < 1000 else 0.25, c=DATASET_COLORS[ds], alpha=0.5,
                        linewidths=0, rasterized=True)
        axes[1].scatter([], [], s=30, c=DATASET_COLORS[ds], label=ds)
    axes[1].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    style(axes[1], "MS/MS datasets (the fitted map)", "Drawn largest first; smaller datasets on top")
    if nn is not None:
        joined = proj.merge(nn, on="smiles")
        for lo, hi, _label, color in SIMILARITY_BINS[::-1]:
            sub = joined[(joined["nn_all"] >= lo) & (joined["nn_all"] < hi)]
            axes[2].scatter(sub["x"], sub["y"], s=1.0, c=color, alpha=0.7, linewidths=0, rasterized=True)
        for lo, hi, label, color in SIMILARITY_BINS:
            n = int(((nn["nn_all"] >= lo) & (nn["nn_all"] < hi)).sum())
            axes[2].scatter([], [], s=30, c=color, label=f"{label}: {n / len(nn):.0%}")
        axes[2].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK,
                       title="Nearest molecule in any MS/MS dataset (Tanimoto)", title_fontsize=9)
        style(axes[2], f"{name} by nearest library molecule",
              f"The {len(joined):,}-molecule similarity sample (nn_fit.py)")
    fig.text(0.01, 0.01, f"Morgan r=2, 2048 bits, Jaccard; map fitted on MS/MS datasets only, {name} placed by UMAP transform.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.map_dp / f"umap_{slug(name)}_projected.png", dpi=150)
    print(f"projected {len(proj)}, off view {int(off.sum())}")


if __name__ == "__main__":
    main()
