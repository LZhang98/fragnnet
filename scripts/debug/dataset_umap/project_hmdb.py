"""Project HMDB onto a UMAP fitted on the MS/MS datasets only.

A joint fit with HMDB fails spectral initialisation (HMDB's lipid series form many disconnected
components) and scatters them as islands across the frame. Here the map is fitted without HMDB
(``plot_umap.py`` on the MS/MS molecules, which saves ``umap_model.pkl``) and a seeded HMDB sample
is placed with ``reducer.transform``, so HMDB is shown where it lands relative to the training data.
Each HMDB point is coloured by its nearest-neighbour Tanimoto to all MS/MS datasets
(``hmdb_nn.parquet`` from ``hmdb_fit.py``) when available.

Usage (from the repository root, ``umap-viz`` env):
    python -I scripts/debug/dataset_umap/project_hmdb.py \
        --map_dp output/dataset_umap_proj_2026_10_17 \
        --hmdb_mols_fp output/dataset_umap_hmdb_2026_10_17/mols.parquet \
        --hmdb_nn_fp output/dataset_umap_hmdb_2026_10_17/hmdb_nn.parquet
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
from plot_umap import DATASET_COLORS, GREY, INK, MUTED, fingerprints, note, style  # noqa: E402

NN_BINS = [(0.999, 1.01, "identical", "#2a78d6"), (0.7, 0.999, "close ≥0.7", "#1baf7a"),
           (0.4, 0.7, "related 0.4–0.7", "#eda100"), (0.0, 0.4, "nothing similar <0.4", "#e34948")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_dp", type=Path, required=True)
    ap.add_argument("--hmdb_mols_fp", type=Path, required=True)
    ap.add_argument("--hmdb_nn_fp", type=Path, default=None)
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--view_pct", type=float, default=0.2)
    ap.add_argument("--n_label", type=int, default=5, help="Label this many densest HMDB landing spots")
    args = ap.parse_args()

    with open(args.map_dp / "umap_model.pkl", "rb") as fh:
        reducer = pickle.load(fh)
    coords = pd.read_parquet(args.map_dp / "umap_coords.parquet")
    sampled = pd.read_parquet(args.map_dp / "umap_sampled_mols.parquet")
    mols = pd.read_parquet(args.hmdb_mols_fp)
    hmdb = mols[mols["dataset"] == "HMDB"]
    nn = None
    if args.hmdb_nn_fp is not None:
        nn = pd.read_parquet(args.hmdb_nn_fp)[["smiles", "nn_all", "long_chain"]]
    # Project the whole similarity sample, topped up with a seeded random sample.
    first = hmdb[hmdb["smiles"].isin(set(nn["smiles"]))] if nn is not None else hmdb.iloc[:0]
    rest = hmdb.drop(first.index)
    rest = rest.sample(n=max(0, min(args.n - len(first), len(rest))), random_state=args.seed)
    hmdb = pd.concat([first, rest]).reset_index(drop=True)
    xy = reducer.transform(fingerprints(hmdb["smiles"].tolist(), 2048))
    hmdb["x"], hmdb["y"] = xy[:, 0], xy[:, 1]
    hmdb.to_parquet(args.map_dp / "hmdb_projected.parquet", index=False)

    hmdb = hmdb.dropna(subset=["x", "y"])
    # HMDB's lipid series land on small, far-out lipid islands of the map; keep them in view.
    pct = [args.view_pct, 100 - args.view_pct]
    both = pd.concat([coords[["x", "y"]], hmdb[["x", "y"]]])
    lims = [np.percentile(both[c], pct) for c in ("x", "y")]
    lims = [lo_hi + np.array([-0.08, 0.08]) * (lo_hi[1] - lo_hi[0]) for lo_hi in lims]
    off = ~(hmdb["x"].between(*lims[0]) & hmdb["y"].between(*lims[1]))

    panels = ["HMDB projected", "Training datasets"] + (["HMDB by nearest library molecule"] if nn is not None else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 8), facecolor="white")
    for ax in axes:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.scatter(coords["x"], coords["y"], s=0.2, c=GREY, linewidths=0, rasterized=True)
    axes[0].scatter(hmdb["x"], hmdb["y"], s=0.4, c=DATASET_COLORS["HMDB"], alpha=0.5, linewidths=0, rasterized=True)
    style(axes[0], "HMDB, projected onto the MS/MS-only map",
          f"{len(hmdb):,} of {int((mols['dataset'] == 'HMDB').sum()):,} HMDB molecules (seeded sample)")
    note(axes[0], f"{int(off.sum()):,} projected HMDB molecules lie outside the view")
    # Label the densest HMDB landing spots with HMDB and training-molecule counts (radius 1.5).
    grid = hmdb[["x", "y"]].round(0).value_counts()
    labelled = []
    for (gx, gy), _ in grid.items():
        if len(labelled) == args.n_label:
            break
        if any(np.hypot(gx - lx, gy - ly) < 4 for lx, ly in labelled):
            continue
        labelled.append((gx, gy))
        n_h = int((np.hypot(hmdb["x"] - gx, hmdb["y"] - gy) < 1.5).sum())
        near = sampled[np.hypot(sampled["x"] - gx, sampled["y"] - gy) < 1.5].drop_duplicates("ik14")
        top = near["dataset"].value_counts()
        axes[0].annotate(f"HMDB {n_h:,} / train {len(near):,}" + (f"\n({top.index[0]})" if len(top) else ""),
                         (gx, gy), xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK,
                         arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6})
    order = sorted(sampled["dataset"].unique(), key=lambda d: -int((sampled["dataset"] == d).sum()))
    for name in order:
        sub = sampled[sampled["dataset"] == name]
        axes[1].scatter(sub["x"], sub["y"], s=6 if len(sub) < 1000 else 0.25, c=DATASET_COLORS[name], alpha=0.5,
                        linewidths=0, rasterized=True)
        axes[1].scatter([], [], s=30, c=DATASET_COLORS[name], label=name)
    axes[1].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    style(axes[1], "MS/MS datasets (the fitted map)", "Drawn largest first; smaller datasets on top")
    if nn is not None:
        joined = hmdb.merge(nn, on="smiles")
        for lo, hi, label, color in NN_BINS[::-1]:
            sub = joined[(joined["nn_all"] >= lo) & (joined["nn_all"] < hi)]
            axes[2].scatter(sub["x"], sub["y"], s=1.0, c=color, alpha=0.7, linewidths=0, rasterized=True)
        for lo, hi, label, color in NN_BINS:
            n = int(((nn["nn_all"] >= lo) & (nn["nn_all"] < hi)).sum())
            axes[2].scatter([], [], s=30, c=color, label=f"{label}: {n / len(nn):.0%}")
        axes[2].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK,
                       title="Nearest molecule in any MS/MS dataset (Tanimoto)", title_fontsize=9)
        style(axes[2], "HMDB by nearest library molecule",
              f"The {len(joined):,}-molecule similarity sample (hmdb_fit.py)")
    fig.text(0.01, 0.01, "Morgan r=2, 2048 bits, Jaccard; map fitted without HMDB, HMDB placed by UMAP transform.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.map_dp / "umap_hmdb_projected.png", dpi=150)
    print(f"projected {len(hmdb)}, off view {int(off.sum())}")


if __name__ == "__main__":
    main()
