"""Joint chemical-space UMAP of the MS/MS datasets, one panel per dataset.

Fits ONE UMAP (Morgan radius 2, 2048 bits, Jaccard) on a molecule sample drawn so that every
dataset contributes up to ``--per_dataset`` molecules, so all panels share coordinates. Each
panel shows that dataset's sampled molecules over the grey union; a second figure marks the
Br, Cl and I carriers per dataset. Input is ``collect_mols.py`` output. Runs in the
``umap-viz`` conda env (umap-learn 0.5.7, RDKit 2024.09); ``FRAGNNET-GPU`` has no UMAP.

Usage (from the repository root):
    conda run -n umap-viz python scripts/debug/dataset_umap/plot_umap.py \
        --mols_fp output/dataset_umap_2026_10_16/mols.parquet \
        --out_dp output/dataset_umap_2026_10_16
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import umap  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ORDER = ["NIST20 HR", "NIST23 HR", "MassSpecGym", "Spectraverse", "MSnLib", "Enveda-180", "CASMI NP proxy"]
HIGHLIGHT = "#2a78d6"
HALOGEN_COLORS = {"Cl": "#2a78d6", "Br": "#eb6834", "I": "#1baf7a"}  # reference palette slots 1-3
GREY = "#d4d3cd"
INK, MUTED = "#1a1a19", "#6b6a63"


def fingerprints(smiles: list[str], n_bits: int) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr := np.zeros(n_bits, dtype=np.uint8))
            out[i] = arr.astype(bool)
    return out


def style(ax, title: str, subtitle: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="semibold", pad=14)
    ax.text(0, 1.005, subtitle, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols_fp", type=Path, required=True)
    ap.add_argument("--out_dp", type=Path, required=True)
    ap.add_argument("--per_dataset", type=int, default=25000)
    ap.add_argument("--n_bits", type=int, default=2048)
    ap.add_argument("--n_neighbors", type=int, default=30)
    ap.add_argument("--min_dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args()
    args.out_dp.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.mols_fp)
    totals = df.groupby("dataset").agg(molecules=("ik14", "size"), Br=("Br", "sum"), Cl=("Cl", "sum"), I=("I", "sum"))
    sampled = pd.concat(
        g.sample(n=min(len(g), args.per_dataset), random_state=args.seed) for _, g in df.groupby("dataset")
    )
    keys = sampled.drop_duplicates("ik14")[["ik14", "smiles"]].reset_index(drop=True)
    t0 = time.time()
    fps = fingerprints(keys["smiles"].tolist(), args.n_bits)
    print(f"{len(keys)} molecules fingerprinted in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    # No random_state: a fixed seed makes umap-learn single-threaded. Coordinates are saved instead.
    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, metric="jaccard", low_memory=True)
    xy = reducer.fit_transform(fps)
    print(f"UMAP fit in {time.time() - t0:.0f}s", flush=True)
    keys["x"], keys["y"] = xy[:, 0], xy[:, 1]
    keys.to_parquet(args.out_dp / "umap_coords.parquet", index=False)
    sampled = sampled.merge(keys[["ik14", "x", "y"]], on="ik14")
    sampled.to_parquet(args.out_dp / "umap_sampled_mols.parquet", index=False)

    datasets = [d for d in ORDER if d in totals.index]
    ncol = 4
    nrow = int(np.ceil((len(datasets) + 1) / ncol))
    lims = (np.percentile(keys["x"], [0.2, 99.8]), np.percentile(keys["y"], [0.2, 99.8]))

    def grid():
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow), facecolor="white")
        axes = axes.ravel()
        for ax in axes:
            ax.set_xlim(*(lims[0] + np.array([-0.5, 0.5])))
            ax.set_ylim(*(lims[1] + np.array([-0.5, 0.5])))
            ax.scatter(keys["x"], keys["y"], s=0.3, c=GREY, linewidths=0, rasterized=True)
        return fig, axes

    # Figure 1: where each dataset sits.
    fig, axes = grid()
    for ax, name in zip(axes, datasets):
        sub = sampled[sampled["dataset"] == name]
        size = 6 if len(sub) < 1000 else (1.2 if len(sub) < 20000 else 0.6)
        ax.scatter(sub["x"], sub["y"], s=size, c=HIGHLIGHT, alpha=0.6, linewidths=0, rasterized=True)
        n = int(totals.loc[name, "molecules"])
        style(ax, name, f"{n:,} molecules" + (f", {len(sub):,} plotted" if len(sub) < n else ""))
    overlap = df.groupby("ik14")["dataset"].nunique()
    style(axes[len(datasets)], "All datasets (grey)", f"{len(keys):,} unique sampled molecules; "
          f"{int((overlap > 1).sum()):,} of {len(overlap):,} in 2+ datasets")
    for ax in axes[len(datasets) + 1:]:
        ax.set_visible(False)
    fig.suptitle("Chemical space by dataset: joint UMAP of Morgan fingerprints (r=2, 2048 bits, Jaccard)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_dp / "umap_by_dataset.png", dpi=150)
    plt.close(fig)

    # Figure 2: halogen carriers per dataset (Cl first so rarer Br and I draw on top).
    fig, axes = grid()
    for ax, name in zip(axes, datasets):
        sub = sampled[sampled["dataset"] == name]
        parts = []
        for el in ("Cl", "Br", "I"):
            hal = sub[sub[el]]
            ax.scatter(hal["x"], hal["y"], s=2.5 if len(hal) > 2000 else 6, c=HALOGEN_COLORS[el], alpha=0.75,
                       linewidths=0, rasterized=True)
            parts.append(f"{el} {int(totals.loc[name, el]):,}")
        style(ax, name, "  ·  ".join(parts) + " molecules")
    legend_ax = axes[len(datasets)]
    style(legend_ax, "All datasets (grey)", "Halogen carriers drawn over all sampled molecules")
    for el, color in HALOGEN_COLORS.items():
        legend_ax.scatter([], [], s=30, c=color, label={"Cl": "Cl carrier", "Br": "Br carrier", "I": "I carrier"}[el])
    legend_ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
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
