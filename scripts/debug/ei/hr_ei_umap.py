"""Chemical-space UMAP of e5's NIST23 EI training molecules and the three HR-EI reference sets.

Two stages (run from the repository root):

1. ``collect`` (``FRAGNNET-GPU``; ``mol_df.pkl`` holds RDKit objects): unique 2D-InChIKey training
   molecules of the HR-EI-excluded split, the HR-EI reference molecules with their e5 10 ppm cosine,
   and each reference molecule's nearest training-molecule Tanimoto.
2. ``plot`` (``umap-viz``): one joint UMAP (Morgan r=2, 2048 bits, Jaccard) of a seeded training
   sample plus every reference molecule, drawn by source and by e5 score.

    /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -I scripts/debug/ei/hr_ei_umap.py collect \
        --eval_dp output/hr_ei_e5_eugy8s9n_eval_2026_10_18/natural_exact --out_dp output/hr_ei_umap_2026_10_18
    /home/feiw/miniconda3/envs/umap-viz/bin/python -I scripts/debug/ei/hr_ei_umap.py plot \
        --out_dp output/hr_ei_umap_2026_10_18
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/debug/dataset_umap"))
from common import FP_BITS, fingerprints, mol_info_frame, nearest, similarity_summary  # noqa: E402

SOURCE_COLORS = {"hrei_msdb": "#2a78d6", "recetox_exposome": "#eb6834", "recetox_metabolome": "#1baf7a"}
GREY, INK, MUTED = "#bdbcb5", "#1a1a19", "#6b6a63"


def collect(args: argparse.Namespace) -> None:
    """Write ``train_mols.parquet`` and ``hr_mols.parquet`` to ``--out_dp``."""
    args.out_dp.mkdir(parents=True, exist_ok=True)
    train_ids = pd.read_csv(args.split_dp / "train_ids.csv")
    mol = pd.read_pickle(ROOT / "data/proc/nist23_ei/mol_df.pkl")[["mol_id", "smiles"]]
    train = mol[mol["mol_id"].isin(set(train_ids["mol_id"]))]
    info = mol_info_frame(train["smiles"].tolist(), args.workers)[["smiles", "ik14"]]
    train = info.dropna(subset=["ik14"]).drop_duplicates("ik14")
    train.to_parquet(args.out_dp / "train_mols.parquet", index=False)

    sel = pd.read_csv(args.eval_dp / "selected_records.csv")
    scores = pd.read_csv(args.eval_dp / "scores.csv")[["spec_id", "10ppm_cosine"]]
    hr = sel[["spec_id", "source", "name", "smiles"]].merge(scores, on="spec_id")
    hr = hr.merge(mol_info_frame(hr["smiles"].tolist(), args.workers)[["smiles", "ik14"]], on="smiles")
    hr = (hr.groupby(["source", "ik14"], as_index=False)
          .agg(smiles=("smiles", "first"), name=("name", "first"), n_spectra=("spec_id", "size"),
               cosine_10ppm=("10ppm_cosine", "mean")))
    hr["nn_train"] = nearest(hr["smiles"].tolist(), train["smiles"].tolist(), args.workers)
    hr.to_parquet(args.out_dp / "hr_mols.parquet", index=False)
    summary = {src: {"molecules": len(g), **similarity_summary(g["nn_train"])} for src, g in hr.groupby("source")}
    print(f"train molecules {len(train):,}")
    print(pd.DataFrame(summary).T.to_string())


def plot(args: argparse.Namespace) -> None:
    """Fit the joint UMAP and draw the source and score panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    train = pd.read_parquet(args.out_dp / "train_mols.parquet")
    hr = pd.read_parquet(args.out_dp / "hr_mols.parquet")
    sample = train.sample(n=min(args.n_train, len(train)), random_state=args.seed)
    smiles = sample["smiles"].tolist() + hr["smiles"].tolist()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="jaccard", random_state=args.seed)
    xy = reducer.fit_transform(fingerprints(smiles, FP_BITS))
    sample = sample.assign(x=xy[: len(sample), 0], y=xy[: len(sample), 1])
    hr = hr.assign(x=xy[len(sample):, 0], y=xy[len(sample):, 1])
    sample.to_parquet(args.out_dp / "umap_train_coords.parquet", index=False)
    hr.to_parquet(args.out_dp / "umap_hr_coords.parquet", index=False)

    lims = [np.percentile(xy[:, i], [0.2, 99.8]) for i in (0, 1)]
    lims = [lo_hi + np.array([-0.06, 0.06]) * np.ptp(lo_hi) for lo_hi in lims]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), facecolor="white")
    for ax in axes:
        ax.scatter(sample["x"], sample["y"], s=0.8, c=GREY, alpha=0.6, linewidths=0, rasterized=True)
        ax.set(xlim=lims[0], ylim=lims[1], xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_visible(False)
    for src, color in SOURCE_COLORS.items():
        g = hr[hr["source"] == src]
        axes[0].scatter(g["x"], g["y"], s=14, c=color, edgecolors="white", linewidths=0.4,
                        label=f"{src} ({len(g)}; median nearest-train Tanimoto {g['nn_train'].median():.2f})")
    axes[0].scatter([], [], s=14, c=GREY, label=f"e5 training molecules (seeded sample {len(sample):,} of {len(train):,})")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.01), frameon=False, fontsize=9, labelcolor=INK)
    axes[0].set_title("HR-EI reference molecules on e5's training chemical space", loc="left", fontsize=11, color=INK)
    order = hr.sort_values("cosine_10ppm", ascending=False)
    # One hue, dark = low score, so failures stand out; the lightest step stays visible on white.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("low_dark", ["#7f1d1d", "#e34948", "#f6c9c4"])
    points = axes[1].scatter(order["x"], order["y"], s=14, c=order["cosine_10ppm"], cmap=cmap,
                             vmin=0, vmax=1, edgecolors="#6b6a63", linewidths=0.3)
    bar = fig.colorbar(points, ax=axes[1], fraction=0.035, pad=0.01)
    bar.set_label("e5 cosine, 10 ppm, natural isotopes (mean per molecule)", color=INK, fontsize=9)
    axes[1].set_title("Same molecules by e5 score (low scores drawn on top)", loc="left", fontsize=11, color=INK)
    fig.text(0.01, 0.01, "Morgan r=2, 2048 bits, Jaccard; one joint fit of the training sample and all "
             "reference molecules. Metabolome structures are the measured TMS/MeOx derivatives.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out_dp / "umap_train_vs_hr_ei.png", dpi=160)
    print(args.out_dp / "umap_train_vs_hr_ei.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--eval_dp", type=Path, required=True)
    c.add_argument("--out_dp", type=Path, required=True)
    c.add_argument("--split_dp", type=Path,
                   default=ROOT / "data/split/nist23_ei_inchikey_hr_ei_excluded_2026_10_17")
    c.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("plot")
    p.add_argument("--out_dp", type=Path, required=True)
    p.add_argument("--n_train", type=int, default=40000)
    p.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()
    collect(args) if args.stage == "collect" else plot(args)


if __name__ == "__main__":
    main()
