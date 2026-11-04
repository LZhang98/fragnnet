"""Compare the chemical space of two molecule tables (e.g. NIST20 vs Enveda-180).

A two-dataset version of the ``scripts/debug/dataset_umap`` pipeline, reusing its helpers and
defaults (Morgan r=2, 2048 bits; 2D InChIKey block as molecule identity):

1. Dedupe each table to one row per 2D InChIKey block and count Br/Cl/I carriers.
2. Overlap: molecules present in both tables (same 2D block).
3. Nearest-molecule Tanimoto of a seeded sample of each table to the other (both directions),
   summarised and binned as identical / close analogue / related / no analogue.
4. One joint UMAP (Jaccard) over both tables, one panel per dataset plus the overlay.

Each table is a parquet / pkl / csv / tsv with a ``smiles`` column; the dataset name is the file stem.
Writes ``{a}_{b}_umap.png``, ``{a}_{b}_similarity.png``, ``{a}_{b}_coords.csv``,
``{a}_{b}_report.csv`` and ``{a}_{b}_report.json`` to ``--out_dp`` (``a``/``b`` are the input
file stems), so several runs can share one output directory without clobbering. ``--out_dp`` defaults to
``~/fragnnet/output/chem_space``. Runs in the ``umap-viz`` env (needs umap-learn).

Usage (from the repository root):
    uv run python scripts/chem_space/compare_datasets.py \
        -a data/proc/nist20/mol_df.pkl -b data/proc/enveda_180/mol_df.pkl
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from tqdm import tqdm  # noqa: E402
import umap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "debug" / "dataset_umap"))
from common import (  # noqa: E402
    INK,
    SIMILARITY_BINS,
    base_map,
    fingerprints,
    mol_info_frame,
    nearest,
    similarity_bin,
    similarity_summary,
    style,
    view_limits,
)

COLORS = ["#2a78d6", "#eb6834"]
N_NN, PER_DATASET, SEED = 10000, 25000, 20261016
K_NN = 10


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--mols_a", type=Path, required=True, help="First molecule table (parquet / pkl / csv / tsv) with a smiles column")
    ap.add_argument("-b", "--mols_b", type=Path, required=True, help="Second molecule table (parquet / pkl / csv / tsv) with a smiles column")
    default_out = Path.home() / "fragnnet" / "output" / "chem_space"
    ap.add_argument("-o", "--out_dp", type=Path, default=default_out, help=f"Output directory (default {default_out})")
    ap.add_argument("-w", "--workers", type=int, default=6, help="Parallel workers for fingerprinting / nearest search (default 6)")
    return ap.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    """Read a molecule table, dispatching on file suffix: parquet / pkl / tsv / csv."""
    sep = "\t" if path.suffix == ".tsv" else ","
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path, sep=sep)


def load_molecules(path: Path, workers: int) -> pd.DataFrame:
    """One row per 2D InChIKey block (columns smiles, ik14, Br, Cl, I) of a table with a ``smiles`` column."""
    table = read_table(path)
    info = mol_info_frame(table["smiles"].dropna().tolist(), workers)
    return info.dropna(subset=["ik14"]).drop_duplicates("ik14").reset_index(drop=True)


def halogen_counts(mols: dict[str, pd.DataFrame]) -> dict:
    """Per-dataset molecule count and Br/Cl/I carrier counts."""
    return {n: {"molecules": len(m), **{e: int(m[e].sum()) for e in ("Br", "Cl", "I")}} for n, m in mols.items()}


def overlap(mols: dict[str, pd.DataFrame], names: list[str], report: dict) -> set:
    """2D-InChIKey-block molecules present in both tables; store count and per-dataset fraction."""
    a, b = names
    shared = set(mols[a]["ik14"]) & set(mols[b]["ik14"])
    report["shared_molecules"] = {"n": len(shared), **{n: round(len(shared) / len(mols[n]), 3) for n in names}}
    return shared


def nearest_molecule_similarity(mols: dict[str, pd.DataFrame], names: list[str], workers: int, report: dict) -> dict:
    """Nearest-molecule Tanimoto of a seeded sample of each table to the whole other table (both directions)."""
    nn = {}
    for query, ref in tqdm([(names[0], names[1]), (names[1], names[0])], desc="nearest-molecule similarity"):
        sample = mols[query].sample(n=min(N_NN, len(mols[query])), random_state=SEED)
        nn[f"{query} -> {ref}"] = nearest(sample["smiles"].tolist(), mols[ref]["smiles"].tolist(), workers)
        report[f"nn {query} -> {ref}"] = similarity_summary(nn[f"{query} -> {ref}"])
    return nn


def umap_embedding(mols: dict[str, pd.DataFrame], names: list[str], tag: str, out_dp: Path) -> pd.DataFrame:
    """Joint Jaccard UMAP of a seeded sample of each table (shared molecules drawn once); save coordinates."""
    sampled = pd.concat(
        m.sample(n=min(PER_DATASET, len(m)), random_state=SEED).assign(dataset=n) for n, m in mols.items()
    )
    keys = sampled.drop_duplicates("ik14")[["ik14", "smiles"]].reset_index(drop=True)
    xy = umap.UMAP(metric="jaccard", low_memory=True).fit_transform(fingerprints(keys["smiles"].tolist()))
    keys["x"], keys["y"] = xy[:, 0], xy[:, 1]
    keys.to_csv(out_dp / f"{tag}_coords.csv", index=False)
    return sampled.merge(keys[["ik14", "x", "y"]], on="ik14")


def knn_separation(sampled: pd.DataFrame, names: list[str], report: dict) -> None:
    """KNN separation: fraction of each pooled point's K nearest neighbours that share its dataset label.
    1 means perfectly separated, ~0.5 means fully mixed. Reported per dataset and overall."""
    labels = sampled[["x", "y", "dataset"]].reset_index(drop=True)
    nn_idx = cKDTree(labels[["x", "y"]].to_numpy()).query(labels[["x", "y"]].to_numpy(), k=K_NN + 1)[1][:, 1:]
    same = (labels["dataset"].to_numpy()[:, None] == labels["dataset"].to_numpy()[nn_idx]).mean(axis=1)
    report["knn_separation"] = {"k": K_NN, "n": int(len(labels))}
    for n in names:
        report["knn_separation"][f"mean_same_{n}"] = round(float(same[labels["dataset"] == n].mean()), 3)
    report["knn_separation"]["mean_same_overall"] = round(float(same.mean()), 3)


def plot_umap(sampled: pd.DataFrame, keys: pd.DataFrame, names: list[str], len_mols: dict[str, int], n_shared: int, tag: str, out_dp: Path) -> None:
    """One joint UMAP figure: a panel per dataset plus the overlay."""
    lims = view_limits(keys, 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), facecolor="white")
    for ax in axes:
        base_map(ax, keys, lims)
    for ax, name, color in zip(axes, names, COLORS):
        sub = sampled[sampled["dataset"] == name]
        ax.scatter(sub["x"], sub["y"], s=0.25, c=color, alpha=0.5, linewidths=0, rasterized=True)
        style(ax, name, f"{len_mols[name]:,} molecules, {len(sub):,} plotted")
    for name, color in zip(names, COLORS):
        sub = sampled[sampled["dataset"] == name]
        axes[2].scatter(sub["x"], sub["y"], s=0.25, c=color, alpha=0.5, linewidths=0, rasterized=True)
        axes[2].scatter([], [], s=30, c=color, label=name)
    axes[2].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    style(axes[2], "Overlay", f"{n_shared:,} molecules in both")
    fig.suptitle("Joint UMAP of Morgan fingerprints (r=2, 2048 bits, Jaccard)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dp / f"{tag}_umap.png", dpi=150)
    plt.close(fig)


def plot_similarity_bins(nn: dict[str, np.ndarray], tag: str, out_dp: Path) -> None:
    """Nearest-molecule similarity bins, a grouped bar per direction (one bar per bin)."""
    labels = [bn[2] for bn in SIMILARITY_BINS]
    colors = [bn[3] for bn in SIMILARITY_BINS]
    fracs = pd.DataFrame({k: similarity_bin(v).value_counts(normalize=True) for k, v in nn.items()}).reindex(labels)
    directions = list(fracs.columns)
    x = np.arange(len(directions))
    width = 0.9 / len(labels)
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor="white")
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, fracs.loc[label], width, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(directions, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of query molecules")
    ax.set_xlabel("nearest molecule in the other dataset (Tanimoto bin)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dp / f"{tag}_similarity.png", dpi=150)
    plt.close(fig)


def write_report(report: dict, tag: str, out_dp: Path) -> None:
    """Write the full report as JSON plus a flattened long-form CSV for human reading."""
    (out_dp / f"{tag}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    rows = []
    for section, value in report.items():
        if isinstance(value, dict):
            for key, val in value.items():
                rows.append((f"{section}.{key}", val))
        else:
            rows.append((section, value))
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(out_dp / f"{tag}_report.csv", index=False)


def main() -> None:
    args = parse_args()
    args.out_dp.mkdir(parents=True, exist_ok=True)

    tables = (args.mols_a, args.mols_b)
    # Use the parent folder name for the dataset label when the file stem alone is not distinct
    # (e.g. two data/proc/*/mol_df.pkl), so the two inputs and output tag never collide.
    stems = [p.stem for p in tables]
    names = [p.parent.name if stems.count(s) > 1 else s for s, p in zip(stems, tables)]
    tag = f"{names[0]}_{names[1]}"
    print(f"[1/6] Loading molecules from {names[0]} and {names[1]}")
    mols = {n: load_molecules(p, args.workers) for p, n in tqdm(zip(tables, names), desc="load datasets", total=2)}
    report = halogen_counts(mols)
    for n in names:
        print(f"  {n}: {report[n]['molecules']:,} molecules")
    print(f"[2/6] Computing 2D InChIKey-block overlap")
    shared = overlap(mols, names, report)
    print(f"  {len(shared):,} molecules shared between {names[0]} and {names[1]}")
    print(f"[3/6] Nearest-molecule Tanimoto, both directions")
    nn = nearest_molecule_similarity(mols, names, args.workers, report)
    print(f"[4/6] Joint UMAP embedding")
    sampled = umap_embedding(mols, names, tag, args.out_dp)
    umap_keys = sampled.drop_duplicates("ik14")[["ik14", "x", "y"]]
    print(f"[5/6] KNN separation in UMAP space")
    knn_separation(sampled, names, report)
    print(f"[6/6] Writing figures and report to {args.out_dp}")
    plot_umap(sampled, umap_keys, names, {n: len(mols[n]) for n in names}, len(shared), tag, args.out_dp)
    plot_similarity_bins(nn, tag, args.out_dp)

    write_report(report, tag, args.out_dp)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
