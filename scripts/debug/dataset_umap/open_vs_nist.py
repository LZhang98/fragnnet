"""What open (non-NIST) MS/MS data covers, and what dropping NIST20/23 HR loses.

"Open" = every CASMI 2026 train library except the natural-product proxy ``enveda-np-examples``
(kept out so it can be scored), plus MassSpecGym benchmark 1.5 and Spectraverse. For each NIST20 ∪
NIST23 HR molecule, the nearest open neighbour (Morgan r=2, 2048 bits, Tanimoto) says whether open
data has the same or a close analogue. The NP proxy is scored against open and open + NIST.
NIST molecules are drawn on the joint UMAP of ``plot_umap.py``, coloured by that similarity.

Runs in the ``umap-viz`` env; reads ``collect_mols.py`` output, CASMI train and the UMAP coordinates.

Usage (from the repository root):
    conda run -n umap-viz python scripts/debug/dataset_umap/open_vs_nist.py \
        --umap_dp output/dataset_umap_2026_10_16 --workers 10
    # Re-plot saved similarities on another map (e.g. the one fitted with HMDB):
    conda run -n umap-viz python scripts/debug/dataset_umap/open_vs_nist.py \
        --umap_dp output/dataset_umap_hmdb_2026_10_17 --nn_dp output/dataset_umap_2026_10_16
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402

RDLogger.DisableLog("rdApp.*")

NIST = ["NIST20 HR", "NIST23 HR"]
PROXY_LIB = "enveda-np-examples"
BINS = [(1.0, 1.01, "same 2D molecule or identical fingerprint"), (0.7, 1.0, "close analogue ≥0.7"),
        (0.4, 0.7, "related 0.4–0.7"), (0.0, 0.4, "no analogue <0.4")]
BIN_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]  # reference palette slots 1, 3, 4, 8
GREY, INK, MUTED = "#e4e3de", "#1a1a19", "#6b6a63"

_REF: list = []


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol) if mol is not None else None


def _init(ref_smiles: list[str]) -> None:
    global _REF
    _REF = [fp for fp in map(_fp, ref_smiles) if fp is not None]


def _nn(smiles_chunk: list[str]) -> list[float]:
    out = []
    for smi in smiles_chunk:
        fp = _fp(smi)
        out.append(max(DataStructs.BulkTanimotoSimilarity(fp, _REF)) if fp is not None else np.nan)
    return out


def nearest(query: list[str], ref: list[str], workers: int) -> np.ndarray:
    chunks = [query[i : i + 200] for i in range(0, len(query), 200)]
    with Pool(workers, initializer=_init, initargs=(ref,)) as pool:
        return np.array([v for part in pool.map(_nn, chunks) for v in part])


def counts(frame: pd.DataFrame) -> dict:
    mols = frame.drop_duplicates("ik14")
    return {"molecules": len(mols), "Br": int(mols["Br"].sum()), "Cl": int(mols["Cl"].sum()),
            "I": int(mols["I"].sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--umap_dp", type=Path, required=True)
    ap.add_argument("--casmi_train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--nn_dp", type=Path, default=None,
                    help="Reuse nist_nn_open.parquet from this directory and only plot onto --umap_dp")
    args = ap.parse_args()
    if args.nn_dp is not None:
        plot(args.umap_dp, pd.read_parquet(args.nn_dp / "nist_nn_open.parquet"))
        return

    mols = pd.read_parquet(args.umap_dp / "mols.parquet")
    casmi = pq.read_table(args.casmi_train_fp, columns=["ingest_lib", "normalized_smiles"]).to_pandas()
    casmi = casmi.drop_duplicates()
    # CASMI libraries not already in mols.parquet; 2D key recomputed with RDKit as collect_mols does.
    extra = casmi[~casmi["ingest_lib"].isin(["pluskal_ms2", "enveda-180", PROXY_LIB])]
    rows = []
    for lib, smi in extra.itertuples(index=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        sym = {a.GetSymbol() for a in mol.GetAtoms()}
        rows.append({"dataset": f"CASMI {lib}", "ik14": Chem.MolToInchiKey(mol)[:14], "smiles": smi,
                     "Br": "Br" in sym, "Cl": "Cl" in sym, "I": "I" in sym})
    allm = pd.concat([mols, pd.DataFrame(rows)], ignore_index=True)
    is_nist = allm["dataset"].isin(NIST)
    is_proxy = allm["dataset"] == "CASMI NP proxy"
    open_m = allm[~is_nist & ~is_proxy]
    nist_m = allm[is_nist]
    open_keys, nist_keys = set(open_m["ik14"]), set(nist_m["ik14"])

    report = {
        "open": counts(open_m),
        "nist20_or_23": counts(nist_m),
        "open_plus_nist": counts(pd.concat([open_m, nist_m])),
        "nist_only": counts(nist_m[~nist_m["ik14"].isin(open_keys)]),
        "open_per_source": {d: counts(g) for d, g in open_m.groupby("dataset")},
    }

    ref_open = open_m.drop_duplicates("ik14")
    nist_u = nist_m.drop_duplicates("ik14").copy()
    nist_u["nn_open"] = nearest(nist_u["smiles"].tolist(), ref_open["smiles"].tolist(), args.workers)
    nist_u.loc[nist_u["ik14"].isin(open_keys), "nn_open"] = 1.0
    nist_u["bin"] = pd.cut(nist_u["nn_open"], [0, 0.4, 0.7, 0.999999, 1.01], right=False,
                           labels=[b[2] for b in BINS[::-1]])
    nist_u.to_parquet(args.umap_dp / "nist_nn_open.parquet", index=False)
    report["nist_nn_open_bins"] = nist_u["bin"].value_counts().reindex([b[2] for b in BINS]).to_dict()
    for el in ("Br", "Cl", "I"):
        sub = nist_u[nist_u[el]]
        report[f"nist_{el}_nn_open_bins"] = sub["bin"].value_counts().reindex([b[2] for b in BINS]).to_dict()

    proxy = allm[is_proxy].drop_duplicates("ik14")
    proxy_open = nearest(proxy["smiles"].tolist(), ref_open["smiles"].tolist(), args.workers)
    ref_all = pd.concat([open_m, nist_m]).drop_duplicates("ik14")
    proxy_all = nearest(proxy["smiles"].tolist(), ref_all["smiles"].tolist(), args.workers)
    report["np_proxy"] = {
        "molecules": len(proxy),
        "exact_in_open": int(proxy["ik14"].isin(open_keys).sum()),
        "exact_in_nist": int(proxy["ik14"].isin(nist_keys).sum()),
        "nn_open_median": float(np.nanmedian(proxy_open)),
        "nn_open_plus_nist_median": float(np.nanmedian(proxy_all)),
        "frac_nn_ge_0.7_open": float(np.nanmean(proxy_open >= 0.7)),
        "frac_nn_ge_0.7_open_plus_nist": float(np.nanmean(proxy_all >= 0.7)),
        "molecules_whose_best_neighbour_improves_with_nist": int((proxy_all > proxy_open + 1e-9).sum()),
    }
    (args.umap_dp / "open_vs_nist.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))

    plot(args.umap_dp, nist_u)


def plot(umap_dp: Path, nist_u: pd.DataFrame) -> None:
    """Open data on the joint map, and NIST molecules by their nearest open neighbour."""
    coords = pd.read_parquet(umap_dp / "umap_coords.parquet")[["ik14", "x", "y"]]
    sampled = pd.read_parquet(umap_dp / "umap_sampled_mols.parquet")
    lims = [np.percentile(coords[c], [2, 98]) for c in ("x", "y")]
    lims = [lo_hi + np.array([-0.08, 0.08]) * (lo_hi[1] - lo_hi[0]) for lo_hi in lims]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), facecolor="white")
    for ax in axes:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.scatter(coords["x"], coords["y"], s=0.2, c=GREY, linewidths=0, rasterized=True)
    # HMDB has no spectra: drawn as its own layer, never counted as open MS/MS data.
    open_pts = sampled[~sampled["dataset"].isin(NIST + ["CASMI NP proxy", "HMDB"])].drop_duplicates("ik14")
    hmdb_pts = sampled[sampled["dataset"] == "HMDB"]
    if len(hmdb_pts):
        axes[0].scatter(hmdb_pts["x"], hmdb_pts["y"], s=0.3, c="#e34948", alpha=0.35, linewidths=0, rasterized=True)
    axes[0].scatter(open_pts["x"], open_pts["y"], s=0.3, c="#2a78d6", alpha=0.5, linewidths=0, rasterized=True)
    proxy_pts = sampled[sampled["dataset"] == "CASMI NP proxy"]
    axes[0].scatter(proxy_pts["x"], proxy_pts["y"], s=10, c="#6250d6", linewidths=0, label="CASMI NP proxy")
    axes[0].scatter([], [], s=30, c="#2a78d6", label="Open data (MSGYM, Spectraverse, MSnLib, Enveda-180)")
    if len(hmdb_pts):
        axes[0].scatter([], [], s=30, c="#e34948", label="HMDB (structures only, no spectra; under open data)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    axes[0].set_title("Open data only (sampled molecules on the joint map)", loc="left", fontsize=12, color=INK)
    nist_pts = coords.merge(nist_u[["ik14", "bin"]], on="ik14")
    for (_lo, _hi, label), color in list(zip(BINS, BIN_COLORS)):
        sub = nist_pts[nist_pts["bin"] == label]
        n_all = int((nist_u["bin"] == label).sum())
        axes[1].scatter(sub["x"], sub["y"], s=0.6, c=color, alpha=0.7,
                        linewidths=0, rasterized=True)
        axes[1].scatter([], [], s=30, c=color, label=f"{label}: {n_all:,} of {len(nist_u):,}")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK,
                   title="NIST20 ∪ NIST23 HR molecule, nearest open neighbour (Tanimoto)", title_fontsize=9)
    axes[1].set_title("What open data covers of NIST", loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "Map shows the UMAP sample (25,000 per dataset); counts in the legend are all molecules. "
             "Red is drawn last, on top.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(umap_dp / "umap_open_vs_nist.png", dpi=150)


if __name__ == "__main__":
    main()
