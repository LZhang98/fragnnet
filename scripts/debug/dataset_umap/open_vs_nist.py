"""What open (non-NIST) MS/MS data covers, and what dropping NIST20/23 HR loses.

"Open" = every CASMI 2026 train library except the natural-product proxy ``enveda-np-examples``
(kept out so it can be scored), plus MassSpecGym benchmark 1.5 and Spectraverse. For each NIST20 ∪
NIST23 HR molecule, the nearest open neighbour (Morgan r=2, 2048 bits, Tanimoto) says whether open
data has the same or a close analogue. The NP proxy is scored against open and open + NIST.
NIST molecules are drawn on the joint UMAP of ``plot_umap.py``, coloured by that similarity.

Runs in the ``umap-viz`` env; reads ``collect_mols.py`` output (structure-only sets ignored), CASMI
train and the UMAP coordinates.

Usage (from the repository root):
    conda run -n umap-viz python scripts/debug/dataset_umap/open_vs_nist.py \
        --umap_dp output/dataset_umap_2026_10_16 --workers 10
    # Re-plot saved similarities (nist_nn_open.parquet in --nn_dp) on the map in --umap_dp:
    conda run -n umap-viz python scripts/debug/dataset_umap/open_vs_nist.py \
        --umap_dp output/dataset_umap_2026_10_16 --nn_dp output/dataset_umap_2026_10_16
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
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATASET_COLORS,
    INK,
    MUTED,
    NIST,
    NP_PROXY,
    SIMILARITY_BINS,
    base_map,
    mol_info,
    nearest,
    similarity_bin,
    style,
    view_limits,
    with_spectra,
)

PROXY_LIB = "enveda-np-examples"
BIN_LABELS = [b[2] for b in SIMILARITY_BINS]


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

    mols = with_spectra(pd.read_parquet(args.umap_dp / "mols.parquet"))
    casmi = pq.read_table(args.casmi_train_fp, columns=["ingest_lib", "normalized_smiles"]).to_pandas()
    casmi = casmi.drop_duplicates()
    # CASMI libraries not already in mols.parquet; 2D key recomputed with RDKit as collect_mols does.
    extra = casmi[~casmi["ingest_lib"].isin(["pluskal_ms2", "enveda-180", PROXY_LIB])]
    info = pd.DataFrame([mol_info(smi) for smi in extra["normalized_smiles"]],
                        columns=["smiles", "ik14", "Br", "Cl", "I"], index=extra.index)
    rows = info.assign(dataset="CASMI " + extra["ingest_lib"]).dropna(subset=["ik14"])
    allm = pd.concat([mols, rows[["dataset", "ik14", "smiles", "Br", "Cl", "I"]]], ignore_index=True)
    is_nist = allm["dataset"].isin(NIST)
    is_proxy = allm["dataset"] == NP_PROXY
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
    nist_u["bin"] = similarity_bin(nist_u["nn_open"]).to_numpy()
    nist_u.to_parquet(args.umap_dp / "nist_nn_open.parquet", index=False)
    report["nist_nn_open_bins"] = nist_u["bin"].value_counts().reindex(BIN_LABELS).to_dict()
    for el in ("Br", "Cl", "I"):
        sub = nist_u[nist_u[el]]
        report[f"nist_{el}_nn_open_bins"] = sub["bin"].value_counts().reindex(BIN_LABELS).to_dict()

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
    lims = view_limits(coords, 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4), facecolor="white")
    for ax in axes:
        base_map(ax, coords, lims)
    open_color = DATASET_COLORS["NIST20 HR"]
    open_pts = sampled[~sampled["dataset"].isin(NIST + [NP_PROXY])].drop_duplicates("ik14")
    axes[0].scatter(open_pts["x"], open_pts["y"], s=0.3, c=open_color, alpha=0.5, linewidths=0, rasterized=True)
    proxy_pts = sampled[sampled["dataset"] == NP_PROXY]
    axes[0].scatter(proxy_pts["x"], proxy_pts["y"], s=10, c=DATASET_COLORS[NP_PROXY], linewidths=0, label=NP_PROXY)
    axes[0].scatter([], [], s=30, c=open_color, label="Open data (MSGYM, Spectraverse, MSnLib, Enveda-180)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    style(axes[0], "Open data only", "Sampled molecules on the joint map")
    # Bins come from the similarity, not the stored label, so older nist_nn_open.parquet files re-plot.
    bins = similarity_bin(nist_u["nn_open"]).to_numpy()
    nist_pts = coords.merge(nist_u[["ik14"]].assign(bin=bins), on="ik14")
    for _lo, _hi, label, color in SIMILARITY_BINS:
        sub = nist_pts[nist_pts["bin"] == label]
        n_all = int((bins == label).sum())
        axes[1].scatter(sub["x"], sub["y"], s=0.6, c=color, alpha=0.7,
                        linewidths=0, rasterized=True)
        axes[1].scatter([], [], s=30, c=color, label=f"{label}: {n_all:,} of {len(nist_u):,}")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK,
                   title="NIST20 ∪ NIST23 HR molecule, nearest open neighbour (Tanimoto)", title_fontsize=9)
    style(axes[1], "What open data covers of NIST", "Coloured by the nearest open-data molecule")
    fig.text(0.01, 0.01, "Map shows the UMAP sample (25,000 per dataset); counts in the legend are all molecules. "
             "Red is drawn last, on top.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(umap_dp / "umap_open_vs_nist.png", dpi=150)


if __name__ == "__main__":
    main()
