"""Build Spectraverse fold splits whose TRAIN part keeps one spectrum per near-duplicate cluster.

Near-duplicates are spectra of the same molecule and adduct whose cosine exceeds ``--min-cosine``
and whose instrument type and collision energy match (user decisions, 2026-10-04). A missing value
(instrument "NA", or no CE) matches anything; known values must be equal. CE is compared as the
ACE readings (primary and up to two extra, rounded to 0.1 eV). This follows the curation paper's
rule (Gupta et al., Anal. Chem. 2026; cosine > 0.99), which v1.0.2 does not satisfy
(docs/notes/spectraverse_curation_paper.md); the paper groups more widely, by InChIKey first block
and polarity. Identical m/z does not prove a copy: libraries that report one m/z list per compound
give every CE the same m/z.

The decision cosine is the repo's tolerance-matched ``cos_hun_helper`` (one-to-one Hungarian
peak matching, stored base-peak-normalized intensities) at ``--ppm`` (default 5, the paper's
tolerance) with an m/z floor ``--min-mz`` (default 200, so at least 1 mDa: copies stored at 3 vs
5 decimals still match). It runs only on pairs passing a fast prefilter: cosine of intensities
binned at ``--bin-da`` on two half-offset grids (max of the two) above ``--prefilter``. On 27,626
sampled pairs every 5 ppm pair above 0.99 had a binned score of at least 0.965.

A pair with identical instrument and CE is a ``same`` edge; ``same`` edges are joined into clusters
(connected components) and each cluster keeps the spectrum with the most peaks, then the lowest
``spec_id``. A pair matched only through a missing value is a ``wildcard`` edge: it drops the
less-specified spectrum (more missing values; on a tie, fewer peaks, then higher ``spec_id``) and
never merges two labelled spectra, so a missing-value spectrum cannot bridge different
instruments or CEs. Every cluster lies inside one fold, because folds are InChIKey-disjoint. Each
near-duplicate pair is written to ``--edges-fp`` with its median absolute m/z difference over
matched peaks, which separates redistributed copies (identical m/z) from independent acquisitions
that merely look alike.

Val and test ids are copied unchanged, so held-out scores stay comparable with the source split.
The cluster table for all spectra is written to ``--cluster-fp``.

    python preproc_scripts/spectraverse/dedup_train_split.py --folds 0 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from fragnnet.utils.spec_utils import calculate_match_mzs, cos_hun_helper


def _binned(peaks_list: list, bin_da: float, offset: float) -> csr_matrix:
    """Row-normalized sparse intensity matrix, one row per spectrum, on one bin grid."""
    rows, cols, vals = [], [], []
    for r, peaks in enumerate(peaks_list):
        for mz, intensity in peaks:
            rows.append(r)
            cols.append(int(np.floor(mz / bin_da + offset)))
            vals.append(intensity)
    cols = np.asarray(cols)
    _, cols = np.unique(cols, return_inverse=True)
    mat = csr_matrix((vals, (rows, cols)), shape=(len(peaks_list), cols.max() + 1))
    mat.sum_duplicates()
    norm = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    return csr_matrix(mat.multiply(1.0 / norm[:, None]))


def group_cosine(peaks_list: list, bin_da: float) -> np.ndarray:
    """Pairwise cosine within one group: the max over two half-offset bin grids."""
    best = None
    for offset in (0.0, 0.5):
        mat = _binned(peaks_list, bin_da, offset)
        sim = (mat @ mat.T).toarray()
        best = sim if best is None else np.maximum(best, sim)
    return best


def tolerance_cosine(pa: list, pb: list, ppm: float, min_mz: float) -> tuple[float, float]:
    """Repo ``cos_hun`` at ``ppm``, and the median |m/z difference| (Da) over matched peaks."""
    # NumPy inputs take calculate_match_mzs' path without a CUDA autocast context, which cannot
    # initialize inside forked pool workers.
    ma_np = np.array([p[0] for p in pa], dtype=np.float64)
    mb_np = np.array([p[0] for p in pb], dtype=np.float64)
    mask = th.from_numpy(
        calculate_match_mzs(ma_np, mb_np, tolerance=ppm * 1e-6, relative=True, tolerance_min_mz=min_mz)
    )
    ma, mb = th.from_numpy(ma_np), th.from_numpy(mb_np)
    true_mask, pred_mask = mask.any(1), mask.any(0)
    if not true_mask.any():
        return 0.0, float("nan")
    ia = th.tensor([p[1] for p in pa], dtype=th.float64)
    ib = th.tensor([p[1] for p in pb], dtype=th.float64)
    score = float(cos_hun_helper(ia, ib, mask, true_mask, pred_mask, False, None, None))
    diff = (ma[:, None] - mb[None, :]).abs().masked_fill(~mask, float("inf")).min(1).values
    return score, float(diff[true_mask].median())


def pair_kind(inst_a: str, ce_a: str, inst_b: str, ce_b: str) -> str | None:
    """``"same"``, ``"wildcard"`` (matched through a missing value) or ``None`` (incompatible).

    A missing instrument is ``"NA"`` and a missing CE is ``""``.
    """
    inst_ok = inst_a == inst_b or "NA" in (inst_a, inst_b)
    ce_ok = ce_a == ce_b or "" in (ce_a, ce_b)
    if not (inst_ok and ce_ok):
        return None
    return "same" if (inst_a, ce_a) == (inst_b, ce_b) else "wildcard"


def _rank(inst: str, ce: str, npk: int, spec_id: int) -> tuple:
    """Sort key, best first: fewer missing values, then more peaks, then lower ``spec_id``."""
    return ((inst == "NA") + (ce == ""), -npk, spec_id)


def _group_edges(task: tuple) -> list[tuple]:
    """Near-duplicate edges ``(spec_a, spec_b, cos_5ppm, median_dmz, kind, loser)`` in one group.

    ``loser`` is the spectrum a ``wildcard`` edge drops, and -1 for a ``same`` edge.
    """
    th.set_num_threads(1)  # one thread per pool worker
    spec_ids, peaks, insts, ces, npks, bin_da, prefilter, ppm, min_mz, min_cosine = task
    sim = group_cosine(peaks, bin_da)
    edges = []
    for a, b in zip(*np.nonzero(np.triu(sim > prefilter, k=1))):
        kind = pair_kind(insts[a], ces[a], insts[b], ces[b])
        if kind is None:
            continue
        score, dmz = tolerance_cosine(peaks[a], peaks[b], ppm, min_mz)
        if score <= min_cosine:
            continue
        loser = -1
        if kind == "wildcard":
            ranks = {i: _rank(insts[i], ces[i], npks[i], spec_ids[i]) for i in (a, b)}
            loser = spec_ids[max(ranks, key=ranks.get)]
        edges.append((spec_ids[a], spec_ids[b], score, dmz, kind, loser))
    return edges


def ce_signature(spec: pd.DataFrame) -> pd.Series:
    """ACE readings (primary and up to two extra, for ramped/stepped CE) as a string; "" if none."""
    cols = [c for c in ("ace", "ace_extra_1", "ace_extra_2") if c in spec.columns]
    return spec[cols].apply(
        lambda r: "|".join(f"{v:.1f}" for v in r if pd.notna(v) and v != -1), axis=1
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-fp", default="data/proc/spectraverse/spec_df.pkl")
    parser.add_argument("--src-split-dp", default="data/split/spectraverse_benchmark")
    parser.add_argument("--dst-split-dp", default="data/split/spectraverse_benchmark_dedup")
    parser.add_argument("--cluster-fp", default="data/proc/spectraverse/near_dup_clusters.csv")
    parser.add_argument("--source-fp", default="data/proc/spectraverse/spec_source.csv")
    parser.add_argument("--edges-fp", default="data/proc/spectraverse/near_dup_edges.csv")
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--ppm", type=float, default=5.0)
    parser.add_argument("--min-mz", type=float, default=200.0)
    parser.add_argument("--bin-da", type=float, default=0.01)
    parser.add_argument("--prefilter", type=float, default=0.95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    # 1. Candidate pairs: same molecule and adduct; instrument and CE are checked per pair.
    spec = pd.read_pickle(args.spec_fp)
    spec["npk"] = spec["peaks"].map(len)
    spec["ce_sig"] = ce_signature(spec)

    # 2. Near-duplicate edges per group (5 ppm cos_hun), then clusters and wildcard drops.
    # spec_id is stored as object dtype (its to_numpy() can alias the column); use int64 copies.
    spec["spec_id"] = spec["spec_id"].astype(np.int64)
    tasks = [
        (spec["spec_id"].iloc[idx].tolist(), spec["peaks"].iloc[idx].tolist(),
         spec["inst_type"].iloc[idx].tolist(), spec["ce_sig"].iloc[idx].tolist(),
         spec["npk"].iloc[idx].tolist(), args.bin_da, args.prefilter, args.ppm, args.min_mz,
         args.min_cosine)
        for idx in spec.groupby(["mol_id", "prec_type"]).indices.values() if len(idx) >= 2
    ]
    with Pool(args.workers) as pool:
        edge_lists = pool.map(_group_edges, tasks, chunksize=64)
    edges = pd.DataFrame(
        [e for el in edge_lists for e in el],
        columns=["spec_a", "spec_b", "cos_5ppm", "median_dmz", "kind", "loser"],
    )
    edges.to_csv(args.edges_fp, index=False)
    same = edges[edges["kind"] == "same"]
    pos = pd.Series(np.arange(len(spec)), index=spec["spec_id"].to_numpy())
    adj = coo_matrix(
        (np.ones(len(same)), (pos.loc[same.spec_a].to_numpy(), pos.loc[same.spec_b].to_numpy())),
        shape=(len(spec), len(spec)),
    )
    _, label = connected_components(adj, directed=False)
    spec["label"] = label
    rep = (
        spec.sort_values(["npk", "spec_id"], ascending=[False, True])
        .groupby("label")["spec_id"]
        .first()
    )
    cluster = spec["label"].map(rep).to_numpy(dtype=np.int64, copy=True)
    # A wildcard loser is dropped in favour of its better-specified twin.
    wild = edges[edges["kind"] == "wildcard"]
    winner = np.where(wild["loser"] == wild["spec_a"], wild["spec_b"], wild["spec_a"])
    winner_of = pd.Series(winner, index=wild["loser"].to_numpy()).groupby(level=0).first()
    is_loser = spec["spec_id"].isin(winner_of.index).to_numpy()
    cluster[is_loser] = spec.loc[is_loser, "spec_id"].map(winner_of).to_numpy()
    if spec["spec_id"].duplicated().any():
        raise RuntimeError("spec_id was modified during clustering")
    spec["cluster"] = cluster
    spec["kept"] = (spec["cluster"] == spec["spec_id"]) & ~is_loser
    spec["cluster_size"] = spec.groupby("cluster")["spec_id"].transform("size")
    spec[["spec_id", "cluster", "cluster_size", "kept"]].to_csv(args.cluster_fp, index=False)

    # 3. Drop non-kept spectra from train; copy val/test.
    drop = set(spec.loc[~spec["kept"], "spec_id"])
    stats = {
        "min_cosine": args.min_cosine, "ppm": args.ppm, "min_mz": args.min_mz,
        "prefilter": args.prefilter, "bin_da": args.bin_da, "n_spectra": len(spec),
        "n_edges": len(edges), "n_same_edges": len(same), "n_wildcard_edges": len(wild),
        "n_wildcard_losers": int(is_loser.sum()),
        "n_dropped_all_splits": len(drop),
        "n_clusters_gt1": int((spec.loc[spec["kept"], "cluster_size"] > 1).sum()),
        "folds": {},
    }
    if Path(args.source_fp).exists():
        source = pd.read_csv(args.source_fp).set_index("spec_id")["source"]
        dropped_src = source.loc[sorted(drop)].value_counts()
        total_src = source.value_counts()
        stats["dropped_share_by_source"] = (
            (dropped_src / total_src).fillna(0).sort_values(ascending=False).round(4).to_dict()
        )
    for fold in args.folds:
        src = Path(args.src_split_dp) / f"cv{fold}"
        dst = Path(args.dst_split_dp) / f"cv{fold}"
        dst.mkdir(parents=True, exist_ok=True)
        train = pd.read_csv(src / "train_ids.csv")
        kept = train[~train["spec_id"].isin(drop)]
        kept.to_csv(dst / "train_ids.csv", index=False)
        for name in ("val_ids.csv", "test_ids.csv"):
            shutil.copyfile(src / name, dst / name)
        held_out = pd.concat([pd.read_csv(src / f"{n}_ids.csv") for n in ("val", "test")])
        stats["folds"][f"cv{fold}"] = {
            "train_before": len(train), "train_after": len(kept),
            "train_dropped": len(train) - len(kept),
            "train_mols_before": int(train["mol_id"].nunique()),
            "train_mols_after": int(kept["mol_id"].nunique()),
            "val_test_near_dups_left_in_place": int(held_out["spec_id"].isin(drop).sum()),
        }
    (Path(args.dst_split_dp) / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
