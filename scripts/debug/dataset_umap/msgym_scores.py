"""MassSpecGym test performance of a MASSter checkpoint on the dataset chemical-space map.

The 2026-10-16 map (``plot_umap.py``) was fitted on every MassSpecGym molecule, so test and train
molecules already have coordinates in ``umap_sampled_mols.parquet``; nothing is refitted.

Two stages (run from the repository root; both run in ``FRAGNNET-GPU``):

1. ``collect``: per formula-test query, the instrument-tolerance Hungarian cosine of the true
   candidate's saved prediction (linear intensities, peaks rounded to 4 decimals and summed, L2
   norm, one-to-one matching within ``rel * max(min(mz, 1500), min_mz)`` of each query peak, the
   ``cos_hun_inst_tol`` matching; precursor kept; the truth listed first when a query has two), the
   logged ``cos_sim_0.01``, and the true
   molecule's tie-aware expected hit@1 / hit@5 per retrieval score. Per molecule (2D InChIKey):
   means of those, and the nearest-train Tanimoto. Writes ``queries.parquet``, ``mols.parquet``
   and ``train_mols.parquet``.
2. ``plot``: train molecules in grey, test molecules coloured by score (low scores on top), and
   the per-molecule cosine against nearest-train Tanimoto.

    /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -I scripts/debug/dataset_umap/msgym_scores.py collect \
        --out_dp output/msgym_umap_scores_2026_10_18
    /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -I scripts/debug/dataset_umap/msgym_scores.py plot \
        --out_dp output/msgym_umap_scores_2026_10_18
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/debug/dataset_umap"))
sys.path.insert(0, str(ROOT / "src"))
from common import GREY, INK, MUTED, mol_info_frame, nearest, view_limits  # noqa: E402

from fragnnet.utils.retrieval_utils import MSPRED_UPPER_LIMIT, expected_hits  # noqa: E402

INST_TOL = {"ORB": (1e-5, 200.0), "QTOF": (2e-5, 750.0)}
RETRIEVAL_SCORES = ("loglik_cal", "mspred_ent")
# One hue, dark = low score, so failures stand out (as in ``scripts/debug/ei/hr_ei_umap.py``).
LOW_DARK = ["#7f1d1d", "#e34948", "#f6c9c4"]


def _aggregate(mzs: np.ndarray, ints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round m/z to 4 decimals, sum intensities per rounded m/z, drop zeros, L2-normalise."""
    keys, inv = np.unique(np.round(mzs, 4), return_inverse=True)
    summed = np.bincount(inv, weights=ints, minlength=len(keys))
    keep = summed > 0
    keys, summed = keys[keep], summed[keep]
    norm = np.linalg.norm(summed)
    return keys, (summed / norm if norm > 0 else summed)


def cosine_inst_tol(true_mzs, true_ints, pred_mzs, pred_ints, rel: float, min_mz: float) -> float:
    """One-to-one (Hungarian) cosine with each query peak's tolerance ``rel * max(min(mz, 1500), min_mz)``."""
    tm, tq = _aggregate(np.asarray(true_mzs, float), np.asarray(true_ints, float))
    order = np.argsort(pred_mzs)
    pm, pp = _aggregate(np.asarray(pred_mzs, float)[order], np.asarray(pred_ints, float)[order])
    if len(tm) == 0 or len(pm) == 0:
        return 0.0
    tol = rel * np.maximum(np.minimum(tm, MSPRED_UPPER_LIMIT), min_mz)
    lo = np.searchsorted(pm, tm - tol, "left")
    hi = np.searchsorted(pm, tm + tol, "right")
    rows = np.repeat(np.arange(len(tm)), hi - lo)
    if len(rows) == 0:
        return 0.0
    cols = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
    r_ids, r_inv = np.unique(rows, return_inverse=True)
    c_ids, c_inv = np.unique(cols, return_inverse=True)
    w = np.zeros((len(r_ids), len(c_ids)))
    w[r_inv, c_inv] = tq[rows] * pp[cols]
    r, c = linear_sum_assignment(w, maximize=True)
    return float(w[r, c].sum())


def _score_shard(job: tuple) -> pd.DataFrame:
    """Per-query scores of one retrieval shard."""
    shard_dp, run, query_peaks = job
    run_dp = shard_dp / run
    cand = pd.read_pickle(run_dp / "scores.pkl")
    out = []
    for qid, g in cand.groupby("query_spec_id"):
        row = {"spec_id": qid}
        for s in RETRIEVAL_SCORES:
            row[f"hit1_{s}"], row[f"hit5_{s}"] = expected_hits(g[s].to_numpy(), g["is_truth"].to_numpy(), (1, 5))
        out.append(row)
    hits = pd.DataFrame(out)
    # 19 queries have a second truth (same 2D key, another candidate entry); the listed truth is
    # MassSpecGym's first candidate.
    first = pd.read_csv(shard_dp / "candidates.csv", usecols=["cand_uid", "list_pos"])
    truth = cand[cand["is_truth"]].merge(first[first["list_pos"] == 0], on="cand_uid")
    truth = truth[["cand_uid", "query_spec_id", "cos_sim_0.01"]]
    rows = []
    with h5py.File(run_dp / "predictions_test.hdf5", "r") as f:
        pos = pd.Series(np.arange(len(f["spec_id"])), index=f["spec_id"][:])
        starts, ends = f["start_points"][:], f["end_points"][:]
        mzs, ints = f["pred_mzs"], f["pred_ints"]
        for uid, qid, cos01 in truth.itertuples(index=False):
            i = pos[uid]
            pm, pi = mzs[starts[i] : ends[i]], ints[starts[i] : ends[i]]
            peaks, inst = query_peaks[qid]
            t = np.asarray(peaks, float)
            rows.append({"spec_id": qid, "inst_type": inst, "cos_sim_0.01": cos01,
                         "cos_inst_tol": cosine_inst_tol(t[:, 0], t[:, 1], pm, pi, *INST_TOL[inst])})
    return pd.DataFrame(rows).merge(hits, on="spec_id")


def collect(args: argparse.Namespace) -> None:
    args.out_dp.mkdir(parents=True, exist_ok=True)
    queries = pd.read_csv(args.retrieval_dp / "queries.csv")
    spec = pd.read_pickle(args.data_dp / "spec_df.pkl").set_index("spec_id")
    spec = spec.loc[queries["spec_id"]]
    query_peaks = dict(zip(spec.index, zip(spec["peaks"], spec["inst_type"])))
    shards = sorted(p for p in args.retrieval_dp.glob("shard_*") if (p / args.run).is_dir())
    jobs = [(p, args.run, query_peaks) for p in shards]
    with Pool(min(args.workers, len(jobs))) as pool:
        scores = pd.concat(pool.map(_score_shard, jobs), ignore_index=True)
    if len(scores) != len(queries):
        raise ValueError(f"{len(scores)} scored queries, expected {len(queries)}")

    mol = pd.read_pickle(args.data_dp / "mol_df.pkl")[["mol_id", "smiles"]]
    train_ids = set(pd.read_csv(args.split_dp / "train_ids.csv")["mol_id"])
    train = mol[mol["mol_id"].isin(train_ids)]
    train = mol_info_frame(train["smiles"].tolist(), args.workers)[["smiles", "ik14"]]
    train = train.dropna(subset=["ik14"]).drop_duplicates("ik14")
    train.to_parquet(args.out_dp / "train_mols.parquet", index=False)

    q = queries[["spec_id", "mol_id"]].merge(scores, on="spec_id").merge(mol, on="mol_id")
    q = q.merge(mol_info_frame(q["smiles"].unique().tolist(), args.workers)[["smiles", "ik14"]], on="smiles")
    q.to_parquet(args.out_dp / "queries.parquet", index=False)
    metrics = ["cos_inst_tol", "cos_sim_0.01", *[f"hit{k}_{s}" for s in RETRIEVAL_SCORES for k in (1, 5)]]
    mols = q.groupby("ik14", as_index=False).agg(
        smiles=("smiles", "first"), n_spectra=("spec_id", "size"),
        frac_orb=("inst_type", lambda s: float((s == "ORB").mean())), **{m: (m, "mean") for m in metrics})
    mols["nn_train"] = nearest(mols["smiles"].tolist(), train["smiles"].tolist(), args.workers)
    mols.to_parquet(args.out_dp / "mols.parquet", index=False)
    print(q[metrics].mean().round(4).to_string())
    print(f"{len(q)} queries, {len(mols)} molecules, {len(train)} train molecules")


def plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = pd.read_parquet(args.map_dp / "umap_sampled_mols.parquet")
    coords = coords[coords["dataset"] == "MassSpecGym"][["ik14", "x", "y"]]
    train = pd.read_parquet(args.out_dp / "train_mols.parquet").merge(coords, on="ik14")
    mols = pd.read_parquet(args.out_dp / "mols.parquet")
    placed = mols.merge(coords, on="ik14")
    lims = view_limits(coords, 0.5)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("low_dark", LOW_DARK)

    fig = plt.figure(figsize=(18, 12.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 0.55])
    panels = [("cos_inst_tol", "True-molecule cosine at instrument tolerance (mean per molecule)"),
              ("hit1_loglik_cal", "Retrieval hit@1 under loglik_cal (fraction of the molecule's queries)")]
    for i, (col, label) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        ax.scatter(train["x"], train["y"], s=2, c="#bdbcb5", alpha=0.6, linewidths=0, rasterized=True)
        order = placed.sort_values(col, ascending=False)
        points = ax.scatter(order["x"], order["y"], s=10, c=order[col], cmap=cmap, vmin=0, vmax=1,
                            edgecolors=MUTED, linewidths=0.25)
        ax.set(xlim=lims[0], ylim=lims[1], xticks=[], yticks=[])
        for side in ax.spines.values():
            side.set_visible(False)
        bar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.01)
        bar.set_label(label, color=INK, fontsize=9)
        title = ("Formula-test molecules by cosine" if i == 0 else "Same molecules by retrieval hit@1")
        ax.set_title(f"{title} (low drawn on top; grey = {len(train):,} train molecules)",
                     loc="left", fontsize=11, color=INK)

    # Score against nearest-train Tanimoto: per-molecule points, bin means with a 95% bootstrap
    # interval over molecules, each drawn at its bin's median Tanimoto.
    # The MCES split keeps test far from train (median nearest-train Tanimoto 0.36), so bins are fine low down.
    edges = [0, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.01]
    rng = np.random.default_rng(0)
    bins = pd.cut(mols["nn_train"], edges, right=False)
    for i, (col, label) in enumerate(panels):
        ax = fig.add_subplot(grid[1, i])
        ax.scatter(mols["nn_train"], mols[col], s=4, c=GREY, linewidths=0, rasterized=True)
        rows = []
        for _, g in mols.groupby(bins, observed=True):
            v = g[col].to_numpy()
            boot = rng.choice(v, (2000, len(v))).mean(1)
            rows.append((g["nn_train"].median(), v.mean(), *np.percentile(boot, [2.5, 97.5]), len(v)))
        x, mean, lo, hi, n = map(np.array, zip(*rows))
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="o-", color="#7f1d1d", ms=6, lw=2, capsize=0)
        for xi, ni in zip(x, n):
            ax.annotate(f"n={ni}", (xi, 1.02), ha="center", fontsize=8, color=MUTED,
                        xycoords=("data", "axes fraction"))
        rho, p = spearmanr(mols["nn_train"], mols[col], nan_policy="omit")
        ax.set_xlabel("Nearest-train Tanimoto (Morgan r=2, 2048 bits)", color=INK, fontsize=9)
        ax.set_ylabel(label.split(" (")[0], color=INK, fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.text(0.01, 0.97, f"Spearman {rho:.2f} (p {p:.1g}); mean and 95% bootstrap interval per bin",
                transform=ax.transAxes, fontsize=9, color=INK, va="top")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    missing = len(mols) - len(placed)
    fig.text(0.01, 0.005, f"{args.label}. Map: 2026-10-16 joint MS/MS fit (Morgan r=2, 2048 bits, Jaccard), "
             f"zoomed to MassSpecGym's 0.5-99.5 percentiles; {len(placed):,} of {len(mols):,} test molecules placed"
             + (f" ({missing} unmatched InChIKey blocks)" if missing else "")
             + ". Instrument tolerance: ORB 10 ppm (200 Da floor), QTOF 20 ppm (750 Da floor).",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = args.out_dp / "umap_msgym_test_scores.png"
    fig.savefig(out, dpi=150)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out_dp", type=Path, required=True)
    c.add_argument("--retrieval_dp", type=Path, default=ROOT / "output/msgym_retrieval/formula_test")
    c.add_argument("--run", default="msgym_masster_sweep_leader_720jwmai_f44tzxoc")
    c.add_argument("--data_dp", type=Path, default=ROOT / "data/proc/ms_gym_benchmark_1_5")
    c.add_argument("--split_dp", type=Path, default=ROOT / "data/split/ms_gym_benchmark_1_5")
    c.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("plot")
    p.add_argument("--out_dp", type=Path, required=True)
    p.add_argument("--map_dp", type=Path, default=ROOT / "output/dataset_umap_2026_10_16")
    p.add_argument("--label", default="Sweep leader f44tzxoc, MassSpecGym formula-test")
    args = ap.parse_args()
    collect(args) if args.stage == "collect" else plot(args)


if __name__ == "__main__":
    main()
