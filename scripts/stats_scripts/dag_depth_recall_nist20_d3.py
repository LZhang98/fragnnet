"""Fast per-depth peak recall for NIST20 D3 (vectorized numpy matching).

Pass 1 (node stats) was already run by dag_depth_stats_nist20_d3.py.
This script adds the peak recall analysis on a 2 000-molecule random sample.

Usage:
    conda run -n FRAGNNET-GPU python scripts/stats_scripts/dag_depth_recall_nist20_d3.py
"""

import bz2
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.frag_utils import PREC_TYPE_TO_CMF_MASS_DIFF

DAG_DP = "/home/feiw/gitRepos/frag-gnn/data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags"
SPEC_DF_FP = "/home/feiw/gitRepos/frag-gnn/data/proc/nist20/spec_df.pkl"
OUT_DIR = "/home/feiw/gitRepos/frag-gnn/data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse"

MAX_DEPTH = 3
H_TRANSFER = 4     # h_delta = 0, ±1, ±2, ±3, ±4

# Per-instrument-type tolerance: rel (ppm as fraction) + min_mz floor
# tol(mz) = rel * max(mz, min_mz)
INST_TOL: dict[str, dict] = {
    "OB":   {"rel": 1e-5, "min_mz": 200.0},   # 10 ppm, floor 0.002 Da
    "OB_F":   {"rel": 1e-5, "min_mz": 200.0},   # 10 ppm, floor 0.002 Da
    "QTOF": {"rel": 2e-5, "min_mz": 750.0},   # 20 ppm, floor 0.015 Da
    "IT":   {"rel": 1e-5, "min_mz": 200.0},   # same as global default
}

PREC_TYPES = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"}
INST_TYPES  = {"OB", "OB_F", "QTOF", "IT"}
FRAG_MODES  = {"CID", "HCD"}

SAMPLE_N    = 2000
RANDOM_SEED = 42


def load_dag(fp: str) -> dict:
    with bz2.open(fp, "rb") as f:
        return pickle.load(f)


def formula_depth_map(dag_d: dict) -> dict[int, int]:
    """Return {formula_idx: min_depth} for all valid (non-sentinel) formula indices."""
    dag = dag_d["dag"]
    feat_start = dag.node_feat_idxs[0, 3].item()
    feat_end   = dag.node_feat_idxs[0, 4].item()
    formula_idx_block = dag.x[:, feat_start:feat_end].numpy()   # (num_nodes, 9)
    nodes_min_depth   = dag_d["nodes_min_depth"]

    f_depth: dict[int, int] = {}
    for node_idx, dep in enumerate(nodes_min_depth):
        for f_idx in formula_idx_block[node_idx]:
            f_idx = int(f_idx)
            if f_idx == 0:
                continue
            if f_idx not in f_depth or dep < f_depth[f_idx]:
                f_depth[f_idx] = int(dep)
    return f_depth


def inst_tolerances(true_mzs: np.ndarray, inst_type: str) -> np.ndarray:
    """Return per-peak tolerance array using inst_tol config.

    Args:
        true_mzs: Observed m/z values of shape (n_peaks,).
        inst_type: Instrument type string.

    Returns:
        Per-peak tolerance array of shape (n_peaks,).
    """
    cfg = INST_TOL[inst_type]
    return cfg["rel"] * np.maximum(true_mzs, cfg["min_mz"])


def vectorized_recall(
    peaks: list,
    formula_peak_mzs_np: np.ndarray,   # (num_formulas, 5) float32
    formula_peak_probs_np: np.ndarray,  # (num_formulas, 5) float32
    allowed_idxs: set[int],
    prec_type: str,
    inst_type: str,
) -> tuple[float, float]:
    """Fast vectorized recall / wrecall using per-instrument ppm tolerance."""
    if not peaks or not allowed_idxs:
        return 0.0, 0.0

    true_mzs = np.array([p[0] for p in peaks], dtype=np.float64)
    true_ints = np.array([p[1] for p in peaks], dtype=np.float64)
    total_int = true_ints.sum()
    if total_int == 0:
        return 0.0, 0.0

    idx_list = list(allowed_idxs)
    theo_raw  = formula_peak_mzs_np[idx_list, 0].astype(np.float64)
    prob_raw  = formula_peak_probs_np[idx_list, 0]
    theo_mono = theo_raw[prob_raw > 0.0]

    if len(theo_mono) == 0:
        return 0.0, 0.0

    std_diff = PREC_TYPE_TO_MASS_DIFF.get(prec_type, 0.0)
    cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF.get(prec_type, std_diff)

    theo_std = theo_mono + std_diff
    theo_cmf = theo_mono + cmf_diff
    theo_all = theo_std if std_diff == cmf_diff else np.unique(
        np.concatenate([theo_std, theo_cmf])
    )

    # Per-peak tolerance (ppm-based with floor)
    tols = inst_tolerances(true_mzs, inst_type)   # (n_peaks,)

    # diffs: (n_peaks, n_theo); tols: (n_peaks, 1)
    diffs   = np.abs(true_mzs[:, None] - theo_all[None, :])
    matched = np.any(diffs < tols[:, None], axis=1)

    recall  = matched.mean()
    wrecall = (true_ints * matched).sum() / total_int
    return float(recall), float(wrecall)


def filtered_allowed(
    f_depth: dict[int, int], dep_thresh: int
) -> set[int]:
    return {f for f, fd in f_depth.items() if fd <= dep_thresh}


# idx_by_h_delta encoding: h_delta=0->0, h_delta=-1->1, h_delta=+1->2,
#   h_delta=-2->3, h_delta=+2->4, ..., h_delta=-k->2k-1, h_delta=+k->2k
# For H_TRANSFER=4: use positions 0..8 (h_delta = 0, ±1, ±2, ±3, ±4)
H_DELTA_IDXS = list(range(1 + 2 * H_TRANSFER))  # [0,1,2,3,4,5,6,7,8]


def ibd_allowed(idx_by_h_delta: list[set], allowed: set[int]) -> set[int]:
    """Return union of formula indices across h_delta positions, filtered to allowed."""
    out: set[int] = set()
    for i in H_DELTA_IDXS:
        out |= (idx_by_h_delta[i] & allowed)
    return out


def process_mol(mol_id: int, spec_rows: list[dict]) -> list[dict]:
    fp = os.path.join(DAG_DP, f"{mol_id}.pkl.bz2")
    if not os.path.exists(fp):
        return []
    dag_d = load_dag(fp)

    f_depth = formula_depth_map(dag_d)
    idx_by_h_delta = dag_d["idx_by_h_delta"]
    fpm  = dag_d["formula_peak_mzs"].numpy()    # (n_formulas, 5)
    fpp  = dag_d["formula_peak_probs"].numpy()   # (n_formulas, 5)

    # Pre-compute allowed sets per depth
    depth_allowed = {
        dep: ibd_allowed(idx_by_h_delta, filtered_allowed(f_depth, dep))
        for dep in range(MAX_DEPTH + 1)
    }

    out = []
    for sr in spec_rows:
        peaks = sr["peaks"]
        if len(peaks) == 0:
            continue
        inst = sr["inst_type"]
        for dep in range(MAX_DEPTH + 1):
            r, w = vectorized_recall(peaks, fpm, fpp, depth_allowed[dep],
                                     sr["prec_type"], inst)
            out.append({
                "mol_id":       mol_id,
                "spec_id":      sr["spec_id"],
                "inst_type":    inst,
                "frag_mode":    sr["frag_mode"],
                "prec_type":    sr["prec_type"],
                "depth_thresh": dep,
                "recall":  r,
                "wrecall": w,
            })
    return out


def main() -> None:
    dag_mol_ids = sorted(int(fn.split(".")[0]) for fn in os.listdir(DAG_DP))
    print(f"Total DAG molecules: {len(dag_mol_ids):,}", flush=True)

    # Load and filter spec_df
    print("Loading spec_df...", flush=True)
    spec_df = pd.read_pickle(SPEC_DF_FP)
    spec_filt = spec_df[
        spec_df["prec_type"].isin(PREC_TYPES)
        & spec_df["inst_type"].isin(INST_TYPES)
        & spec_df["frag_mode"].isin(FRAG_MODES)
        & (spec_df["spec_type"] == "MS2")
        & spec_df["mol_id"].isin(set(dag_mol_ids))
    ][["spec_id", "mol_id", "prec_mz", "prec_type", "inst_type", "frag_mode", "peaks"]].copy()
    del spec_df
    print(f"  {len(spec_filt):,} spectra, {spec_filt['mol_id'].nunique():,} mols", flush=True)

    # Sample molecules
    rng = random.Random(RANDOM_SEED)
    sample_mol_ids = set(rng.sample(dag_mol_ids, SAMPLE_N))

    spec_sample = spec_filt[spec_filt["mol_id"].isin(sample_mol_ids)]
    spec_by_mol: dict[int, list] = {}
    for mol_id, grp in spec_sample.groupby("mol_id"):
        spec_by_mol[int(mol_id)] = grp[
            ["spec_id", "prec_mz", "prec_type", "inst_type", "frag_mode", "peaks"]
        ].to_dict("records")
    del spec_sample, spec_filt
    print(f"  Sample: {len(spec_by_mol):,} mols", flush=True)

    # Per-mol recall
    print(f"Running per-depth recall on {SAMPLE_N} mols...", flush=True)
    all_rows = []
    for i, mol_id in enumerate(sorted(sample_mol_ids)):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{SAMPLE_N}", flush=True)
        rows = process_mol(mol_id, spec_by_mol.get(mol_id, []))
        all_rows.extend(rows)

    spec_df_out = pd.DataFrame(all_rows)
    out_fp = os.path.join(OUT_DIR, "dag_depth_recall_sample.pkl")
    spec_df_out.to_pickle(out_fp)
    print(f"Saved: {out_fp}  ({len(spec_df_out):,} rows)", flush=True)

    # Summary
    print(f"\n{'='*72}", flush=True)
    print(f"NIST20 D3 — per-depth peak recall  ({SAMPLE_N}-mol sample)", flush=True)
    print(f"inst_tol (FT=10ppm/0.002Da, QTOF=20ppm/0.015Da, IT=10ppm/0.002Da), h≤{H_TRANSFER}", flush=True)
    print(f"{'='*72}", flush=True)

    print("\n--- Cumulative recall by depth threshold (all inst/mode) ---")
    print(f"  {'d':>2}  {'recall':>8}  {'wrecall':>9}  "
          f"{'Δrecall':>9}  {'Δwrecall':>9}  {'n_specs':>8}")
    prev_r = prev_w = 0.0
    for dep in range(MAX_DEPTH + 1):
        sub = spec_df_out[spec_df_out["depth_thresh"] == dep]
        r = sub["recall"].mean()
        w = sub["wrecall"].mean()
        n = len(sub)
        print(f"  {dep:>2}  {r:>8.4f}  {w:>9.4f}  "
              f"{r-prev_r:>+9.4f}  {w-prev_w:>+9.4f}  {n:>8,}")
        prev_r, prev_w = r, w

    print("\n--- By frag_mode (wrecall, depth=3 full DAG) ---")
    full = spec_df_out[spec_df_out["depth_thresh"] == MAX_DEPTH]
    for mode, grp in full.groupby("frag_mode"):
        print(f"  {mode:5s}  recall={grp['recall'].mean():.4f}  "
              f"wrecall={grp['wrecall'].mean():.4f}  n={len(grp):,}")

    print("\n--- By inst_type (wrecall, depth=3 full DAG) ---")
    for inst, grp in full.groupby("inst_type"):
        print(f"  {inst:5s}  recall={grp['recall'].mean():.4f}  "
              f"wrecall={grp['wrecall'].mean():.4f}  n={len(grp):,}")

    print("\n--- Per-depth wrecall by inst_type ---")
    print(f"  {'inst':>5}  {'d0':>7}  {'d1':>7}  {'d2':>7}  {'d3':>7}")
    for inst, grp in spec_df_out.groupby("inst_type"):
        vals = [grp[grp["depth_thresh"] == d]["wrecall"].mean() for d in range(MAX_DEPTH + 1)]
        print(f"  {inst:>5}  " + "  ".join(f"{v:>7.4f}" for v in vals))


if __name__ == "__main__":
    main()
