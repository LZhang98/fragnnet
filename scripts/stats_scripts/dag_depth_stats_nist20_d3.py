"""Per-depth DAG contribution stats for NIST20 D3 molecules.

For each BFS depth (0–3), computes:
  - New fragments (DAG nodes) and unique formulas first accessible at that depth
  - Cumulative peak recall / weighted-recall at each depth threshold
    (exact, on a representative 2 000-molecule random sample)

Usage:
    conda run -n FRAGNNET-GPU python scripts/stats_scripts/dag_depth_stats_nist20_d3.py
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
    "IT":   {"rel": 1e-5, "min_mz": 200.0},   # same as FT
}

PREC_TYPES = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"}
INST_TYPES  = {"OB", "OB_F", "QTOF", "IT"}
FRAG_MODES  = {"CID", "HCD"}

SAMPLE_N    = 2000   # molecules used for recall estimation (random, seeded)
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
            if f_idx == 0:                      # sentinel = no formula at this h_delta
                continue
            if f_idx not in f_depth or dep < f_depth[f_idx]:
                f_depth[f_idx] = int(dep)
    return f_depth


H_DELTA_IDXS = list(range(1 + 2 * H_TRANSFER))  # [0..8] for H_TRANSFER=4


def inst_tolerances(true_mzs: np.ndarray, inst_type: str) -> np.ndarray:
    """Return per-peak tolerance array (ppm-based with absolute floor).

    Args:
        true_mzs: Observed m/z values of shape (n_peaks,).
        inst_type: Instrument type string.

    Returns:
        Per-peak tolerance array of shape (n_peaks,).
    """
    cfg = INST_TOL[inst_type]
    return cfg["rel"] * np.maximum(true_mzs, cfg["min_mz"])


def ibd_allowed(idx_by_h_delta: list[set], allowed: set[int]) -> set[int]:
    """Return union of formula indices across all h_delta positions, filtered to allowed."""
    out: set[int] = set()
    for i in H_DELTA_IDXS:
        out |= (idx_by_h_delta[i] & allowed)
    return out


def vectorized_recall(
    peaks: list,
    formula_peak_mzs_np: np.ndarray,
    formula_peak_probs_np: np.ndarray,
    allowed_idxs: set[int],
    prec_type: str,
    inst_type: str,
) -> tuple[float, float]:
    """Fast vectorized recall / wrecall using per-instrument ppm tolerance and CMF."""
    if not peaks or not allowed_idxs:
        return 0.0, 0.0

    true_mzs  = np.array([p[0] for p in peaks], dtype=np.float64)
    true_ints = np.array([p[1] for p in peaks], dtype=np.float64)
    total_int = true_ints.sum()
    if total_int == 0:
        return 0.0, 0.0

    idx_list  = list(allowed_idxs)
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

    tols    = inst_tolerances(true_mzs, inst_type)
    diffs   = np.abs(true_mzs[:, None] - theo_all[None, :])
    matched = np.any(diffs < tols[:, None], axis=1)

    recall  = matched.mean()
    wrecall = (true_ints * matched).sum() / total_int
    return float(recall), float(wrecall)


# ── Pass 1: fast node/formula counts from ALL molecules ────────────────────────

def fast_pass_mol(mol_id: int) -> dict | None:
    fp = os.path.join(DAG_DP, f"{mol_id}.pkl.bz2")
    if not os.path.exists(fp):
        return None
    dag_d = load_dag(fp)

    f_depth = formula_depth_map(dag_d)
    total_valid_f = len(f_depth)
    nodes_by_depth = dag_d["dag_num_nodes_by_depth"]
    edges_by_depth = dag_d["dag_num_edges_by_depth"]

    row: dict = {
        "mol_id": mol_id,
        "reached_depth": dag_d["reached_depth"],
        "total_nodes": dag_d["dag_num_nodes"],
        "total_edges": dag_d["dag_num_edges"],
        "total_formulas_valid": total_valid_f,
        "force_stopped": dag_d.get("force_stopped", False),
    }
    cum_f = 0
    for dep in range(MAX_DEPTH + 1):
        row[f"new_nodes_d{dep}"]  = nodes_by_depth.get(dep, 0)
        row[f"new_edges_d{dep}"]  = edges_by_depth.get(dep, 0)
        n_new_f = sum(1 for fd in f_depth.values() if fd == dep)
        row[f"new_formulas_d{dep}"] = n_new_f
        cum_f += n_new_f
        row[f"cum_formulas_thru_d{dep}"] = cum_f
        row[f"frac_formulas_thru_d{dep}"] = (
            cum_f / total_valid_f if total_valid_f > 0 else 0.0
        )
    return row


# ── Pass 2: per-depth peak recall on sampled molecules ─────────────────────────

def recall_pass_mol(
    mol_id: int,
    spec_rows: list[dict],
) -> list[dict]:
    fp = os.path.join(DAG_DP, f"{mol_id}.pkl.bz2")
    if not os.path.exists(fp):
        return []
    dag_d = load_dag(fp)

    f_depth        = formula_depth_map(dag_d)
    idx_by_h_delta = dag_d["idx_by_h_delta"]
    fpm            = dag_d["formula_peak_mzs"].numpy()   # (n_formulas, 5)
    fpp            = dag_d["formula_peak_probs"].numpy()  # (n_formulas, 5)

    depth_allowed: dict[int, set[int]] = {
        dep: ibd_allowed(idx_by_h_delta, {f for f, fd in f_depth.items() if fd <= dep})
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

    # ── Pass 1: node / formula counts (all molecules) ──────────────────────────
    print("\nPass 1: scanning all DAGs for node/formula counts...", flush=True)
    node_rows = []
    for i, mol_id in enumerate(dag_mol_ids):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(dag_mol_ids)}", flush=True)
        row = fast_pass_mol(mol_id)
        if row is not None:
            node_rows.append(row)

    node_df = pd.DataFrame(node_rows)
    node_df.to_pickle(os.path.join(OUT_DIR, "dag_depth_node_stats.pkl"))
    print(f"  Done. n={len(node_df):,}", flush=True)

    # ── Pass 2: recall on sample ───────────────────────────────────────────────
    print(f"\nPass 2: loading spec_df for recall on {SAMPLE_N} sampled molecules...",
          flush=True)
    spec_df = pd.read_pickle(SPEC_DF_FP)
    spec_filt = spec_df[
        spec_df["prec_type"].isin(PREC_TYPES)
        & spec_df["inst_type"].isin(INST_TYPES)
        & spec_df["frag_mode"].isin(FRAG_MODES)
        & (spec_df["spec_type"] == "MS2")
        & spec_df["mol_id"].isin(set(dag_mol_ids))
    ][["spec_id", "mol_id", "prec_mz", "prec_type", "inst_type", "frag_mode", "peaks"]].copy()
    del spec_df

    rng = random.Random(RANDOM_SEED)
    sample_mol_ids = set(rng.sample(dag_mol_ids, SAMPLE_N))
    spec_sample = spec_filt[spec_filt["mol_id"].isin(sample_mol_ids)]
    print(f"  Sample: {len(spec_sample):,} spectra across {spec_sample['mol_id'].nunique()} mols",
          flush=True)

    # Build per-mol dict of spec rows (no heavy Python loop — use groupby)
    spec_by_mol: dict[int, list[dict]] = {}
    for mol_id, grp in spec_sample.groupby("mol_id"):
        spec_by_mol[int(mol_id)] = grp[
            ["spec_id", "prec_mz", "prec_type", "inst_type", "frag_mode", "peaks"]
        ].to_dict("records")
    del spec_sample, spec_filt

    print("  Running per-depth recall matching...", flush=True)
    spec_recall_rows = []
    for i, mol_id in enumerate(sorted(sample_mol_ids)):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{SAMPLE_N}", flush=True)
        rows = recall_pass_mol(mol_id, spec_by_mol.get(mol_id, []))
        spec_recall_rows.extend(rows)

    spec_df_out = pd.DataFrame(spec_recall_rows)
    spec_df_out.to_pickle(os.path.join(OUT_DIR, "dag_depth_recall_sample.pkl"))
    print(f"  Done. {len(spec_df_out):,} spec×depth rows", flush=True)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"NIST20 D3 — per-depth DAG contribution  (node stats n={len(node_df):,})")
    print(f"Peak recall on {SAMPLE_N}-mol random sample; "
          f"inst_tol (FT=10ppm/0.002Da, QTOF=20ppm/0.015Da, IT=10ppm/0.002Da), "
          f"h≤{H_TRANSFER}, CMF+std matching")
    print(f"{'='*72}")

    print("\n--- reached_depth distribution ---")
    for dep, cnt in node_df["reached_depth"].value_counts().sort_index().items():
        print(f"  depth {dep}: {cnt:6d} mols  ({100*cnt/len(node_df):.1f}%)")
    print(f"  force_stopped: {int(node_df['force_stopped'].sum())}")

    print("\n--- Overall DAG size ---")
    for col, label in [("total_nodes","nodes"), ("total_edges","edges"),
                       ("total_formulas_valid","formulas")]:
        s = node_df[col]
        print(f"  {label:10s}  mean={s.mean():.1f}  med={s.median():.0f}  "
              f"std={s.std():.1f}  p10={s.quantile(0.1):.0f}  p90={s.quantile(0.9):.0f}")

    print("\n--- New nodes per depth (how many new fragments BFS adds) ---")
    print(f"  {'d':>2}  {'mean':>7}  {'med':>5}  {'p10':>5}  {'p90':>5}  "
          f"{'mols>0':>9}  {'mean% total':>11}")
    for dep in range(MAX_DEPTH + 1):
        s   = node_df[f"new_nodes_d{dep}"]
        frac = s / node_df["total_nodes"]
        act  = (s > 0).sum()
        print(f"  {dep:>2}  {s.mean():>7.1f}  {s.median():>5.0f}  "
              f"{s.quantile(0.1):>5.0f}  {s.quantile(0.9):>5.0f}  "
              f"{act:>6} ({100*act/len(node_df):.0f}%)  {100*frac.mean():>9.1f}%")

    print("\n--- New unique formulas per depth ---")
    print(f"  {'d':>2}  {'mean new':>8}  {'med':>5}  {'cum% formula (mean)':>20}")
    for dep in range(MAX_DEPTH + 1):
        s   = node_df[f"new_formulas_d{dep}"]
        cum = node_df[f"frac_formulas_thru_d{dep}"]
        print(f"  {dep:>2}  {s.mean():>8.1f}  {s.median():>5.0f}  "
              f"{'cumulative thru d'+str(dep)+':':>22} {100*cum.mean():.1f}%")

    if len(spec_df_out) > 0:
        print("\n--- Cumulative peak recall by depth (sample mols, all inst/mode) ---")
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

        print("\n--- By frag_mode (depth=3, full DAG) ---")
        full = spec_df_out[spec_df_out["depth_thresh"] == MAX_DEPTH]
        for mode, grp in full.groupby("frag_mode"):
            print(f"  {mode:6s}  recall={grp['recall'].mean():.4f}  "
                  f"wrecall={grp['wrecall'].mean():.4f}  n={len(grp):,}")

        print("\n--- By inst_type (depth=3, full DAG) ---")
        for inst, grp in full.groupby("inst_type"):
            print(f"  {inst:6s}  recall={grp['recall'].mean():.4f}  "
                  f"wrecall={grp['wrecall'].mean():.4f}  n={len(grp):,}")

        print("\n--- Per-depth wrecall by inst_type ---")
        print(f"  {'inst':>5}  {'d0':>7}  {'d1':>7}  {'d2':>7}  {'d3':>7}")
        for inst, grp in spec_df_out.groupby("inst_type"):
            vals = [grp[grp["depth_thresh"] == d]["wrecall"].mean() for d in range(MAX_DEPTH + 1)]
            print(f"  {inst:>5}  " + "  ".join(f"{v:>7.4f}" for v in vals))


if __name__ == "__main__":
    main()
