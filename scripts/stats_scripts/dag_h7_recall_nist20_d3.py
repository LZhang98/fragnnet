"""Compare peak recall at H=4 vs H=7 using existing D3/H=4 DAG pickles.

H=7 is computed by offsetting h_delta=0 formula m/z values by ±5, ±6, ±7 × H_MASS.
No DAG rebuild required — the extension is exact because adding k hydrogen atoms
shifts the exact monoisotopic mass by exactly k × H_MASS.

Usage:
    conda run -n FRAGNNET-GPU python scripts/stats_scripts/dag_h7_recall_nist20_d3.py
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

H_MASS = 1.00782503207     # hydrogen atomic mass in Da
MAX_DEPTH = 3
H_TRANSFER_DAG = 4         # DAGs were built with H=4 (9 positions in idx_by_h_delta)
H_TRANSFER_EVAL = 7        # extended H we want to evaluate

INST_TOL: dict[str, dict] = {
    "FT":   {"rel": 1e-5, "min_mz": 200.0},   # 10 ppm, floor 0.002 Da
    "QTOF": {"rel": 2e-5, "min_mz": 750.0},   # 20 ppm, floor 0.015 Da
    "IT":   {"rel": 1e-5, "min_mz": 200.0},
}
PREC_TYPES = {"[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-"}
INST_TYPES  = {"FT", "QTOF", "IT"}
FRAG_MODES  = {"CID", "HCD"}
SAMPLE_N    = 2000
RANDOM_SEED = 42

# Positions in idx_by_h_delta covered by the existing H=4 DAG
H_DELTA_IDXS_H4 = list(range(1 + 2 * H_TRANSFER_DAG))   # [0..8]

# For +5,+6,+7: extend from h_delta=+4 (idx_by_h_delta[8]) + 1,2,3 × H_MASS
# For -5,-6,-7: extend from h_delta=-4 (idx_by_h_delta[7]) - 1,2,3 × H_MASS
# Using ±4 as basis is more robust than 0: nodes missing h_delta=0 (e.g. fully
# unsaturated fragments) still have ±4 valid if ±5..7 are reachable at all.
POS_H4_IDX = 2 * H_TRANSFER_DAG       # idx_by_h_delta position for h_delta=+4  (= 8)
NEG_H4_IDX = 2 * H_TRANSFER_DAG - 1   # idx_by_h_delta position for h_delta=-4  (= 7)
# Offsets (Da) applied to the ±4 basis to reach ±5, ±6, ±7
POS_EXTRA_OFFSETS = [k * H_MASS for k in range(1, H_TRANSFER_EVAL - H_TRANSFER_DAG + 1)]
NEG_EXTRA_OFFSETS = [k * H_MASS for k in range(1, H_TRANSFER_EVAL - H_TRANSFER_DAG + 1)]


def load_dag(fp: str) -> dict:
    with bz2.open(fp, "rb") as f:
        return pickle.load(f)


def formula_depth_map(dag_d: dict) -> dict[int, int]:
    """Return {formula_idx: min_depth} for all valid (non-sentinel) formula indices."""
    dag = dag_d["dag"]
    feat_start = dag.node_feat_idxs[0, 3].item()
    feat_end   = dag.node_feat_idxs[0, 4].item()
    formula_idx_block = dag.x[:, feat_start:feat_end].numpy()
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
    cfg = INST_TOL[inst_type]
    return cfg["rel"] * np.maximum(true_mzs, cfg["min_mz"])


def build_theo_mzs(
    fpm: np.ndarray,
    fpp: np.ndarray,
    idx_by_h_delta: list[set],
    allowed: set[int],
    prec_type: str,
    use_h7: bool,
) -> np.ndarray:
    """Build sorted unique theoretical m/z array.

    Args:
        fpm: formula_peak_mzs of shape (n_formulas, 5).
        fpp: formula_peak_probs of shape (n_formulas, 5).
        idx_by_h_delta: list of 9 sets (H=4 DAG), formula indices per h_delta slot.
        allowed: formula indices reachable within depth threshold.
        prec_type: adduct type string.
        use_h7: if True, also add h_delta=±5,6,7 offsets from h_delta=0 m/z.

    Returns:
        Sorted unique theoretical m/z array of shape (n_theo,).
    """
    # Collect formula indices from existing H=4 positions
    h4_idxs: set[int] = set()
    for i in H_DELTA_IDXS_H4:
        h4_idxs |= (idx_by_h_delta[i] & allowed)

    if not h4_idxs:
        return np.array([], dtype=np.float64)

    idx_list = list(h4_idxs)
    theo_raw = fpm[idx_list, 0].astype(np.float64)
    prob_raw = fpp[idx_list, 0]
    theo_mono = theo_raw[prob_raw > 0.0]   # valid monoisotopic m/z values

    std_diff = PREC_TYPE_TO_MASS_DIFF.get(prec_type, 0.0)
    cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF.get(prec_type, std_diff)

    parts: list[np.ndarray] = [theo_mono + std_diff]
    if std_diff != cmf_diff:
        parts.append(theo_mono + cmf_diff)

    if use_h7:
        # Positive extensions (+5,+6,+7): step up from h_delta=+4 basis
        pos4_idxs = list(idx_by_h_delta[POS_H4_IDX] & allowed)
        if pos4_idxs:
            pos4_raw  = fpm[pos4_idxs, 0].astype(np.float64)
            pos4_prob = fpp[pos4_idxs, 0]
            pos4_mono = pos4_raw[pos4_prob > 0.0]
            for step in POS_EXTRA_OFFSETS:   # +H_MASS, +2*H_MASS, +3*H_MASS
                ext = pos4_mono + step
                parts.append(ext + std_diff)
                if std_diff != cmf_diff:
                    parts.append(ext + cmf_diff)

        # Negative extensions (-5,-6,-7): step down from h_delta=-4 basis
        neg4_idxs = list(idx_by_h_delta[NEG_H4_IDX] & allowed)
        if neg4_idxs:
            neg4_raw  = fpm[neg4_idxs, 0].astype(np.float64)
            neg4_prob = fpp[neg4_idxs, 0]
            neg4_mono = neg4_raw[neg4_prob > 0.0]
            for step in NEG_EXTRA_OFFSETS:   # +H_MASS, +2*H_MASS, +3*H_MASS (subtract)
                ext = neg4_mono - step
                parts.append(ext + std_diff)
                if std_diff != cmf_diff:
                    parts.append(ext + cmf_diff)

    return np.unique(np.concatenate(parts))


def vectorized_recall(
    true_mzs: np.ndarray,
    true_ints: np.ndarray,
    theo_mzs: np.ndarray,
    inst_type: str,
) -> tuple[float, float]:
    if len(theo_mzs) == 0:
        return 0.0, 0.0
    total_int = true_ints.sum()
    if total_int == 0:
        return 0.0, 0.0
    tols    = inst_tolerances(true_mzs, inst_type)
    diffs   = np.abs(true_mzs[:, None] - theo_mzs[None, :])
    matched = np.any(diffs < tols[:, None], axis=1)
    return float(matched.mean()), float((true_ints * matched).sum() / total_int)


def process_mol(mol_id: int, spec_rows: list[dict]) -> list[dict]:
    fp = os.path.join(DAG_DP, f"{mol_id}.pkl.bz2")
    if not os.path.exists(fp):
        return []
    dag_d = load_dag(fp)
    f_depth = formula_depth_map(dag_d)
    ibd     = dag_d["idx_by_h_delta"]
    fpm     = dag_d["formula_peak_mzs"].numpy()
    fpp     = dag_d["formula_peak_probs"].numpy()

    # Pre-compute allowed set at d3 (full DAG)
    allowed = {f for f, fd in f_depth.items() if fd <= MAX_DEPTH}

    out = []
    for sr in spec_rows:
        peaks = sr["peaks"]
        if not peaks:
            continue
        true_mzs  = np.array([p[0] for p in peaks], dtype=np.float64)
        true_ints = np.array([p[1] for p in peaks], dtype=np.float64)
        inst  = sr["inst_type"]
        prec  = sr["prec_type"]

        theo_h4 = build_theo_mzs(fpm, fpp, ibd, allowed, prec, use_h7=False)
        theo_h7 = build_theo_mzs(fpm, fpp, ibd, allowed, prec, use_h7=True)

        r4, w4 = vectorized_recall(true_mzs, true_ints, theo_h4, inst)
        r7, w7 = vectorized_recall(true_mzs, true_ints, theo_h7, inst)

        out.append({
            "mol_id":        mol_id,
            "spec_id":       sr["spec_id"],
            "inst_type":     inst,
            "frag_mode":     sr["frag_mode"],
            "prec_type":     prec,
            "recall_h4":     r4,
            "wrecall_h4":    w4,
            "recall_h7":     r7,
            "wrecall_h7":    w7,
            "delta_recall":  r7 - r4,
            "delta_wrecall": w7 - w4,
        })
    return out


def main() -> None:
    dag_mol_ids = sorted(int(fn.split(".")[0]) for fn in os.listdir(DAG_DP))
    print(f"Total DAG molecules: {len(dag_mol_ids):,}", flush=True)

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

    print(f"Running H=4 vs H=7 recall on {SAMPLE_N} mols (D3 only)...", flush=True)
    all_rows: list[dict] = []
    for i, mol_id in enumerate(sorted(sample_mol_ids)):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{SAMPLE_N}", flush=True)
        all_rows.extend(process_mol(mol_id, spec_by_mol.get(mol_id, [])))

    df = pd.DataFrame(all_rows)
    out_fp = os.path.join(OUT_DIR, "dag_h7_recall_sample.pkl")
    df.to_pickle(out_fp)
    print(f"Saved: {out_fp}  ({len(df):,} rows)", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}", flush=True)
    print(f"H=4 vs H=7 recall at D3  ({SAMPLE_N}-mol sample, inst_tol)", flush=True)
    print(f"{'='*72}", flush=True)

    print("\n--- Overall ---")
    for col_r, col_w, label in [
        ("recall_h4", "wrecall_h4", "H=4"),
        ("recall_h7", "wrecall_h7", "H=7"),
    ]:
        print(f"  {label}  recall={df[col_r].mean():.4f}  wrecall={df[col_w].mean():.4f}")
    print(f"  Δ     recall={df['delta_recall'].mean():+.4f}  "
          f"wrecall={df['delta_wrecall'].mean():+.4f}")

    print("\n--- By inst_type ---")
    for inst, grp in df.groupby("inst_type"):
        print(f"  {inst:5s}  H4={grp['wrecall_h4'].mean():.4f}  "
              f"H7={grp['wrecall_h7'].mean():.4f}  "
              f"Δ={grp['delta_wrecall'].mean():+.4f}  n={len(grp):,}")

    print("\n--- By frag_mode ---")
    for mode, grp in df.groupby("frag_mode"):
        print(f"  {mode:5s}  H4={grp['wrecall_h4'].mean():.4f}  "
              f"H7={grp['wrecall_h7'].mean():.4f}  "
              f"Δ={grp['delta_wrecall'].mean():+.4f}  n={len(grp):,}")

    print("\n--- Delta wrecall distribution ---")
    dw = df["delta_wrecall"]
    print(f"  mean={dw.mean():+.4f}  med={dw.median():+.4f}  "
          f"p10={dw.quantile(0.1):+.4f}  p90={dw.quantile(0.9):+.4f}")
    print(f"  spectra with Δwrecall > 0.01: {(dw > 0.01).mean() * 100:.1f}%")
    print(f"  spectra with Δwrecall > 0.05: {(dw > 0.05).mean() * 100:.1f}%")
    print(f"  spectra with Δwrecall > 0.10: {(dw > 0.10).mean() * 100:.1f}%")

    print("\n--- By prec_type (top adducts) ---")
    for prec, grp in df.groupby("prec_type"):
        if len(grp) < 100:
            continue
        print(f"  {prec:12s}  H4={grp['wrecall_h4'].mean():.4f}  "
              f"H7={grp['wrecall_h7'].mean():.4f}  "
              f"Δ={grp['delta_wrecall'].mean():+.4f}  n={len(grp):,}")


if __name__ == "__main__":
    main()
