from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np
import torch as th

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eval")
)
from build_cut_pair_targets import load_spec_df_with_ce_fill

import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils.feat_utils import get_node_feats
from fragnnet.utils.frag_utils import th_long_to_mask

CACHE_FP = "/tmp/cut-pair-review-BJ5z6X/cache_500mol.pt"
PRED_FP = "output/inference_eval/9t63zils/predictions_val.hdf5"
SPEC_FP = "data/proc/nist20/spec_df.pkl"
OUT_FP = "/tmp/cut-pair-review-BJ5z6X/cut_pair_control_stats.json"


def metrics(q: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    if q.size == 0 or w.sum() <= 0:
        return {"pairs": 0}
    p_clip = np.clip(p, 1e-30, 1.0 - 1e-15)
    kl = q * np.log(q / p_clip) + (1.0 - q) * np.log((1.0 - q) / (1.0 - p_clip))
    tv = np.abs(p - q)
    tie = np.isclose(q, 0.5)
    mismatch = (p > 0.5) != (q > 0.5)
    valid_w = w[~tie]
    return {
        "pairs": int(q.size),
        "weight_sum": float(w.sum()),
        "kish_ess": float(w.sum() ** 2 / np.square(w).sum()),
        "weighted_tv": float(np.sum(w * tv) / w.sum()),
        "weighted_kl": float(np.sum(w * kl) / w.sum()),
        "weighted_dominant_error": (
            float(np.sum(valid_w * mismatch[~tie]) / valid_w.sum()) if valid_w.size else None
        ),
        "uniform_weighted_tv": float(np.sum(w * np.abs(q - 0.5)) / w.sum()),
        "target_confidence": float(np.sum(w * np.maximum(q, 1.0 - q)) / w.sum()),
        "model_confidence": float(np.sum(w * np.maximum(p, 1.0 - p)) / w.sum()),
    }


def cluster_ci(
    mol: np.ndarray,
    q: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    selector: np.ndarray,
    seed: int = 0,
) -> dict:
    p_clip = np.clip(p, 1e-30, 1.0 - 1e-15)
    values = {
        "tv": np.abs(p - q),
        "kl": q * np.log(q / p_clip) + (1.0 - q) * np.log((1.0 - q) / (1.0 - p_clip)),
        "dominant_error": ((p > 0.5) != (q > 0.5)).astype(np.float64),
    }
    mol_ids = np.unique(mol[selector])
    by_mol: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, value in values.items():
        numer = np.asarray(
            [np.sum(w[selector & (mol == m)] * value[selector & (mol == m)]) for m in mol_ids]
        )
        denom = np.asarray([np.sum(w[selector & (mol == m)]) for m in mol_ids])
        by_mol[name] = numer, denom

    rng = np.random.default_rng(seed)
    out = {}
    for name, (numer, denom) in by_mol.items():
        boot = np.empty(5000, dtype=np.float64)
        for start in range(0, 5000, 100):
            idx = rng.integers(0, len(mol_ids), size=(100, len(mol_ids)))
            boot[start : start + 100] = numer[idx].sum(axis=1) / denom[idx].sum(axis=1)
        out[name] = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
    return out


def main() -> None:
    cache = th.load(CACHE_FP, weights_only=False)
    spec_df, _ = load_spec_df_with_ce_fill(SPEC_FP, ["nist20_hr"])
    spec_by_id = spec_df.set_index("spec_id")

    spec_ptr = cache["spec_group_ptr"].numpy()
    group_ptr = cache["group_child_ptr"].numpy()
    child_nodes = cache["child_nodes"].numpy()
    q_all = cache["q_obs"].numpy()
    weight_all = cache["group_weights"].numpy()

    records: dict[str, list] = defaultdict(list)
    per_spec: list[dict] = []
    atom_counts_by_mol: dict[int, np.ndarray] = {}
    with h5py.File(PRED_FP, "r") as pred:
        pred_row = {int(s): i for i, s in enumerate(pred["spec_id"][:])}
        for cache_i, spec_id_raw in enumerate(cache["spec_ids"].tolist()):
            spec_id = int(spec_id_raw)
            if spec_id not in pred_row:
                continue
            gs, ge = int(spec_ptr[cache_i]), int(spec_ptr[cache_i + 1])
            if gs == ge:
                continue
            es, ee = int(group_ptr[gs]), int(group_ptr[ge])
            children = child_nodes[es:ee].reshape(-1, 2)
            q = q_all[es:ee].reshape(-1, 2)[:, 0].astype(np.float64)
            w = weight_all[gs:ge].astype(np.float64)

            pred_i = pred_row[spec_id]
            ns = int(pred["node_start_points"][pred_i])
            ne = int(pred["node_end_points"][pred_i])
            node_ids = pred["pred_node_node_idxs"][ns:ne].astype(np.int64)
            node_probs = pred["pred_node_probs"][ns:ne].astype(np.float64)
            node_map = dict(zip(node_ids.tolist(), node_probs.tolist()))
            pair_probs = np.asarray(
                [[node_map[int(a)], node_map[int(b)]] for a, b in children], dtype=np.float64
            )
            p = pair_probs[:, 0] / pair_probs.sum(axis=1)

            row = spec_by_id.loc[spec_id]
            mol_id = int(row["mol_id"])
            if mol_id not in atom_counts_by_mol:
                dag_d = frag_utils.load_frag_d(
                    mol_id, cache["metadata"]["frag_dp"], is_compressed=True
                )
                dag = dag_d["dag"]
                nfi = dag.node_feat_idxs[0]
                masks = th_long_to_mask(get_node_feats(dag.x, nfi, "cc").long())
                atom_counts_by_mol[mol_id] = masks.sum(dim=1).numpy().astype(np.float64)
            atom_counts = atom_counts_by_mol[mol_id]
            pair_atom_counts = atom_counts[children]
            p_atom = pair_atom_counts[:, 0] / pair_atom_counts.sum(axis=1)
            n = len(q)
            records["q"].extend(q.tolist())
            records["p"].extend(p.tolist())
            records["w"].extend(w.tolist())
            records["p_atom"].extend(p_atom.tolist())
            records["mol"].extend([mol_id] * n)
            records["spec"].extend([spec_id] * n)
            records["nce"].extend([float(row["nce"])] * n)
            records["inst"].extend([str(row["inst_type"])] * n)
            records["prec"].extend([str(row["prec_type"])] * n)
            records["frag"].extend([str(row["frag_mode"])] * n)
            spec_metrics = metrics(q, p, w)
            per_spec.append(
                {
                    "spec_id": spec_id,
                    "mol_id": mol_id,
                    "cut_tv": spec_metrics["weighted_tv"],
                    "cut_kl": spec_metrics["weighted_kl"],
                    "cos": float(pred["cos_hun_inst_tol"][pred_i]),
                }
            )

    q = np.asarray(records["q"])
    p = np.asarray(records["p"])
    w = np.asarray(records["w"])
    p_atom = np.asarray(records["p_atom"])
    mol = np.asarray(records["mol"])
    nce = np.asarray(records["nce"])
    inst = np.asarray(records["inst"])
    prec = np.asarray(records["prec"])
    frag = np.asarray(records["frag"])
    confidence = np.maximum(q, 1.0 - q)

    report: dict = {
        "spectra": len(per_spec),
        "molecules": int(np.unique(mol).size),
        "overall": metrics(q, p, w),
        "unweighted": metrics(q, p, np.ones_like(w)),
        "baselines": {
            "uniform_weighted": metrics(q, np.full_like(q, 0.5), w),
            "uniform_unweighted": metrics(q, np.full_like(q, 0.5), np.ones_like(w)),
            "atom_count_ratio_weighted": metrics(q, p_atom, w),
            "atom_count_ratio_unweighted": metrics(q, p_atom, np.ones_like(w)),
        },
    }
    all_selector = np.ones(q.size, dtype=bool)
    high_selector = confidence >= 0.8
    report["overall"]["cluster_ci95"] = cluster_ci(mol, q, p, w, all_selector)
    report["high_confidence"] = metrics(q[high_selector], p[high_selector], w[high_selector])
    report["high_confidence"]["cluster_ci95"] = cluster_ci(mol, q, p, w, high_selector)

    report["by_confidence"] = {}
    for lo, hi in zip([0.5, 0.6, 0.7, 0.8, 0.9, 0.99], [0.6, 0.7, 0.8, 0.9, 0.99, 1.001]):
        sel = (confidence >= lo) & (confidence < hi)
        report["by_confidence"][f"{lo:g}-{min(hi, 1.0):g}"] = metrics(q[sel], p[sel], w[sel])

    report["by_final_weight_decile"] = {}
    weight_edges = np.quantile(w, np.linspace(0.0, 1.0, 11))
    for decile in range(10):
        lo, hi = weight_edges[decile], weight_edges[decile + 1]
        sel = (w >= lo) & ((w <= hi) if decile == 9 else (w < hi))
        report["by_final_weight_decile"][str(decile + 1)] = metrics(
            q[sel], p[sel], np.ones(sel.sum(), dtype=np.float64)
        )

    report["by_nce"] = {}
    for lo, hi in zip([0, 20, 35, 50, 85, 120], [20, 35, 50, 85, 120, np.inf]):
        sel = (nce >= lo) & (nce < hi)
        report["by_nce"][f"{lo:g}-{hi:g}"] = metrics(q[sel], p[sel], w[sel])

    for name, values in (("by_inst", inst), ("by_prec", prec), ("by_frag", frag)):
        report[name] = {
            str(v): metrics(q[values == v], p[values == v], w[values == v])
            for v in np.unique(values)
        }

    import pandas as pd

    per_spec_df = pd.DataFrame(per_spec)
    intensity_df = pd.read_csv("output/diagnostics/intensity_error_9t63zils_val.csv")
    overlap_global = per_spec_df[["spec_id", "cos"]].merge(
        intensity_df, on="spec_id", how="inner", validate="one_to_one"
    )
    hit = overlap_global["base_peak_hit"].astype(bool)
    report["same_cohort_global"] = {
        "spectra": int(len(overlap_global)),
        "base_peak_accuracy": float(hit.mean()),
        "cos_base_peak_hit": float(overlap_global.loc[hit, "cos"].mean()),
        "cos_base_peak_miss": float(overlap_global.loc[~hit, "cos"].mean()),
        "mean_spearman": float(overlap_global["spearman"].mean()),
        "mean_entropy_diff": float(overlap_global["entropy_diff"].mean()),
        "mean_true_mass_matched": float(overlap_global["true_mass_matched"].mean()),
    }
    report["global_relationship"] = {
        "mean_cos": float(per_spec_df["cos"].mean()),
        "corr_cos_cut_tv": float(per_spec_df[["cos", "cut_tv"]].corr().iloc[0, 1]),
        "corr_cos_cut_kl": float(per_spec_df[["cos", "cut_kl"]].corr().iloc[0, 1]),
        "mean_cut_tv_lowest_cos_quartile": float(
            per_spec_df.nsmallest(len(per_spec_df) // 4, "cos")["cut_tv"].mean()
        ),
        "mean_cut_tv_highest_cos_quartile": float(
            per_spec_df.nlargest(len(per_spec_df) // 4, "cos")["cut_tv"].mean()
        ),
    }
    with open(OUT_FP, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
