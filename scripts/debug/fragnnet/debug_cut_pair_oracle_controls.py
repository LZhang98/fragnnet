from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch as th

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eval")
)

from dag_oracle_scores import _tolerance_for_inst_type

from fragnnet.utils.spec_utils import calculate_match_mzs, cos_hun_helper

CACHE_FP = "/tmp/cut-pair-review-BJ5z6X/cache_500mol.pt"
PRED_FP = "output/inference_eval/9t63zils/predictions_val.hdf5"
SPEC_FP = "data/proc/nist20/spec_df.pkl"
OUT_FP = "/tmp/cut-pair-review-BJ5z6X/oracle_edit_controls_500mol_val.csv"
EDIT_STEPS = 5
EDIT_LR = 0.05
EDIT_L2 = 1e-3


def edit_logits(
    node_probs: np.ndarray,
    child_positions: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Jointly fit all overlapping binary cut constraints for one spectrum."""
    z0 = th.log(th.as_tensor(node_probs, dtype=th.float64).clamp_min(1e-30))
    z = z0.clone().requires_grad_(True)
    child = th.as_tensor(child_positions, dtype=th.long)
    target = th.as_tensor(targets, dtype=th.float64)
    weight = th.as_tensor(weights, dtype=th.float64)
    optimizer = th.optim.Adam([z], lr=EDIT_LR)

    for _ in range(EDIT_STEPS):
        optimizer.zero_grad()
        log_q = th.log_softmax(z[child], dim=1)
        pair_ce = -(target * log_q).sum(dim=1)
        loss = (weight * pair_ce).sum() / weight.sum() + EDIT_L2 * (z - z0).square().mean()
        loss.backward()
        optimizer.step()
    return th.softmax(z.detach(), dim=0).numpy()


def rebuild_peak_intensities(
    node_probs: np.ndarray,
    joint_node_positions: np.ndarray,
    joint_formula_idxs: np.ndarray,
    formula_given_node: np.ndarray,
    peak_formula_idxs: np.ndarray,
    peak_response: np.ndarray,
    num_formulas: int,
) -> np.ndarray:
    """Rebuild spectrum intensities with P(f|n) and isotope response frozen."""
    formula_probs = np.zeros(num_formulas, dtype=np.float64)
    np.add.at(
        formula_probs,
        joint_formula_idxs,
        node_probs[joint_node_positions] * formula_given_node,
    )
    return formula_probs[peak_formula_idxs] * peak_response


def score_spectrum(
    true_mzs: np.ndarray,
    true_ints: np.ndarray,
    pred_mzs: np.ndarray,
    pred_ints: np.ndarray,
    inst_type: str,
) -> float:
    tolerance, tolerance_min_mz = _tolerance_for_inst_type(inst_type)
    true_mzs_t = th.as_tensor(true_mzs, dtype=th.float64)
    true_ints_t = th.as_tensor(true_ints, dtype=th.float64)
    pred_mzs_t = th.as_tensor(pred_mzs, dtype=th.float64)
    pred_ints_t = th.as_tensor(pred_ints, dtype=th.float64)
    match = calculate_match_mzs(
        true_mzs_t,
        pred_mzs_t,
        tolerance=tolerance,
        relative=True,
        tolerance_min_mz=tolerance_min_mz,
    )
    score = cos_hun_helper(
        true_ints_t,
        pred_ints_t,
        match,
        match.any(dim=1),
        match.any(dim=0),
        False,
        th.zeros(true_mzs_t.shape[0], dtype=th.bool),
        th.zeros(pred_mzs_t.shape[0], dtype=th.bool),
    )
    return float(score)


def molecule_cluster_mean_ci(
    frame: pd.DataFrame, column: str, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap a spectrum-level mean by resampling whole molecules."""
    grouped = frame.groupby("mol_id")[column].agg(["sum", "count"])
    numer = grouped["sum"].to_numpy()
    denom = grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    means = np.empty(5000, dtype=np.float64)
    for start in range(0, 5000, 100):
        idx = rng.integers(0, len(grouped), size=(100, len(grouped)))
        means[start : start + 100] = numer[idx].sum(axis=1) / denom[idx].sum(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def main() -> None:
    cache = th.load(CACHE_FP, weights_only=False)
    spec_df = pd.read_pickle(SPEC_FP).set_index("spec_id")
    spec_ids = cache["spec_ids"].numpy()
    spec_ptr = cache["spec_group_ptr"].numpy()
    group_ptr = cache["group_child_ptr"].numpy()
    child_nodes = cache["child_nodes"].numpy()
    q_true_all = cache["q_obs"].numpy()
    q_shuffle_all = cache["q_obs_shuffled"].numpy()
    group_weights = cache["group_weights"].numpy()

    rows: list[dict] = []
    with h5py.File(PRED_FP, "r") as pred:
        pred_row = {int(s): i for i, s in enumerate(pred["spec_id"][:])}
        for cache_i, spec_id_raw in enumerate(spec_ids):
            spec_id = int(spec_id_raw)
            if spec_id not in pred_row:
                continue
            group_start, group_end = int(spec_ptr[cache_i]), int(spec_ptr[cache_i + 1])
            if group_start == group_end:
                continue

            i = pred_row[spec_id]
            peak_start, peak_end = int(pred["start_points"][i]), int(pred["end_points"][i])
            node_start, node_end = (
                int(pred["node_start_points"][i]),
                int(pred["node_end_points"][i]),
            )
            joint_start, joint_end = (
                int(pred["joint_start_points"][i]),
                int(pred["joint_end_points"][i]),
            )
            node_ids = pred["pred_node_node_idxs"][node_start:node_end].astype(np.int64)
            node_probs = pred["pred_node_probs"][node_start:node_end].astype(np.float64)
            node_to_position = {int(node): pos for pos, node in enumerate(node_ids)}

            entry_start, entry_end = int(group_ptr[group_start]), int(group_ptr[group_end])
            local_children = child_nodes[entry_start:entry_end].reshape(-1, 2)
            try:
                child_positions = np.asarray(
                    [
                        [node_to_position[int(a)], node_to_position[int(b)]]
                        for a, b in local_children
                    ],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise ValueError(f"spec_id={spec_id}: cut node absent from prediction") from exc
            q_true = q_true_all[entry_start:entry_end].reshape(-1, 2).astype(np.float64)
            q_shuffle = q_shuffle_all[entry_start:entry_end].reshape(-1, 2).astype(np.float64)
            weights = group_weights[group_start:group_end].astype(np.float64)

            current_pair_logits = np.log(np.clip(node_probs[child_positions], 1e-30, None))
            q_model = np.exp(
                current_pair_logits
                - np.logaddexp.reduce(current_pair_logits, axis=1, keepdims=True)
            )
            target_confidence = q_true.max(axis=1)
            model_winner = q_model.argmax(axis=1)
            q_model_direction = np.full_like(q_true, 1.0 - target_confidence[:, None])
            q_model_direction[np.arange(len(q_true)), model_winner] = target_confidence

            edited_true = edit_logits(node_probs, child_positions, q_true, weights)
            edited_shuffle = edit_logits(node_probs, child_positions, q_shuffle, weights)
            edited_model_ratio = edit_logits(node_probs, child_positions, q_model, weights)
            edited_model_direction = edit_logits(
                node_probs, child_positions, q_model_direction, weights
            )

            joint_node_ids = pred["pred_joint_node_idxs"][joint_start:joint_end].astype(np.int64)
            joint_node_positions = np.asarray(
                [node_to_position[int(node)] for node in joint_node_ids], dtype=np.int64
            )
            joint_formula = pred["pred_joint_formula_idxs"][joint_start:joint_end].astype(np.int64)
            formula_given_node = pred["pred_node_formula_probs"][joint_start:joint_end].astype(
                np.float64
            )
            peak_formula = pred["frag_formula_peak_idxs"][peak_start:peak_end].astype(np.int64)
            pred_mzs = pred["pred_mzs"][peak_start:peak_end].astype(np.float64)
            current_peak_ints = pred["pred_ints"][peak_start:peak_end].astype(np.float64)
            num_formulas = int(max(joint_formula.max(), peak_formula.max()) + 1)

            current_formula = np.zeros(num_formulas, dtype=np.float64)
            np.add.at(
                current_formula,
                joint_formula,
                node_probs[joint_node_positions] * formula_given_node,
            )
            denom = current_formula[peak_formula]
            if np.any(denom <= 0):
                raise ValueError(f"spec_id={spec_id}: cannot recover frozen peak response")
            peak_response = current_peak_ints / denom
            rebuilt_current = rebuild_peak_intensities(
                node_probs,
                joint_node_positions,
                joint_formula,
                formula_given_node,
                peak_formula,
                peak_response,
                num_formulas,
            )
            if not np.allclose(rebuilt_current, current_peak_ints, rtol=2e-5, atol=1e-7):
                raise ValueError(f"spec_id={spec_id}: frozen-decoder reconstruction mismatch")

            true_peak_ints = rebuild_peak_intensities(
                edited_true,
                joint_node_positions,
                joint_formula,
                formula_given_node,
                peak_formula,
                peak_response,
                num_formulas,
            )
            shuffle_peak_ints = rebuild_peak_intensities(
                edited_shuffle,
                joint_node_positions,
                joint_formula,
                formula_given_node,
                peak_formula,
                peak_response,
                num_formulas,
            )
            model_ratio_peak_ints = rebuild_peak_intensities(
                edited_model_ratio,
                joint_node_positions,
                joint_formula,
                formula_given_node,
                peak_formula,
                peak_response,
                num_formulas,
            )
            model_direction_peak_ints = rebuild_peak_intensities(
                edited_model_direction,
                joint_node_positions,
                joint_formula,
                formula_given_node,
                peak_formula,
                peak_response,
                num_formulas,
            )

            spec_row = spec_df.loc[spec_id]
            true_mzs, true_ints = (np.asarray(x, dtype=np.float64) for x in zip(*spec_row["peaks"]))
            current_score = score_spectrum(
                true_mzs, true_ints, pred_mzs, current_peak_ints, str(spec_row["inst_type"])
            )
            true_score = score_spectrum(
                true_mzs, true_ints, pred_mzs, true_peak_ints, str(spec_row["inst_type"])
            )
            shuffle_score = score_spectrum(
                true_mzs, true_ints, pred_mzs, shuffle_peak_ints, str(spec_row["inst_type"])
            )
            model_ratio_score = score_spectrum(
                true_mzs,
                true_ints,
                pred_mzs,
                model_ratio_peak_ints,
                str(spec_row["inst_type"]),
            )
            model_direction_score = score_spectrum(
                true_mzs,
                true_ints,
                pred_mzs,
                model_direction_peak_ints,
                str(spec_row["inst_type"]),
            )
            rows.append(
                {
                    "spec_id": spec_id,
                    "mol_id": int(spec_row["mol_id"]),
                    "num_pairs": group_end - group_start,
                    "cos_current": current_score,
                    "cos_stored": float(pred["cos_hun_inst_tol"][i]),
                    "cos_true_edit": true_score,
                    "cos_shuffled_edit": shuffle_score,
                    "cos_model_ratio_edit": model_ratio_score,
                    "cos_model_direction_edit": model_direction_score,
                    "delta_true": true_score - current_score,
                    "delta_shuffled": shuffle_score - current_score,
                    "delta_model_ratio": model_ratio_score - current_score,
                    "delta_model_direction": model_direction_score - current_score,
                    "delta_specific": true_score - shuffle_score,
                    "delta_vs_model_direction": true_score - model_direction_score,
                }
            )
            if len(rows) % 50 == 0:
                print(f"processed {len(rows)} spectra", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FP, index=False)
    delta = out["delta_specific"].to_numpy()
    perfect_gap = float((1.0 - out["cos_current"]).sum())
    print(
        {
            "spectra": len(out),
            "molecules": int(out["mol_id"].nunique()),
            "mean_current": float(out["cos_current"].mean()),
            "mean_true_edit": float(out["cos_true_edit"].mean()),
            "mean_shuffled_edit": float(out["cos_shuffled_edit"].mean()),
            "mean_delta_true": float(out["delta_true"].mean()),
            "mean_delta_shuffled": float(out["delta_shuffled"].mean()),
            "mean_delta_model_ratio": float(out["delta_model_ratio"].mean()),
            "mean_delta_model_direction": float(out["delta_model_direction"].mean()),
            "mean_delta_specific": float(delta.mean()),
            "mean_delta_vs_model_direction": float(out["delta_vs_model_direction"].mean()),
            "delta_specific_molecule_cluster_95ci": molecule_cluster_mean_ci(out, "delta_specific"),
            "delta_true_molecule_cluster_95ci": molecule_cluster_mean_ci(out, "delta_true"),
            "delta_model_direction_molecule_cluster_95ci": molecule_cluster_mean_ci(
                out, "delta_model_direction"
            ),
            "delta_vs_model_direction_molecule_cluster_95ci": molecule_cluster_mean_ci(
                out, "delta_vs_model_direction"
            ),
            "delta_specific_fraction_of_perfect_gap": float(delta.sum() / perfect_gap),
            "mean_abs_recomputed_minus_stored": float(
                np.abs(out["cos_current"] - out["cos_stored"]).mean()
            ),
            "output": OUT_FP,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
