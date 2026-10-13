"""Test whether fragment probabilities are effectively static across collision energies.

For fixed ``(molecule, adduct, instrument, fragmentation mode)`` contexts, this diagnostic
compares observed and predicted variation across CE, evaluates each prediction against its
CE-specific target and a leave-one-CE-out context average, and measures exact node-probability
stability across adjacent CE levels.

The constants below intentionally pin the provisional ``9t63zils`` checkpoint export and the
base-peak analysis inputs used by ``fragment_intensity_ranking_findings.md``. Change the prediction,
metadata, and output paths together when repeating the test on a matched GINE-1 checkpoint.

Usage:
    /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
        scripts/debug/fragnnet/debug_ce_static_fragment_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PRED_FP = Path("output/inference_eval/9t63zils/predictions_val.hdf5")
SPEC_FP = Path("data/proc/nist20/spec_df.pkl")
BASE_DP = Path("/tmp/base-peak-ce-structure-9t63zils")
OUT_DP = Path("/tmp/ce-static-fragment-weight-9t63zils")
BIN_WIDTH = 0.01
NODE_SAMPLE_MOLS = 1500
SEED = 17
CTX = ["mol_id", "prec_type", "inst_type", "frag_mode"]


def binned_dist(mzs: np.ndarray, intensities: np.ndarray) -> dict[int, float]:
    intensities = np.asarray(intensities, dtype=float)
    total = intensities.sum()
    if len(intensities) == 0 or total <= 0:
        return {}
    bins = np.rint(np.asarray(mzs, dtype=float) / BIN_WIDTH).astype(np.int64)
    unique, inverse = np.unique(bins, return_inverse=True)
    values = np.bincount(inverse, weights=intensities) / total
    return {int(k): float(v) for k, v in zip(unique, values) if v > 0}


def average_sparse(distributions: list[dict[int, float]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for dist in distributions:
        for key, value in dist.items():
            out[key] = out.get(key, 0.0) + value
    scale = 1.0 / len(distributions)
    return {key: value * scale for key, value in out.items()}


def sparse_linear_combination(
    a: dict[int, float], a_scale: float, b: dict[int, float], b_scale: float
) -> dict[int, float]:
    out = {key: a_scale * value for key, value in a.items()}
    for key, value in b.items():
        out[key] = out.get(key, 0.0) + b_scale * value
    return {key: value for key, value in out.items() if abs(value) > 1e-15}


def sparse_tv(a: dict[int, float], b: dict[int, float]) -> float:
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in a.keys() | b.keys())


def sparse_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    common = a.keys() & b.keys()
    dot = sum(a[key] * b[key] for key in common)
    norm_a = np.sqrt(sum(value * value for value in a.values()))
    norm_b = np.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return np.nan
    return float(dot / (norm_a * norm_b))


def sparse_dot(a: dict[int, float], b: dict[int, float]) -> float:
    """Return the dot product of two sparse vectors."""
    return float(sum(a[key] * b[key] for key in a.keys() & b.keys()))


def sparse_l2_sq(a: dict[int, float]) -> float:
    """Return the squared L2 norm of a sparse vector."""
    return float(sum(value * value for value in a.values()))


def top_bin(dist: dict[int, float]) -> int:
    return max(dist, key=dist.__getitem__)


def bootstrap_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    condition_col: str | None = None,
    n_boot: int = 3000,
) -> tuple[float, float, float]:
    work = df if condition_col is None else df[df[condition_col]]
    by_mol = work.groupby("mol_id", observed=True)[value_col].agg(["sum", "count"]).to_numpy(float)
    point = by_mol[:, 0].sum() / by_mol[:, 1].sum()
    rng = np.random.default_rng(SEED)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sample = by_mol[rng.integers(0, len(by_mol), len(by_mol))]
        estimates[i] = sample[:, 0].sum() / sample[:, 1].sum()
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def bootstrap_ratio_of_sums_ci(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    condition_col: str | None = None,
    n_boot: int = 3000,
) -> tuple[float, float, float]:
    """Bootstrap a ratio of sums with molecule-level resampling."""
    work = df if condition_col is None else df[df[condition_col]]
    by_mol = (
        work.groupby("mol_id", observed=True)[[numerator_col, denominator_col]]
        .sum()
        .to_numpy(float)
    )
    denominator = by_mol[:, 1].sum()
    if denominator <= 0:
        return np.nan, np.nan, np.nan
    point = by_mol[:, 0].sum() / denominator
    rng = np.random.default_rng(SEED)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sample = by_mol[rng.integers(0, len(by_mol), len(by_mol))]
        estimates[i] = sample[:, 0].sum() / sample[:, 1].sum()
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main() -> None:
    OUT_DP.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(BASE_DP / "per_spectrum.csv")
    metadata = metadata[metadata["model_nce"].notna()].copy()
    metadata["ce_key"] = metadata["model_nce"].round(6)
    wanted_ids = set(metadata["spec_id"].astype(int))

    spec_df = pd.read_pickle(SPEC_FP)
    spec_df = spec_df[spec_df["spec_id"].astype(int).isin(wanted_ids)][["spec_id", "peaks"]]
    peaks_by_sid = {int(row.spec_id): row.peaks for row in spec_df.itertuples(index=False)}
    del spec_df

    rng = np.random.default_rng(SEED)
    unique_mols = metadata["mol_id"].unique()
    node_sample = set(
        rng.choice(unique_mols, size=min(NODE_SAMPLE_MOLS, len(unique_mols)), replace=False)
    )

    context_rows = []
    level_rows = []
    adjacent_rows = []

    with h5py.File(PRED_FP, "r") as h5:
        h5_spec_ids = h5["spec_id"][:].astype(int)
        pos_by_sid = {sid: i for i, sid in enumerate(h5_spec_ids)}
        starts = h5["start_points"][:].astype(int)
        ends = h5["end_points"][:].astype(int)
        pred_mzs_all = h5["pred_mzs"][:].astype(float)
        pred_ints_all = h5["pred_ints"][:].astype(float)
        node_starts = h5["node_start_points"][:].astype(int)
        node_ends = h5["node_end_points"][:].astype(int)

        for key, context in metadata.groupby(CTX, observed=True):
            if context["ce_key"].nunique() < 2:
                continue
            use_nodes = int(key[0]) in node_sample
            levels = []
            node_length = None
            node_valid = use_nodes

            for ce_key, ce_group in context.groupby("ce_key", observed=True):
                true_dists = []
                pred_dists = []
                node_dists = []
                for row in ce_group.itertuples(index=False):
                    sid = int(row.spec_id)
                    pos = pos_by_sid[sid]
                    peaks = peaks_by_sid[sid]
                    true_mzs = np.asarray([peak[0] for peak in peaks], dtype=float)
                    true_ints = np.asarray([peak[1] for peak in peaks], dtype=float)
                    true_dists.append(binned_dist(true_mzs, true_ints))

                    start, end = starts[pos], ends[pos]
                    pred_dists.append(
                        binned_dist(pred_mzs_all[start:end], pred_ints_all[start:end])
                    )

                    if node_valid:
                        node_start, node_end = node_starts[pos], node_ends[pos]
                        node_ids = h5["pred_node_node_idxs"][node_start:node_end]
                        expected = np.arange(node_end - node_start, dtype=node_ids.dtype)
                        if not np.array_equal(node_ids, expected):
                            node_valid = False
                            node_dists = []
                        else:
                            values = h5["pred_node_probs"][node_start:node_end].astype(float)
                            if node_length is None:
                                node_length = len(values)
                            if len(values) != node_length:
                                node_valid = False
                                node_dists = []
                            else:
                                node_dists.append(values)

                levels.append(
                    {
                        "ce": float(ce_key),
                        "true": average_sparse(true_dists),
                        "pred": average_sparse(pred_dists),
                        "node": np.mean(node_dists, axis=0) if node_valid and node_dists else None,
                        "n_replicates": len(ce_group),
                    }
                )

            levels.sort(key=lambda item: item["ce"])
            true_mean = average_sparse([level["true"] for level in levels])
            pred_mean = average_sparse([level["pred"] for level in levels])
            node_mean = None
            if node_valid and all(level["node"] is not None for level in levels):
                node_mean = np.mean([level["node"] for level in levels], axis=0)

            true_tv_mean = np.mean([sparse_tv(level["true"], true_mean) for level in levels])
            pred_tv_mean = np.mean([sparse_tv(level["pred"], pred_mean) for level in levels])
            residual_cosines = []
            node_tv_mean = np.nan
            if node_mean is not None:
                node_tv_mean = float(
                    np.mean([0.5 * np.abs(level["node"] - node_mean).sum() for level in levels])
                )

            n_levels = len(levels)
            for level_idx, level in enumerate(levels):
                true_loo = sparse_linear_combination(
                    true_mean,
                    n_levels / (n_levels - 1),
                    level["true"],
                    -1.0 / (n_levels - 1),
                )
                tv_specific = sparse_tv(level["pred"], level["true"])
                tv_average = sparse_tv(level["pred"], true_loo)
                true_residual = sparse_linear_combination(level["true"], 1.0, true_mean, -1.0)
                pred_residual = sparse_linear_combination(level["pred"], 1.0, pred_mean, -1.0)
                residual_cos = sparse_cosine(true_residual, pred_residual)
                residual_cosines.append(residual_cos)
                level_rows.append(
                    {
                        **dict(zip(CTX, key)),
                        "ce": level["ce"],
                        "n_levels": n_levels,
                        "tv_pred_specific": tv_specific,
                        "tv_pred_loo_average": tv_average,
                        "specific_minus_average_tv": tv_specific - tv_average,
                        "closer_to_loo_average": tv_average < tv_specific,
                        "true_tv_from_context_mean": sparse_tv(level["true"], true_mean),
                        "pred_tv_from_context_mean": sparse_tv(level["pred"], pred_mean),
                        "ce_residual_cosine": residual_cos,
                    }
                )

            context_rows.append(
                {
                    **dict(zip(CTX, key)),
                    "n_levels": n_levels,
                    "ce_span": levels[-1]["ce"] - levels[0]["ce"],
                    "true_tv_from_mean": true_tv_mean,
                    "pred_tv_from_mean": pred_tv_mean,
                    "variation_ratio": pred_tv_mean / true_tv_mean if true_tv_mean > 0 else np.nan,
                    "mean_ce_residual_cosine": float(np.nanmean(residual_cosines)),
                    "node_tv_from_mean": node_tv_mean,
                    "node_sampled": node_mean is not None,
                }
            )

            for low, high in zip(levels[:-1], levels[1:]):
                true_delta = sparse_linear_combination(high["true"], 1.0, low["true"], -1.0)
                pred_delta = sparse_linear_combination(high["pred"], 1.0, low["pred"], -1.0)
                row = {
                    **dict(zip(CTX, key)),
                    "ce_low": low["ce"],
                    "ce_high": high["ce"],
                    "ce_gap": high["ce"] - low["ce"],
                    "true_peak_tv": sparse_tv(low["true"], high["true"]),
                    "pred_peak_tv": sparse_tv(low["pred"], high["pred"]),
                    "delta_cosine": sparse_cosine(true_delta, pred_delta),
                    "delta_dot": sparse_dot(true_delta, pred_delta),
                    "true_delta_l2_sq": sparse_l2_sq(true_delta),
                    "pred_delta_l2_sq": sparse_l2_sq(pred_delta),
                    "observed_peak_switch": top_bin(low["true"]) != top_bin(high["true"]),
                    "predicted_peak_switch": top_bin(low["pred"]) != top_bin(high["pred"]),
                    "node_sampled": low["node"] is not None and high["node"] is not None,
                    "node_tv": np.nan,
                    "node_cosine": np.nan,
                    "node_top_switch": np.nan,
                }
                if row["node_sampled"]:
                    row["node_tv"] = float(0.5 * np.abs(low["node"] - high["node"]).sum())
                    denom = np.linalg.norm(low["node"]) * np.linalg.norm(high["node"])
                    row["node_cosine"] = float(low["node"].dot(high["node"]) / denom)
                    row["node_top_switch"] = bool(
                        np.argmax(low["node"]) != np.argmax(high["node"])
                    )
                adjacent_rows.append(row)

    contexts = pd.DataFrame(context_rows)
    level_df = pd.DataFrame(level_rows)
    adjacent = pd.DataFrame(adjacent_rows)

    summaries = {}
    for inst in sorted(contexts["inst_type"].unique()):
        c = contexts[contexts["inst_type"] == inst]
        levels_i = level_df[level_df["inst_type"] == inst]
        adj = adjacent[adjacent["inst_type"] == inst]
        adj_nodes = adj[adj["node_sampled"]]
        summaries[inst] = {
            "n_contexts": len(c),
            "n_molecules": int(c["mol_id"].nunique()),
            "true_tv_from_context_mean": bootstrap_mean_ci(c, "true_tv_from_mean"),
            "pred_tv_from_context_mean": bootstrap_mean_ci(c, "pred_tv_from_mean"),
            "variation_ratio_context_mean": bootstrap_mean_ci(c.dropna(), "variation_ratio"),
            "ce_residual_cosine": bootstrap_mean_ci(
                levels_i.dropna(subset=["ce_residual_cosine"]), "ce_residual_cosine"
            ),
            "specific_minus_loo_average_tv": bootstrap_mean_ci(
                levels_i, "specific_minus_average_tv"
            ),
            "fraction_predictions_closer_to_loo_average": bootstrap_mean_ci(
                levels_i, "closer_to_loo_average"
            ),
            "adjacent_true_peak_tv": bootstrap_mean_ci(adj, "true_peak_tv"),
            "adjacent_pred_peak_tv": bootstrap_mean_ci(adj, "pred_peak_tv"),
            "adjacent_delta_cosine": bootstrap_mean_ci(
                adj.dropna(subset=["delta_cosine"]), "delta_cosine"
            ),
            "adjacent_delta_cosine_given_observed_peak_switch": bootstrap_mean_ci(
                adj.dropna(subset=["delta_cosine"]),
                "delta_cosine",
                "observed_peak_switch",
            ),
            "adjacent_optimal_l2_gain": bootstrap_ratio_of_sums_ci(
                adj,
                "delta_dot",
                "pred_delta_l2_sq",
            ),
            "adjacent_optimal_l2_gain_given_observed_peak_switch": (
                bootstrap_ratio_of_sums_ci(
                    adj,
                    "delta_dot",
                    "pred_delta_l2_sq",
                    "observed_peak_switch",
                )
            ),
            "node_sample_n_molecules": int(adj_nodes["mol_id"].nunique()),
            "adjacent_node_tv": bootstrap_mean_ci(adj_nodes, "node_tv"),
            "adjacent_node_cosine": bootstrap_mean_ci(adj_nodes, "node_cosine"),
            "node_top_switch_given_observed_peak_switch": bootstrap_mean_ci(
                adj_nodes, "node_top_switch", "observed_peak_switch"
            ),
        }

    contexts.to_csv(OUT_DP / "contexts.csv", index=False)
    level_df.to_csv(OUT_DP / "ce_levels.csv", index=False)
    adjacent.to_csv(OUT_DP / "adjacent_ce.csv", index=False)
    (OUT_DP / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
