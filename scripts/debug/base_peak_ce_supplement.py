"""Analyze CE-dependent base-peak transitions from a completed base-peak diagnostic.

The input directory must contain ``per_spectrum.csv``, ``ce_levels.csv``, and
``cross_ce_pairs.csv`` produced by the larger base-peak analysis. This supplement measures
within-context low-to-high CE degradation, adjacent-CE winner changes, and molecule-clustered
uncertainty.

Example:
    python scripts/debug/base_peak_ce_supplement.py \
        --analysis_dp /tmp/base-peak-ce-structure-9t63zils
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CTX = ["mol_id", "prec_type", "inst_type", "frag_mode"]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--analysis_dp",
        type=Path,
        default=Path("/tmp/base-peak-ce-structure-9t63zils"),
        help="Directory containing the main base-peak analysis CSV files.",
    )
    return parser.parse_args()


def bootstrap_context_delta(
    df: pd.DataFrame,
    low_col: str,
    high_col: str,
    n_boot: int = 5000,
    seed: int = 17,
) -> tuple[float, float, float]:
    by_mol = df.assign(delta=df[high_col] - df[low_col]).groupby("mol_id")["delta"].mean()
    values = by_mol.to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        estimates[i] = rng.choice(values, size=len(values), replace=True).mean()
    return float(values.mean()), *np.quantile(estimates, [0.025, 0.975]).tolist()


def conditional_cluster_ci(
    df: pd.DataFrame,
    value: str,
    condition: str,
    n_boot: int = 3000,
    seed: int = 17,
) -> tuple[float, float, float]:
    work = df[df[condition]].copy()
    grouped = work.groupby("mol_id")[value].agg(["sum", "count"]).to_numpy(float)
    point = grouped[:, 0].sum() / grouped[:, 1].sum()
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sample = grouped[rng.integers(0, len(grouped), len(grouped))]
        estimates[i] = sample[:, 0].sum() / sample[:, 1].sum()
    return float(point), *np.quantile(estimates, [0.025, 0.975]).tolist()


def main() -> None:
    args = get_args()
    analysis_dp = args.analysis_dp
    required = ["per_spectrum.csv", "ce_levels.csv", "cross_ce_pairs.csv"]
    missing = [name for name in required if not (analysis_dp / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required analysis files in {analysis_dp}: {', '.join(missing)}"
        )

    spectra = pd.read_csv(analysis_dp / "per_spectrum.csv")
    levels = pd.read_csv(analysis_dp / "ce_levels.csv")
    pairs = pd.read_csv(analysis_dp / "cross_ce_pairs.csv")

    # Conditional accuracy among spectra whose correct base mass survives export filtering.
    nce = (
        spectra.groupby(["inst_type", "nce_bin"], observed=True)
        .agg(
            n=("spec_id", "size"),
            n_mols=("mol_id", "nunique"),
            hit=("base_peak_hit", "mean"),
            support=("base_supported_export", "mean"),
            mean_rel_int=("base_pred_rel_int", "mean"),
        )
        .reset_index()
    )
    nce["hit_given_support"] = nce["hit"] / nce["support"]

    # Paired first-vs-last energy in each fixed context.
    endpoints = []
    adjacent = []
    for key, group in levels.groupby(CTX, observed=True):
        g = group.sort_values("ce_key")
        if len(g) < 2:
            continue
        lo, hi = g.iloc[0], g.iloc[-1]
        endpoints.append(
            {
                **dict(zip(CTX, key)),
                "n_levels": len(g),
                "ce_span": hi.ce_key - lo.ce_key,
                "low_hit": float(lo.consensus_hit),
                "high_hit": float(hi.consensus_hit),
                "low_mean_hit": lo.mean_hit,
                "high_mean_hit": hi.mean_hit,
            }
        )
        records = g.to_dict("records")
        for lo_r, hi_r in zip(records[:-1], records[1:]):
            # m/z comparisons use a conservative fixed 0.02 Da here only for sign filtering;
            # exact hit/switch fields were already instrument-tolerance matched in the main scan.
            true_delta = hi_r["true_base_mz"] - lo_r["true_base_mz"]
            pred_delta = hi_r["pred_base_mz"] - lo_r["pred_base_mz"]
            true_changed = abs(true_delta) > 0.02
            pred_changed = abs(pred_delta) > 0.02
            adjacent.append(
                {
                    **dict(zip(CTX, key)),
                    "ce_gap": hi_r["ce_key"] - lo_r["ce_key"],
                    "true_delta_mz": true_delta,
                    "pred_delta_mz": pred_delta,
                    "true_changed": true_changed,
                    "pred_changed": pred_changed,
                    "both_hit": bool(lo_r["consensus_hit"] and hi_r["consensus_hit"]),
                    "direction_agree": bool(
                        true_changed and pred_changed and np.sign(true_delta) == np.sign(pred_delta)
                    ),
                }
            )
    endpoints = pd.DataFrame(endpoints)
    adjacent = pd.DataFrame(adjacent)

    # Same endpoint comparison at spectrum level, retaining support and ranking components.
    endpoint_components = []
    spectra_finite = spectra[spectra["model_nce"].notna()].copy()
    spectra_finite["ce_key"] = spectra_finite["model_nce"].round(6)
    ce_agg = (
        spectra_finite.groupby(CTX + ["ce_key"], observed=True)
        .agg(
            hit=("base_peak_hit", "mean"),
            support=("base_supported_export", "mean"),
            rel_int=("base_pred_rel_int", "mean"),
            cosine=("cosine", "mean"),
            oracle=("oracle_cosine", "mean"),
        )
        .reset_index()
    )
    for key, group in ce_agg.groupby(CTX, observed=True):
        g = group.sort_values("ce_key")
        if len(g) < 2:
            continue
        lo, hi = g.iloc[0], g.iloc[-1]
        endpoint_components.append(
            {
                **dict(zip(CTX, key)),
                "ce_span": hi.ce_key - lo.ce_key,
                **{f"low_{c}": lo[c] for c in ["hit", "support", "rel_int", "cosine", "oracle"]},
                **{f"high_{c}": hi[c] for c in ["hit", "support", "rel_int", "cosine", "oracle"]},
            }
        )
    endpoint_components = pd.DataFrame(endpoint_components)

    endpoint_summary = []
    for inst, g in endpoints.groupby("inst_type", observed=True):
        endpoint_summary.append(
            {
                "inst_type": inst,
                "n_contexts": len(g),
                "n_mols": g.mol_id.nunique(),
                "mean_ce_span": g.ce_span.mean(),
                "low_consensus_hit": g.low_hit.mean(),
                "high_consensus_hit": g.high_hit.mean(),
                "paired_high_minus_low": bootstrap_context_delta(g, "low_hit", "high_hit"),
                "low_spectrum_hit": g.low_mean_hit.mean(),
                "high_spectrum_hit": g.high_mean_hit.mean(),
                "paired_spectrum_high_minus_low": bootstrap_context_delta(
                    g, "low_mean_hit", "high_mean_hit"
                ),
            }
        )

    adjacent_summary = []
    for inst, g in adjacent.groupby("inst_type", observed=True):
        changed = g[g.true_changed]
        adjacent_summary.append(
            {
                "inst_type": inst,
                "n_adjacent_transitions": len(g),
                "n_mols": g.mol_id.nunique(),
                "observed_change": g.true_changed.mean(),
                "predicted_change": g.pred_changed.mean(),
                "predicted_change_given_observed": changed.pred_changed.mean(),
                "correct_mass_direction_given_observed": changed.direction_agree.mean(),
                "both_endpoints_correct_given_observed": changed.both_hit.mean(),
                "median_true_delta_given_observed": changed.true_delta_mz.median(),
                "median_pred_delta_given_observed": changed.pred_delta_mz.median(),
                "corr_true_pred_delta_given_observed": changed[["true_delta_mz", "pred_delta_mz"]]
                .corr()
                .iloc[0, 1],
            }
        )

    component_summary = []
    for inst, g in endpoint_components.groupby("inst_type", observed=True):
        row = {
            "inst_type": inst,
            "n_contexts": len(g),
            "n_mols": g.mol_id.nunique(),
        }
        for metric in ["hit", "support", "rel_int", "cosine", "oracle"]:
            row[f"low_{metric}"] = g[f"low_{metric}"].mean()
            row[f"high_{metric}"] = g[f"high_{metric}"].mean()
            row[f"delta_{metric}"] = bootstrap_context_delta(g, f"low_{metric}", f"high_{metric}")
        row["low_hit_given_support"] = row["low_hit"] / row["low_support"]
        row["high_hit_given_support"] = row["high_hit"] / row["high_support"]
        component_summary.append(row)

    # Pair-level molecule bootstrap for the key distinction: responsiveness vs identity.
    pair_key = {}
    for inst in ["FT", "QTOF"]:
        g = pairs[pairs.inst_type.eq(inst)].copy()
        pair_key[inst] = {
            "pred_switch_given_obs_switch": conditional_cluster_ci(g, "pred_switch", "obs_switch"),
            "both_endpoint_hit_given_obs_switch": conditional_cluster_ci(
                g, "both_endpoint_hit", "obs_switch"
            ),
        }

    out = {
        "endpoint_summary": endpoint_summary,
        "endpoint_component_summary": component_summary,
        "adjacent_summary": adjacent_summary,
        "pair_cluster_intervals": pair_key,
    }
    (analysis_dp / "supplement.json").write_text(json.dumps(out, indent=2))
    nce.to_csv(analysis_dp / "by_nce_conditional.csv", index=False)
    endpoints.to_csv(analysis_dp / "ce_context_endpoints.csv", index=False)
    endpoint_components.to_csv(analysis_dp / "ce_context_endpoint_components.csv", index=False)
    adjacent.to_csv(analysis_dp / "ce_adjacent_transitions.csv", index=False)
    print(json.dumps(out, indent=2))
    print("\nCONDITIONAL NCE\n", nce.to_string(index=False))


if __name__ == "__main__":
    main()
