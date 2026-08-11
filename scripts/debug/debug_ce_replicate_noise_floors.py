"""Measure replicate floors for CE-dependent winner and direction changes.

The analysis uses acquisition metadata from a completed validation scan only to identify the
validation spectrum IDs, fixed contexts, and the exact filled NCE values used by the model. All
reported replicate targets are computed directly from observed spectra. An adjacent-transition
model diagnostic is read only to make a cohort-matched directional comparison.

For same-CE groups, the winner-flip floor is the fraction of replicate pairs whose base masses
disagree under the configured instrument tolerance. For adjacent CE levels with at least two
replicates at both endpoints, two delta estimates are formed from disjoint replicate choices and
their cosine is measured on the same 0.01-Da grid as the provisional model diagnostic.

Example:
    /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
        scripts/debug/debug_ce_replicate_noise_floors.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

CTX = ["mol_id", "prec_type", "inst_type", "frag_mode"]
BIN_WIDTH = 0.01
BOOTSTRAP_SEED = 17
N_BOOT = 5000
MIN_CONCLUSIVE_MOLS = 30
MIN_CONCLUSIVE_TRANSITIONS = 50
MODEL_JOIN_KEYS = CTX + ["ce_low", "ce_high"]

INST_TOLERANCE = {
    "FT": (1.0e-5, 200.0),
    "QTOF": (2.0e-5, 750.0),
}
DEFAULT_TOLERANCE = (1.0e-5, 200.0)

def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--spec_fp",
        type=Path,
        default=Path("data/proc/nist20/spec_df.pkl"),
        help="Observed NIST spectrum dataframe.",
    )
    parser.add_argument(
        "--metadata_fp",
        type=Path,
        default=Path("/tmp/base-peak-ce-structure-9t63zils/per_spectrum.csv"),
        help="Validation IDs, contexts, and filled model_nce values from the base-peak scan.",
    )
    parser.add_argument(
        "--out_dp",
        type=Path,
        default=Path("/tmp/ce-replicate-noise-floors-9t63zils"),
        help="Output directory.",
    )
    parser.add_argument(
        "--model_adjacent_fp",
        type=Path,
        default=Path("/tmp/ce-static-fragment-weight-9t63zils/adjacent_ce.csv"),
        help="Adjacent-transition output used for the cohort-matched model comparison.",
    )
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def binned_dist(peaks: object) -> dict[int, float]:
    """Return a sum-normalized sparse spectrum on the frozen 0.01-Da grid."""
    peak_array = np.asarray(peaks, dtype=float)
    if peak_array.size == 0:
        return {}
    if peak_array.ndim != 2 or peak_array.shape[1] != 2:
        raise ValueError(f"Expected an [n, 2] peak array, got {peak_array.shape}")
    mzs = peak_array[:, 0]
    intensities = peak_array[:, 1]
    total = float(intensities.sum())
    if total <= 0:
        return {}
    bins = np.rint(mzs / BIN_WIDTH).astype(np.int64)
    unique, inverse = np.unique(bins, return_inverse=True)
    values = np.bincount(inverse, weights=intensities) / total
    return {int(key): float(value) for key, value in zip(unique, values) if value > 0}


def raw_base_mz(peaks: object) -> float:
    """Return the raw observed base-peak mass, breaking exact intensity ties by lower mass."""
    peak_array = np.asarray(peaks, dtype=float)
    if peak_array.size == 0:
        return np.nan
    if peak_array.ndim != 2 or peak_array.shape[1] != 2:
        raise ValueError(f"Expected an [n, 2] peak array, got {peak_array.shape}")
    max_intensity = float(peak_array[:, 1].max())
    return float(peak_array[peak_array[:, 1] == max_intensity, 0].min())


def average_sparse(distributions: list[dict[int, float]]) -> dict[int, float]:
    """Average sparse distributions without changing their common mass grid."""
    if not distributions:
        return {}
    result: dict[int, float] = {}
    for distribution in distributions:
        for key, value in distribution.items():
            result[key] = result.get(key, 0.0) + value
    scale = 1.0 / len(distributions)
    return {key: value * scale for key, value in result.items()}


def sparse_delta(high: dict[int, float], low: dict[int, float]) -> dict[int, float]:
    """Return high minus low on the union of two sparse supports."""
    result = {key: -value for key, value in low.items()}
    for key, value in high.items():
        result[key] = result.get(key, 0.0) + value
    return {key: value for key, value in result.items() if abs(value) > 1.0e-15}


def sparse_cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Return cosine similarity for two sparse vectors."""
    norm_a = np.sqrt(sum(value * value for value in a.values()))
    norm_b = np.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return np.nan
    dot = sum(a[key] * b[key] for key in a.keys() & b.keys())
    return float(dot / (norm_a * norm_b))


def top_mz(distribution: dict[int, float]) -> float:
    """Return the deterministic base-peak mass represented by a sparse distribution."""
    if not distribution:
        return np.nan
    top_bin = min(distribution, key=lambda key: (-distribution[key], key))
    return float(top_bin * BIN_WIDTH)


def same_mass(mz_a: float, mz_b: float, inst_type: str) -> bool:
    """Compare two observed masses under a symmetric instrument-relative tolerance."""
    if not np.isfinite(mz_a) or not np.isfinite(mz_b):
        return False
    tolerance, min_mz = INST_TOLERANCE.get(inst_type, DEFAULT_TOLERANCE)
    divisor = max(min(mz_a, mz_b), min_mz)
    return bool(abs(mz_a - mz_b) / divisor < tolerance)


def cluster_ratio_ci(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap a ratio of sums by resampling molecules."""
    grouped = (
        df.groupby("mol_id", observed=True)[[numerator_col, denominator_col]]
        .sum()
        .to_numpy(float)
    )
    denominator = grouped[:, 1].sum()
    if len(grouped) == 0 or denominator <= 0:
        return np.nan, np.nan, np.nan
    point = grouped[:, 0].sum() / denominator
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        sample = grouped[rng.integers(0, len(grouped), len(grouped))]
        estimates[index] = sample[:, 0].sum() / sample[:, 1].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(point), float(low), float(high)


def cluster_mean_ci(
    df: pd.DataFrame,
    value_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap a row-weighted mean with molecule-cluster resampling."""
    grouped = (
        df.groupby("mol_id", observed=True)[value_col]
        .agg(["sum", "count"])
        .to_numpy(float)
    )
    if len(grouped) == 0 or grouped[:, 1].sum() <= 0:
        return np.nan, np.nan, np.nan
    point = grouped[:, 0].sum() / grouped[:, 1].sum()
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        sample = grouped[rng.integers(0, len(grouped), len(grouped))]
        estimates[index] = sample[:, 0].sum() / sample[:, 1].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(point), float(low), float(high)


def summarize_winner_floor(
    winner_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    """Summarize replicate winner disagreement by instrument and adduct."""
    rows: list[dict[str, object]] = []
    groupings = [(["inst_type"], True), (["inst_type", "prec_type"], False)]
    for columns, aggregate_adduct in groupings:
        for key, group in winner_df.groupby(columns, observed=True):
            keys = (key,) if isinstance(key, str) else key
            inst_type = str(keys[0])
            prec_type = "ALL" if aggregate_adduct else str(keys[1])
            estimate = cluster_ratio_ci(group, "n_winner_flips", "n_pairs", n_boot, seed)
            rows.append(
                {
                    "inst_type": inst_type,
                    "prec_type": prec_type,
                    "n_replicate_levels": int(len(group)),
                    "n_replicate_pairs": int(group["n_pairs"].sum()),
                    "n_molecules": int(group["mol_id"].nunique()),
                    "winner_flip_floor": estimate,
                }
            )
    return rows


def summarize_delta_floor(
    delta_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    """Summarize replicate delta agreement by instrument, adduct, and switch status."""
    rows: list[dict[str, object]] = []
    groupings = [(["inst_type"], True), (["inst_type", "prec_type"], False)]
    for columns, aggregate_adduct in groupings:
        for key, stratum in delta_df.groupby(columns, observed=True):
            keys = (key,) if isinstance(key, str) else key
            inst_type = str(keys[0])
            prec_type = "ALL" if aggregate_adduct else str(keys[1])
            for subset_name, subset in [
                ("all", stratum),
                ("stable", stratum[~stratum["consensus_winner_switch"]]),
                ("switch", stratum[stratum["consensus_winner_switch"]]),
            ]:
                valid = subset.dropna(subset=["replicate_delta_cosine"])
                estimate = cluster_mean_ci(valid, "replicate_delta_cosine", n_boot, seed)
                n_molecules = int(valid["mol_id"].nunique())
                n_transitions = int(len(valid))
                rows.append(
                    {
                        "inst_type": inst_type,
                        "prec_type": prec_type,
                        "transition_subset": subset_name,
                        "n_transitions": n_transitions,
                        "n_molecules": n_molecules,
                        "n_disjoint_comparisons": int(valid["n_delta_comparisons"].sum()),
                        "replicate_delta_cosine": estimate,
                        "coverage_conclusive": bool(
                            n_molecules >= MIN_CONCLUSIVE_MOLS
                            and n_transitions >= MIN_CONCLUSIVE_TRANSITIONS
                        ),
                    }
                )
    return rows


def summarize_paired_model_comparison(
    delta_df: pd.DataFrame,
    model_adjacent_fp: Path,
    n_boot: int,
    seed: int,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    """Compare replicate and model delta cosines on identical switching transitions."""
    model_df = pd.read_csv(model_adjacent_fp)
    model_values = [
        "delta_cosine",
        "observed_peak_switch",
        "true_delta_l2_sq",
        "ce_gap",
    ]
    required = set(MODEL_JOIN_KEYS + model_values)
    missing = required - set(model_df.columns)
    if missing:
        raise KeyError(f"Missing model-adjacent columns: {sorted(missing)}")
    model_df = model_df[MODEL_JOIN_KEYS + model_values]
    if model_df.duplicated(MODEL_JOIN_KEYS).any():
        raise ValueError("Model-adjacent output contains duplicate transition rows.")

    paired = delta_df.merge(
        model_df,
        on=MODEL_JOIN_KEYS,
        how="left",
        validate="one_to_one",
    )
    if paired["delta_cosine"].isna().any():
        n_missing = int(paired["delta_cosine"].isna().sum())
        raise ValueError(f"Model-adjacent output is missing {n_missing} replicate transitions.")
    switch_disagreement = (
        paired["consensus_winner_switch"] != paired["observed_peak_switch"]
    )
    n_switch_disagreements = int(switch_disagreement.sum())
    n_ft_switch_disagreements = int(
        (switch_disagreement & (paired["inst_type"] == "FT")).sum()
    )
    if n_ft_switch_disagreements:
        raise ValueError(
            "Replicate and model diagnostics disagree on "
            f"{n_ft_switch_disagreements} FT switch labels."
        )
    paired["replicate_minus_model"] = (
        paired["replicate_delta_cosine"] - paired["delta_cosine"]
    )

    rows: list[dict[str, object]] = []
    paired_switches = paired[paired["consensus_winner_switch"]]
    groupings = [(["inst_type"], True), (["inst_type", "prec_type"], False)]
    for columns, aggregate_adduct in groupings:
        for key, group in paired_switches.groupby(columns, observed=True):
            keys = (key,) if isinstance(key, str) else key
            n_molecules = int(group["mol_id"].nunique())
            n_transitions = int(len(group))
            rows.append(
                {
                    "inst_type": str(keys[0]),
                    "prec_type": "ALL" if aggregate_adduct else str(keys[1]),
                    "n_transitions": n_transitions,
                    "n_molecules": n_molecules,
                    "replicate_delta_cosine": cluster_mean_ci(
                        group, "replicate_delta_cosine", n_boot, seed
                    ),
                    "model_delta_cosine": cluster_mean_ci(
                        group, "delta_cosine", n_boot, seed
                    ),
                    "replicate_minus_model": cluster_mean_ci(
                        group, "replicate_minus_model", n_boot, seed
                    ),
                    "coverage_conclusive": bool(
                        n_molecules >= MIN_CONCLUSIVE_MOLS
                        and n_transitions >= MIN_CONCLUSIVE_TRANSITIONS
                    ),
                }
            )
    ft_switches = model_df[
        (model_df["inst_type"] == "FT") & model_df["observed_peak_switch"]
    ].copy()
    replicate_switch_keys = pd.MultiIndex.from_frame(
        delta_df[
            (delta_df["inst_type"] == "FT") & delta_df["consensus_winner_switch"]
        ][MODEL_JOIN_KEYS]
    )
    ft_switches["replicate_matched"] = pd.MultiIndex.from_frame(
        ft_switches[MODEL_JOIN_KEYS]
    ).isin(replicate_switch_keys)
    ft_switches["true_delta_l2"] = np.sqrt(ft_switches["true_delta_l2_sq"])

    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]
    ft_switches["true_delta_l2_quartile"] = pd.qcut(
        ft_switches["true_delta_l2"],
        len(quartile_labels),
        labels=quartile_labels,
    )
    matched = ft_switches[ft_switches["replicate_matched"]]
    unmatched = ft_switches[~ft_switches["replicate_matched"]]
    shared_molecules = set(matched["mol_id"]) & set(unmatched["mol_id"])
    shared = ft_switches[ft_switches["mol_id"].isin(shared_molecules)]
    quartile_rows = []
    for quartile, group in ft_switches.groupby("true_delta_l2_quartile", observed=True):
        row: dict[str, object] = {"quartile": str(quartile)}
        for name, subset in [
            ("matched", group[group["replicate_matched"]]),
            ("unmatched", group[~group["replicate_matched"]]),
        ]:
            row[f"n_{name}"] = int(len(subset))
            row[f"model_delta_cosine_{name}"] = float(subset["delta_cosine"].mean())
        quartile_rows.append(row)

    coverage_summary = {
        "n_ft_switches": int(len(ft_switches)),
        "n_replicate_matched": int(len(matched)),
        "replicate_coverage_fraction": float(len(matched) / len(ft_switches)),
        "matched_median_true_delta_l2": float(matched["true_delta_l2"].median()),
        "unmatched_median_true_delta_l2": float(unmatched["true_delta_l2"].median()),
        "matched_median_ce_gap": float(matched["ce_gap"].median()),
        "unmatched_median_ce_gap": float(unmatched["ce_gap"].median()),
        "matched_model_delta_cosine": float(matched["delta_cosine"].mean()),
        "unmatched_model_delta_cosine": float(unmatched["delta_cosine"].mean()),
        "matched_fraction_in_q4": float(
            (matched["true_delta_l2_quartile"] == "Q4").mean()
        ),
        "n_shared_molecules": int(len(shared_molecules)),
        "shared_molecule_matched_model_delta_cosine": cluster_mean_ci(
            shared[shared["replicate_matched"]], "delta_cosine", n_boot, seed
        ),
        "shared_molecule_unmatched_model_delta_cosine": cluster_mean_ci(
            shared[~shared["replicate_matched"]], "delta_cosine", n_boot, seed
        ),
        "true_delta_l2_quartiles": quartile_rows,
    }
    magnitude_rows: list[dict[str, object]] = []
    for subset_name, is_switch in [("stable", False), ("switch", True)]:
        population = model_df[
            (model_df["inst_type"] == "FT")
            & (model_df["observed_peak_switch"] == is_switch)
        ]
        boundaries = np.quantile(
            np.sqrt(population["true_delta_l2_sq"]),
            [0.25, 0.5, 0.75],
        )
        subset = paired[
            (paired["inst_type"] == "FT")
            & (paired["consensus_winner_switch"] == is_switch)
        ].copy()
        subset["magnitude_quartile"] = np.digitize(
            np.sqrt(subset["true_delta_l2_sq"]),
            boundaries,
            right=True,
        ) + 1
        for quartile in range(1, 5):
            group = subset[subset["magnitude_quartile"] == quartile]
            magnitude_rows.append(
                {
                    "transition_subset": subset_name,
                    "magnitude_quartile": f"Q{quartile}",
                    "upper_true_delta_l2": (
                        float(boundaries[quartile - 1]) if quartile < 4 else None
                    ),
                    "n_transitions": int(len(group)),
                    "n_molecules": int(group["mol_id"].nunique()),
                    "replicate_delta_cosine": cluster_mean_ci(
                        group, "replicate_delta_cosine", n_boot, seed
                    ),
                    "model_delta_cosine": cluster_mean_ci(
                        group, "delta_cosine", n_boot, seed
                    ),
                    "replicate_minus_model": cluster_mean_ci(
                        group, "replicate_minus_model", n_boot, seed
                    ),
                }
            )
        magnitude_rows.append(
            {
                "transition_subset": subset_name,
                "magnitude_quartile": "ALL",
                "upper_true_delta_l2": None,
                "n_transitions": int(len(subset)),
                "n_molecules": int(subset["mol_id"].nunique()),
                "replicate_delta_cosine": cluster_mean_ci(
                    subset, "replicate_delta_cosine", n_boot, seed
                ),
                "model_delta_cosine": cluster_mean_ci(
                    subset, "delta_cosine", n_boot, seed
                ),
                "replicate_minus_model": cluster_mean_ci(
                    subset, "replicate_minus_model", n_boot, seed
                ),
            }
        )
    join_integrity = {
        "n_replicate_rows": int(len(delta_df)),
        "n_merged_rows": int(len(delta_df)),
        "n_replicate_duplicate_keys": int(delta_df.duplicated(MODEL_JOIN_KEYS).sum()),
        "n_model_duplicate_keys": int(model_df.duplicated(MODEL_JOIN_KEYS).sum()),
        "n_switch_label_disagreements": n_switch_disagreements,
        "n_ft_switch_label_disagreements": n_ft_switch_disagreements,
    }
    return rows, coverage_summary, magnitude_rows, join_integrity


def json_safe(value: object) -> object:
    """Recursively replace non-finite floats so the report is strict JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    """Run the model-free replicate analysis and write auditable outputs."""
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.n_boot < 1:
        raise ValueError(f"n_boot must be positive, got {args.n_boot}")
    args.out_dp.mkdir(parents=True, exist_ok=True)

    # 1. Load the validation cohort and observed spectra.
    metadata = pd.read_csv(args.metadata_fp)
    required = set(CTX + ["spec_id", "model_nce"])
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"Missing required metadata columns: {sorted(missing)}")
    metadata = metadata[metadata["model_nce"].notna()][list(required)].copy()
    metadata["ce_key"] = metadata["model_nce"].round(6)
    if metadata["spec_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate spec_id rows.")

    LOGGER.info("Loading observed spectrum dataframe: %s", args.spec_fp)
    spectra = pd.read_pickle(args.spec_fp)
    wanted_ids = set(metadata["spec_id"].astype(int))
    spectra = spectra[spectra["spec_id"].astype(int).isin(wanted_ids)][["spec_id", "peaks"]]
    if spectra["spec_id"].duplicated().any():
        raise ValueError("Observed spectrum dataframe contains duplicate selected spec_id rows.")
    peaks_by_spec_id: dict[int, dict[int, float]] = {}
    base_mz_by_spec_id: dict[int, float] = {}
    for row in spectra.itertuples(index=False):
        spec_id = int(row.spec_id)
        peaks_by_spec_id[spec_id] = binned_dist(row.peaks)
        base_mz_by_spec_id[spec_id] = raw_base_mz(row.peaks)
    missing_ids = wanted_ids - peaks_by_spec_id.keys()
    if missing_ids:
        raise KeyError(f"Missing {len(missing_ids)} validation spectra from observed dataframe.")
    del spectra

    # 2. Build same-CE winner-pair rows and adjacent double-replicate transition rows.
    winner_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for context_key, context in metadata.groupby(CTX, observed=True):
        mol_id, prec_type, inst_type, frag_mode = context_key
        levels: list[dict[str, object]] = []
        for ce_key, ce_group in context.groupby("ce_key", observed=True):
            spec_ids = sorted(ce_group["spec_id"].astype(int).tolist())
            distributions = [peaks_by_spec_id[spec_id] for spec_id in spec_ids]
            valid = [
                (spec_id, distribution, base_mz_by_spec_id[spec_id])
                for spec_id, distribution in zip(spec_ids, distributions)
                if distribution and np.isfinite(base_mz_by_spec_id[spec_id])
            ]
            if not valid:
                continue
            spec_ids = [item[0] for item in valid]
            distributions = [item[1] for item in valid]
            base_mzs = [item[2] for item in valid]
            consensus = average_sparse(distributions)
            levels.append(
                {
                    "ce_key": float(ce_key),
                    "spec_ids": spec_ids,
                    "distributions": distributions,
                    "base_mzs": base_mzs,
                    "consensus": consensus,
                }
            )

            if len(distributions) >= 2:
                n_pairs = 0
                n_flips = 0
                for mz_a, mz_b in itertools.combinations(base_mzs, 2):
                    n_pairs += 1
                    n_flips += int(not same_mass(mz_a, mz_b, str(inst_type)))
                winner_rows.append(
                    {
                        "mol_id": int(mol_id),
                        "prec_type": str(prec_type),
                        "inst_type": str(inst_type),
                        "frag_mode": str(frag_mode),
                        "ce_key": float(ce_key),
                        "n_replicates": len(distributions),
                        "n_pairs": n_pairs,
                        "n_winner_flips": n_flips,
                    }
                )

        levels.sort(key=lambda item: item["ce_key"])
        for low, high in zip(levels[:-1], levels[1:]):
            low_dists = low["distributions"]
            high_dists = high["distributions"]
            if len(low_dists) < 2 or len(high_dists) < 2:
                continue

            cosines = []
            for low_a, low_b in itertools.combinations(low_dists, 2):
                for high_a, high_b in itertools.combinations(high_dists, 2):
                    aligned = sparse_cosine(
                        sparse_delta(high_a, low_a),
                        sparse_delta(high_b, low_b),
                    )
                    crossed = sparse_cosine(
                        sparse_delta(high_b, low_a),
                        sparse_delta(high_a, low_b),
                    )
                    if np.isfinite(aligned):
                        cosines.append(aligned)
                    if np.isfinite(crossed):
                        cosines.append(crossed)

            low_top = top_mz(low["consensus"])
            high_top = top_mz(high["consensus"])
            delta_rows.append(
                {
                    "mol_id": int(mol_id),
                    "prec_type": str(prec_type),
                    "inst_type": str(inst_type),
                    "frag_mode": str(frag_mode),
                    "ce_low": float(low["ce_key"]),
                    "ce_high": float(high["ce_key"]),
                    "n_low_replicates": len(low_dists),
                    "n_high_replicates": len(high_dists),
                    "n_delta_comparisons": len(cosines),
                    "consensus_winner_switch": not same_mass(
                        low_top,
                        high_top,
                        str(inst_type),
                    ),
                    "replicate_delta_cosine": (
                        float(np.mean(cosines)) if cosines else np.nan
                    ),
                }
            )

    winner_df = pd.DataFrame(winner_rows)
    delta_df = pd.DataFrame(delta_rows)
    if winner_df.empty:
        raise ValueError("No same-CE replicate groups were found.")
    if delta_df.empty:
        raise ValueError("No adjacent CE pairs had at least two replicates at both endpoints.")

    # 3. Summarize coverage before interpreting either floor.
    winner_summary = summarize_winner_floor(winner_df, args.n_boot, args.seed)
    delta_summary = summarize_delta_floor(delta_df, args.n_boot, args.seed)
    paired_model_summary, ft_ceiling_coverage, magnitude_summary, join_integrity = (
        summarize_paired_model_comparison(
            delta_df,
            args.model_adjacent_fp,
            args.n_boot,
            args.seed,
        )
    )
    report = json_safe(
        {
            "method": {
                "replicate_statistics_model_free": True,
                "paired_model_comparison_fp": str(args.model_adjacent_fp),
                "mass_grid_da": BIN_WIDTH,
                "winner_identity": "instrument-relative tolerance",
                "delta_unit": "one adjacent transition, averaged over disjoint replicate pairings",
                "bootstrap_unit": "molecule",
                "n_boot": args.n_boot,
                "seed": args.seed,
                "conclusive_coverage": {
                    "min_molecules": MIN_CONCLUSIVE_MOLS,
                    "min_transitions": MIN_CONCLUSIVE_TRANSITIONS,
                },
            },
            "winner_flip_floor": winner_summary,
            "replicate_delta_cosine": delta_summary,
            "paired_model_delta_cosine": paired_model_summary,
            "ft_switch_ceiling_coverage": ft_ceiling_coverage,
            "ft_paired_magnitude_strata": magnitude_summary,
            "join_integrity": join_integrity,
        }
    )

    # 4. Write detailed rows and the compact report.
    winner_df.to_csv(args.out_dp / "same_ce_winner_pairs.csv", index=False)
    delta_df.to_csv(args.out_dp / "adjacent_replicate_delta_cosine.csv", index=False)
    (args.out_dp / "summary.json").write_text(json.dumps(report, indent=2))
    LOGGER.info("Wrote outputs to %s", args.out_dp)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
