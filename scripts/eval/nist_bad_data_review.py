"""Build a ranked, human-reviewable queue of suspicious NIST spectra.

The detector deliberately emits independent flags rather than an automatic delete decision.
``review_priority`` is only a triage score; it is not a probability that a spectrum is bad.

Example:
    python scripts/eval/nist_bad_data_review.py \
        --spec_fp data/proc/nist20/spec_df.pkl \
        --mol_fp data/proc/nist20/mol_df.pkl \
        --out_csv data/eval/nist20_bad_data_review.csv \
        --out_html data/eval/nist20_bad_data_review.html
"""

from __future__ import annotations

import argparse
import hashlib
import html
import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from oracle_dataset_utils import resolve_dsets_for_spec_fp

from fragnnet.utils.formula_utils import (
    C_MASS,
    H_MASS,
    N_MASS,
    O_MASS,
    PREC_TYPE_TO_MASS_DIFF,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Keep aligned with INST_TYPE_LOSS_TOL in dag_oracle_scores.py and the NIST20 experiment config.
INSTRUMENT_TOLERANCE = {
    "FT": (1.0e-5, 200.0),
    "QTOF": (2.0e-5, 750.0),
}
DEFAULT_TOLERANCE = (1.0e-5, 200.0)
# PREC_TYPE_TO_MASS_DIFF is primarily an ion-carrier mapping used for fragment peaks. Neutral-loss
# labels require distinct precursor-m/z shifts. Formula-dependent ``+2i`` isotope-selected labels
# are deliberately excluded from this simple monoisotopic precursor check.
PRECURSOR_TYPE_TO_MASS_DIFF = {
    **{
        prec_type: mass_diff
        for prec_type, mass_diff in PREC_TYPE_TO_MASS_DIFF.items()
        if "+2i" not in prec_type
    },
    "[M+H-NH3]+": PREC_TYPE_TO_MASS_DIFF["[M+H]+"] - N_MASS - 3 * H_MASS,
    "[M-H-CO2]-": PREC_TYPE_TO_MASS_DIFF["[M-H]-"] - C_MASS - 2 * O_MASS,
}

HARD_FLAG_WEIGHTS = {
    "invalid_peaks": 5,
    "precursor_mass_mismatch": 4,
    "missing_precursor_mz": 4,
    "polarity_mismatch": 4,
    "duplicate_spectrum_cross_molecule": 4,
    "missing_molecule_metadata": 4,
    "empty_spectrum": 3,
}
HEURISTIC_FLAG_WEIGHTS = {
    "substantial_intensity_above_precursor": 2,
    "missing_collision_energy": 1,
    "negative_collision_energy": 2,
    "sparse_spectrum": 1,
    "single_peak_dominates": 1,
    "unsorted_peak_mz": 1,
    "duplicate_peak_mz": 1,
    "replicate_consensus_disagreement": 3,
    "substantial_cross_ce_orphan_intensity": 2,
    "substantial_uwpr_contaminant_only_intensity": 2,
}

REPLICATE_LOW_SIMILARITY = 0.5
REPLICATE_NORMAL_SIMILARITY = 0.8
CROSS_CE_ORPHAN_FRACTION = 0.2
UWPR_CONTAMINANT_ONLY_FRACTION = 0.05


def _is_missing(value: object) -> bool:
    """Return whether a scalar value is missing."""
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(value))


def _tolerance_for_inst_type(inst_type: str) -> tuple[float, float]:
    """Return relative tolerance and minimum-m/z floor for one instrument type."""
    return INSTRUMENT_TOLERANCE.get(inst_type, DEFAULT_TOLERANCE)


def _spectrum_hash(mzs: np.ndarray, intensities: np.ndarray) -> str | None:
    """Hash a normalized spectrum for exact-export duplicate detection."""
    if len(mzs) == 0 or not np.isfinite(intensities).all() or intensities.max(initial=0.0) <= 0:
        return None
    normalized = intensities / intensities.max()
    payload = np.column_stack((np.round(mzs, 4), np.round(normalized, 5))).tobytes()
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def _polarity_mismatch(prec_type: object, ion_mode: object) -> bool:
    """Check whether the adduct charge sign conflicts with NIST ion mode."""
    if not isinstance(prec_type, str) or ion_mode not in {"P", "N"}:
        return False
    if prec_type.endswith("+") or prec_type.endswith("+_NL"):
        return ion_mode != "P"
    if prec_type.endswith("-") or prec_type.endswith("-_NL"):
        return ion_mode != "N"
    return False


def _collision_energy_flags(row: object) -> tuple[bool, bool]:
    """Return missing-all-CE and any-negative-CE flags."""
    values = [
        getattr(row, column)
        for column in ("ace", "nce", "ace_extra_1", "nce_extra_1", "ace_extra_2", "nce_extra_2")
    ]
    finite_values = [float(value) for value in values if not _is_missing(value)]
    return len(finite_values) == 0, any(value < 0 for value in finite_values)


def _valid_peak_arrays(peaks: object) -> tuple[np.ndarray, np.ndarray] | None:
    """Return finite, positive-total peak arrays or None for an unusable spectrum."""
    try:
        peak_array = np.asarray(peaks, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        peak_array.ndim != 2
        or peak_array.shape[1] != 2
        or len(peak_array) == 0
        or not np.isfinite(peak_array).all()
        or (peak_array[:, 0] <= 0).any()
        or (peak_array[:, 1] < 0).any()
        or peak_array[:, 1].sum() <= 0
    ):
        return None
    return peak_array[:, 0], peak_array[:, 1]


def _mz_tolerance(inst_type: str) -> float:
    """Return a conservative absolute tolerance for spectrum-to-spectrum matching."""
    return {"FT": 0.02, "QTOF": 0.05}.get(inst_type, 0.5)


def _spectral_cosine(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    mz_tol: float,
) -> float:
    """Calculate one-to-one, square-root-intensity cosine similarity."""
    first_mzs, first_ints = first
    second_mzs, second_ints = second
    first_weights = np.sqrt(first_ints)
    second_weights = np.sqrt(second_ints)
    denominator = np.linalg.norm(first_weights) * np.linalg.norm(second_weights)
    if denominator <= 0:
        return math.nan

    candidate_pairs: list[tuple[float, int, int]] = []
    second_order = np.argsort(second_mzs)
    sorted_second_mzs = second_mzs[second_order]
    for first_idx, mz in enumerate(first_mzs):
        lo = np.searchsorted(sorted_second_mzs, mz - mz_tol, side="left")
        hi = np.searchsorted(sorted_second_mzs, mz + mz_tol, side="right")
        for sorted_idx in range(lo, hi):
            second_idx = int(second_order[sorted_idx])
            contribution = float(first_weights[first_idx] * second_weights[second_idx])
            candidate_pairs.append((contribution, first_idx, second_idx))

    numerator = 0.0
    used_first: set[int] = set()
    used_second: set[int] = set()
    for contribution, first_idx, second_idx in sorted(candidate_pairs, reverse=True):
        if first_idx in used_first or second_idx in used_second:
            continue
        numerator += contribution
        used_first.add(first_idx)
        used_second.add(second_idx)
    return numerator / denominator


def _add_ce_columns(spec_df: pd.DataFrame) -> pd.DataFrame:
    """Add the same primary CE representation used by the Oracle sibling audit."""
    out = spec_df.copy()
    ace = pd.to_numeric(out["ace"], errors="coerce")
    nce = pd.to_numeric(out["nce"], errors="coerce")
    use_nce = out["inst_type"].eq("FT") & nce.notna()
    out["ce_value"] = ace.where(ace.notna(), nce).mask(use_nce, nce)
    out["ce_source"] = np.where(
        use_nce,
        "nce",
        np.where(ace.notna(), "ace", np.where(nce.notna(), "nce", "none")),
    )
    out["ce_bin"] = out["ce_value"].round(6)
    return out


def _attach_signals(
    review_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    defaults: dict[str, object],
) -> pd.DataFrame:
    """Attach spec_id-indexed signal columns without changing row order."""
    out = review_df.copy()
    if signal_df.empty:
        for column, default in defaults.items():
            out[column] = default
        return out
    indexed = signal_df.drop_duplicates("spec_id").set_index("spec_id")
    for column, default in defaults.items():
        mapped = out["spec_id"].map(indexed[column])
        if isinstance(default, bool):
            out[column] = mapped.astype("boolean").fillna(default).astype(bool)
        elif isinstance(default, int):
            out[column] = pd.to_numeric(mapped, errors="coerce").fillna(default).astype(int)
        else:
            out[column] = pd.to_numeric(mapped, errors="coerce").fillna(default)
    return out


def add_replicate_consensus_signals(spec_df: pd.DataFrame, review_df: pd.DataFrame) -> pd.DataFrame:
    """Flag a same-CE replicate only when the remaining replicates agree."""
    scoped = _add_ce_columns(spec_df)
    scoped = scoped[scoped["ce_bin"].notna()]
    records: list[dict[str, object]] = []

    # 1. Compare groups with enough replicates to form an independent peer consensus.
    for _, group in scoped.groupby(["group_id", "ce_source", "ce_bin"], observed=True):
        if len(group) < 3:
            continue
        valid_rows = []
        for row in group.itertuples(index=False):
            peak_arrays = _valid_peak_arrays(row.peaks)
            if peak_arrays is not None:
                valid_rows.append((row, peak_arrays))
        if len(valid_rows) < 3:
            continue

        similarities = np.eye(len(valid_rows), dtype=float)
        for first_idx in range(len(valid_rows)):
            for second_idx in range(first_idx + 1, len(valid_rows)):
                similarity = _spectral_cosine(
                    valid_rows[first_idx][1],
                    valid_rows[second_idx][1],
                    _mz_tolerance(str(valid_rows[first_idx][0].inst_type)),
                )
                similarities[first_idx, second_idx] = similarity
                similarities[second_idx, first_idx] = similarity

        # 2. A candidate must disagree with every peer while those peers agree with each other.
        for candidate_idx, (row, _) in enumerate(valid_rows):
            peer_idxs = [idx for idx in range(len(valid_rows)) if idx != candidate_idx]
            best_similarity = float(similarities[candidate_idx, peer_idxs].max())
            peer_pairs = [
                similarities[first_idx, second_idx]
                for offset, first_idx in enumerate(peer_idxs)
                for second_idx in peer_idxs[offset + 1 :]
            ]
            peer_consensus = float(np.median(peer_pairs))
            records.append(
                {
                    "spec_id": int(row.spec_id),
                    "n_same_ce_replicates": len(valid_rows),
                    "replicate_best_similarity": best_similarity,
                    "other_replicate_consensus": peer_consensus,
                    "replicate_consensus_disagreement": (
                        best_similarity <= REPLICATE_LOW_SIMILARITY
                        and peer_consensus >= REPLICATE_NORMAL_SIMILARITY
                    ),
                }
            )

    defaults = {
        "n_same_ce_replicates": 0,
        "replicate_best_similarity": math.nan,
        "other_replicate_consensus": math.nan,
        "replicate_consensus_disagreement": False,
    }
    return _attach_signals(review_df, pd.DataFrame.from_records(records), defaults)


def _peaks_match_any(
    mzs: np.ndarray, sorted_comparison_mzs: np.ndarray, mz_tol: float
) -> np.ndarray:
    """Vectorize whether each peak has a comparison peak within tolerance."""
    insertion = np.searchsorted(sorted_comparison_mzs, mzs, side="left")
    right_idx = np.minimum(insertion, len(sorted_comparison_mzs) - 1)
    left_idx = np.maximum(insertion - 1, 0)
    right_match = (insertion < len(sorted_comparison_mzs)) & (
        np.abs(sorted_comparison_mzs[right_idx] - mzs) <= mz_tol
    )
    left_match = (insertion > 0) & (np.abs(sorted_comparison_mzs[left_idx] - mzs) <= mz_tol)
    return right_match | left_match


def add_cross_ce_orphan_signals(spec_df: pd.DataFrame, review_df: pd.DataFrame) -> pd.DataFrame:
    """Measure intensity absent from every one of at least two other CE bins."""
    scoped = _add_ce_columns(spec_df)
    scoped = scoped[scoped["ce_bin"].notna()]
    records: list[dict[str, object]] = []

    # 1. Build one union of observed m/z values for each CE in a series.
    for _, group in scoped.groupby(["group_id", "ce_source"], observed=True):
        ce_mzs: dict[float, np.ndarray] = {}
        valid_rows: list[tuple[object, tuple[np.ndarray, np.ndarray]]] = []
        for row in group.itertuples(index=False):
            peak_arrays = _valid_peak_arrays(row.peaks)
            if peak_arrays is None:
                continue
            valid_rows.append((row, peak_arrays))
            ce_bin = float(row.ce_bin)
            ce_mzs.setdefault(ce_bin, np.array([], dtype=float))
            ce_mzs[ce_bin] = np.concatenate((ce_mzs[ce_bin], peak_arrays[0]))
        ce_mzs = {ce: np.unique(mzs) for ce, mzs in ce_mzs.items()}
        if len(ce_mzs) < 3:
            continue
        comparison_mzs_by_ce = {
            ce: np.unique(
                np.concatenate(
                    [other_mzs for other_ce, other_mzs in ce_mzs.items() if other_ce != ce]
                )
            )
            for ce in ce_mzs
        }

        # 2. A peak is orphaned only when it is absent from every other CE bin.
        for row, (mzs, intensities) in valid_rows:
            comparison_mzs = comparison_mzs_by_ce[float(row.ce_bin)]
            mz_tol = _mz_tolerance(str(row.inst_type))
            orphan_mask = ~_peaks_match_any(mzs, comparison_mzs, mz_tol)
            orphan_fraction = float(intensities[orphan_mask].sum() / intensities.sum())
            records.append(
                {
                    "spec_id": int(row.spec_id),
                    "n_other_ce_bins": len(ce_mzs) - 1,
                    "n_cross_ce_orphan_peaks": int(orphan_mask.sum()),
                    "cross_ce_orphan_intensity_fraction": orphan_fraction,
                    "substantial_cross_ce_orphan_intensity": (
                        orphan_fraction >= CROSS_CE_ORPHAN_FRACTION
                    ),
                }
            )

    defaults = {
        "n_other_ce_bins": 0,
        "n_cross_ce_orphan_peaks": 0,
        "cross_ce_orphan_intensity_fraction": math.nan,
        "substantial_cross_ce_orphan_intensity": False,
    }
    return _attach_signals(review_df, pd.DataFrame.from_records(records), defaults)


def _matched_peak_stats(
    peaks: object, target_mzs: Sequence[float], mz_tol: float = 1.0e-4
) -> tuple[int, float]:
    """Return unique matched-peak count and matched intensity fraction."""
    peak_arrays = _valid_peak_arrays(peaks)
    if peak_arrays is None or not target_mzs:
        return 0, 0.0
    mzs, intensities = peak_arrays
    matched = np.zeros(len(mzs), dtype=bool)
    for target_mz in np.unique(np.asarray(target_mzs, dtype=float)):
        matched |= np.abs(mzs - target_mz) <= mz_tol
    return int(matched.sum()), float(intensities[matched].sum() / intensities.sum())


def add_uwpr_signals(
    spec_df: pd.DataFrame,
    review_df: pd.DataFrame,
    feasibility_csv: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> pd.DataFrame:
    """Attach contaminant matches, requiring consensus across supplied DAG audits."""
    if isinstance(feasibility_csv, (str, os.PathLike)):
        feasibility_paths = [feasibility_csv]
    else:
        feasibility_paths = list(feasibility_csv)
    if not feasibility_paths:
        raise ValueError("At least one UWPR feasibility CSV is required")

    required_cols = ["spec_id", "peak_mz", "feasibility"]
    known_labels = {
        "dag_feasible",
        "contaminant_only",
        "no_dag_file",
        "unsupported_prec",
        "missing_metadata",
    }
    labels_by_source: list[dict[tuple[int, float], str]] = []
    all_match_keys: set[tuple[int, float]] = set()
    for path in feasibility_paths:
        hits = pd.read_csv(path, usecols=required_cols)
        if hits[required_cols].isna().any().any():
            raise ValueError(f"UWPR feasibility CSV contains missing required values: {path}")
        unknown_labels = set(hits["feasibility"].unique()) - known_labels
        if unknown_labels:
            raise ValueError(f"Unknown UWPR feasibility labels: {sorted(unknown_labels)}")
        hits = hits.drop_duplicates(["spec_id", "peak_mz", "feasibility"])
        conflicting = hits.groupby(["spec_id", "peak_mz"])["feasibility"].nunique().gt(1)
        if conflicting.any():
            raise ValueError(f"UWPR feasibility CSV has conflicting labels for one peak: {path}")
        source_labels = {
            (int(row.spec_id), float(row.peak_mz)): str(row.feasibility)
            for row in hits.itertuples(index=False)
        }
        labels_by_source.append(source_labels)
        all_match_keys.update(source_labels)

    all_matches: dict[int, list[float]] = {}
    conservative_matches: dict[int, list[float]] = {}
    disagreement_matches: dict[int, list[float]] = {}
    for key in all_match_keys:
        labels = [source.get(key) for source in labels_by_source]
        spec_id, peak_mz = key
        all_matches.setdefault(spec_id, []).append(peak_mz)
        if all(label == "contaminant_only" for label in labels):
            conservative_matches.setdefault(spec_id, []).append(peak_mz)
        evaluable_labels = {
            label for label in labels if label in {"dag_feasible", "contaminant_only"}
        }
        if len(evaluable_labels) > 1:
            disagreement_matches.setdefault(spec_id, []).append(peak_mz)

    records = []
    wanted_ids = set(all_matches)
    for row in spec_df[spec_df["spec_id"].isin(wanted_ids)].itertuples(index=False):
        n_all, fraction_all = _matched_peak_stats(row.peaks, all_matches[int(row.spec_id)])
        n_conservative, fraction_conservative = _matched_peak_stats(
            row.peaks, conservative_matches.get(int(row.spec_id), [])
        )
        n_disagreement, fraction_disagreement = _matched_peak_stats(
            row.peaks, disagreement_matches.get(int(row.spec_id), [])
        )
        records.append(
            {
                "spec_id": int(row.spec_id),
                "n_uwpr_feasibility_sources": len(feasibility_paths),
                "n_uwpr_matched_peaks": n_all,
                "uwpr_matched_intensity_fraction": fraction_all,
                "n_uwpr_contaminant_only_peaks": n_conservative,
                "uwpr_contaminant_only_intensity_fraction": fraction_conservative,
                "n_uwpr_dag_disagreement_peaks": n_disagreement,
                "uwpr_dag_disagreement_intensity_fraction": fraction_disagreement,
                "substantial_uwpr_contaminant_only_intensity": (
                    fraction_conservative >= UWPR_CONTAMINANT_ONLY_FRACTION
                ),
            }
        )

    defaults = {
        "n_uwpr_feasibility_sources": len(feasibility_paths),
        "n_uwpr_matched_peaks": 0,
        "uwpr_matched_intensity_fraction": 0.0,
        "n_uwpr_contaminant_only_peaks": 0,
        "uwpr_contaminant_only_intensity_fraction": 0.0,
        "n_uwpr_dag_disagreement_peaks": 0,
        "uwpr_dag_disagreement_intensity_fraction": 0.0,
        "substantial_uwpr_contaminant_only_intensity": False,
    }
    return _attach_signals(review_df, pd.DataFrame.from_records(records), defaults)


def analyze_spectrum(row: object, mol_meta: dict[int, tuple[float, str]]) -> dict[str, object]:
    """Compute independent hard flags and softer review heuristics for one spectrum."""
    reasons: list[str] = []
    hard_reasons: list[str] = []
    heuristic_reasons: list[str] = []
    peaks = row.peaks

    empty_spectrum = peaks is None or len(peaks) == 0
    invalid_peaks = False
    mzs = np.array([], dtype=float)
    intensities = np.array([], dtype=float)
    if not empty_spectrum:
        try:
            peak_array = np.asarray(peaks, dtype=float)
            invalid_peaks = peak_array.ndim != 2 or peak_array.shape[1] != 2
            if not invalid_peaks:
                mzs = peak_array[:, 0]
                intensities = peak_array[:, 1]
                invalid_peaks = bool(
                    (~np.isfinite(peak_array)).any()
                    or (mzs <= 0).any()
                    or (intensities < 0).any()
                    or intensities.sum() <= 0
                )
        except (TypeError, ValueError):
            invalid_peaks = True

    exact_mw, formula = mol_meta.get(int(row.mol_id), (math.nan, ""))
    missing_molecule_metadata = not np.isfinite(exact_mw)
    expected_prec_mz = math.nan
    precursor_error_da = math.nan
    precursor_error_ppm = math.nan
    precursor_mass_mismatch = False
    if (
        not missing_molecule_metadata
        and row.prec_type in PRECURSOR_TYPE_TO_MASS_DIFF
        and not _is_missing(row.prec_mz)
    ):
        expected_prec_mz = exact_mw + PRECURSOR_TYPE_TO_MASS_DIFF[row.prec_type]
        precursor_error_da = float(row.prec_mz - expected_prec_mz)
        precursor_error_ppm = 1.0e6 * precursor_error_da / expected_prec_mz
        tolerance, tolerance_min_mz = _tolerance_for_inst_type(row.inst_type)
        allowed_error_da = tolerance * max(expected_prec_mz, tolerance_min_mz)
        precursor_mass_mismatch = abs(precursor_error_da) > allowed_error_da

    total_intensity = (
        float(intensities.sum()) if len(intensities) and not invalid_peaks else math.nan
    )
    intensity_above_precursor_fraction = math.nan
    dominant_peak_fraction = math.nan
    n_peaks = int(len(mzs)) if not invalid_peaks else 0
    if n_peaks and not _is_missing(row.prec_mz):
        tolerance, tolerance_min_mz = _tolerance_for_inst_type(row.inst_type)
        cutoff = float(row.prec_mz) + tolerance * max(float(row.prec_mz), tolerance_min_mz)
        intensity_above_precursor_fraction = float(
            intensities[mzs > cutoff].sum() / total_intensity
        )
        dominant_peak_fraction = float(intensities.max() / total_intensity)

    missing_precursor_mz = _is_missing(row.prec_mz) or not np.isfinite(float(row.prec_mz))
    flags = {
        "empty_spectrum": empty_spectrum,
        "invalid_peaks": invalid_peaks,
        "missing_precursor_mz": missing_precursor_mz,
        "missing_molecule_metadata": missing_molecule_metadata,
        "precursor_mass_mismatch": precursor_mass_mismatch,
        "polarity_mismatch": _polarity_mismatch(row.prec_type, row.ion_mode),
    }
    missing_ce, negative_ce = _collision_energy_flags(row)
    heuristic_flags = {
        "missing_collision_energy": missing_ce,
        "negative_collision_energy": negative_ce,
        "sparse_spectrum": not invalid_peaks and n_peaks <= 2,
        "substantial_intensity_above_precursor": (
            np.isfinite(intensity_above_precursor_fraction)
            and intensity_above_precursor_fraction >= 0.05
        ),
        "single_peak_dominates": (
            n_peaks >= 3 and np.isfinite(dominant_peak_fraction) and dominant_peak_fraction >= 0.98
        ),
        "unsupported_precursor_type": row.prec_type not in PRECURSOR_TYPE_TO_MASS_DIFF,
        "unsorted_peak_mz": n_peaks >= 2 and bool(np.any(np.diff(mzs) < 0)),
        "duplicate_peak_mz": n_peaks >= 2 and len(np.unique(mzs)) != n_peaks,
    }
    hard_reasons.extend(name for name, enabled in flags.items() if enabled)
    heuristic_reasons.extend(name for name, enabled in heuristic_flags.items() if enabled)
    reasons.extend(hard_reasons)
    reasons.extend(heuristic_reasons)

    return {
        "spec_id": int(row.spec_id),
        "mol_id": int(row.mol_id),
        "dset_spec_id": row.dset_spec_id,
        "formula": formula,
        "prec_type": row.prec_type,
        "ion_mode": row.ion_mode,
        "inst_type": row.inst_type,
        "frag_mode": row.frag_mode,
        "ace": row.ace,
        "nce": row.nce,
        "prec_mz": row.prec_mz,
        "expected_prec_mz": expected_prec_mz,
        "precursor_error_da": precursor_error_da,
        "precursor_error_ppm": precursor_error_ppm,
        "n_peaks": n_peaks,
        "intensity_above_precursor_fraction": intensity_above_precursor_fraction,
        "dominant_peak_fraction": dominant_peak_fraction,
        "spectrum_hash": _spectrum_hash(mzs, intensities) if not invalid_peaks else None,
        **flags,
        **heuristic_flags,
        "hard_reasons": ";".join(hard_reasons),
        "heuristic_reasons": ";".join(heuristic_reasons),
        "review_reasons": ";".join(reasons),
    }


def mark_cross_molecule_duplicates(review_df: pd.DataFrame) -> None:
    """Flag identical normalized spectra assigned to more than one molecule, in place."""
    valid = review_df[review_df["spectrum_hash"].notna()]
    molecule_counts = valid.groupby("spectrum_hash")["mol_id"].nunique()
    suspicious_hashes = set(molecule_counts[molecule_counts > 1].index)
    review_df["duplicate_spectrum_cross_molecule"] = review_df["spectrum_hash"].isin(
        suspicious_hashes
    )


def finalize_review_queue(review_df: pd.DataFrame) -> pd.DataFrame:
    """Add duplicate reasons, priority, and blank human-review fields."""
    mark_cross_molecule_duplicates(review_df)
    duplicate_flag = review_df["duplicate_spectrum_cross_molecule"]
    review_df.loc[duplicate_flag, "hard_reasons"] = review_df.loc[
        duplicate_flag, "hard_reasons"
    ].map(lambda value: ";".join(filter(None, [value, "duplicate_spectrum_cross_molecule"])))
    review_df.loc[duplicate_flag, "review_reasons"] = review_df.loc[
        duplicate_flag, "review_reasons"
    ].map(lambda value: ";".join(filter(None, [value, "duplicate_spectrum_cross_molecule"])))

    dynamic_flags = [
        "replicate_consensus_disagreement",
        "substantial_cross_ce_orphan_intensity",
        "substantial_uwpr_contaminant_only_intensity",
    ]
    for flag in dynamic_flags:
        if flag not in review_df:
            review_df[flag] = False
        enabled = review_df[flag].astype(bool)
        review_df.loc[enabled, "heuristic_reasons"] = review_df.loc[
            enabled, "heuristic_reasons"
        ].map(lambda value, reason=flag: ";".join(filter(None, [value, reason])))
        review_df.loc[enabled, "review_reasons"] = review_df.loc[enabled, "review_reasons"].map(
            lambda value, reason=flag: ";".join(filter(None, [value, reason]))
        )

    review_df["review_priority"] = 0
    for flag, weight in {**HARD_FLAG_WEIGHTS, **HEURISTIC_FLAG_WEIGHTS}.items():
        if flag not in review_df:
            review_df[flag] = False
        review_df["review_priority"] += review_df[flag].astype(int) * weight

    family_flags = {
        "record_integrity": list(HARD_FLAG_WEIGHTS),
        "replicate_consensus": ["replicate_consensus_disagreement"],
        "cross_ce_orphans": ["substantial_cross_ce_orphan_intensity"],
        "contaminant_library": ["substantial_uwpr_contaminant_only_intensity"],
    }
    family_enabled = pd.DataFrame(
        {family: review_df[flags].any(axis=1) for family, flags in family_flags.items()},
        index=review_df.index,
    )
    review_df["evidence_family_count"] = family_enabled.sum(axis=1)
    review_df["evidence_families"] = family_enabled.apply(
        lambda row: ";".join(row.index[row].tolist()), axis=1
    )
    review_df["review_status"] = "unreviewed"
    review_df["reviewer"] = ""
    review_df["review_notes"] = ""
    candidates = review_df[review_df["review_priority"] > 0].copy()
    return candidates.sort_values(
        ["review_priority", "spec_id"], ascending=[False, True]
    ).reset_index(drop=True)


def _spectrum_svg(peaks: Sequence[Sequence[float]], width: int = 520, height: int = 160) -> str:
    """Render a dependency-free SVG stick spectrum for the HTML review report."""
    if not peaks:
        return "<em>Empty spectrum</em>"
    peak_array = np.asarray(peaks, dtype=float)
    if peak_array.ndim != 2 or peak_array.shape[1] != 2 or not np.isfinite(peak_array).all():
        return "<em>Malformed spectrum</em>"
    max_mz = max(float(peak_array[:, 0].max()), 1.0)
    max_intensity = max(float(peak_array[:, 1].max()), 1.0)
    lines = []
    for mz, intensity in peak_array:
        x = 5 + (width - 10) * float(mz) / max_mz
        y = height - 5 - (height - 10) * max(float(intensity), 0.0) / max_intensity
        lines.append(f'<line x1="{x:.1f}" y1="{height - 5}" x2="{x:.1f}" y2="{y:.1f}"/>')
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="mass spectrum">'
        f'<g stroke="#2457a6" stroke-width="1">{"".join(lines)}</g>'
        f'<line x1="5" y1="{height - 5}" x2="{width - 5}" y2="{height - 5}" '
        'stroke="#555"/></svg>'
    )


def write_review_html(
    candidates: pd.DataFrame, peaks_by_spec_id: dict[int, Sequence[Sequence[float]]], out_fp: str
) -> None:
    """Write a visual companion to the editable review CSV."""
    cards = []
    for row in candidates.itertuples(index=False):
        precursor_text = f"{float(row.prec_mz):.5f}" if not _is_missing(row.prec_mz) else "missing"
        metadata = (
            f"spec_id={row.spec_id} | NIST id={html.escape(str(row.dset_spec_id))} | "
            f"mol_id={row.mol_id} | {html.escape(str(row.formula))} | "
            f"{html.escape(str(row.prec_type))} | {html.escape(str(row.inst_type))} | "
            f"precursor={precursor_text}"
        )
        evidence = (
            f"families={html.escape(str(row.evidence_families))} | "
            f"replicate_best={row.replicate_best_similarity:.3f} | "
            f"cross_CE_orphan_intensity={row.cross_ce_orphan_intensity_fraction:.3f} | "
            f"UWPR_contaminant_only_intensity="
            f"{row.uwpr_contaminant_only_intensity_fraction:.3f} | "
            f"UWPR_DAG_disagreement_intensity="
            f"{row.uwpr_dag_disagreement_intensity_fraction:.3f}"
        )
        cards.append(
            "<article><h2>Priority "
            f"{row.review_priority}: {html.escape(row.review_reasons)}</h2><p>{metadata}</p>"
            f"<p>{evidence}</p>"
            f"{_spectrum_svg(peaks_by_spec_id.get(row.spec_id, []))}</article>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>NIST spectrum review</title>
<style>body{{font:14px sans-serif;max-width:1100px;margin:auto}}article{{border-bottom:1px solid #bbb;
padding:12px}}h2{{font-size:16px}}svg{{width:100%;height:180px;background:#fafafa}}</style></head>
<body><h1>NIST suspicious-spectrum review</h1>
<p>Use the companion CSV to record keep/exclude/needs_followup decisions. Priority is a triage
score, not a bad-data probability.</p>{"".join(cards)}</body></html>"""
    os.makedirs(os.path.dirname(out_fp) or ".", exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as handle:
        handle.write(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--dsets", "--dset", nargs="+", default=None)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_html", default=None)
    parser.add_argument(
        "--uwpr_feasibility_csv",
        nargs="+",
        default=["data/uwpr/nist20_uwpr_hits_feasibility.csv"],
        help=(
            "One or more UWPR/MaConDa feasibility files. Multiple files require unanimous "
            "contaminant_only labels before increasing review priority."
        ),
    )
    parser.add_argument("--max_html_spectra", type=int, default=500)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug only; disables full-dataset duplicate detection.",
    )
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)

    log.info("Loading spectra from %s", args.spec_fp)
    spec_df = pd.read_pickle(args.spec_fp)
    if args.dsets:
        spec_df = spec_df[spec_df["dset"].isin(args.dsets)]
    if args.limit is not None:
        spec_df = spec_df.head(args.limit)
        log.warning("--limit is active; duplicate flags apply only to this subset")

    mol_df = pd.read_pickle(args.mol_fp)[["mol_id", "exact_mw", "formula"]]
    mol_meta = {
        int(row.mol_id): (float(row.exact_mw), str(row.formula))
        for row in mol_df.itertuples(index=False)
    }
    log.info("Computing review signals for %d spectra", len(spec_df))
    records = [analyze_spectrum(row, mol_meta) for row in spec_df.itertuples(index=False)]
    review_df = pd.DataFrame.from_records(records)
    log.info("Computing same-CE replicate consensus signals")
    review_df = add_replicate_consensus_signals(spec_df, review_df)
    log.info("Computing cross-CE orphan-intensity signals")
    review_df = add_cross_ce_orphan_signals(spec_df, review_df)
    if args.uwpr_feasibility_csv:
        uwpr_paths = [Path(path) for path in args.uwpr_feasibility_csv]
        missing_paths = [path for path in uwpr_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"UWPR feasibility CSV does not exist: {missing_paths}")
        log.info("Loading UWPR/MaConDa evidence from %s", uwpr_paths)
        review_df = add_uwpr_signals(spec_df, review_df, uwpr_paths)
    else:
        review_df = _attach_signals(
            review_df,
            pd.DataFrame(),
            {
                "n_uwpr_matched_peaks": 0,
                "uwpr_matched_intensity_fraction": 0.0,
                "n_uwpr_contaminant_only_peaks": 0,
                "uwpr_contaminant_only_intensity_fraction": 0.0,
                "n_uwpr_feasibility_sources": 0,
                "n_uwpr_dag_disagreement_peaks": 0,
                "uwpr_dag_disagreement_intensity_fraction": 0.0,
                "substantial_uwpr_contaminant_only_intensity": False,
            },
        )
    candidates = finalize_review_queue(review_df)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    candidates.to_csv(args.out_csv, index=False)
    log.info("Saved %d review candidates to %s", len(candidates), args.out_csv)

    if args.out_html:
        html_candidates = candidates.head(args.max_html_spectra)
        wanted_ids = set(html_candidates["spec_id"])
        peaks_by_spec_id = {
            int(row.spec_id): row.peaks
            for row in spec_df[spec_df["spec_id"].isin(wanted_ids)][
                ["spec_id", "peaks"]
            ].itertuples(index=False)
        }
        write_review_html(html_candidates, peaks_by_spec_id, args.out_html)
        log.info("Saved visual review report to %s", args.out_html)


if __name__ == "__main__":
    main()
