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
import functools
import hashlib
import html
import logging
import math
import os
import textwrap
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from dag_oracle_scores import _allowed_formula_idxs, _dag_ion_mzs, _detect_is_compressed
from matplotlib.backends.backend_pdf import PdfPages
from oracle_dataset_utils import resolve_dsets_for_spec_fp
from rdkit import Chem
from rdkit.Chem import Draw

import fragnnet.utils.frag_utils as frag_utils
from fragnnet.utils import data_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Keep aligned with INST_TYPE_LOSS_TOL in dag_oracle_scores.py and the NIST20 experiment config.
INSTRUMENT_TOLERANCE = {
    "FT": (1.0e-5, 200.0),
    "QTOF": (2.0e-5, 750.0),
}
DEFAULT_TOLERANCE = (1.0e-5, 200.0)
PRECURSOR_MASS_CHECK_INSTRUMENTS = set(INSTRUMENT_TOLERANCE)
# Formula-dependent ``+2i`` isotope-selected labels are deliberately excluded from this simple
# monoisotopic precursor check.
PRECURSOR_TYPE_TO_MASS_DIFF = {
    prec_type: mass_diff
    for prec_type, mass_diff in PREC_TYPE_TO_MASS_DIFF.items()
    if prec_type and "+2i" not in prec_type
}
PRECURSOR_TYPE_ALIASES = {
    "[M+H-H2O]+": "[M-H2O+H]+",
    "[M+H-2H2O]+": "[M-2H2O+H]+",
    "[M+CHO2]-": "[M+HCOOH-H]-",
    "M-H": "[M-H]-",
}

HARD_FLAG_WEIGHTS = {
    "invalid_peaks": 5,
    "polarity_mismatch": 4,
    "duplicate_spectrum_formula_conflict": 5,
    "missing_molecule_metadata": 4,
    "empty_spectrum": 3,
}
HEURISTIC_FLAG_WEIGHTS = {
    "precursor_mass_mismatch": 1,
    "missing_precursor_mz": 1,
    "substantial_intensity_above_precursor": 2,
    "missing_collision_energy": 1,
    "negative_collision_energy": 2,
    "single_peak_dominates": 1,
    "unsorted_peak_mz": 1,
    "duplicate_peak_mz": 1,
    "replicate_consensus_disagreement": 3,
    "substantial_cross_ce_orphan_intensity": 2,
    "substantial_uwpr_contaminant_only_intensity": 2,
    "duplicate_spectrum_structure_ambiguity": 1,
}
REVIEW_FLAG_GLOSSARY = {
    "invalid_peaks": (
        "Hard",
        "The peak array is malformed or contains non-finite m/z values, non-positive m/z "
        "values, negative intensities, or no positive total intensity.",
    ),
    "polarity_mismatch": (
        "Hard",
        "The charge sign written in the precursor/adduct label disagrees with the recorded "
        "positive or negative ion mode.",
    ),
    "duplicate_spectrum_formula_conflict": (
        "Hard",
        "The same exact-export spectrum with at least three peaks is assigned to molecules "
        "having different molecular formulas.",
    ),
    "missing_molecule_metadata": (
        "Hard",
        "The spectrum's molecule ID has no usable exact molecular mass in the molecule table.",
    ),
    "empty_spectrum": (
        "Hard",
        "The spectrum contains no peaks.",
    ),
    "precursor_mass_mismatch": (
        "Heuristic",
        "The recorded precursor m/z differs from exact molecular mass plus the canonicalized "
        "adduct shift by more than the instrument tolerance for an FT or QTOF spectrum. NIST "
        "precursor m/z is usually theoretical, so this is an integrity/regression guard for "
        "label or structure inconsistency, not a measured-mass-quality test.",
    ),
    "missing_precursor_mz": (
        "Heuristic",
        "The record has no finite precursor m/z, so precursor-based checks cannot be run.",
    ),
    "substantial_intensity_above_precursor": (
        "Heuristic",
        "At least 5% of total intensity lies more than 4.5 Da above the precursor, outside the "
        "protected precursor isotope-envelope window.",
    ),
    "missing_collision_energy": (
        "Heuristic",
        "None of the supported absolute or normalized collision-energy fields is present.",
    ),
    "negative_collision_energy": (
        "Heuristic",
        "At least one recorded collision-energy value is negative.",
    ),
    "sparse_spectrum": (
        "Context",
        "The spectrum has only one or two peaks. This is a low-priority prompt for human "
        "review, not evidence that the spectrum is wrong.",
    ),
    "single_peak_dominates": (
        "Heuristic",
        "The spectrum has at least three peaks and one peak carries at least 98% of total "
        "intensity.",
    ),
    "unsorted_peak_mz": (
        "Heuristic",
        "Peak m/z values are not stored in ascending order.",
    ),
    "duplicate_peak_mz": (
        "Heuristic",
        "Two or more entries in the same spectrum have exactly the same m/z value.",
    ),
    "replicate_consensus_disagreement": (
        "Heuristic",
        "Within at least three same-collision-energy replicates, this spectrum has cosine "
        "similarity at most 0.50 to every peer while the other peers agree at 0.80 or higher.",
    ),
    "substantial_cross_ce_orphan_intensity": (
        "Heuristic",
        "At least 20% of intensity comes from peaks absent from every spectrum in at least two "
        "other collision-energy bins for the same molecule/adduct.",
    ),
    "substantial_uwpr_contaminant_only_intensity": (
        "Heuristic",
        "At least 5% of intensity matches UWPR/MaConDa contaminant ions that every DAG source "
        "returning a feasibility decision labels contaminant-only; undecidable sources abstain.",
    ),
    "duplicate_spectrum_structure_ambiguity": (
        "Heuristic",
        "The same exact-export spectrum with at least three peaks is assigned to different "
        "structures sharing one molecular formula; this may be label ambiguity among isomers.",
    ),
    "unsupported_precursor_type": (
        "Context",
        "The canonicalized precursor label has no entry in the simple monoisotopic mass-shift "
        "table, so the precursor-mass check is skipped. This does not increase review priority "
        "and does not appear in review_reasons.",
    ),
    "duplicate_spectrum_cross_molecule": (
        "Context",
        "The same exact-export spectrum with at least three peaks occurs under more than one "
        "molecule ID. The formula-conflict and structure-ambiguity flags provide the actionable "
        "classification; this union flag does not itself increase priority.",
    ),
    "prec_type_was_normalized": (
        "Context",
        "The original NIST precursor label was mapped to a known equivalent spelling before "
        "mass and polarity checks. The original label remains available in prec_type.",
    ),
    "duplicate_spectrum_same_structure": (
        "Context",
        "The same exact-export spectrum appears under multiple molecule IDs having the same "
        "formula and structure key. This may reflect duplicate identifiers and does not itself "
        "increase review priority.",
    ),
}

REPLICATE_LOW_SIMILARITY = 0.5
REPLICATE_NORMAL_SIMILARITY = 0.8
CROSS_CE_ORPHAN_FRACTION = 0.2
UWPR_CONTAMINANT_ONLY_FRACTION = 0.05
PRECURSOR_ENVELOPE_WINDOW_DA = 4.5
DAG_WRECALL_REVIEW_CUTOFF = 0.75
MAX_OTHER_CE_PDF_PANELS = 11
MAX_GROUPED_PDF_PANELS = 12
MAX_DEDICATED_COMPARISON_PANELS = 11
MAX_PDF_STRUCTURE_IMAGES = 4
DEFAULT_PDF_PRECURSOR_TYPES = [
    "[M+H]+",
    "[M-H]-",
    "[M+Na]+",
    "[M+K]+",
    "[M+Cl]-",
    "[M]+",
]


def _is_missing(value: object) -> bool:
    """Return whether a scalar value is missing."""
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(value))


def canonicalize_prec_type(prec_type: object) -> object:
    """Map known NIST spelling variants to the precursor mass-check vocabulary."""
    if not isinstance(prec_type, str):
        return prec_type
    return PRECURSOR_TYPE_ALIASES.get(prec_type, prec_type)


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


def collect_cross_ce_comparison_spectra(
    spec_df: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[int, list[dict[str, object]]]:
    """Collect other-CE spectra used to contextualize flagged orphan-intensity rows."""
    flagged_ids = set(
        candidates.loc[
            candidates["substantial_cross_ce_orphan_intensity"].astype(bool), "spec_id"
        ].astype(int)
    )
    if not flagged_ids:
        return {}

    scoped = _add_ce_columns(spec_df)
    primary_rows = scoped[scoped["spec_id"].isin(flagged_ids)].set_index("spec_id")
    comparisons: dict[int, list[dict[str, object]]] = {}
    for spec_id in flagged_ids:
        if spec_id not in primary_rows.index:
            continue
        primary = primary_rows.loc[spec_id]
        if pd.isna(primary["ce_bin"]):
            continue
        siblings = scoped[
            scoped["group_id"].eq(primary["group_id"])
            & scoped["ce_source"].eq(primary["ce_source"])
            & scoped["ce_bin"].notna()
            & scoped["ce_bin"].ne(primary["ce_bin"])
        ].sort_values(["ce_bin", "spec_id"])
        representatives = siblings.drop_duplicates("ce_bin").copy()
        n_available_bins = len(representatives)
        if n_available_bins > MAX_OTHER_CE_PDF_PANELS:
            representatives["ce_distance"] = (
                representatives["ce_bin"] - float(primary["ce_bin"])
            ).abs()
            representatives["spec_sort_id"] = pd.to_numeric(
                representatives["spec_id"], errors="raise"
            )
            representatives = representatives.nsmallest(
                MAX_OTHER_CE_PDF_PANELS,
                ["ce_distance", "ce_bin", "spec_sort_id"],
            ).sort_values(["ce_bin", "spec_sort_id"])
        comparisons[spec_id] = [
            {
                "spec_id": int(row.spec_id),
                "dset_spec_id": row.dset_spec_id,
                "ce_source": str(row.ce_source),
                "ce_value": float(row.ce_value),
                "inst_type": row.inst_type,
                "prec_mz": getattr(row, "prec_mz", math.nan),
                "peaks": row.peaks,
                "n_other_ce_bins_available": n_available_bins,
            }
            for row in representatives.itertuples(index=False)
        ]
    return comparisons


def _ordered_cross_ce_spectra(
    row: object,
    primary_peaks: Sequence[Sequence[float]],
    comparisons: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Insert the review target into its low-to-high selected-CE sequence."""
    ce_source = str(comparisons[0]["ce_source"])
    primary = {
        "is_review_target": True,
        "spec_id": int(getattr(row, "spec_id")),
        "dset_spec_id": getattr(row, "dset_spec_id"),
        "ce_source": ce_source,
        "ce_value": float(getattr(row, ce_source)),
        "prec_mz": getattr(row, "prec_mz", math.nan),
        "expected_prec_mz": getattr(row, "expected_prec_mz", math.nan),
        "peaks": primary_peaks,
        "n_other_ce_bins_shown": len(comparisons),
        "n_other_ce_bins_available": int(
            comparisons[0].get("n_other_ce_bins_available", len(comparisons))
        ),
    }
    siblings = [dict(comparison, is_review_target=False) for comparison in comparisons]
    return sorted(
        [primary, *siblings],
        key=lambda spectrum: (float(spectrum["ce_value"]), int(spectrum["spec_id"])),
    )


def collect_same_ce_replicate_spectra(
    spec_df: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[int, list[dict[str, object]]]:
    """Collect same-CE peers used to contextualize replicate-disagreement rows."""
    flagged_ids = set(
        candidates.loc[
            candidates["replicate_consensus_disagreement"].astype(bool), "spec_id"
        ].astype(int)
    )
    if not flagged_ids:
        return {}

    scoped = _add_ce_columns(spec_df)
    primary_rows = scoped[scoped["spec_id"].isin(flagged_ids)].set_index("spec_id")
    comparisons: dict[int, list[dict[str, object]]] = {}
    for spec_id in flagged_ids:
        if spec_id not in primary_rows.index:
            continue
        primary = primary_rows.loc[spec_id]
        if pd.isna(primary["ce_bin"]):
            continue
        siblings = scoped[
            scoped["group_id"].eq(primary["group_id"])
            & scoped["ce_source"].eq(primary["ce_source"])
            & scoped["ce_bin"].eq(primary["ce_bin"])
            & scoped["spec_id"].ne(spec_id)
        ].sort_values("spec_id")
        comparisons[spec_id] = [
            {
                "spec_id": int(row.spec_id),
                "dset_spec_id": row.dset_spec_id,
                "ace": row.ace,
                "nce": row.nce,
                "inst_type": row.inst_type,
                "prec_mz": getattr(row, "prec_mz", math.nan),
                "peaks": row.peaks,
            }
            for row in siblings.itertuples(index=False)
            if _valid_peak_arrays(row.peaks) is not None
        ]
    return comparisons


def collect_formula_conflict_comparison_spectra(
    spec_df: pd.DataFrame,
    review_df: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[int, list[dict[str, object]]]:
    """Collect exact-export duplicate partners having conflicting formulas."""
    flagged_ids = set(
        candidates.loc[
            candidates["duplicate_spectrum_formula_conflict"].astype(bool), "spec_id"
        ].astype(int)
    )
    if not flagged_ids:
        return {}

    peaks_by_spec_id = dict(spec_df.loc[:, ["spec_id", "peaks"]].itertuples(index=False, name=None))
    conflict_rows = review_df[
        review_df["duplicate_spectrum_formula_conflict"].astype(bool)
        & review_df["spectrum_hash"].notna()
    ]
    comparisons: dict[int, list[dict[str, object]]] = {}
    for primary in conflict_rows[conflict_rows["spec_id"].isin(flagged_ids)].itertuples(
        index=False
    ):
        primary_peaks = peaks_by_spec_id[int(primary.spec_id)]
        primary_peak_arrays = _valid_peak_arrays(primary_peaks)
        partners = conflict_rows[
            conflict_rows["spectrum_hash"].eq(primary.spectrum_hash)
            & conflict_rows["spec_id"].ne(primary.spec_id)
        ].sort_values("spec_id")
        comparisons[int(primary.spec_id)] = [
            {
                "spec_id": int(row.spec_id),
                "mol_id": int(row.mol_id),
                "dset_spec_id": row.dset_spec_id,
                "formula": row.formula,
                "prec_type": row.prec_type,
                "ace": row.ace,
                "nce": row.nce,
                "inst_type": row.inst_type,
                "prec_mz": row.prec_mz,
                "expected_prec_mz": row.expected_prec_mz,
                "peaks": peaks_by_spec_id[int(row.spec_id)],
                "cosine_similarity": _spectral_cosine(
                    primary_peak_arrays,
                    _valid_peak_arrays(peaks_by_spec_id[int(row.spec_id)]),
                    _mz_tolerance(str(primary.inst_type)),
                ),
            }
            for row in partners.itertuples(index=False)
        ]
    return comparisons


def _is_contaminant_only_consensus(labels: Iterable[str | None]) -> bool:
    """Return whether at least one source decided and none found DAG feasibility."""
    decided = [label for label in labels if label in {"dag_feasible", "contaminant_only"}]
    return bool(decided) and "dag_feasible" not in decided


def collect_uwpr_peak_annotations(
    feasibility_paths: Sequence[str | os.PathLike[str]],
    candidates: pd.DataFrame,
) -> dict[int, dict[str, object]]:
    """Collect DAG labels and contaminant formulas for annotated UWPR peaks."""
    needs_annotations = candidates["substantial_uwpr_contaminant_only_intensity"].astype(bool)
    flagged_ids = set(candidates.loc[needs_annotations, "spec_id"].astype(int))
    if not flagged_ids or not feasibility_paths:
        return {}

    labels_by_source: list[dict[tuple[int, float], str]] = []
    contaminant_formulas_by_key: dict[tuple[int, float], set[str]] = {}
    for path in feasibility_paths:
        source_labels: dict[tuple[int, float], str] = {}
        available_columns = set(pd.read_csv(path, nrows=0).columns)
        usecols = ["spec_id", "peak_mz", "feasibility"]
        if "cont_formula" in available_columns:
            usecols.append("cont_formula")
        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            chunksize=500_000,
        ):
            hits = chunk[chunk["spec_id"].isin(flagged_ids)]
            for row in hits.itertuples(index=False):
                key = (int(row.spec_id), float(row.peak_mz))
                label = str(row.feasibility)
                previous = source_labels.get(key)
                if previous is not None and previous != label:
                    raise ValueError(f"UWPR feasibility CSV has conflicting labels: {path}")
                source_labels[key] = label
                formula = getattr(row, "cont_formula", None)
                if isinstance(formula, str) and formula:
                    contaminant_formulas_by_key.setdefault(key, set()).add(formula)
        labels_by_source.append(source_labels)

    annotations: dict[int, dict[str, object]] = {}
    all_keys = set().union(*(source.keys() for source in labels_by_source))
    d3_feasible = {key for key in all_keys if labels_by_source[0].get(key) == "dag_feasible"}
    d4_additional = (
        {
            key
            for key in all_keys
            if labels_by_source[1].get(key) == "dag_feasible" and key not in d3_feasible
        }
        if len(labels_by_source) >= 2
        else set()
    )
    contaminants = {
        key
        for key in all_keys
        if _is_contaminant_only_consensus(source.get(key) for source in labels_by_source)
    }
    for label, keys in (
        ("d3_dag", d3_feasible),
        ("d4_additional_dag", d4_additional),
        ("uwpr_contaminant", contaminants),
    ):
        for spec_id, peak_mz in keys:
            annotations.setdefault(spec_id, {}).setdefault(label, []).append(peak_mz)
    for spec_id, peak_mz in contaminants:
        formulas = contaminant_formulas_by_key.get((spec_id, peak_mz), set())
        if formulas:
            annotations.setdefault(spec_id, {}).setdefault("uwpr_formula_by_mz", {})[peak_mz] = (
                "/".join(sorted(formulas))
            )
    for labels in annotations.values():
        for label in ("d3_dag", "d4_additional_dag", "uwpr_contaminant"):
            if label in labels:
                labels[label] = sorted(labels[label])
    return annotations


def collect_uwpr_contaminant_matches(
    feasibility_paths: Sequence[str | os.PathLike[str]],
    candidates: pd.DataFrame,
) -> dict[int, list[float]]:
    """Collect candidate peaks called contaminant-only by every UWPR source."""
    annotations = collect_uwpr_peak_annotations(feasibility_paths, candidates)
    matches: dict[int, list[float]] = {}
    for spec_id, labels in annotations.items():
        contaminant_mzs = labels.get("uwpr_contaminant", [])
        if contaminant_mzs:
            matches[spec_id] = contaminant_mzs
    return matches


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


def _dag_weighted_recall(
    peaks: object,
    dag_ion_mzs: np.ndarray,
    inst_type: str,
) -> float:
    """Return the fraction of observed intensity matched by candidate DAG ions."""
    peak_arrays = _valid_peak_arrays(peaks)
    if peak_arrays is None or len(dag_ion_mzs) == 0:
        return math.nan
    mzs, intensities = peak_arrays
    candidate_mzs = np.sort(np.asarray(dag_ion_mzs, dtype=float))
    insertion = np.searchsorted(candidate_mzs, mzs, side="left")
    right_idx = np.minimum(insertion, len(candidate_mzs) - 1)
    left_idx = np.maximum(insertion - 1, 0)
    nearest_error = np.minimum(
        np.abs(candidate_mzs[right_idx] - mzs),
        np.abs(candidate_mzs[left_idx] - mzs),
    )
    tolerance, tolerance_min_mz = _tolerance_for_inst_type(inst_type)
    matched = nearest_error <= tolerance * np.maximum(mzs, tolerance_min_mz)
    return float(intensities[matched].sum() / intensities.sum())


def _dag_weighted_recall_for_mol(
    mol_id: int,
    group: pd.DataFrame,
    dag_paths: dict[str, str],
    is_compressed_by_label: dict[str, bool],
) -> dict[int, dict[str, float]]:
    """Calculate every requested DAG wRecall for one molecule's spectra."""
    recalls = {int(spec_id): dict.fromkeys(dag_paths, math.nan) for spec_id in group["spec_id"]}
    for label, dag_path in dag_paths.items():
        try:
            dag = frag_utils.load_frag_d(
                mol_id,
                dag_path,
                is_compressed=is_compressed_by_label[label],
            )
        except FileNotFoundError:
            continue
        if dag is None:
            continue
        allowed_idx = _allowed_formula_idxs(dag["idx_by_h_delta"], None)
        idx_t = th.tensor(allowed_idx, dtype=th.long)
        theo_mzs = dag["formula_peak_mzs"].index_select(0, idx_t)
        prec_mask = (dag["formula_peak_probs"].index_select(0, idx_t) > 0.0).float()
        ion_mzs_by_prec_type: dict[str, np.ndarray] = {}
        for row in group.itertuples(index=False):
            prec_type = canonicalize_prec_type(row.prec_type)
            if prec_type not in PRECURSOR_TYPE_TO_MASS_DIFF:
                continue
            if prec_type not in ion_mzs_by_prec_type:
                ion_mzs_by_prec_type[prec_type] = (
                    _dag_ion_mzs(theo_mzs, prec_mask, prec_type).detach().cpu().numpy()
                )
            recalls[int(row.spec_id)][label] = _dag_weighted_recall(
                row.peaks,
                ion_mzs_by_prec_type[prec_type],
                str(row.inst_type),
            )
    return recalls


def collect_dag_weighted_recall(
    spec_df: pd.DataFrame,
    spec_ids: set[int],
    dag_paths: dict[str, str],
    n_jobs: int = data_utils.JOBLIB_N_JOBS,
) -> dict[int, dict[str, float]]:
    """Calculate per-spectrum intensity-weighted recall for supplied DAG variants."""
    scoped = spec_df[spec_df["spec_id"].isin(spec_ids)]
    if scoped.empty:
        return {}

    sample_mol_id = int(scoped["mol_id"].iloc[0])
    active_dag_paths: dict[str, str] = {}
    is_compressed_by_label: dict[str, bool] = {}
    for label, dag_path in dag_paths.items():
        try:
            is_compressed_by_label[label] = _detect_is_compressed(dag_path, sample_mol_id)
        except FileNotFoundError as error:
            log.warning("Skipping %s wRecall: %s", label, error)
            continue
        active_dag_paths[label] = dag_path
    if not active_dag_paths:
        return {}
    mol_inputs = [(int(mol_id), group) for mol_id, group in scoped.groupby("mol_id")]
    worker = functools.partial(
        _dag_weighted_recall_for_mol,
        dag_paths=active_dag_paths,
        is_compressed_by_label=is_compressed_by_label,
    )
    per_mol = data_utils.par_apply(
        mol_inputs,
        worker,
        need_unpack=True,
        n_jobs=n_jobs,
    )
    return {
        spec_id: values
        for molecule_recalls in per_mol
        for spec_id, values in molecule_recalls.items()
    }


def apply_dag_wrecall_review_gate(
    candidates: pd.DataFrame,
    dag_wrecall_by_spec_id: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """Annotate soft-only candidates adequately explained by either DAG."""
    gated = candidates.copy()
    for label in ("D3", "D4"):
        gated[f"dag_wrecall_{label.lower()}"] = gated["spec_id"].map(
            lambda spec_id, dag_label=label: dag_wrecall_by_spec_id.get(int(spec_id), {}).get(
                dag_label, math.nan
            )
        )
    gated["dag_wrecall_max"] = gated[["dag_wrecall_d3", "dag_wrecall_d4"]].max(axis=1, skipna=True)
    gateable_families = {
        "replicate_consensus",
        "cross_ce_orphans",
        "contaminant_library",
    }
    gateable_reasons = {
        "replicate_consensus_disagreement",
        "substantial_cross_ce_orphan_intensity",
        "substantial_uwpr_contaminant_only_intensity",
    }
    evidence_families = (
        gated["evidence_families"]
        .fillna("")
        .map(lambda value: set(filter(None, str(value).split(";"))))
    )
    review_reasons = (
        gated["review_reasons"]
        .fillna("")
        .map(lambda value: set(filter(None, str(value).split(";"))))
    )
    gated["dag_wrecall_gate_exempt"] = (
        gated["has_hard_evidence"].astype(bool)
        | evidence_families.map(
            lambda families: not families or not families.issubset(gateable_families)
        )
        | review_reasons.map(lambda reasons: not reasons or not reasons.issubset(gateable_reasons))
    )
    gated["dag_wrecall_gated"] = ~gated["dag_wrecall_gate_exempt"] & gated["dag_wrecall_max"].ge(
        DAG_WRECALL_REVIEW_CUTOFF
    )
    return gated


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
        log.info(
            "UWPR feasibility labels in %s:\n%s",
            path,
            hits["feasibility"].value_counts(dropna=False).to_string(),
        )

    all_matches: dict[int, list[float]] = {}
    conservative_matches: dict[int, list[float]] = {}
    disagreement_matches: dict[int, list[float]] = {}
    for key in all_match_keys:
        labels = [source.get(key) for source in labels_by_source]
        spec_id, peak_mz = key
        all_matches.setdefault(spec_id, []).append(peak_mz)
        if _is_contaminant_only_consensus(labels):
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


def analyze_spectrum(
    row: object,
    mol_meta: dict[int, tuple[float, str] | tuple[float, str, str]],
) -> dict[str, object]:
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

    molecule_values = mol_meta.get(int(row.mol_id), (math.nan, "", ""))
    if len(molecule_values) == 2:
        exact_mw, formula = molecule_values
        inchikey_s = ""
    else:
        exact_mw, formula, inchikey_s = molecule_values
    missing_molecule_metadata = not np.isfinite(exact_mw)
    canonical_prec_type = canonicalize_prec_type(row.prec_type)
    expected_prec_mz = math.nan
    precursor_error_da = math.nan
    precursor_error_ppm = math.nan
    precursor_mass_mismatch = False
    if (
        not missing_molecule_metadata
        and canonical_prec_type in PRECURSOR_TYPE_TO_MASS_DIFF
        and not _is_missing(row.prec_mz)
        and row.inst_type in PRECURSOR_MASS_CHECK_INSTRUMENTS
    ):
        expected_prec_mz = exact_mw + PRECURSOR_TYPE_TO_MASS_DIFF[canonical_prec_type]
        precursor_error_da = float(row.prec_mz - expected_prec_mz)
        precursor_error_ppm = 1.0e6 * precursor_error_da / expected_prec_mz
        tolerance, tolerance_min_mz = _tolerance_for_inst_type(row.inst_type)
        allowed_error_da = tolerance * max(expected_prec_mz, tolerance_min_mz)
        precursor_mass_mismatch = abs(precursor_error_da) > allowed_error_da

    total_intensity = (
        float(intensities.sum()) if len(intensities) and not invalid_peaks else math.nan
    )
    intensity_above_precursor_fraction = math.nan
    precursor_envelope_intensity_fraction = math.nan
    non_envelope_intensity_above_precursor_fraction = math.nan
    dominant_peak_fraction = math.nan
    n_peaks = int(len(mzs)) if not invalid_peaks else 0
    if n_peaks:
        dominant_peak_fraction = float(intensities.max() / total_intensity)
    if n_peaks and not _is_missing(row.prec_mz):
        tolerance, tolerance_min_mz = _tolerance_for_inst_type(row.inst_type)
        cutoff = float(row.prec_mz) + tolerance * max(float(row.prec_mz), tolerance_min_mz)
        above_precursor = mzs > cutoff
        precursor_envelope = above_precursor & (
            mzs <= float(row.prec_mz) + PRECURSOR_ENVELOPE_WINDOW_DA
        )
        non_envelope_above_precursor = mzs > (float(row.prec_mz) + PRECURSOR_ENVELOPE_WINDOW_DA)
        intensity_above_precursor_fraction = float(
            intensities[above_precursor].sum() / total_intensity
        )
        precursor_envelope_intensity_fraction = float(
            intensities[precursor_envelope].sum() / total_intensity
        )
        non_envelope_intensity_above_precursor_fraction = float(
            intensities[non_envelope_above_precursor].sum() / total_intensity
        )

    missing_precursor_mz = _is_missing(row.prec_mz) or not np.isfinite(float(row.prec_mz))
    flags = {
        "empty_spectrum": empty_spectrum,
        "invalid_peaks": invalid_peaks,
        "missing_molecule_metadata": missing_molecule_metadata,
        "polarity_mismatch": _polarity_mismatch(canonical_prec_type, row.ion_mode),
    }
    missing_ce, negative_ce = _collision_energy_flags(row)
    sparse_spectrum = not invalid_peaks and 1 <= n_peaks <= 2
    heuristic_flags = {
        "missing_precursor_mz": missing_precursor_mz,
        "precursor_mass_mismatch": precursor_mass_mismatch,
        "missing_collision_energy": missing_ce,
        "negative_collision_energy": negative_ce,
        "substantial_intensity_above_precursor": (
            np.isfinite(non_envelope_intensity_above_precursor_fraction)
            and non_envelope_intensity_above_precursor_fraction >= 0.05
        ),
        "single_peak_dominates": (
            n_peaks >= 3 and np.isfinite(dominant_peak_fraction) and dominant_peak_fraction >= 0.98
        ),
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
        "inchikey_s": inchikey_s,
        "prec_type": row.prec_type,
        "prec_type_canonical": canonical_prec_type,
        "prec_type_was_normalized": (
            isinstance(row.prec_type, str) and canonical_prec_type != row.prec_type
        ),
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
        "precursor_envelope_intensity_fraction": precursor_envelope_intensity_fraction,
        "non_envelope_intensity_above_precursor_fraction": (
            non_envelope_intensity_above_precursor_fraction
        ),
        "dominant_peak_fraction": dominant_peak_fraction,
        "spectrum_hash": _spectrum_hash(mzs, intensities) if not invalid_peaks else None,
        "sparse_spectrum": sparse_spectrum,
        **flags,
        **heuristic_flags,
        "unsupported_precursor_type": canonical_prec_type not in PRECURSOR_TYPE_TO_MASS_DIFF,
        "hard_reasons": ";".join(hard_reasons),
        "heuristic_reasons": ";".join(heuristic_reasons),
        "review_reasons": ";".join(reasons),
    }


def mark_cross_molecule_duplicates(review_df: pd.DataFrame) -> None:
    """Split exact-export duplicates by molecular formula and structure identity."""
    valid = review_df[review_df["spectrum_hash"].notna() & review_df["n_peaks"].ge(3)]
    group_stats = valid.groupby("spectrum_hash").agg(
        n_molecules=("mol_id", "nunique"),
        n_formulas=("formula", "nunique"),
        n_structures=("inchikey_s", "nunique"),
    )
    duplicate_stats = group_stats[group_stats["n_molecules"].gt(1)]
    suspicious_hashes = set(duplicate_stats.index)
    formula_conflict_hashes = set(duplicate_stats[duplicate_stats["n_formulas"].gt(1)].index)
    structure_ambiguity_hashes = set(
        duplicate_stats[
            duplicate_stats["n_formulas"].eq(1) & duplicate_stats["n_structures"].gt(1)
        ].index
    )
    same_structure_hashes = suspicious_hashes - formula_conflict_hashes - structure_ambiguity_hashes
    informative = review_df["n_peaks"].ge(3)
    review_df["duplicate_spectrum_cross_molecule"] = informative & review_df["spectrum_hash"].isin(
        suspicious_hashes
    )
    review_df["duplicate_spectrum_formula_conflict"] = informative & review_df[
        "spectrum_hash"
    ].isin(formula_conflict_hashes)
    review_df["duplicate_spectrum_structure_ambiguity"] = informative & review_df[
        "spectrum_hash"
    ].isin(structure_ambiguity_hashes)
    review_df["duplicate_spectrum_same_structure"] = informative & review_df["spectrum_hash"].isin(
        same_structure_hashes
    )


def finalize_review_queue(review_df: pd.DataFrame) -> pd.DataFrame:
    """Add duplicate reasons, priority, and blank human-review fields."""
    mark_cross_molecule_duplicates(review_df)
    formula_conflict = review_df["duplicate_spectrum_formula_conflict"]
    review_df.loc[formula_conflict, "hard_reasons"] = review_df.loc[
        formula_conflict, "hard_reasons"
    ].map(lambda value: ";".join(filter(None, [value, "duplicate_spectrum_formula_conflict"])))
    review_df.loc[formula_conflict, "review_reasons"] = review_df.loc[
        formula_conflict, "review_reasons"
    ].map(lambda value: ";".join(filter(None, [value, "duplicate_spectrum_formula_conflict"])))

    structure_ambiguity = review_df["duplicate_spectrum_structure_ambiguity"]
    review_df.loc[structure_ambiguity, "heuristic_reasons"] = review_df.loc[
        structure_ambiguity, "heuristic_reasons"
    ].map(lambda value: ";".join(filter(None, [value, "duplicate_spectrum_structure_ambiguity"])))
    review_df.loc[structure_ambiguity, "review_reasons"] = review_df.loc[
        structure_ambiguity, "review_reasons"
    ].map(lambda value: ";".join(filter(None, [value, "duplicate_spectrum_structure_ambiguity"])))

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
        "metadata_completeness": ["missing_precursor_mz", "missing_collision_energy"],
        "metadata_consistency": ["precursor_mass_mismatch", "negative_collision_energy"],
        "label_ambiguity": ["duplicate_spectrum_structure_ambiguity"],
        "replicate_consensus": ["replicate_consensus_disagreement"],
        "cross_ce_orphans": ["substantial_cross_ce_orphan_intensity"],
        "contaminant_library": ["substantial_uwpr_contaminant_only_intensity"],
    }
    family_enabled = pd.DataFrame(
        {family: review_df[flags].any(axis=1) for family, flags in family_flags.items()},
        index=review_df.index,
    )
    review_df["evidence_family_count"] = family_enabled.sum(axis=1)
    review_df["evidence_families"] = ""
    for family in family_enabled:
        enabled = family_enabled[family]
        review_df.loc[enabled, "evidence_families"] = review_df.loc[
            enabled, "evidence_families"
        ].map(lambda value, name=family: ";".join(filter(None, [value, name])))
    review_df["review_status"] = "unreviewed"
    review_df["hard_flag_count"] = review_df[list(HARD_FLAG_WEIGHTS)].sum(axis=1)
    review_df["has_hard_evidence"] = review_df["hard_flag_count"].gt(0)
    review_df["review_tier"] = np.select(
        [
            review_df["has_hard_evidence"],
            review_df["evidence_family_count"].ge(2),
            review_df["evidence_family_count"].eq(1),
        ],
        ["hard_conflict", "corroborated", "single_evidence_family"],
        default="heuristic_only",
    )
    review_df["reviewer"] = ""
    review_df["review_notes"] = ""
    candidates = review_df[review_df["review_priority"] > 0].copy()
    return candidates.sort_values(
        ["has_hard_evidence", "evidence_family_count", "review_priority", "spec_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _spectrum_svg(
    peaks: Sequence[Sequence[float]],
    precursor_mz: float | None = None,
    width: int = 520,
    height: int = 160,
) -> str:
    """Render a dependency-free SVG stick spectrum for the HTML review report."""
    if not peaks:
        return "<em>Empty spectrum</em>"
    peak_array = np.asarray(peaks, dtype=float)
    if peak_array.ndim != 2 or peak_array.shape[1] != 2 or not np.isfinite(peak_array).all():
        return "<em>Malformed spectrum</em>"
    max_mz = max(float(peak_array[:, 0].max()), precursor_mz or 0.0, 1.0)
    max_intensity = max(float(peak_array[:, 1].max()), 1.0)
    lines = []
    for mz, intensity in peak_array:
        x = 5 + (width - 10) * float(mz) / max_mz
        y = height - 5 - (height - 10) * max(float(intensity), 0.0) / max_intensity
        lines.append(f'<line x1="{x:.1f}" y1="{height - 5}" x2="{x:.1f}" y2="{y:.1f}"/>')
    precursor_line = ""
    if precursor_mz is not None:
        precursor_x = 5 + (width - 10) * precursor_mz / max_mz
        precursor_line = (
            f'<line x1="{precursor_x:.1f}" y1="5" x2="{precursor_x:.1f}" '
            f'y2="{height - 5}" stroke="#555" stroke-width="1" stroke-dasharray="4 3"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="mass spectrum">'
        f'<g stroke="#2457a6" stroke-width="1">{"".join(lines)}</g>'
        f"{precursor_line}"
        f'<line x1="5" y1="{height - 5}" x2="{width - 5}" y2="{height - 5}" '
        'stroke="#555"/></svg>'
    )


def write_review_html(
    candidates: pd.DataFrame,
    peaks_by_spec_id: dict[int, Sequence[Sequence[float]]],
    out_fp: str,
    n_wrecall_gated: int = 0,
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
            f"{row.uwpr_contaminant_only_intensity_fraction:.3f}"
        )
        cards.append(
            "<article><h2>Priority "
            f"{row.review_priority}: {html.escape(row.review_reasons)}</h2><p>{metadata}</p>"
            f"<p>{evidence}</p>"
            f"{_spectrum_svg(peaks_by_spec_id.get(row.spec_id, []), _pdf_precursor_mz(row))}"
            "</article>"
        )
    glossary = "".join(
        f"<dt><code>{html.escape(reason)}</code> ({level})</dt><dd>{html.escape(explanation)}</dd>"
        for reason, (level, explanation) in REVIEW_FLAG_GLOSSARY.items()
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>NIST spectrum review</title>
<style>body{{font:14px sans-serif;max-width:1100px;margin:auto}}article{{border-bottom:1px solid #bbb;
padding:12px}}h2{{font-size:16px}}dt{{font-weight:bold;margin-top:10px}}svg{{width:100%;height:180px;
background:#fafafa}}</style></head>
<body><h1>NIST suspicious-spectrum review</h1>
<p>Use the companion CSV to record keep/exclude/needs_followup decisions. Priority is a triage
score, not a bad-data probability. This HTML omits {n_wrecall_gated:,} soft spectral candidates
with D3 or D4 wRecall at or above {DAG_WRECALL_REVIEW_CUTOFF:.2f}; they remain in the CSV with
<code>dag_wrecall_gated=true</code>.</p><h2>Review flag glossary</h2><dl>{glossary}</dl>
{"".join(cards)}</body></html>"""
    os.makedirs(os.path.dirname(out_fp) or ".", exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as handle:
        handle.write(document)


def _pdf_metric(row: object, name: str) -> str:
    """Format one optional review metric for a PDF spectrum card."""
    value = getattr(row, name, math.nan)
    return f"{float(value):.3f}" if pd.notna(value) else "n/a"


def _pdf_mz_value(value: object) -> str:
    """Format one optional m/z value for a PDF label."""
    return f"{float(value):.5f}" if pd.notna(value) else "n/a"


def _pdf_mz(row: object, name: str) -> str:
    """Format one optional m/z attribute for a PDF header."""
    return _pdf_mz_value(getattr(row, name, math.nan))


def _pdf_wrecall(
    spec_id: int,
    inst_type: object,
    dag_wrecall_by_spec_id: dict[int, dict[str, float]],
) -> str:
    """Format D3 and D4 intensity-weighted recall for one spectrum."""
    if inst_type not in INSTRUMENT_TOLERANCE:
        return "wRecall D3=n/a | D4=n/a"
    values = dag_wrecall_by_spec_id.get(spec_id, {})
    d3 = values.get("D3", math.nan)
    d4 = values.get("D4", math.nan)
    d3_text = f"{d3:.3f}" if np.isfinite(d3) else "n/a"
    d4_text = f"{d4:.3f}" if np.isfinite(d4) else "n/a"
    return f"wRecall D3={d3_text} | D4={d4_text}"


def _pdf_ce_label(row: object) -> str:
    """Format available absolute and normalized collision energies."""
    labels = []
    for name, label in (("ace", "ACE"), ("nce", "NCE")):
        value = row.get(name, math.nan) if isinstance(row, dict) else getattr(row, name, math.nan)
        if pd.notna(value):
            labels.append(f"{label}={float(value):g}")
    return ", ".join(labels) if labels else "CE unavailable"


def _pdf_precursor_mz(row: object) -> float | None:
    """Return recorded precursor m/z, falling back to the computed value."""
    for name in ("prec_mz", "expected_prec_mz"):
        value = row.get(name, math.nan) if isinstance(row, dict) else getattr(row, name, math.nan)
        if pd.notna(value) and float(value) > 0.0:
            return float(value)
    return None


def _plot_pdf_precursor_line(
    axis: object,
    precursor_mz: float | None,
    color: str = "#555555",
    label: str = "prec m/z",
    linestyle: str = "--",
    label_y: float = 0.98,
) -> None:
    """Draw and label a precursor marker without obscuring fragment sticks."""
    if precursor_mz is None:
        return
    axis.axvline(
        precursor_mz,
        color=color,
        linestyle=linestyle,
        linewidth=0.9,
        alpha=0.85,
        zorder=0.5,
    )
    axis.text(
        precursor_mz,
        label_y,
        label,
        transform=axis.get_xaxis_transform(),
        color=color,
        fontsize=6,
        rotation=90,
        ha="right",
        va="top",
    )


def _pdf_shared_x_max(
    peaks_by_panel: Sequence[Sequence[Sequence[float]]],
    records: Sequence[object],
) -> float:
    """Return an x limit containing every peak and precursor marker on a page."""
    values = [
        float(peak_arrays[0].max())
        for peaks in peaks_by_panel
        if (peak_arrays := _valid_peak_arrays(peaks)) is not None
    ]
    values.extend(
        precursor_mz
        for record in records
        if (precursor_mz := _pdf_precursor_mz(record)) is not None
    )
    return max(values, default=1.0) * 1.03


def _flag_count(candidates: pd.DataFrame, column: str) -> int:
    """Count an optional boolean review flag."""
    return int(candidates[column].sum()) if column in candidates else 0


def _mol_image_array(smiles: str | None) -> np.ndarray | None:
    """Render a molecule image array from SMILES, returning None on failure."""
    if not smiles or smiles == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.asarray(Draw.MolToImage(mol, size=(360, 180), kekulize=False))


def _plot_pdf_spectrum(
    axis: object,
    peaks: Sequence[Sequence[float]],
    title: str,
    color: str,
    x_max: float | None = None,
    peak_annotations: dict[str, object] | None = None,
    precursor_mz: float | None = None,
) -> None:
    """Draw one full-width PDF spectrum axis."""
    peak_arrays = _valid_peak_arrays(peaks)
    if peak_arrays is None:
        axis.text(0.5, 0.5, "Malformed or empty spectrum", ha="center", va="center")
    else:
        mzs, intensities = peak_arrays
        normalized = intensities / intensities.max()
        axis.vlines(mzs, 0.0, normalized, color=color, linewidth=0.9)
        annotation_styles = {
            "d3_dag": ("#269653", "D3 DAG"),
            "d4_additional_dag": ("#7b4ab5", "D4+ DAG"),
            "uwpr_contaminant": ("#e07a19", "UWPR"),
        }
        formula_by_mz = (peak_annotations or {}).get("uwpr_formula_by_mz", {})
        for annotation, target_mzs in (peak_annotations or {}).items():
            if annotation == "uwpr_formula_by_mz":
                continue
            annotation_color, annotation_label = annotation_styles[annotation]
            for target_mz in target_mzs:
                matched = np.flatnonzero(np.abs(mzs - target_mz) <= 1.0e-4)
                if len(matched) == 0:
                    continue
                peak_idx = int(matched[0])
                axis.vlines(
                    mzs[peak_idx],
                    0.0,
                    normalized[peak_idx],
                    color=annotation_color,
                    linewidth=2.0,
                )
                peak_label = annotation_label
                if annotation == "uwpr_contaminant" and isinstance(formula_by_mz, dict):
                    formula = formula_by_mz.get(float(target_mz))
                    if formula:
                        peak_label = f"{annotation_label} {formula}"
                axis.text(
                    mzs[peak_idx],
                    min(float(normalized[peak_idx]) + 0.025, 1.02),
                    peak_label,
                    color=annotation_color,
                    fontsize=6,
                    rotation=90,
                    ha="center",
                    va="bottom",
                )
        axis.set_ylim(0.0, 1.08)
    if x_max is not None:
        axis.set_xlim(0.0, x_max)
    elif peak_arrays is not None:
        axis.set_xlim(0.0, max(float(peak_arrays[0].max()) * 1.03, 1.0))
    _plot_pdf_precursor_line(axis, precursor_mz)
    axis.set_title(title, fontsize=8, loc="left")
    axis.grid(axis="y", alpha=0.25, linewidth=0.4)
    axis.set_xlabel("m/z", fontsize=8)
    axis.set_ylabel("relative intensity", fontsize=8)
    axis.tick_params(labelsize=7)


def _plot_pdf_mirror_spectrum(
    axis: object,
    primary_peaks: Sequence[Sequence[float]],
    comparison_peaks: Sequence[Sequence[float]],
    title: str,
    x_max: float,
    primary_precursor_mz: float | None = None,
    comparison_precursor_mz: float | None = None,
) -> None:
    """Draw primary and comparison spectra upward and downward on one axis."""
    primary = _valid_peak_arrays(primary_peaks)
    comparison = _valid_peak_arrays(comparison_peaks)
    if primary is None or comparison is None:
        axis.text(0.5, 0.5, "Malformed or empty spectrum", ha="center", va="center")
    else:
        primary_mzs, primary_intensities = primary
        comparison_mzs, comparison_intensities = comparison
        axis.vlines(
            primary_mzs,
            0.0,
            primary_intensities / primary_intensities.max(),
            color="#2457a6",
            linewidth=0.9,
        )
        axis.vlines(
            comparison_mzs,
            0.0,
            -(comparison_intensities / comparison_intensities.max()),
            color="#8b2f2f",
            linewidth=0.9,
        )
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_xlim(0.0, x_max)
    axis.set_ylim(-1.08, 1.08)
    precursors_overlap = (
        primary_precursor_mz is not None
        and comparison_precursor_mz is not None
        and math.isclose(primary_precursor_mz, comparison_precursor_mz, abs_tol=0.01)
    )
    if precursors_overlap:
        _plot_pdf_precursor_line(
            axis,
            primary_precursor_mz,
            label="target/conflict prec m/z",
        )
    else:
        _plot_pdf_precursor_line(
            axis,
            primary_precursor_mz,
            color="#2457a6",
            label="target prec m/z",
            linestyle="--",
        )
        _plot_pdf_precursor_line(
            axis,
            comparison_precursor_mz,
            color="#8b2f2f",
            label="conflict prec m/z",
            linestyle=":",
            label_y=0.72,
        )
    axis.set_title(title, fontsize=8, loc="left")
    axis.grid(axis="y", alpha=0.25, linewidth=0.4)
    axis.set_xlabel("m/z", fontsize=8)
    axis.set_ylabel(
        "relative intensity\nreview target ↑ / conflicting assignment ↓",
        fontsize=8,
    )
    axis.tick_params(labelsize=7)


def _group_pdf_candidate_pages(
    rows: list[object],
    formula_conflicts_by_spec_id: dict[int, list[dict[str, object]]],
    same_ce_replicates_by_spec_id: dict[int, list[dict[str, object]]],
    cross_ce_comparisons_by_spec_id: dict[int, list[dict[str, object]]],
) -> list[list[object]]:
    """Pack comparison-free candidates with the same molecule/adduct/instrument."""
    comparison_spec_ids = {
        int(spec_id)
        for comparisons in (
            formula_conflicts_by_spec_id,
            same_ce_replicates_by_spec_id,
            cross_ce_comparisons_by_spec_id,
        )
        for spec_id, spectra in comparisons.items()
        if spectra
    }
    simple_groups: dict[tuple[int, str, str], list[object]] = {}
    for row in rows:
        if int(row.spec_id) in comparison_spec_ids:
            continue
        key = (int(row.mol_id), str(row.prec_type), str(row.inst_type))
        simple_groups.setdefault(key, []).append(row)
    for group in simple_groups.values():
        group.sort(key=_pdf_candidate_ce_sort_key)

    pages: list[list[object]] = []
    emitted_groups: set[tuple[int, str, str]] = set()
    for row in rows:
        spec_id = int(row.spec_id)
        if spec_id in comparison_spec_ids:
            pages.append([row])
            continue
        key = (int(row.mol_id), str(row.prec_type), str(row.inst_type))
        if key in emitted_groups:
            continue
        emitted_groups.add(key)
        group = simple_groups[key]
        pages.extend(
            group[start : start + MAX_GROUPED_PDF_PANELS]
            for start in range(0, len(group), MAX_GROUPED_PDF_PANELS)
        )
    return pages


def _pdf_candidate_ce_sort_key(row: object) -> tuple[bool, float, int]:
    """Sort grouped candidates from lower to higher preferred collision energy."""
    nce = getattr(row, "nce", math.nan)
    ace = getattr(row, "ace", math.nan)
    if str(row.inst_type) == "FT" and not _is_missing(nce):
        ce_value = float(nce)
    elif not _is_missing(ace):
        ce_value = float(ace)
    elif not _is_missing(nce):
        ce_value = float(nce)
    else:
        ce_value = math.inf
    return not math.isfinite(ce_value), ce_value, int(row.spec_id)


def _select_dedicated_comparisons(
    formula_conflicts: list[dict[str, object]],
    same_ce_replicates: list[dict[str, object]],
    cross_ce_comparisons: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Share the fixed panel budget without hiding an available evidence type."""
    sources = [formula_conflicts, same_ce_replicates, cross_ce_comparisons]
    selected: list[list[dict[str, object]]] = [[], [], []]
    while sum(map(len, selected)) < MAX_DEDICATED_COMPARISON_PANELS:
        added = False
        for source, destination in zip(sources, selected, strict=True):
            if len(destination) >= len(source):
                continue
            destination.append(source[len(destination)])
            added = True
            if sum(map(len, selected)) == MAX_DEDICATED_COMPARISON_PANELS:
                break
        if not added:
            break
    return selected[0], selected[1], selected[2]


def write_review_pdf(
    candidates: pd.DataFrame,
    peaks_by_spec_id: dict[int, Sequence[Sequence[float]]],
    out_fp: str,
    total_candidate_count: int | None = None,
    smiles_by_mol_id: dict[int, str] | None = None,
    cross_ce_comparisons_by_spec_id: dict[int, list[dict[str, object]]] | None = None,
    same_ce_replicates_by_spec_id: dict[int, list[dict[str, object]]] | None = None,
    uwpr_annotations_by_spec_id: dict[int, dict[str, object]] | None = None,
    dag_wrecall_by_spec_id: dict[int, dict[str, float]] | None = None,
    formula_conflict_comparisons_by_spec_id: dict[int, list[dict[str, object]]] | None = None,
) -> None:
    """Write candidate pages with grouped targets or dedicated comparison spectra."""
    if total_candidate_count is None:
        total_candidate_count = len(candidates)
    if smiles_by_mol_id is None:
        smiles_by_mol_id = {}
    if cross_ce_comparisons_by_spec_id is None:
        cross_ce_comparisons_by_spec_id = {}
    if same_ce_replicates_by_spec_id is None:
        same_ce_replicates_by_spec_id = {}
    if uwpr_annotations_by_spec_id is None:
        uwpr_annotations_by_spec_id = {}
    if dag_wrecall_by_spec_id is None:
        dag_wrecall_by_spec_id = {}
    if formula_conflict_comparisons_by_spec_id is None:
        formula_conflict_comparisons_by_spec_id = {}
    os.makedirs(os.path.dirname(out_fp) or ".", exist_ok=True)
    with PdfPages(out_fp) as pdf:
        # 1. Summarize the queue represented by this review document.
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("NIST suspicious-spectrum human review", fontsize=20, y=0.94)
        signal_counts = [
            ("Candidates in complete queue", total_candidate_count),
            ("Spectra included in this PDF", len(peaks_by_spec_id)),
            (
                "Within PDF: same-CE replicate disagreements",
                _flag_count(candidates, "replicate_consensus_disagreement"),
            ),
            (
                "Within PDF: substantial cross-CE orphan intensity",
                _flag_count(candidates, "substantial_cross_ce_orphan_intensity"),
            ),
            (
                "Within PDF: D3+D4 contaminant consensus",
                _flag_count(candidates, "substantial_uwpr_contaminant_only_intensity"),
            ),
        ]
        summary = "\n".join(f"{label}: {count:,}" for label, count in signal_counts)
        guidance = (
            "Priority is a triage score, not a probability of bad data. Review the molecular "
            "assignment, precursor/adduct/polarity, same-CE replicates, adjacent collision "
            "energies, and evidence fields before choosing keep, exclude, or needs_followup. "
            "Peak-level DAG feasibility is supporting context; candidates containing only "
            "replicate, cross-CE-orphan, and/or contaminant-library evidence with D3 or D4 "
            "wRecall at or above 0.75 are omitted from visual review but retained in CSV."
        )
        fig.text(0.12, 0.74, summary, fontsize=14, linespacing=1.7, va="top")
        fig.text(
            0.12,
            0.40,
            textwrap.fill(guidance, 95),
            fontsize=12,
            linespacing=1.5,
            va="top",
        )
        fig.text(
            0.12,
            0.18,
            "Thresholds: D3 or D4 wRecall ≥0.75 excluded; "
            "replicate ≤0.50 with peer consensus ≥0.80; "
            "cross-CE orphan intensity ≥20%; D3+D4 contaminant consensus intensity ≥5%.",
            fontsize=10,
        )
        pdf.savefig(fig)
        plt.close(fig)

        # 2. Define every review reason and context-only flag.
        glossary_items = list(REVIEW_FLAG_GLOSSARY.items())
        for glossary_start in range(0, len(glossary_items), 9):
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle("Review flag glossary", fontsize=18, y=0.96)
            page_items = glossary_items[glossary_start : glossary_start + 9]
            y = 0.89
            for reason, (level, explanation) in page_items:
                fig.text(0.08, y, f"{reason}  [{level}]", fontsize=10.5, weight="bold", va="top")
                fig.text(
                    0.08,
                    y - 0.027,
                    textwrap.fill(explanation, 115),
                    fontsize=9,
                    linespacing=1.25,
                    va="top",
                )
                y -= 0.095
            pdf.savefig(fig)
            plt.close(fig)

        # 3. Plot each candidate with exact-duplicate partners and other-CE siblings.
        rows = list(
            candidates[candidates["spec_id"].isin(peaks_by_spec_id)].itertuples(index=False)
        )
        candidate_pages = _group_pdf_candidate_pages(
            rows,
            formula_conflict_comparisons_by_spec_id,
            same_ce_replicates_by_spec_id,
            cross_ce_comparisons_by_spec_id,
        )
        page_number = 0
        for page_rows in candidate_pages:
            page_number += 1
            if len(page_rows) > 1:
                first_row = page_rows[0]
                mol_id = int(first_row.mol_id)
                n_plots = len(page_rows)
                ncols = 2
                nrows = math.ceil(n_plots / ncols)
                page_height = max(8.5, 2.25 * nrows + 2.4)
                fig = plt.figure(figsize=(13.5, page_height), constrained_layout=False)
                grid_top = 1.0 - (1.65 / page_height)
                grid_bottom = 0.55 / page_height
                grid = fig.add_gridspec(
                    nrows,
                    ncols,
                    left=0.045,
                    right=0.985,
                    top=grid_top,
                    bottom=grid_bottom,
                    hspace=0.9,
                    wspace=0.18,
                )
                header_bottom = grid_top + 0.01
                header_top = 1.0 - (0.1 / page_height)
                header_axis = fig.add_axes([0.055, header_bottom, 0.91, header_top - header_bottom])
                header_axis.axis("off")
                plot_axes = np.array(
                    [
                        [fig.add_subplot(grid[plot_row, col]) for col in range(ncols)]
                        for plot_row in range(nrows)
                    ]
                ).ravel()
                grouped_page_peaks = [
                    peaks_by_spec_id.get(int(grouped_row.spec_id), []) for grouped_row in page_rows
                ]
                shared_x_max = _pdf_shared_x_max(grouped_page_peaks, page_rows)
                for axis, grouped_row in zip(plot_axes, page_rows, strict=False):
                    grouped_spec_id = int(grouped_row.spec_id)
                    reason = str(grouped_row.review_reasons).replace(";", "; ")
                    title = (
                        f"REVIEW TARGET | Priority {grouped_row.review_priority} | "
                        f"spec {grouped_spec_id} | NIST {grouped_row.dset_spec_id} | "
                        f"{_pdf_ce_label(grouped_row)}\n"
                        f"{_pdf_wrecall(grouped_spec_id, grouped_row.inst_type, dag_wrecall_by_spec_id)} | "
                        f"prec m/z recorded={_pdf_mz(grouped_row, 'prec_mz')}, "
                        f"computed={_pdf_mz(grouped_row, 'expected_prec_mz')}\n"
                        f"{reason}"
                    )
                    title = "\n".join(
                        textwrap.fill(
                            line,
                            90,
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                        for line in title.splitlines()
                    )
                    _plot_pdf_spectrum(
                        axis,
                        peaks_by_spec_id.get(grouped_spec_id, []),
                        title,
                        "#2457a6",
                        shared_x_max,
                        uwpr_annotations_by_spec_id.get(grouped_spec_id, {}),
                        precursor_mz=_pdf_precursor_mz(grouped_row),
                    )
                    axis.set_facecolor("#fff6cf")
                    for spine in axis.spines.values():
                        spine.set_color("#d18b00")
                        spine.set_linewidth(1.4)
                for axis in plot_axes[n_plots:]:
                    axis.axis("off")

                metadata = (
                    f"review page {page_number} | grouped review targets={len(page_rows)} | "
                    f"mol {mol_id} | {first_row.prec_type} | {first_row.formula} | "
                    f"{first_row.inst_type}\n"
                    "same molecule + adduct + instrument; each highlighted panel is a "
                    "separate review candidate"
                )
                header_axis.text(
                    0.01,
                    0.96,
                    "\n".join(textwrap.fill(line, 95) for line in metadata.splitlines()),
                    transform=header_axis.transAxes,
                    fontsize=7.5,
                    va="top",
                )
                mol_image = _mol_image_array(smiles_by_mol_id.get(mol_id))
                if mol_image is None:
                    header_axis.text(
                        0.86,
                        0.5,
                        "Molecule structure unavailable",
                        transform=header_axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                else:
                    structure_axis = header_axis.inset_axes([0.72, 0.02, 0.26, 0.94])
                    structure_axis.imshow(mol_image)
                    structure_axis.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                continue

            row = page_rows[0]
            spec_id = int(row.spec_id)
            mol_id = int(row.mol_id)
            all_formula_conflicts = (
                formula_conflict_comparisons_by_spec_id.get(spec_id, [])
                if bool(row.duplicate_spectrum_formula_conflict)
                else []
            )
            all_same_ce_replicates = (
                same_ce_replicates_by_spec_id.get(spec_id, [])
                if bool(row.replicate_consensus_disagreement)
                else []
            )
            all_cross_ce_comparisons = (
                cross_ce_comparisons_by_spec_id.get(spec_id, [])
                if bool(row.substantial_cross_ce_orphan_intensity)
                else []
            )
            formula_conflicts, same_ce_replicates, cross_ce_comparisons = (
                _select_dedicated_comparisons(
                    all_formula_conflicts,
                    all_same_ce_replicates,
                    all_cross_ce_comparisons,
                )
            )
            comparisons = formula_conflicts + same_ce_replicates + cross_ce_comparisons
            n_plots = 1 + len(comparisons)
            ncols = 2
            nrows = math.ceil(n_plots / ncols)
            page_height = max(8.5, 2.25 * nrows + 2.4)
            fig = plt.figure(figsize=(13.5, page_height), constrained_layout=False)
            grid_top = 1.0 - (1.65 / page_height)
            grid_bottom = 0.55 / page_height
            grid = fig.add_gridspec(
                nrows,
                ncols,
                left=0.045,
                right=0.985,
                top=grid_top,
                bottom=grid_bottom,
                hspace=0.74,
                wspace=0.18,
            )
            header_bottom = grid_top + 0.01
            header_top = 1.0 - (0.1 / page_height)
            header_axis = fig.add_axes([0.055, header_bottom, 0.91, header_top - header_bottom])
            header_axis.axis("off")
            axes = np.array(
                [
                    [fig.add_subplot(grid[plot_row, col]) for col in range(ncols)]
                    for plot_row in range(nrows)
                ]
            )
            axes_flat = axes.ravel()
            plot_axes = axes.T.ravel() if cross_ce_comparisons else axes_flat
            page_peaks = [peaks_by_spec_id.get(spec_id, [])] + [
                comparison["peaks"] for comparison in comparisons
            ]
            shared_x_max = _pdf_shared_x_max(page_peaks, [row, *comparisons])
            reason = textwrap.fill(
                str(row.review_reasons).replace(";", "; "),
                width=55,
                break_long_words=False,
                break_on_hyphens=False,
            )
            primary_peaks = peaks_by_spec_id.get(spec_id, [])
            if cross_ce_comparisons:
                cross_ce_spectra = _ordered_cross_ce_spectra(
                    row, primary_peaks, cross_ce_comparisons
                )
                for axis, spectrum in zip(plot_axes, cross_ce_spectra, strict=False):
                    ce_label = (
                        f"{str(spectrum['ce_source']).upper()}=" f"{float(spectrum['ce_value']):g}"
                    )
                    if bool(spectrum["is_review_target"]):
                        _plot_pdf_spectrum(
                            axis,
                            spectrum["peaks"],
                            f"REVIEW TARGET — CROSS-CE ORPHAN | Priority "
                            f"{row.review_priority} | spec {spec_id} | {ce_label}\n"
                            f"{_pdf_wrecall(spec_id, row.inst_type, dag_wrecall_by_spec_id)} | "
                            f"CE bins shown={spectrum['n_other_ce_bins_shown']}/"
                            f"{spectrum['n_other_ce_bins_available']} | {reason}",
                            "#2457a6",
                            shared_x_max,
                            uwpr_annotations_by_spec_id.get(spec_id, {}),
                            precursor_mz=_pdf_precursor_mz(spectrum),
                        )
                        axis.set_facecolor("#fff6cf")
                        for spine in axis.spines.values():
                            spine.set_color("#d18b00")
                            spine.set_linewidth(1.8)
                    else:
                        _plot_pdf_spectrum(
                            axis,
                            spectrum["peaks"],
                            f"OTHER CE | {ce_label} | spec {spectrum['spec_id']} | "
                            f"NIST {spectrum['dset_spec_id']}\n"
                            f"{_pdf_wrecall(int(spectrum['spec_id']), row.inst_type, dag_wrecall_by_spec_id)}",
                            "#b04a35",
                            shared_x_max,
                            precursor_mz=_pdf_precursor_mz(spectrum),
                        )
                comparison_axes = iter(plot_axes[len(cross_ce_spectra) :])
            else:
                _plot_pdf_spectrum(
                    plot_axes[0],
                    primary_peaks,
                    f"REVIEW TARGET | Priority {row.review_priority} | spec {spec_id} | "
                    f"{_pdf_ce_label(row)}\n"
                    f"{_pdf_wrecall(spec_id, row.inst_type, dag_wrecall_by_spec_id)} | {reason}",
                    "#2457a6",
                    shared_x_max,
                    uwpr_annotations_by_spec_id.get(spec_id, {}),
                    precursor_mz=_pdf_precursor_mz(row),
                )
                comparison_axes = iter(plot_axes[1:])
            for comparison in formula_conflicts:
                axis = next(comparison_axes)
                _plot_pdf_mirror_spectrum(
                    axis,
                    primary_peaks,
                    comparison["peaks"],
                    "CONFLICTING ASSIGNMENT | "
                    f"spec {comparison['spec_id']} | mol {comparison['mol_id']} | "
                    f"NIST {comparison['dset_spec_id']} | {_pdf_ce_label(comparison)} | "
                    f"cos={float(comparison['cosine_similarity']):.4f}\n"
                    "EXACT DUPLICATE, DIFFERENT FORMULA | "
                    f"{comparison['formula']} | {comparison['prec_type']}\n"
                    f"{_pdf_wrecall(int(comparison['spec_id']), comparison.get('inst_type', row.inst_type), dag_wrecall_by_spec_id)} | "
                    f"prec m/z recorded={_pdf_mz_value(comparison['prec_mz'])}, "
                    f"computed={_pdf_mz_value(comparison['expected_prec_mz'])}",
                    shared_x_max,
                    primary_precursor_mz=_pdf_precursor_mz(row),
                    comparison_precursor_mz=_pdf_precursor_mz(comparison),
                )
            for comparison in same_ce_replicates:
                axis = next(comparison_axes)
                _plot_pdf_spectrum(
                    axis,
                    comparison["peaks"],
                    "SAME-CE REPLICATE | "
                    f"spec {comparison['spec_id']} | NIST {comparison['dset_spec_id']} | "
                    f"{_pdf_ce_label(comparison)}\n"
                    f"{_pdf_wrecall(int(comparison['spec_id']), comparison.get('inst_type', row.inst_type), dag_wrecall_by_spec_id)}",
                    "#6f4a8e",
                    shared_x_max,
                    precursor_mz=_pdf_precursor_mz(comparison),
                )
            for axis in plot_axes[n_plots:]:
                axis.axis("off")

            families = str(row.evidence_families) or "basic heuristic"
            metadata = (
                f"review page {page_number} | mol {mol_id} | {row.prec_type} | "
                f"NIST {row.dset_spec_id} | {row.formula} | {row.inst_type}\n"
                f"prec m/z: recorded={_pdf_mz(row, 'prec_mz')} | "
                f"computed={_pdf_mz(row, 'expected_prec_mz')} | "
                f"error={_pdf_metric(row, 'precursor_error_ppm')} ppm\n"
                f"families: {families} | "
                f"replicate best={_pdf_metric(row, 'replicate_best_similarity')} | "
                f"orphan={_pdf_metric(row, 'cross_ce_orphan_intensity_fraction')} | "
                f"contaminant={_pdf_metric(row, 'uwpr_contaminant_only_intensity_fraction')}"
            )
            if all_cross_ce_comparisons:
                metadata += (
                    "\ncross-CE display uses one representative per CE bin (lowest spec_id); "
                    f"shown={len(cross_ce_comparisons)}/{len(all_cross_ce_comparisons)} bins"
                )
            if all_formula_conflicts:
                metadata += (
                    f"\nformula-conflict panels shown={len(formula_conflicts)}/"
                    f"{len(all_formula_conflicts)}"
                )
            header_axis.text(
                0.01,
                0.96,
                "\n".join(
                    textwrap.fill(line, 65 if formula_conflicts else 95)
                    for line in metadata.splitlines()
                ),
                transform=header_axis.transAxes,
                fontsize=7.5,
                va="top",
            )
            mol_image = _mol_image_array(smiles_by_mol_id.get(mol_id))
            if formula_conflicts:
                structure_entries = [("REVIEW TARGET", mol_image)]
                structure_entries.extend(
                    (
                        f"CONFLICT spec {comparison['spec_id']}",
                        _mol_image_array(smiles_by_mol_id.get(int(comparison["mol_id"]))),
                    )
                    for comparison in formula_conflicts[: MAX_PDF_STRUCTURE_IMAGES - 1]
                )
                n_structure_entries = 1 + len(all_formula_conflicts)
                if n_structure_entries > len(structure_entries):
                    header_axis.text(
                        0.985,
                        0.02,
                        f"structures shown={len(structure_entries)}/{n_structure_entries}",
                        transform=header_axis.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=6,
                    )
                structure_width = 0.43 / len(structure_entries)
                for structure_idx, (label, image) in enumerate(structure_entries):
                    structure_axis = header_axis.inset_axes(
                        [
                            0.56 + structure_idx * structure_width,
                            0.02,
                            structure_width * 0.96,
                            0.9,
                        ]
                    )
                    if image is None:
                        structure_axis.text(
                            0.5,
                            0.5,
                            "Structure unavailable",
                            ha="center",
                            va="center",
                            fontsize=6,
                        )
                    else:
                        structure_axis.imshow(image)
                    structure_axis.set_title(label, fontsize=6, pad=1)
                    structure_axis.axis("off")
            elif mol_image is None:
                header_axis.text(
                    0.86,
                    0.5,
                    "Molecule structure unavailable",
                    transform=header_axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            else:
                structure_axis = header_axis.inset_axes([0.72, 0.02, 0.26, 0.94])
                structure_axis.imshow(mol_image)
                structure_axis.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--dsets", "--dset", nargs="+", default=None)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_html", default=None)
    parser.add_argument("--out_pdf", default=None)
    parser.add_argument(
        "--d3_frag_dp",
        default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_mbfs/dags",
        help="D3 MBFS DAG directory used for the wRecall gate and PDF labels.",
    )
    parser.add_argument(
        "--d4_frag_dp",
        default="data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse/dags",
        help="D4 DAG directory used for PDF wRecall.",
    )
    parser.add_argument(
        "--dag_n_jobs",
        type=int,
        default=data_utils.JOBLIB_N_JOBS,
        help="Parallel molecule workers for D3/D4 wRecall scoring.",
    )
    parser.add_argument(
        "--uwpr_feasibility_csv",
        nargs="+",
        default=["data/uwpr/nist20_uwpr_hits_feasibility.csv"],
        help=(
            "One or more UWPR/MaConDa feasibility files. Every source returning a decision "
            "must label a peak contaminant_only before it increases priority; undecidable "
            "sources abstain."
        ),
    )
    parser.add_argument("--max_html_spectra", type=int, default=500)
    parser.add_argument("--max_pdf_spectra", type=int, default=500)
    parser.add_argument(
        "--pdf_prec_types",
        nargs="+",
        default=DEFAULT_PDF_PRECURSOR_TYPES,
        help="Precursor types included in the PDF queue.",
    )
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

    mol_df = pd.read_pickle(args.mol_fp)[["mol_id", "exact_mw", "formula", "inchikey_s", "smiles"]]
    mol_meta = {
        int(row.mol_id): (float(row.exact_mw), str(row.formula), str(row.inchikey_s))
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
    candidate_count_before_dag_gate = len(candidates)
    log.info(
        "Computing D3/D4 wRecall review gate for %d candidates",
        candidate_count_before_dag_gate,
    )
    candidate_dag_wrecall = collect_dag_weighted_recall(
        spec_df,
        set(candidates["spec_id"].astype(int)),
        {"D3": args.d3_frag_dp, "D4": args.d4_frag_dp},
        n_jobs=args.dag_n_jobs,
    )
    candidates = apply_dag_wrecall_review_gate(candidates, candidate_dag_wrecall)
    n_wrecall_gated = int(candidates["dag_wrecall_gated"].sum())
    log.info(
        "D3/D4 wRecall gate marked %d soft-only candidates at or above %.2f; "
        "all candidates remain in the CSV",
        n_wrecall_gated,
        DAG_WRECALL_REVIEW_CUTOFF,
    )
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    candidates.to_csv(args.out_csv, index=False)
    log.info("Saved %d review candidates to %s", len(candidates), args.out_csv)

    visual_candidates = candidates[~candidates["dag_wrecall_gated"]]
    if args.out_pdf:
        pdf_queue = visual_candidates[visual_candidates["prec_type"].isin(args.pdf_prec_types)]
    if args.out_html or args.out_pdf:
        wanted_ids: set[int] = set()
        if args.out_html:
            wanted_ids.update(visual_candidates.head(args.max_html_spectra)["spec_id"])
        if args.out_pdf:
            wanted_ids.update(pdf_queue.head(args.max_pdf_spectra)["spec_id"])
        peaks_by_spec_id = {
            int(row.spec_id): row.peaks
            for row in spec_df[spec_df["spec_id"].isin(wanted_ids)][
                ["spec_id", "peaks"]
            ].itertuples(index=False)
        }
    if args.out_html:
        html_candidates = visual_candidates.head(args.max_html_spectra)
        write_review_html(
            html_candidates,
            peaks_by_spec_id,
            args.out_html,
            n_wrecall_gated=n_wrecall_gated,
        )
        log.info("Saved visual review report to %s", args.out_html)
    if args.out_pdf:
        pdf_candidates = pdf_queue.head(args.max_pdf_spectra)
        cross_ce_comparisons_by_spec_id = collect_cross_ce_comparison_spectra(
            spec_df, pdf_candidates
        )
        same_ce_replicates_by_spec_id = collect_same_ce_replicate_spectra(spec_df, pdf_candidates)
        uwpr_annotations_by_spec_id = collect_uwpr_peak_annotations(
            uwpr_paths if args.uwpr_feasibility_csv else [], pdf_candidates
        )
        formula_conflict_comparisons_by_spec_id = collect_formula_conflict_comparison_spectra(
            spec_df, review_df, pdf_candidates
        )
        pdf_plot_spec_ids = set(pdf_candidates["spec_id"].astype(int))
        for comparisons_by_spec_id in (
            cross_ce_comparisons_by_spec_id,
            same_ce_replicates_by_spec_id,
            formula_conflict_comparisons_by_spec_id,
        ):
            pdf_plot_spec_ids.update(
                int(comparison["spec_id"])
                for comparisons in comparisons_by_spec_id.values()
                for comparison in comparisons
            )
        dag_wrecall_by_spec_id = candidate_dag_wrecall.copy()
        missing_pdf_wrecall_ids = pdf_plot_spec_ids - set(dag_wrecall_by_spec_id)
        if missing_pdf_wrecall_ids:
            dag_wrecall_by_spec_id.update(
                collect_dag_weighted_recall(
                    spec_df,
                    missing_pdf_wrecall_ids,
                    {"D3": args.d3_frag_dp, "D4": args.d4_frag_dp},
                    n_jobs=args.dag_n_jobs,
                )
            )
        wanted_mol_ids = set(pdf_candidates["mol_id"])
        wanted_mol_ids.update(
            int(comparison["mol_id"])
            for comparisons in formula_conflict_comparisons_by_spec_id.values()
            for comparison in comparisons
        )
        smiles_by_mol_id = {
            int(row.mol_id): str(row.smiles)
            for row in mol_df[mol_df["mol_id"].isin(wanted_mol_ids)].itertuples(index=False)
        }
        write_review_pdf(
            pdf_candidates,
            peaks_by_spec_id,
            args.out_pdf,
            total_candidate_count=len(candidates),
            smiles_by_mol_id=smiles_by_mol_id,
            cross_ce_comparisons_by_spec_id=cross_ce_comparisons_by_spec_id,
            same_ce_replicates_by_spec_id=same_ce_replicates_by_spec_id,
            uwpr_annotations_by_spec_id=uwpr_annotations_by_spec_id,
            dag_wrecall_by_spec_id=dag_wrecall_by_spec_id,
            formula_conflict_comparisons_by_spec_id=(formula_conflict_comparisons_by_spec_id),
        )
        log.info("Saved PDF review report to %s", args.out_pdf)


if __name__ == "__main__":
    main()
