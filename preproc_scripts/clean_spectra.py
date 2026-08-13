"""Optionally merge nearby peaks and filter peaks by molecular subformulas."""

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from fragnnet.iceberg.common import chem_utils
from fragnnet.utils.formula_utils import (
    PREC_TYPE_TO_COMP_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    parse_formula,
)

logger = logging.getLogger(__name__)

PRECURSOR_TYPE_ALIASES = {
    "[M+H-H2O]+": "[M-H2O+H]+",
    "[M+H-2H2O]+": "[M-2H2O+H]+",
    "[M-H4O2+H]+": "[M-2H2O+H]+",
    "[M+H3N+H]+": "[M+NH4]+",
}

INSTRUMENT_TOLERANCES = {
    "FT": {"relative": 1.0e-5, "min_mz": 200.0},
    "QTOF": {"relative": 2.0e-5, "min_mz": 750.0},
}


def ppm_error(observed_mz: float, theoretical_mz: float) -> float:
    """Return the absolute mass error in parts per million."""
    return abs(observed_mz - theoretical_mz) / theoretical_mz * 1e6


def merge_peaks(peaks: list[tuple[float, float]], tolerance_ppm: float) -> list[tuple[float, float]]:
    """Merge peaks against an intensity-weighted cluster centroid."""
    if tolerance_ppm <= 0:
        raise ValueError("tolerance_ppm must be positive")
    if not peaks:
        return []

    sorted_peaks = sorted((float(mz), float(intensity)) for mz, intensity in peaks)
    for mz, intensity in sorted_peaks:
        if not np.isfinite(mz) or mz <= 0:
            raise ValueError(f"Peak m/z must be finite and positive, got {mz}")
        if not np.isfinite(intensity) or intensity < 0:
            raise ValueError(f"Peak intensity must be finite and nonnegative, got {intensity}")

    merged = []
    cluster = [sorted_peaks[0]]
    for peak in sorted_peaks[1:]:
        centroid = _cluster_centroid(cluster)
        if ppm_error(peak[0], centroid) <= tolerance_ppm:
            cluster.append(peak)
        else:
            merged.append((_cluster_centroid(cluster), sum(item[1] for item in cluster)))
            cluster = [peak]
    merged.append((_cluster_centroid(cluster), sum(item[1] for item in cluster)))
    return merged


def _cluster_centroid(cluster: list[tuple[float, float]]) -> float:
    total_intensity = sum(intensity for _, intensity in cluster)
    if total_intensity == 0:
        return float(np.mean([mz for mz, _ in cluster]))
    return sum(mz * intensity for mz, intensity in cluster) / total_intensity


@lru_cache(maxsize=2)
def _subformula_masses(formula: str, precursor_type: str) -> np.ndarray:
    normalized_precursor_type = PRECURSOR_TYPE_ALIASES.get(precursor_type, precursor_type)
    if normalized_precursor_type not in PREC_TYPE_TO_MASS_DIFF:
        supported_types = sorted(set(PREC_TYPE_TO_MASS_DIFF) | set(PRECURSOR_TYPE_ALIASES))
        raise ValueError(
            f"Unsupported precursor type {precursor_type!r}; supported types are "
            f"{supported_types}"
        )

    element_counts = parse_formula(formula)
    unsupported_elements = set(element_counts) - set(chem_utils.VALID_ELEMENTS)
    if unsupported_elements:
        raise ValueError(f"Unsupported formula elements: {sorted(unsupported_elements)}")

    required_counts = {
        element: -count_change
        for element, count_change in PREC_TYPE_TO_COMP_DIFF[normalized_precursor_type].items()
        if count_change < 0
    }
    if any(element_counts.get(element, 0) < count for element, count in required_counts.items()):
        return np.array([], dtype=float)
    neutral_masses = np.array([0.0])
    rdbe_twice = np.array([2.0])
    rdbe_coefficients = {"C": 2, "N": 1, "P": 1, "H": -1, "Cl": -1, "Br": -1, "I": -1, "F": -1}
    for element, max_count in element_counts.items():
        min_count = required_counts.get(element, 0)
        if min_count > max_count:
            return np.array([], dtype=float)
        levels = np.arange(min_count, max_count + 1)
        neutral_masses = (
            neutral_masses[:, None] + levels * chem_utils.ELEMENT_TO_MASS[element]
        ).ravel()
        rdbe_twice = (
            rdbe_twice[:, None] + levels * rdbe_coefficients.get(element, 0)
        ).ravel()

    neutral_masses = neutral_masses[(neutral_masses > 0) & (rdbe_twice >= 0)]
    adduct_mass_shift = PREC_TYPE_TO_MASS_DIFF[normalized_precursor_type]
    ion_masses = neutral_masses + adduct_mass_shift
    return np.sort(ion_masses[ion_masses > 0])


def filter_subformula_peaks(
    peaks: list[tuple[float, float]],
    formula: str,
    precursor_type: str,
    tolerance_ppm: float,
    tolerance_min_mz: float = 0.0,
) -> list[tuple[float, float]]:
    """Keep peaks matching a precursor-bounded, nonnegative-RDBE subformula."""
    if tolerance_ppm <= 0:
        raise ValueError("tolerance_ppm must be positive")

    candidate_masses = _subformula_masses(formula, precursor_type)
    if candidate_masses.size == 0:
        return []

    filtered = []
    for peak in peaks:
        mz = float(peak[0])
        insertion_idx = int(np.searchsorted(candidate_masses, mz))
        candidate_idxs = [insertion_idx - 1, insertion_idx]
        if any(
            0 <= idx < len(candidate_masses)
            and abs(mz - float(candidate_masses[idx]))
            <= tolerance_ppm * 1e-6 * max(mz, tolerance_min_mz)
            for idx in candidate_idxs
        ):
            filtered.append((float(peak[0]), float(peak[1])))
    return filtered


def summarize_spectra(spec_df: pd.DataFrame) -> dict[str, int | float]:
    """Summarize spectrum and peak counts for a processing stage."""
    peak_counts = spec_df["peaks"].apply(len).to_numpy()
    total_intensity = float(
        sum(sum(float(intensity) for _, intensity in peaks) for peaks in spec_df["peaks"])
    )
    return {
        "spectra": int(len(spec_df)),
        "nonempty_spectra": int((peak_counts > 0).sum()),
        "empty_spectra": int((peak_counts == 0).sum()),
        "peaks": int(peak_counts.sum()),
        "mean_peaks_per_spectrum": float(peak_counts.mean()) if len(peak_counts) else 0.0,
        "median_peaks_per_spectrum": float(np.median(peak_counts)) if len(peak_counts) else 0.0,
        "total_intensity": total_intensity,
    }


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_retention(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> dict[str, int | float]:
    """Summarize per-spectrum peak and intensity retention between aligned tables."""
    if len(before_df) != len(after_df):
        raise ValueError("Retention statistics require aligned tables with equal row counts")

    peak_fractions = []
    intensity_fractions = []
    spectra_with_no_retained_peaks = 0
    spectra_with_base_peak_retained = 0
    for before_peaks, after_peaks in zip(before_df["peaks"], after_df["peaks"], strict=True):
        if not after_peaks:
            spectra_with_no_retained_peaks += 1
        if before_peaks:
            peak_fractions.append(len(after_peaks) / len(before_peaks))
            before_base_intensity = max(float(intensity) for _, intensity in before_peaks)
            after_base_intensity = max(
                (float(intensity) for _, intensity in after_peaks), default=-np.inf
            )
            spectra_with_base_peak_retained += after_base_intensity >= before_base_intensity

        before_intensity = sum(float(intensity) for _, intensity in before_peaks)
        after_intensity = sum(float(intensity) for _, intensity in after_peaks)
        if before_intensity > 0:
            intensity_fractions.append(after_intensity / before_intensity)

    before_stats = summarize_spectra(before_df)
    after_stats = summarize_spectra(after_df)
    return {
        "spectra": int(len(before_df)),
        "aggregate_peak_fraction": _safe_fraction(after_stats["peaks"], before_stats["peaks"]),
        "aggregate_intensity_fraction": _safe_fraction(
            after_stats["total_intensity"], before_stats["total_intensity"]
        ),
        "mean_per_spectrum_peak_fraction": float(np.mean(peak_fractions))
        if peak_fractions
        else 0.0,
        "median_per_spectrum_peak_fraction": float(np.median(peak_fractions))
        if peak_fractions
        else 0.0,
        "mean_per_spectrum_intensity_fraction": float(np.mean(intensity_fractions))
        if intensity_fractions
        else 0.0,
        "median_per_spectrum_intensity_fraction": float(np.median(intensity_fractions))
        if intensity_fractions
        else 0.0,
        "spectra_with_no_retained_peaks": spectra_with_no_retained_peaks,
        "spectra_with_base_peak_retained": spectra_with_base_peak_retained,
    }


def summarize_retention_by_column(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    column: str,
) -> dict[str, dict[str, int | float]]:
    """Calculate aligned retention statistics separately for each group."""
    summaries = {}
    for value in sorted(before_df[column].dropna().unique()):
        mask = before_df[column] == value
        summaries[str(value)] = summarize_retention(before_df[mask], after_df[mask])
    return summaries


def summarize_retention_by_columns(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, int | float]]:
    """Calculate retention for intersections such as precursor type × instrument."""
    summaries = {}
    for values, indices in before_df.groupby(columns, dropna=False, sort=True).groups.items():
        values = values if isinstance(values, tuple) else (values,)
        group_name = " | ".join(f"{column}={value}" for column, value in zip(columns, values))
        summaries[group_name] = summarize_retention(
            before_df.loc[indices], after_df.loc[indices]
        )
    return summaries


def select_precursor_types(
    spec_df: pd.DataFrame,
    precursor_types: list[str] | None,
) -> pd.DataFrame:
    """Select requested precursor types without changing spectrum identifiers."""
    if precursor_types is None:
        return spec_df
    if "prec_type" not in spec_df:
        raise ValueError("spec_df must contain prec_type when --precursor-types is used")
    selected_df = spec_df[spec_df["prec_type"].isin(precursor_types)].copy()
    if selected_df.empty:
        raise ValueError(f"No spectra found for precursor types: {precursor_types}")
    return selected_df.reset_index(drop=True)


def select_instrument_types(
    spec_df: pd.DataFrame,
    instrument_types: list[str] | None,
) -> pd.DataFrame:
    """Select requested instrument types without changing spectrum identifiers."""
    if instrument_types is None:
        return spec_df
    if "inst_type" not in spec_df:
        raise ValueError("spec_df must contain inst_type when --instrument-types is used")
    selected_df = spec_df[spec_df["inst_type"].isin(instrument_types)].copy()
    if selected_df.empty:
        raise ValueError(f"No spectra found for instrument types: {instrument_types}")
    return selected_df.reset_index(drop=True)


def select_dsets(
    spec_df: pd.DataFrame,
    dsets: list[str] | None,
) -> pd.DataFrame:
    """Select requested source datasets without changing spectrum identifiers."""
    if dsets is None:
        return spec_df
    if "dset" not in spec_df:
        raise ValueError("spec_df must contain dset when --dsets is used")
    selected_df = spec_df[spec_df["dset"].isin(dsets)].copy()
    if selected_df.empty:
        raise ValueError(f"No spectra found for datasets: {dsets}")
    return selected_df.reset_index(drop=True)


def select_supported_formula_spectra(
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | list[str]]]:
    """Remove spectra whose molecular formulas contain unsupported RDBE elements."""
    formula_by_mol_id = mol_df.set_index("mol_id")["formula"]
    relevant_formulas = formula_by_mol_id[formula_by_mol_id.index.isin(spec_df["mol_id"])]
    unsupported_by_mol_id = relevant_formulas.map(
        lambda formula: sorted(set(parse_formula(formula)) - set(chem_utils.VALID_ELEMENTS))
    )
    unsupported_by_mol_id = unsupported_by_mol_id[unsupported_by_mol_id.apply(bool)]
    unsupported_mask = spec_df["mol_id"].isin(unsupported_by_mol_id.index)
    summary = {
        "excluded_spectra": int(unsupported_mask.sum()),
        "excluded_molecules": int(unsupported_by_mol_id.size),
        "unsupported_elements": sorted(
            {element for elements in unsupported_by_mol_id for element in elements}
        ),
    }
    return spec_df[~unsupported_mask].reset_index(drop=True), summary


def clean_spectra(
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    merge_peak_signals: bool,
    merge_ppm: float,
    filter_by_subformula: bool,
    subformula_ppm: float,
    drop_empty_spectra: bool,
    use_instrument_tolerance: bool = False,
) -> pd.DataFrame:
    """Apply the selected per-spectrum cleanup operations."""
    cleaned_df, _ = clean_spectra_with_stats(
        spec_df,
        mol_df,
        merge_peak_signals,
        merge_ppm,
        filter_by_subformula,
        subformula_ppm,
        drop_empty_spectra,
        use_instrument_tolerance,
    )
    return cleaned_df


def clean_spectra_with_stats(
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    merge_peak_signals: bool,
    merge_ppm: float,
    filter_by_subformula: bool,
    subformula_ppm: float,
    drop_empty_spectra: bool,
    use_instrument_tolerance: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Apply cleanup and return stage-level and retention statistics."""
    required_spec_columns = {"peaks"}
    if filter_by_subformula:
        required_spec_columns.update({"mol_id", "prec_type"})
        if use_instrument_tolerance:
            required_spec_columns.add("inst_type")
    missing_spec_columns = required_spec_columns - set(spec_df.columns)
    if missing_spec_columns:
        raise ValueError(f"spec_df is missing required columns: {sorted(missing_spec_columns)}")
    if filter_by_subformula and not {"mol_id", "formula"}.issubset(mol_df.columns):
        raise ValueError("mol_df must contain mol_id and formula columns for subformula filtering")

    stats = {
        "parameters": {
            "merge_peaks": merge_peak_signals,
            "merge_ppm": merge_ppm,
            "filter_subformula_peaks": filter_by_subformula,
            "subformula_ppm": subformula_ppm,
            "subformula_instrument_tolerance": use_instrument_tolerance,
            "drop_empty_spectra": drop_empty_spectra,
        },
        "input": summarize_spectra(spec_df),
    }

    # 1. Attach molecular formulas only when subformula filtering is requested.
    result = spec_df.copy()
    if filter_by_subformula:
        formula_by_mol_id = mol_df.set_index("mol_id")["formula"]
        if not result["mol_id"].isin(formula_by_mol_id.index).all():
            missing_ids = sorted(set(result["mol_id"]) - set(formula_by_mol_id.index))
            raise ValueError(f"No molecular formula found for mol_id values: {missing_ids[:10]}")
        formulas = result["mol_id"].map(formula_by_mol_id)

    # 2. Merge peaks within the requested centroid-based ppm tolerance.
    if merge_peak_signals:
        before_merge = result.copy()
        result.loc[:, "peaks"] = result["peaks"].apply(merge_peaks, tolerance_ppm=merge_ppm)
        stats["after_merge"] = summarize_spectra(result)
        stats["merge_retention"] = summarize_retention(before_merge, result)
        if "prec_type" in result:
            stats["merge_retention_by_prec_type"] = summarize_retention_by_column(
                before_merge, result, "prec_type"
            )
        if "inst_type" in result:
            stats["merge_retention_by_inst_type"] = summarize_retention_by_column(
                before_merge, result, "inst_type"
            )
            if "prec_type" in result:
                stats["merge_retention_by_prec_type_and_inst_type"] = (
                    summarize_retention_by_columns(before_merge, result, ["prec_type", "inst_type"])
                )

    # 3. Retain only peaks explained by a feasible molecular subformula.
    if filter_by_subformula:
        before_subformula_filter = result.copy()
        if use_instrument_tolerance:
            unsupported_instruments = set(result["inst_type"]) - set(INSTRUMENT_TOLERANCES)
            if unsupported_instruments:
                raise ValueError(
                    "No subformula instrument tolerance for: "
                    f"{sorted(unsupported_instruments)}"
                )
            filtered_peaks = (
                filter_subformula_peaks(
                    peaks,
                    formula,
                    precursor_type,
                    INSTRUMENT_TOLERANCES[instrument_type]["relative"] * 1e6,
                    INSTRUMENT_TOLERANCES[instrument_type]["min_mz"],
                )
                for peaks, formula, precursor_type, instrument_type in zip(
                    result["peaks"],
                    formulas,
                    result["prec_type"],
                    result["inst_type"],
                    strict=True,
                )
            )
        else:
            filtered_peaks = (
                filter_subformula_peaks(peaks, formula, precursor_type, subformula_ppm)
                for peaks, formula, precursor_type in zip(
                    result["peaks"], formulas, result["prec_type"], strict=True
                )
            )
        result.loc[:, "peaks"] = pd.Series(filtered_peaks, index=result.index, dtype=object)
        stats["after_subformula_filter"] = summarize_spectra(result)
        stats["subformula_retention"] = summarize_retention(before_subformula_filter, result)
        stats["subformula_retention_by_prec_type"] = summarize_retention_by_column(
            before_subformula_filter, result, "prec_type"
        )
        if "inst_type" in result:
            stats["subformula_retention_by_inst_type"] = summarize_retention_by_column(
                before_subformula_filter, result, "inst_type"
            )
            stats["subformula_retention_by_prec_type_and_inst_type"] = (
                summarize_retention_by_columns(
                    before_subformula_filter, result, ["prec_type", "inst_type"]
                )
            )

    if drop_empty_spectra:
        result = result[result["peaks"].apply(len) > 0].copy()
    result = result.reset_index(drop=True)
    stats["output"] = summarize_spectra(result)
    return result, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-fp", type=Path, required=True, help="Input spec_df.pkl")
    parser.add_argument(
        "--mol-fp",
        type=Path,
        help="Input mol_df.pkl; required with --filter-subformula-peaks",
    )
    parser.add_argument("--output-fp", type=Path, help="Output pickle path")
    parser.add_argument(
        "--stats-fp",
        type=Path,
        help="JSON statistics path; defaults to <output-fp stem>.stats.json",
    )
    parser.add_argument(
        "--precursor-types",
        nargs="+",
        help='Only process these precursor types, e.g. "[M+H]+" "[M-H]-"',
    )
    parser.add_argument(
        "--instrument-types",
        nargs="+",
        help="Only process these instrument types, e.g. FT QTOF",
    )
    parser.add_argument(
        "--dsets",
        nargs="+",
        help="Only process these source datasets, e.g. nist20_hr",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Calculate statistics without writing a cleaned spectrum pickle",
    )
    parser.add_argument(
        "--skip-unsupported-formulas",
        action="store_true",
        help="Exclude and report formulas containing elements unsupported by the RDBE table",
    )
    parser.add_argument(
        "--merge-peaks",
        action="store_true",
        help="Merge peaks using an intensity-weighted centroid",
    )
    parser.add_argument("--merge-ppm", type=float, default=20.0)
    parser.add_argument(
        "--filter-subformula-peaks",
        action="store_true",
        help="Keep only peaks matching precursor-compatible molecular subformulas",
    )
    parser.add_argument("--subformula-ppm", type=float, default=5.0)
    parser.add_argument(
        "--subformula-inst-tol",
        action="store_true",
        help="Use FraGNNet FT/QTOF tolerance and m/z floors instead of --subformula-ppm",
    )
    parser.add_argument(
        "--drop-empty-spectra",
        action="store_true",
        help="Drop spectra with no peaks after the selected operations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.merge_peaks and not args.filter_subformula_peaks:
        raise ValueError("Enable --merge-peaks, --filter-subformula-peaks, or both")
    if args.filter_subformula_peaks and args.mol_fp is None:
        raise ValueError("--mol-fp is required with --filter-subformula-peaks")
    if not args.stats_only and args.output_fp is None:
        raise ValueError("--output-fp is required unless --stats-only is used")
    if args.stats_only and args.stats_fp is None:
        raise ValueError("--stats-fp is required with --stats-only")
    if args.output_fp is not None and args.output_fp.resolve() == args.spec_fp.resolve():
        raise ValueError("--output-fp must differ from --spec-fp")

    source_spec_df = pd.read_pickle(args.spec_fp)
    spec_df = select_dsets(source_spec_df, args.dsets)
    spec_df = select_precursor_types(spec_df, args.precursor_types)
    spec_df = select_instrument_types(spec_df, args.instrument_types)
    mol_df = pd.read_pickle(args.mol_fp) if args.mol_fp is not None else pd.DataFrame()
    selected_spectra = len(spec_df)
    unsupported_summary = None
    if args.filter_subformula_peaks and args.skip_unsupported_formulas:
        spec_df, unsupported_summary = select_supported_formula_spectra(spec_df, mol_df)
    cleaned_df, stats = clean_spectra_with_stats(
        spec_df=spec_df,
        mol_df=mol_df,
        merge_peak_signals=args.merge_peaks,
        merge_ppm=args.merge_ppm,
        filter_by_subformula=args.filter_subformula_peaks,
        subformula_ppm=args.subformula_ppm,
        drop_empty_spectra=args.drop_empty_spectra,
        use_instrument_tolerance=args.subformula_inst_tol,
    )
    stats["selection"] = {
        "precursor_types": args.precursor_types,
        "instrument_types": args.instrument_types,
        "dsets": args.dsets,
        "source_spectra": len(source_spec_df),
        "selected_spectra": selected_spectra,
        "processed_spectra": len(spec_df),
        "unsupported_formula_exclusions": unsupported_summary,
    }

    if not args.stats_only:
        args.output_fp.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_pickle(args.output_fp)
        logger.info("Saved %d spectra to %s", len(cleaned_df), args.output_fp)
    stats_fp = args.stats_fp or args.output_fp.with_suffix(".stats.json")
    stats_fp.parent.mkdir(parents=True, exist_ok=True)
    stats_fp.write_text(json.dumps(stats, indent=2) + "\n")
    logger.info("Saved cleanup statistics to %s", stats_fp)


if __name__ == "__main__":
    main()
