"""Command-line entry point for spectrum cleaning: select spectra, then clean them with
:mod:`fragnnet.utils.clean_spectra_utils` (the argument parsing, the spectrum selection and
``main`` moved here from that module).

Usage: ``python preproc_scripts/clean_spectra.py --help``
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from fragnnet.utils.clean_spectra_utils import MERGE_METHODS, clean_spectra_with_stats, logger
from fragnnet.utils.data_utils import VALID_ELEMENTS
from fragnnet.utils.formula_utils import parse_formula


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
        lambda formula: sorted(set(parse_formula(formula)) - set(VALID_ELEMENTS))
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
        help=(
            'Only process these precursor types, e.g. "[M+H]+" "[M-H]-" '
            '"[M+Na]+" "[M+Cl]-"'
        ),
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
        "--merge-inst-tol",
        action="store_true",
        help="Merge peaks using FraGNNet FT/QTOF tolerances instead of --merge-ppm",
    )
    parser.add_argument(
        "--merge-method",
        choices=MERGE_METHODS,
        default=None,
        help=(
            "Peak grouping rule. Defaults to naive_centroid with --merge-ppm and "
            "common_overlap "
            "with --merge-inst-tol"
        ),
    )
    parser.add_argument(
        "--filter-subformula-peaks",
        action="store_true",
        help="Keep only peaks matching precursor-compatible molecular subformulas",
    )
    parser.add_argument("--subformula-ppm", type=float, default=5.0)
    parser.add_argument(
        "--subformula-da",
        type=float,
        default=None,
        help="Fixed absolute subformula matching tolerance in Da; overrides --subformula-ppm",
    )
    parser.add_argument(
        "--subformula-inst-tol",
        action="store_true",
        help="Use FraGNNet FT/QTOF tolerance and m/z floors instead of --subformula-ppm",
    )
    parser.add_argument(
        "--filter-above-precursor-peaks",
        action="store_true",
        help="Drop peaks above the precursor m/z (also removes the precursor isotopologue ladder)",
    )
    parser.add_argument("--above-precursor-ppm", type=float, default=10.0)
    parser.add_argument(
        "--above-precursor-inst-tol",
        action="store_true",
        help="Use FraGNNet FT/QTOF tolerance and m/z floors instead of --above-precursor-ppm",
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

    if (
        not args.merge_peaks
        and not args.filter_subformula_peaks
        and not args.filter_above_precursor_peaks
    ):
        raise ValueError(
            "Enable at least one of --merge-peaks, --filter-subformula-peaks, "
            "--filter-above-precursor-peaks"
        )
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
        subformula_tolerance_da=args.subformula_da,
        drop_empty_spectra=args.drop_empty_spectra,
        use_instrument_tolerance=args.subformula_inst_tol,
        use_merge_instrument_tolerance=args.merge_inst_tol,
        merge_method=args.merge_method,
        filter_above_precursor=args.filter_above_precursor_peaks,
        above_precursor_ppm=args.above_precursor_ppm,
        above_precursor_use_instrument_tolerance=args.above_precursor_inst_tol,
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
