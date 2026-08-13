import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_FP = Path(__file__).parents[1] / "preproc_scripts" / "clean_spectra.py"
SPEC = importlib.util.spec_from_file_location("clean_spectra", SCRIPT_FP)
clean_spectra = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clean_spectra)


def test_merge_peaks_uses_intensity_weighted_centroid():
    peaks = [(100.0, 1.0), (100.001, 3.0), (101.0, 2.0)]

    merged = clean_spectra.merge_peaks(peaks, tolerance_ppm=20.0)

    assert merged[0] == pytest.approx((100.00075, 4.0))
    assert merged[1] == pytest.approx((101.0, 2.0))


def test_merge_peaks_does_not_use_transitive_chaining():
    peaks = [(100.0, 1.0), (100.0019, 1.0), (100.0038, 1.0)]

    merged = clean_spectra.merge_peaks(peaks, tolerance_ppm=20.0)

    assert len(merged) == 2
    assert merged[0] == pytest.approx((100.00095, 2.0))
    assert merged[1] == pytest.approx((100.0038, 1.0))


def test_merge_only_does_not_require_molecule_metadata():
    spec_df = pd.DataFrame({"peaks": [[(100.0, 1.0), (100.001, 2.0)]]})

    cleaned = clean_spectra.clean_spectra(
        spec_df,
        pd.DataFrame(),
        merge_peak_signals=True,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
    )

    assert cleaned.loc[0, "peaks"][0] == pytest.approx((100.0006666667, 3.0))


def test_filter_subformula_peaks_keeps_only_mass_matches():
    carbon_ion_mz = clean_spectra.chem_utils.formula_mass("C") - clean_spectra.chem_utils.ELECTRON_MASS
    peaks = [(carbon_ion_mz + 4e-5, 10.0), (50.0, 5.0)]

    filtered = clean_spectra.filter_subformula_peaks(
        peaks,
        formula="CH4",
        precursor_type="[M]+",
        tolerance_ppm=5.0,
    )

    assert filtered == [peaks[0]]


@pytest.mark.parametrize(
    ("precursor_type", "mass_shift"),
    [
        ("[M+H]+", clean_spectra.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]),
        ("[M+Na]+", clean_spectra.PREC_TYPE_TO_MASS_DIFF["[M+Na]+"]),
        ("[M-H]-", clean_spectra.PREC_TYPE_TO_MASS_DIFF["[M-H]-"]),
    ],
)
def test_filter_subformula_peaks_applies_adduct_mass_shift(precursor_type, mass_shift):
    ion_mz = clean_spectra.chem_utils.formula_mass("CH4") + mass_shift
    neutral_mz = clean_spectra.chem_utils.formula_mass("CH4")
    peaks = [(ion_mz, 10.0), (neutral_mz, 5.0)]

    filtered = clean_spectra.filter_subformula_peaks(
        peaks,
        formula="CH4",
        precursor_type=precursor_type,
        tolerance_ppm=5.0,
    )

    assert filtered == [peaks[0]]


def test_filter_subformula_peaks_accepts_nist_dehydration_adduct_alias():
    normalized_type = "[M-H2O+H]+"
    nist_type = "[M+H-H2O]+"
    ion_mz = (
        clean_spectra.chem_utils.formula_mass("CH4O")
        + clean_spectra.PREC_TYPE_TO_MASS_DIFF[normalized_type]
    )

    filtered = clean_spectra.filter_subformula_peaks(
        [(ion_mz, 10.0)],
        formula="CH4O",
        precursor_type=nist_type,
        tolerance_ppm=5.0,
    )

    assert filtered == [(ion_mz, 10.0)]


def test_deprotonated_subformula_requires_a_hydrogen_to_remove():
    impossible_mz = (
        clean_spectra.chem_utils.formula_mass("C")
        + clean_spectra.PREC_TYPE_TO_MASS_DIFF["[M-H]-"]
    )

    filtered = clean_spectra.filter_subformula_peaks(
        [(impossible_mz, 10.0)],
        formula="CH4",
        precursor_type="[M-H]-",
        tolerance_ppm=5.0,
    )

    assert filtered == []


def test_cleanup_options_are_independent_and_can_drop_empty_spectra():
    spec_df = pd.DataFrame(
        {
            "spec_id": [0, 1],
            "mol_id": [0, 0],
            "prec_type": ["[M]+", "[M]+"],
            "peaks": [[(100.0, 1.0), (100.001, 2.0)], [(50.0, 1.0)]],
        }
    )
    mol_df = pd.DataFrame({"mol_id": [0], "formula": ["CH4"]})

    merged_only = clean_spectra.clean_spectra(
        spec_df,
        mol_df,
        merge_peak_signals=True,
        merge_ppm=20.0,
        filter_by_subformula=False,
        subformula_ppm=5.0,
        drop_empty_spectra=False,
    )
    filtered = clean_spectra.clean_spectra(
        spec_df.iloc[[1]],
        mol_df,
        merge_peak_signals=False,
        merge_ppm=20.0,
        filter_by_subformula=True,
        subformula_ppm=5.0,
        drop_empty_spectra=True,
    )

    assert merged_only.loc[0, "peaks"][0] == pytest.approx((100.0006666667, 3.0))
    assert filtered.empty


def test_cleanup_stats_report_peak_and_intensity_retention():
    spec_df = pd.DataFrame(
        {
            "mol_id": [0, 0],
            "prec_type": ["[M]+", "[M]+"],
            "peaks": [[(12.0, 3.0), (50.0, 1.0)], [(50.0, 2.0)]],
        }
    )
    mol_df = pd.DataFrame({"mol_id": [0], "formula": ["CH4"]})

    cleaned, stats = clean_spectra.clean_spectra_with_stats(
        spec_df,
        mol_df,
        merge_peak_signals=False,
        merge_ppm=20.0,
        filter_by_subformula=True,
        subformula_ppm=100.0,
        drop_empty_spectra=True,
    )

    assert len(cleaned) == 1
    assert stats["input"]["peaks"] == 3
    assert stats["after_subformula_filter"]["peaks"] == 1
    assert stats["subformula_retention"]["aggregate_peak_fraction"] == pytest.approx(1 / 3)
    assert stats["subformula_retention"]["spectra"] == 2
    assert stats["subformula_retention"]["aggregate_intensity_fraction"] == pytest.approx(0.5)
    assert stats["subformula_retention"]["spectra_with_no_retained_peaks"] == 1
    assert stats["subformula_retention"]["spectra_with_base_peak_retained"] == 1
    assert stats["subformula_retention_by_prec_type"]["[M]+"] == stats[
        "subformula_retention"
    ]


def test_select_precursor_types_focuses_on_protonated_and_deprotonated_spectra():
    spec_df = pd.DataFrame(
        {
            "spec_id": [0, 1, 2],
            "prec_type": ["[M+H]+", "[M-H]-", "[M+Na]+"],
            "peaks": [[], [], []],
        }
    )

    selected = clean_spectra.select_precursor_types(spec_df, ["[M+H]+", "[M-H]-"])

    assert selected["spec_id"].tolist() == [0, 1]


def test_select_instrument_types_focuses_on_ft_and_qtof():
    spec_df = pd.DataFrame(
        {
            "spec_id": [0, 1, 2],
            "inst_type": ["FT", "QTOF", "IT"],
            "peaks": [[], [], []],
        }
    )

    selected = clean_spectra.select_instrument_types(spec_df, ["FT", "QTOF"])

    assert selected["spec_id"].tolist() == [0, 1]


def test_select_supported_formula_spectra_reports_exclusions():
    spec_df = pd.DataFrame({"spec_id": [0, 1], "mol_id": [0, 1], "peaks": [[], []]})
    mol_df = pd.DataFrame({"mol_id": [0, 1], "formula": ["CH4", "CH4Sn"]})

    selected, summary = clean_spectra.select_supported_formula_spectra(spec_df, mol_df)

    assert selected["spec_id"].tolist() == [0]
    assert summary == {
        "excluded_spectra": 1,
        "excluded_molecules": 1,
        "unsupported_elements": ["Sn"],
    }
