"""Focused tests for the NIST data-quality evaluation utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch as th

EVAL_SCRIPTS_DP = Path(__file__).parents[1] / "scripts" / "eval"
sys.path.insert(0, str(EVAL_SCRIPTS_DP))

import dag_oracle_scores  # noqa: E402
from dag_oracle_scores import (  # noqa: E402
    _allowed_formula_idxs,
    _detect_is_compressed,
    _load_uwpr_maps,
)
from nist_bad_data_review import (  # noqa: E402
    PRECURSOR_TYPE_TO_MASS_DIFF,
    analyze_spectrum,
    finalize_review_queue,
)


def test_detect_is_compressed_falls_back_to_another_molecule(tmp_path: Path) -> None:
    (tmp_path / "00000002.pkl.bz2").touch()

    assert _detect_is_compressed(str(tmp_path), sample_mol_id=1)


def test_load_uwpr_maps_separates_aggressive_and_conservative_matches(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "uwpr.csv"
    pd.DataFrame(
        {
            "spec_id": [1, 1, 2, 3],
            "peak_mz": [10.0, 20.0, 30.0, 40.0],
            "feasibility": [
                "contaminant_only",
                "dag_feasible",
                "no_dag_file",
                "unsupported_prec",
            ],
        }
    ).to_csv(csv_path, index=False)

    all_matches, conservative_matches = _load_uwpr_maps(str(csv_path))

    assert all_matches == {1: [10.0, 20.0], 2: [30.0], 3: [40.0]}
    assert conservative_matches == {1: [10.0]}


def test_load_uwpr_maps_rejects_unknown_label(tmp_path: Path) -> None:
    csv_path = tmp_path / "uwpr.csv"
    pd.DataFrame({"spec_id": [1], "peak_mz": [10.0], "feasibility": ["certainly_bad"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="Unknown UWPR feasibility labels"):
        _load_uwpr_maps(str(csv_path))


def test_uwpr_tiers_keep_preclassified_dag_feasible_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    dag = {
        "formula_peak_mzs": th.tensor([[50.0]]),
        "formula_peak_probs": th.tensor([[1.0]]),
        "idx_by_h_delta": [{0}],
        "dag_num_nodes": 1,
    }
    monkeypatch.setattr(dag_oracle_scores.frag_utils, "load_frag_d", lambda *args, **kwargs: dag)
    monkeypatch.setattr(
        dag_oracle_scores,
        "_score_all_tolerances",
        lambda true_mzs, *args, **kwargs: {"n_true_peaks": len(true_mzs)},
    )
    rows = [
        {
            "spec_id": 1,
            "peaks": [(10.0, 1.0), (20.0, 1.0)],
            "prec_mz": 100.0,
            "prec_type": "[M+H]+",
            "inst_type": "FT",
            "frag_mode": "HCD",
            "contam_mzs": [10.0, 20.0],
            "contaminant_only_mzs": [10.0],
        }
    ]

    result = dag_oracle_scores._process_mol_group(
        1,
        rows,
        frag_dp="unused",
        is_compressed=True,
        remove_prec_peak=False,
        max_h_transfer=None,
    )[0]

    assert result["n_true_peaks__0_raw"] == 2
    assert result["n_true_peaks__2_uwpr_rm"] == 0
    assert result["n_true_peaks__3_uwpr_keepexpl"] == 1


def test_allowed_formula_idxs_validates_requested_h_transfer() -> None:
    idx_by_h_delta = [{0}, {1}, {2}]

    assert _allowed_formula_idxs(idx_by_h_delta, max_h_transfer=1) == [0, 1, 2]
    with pytest.raises(ValueError, match="outside the DAG-supported range"):
        _allowed_formula_idxs(idx_by_h_delta, max_h_transfer=2)


def _review_row(**updates: object) -> SimpleNamespace:
    values = {
        "spec_id": 1,
        "mol_id": 10,
        "dset_spec_id": "nist-1",
        "prec_type": "[M+H]+",
        "ion_mode": "P",
        "inst_type": "FT",
        "frag_mode": "HCD",
        "ace": 10.0,
        "nce": None,
        "ace_extra_1": None,
        "nce_extra_1": None,
        "ace_extra_2": None,
        "nce_extra_2": None,
        "prec_mz": 101.007276,
        "peaks": [(20.0, 1.0), (40.0, 1.0), (60.0, 1.0)],
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_review_detector_flags_independent_consistency_errors() -> None:
    row = _review_row(prec_mz=120.0, ion_mode="N")

    result = analyze_spectrum(row, {10: (100.0, "C6H12O")})

    assert result["precursor_mass_mismatch"]
    assert result["polarity_mismatch"]
    assert not result["invalid_peaks"]
    assert "precursor_mass_mismatch" in result["hard_reasons"]


def test_review_detector_does_not_flag_consistent_spectrum() -> None:
    result = analyze_spectrum(_review_row(), {10: (100.0, "C6H12O")})
    candidates = finalize_review_queue(pd.DataFrame([result]))

    assert candidates.empty


@pytest.mark.parametrize(
    "prec_type",
    ["[M+H-NH3]+", "[M-H-CO2]-"],
)
def test_precursor_sanity_uses_precursor_specific_special_adduct_masses(prec_type: str) -> None:
    prec_mz = 100.0 + PRECURSOR_TYPE_TO_MASS_DIFF[prec_type]
    ion_mode = "P" if prec_type.endswith("+") else "N"
    row = _review_row(prec_type=prec_type, ion_mode=ion_mode, prec_mz=prec_mz)

    result = analyze_spectrum(row, {10: (100.0, "C6H12O")})

    assert not result["precursor_mass_mismatch"]


@pytest.mark.parametrize("prec_type", ["[M+H+2i]+", "[M-H+2i]-"])
def test_formula_dependent_isotope_precursors_are_not_mass_checked(prec_type: str) -> None:
    ion_mode = "P" if prec_type.endswith("+") else "N"
    result = analyze_spectrum(
        _review_row(prec_type=prec_type, ion_mode=ion_mode),
        {10: (100.0, "C6H12O")},
    )

    assert not result["precursor_mass_mismatch"]
    assert result["unsupported_precursor_type"]


def test_review_queue_flags_identical_spectrum_across_molecules() -> None:
    first = analyze_spectrum(_review_row(), {10: (100.0, "C6H12O"), 11: (100.0, "C6H12O")})
    second = analyze_spectrum(
        _review_row(spec_id=2, mol_id=11, dset_spec_id="nist-2"),
        {10: (100.0, "C6H12O"), 11: (100.0, "C6H12O")},
    )

    candidates = finalize_review_queue(pd.DataFrame([first, second]))

    assert len(candidates) == 2
    assert candidates["duplicate_spectrum_cross_molecule"].all()
    assert (candidates["review_priority"] == 4).all()
