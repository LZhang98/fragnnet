"""Focused tests for the NIST data-quality evaluation utilities."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch as th
from matplotlib import pyplot as plt

EVAL_SCRIPTS_DP = Path(__file__).parents[1] / "scripts" / "eval"
sys.path.insert(0, str(EVAL_SCRIPTS_DP))

import dag_oracle_scores  # noqa: E402
import nist_bad_data_review  # noqa: E402
from dag_oracle_scores import (  # noqa: E402
    _allowed_formula_idxs,
    _detect_is_compressed,
    _load_uwpr_maps,
)
from nist_bad_data_review import (  # noqa: E402
    HARD_FLAG_WEIGHTS,
    HEURISTIC_FLAG_WEIGHTS,
    PRECURSOR_TYPE_TO_MASS_DIFF,
    REVIEW_FLAG_GLOSSARY,
    _dag_weighted_recall,
    _group_pdf_candidate_pages,
    _ordered_cross_ce_spectra,
    _pdf_shared_x_max,
    _pdf_wrecall,
    _plot_pdf_spectrum,
    _select_dedicated_comparisons,
    add_cross_ce_orphan_signals,
    add_replicate_consensus_signals,
    add_uwpr_signals,
    analyze_spectrum,
    apply_dag_wrecall_review_gate,
    canonicalize_prec_type,
    collect_cross_ce_comparison_spectra,
    collect_dag_weighted_recall,
    collect_formula_conflict_comparison_spectra,
    collect_same_ce_replicate_spectra,
    collect_uwpr_contaminant_matches,
    collect_uwpr_peak_annotations,
    finalize_review_queue,
    write_review_html,
    write_review_pdf,
)
from uwpr_feasibility import main as uwpr_feasibility_main  # noqa: E402


def test_every_review_flag_has_a_glossary_entry() -> None:
    emitted_reasons = set(HARD_FLAG_WEIGHTS) | set(HEURISTIC_FLAG_WEIGHTS)
    context_only_flags = {
        "unsupported_precursor_type",
        "duplicate_spectrum_cross_molecule",
        "prec_type_was_normalized",
        "duplicate_spectrum_same_structure",
        "sparse_spectrum",
    }

    assert set(REVIEW_FLAG_GLOSSARY) == emitted_reasons | context_only_flags
    assert all(
        level in {"Hard", "Heuristic", "Context"} for level, _ in REVIEW_FLAG_GLOSSARY.values()
    )
    assert all(REVIEW_FLAG_GLOSSARY[flag][0] == "Context" for flag in context_only_flags)
    assert all(explanation.strip() for _, explanation in REVIEW_FLAG_GLOSSARY.values())


def test_dag_weighted_recall_uses_observed_intensity() -> None:
    wrecall = _dag_weighted_recall(
        [(20.0, 1.0), (40.0, 3.0)],
        np.array([20.001]),
        "ORB",
    )

    assert wrecall == pytest.approx(0.25)


def test_missing_dag_directories_leave_wrecall_unavailable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    spec_df = pd.DataFrame(
        {
            "spec_id": [1],
            "mol_id": [10],
            "prec_type": ["[M+H]+"],
            "inst_type": ["ORB"],
            "peaks": [[(20.0, 1.0)]],
        }
    )

    recalls = collect_dag_weighted_recall(
        spec_df,
        {1},
        {"D3": str(tmp_path / "missing-d3"), "D4": str(tmp_path / "missing-d4")},
        n_jobs=1,
    )

    assert recalls == {}
    assert "Skipping D3 wRecall" in caplog.text
    assert "Skipping D4 wRecall" in caplog.text


def test_pdf_wrecall_is_not_reported_for_unvalidated_instrument_tolerance() -> None:
    recalls = {1: {"D3": 0.02, "D4": 0.03}}

    assert _pdf_wrecall(1, "IT", recalls) == "wRecall D3=n/a | D4=n/a"
    assert _pdf_wrecall(1, "ORB", recalls) == "wRecall D3=0.020 | D4=0.030"


def test_pdf_spectrum_marks_and_includes_precursor_mz() -> None:
    row = SimpleNamespace(prec_mz=100.0, expected_prec_mz=99.9)
    x_max = _pdf_shared_x_max([[(20.0, 1.0)]], [row])
    figure, axis = plt.subplots()

    _plot_pdf_spectrum(axis, [(20.0, 1.0)], "test", "#2457a6", x_max, precursor_mz=100.0)

    assert x_max == pytest.approx(103.0)
    assert any(np.allclose(line.get_xdata(), [100.0, 100.0]) for line in axis.lines)
    assert any(label.get_text() == "prec m/z" for label in axis.texts)
    plt.close(figure)


def test_pdf_spectrum_labels_uwpr_peak_with_contaminant_formula() -> None:
    figure, axis = plt.subplots()

    _plot_pdf_spectrum(
        axis,
        [(59.0, 1.0)],
        "test",
        "#2457a6",
        peak_annotations={
            "uwpr_contaminant": [59.0],
            "uwpr_formula_by_mz": {59.0: "CH3CN"},
        },
    )

    assert any(label.get_text() == "UWPR CH3CN" for label in axis.texts)
    plt.close(figure)


def test_pdf_spectrum_colors_uwpr_above_precursor_red() -> None:
    figure, axis = plt.subplots()

    _plot_pdf_spectrum(
        axis,
        [(59.0, 1.0), (120.0, 0.5)],
        "test",
        "#2457a6",
        peak_annotations={"uwpr_contaminant": [59.0, 120.0]},
        precursor_mz=100.0,
    )

    uwpr_labels = {float(label.get_position()[0]): label.get_color() for label in axis.texts}
    assert uwpr_labels[59.0] == "#e07a19"
    assert uwpr_labels[120.0] == "#c43b3b"
    plt.close(figure)


def test_dag_wrecall_gate_only_marks_high_coverage_soft_spectral_evidence() -> None:
    candidates = pd.DataFrame(
        {
            "spec_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "has_hard_evidence": [False, False, False, False, False, True, False, False, False],
            "evidence_families": [
                "cross_ce_orphans",
                "cross_ce_orphans",
                "contaminant_library",
                "cross_ce_orphans",
                "cross_ce_orphans",
                "record_integrity",
                "metadata_completeness",
                "",
                "cross_ce_orphans;record_integrity",
            ],
            "review_reasons": [
                "substantial_cross_ce_orphan_intensity",
                "substantial_cross_ce_orphan_intensity",
                "substantial_uwpr_contaminant_only_intensity",
                "substantial_cross_ce_orphan_intensity",
                "substantial_cross_ce_orphan_intensity",
                "polarity_mismatch",
                "missing_precursor_mz",
                "substantial_intensity_above_precursor;duplicate_peak_mz",
                "substantial_cross_ce_orphan_intensity;polarity_mismatch",
            ],
        }
    )
    recalls = {
        1: {"D3": 0.749, "D4": 0.01},
        2: {"D3": 0.75, "D4": 0.01},
        3: {"D3": 0.01, "D4": 0.99},
        4: {"D3": np.nan, "D4": np.nan},
        5: {"D3": 0.1, "D4": 0.2},
        6: {"D3": 0.99, "D4": 0.99},
        7: {"D3": 0.99, "D4": 0.99},
        8: {"D3": 0.99, "D4": 0.99},
        9: {"D3": 0.99, "D4": 0.99},
    }

    gated = apply_dag_wrecall_review_gate(candidates, recalls)

    assert gated["spec_id"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert gated["dag_wrecall_gated"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert gated["dag_wrecall_gate_exempt"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert gated["dag_wrecall_d3"].iloc[0] == pytest.approx(0.749)
    assert gated["dag_wrecall_d4"].iloc[0] == pytest.approx(0.01)
    assert gated["dag_wrecall_max"].iloc[0] == pytest.approx(0.749)
    assert pd.isna(gated["dag_wrecall_max"].iloc[3])


def test_dedicated_comparison_budget_keeps_every_available_evidence_type() -> None:
    conflicts = [{"spec_id": spec_id} for spec_id in range(1, 12)]
    cross_ce = [{"spec_id": 20}, {"spec_id": 21}]

    selected_conflicts, selected_replicates, selected_cross_ce = _select_dedicated_comparisons(
        conflicts, [], cross_ce
    )

    assert selected_replicates == []
    assert len(selected_conflicts) + len(selected_cross_ce) == 11
    assert [row["spec_id"] for row in selected_cross_ce] == [20, 21]


def test_html_discloses_wrecall_gated_candidates(tmp_path: Path) -> None:
    out_fp = tmp_path / "review.html"

    write_review_html(pd.DataFrame(), {}, str(out_fp), n_wrecall_gated=12)

    html_text = out_fp.read_text(encoding="utf-8")
    assert "omits 12 soft spectral candidates" in html_text
    assert "dag_wrecall_gated=true" in html_text


def test_uwpr_feasibility_writes_empty_scoped_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hits_fp = tmp_path / "hits.csv"
    spec_fp = tmp_path / "spec_df.pkl"
    out_fp = tmp_path / "feasibility.csv"
    pd.DataFrame(
        {
            "spec_id": [1],
            "mol_id": [10],
            "peak_mz": [50.0],
            "ion_mode": ["P"],
        }
    ).to_csv(hits_fp, index=False)
    pd.DataFrame(
        {
            "spec_id": [2],
            "dset": ["nist20_hr"],
            "prec_type": ["[M+H]+"],
        }
    ).to_pickle(spec_fp)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "uwpr_feasibility.py",
            "--hits_csv",
            str(hits_fp),
            "--spec_fp",
            str(spec_fp),
            "--out_csv",
            str(out_fp),
            "--dsets",
            "nist20_hr",
        ],
    )

    uwpr_feasibility_main()

    result = pd.read_csv(out_fp)
    assert result.empty
    assert "feasibility" in result.columns


def test_pdf_pages_group_same_molecule_adduct_and_instrument() -> None:
    rows = [
        SimpleNamespace(spec_id=spec_id, mol_id=10, prec_type="[M+H]+", inst_type="ORB")
        for spec_id in range(1, 15)
    ]
    rows.append(SimpleNamespace(spec_id=20, mol_id=11, prec_type="[M+H]+", inst_type="ORB"))

    pages = _group_pdf_candidate_pages(rows, {3: [{"spec_id": 30}]}, {}, {})

    assert [[row.spec_id for row in page] for page in pages] == [
        [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [14],
        [3],
        [20],
    ]


def test_grouped_pdf_candidates_put_higher_ce_after_lower_ce() -> None:
    rows = [
        SimpleNamespace(
            spec_id=1,
            mol_id=10,
            prec_type="[M+H]+",
            inst_type="ORB",
            nce=40.0,
            ace=np.nan,
        ),
        SimpleNamespace(
            spec_id=2,
            mol_id=10,
            prec_type="[M+H]+",
            inst_type="ORB",
            nce=10.0,
            ace=np.nan,
        ),
        SimpleNamespace(
            spec_id=3,
            mol_id=10,
            prec_type="[M+H]+",
            inst_type="ORB",
            nce=np.nan,
            ace=np.nan,
        ),
    ]

    pages = _group_pdf_candidate_pages(rows, {}, {}, {})

    assert [[row.spec_id for row in page] for page in pages] == [[2, 1, 3]]


def test_guide_glossary_matches_code_names_levels_and_order() -> None:
    guide_fp = Path(__file__).parents[1] / "docs" / "guides" / "nist_human_review.md"
    guide = guide_fp.read_text(encoding="utf-8")
    glossary = guide.split("## Review flag glossary", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    guide_entries = re.findall(
        r"^- `([^`]+)` — \*\*(Hard|Heuristic|Context):\*\*",
        glossary,
        flags=re.MULTILINE,
    )
    code_entries = [(name, level) for name, (level, _) in REVIEW_FLAG_GLOSSARY.items()]

    assert guide_entries == code_entries


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
    monkeypatch.setattr(
        dag_oracle_scores.dag_format_utils, "load_frag_d", lambda *args, **kwargs: dag
    )
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
            "inst_type": "ORB",
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
        "inst_type": "ORB",
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
    assert "precursor_mass_mismatch" in result["heuristic_reasons"]
    assert "precursor_mass_mismatch" not in result["hard_reasons"]


def test_review_detector_does_not_flag_consistent_spectrum() -> None:
    result = analyze_spectrum(_review_row(), {10: (100.0, "C6H12O")})
    candidates = finalize_review_queue(pd.DataFrame([result]))

    assert candidates.empty


@pytest.mark.parametrize(
    ("raw_prec_type", "canonical_prec_type"),
    [
        ("[M+H-H2O]+", "[M-H2O+H]+"),
        ("[M+H-2H2O]+", "[M-2H2O+H]+"),
        ("[M+CHO2]-", "[M+HCOOH-H]-"),
        ("M-H", "[M-H]-"),
    ],
)
def test_precursor_spelling_variants_are_canonicalized(
    raw_prec_type: str, canonical_prec_type: str
) -> None:
    ion_mode = "P" if canonical_prec_type.endswith("+") else "N"
    prec_mz = 100.0 + PRECURSOR_TYPE_TO_MASS_DIFF[canonical_prec_type]

    result = analyze_spectrum(
        _review_row(prec_type=raw_prec_type, ion_mode=ion_mode, prec_mz=prec_mz),
        {10: (100.0, "C6H12O")},
    )

    assert canonicalize_prec_type(raw_prec_type) == canonical_prec_type
    assert result["prec_type_canonical"] == canonical_prec_type
    assert result["prec_type_was_normalized"]
    assert not result["unsupported_precursor_type"]
    assert not result["precursor_mass_mismatch"]


def test_empty_precursor_type_is_not_a_supported_zero_mass_adduct() -> None:
    result = analyze_spectrum(
        _review_row(prec_type="", prec_mz=100.0),
        {10: (100.0, "C6H12O")},
    )

    assert "" not in PRECURSOR_TYPE_TO_MASS_DIFF
    assert result["unsupported_precursor_type"]
    assert "unsupported_precursor_type" not in result["review_reasons"]


def test_dominant_peak_is_computed_without_precursor_mz() -> None:
    result = analyze_spectrum(
        _review_row(prec_mz=None, peaks=[(20.0, 98.0), (40.0, 1.0), (60.0, 1.0)]),
        {10: (100.0, "C6H12O")},
    )

    assert result["dominant_peak_fraction"] == pytest.approx(0.98)
    assert result["single_peak_dominates"]


def test_precursor_envelope_intensity_is_not_flagged_as_suspicious_above_precursor() -> None:
    prec_mz = 101.007276
    result = analyze_spectrum(
        _review_row(prec_mz=prec_mz, peaks=[(50.0, 10.0), (prec_mz + 1.003, 90.0)]),
        {10: (100.0, "C6H12O")},
    )

    assert result["intensity_above_precursor_fraction"] == pytest.approx(0.9)
    assert result["precursor_envelope_intensity_fraction"] == pytest.approx(0.9)
    assert result["non_envelope_intensity_above_precursor_fraction"] == 0.0
    assert not result["substantial_intensity_above_precursor"]


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


def test_low_resolution_precursor_is_not_mass_checked() -> None:
    result = analyze_spectrum(
        _review_row(inst_type="IT", prec_mz=101.2),
        {10: (100.0, "C6H12O")},
    )

    assert not result["precursor_mass_mismatch"]
    assert pd.isna(result["expected_prec_mz"])


def test_empty_spectrum_is_not_also_sparse() -> None:
    result = analyze_spectrum(_review_row(peaks=[]), {10: (100.0, "C6H12O")})

    assert result["empty_spectrum"]
    assert not result["sparse_spectrum"]


def test_review_queue_flags_identical_spectrum_across_molecules() -> None:
    mol_meta = {
        10: (100.0, "C6H12O", "STRUCTURE-A"),
        11: (100.0, "C7H16", "STRUCTURE-B"),
    }
    first = analyze_spectrum(_review_row(), mol_meta)
    second = analyze_spectrum(
        _review_row(spec_id=2, mol_id=11, dset_spec_id="nist-2"),
        mol_meta,
    )

    candidates = finalize_review_queue(pd.DataFrame([first, second]))

    assert len(candidates) == 2
    assert candidates["duplicate_spectrum_cross_molecule"].all()
    assert candidates["duplicate_spectrum_formula_conflict"].all()
    assert (candidates["review_priority"] == 5).all()
    assert (candidates["review_tier"] == "hard_conflict").all()


def test_review_queue_does_not_call_sparse_spectra_cross_molecule_duplicates() -> None:
    sparse_peaks = [(20.0, 1.0), (40.0, 0.5)]
    first = analyze_spectrum(
        _review_row(peaks=sparse_peaks),
        {10: (100.0, "C6H12O"), 11: (100.0, "C6H12O")},
    )
    second = analyze_spectrum(
        _review_row(spec_id=2, mol_id=11, dset_spec_id="nist-2", peaks=sparse_peaks),
        {10: (100.0, "C6H12O"), 11: (100.0, "C6H12O")},
    )

    candidates = finalize_review_queue(pd.DataFrame([first, second]))

    assert candidates.empty


def test_same_formula_duplicate_is_label_ambiguity_not_hard_conflict() -> None:
    mol_meta = {
        10: (100.0, "C6H12O", "STRUCTURE-A"),
        11: (100.0, "C6H12O", "STRUCTURE-B"),
    }
    first = analyze_spectrum(_review_row(), mol_meta)
    second = analyze_spectrum(_review_row(spec_id=2, mol_id=11, dset_spec_id="nist-2"), mol_meta)

    candidates = finalize_review_queue(pd.DataFrame([first, second]))

    assert candidates["duplicate_spectrum_structure_ambiguity"].all()
    assert not candidates["duplicate_spectrum_formula_conflict"].any()
    assert not candidates["has_hard_evidence"].any()
    assert (candidates["review_priority"] == 1).all()


def test_same_structure_duplicate_is_context_only() -> None:
    mol_meta = {
        10: (100.0, "C6H12O", "SAME-STRUCTURE"),
        11: (100.0, "C6H12O", "SAME-STRUCTURE"),
    }
    first = analyze_spectrum(_review_row(), mol_meta)
    second = analyze_spectrum(_review_row(spec_id=2, mol_id=11, dset_spec_id="nist-2"), mol_meta)

    candidates = finalize_review_queue(pd.DataFrame([first, second]))

    assert candidates.empty


def test_hard_evidence_ranks_before_higher_scoring_heuristics() -> None:
    hard_conflict = analyze_spectrum(
        _review_row(spec_id=1, ion_mode="N"),
        {10: (100.0, "C6H12O")},
    )
    heuristic_only = analyze_spectrum(
        _review_row(
            spec_id=2,
            prec_mz=None,
            ace=-1.0,
            peaks=[(20.0, 99.0), (20.0, 0.5), (20.0, 0.5)],
        ),
        {10: (100.0, "C6H12O")},
    )

    candidates = finalize_review_queue(pd.DataFrame([heuristic_only, hard_conflict]))

    assert candidates.loc[0, "spec_id"] == 1
    assert candidates.loc[0, "review_tier"] == "hard_conflict"
    assert candidates.loc[1, "review_priority"] > candidates.loc[0, "review_priority"]


def _signal_spec_df(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "mol_id": 10,
        "prec_type": "[M+H]+",
        "inst_type": "ORB",
        "frag_mode": "HCD",
        "group_id": 1,
        "ace": 10.0,
        "nce": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_replicate_consensus_flags_only_the_disagreeing_spectrum() -> None:
    spec_df = _signal_spec_df(
        [
            {"spec_id": 1, "peaks": [(50.0, 10.0), (75.0, 5.0)]},
            {"spec_id": 2, "peaks": [(50.001, 9.0), (75.001, 5.0)]},
            {"spec_id": 3, "peaks": [(120.0, 10.0), (140.0, 5.0)]},
        ]
    )
    review_df = pd.DataFrame({"spec_id": [1, 2, 3]})

    result = add_replicate_consensus_signals(spec_df, review_df)

    flags = result.set_index("spec_id")["replicate_consensus_disagreement"]
    assert flags.to_dict() == {1: False, 2: False, 3: True}
    assert result.set_index("spec_id").loc[3, "other_replicate_consensus"] > 0.99


def test_cross_ce_orphan_signal_requires_two_other_collision_energies() -> None:
    three_ces = _signal_spec_df(
        [
            {"spec_id": 1, "ace": 10.0, "peaks": [(50.0, 1.0), (999.0, 9.0)]},
            {"spec_id": 2, "ace": 20.0, "peaks": [(50.001, 1.0), (75.0, 9.0)]},
            {"spec_id": 3, "ace": 30.0, "peaks": [(50.002, 1.0), (75.001, 9.0)]},
        ]
    )
    review_df = pd.DataFrame({"spec_id": [1, 2, 3]})

    result = add_cross_ce_orphan_signals(three_ces, review_df)

    row = result.set_index("spec_id").loc[1]
    assert row["cross_ce_orphan_intensity_fraction"] == pytest.approx(0.9)
    assert bool(row["substantial_cross_ce_orphan_intensity"])

    two_ces = three_ces.iloc[:2]
    result = add_cross_ce_orphan_signals(two_ces, review_df.iloc[:2])
    assert not result["substantial_cross_ce_orphan_intensity"].any()


def test_cross_ce_comparison_spectra_exclude_the_primary_ce_bin() -> None:
    spec_df = _signal_spec_df(
        [
            {
                "spec_id": 1,
                "dset_spec_id": "nist-1",
                "ace": 10.0,
                "peaks": [(50.0, 1.0)],
            },
            {
                "spec_id": 2,
                "dset_spec_id": "nist-2",
                "ace": 20.0,
                "peaks": [(60.0, 1.0)],
            },
            {
                "spec_id": 3,
                "dset_spec_id": "nist-3",
                "ace": 30.0,
                "peaks": [(70.0, 1.0)],
            },
        ]
    )
    candidates = pd.DataFrame({"spec_id": [1], "substantial_cross_ce_orphan_intensity": [True]})

    comparisons = collect_cross_ce_comparison_spectra(spec_df, candidates)

    assert [row["spec_id"] for row in comparisons[1]] == [2, 3]
    assert [row["ce_value"] for row in comparisons[1]] == [20.0, 30.0]


def test_cross_ce_comparison_spectra_include_every_spectrum_in_each_bin() -> None:
    rows = [{"spec_id": 1, "dset_spec_id": "nist-1", "ace": 50.0, "peaks": [(50.0, 1.0)]}]
    rows.extend(
        {
            "spec_id": idx + 2,
            "dset_spec_id": f"nist-{idx + 2}",
            "ace": float(idx),
            "peaks": [(60.0 + idx, 1.0)],
        }
        for idx in range(15)
    )
    rows.append(
        {"spec_id": 100, "dset_spec_id": "duplicate-ce", "ace": 10.0, "peaks": [(90.0, 1.0)]}
    )
    spec_df = _signal_spec_df(rows)
    spec_df["spec_id"] = spec_df["spec_id"].astype(object)
    candidates = pd.DataFrame({"spec_id": [1], "substantial_cross_ce_orphan_intensity": [True]})

    comparisons = collect_cross_ce_comparison_spectra(spec_df, candidates)[1]

    assert len(comparisons) == 16
    assert [row["spec_id"] for row in comparisons if row["ce_value"] == 10.0] == [12, 100]
    assert all(row["n_other_ce_spectra_available"] == 16 for row in comparisons)


def test_cross_ce_plot_sequence_inserts_target_by_collision_energy() -> None:
    row = SimpleNamespace(spec_id=1, dset_spec_id="nist-1", ace=20.0)
    comparisons = [
        {
            "spec_id": 2,
            "dset_spec_id": "nist-2",
            "ce_source": "ace",
            "ce_value": 10.0,
            "peaks": [(50.0, 1.0)],
        },
        {
            "spec_id": 3,
            "dset_spec_id": "nist-3",
            "ce_source": "ace",
            "ce_value": 30.0,
            "peaks": [(70.0, 1.0)],
        },
    ]

    ordered = _ordered_cross_ce_spectra(row, [(60.0, 1.0)], comparisons)

    assert [spectrum["ce_value"] for spectrum in ordered] == [10.0, 20.0, 30.0]
    assert [spectrum["is_review_target"] for spectrum in ordered] == [False, True, False]


def test_same_ce_replicate_spectra_include_every_peer() -> None:
    spec_df = _signal_spec_df(
        [
            {"spec_id": 1, "dset_spec_id": "nist-1", "ace": 10.0, "peaks": [(50.0, 1.0)]},
            {"spec_id": 2, "dset_spec_id": "nist-2", "ace": 10.0, "peaks": [(60.0, 1.0)]},
            {"spec_id": 3, "dset_spec_id": "nist-3", "ace": 10.0, "peaks": [(70.0, 1.0)]},
            {"spec_id": 4, "dset_spec_id": "nist-4", "ace": 20.0, "peaks": [(80.0, 1.0)]},
        ]
    )
    candidates = pd.DataFrame({"spec_id": [1], "replicate_consensus_disagreement": [True]})

    comparisons = collect_same_ce_replicate_spectra(spec_df, candidates)

    assert [row["spec_id"] for row in comparisons[1]] == [2, 3]
    assert all(row["ace"] == 10.0 for row in comparisons[1])


def test_formula_conflict_comparisons_include_exact_duplicate_metadata() -> None:
    mol_meta = {
        10: (100.0, "C6H12O", "STRUCTURE-A"),
        11: (78.017841, "C5H4N", "STRUCTURE-B"),
    }
    first_row = _review_row()
    second_row = _review_row(
        spec_id=2,
        mol_id=11,
        dset_spec_id="nist-2",
        prec_type="[M+Na]+",
        prec_mz=101.007061,
    )
    review_df = pd.DataFrame(
        [analyze_spectrum(first_row, mol_meta), analyze_spectrum(second_row, mol_meta)]
    )
    candidates = finalize_review_queue(review_df)
    spec_df = pd.DataFrame(
        {
            "spec_id": [1, 2],
            "peaks": [first_row.peaks, second_row.peaks],
        }
    )

    comparisons = collect_formula_conflict_comparison_spectra(
        spec_df, review_df, candidates.iloc[[0]]
    )

    assert len(comparisons[1]) == 1
    assert comparisons[1][0]["spec_id"] == 2
    assert comparisons[1][0]["mol_id"] == 11
    assert comparisons[1][0]["formula"] == "C5H4N"
    assert comparisons[1][0]["prec_type"] == "[M+Na]+"
    assert comparisons[1][0]["ace"] == 10.0
    assert comparisons[1][0]["peaks"] == second_row.peaks
    assert comparisons[1][0]["cosine_similarity"] == pytest.approx(1.0)


def test_uwpr_signal_uses_unique_peak_intensity_and_conservative_labels(tmp_path: Path) -> None:
    spec_df = _signal_spec_df([{"spec_id": 1, "peaks": [(59.0, 8.0), (100.0, 2.0)]}])
    review_df = pd.DataFrame({"spec_id": [1]})
    uwpr_fp = tmp_path / "uwpr.csv"
    pd.DataFrame(
        {
            "spec_id": [1, 1, 1],
            "peak_mz": [59.0, 59.0, 100.0],
            "feasibility": ["contaminant_only", "contaminant_only", "dag_feasible"],
        }
    ).to_csv(uwpr_fp, index=False)

    result = add_uwpr_signals(spec_df, review_df, uwpr_fp)
    row = result.iloc[0]

    assert row["uwpr_matched_intensity_fraction"] == pytest.approx(1.0)
    assert row["uwpr_contaminant_only_intensity_fraction"] == pytest.approx(0.8)
    assert row["n_uwpr_contaminant_only_peaks"] == 1
    assert bool(row["substantial_uwpr_contaminant_only_intensity"])


def test_uwpr_signal_requires_consensus_across_dag_audits(tmp_path: Path) -> None:
    spec_df = _signal_spec_df([{"spec_id": 1, "peaks": [(59.0, 8.0), (100.0, 2.0)]}])
    review_df = pd.DataFrame({"spec_id": [1]})
    d3_fp = tmp_path / "d3.csv"
    d4_fp = tmp_path / "d4.csv"
    pd.DataFrame(
        {
            "spec_id": [1, 1],
            "peak_mz": [59.0, 100.0],
            "feasibility": ["contaminant_only", "dag_feasible"],
        }
    ).to_csv(d3_fp, index=False)
    pd.DataFrame(
        {
            "spec_id": [1, 1],
            "peak_mz": [59.0, 100.0],
            "feasibility": ["dag_feasible", "dag_feasible"],
        }
    ).to_csv(d4_fp, index=False)

    row = add_uwpr_signals(spec_df, review_df, [d3_fp, d4_fp]).iloc[0]

    assert row["n_uwpr_feasibility_sources"] == 2
    assert row["uwpr_contaminant_only_intensity_fraction"] == 0.0
    assert row["uwpr_dag_disagreement_intensity_fraction"] == pytest.approx(0.8)
    assert not bool(row["substantial_uwpr_contaminant_only_intensity"])


def test_uwpr_undecidable_source_abstains_from_consensus(tmp_path: Path) -> None:
    spec_df = _signal_spec_df([{"spec_id": 1, "peaks": [(59.0, 8.0), (100.0, 2.0)]}])
    review_df = pd.DataFrame({"spec_id": [1]})
    d3_fp = tmp_path / "d3.csv"
    d4_fp = tmp_path / "d4.csv"
    pd.DataFrame({"spec_id": [1], "peak_mz": [59.0], "feasibility": ["contaminant_only"]}).to_csv(
        d3_fp, index=False
    )
    pd.DataFrame({"spec_id": [1], "peak_mz": [59.0], "feasibility": ["missing_metadata"]}).to_csv(
        d4_fp, index=False
    )

    row = add_uwpr_signals(spec_df, review_df, [d3_fp, d4_fp]).iloc[0]

    assert row["uwpr_contaminant_only_intensity_fraction"] == pytest.approx(0.8)
    assert bool(row["substantial_uwpr_contaminant_only_intensity"])


def test_collect_uwpr_matches_requires_consensus_across_sources(tmp_path: Path) -> None:
    d3_fp = tmp_path / "d3.csv"
    d4_fp = tmp_path / "d4.csv"
    columns = ["spec_id", "peak_mz", "feasibility"]
    pd.DataFrame(
        [[1, 59.0, "contaminant_only"], [1, 100.0, "contaminant_only"]],
        columns=columns,
    ).to_csv(d3_fp, index=False)
    pd.DataFrame(
        [[1, 59.0, "contaminant_only"], [1, 100.0, "dag_feasible"]],
        columns=columns,
    ).to_csv(d4_fp, index=False)
    candidates = pd.DataFrame(
        {"spec_id": [1], "substantial_uwpr_contaminant_only_intensity": [True]}
    )

    matches = collect_uwpr_contaminant_matches([d3_fp, d4_fp], candidates)

    assert matches == {1: [59.0]}


def test_collect_uwpr_annotations_separates_d3_and_additional_d4(tmp_path: Path) -> None:
    d3_fp = tmp_path / "d3.csv"
    d4_fp = tmp_path / "d4.csv"
    columns = ["spec_id", "peak_mz", "feasibility", "cont_formula"]
    pd.DataFrame(
        [
            [1, 50.0, "dag_feasible", "C2H6O"],
            [1, 75.0, "contaminant_only", "C3H8O"],
            [1, 100.0, "contaminant_only", "CH3CN"],
        ],
        columns=columns,
    ).to_csv(d3_fp, index=False)
    pd.DataFrame(
        [
            [1, 50.0, "dag_feasible", "C2H6O"],
            [1, 75.0, "dag_feasible", "C3H8O"],
            [1, 100.0, "contaminant_only", "HCN"],
        ],
        columns=columns,
    ).to_csv(d4_fp, index=False)
    candidates = pd.DataFrame(
        {"spec_id": [1], "substantial_uwpr_contaminant_only_intensity": [True]}
    )

    annotations = collect_uwpr_peak_annotations([d3_fp, d4_fp], candidates)

    assert annotations == {
        1: {
            "d3_dag": [50.0],
            "d4_additional_dag": [75.0],
            "uwpr_contaminant": [100.0],
            "uwpr_formula_by_mz": {100.0: "CH3CN/HCN"},
        }
    }


def test_collect_uwpr_annotations_skips_disagreement_only_rows(tmp_path: Path) -> None:
    d3_fp = tmp_path / "d3.csv"
    d4_fp = tmp_path / "d4.csv"
    columns = ["spec_id", "peak_mz", "feasibility"]
    pd.DataFrame([[1, 50.0, "dag_feasible"]], columns=columns).to_csv(d3_fp, index=False)
    pd.DataFrame([[1, 50.0, "contaminant_only"]], columns=columns).to_csv(d4_fp, index=False)
    candidates = pd.DataFrame(
        {
            "spec_id": [1],
            "substantial_uwpr_contaminant_only_intensity": [False],
            "n_uwpr_dag_disagreement_peaks": [1],
        }
    )

    annotations = collect_uwpr_peak_annotations([d3_fp, d4_fp], candidates)

    assert annotations == {}


def test_write_review_pdf_creates_summary_and_spectrum_pages(tmp_path: Path) -> None:
    row = _review_row(peaks=[(20.0, 1.0), (40.0, 0.5)])
    review = analyze_spectrum(row, {10: (100.0, "C6H12O")})
    candidates = finalize_review_queue(pd.DataFrame([review]))
    out_fp = tmp_path / "review.pdf"

    write_review_pdf(candidates, {1: row.peaks}, str(out_fp), smiles_by_mol_id={10: "CCO"})

    assert out_fp.read_bytes().startswith(b"%PDF")
    assert out_fp.stat().st_size > 1_000


def test_cross_ce_pdf_highlights_siblings_that_are_review_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        _review_row(spec_id=1, dset_spec_id="nist-1", ace=10.0),
        _review_row(
            spec_id=2,
            dset_spec_id="nist-2",
            ace=20.0,
            peaks=[(21.0, 1.0), (41.0, 1.0), (61.0, 1.0)],
        ),
    ]
    records = [analyze_spectrum(row, {10: (100.0, "C6H12O")}) for row in rows]
    for record in records:
        record["substantial_cross_ce_orphan_intensity"] = False
        record["substantial_uwpr_contaminant_only_intensity"] = False
    records[0]["substantial_cross_ce_orphan_intensity"] = True
    records[1]["substantial_uwpr_contaminant_only_intensity"] = True
    candidates = finalize_review_queue(pd.DataFrame(records))
    plotted: list[tuple[str, dict[str, object] | None]] = []
    original_plot = nist_bad_data_review._plot_pdf_spectrum

    def capture_plot(
        axis: object,
        peaks: object,
        title: str,
        color: str,
        x_max: float | None = None,
        peak_annotations: dict[str, object] | None = None,
        precursor_mz: float | None = None,
    ) -> None:
        plotted.append((title, peak_annotations))
        original_plot(axis, peaks, title, color, x_max, peak_annotations, precursor_mz)

    monkeypatch.setattr(nist_bad_data_review, "_plot_pdf_spectrum", capture_plot)
    write_review_pdf(
        candidates,
        {row.spec_id: row.peaks for row in rows},
        str(tmp_path / "review.pdf"),
        cross_ce_comparisons_by_spec_id={
            1: [
                {
                    "spec_id": 2,
                    "dset_spec_id": "nist-2",
                    "ce_source": "ace",
                    "ce_value": 20.0,
                    "inst_type": "ORB",
                    "prec_mz": 101.007276,
                    "peaks": rows[1].peaks,
                }
            ]
        },
        uwpr_annotations_by_spec_id={
            2: {"uwpr_contaminant": [21.0], "uwpr_formula_by_mz": {21.0: "CH3CN"}}
        },
    )

    sibling_plots = [plot for plot in plotted if "OTHER CE" in plot[0]]
    assert len(sibling_plots) == 1
    assert "REVIEW CANDIDATE — OTHER CE" in sibling_plots[0][0]
    assert "substantial_uwpr_contaminant_only_intensity" in sibling_plots[0][0]
    assert sibling_plots[0][1]["uwpr_formula_by_mz"] == {21.0: "CH3CN"}


def test_cross_ce_pdf_paginates_without_dropping_spectra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _review_row(spec_id=1, dset_spec_id="nist-1", ace=50.0)
    record = analyze_spectrum(row, {10: (100.0, "C6H12O")})
    record["substantial_cross_ce_orphan_intensity"] = True
    candidates = finalize_review_queue(pd.DataFrame([record]))
    comparisons = [
        {
            "spec_id": spec_id,
            "dset_spec_id": f"nist-{spec_id}",
            "ce_source": "ace",
            "ce_value": float(spec_id),
            "inst_type": "ORB",
            "prec_mz": 101.007276,
            "peaks": [(20.0 + spec_id, 1.0)],
            "n_other_ce_spectra_available": 13,
        }
        for spec_id in range(2, 15)
    ]
    plotted_titles: list[str] = []
    original_plot = nist_bad_data_review._plot_pdf_spectrum

    def capture_plot(
        axis: object,
        peaks: object,
        title: str,
        color: str,
        x_max: float | None = None,
        peak_annotations: dict[str, object] | None = None,
        precursor_mz: float | None = None,
    ) -> None:
        plotted_titles.append(title)
        original_plot(axis, peaks, title, color, x_max, peak_annotations, precursor_mz)

    monkeypatch.setattr(nist_bad_data_review, "_plot_pdf_spectrum", capture_plot)
    write_review_pdf(
        candidates,
        {1: row.peaks},
        str(tmp_path / "review.pdf"),
        cross_ce_comparisons_by_spec_id={1: comparisons},
    )

    assert sum("REVIEW TARGET — CROSS-CE ORPHAN" in title for title in plotted_titles) == 2
    assert sum("OTHER CE" in title for title in plotted_titles) == 13
