"""Focused tests for the NIST data-quality evaluation utilities."""

from __future__ import annotations

import re
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
    HARD_FLAG_WEIGHTS,
    HEURISTIC_FLAG_WEIGHTS,
    PRECURSOR_TYPE_TO_MASS_DIFF,
    REVIEW_FLAG_GLOSSARY,
    add_cross_ce_orphan_signals,
    add_replicate_consensus_signals,
    add_uwpr_signals,
    analyze_spectrum,
    canonicalize_prec_type,
    collect_cross_ce_comparison_spectra,
    finalize_review_queue,
    write_review_pdf,
)


def test_every_review_flag_has_a_glossary_entry() -> None:
    emitted_reasons = set(HARD_FLAG_WEIGHTS) | set(HEURISTIC_FLAG_WEIGHTS)
    context_only_flags = {
        "unsupported_precursor_type",
        "duplicate_spectrum_cross_molecule",
        "prec_type_was_normalized",
        "duplicate_spectrum_same_structure",
    }

    assert set(REVIEW_FLAG_GLOSSARY) == emitted_reasons | context_only_flags
    assert all(
        level in {"Hard", "Heuristic", "Context"} for level, _ in REVIEW_FLAG_GLOSSARY.values()
    )
    assert all(REVIEW_FLAG_GLOSSARY[flag][0] == "Context" for flag in context_only_flags)
    assert all(explanation.strip() for _, explanation in REVIEW_FLAG_GLOSSARY.values())


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

    assert not candidates["duplicate_spectrum_cross_molecule"].any()
    assert candidates["sparse_spectrum"].all()
    assert (candidates["review_priority"] == 1).all()


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
            peaks=[(20.0, 99.0), (20.0, 1.0)],
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
        "inst_type": "FT",
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
    candidates = pd.DataFrame(
        {"spec_id": [1], "substantial_cross_ce_orphan_intensity": [True]}
    )

    comparisons = collect_cross_ce_comparison_spectra(spec_df, candidates)

    assert [row["spec_id"] for row in comparisons[1]] == [2, 3]
    assert [row["ce_value"] for row in comparisons[1]] == [20.0, 30.0]


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


def test_write_review_pdf_creates_summary_and_spectrum_pages(tmp_path: Path) -> None:
    row = _review_row(peaks=[(20.0, 1.0), (40.0, 0.5)])
    review = analyze_spectrum(row, {10: (100.0, "C6H12O")})
    candidates = finalize_review_queue(pd.DataFrame([review]))
    out_fp = tmp_path / "review.pdf"

    write_review_pdf(candidates, {1: row.peaks}, str(out_fp), smiles_by_mol_id={10: "CCO"})

    assert out_fp.read_bytes().startswith(b"%PDF")
    assert out_fp.stat().st_size > 1_000
