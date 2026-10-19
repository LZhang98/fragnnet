"""Check HR-EI audit readers against annotated MSP and multi-record JCAMP inputs."""

import pytest

from scripts.debug.ei.audit_hr_ei_raw import formula_counts, read_records


def test_msp_annotations_and_record_boundaries(tmp_path):
    path = tmp_path / "spectra.msp"
    path.write_text(
        "NAME: derivative\nSMILES: C[Si](C)(C)O\nNum Peaks: 2\n"
        '73.04681 6882 "Formula C3H9Si"\n74.04642 3441\n\n'
        'Name: ethanol\nComments: "SMILES=CCO"\nNum peaks: 1\n'
        "46.0413 1.0\n"
    )
    records = read_records(path)
    assert len(records) == 2
    assert records[0][0]["smiles"] == "C[Si](C)(C)O"
    assert records[0][1] == [(73.04681, 6882), (74.04642, 3441)]
    assert records[1][0]["comments"] == '"SMILES=CCO"'
    assert records[1][1] == [(46.0413, 1)]


def test_jcamp_multiple_pairs_and_records(tmp_path):
    path = tmp_path / "spectra.jdx"
    record = (
        "##TITLE=example\n##XFACTOR=1\n##YFACTOR=1\n##NPOINTS=2\n"
        "##PEAK TABLE=(XY..XY)\n43.018,1e2 44.021,3.5\n##END=\n"
    )
    path.write_text(record * 2)
    records = read_records(path)
    assert len(records) == 2
    assert records[0][1] == [(43.018, 100), (44.021, 3.5)]
    assert records[1][0]["npoints"] == "2"


def test_msp_malformed_peak_fails(tmp_path):
    path = tmp_path / "bad.msp"
    path.write_text("Name: bad\nNum peaks: 1\nnot a peak\n")
    with pytest.raises(ValueError, match="Invalid MSP peak"):
        read_records(path)


def test_formula_order_and_unsupported_isotope_notation():
    assert formula_counts("H4C5OF8") == formula_counts("C5H4F8O")
    assert formula_counts("C18H44N2O4S2Si4")["Si"] == 4
    assert formula_counts("C20H12") != formula_counts("C20H24")
    assert formula_counts("[2H]12C20") == {}
