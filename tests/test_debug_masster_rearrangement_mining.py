"""Peak classification of the rearrangement mining script on hand-computed spectra.

Benzenesulfonanilide, PhSO2NHPh (C12H11NO2S), is the worked example: SO2 extrusion gives
[M+H-SO2]+ = C12H12N+, whose 12 C and 1 N are not connected without the S, so no bond-cut
fragment can make it. Exact masses below are written out from monoisotopic atom masses, not
taken from the script.
"""

from __future__ import annotations

import pytest

import scripts.debug.smarts.debug_masster_rearrangement_mining as M

SMILES = "O=S(=O)(Nc1ccccc1)c1ccccc1"
H, E = 1.00782503207, 0.00054858
C, N, O, S, NA = 12.0, 14.0030740048, 15.99491461956, 31.97207100, 22.9897692809


def _mass(c=0, h=0, n=0, o=0, s=0, na=0, charge=1) -> float:
    return c * C + h * H + n * N + o * O + s * S + na * NA - charge * E


PRECURSOR_POS = _mass(12, 12, 1, 2, 1)  # [M+H]+ 234.0583
SO2_LOSS_POS = _mass(12, 12, 1)  # C12H12N+ 170.0964, rearrangement
SULFONYL_POS = _mass(6, 5, 0, 2, 1)  # C6H5O2S+ 141.0005, ordinary S-N cut


@pytest.fixture
def precursor(monkeypatch):
    def set_precursor(name: str) -> None:
        monkeypatch.setattr(M, "PRECURSOR", name)

    return set_precursor


def _labels(peaks: list[tuple[float, float]]) -> dict:
    result = M.process_molecule(("test", 0, SMILES, [(1, "train", peaks)]))
    labels = {round(row["mz"], 3): row["label"] for row in result["peaks"]}
    return labels, result["rearr"]


def test_hand_computed_masses() -> None:
    assert PRECURSOR_POS == pytest.approx(234.05833, abs=1e-5)
    assert SO2_LOSS_POS == pytest.approx(170.09643, abs=1e-5)
    assert SULFONYL_POS == pytest.approx(141.00047, abs=1e-5)


def test_mh_plus_labels_rearrangement_connected_isotope_and_junk(precursor) -> None:
    precursor("[M+H]+")
    peaks = [
        (PRECURSOR_POS, 30.0),
        (SO2_LOSS_POS, 100.0),
        (SO2_LOSS_POS + 1.0033548, 13.0),  # 13C of the SO2-loss ion
        (SULFONYL_POS, 50.0),
        (300.5, 20.0),  # above the precursor: no sub-formula
    ]
    labels, rearr = _labels(peaks)
    assert labels[round(PRECURSOR_POS, 3)] == "connected"
    assert labels[round(SO2_LOSS_POS, 3)] == "rearrangement"
    assert labels[round(SO2_LOSS_POS + 1.0033548, 3)] == "isotope"
    assert labels[round(SULFONYL_POS, 3)] == "connected"
    assert labels[round(300.5, 3)] == "no_formula"
    (row,) = rearr
    assert (row["ion"], row["ion_mol"], row["loss"], row["adduct_kept"]) == ("C12H12N", "C12H12N", "O2S", False)
    assert "S" in row["motif"]


def test_peaks_below_five_percent_of_base_are_not_classified(precursor) -> None:
    precursor("[M+H]+")
    labels, _ = _labels([(SULFONYL_POS, 100.0), (SO2_LOSS_POS, 4.0)])
    assert round(SO2_LOSS_POS, 3) not in labels


def test_mh_minus_uses_anion_masses_and_loss_h(precursor) -> None:
    precursor("[M-H]-")
    so2_loss_neg = _mass(12, 10, 1, charge=-1)  # C12H10N- 168.0820
    labels, rearr = _labels([(_mass(12, 10, 1, 2, 1, charge=-1), 40.0), (so2_loss_neg, 100.0)])
    assert labels[round(so2_loss_neg, 3)] == "rearrangement"
    (row,) = rearr
    assert (row["ion"], row["loss"]) == ("C12H10N", "O2S")
    # The C12H12N cation mass is 1.1 mDa (2 electrons) from the C12H12N anion, inside the 2 mDa floor,
    # so under [M-H]- it is read as the anion with 2 H more than [M-H-SO2]-: loss H is -2.
    _, rearr = _labels([(SO2_LOSS_POS, 100.0)])
    (row,) = rearr
    assert (row["ion"], row["loss"]) == ("C12H12N", "H-2O2S")


def test_sodium_adduct_ions_keep_na_outside_the_graph(precursor) -> None:
    precursor("[M+Na]+")
    kept = _mass(12, 11, 1, na=1)  # C12H11NNa+ 192.0785: SO2 lost, Na kept
    labels, rearr = _labels([(_mass(12, 11, 1, 2, 1, na=1), 30.0), (kept, 100.0)])
    assert labels[round(kept, 3)] == "rearrangement"
    (row,) = rearr
    assert row["adduct_kept"] is True
    assert (row["ion"], row["ion_mol"], row["loss"]) == ("C12H11NNa", "C12H11N", "O2S")


def test_bare_adduct_ion_is_labelled_adduct(precursor) -> None:
    precursor("[M+Na]+")
    labels, rearr = _labels([(_mass(na=1), 100.0)])
    assert labels[round(_mass(na=1), 3)] == "adduct"
    assert rearr == []


def test_exhausted_budget_marks_capped_never_rearrangement(precursor, monkeypatch) -> None:
    precursor("[M+H]+")
    monkeypatch.setattr(M, "MAX_CAPPED_SEARCHES", 0)
    monkeypatch.setattr(M, "REACH_CAP", 1)
    labels, rearr = _labels([(SULFONYL_POS, 50.0), (SO2_LOSS_POS, 100.0)])
    assert labels[round(SO2_LOSS_POS, 3)] == "capped"
    assert rearr == []


def test_isotope_rule_needs_a_more_intense_parent() -> None:
    import numpy as np

    peaks = np.array([[100.0, 10.0], [101.0033548, 50.0]])
    assert not M.is_isotope(101.0033548, 50.0, peaks, 0.002, heavy_m2=False)
    peaks = np.array([[100.0, 80.0], [101.0033548, 50.0]])
    assert M.is_isotope(101.0033548, 50.0, peaks, 0.002, heavy_m2=False)
