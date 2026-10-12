"""Tests for scripts/misc/export_annotated_mgf.py."""

import sys

import numpy as np
import pandas as pd
import pytest
import torch as th
from pyteomics import mgf
from rdkit import Chem

import scripts.misc.export_annotated_mgf as EXPORT
from fragnnet.models.fragnnet.frag_utils import MASK_SIZE, th_mask_to_long

VANILLYL_THIAZOLIUM = "COc1cc(C[n+]2csc(CCO)c2C)ccc1O"
METHOXYETHANE = "CCOC"  # atoms: C0 C1 O2 C3


def _ccs(atom_sets: list[tuple[int, ...]]) -> th.Tensor:
    mask = th.zeros((len(atom_sets), MASK_SIZE), dtype=th.long)
    for row, atoms in enumerate(atom_sets):
        mask[row, list(atoms)] = 1
    return th_mask_to_long(mask)


def _pred_row(ints, formulas, adducts, joint, atom_sets) -> pd.Series:
    """A run_inference row with one formula row per peak; joint is (formula, node, P(n|f))."""
    return pd.Series(
        {
            "pred_mzs": th.tensor([100.0 + k for k in range(len(ints))]),
            "pred_ints": th.tensor(ints, dtype=th.float32),
            "frag_formula_peak_idxs": th.arange(len(ints)),
            "pred_formula_strs": formulas,
            "pred_formula_adducts": adducts,
            "pred_joint_formula_idxs": th.tensor([j[0] for j in joint]),
            "pred_joint_node_idxs": th.tensor([j[1] for j in joint]),
            "pred_formula_node_probs": th.tensor([j[2] for j in joint]),
            "pred_node_ccs": _ccs(atom_sets),
        }
    )


@pytest.fixture
def methoxyethane_row() -> pd.Series:
    # Peak 0: intact [M+H]+. Peak 1: C2H5O from either end, P(n|f) 0.3 / 0.7. Peak 2: the
    # folded row of a methyl that lost all its hydrogens (ion formula, empty adduct).
    # Peak 3 sits under 1% of the base peak.
    return _pred_row(
        ints=[500.0, 1000.0, 200.0, 5.0],
        formulas=["C3H8O", "C2H5O", "C", "CH3"],
        adducts=["+H", "+H", "", "+H"],
        joint=[(0, 0, 1.0), (1, 1, 0.3), (1, 2, 0.7), (2, 3, 1.0), (3, 3, 1.0)],
        atom_sets=[(0, 1, 2, 3), (0, 1, 2), (1, 2, 3), (3,)],
    )


def test_fragment_smiles_keeps_parent_hydrogens_and_charge():
    mol = Chem.MolFromSmiles(VANILLYL_THIAZOLIUM)
    vanillyl = (0, 1, 2, 3, 4, 5, 15, 16, 17, 18)  # methoxy-hydroxy-benzyl, CH2 at atom 5
    thiazolium = (6, 7, 8, 9, 10, 11, 12, 13, 14)  # ring N+ at atom 6
    assert EXPORT.fragment_smiles(mol, vanillyl) == "[CH2]c1ccc(O)c(OC)c1"
    assert EXPORT.fragment_smiles(mol, thiazolium) == "CC1=C(CCO)SC=[N+]1"
    whole = tuple(range(mol.GetNumAtoms()))
    assert EXPORT.fragment_smiles(mol, whole) == Chem.CanonSmiles(VANILLYL_THIAZOLIUM)


@pytest.mark.parametrize(
    "delta, expected",
    [
        ("", {}),
        ("+H", {"H": 1}),
        ("-H2", {"H": -2}),
        ("-H+Na", {"H": -1, "Na": 1}),
        ("-CHO2", {"C": -1, "H": -1, "O": -2}),
    ],
)
def test_parse_signed_delta(delta, expected):
    parsed = {el: n for el, n in EXPORT.parse_signed_delta(delta).items() if n}
    assert parsed == expected


def test_parse_signed_delta_rejects_unsigned():
    with pytest.raises(ValueError, match="not a signed delta"):
        EXPORT.parse_signed_delta("H1")


def test_annotate_spectrum_resolves_delta_and_top_fragment(methoxyethane_row):
    mol = Chem.MolFromSmiles(METHOXYETHANE)
    peaks = EXPORT.annotate_spectrum([methoxyethane_row], mol, "[M+H]+", 0.01)

    assert [p["mz"] for p in peaks] == [100.0, 101.0, 102.0]  # 5/1000 peak dropped
    assert [p["intensity"] for p in peaks] == pytest.approx([500.0, 1000.0, 200.0])
    assert [p["formula"] for p in peaks] == ["C3H9O+", "C2H6O+", "C+"]
    intact, ethoxy, methyl = (p["fragments"] for p in peaks)
    assert intact == [("CCOC", "+H", pytest.approx(1.0))]
    # The heavier joint cell wins; both ends are kept with their shares.
    assert ethoxy == [
        ("[CH2]OC", "+H", pytest.approx(0.7)),
        ("CC[O]", "+H", pytest.approx(0.3)),
    ]
    # Folded row: the formula string is already the ion, so the delta is the full H loss.
    assert methyl == [("[CH3]", "-H3", pytest.approx(1.0))]


def test_annotate_spectrum_averages_predictions(methoxyethane_row):
    mol = Chem.MolFromSmiles(METHOXYETHANE)
    other = methoxyethane_row.copy()
    other["pred_ints"] = th.tensor([1000.0, 500.0, 200.0, 5.0])
    other["pred_formula_node_probs"] = th.tensor([1.0, 0.9, 0.1, 1.0, 1.0])
    peaks = EXPORT.annotate_spectrum([methoxyethane_row, other], mol, "[M+H]+", 0.01)

    # Sum-normalized weights: row one 500/1705, 1000/1705; row two 1000/1705, 500/1705.
    assert [p["intensity"] for p in peaks[:2]] == pytest.approx([1000.0, 1000.0])
    w1, w2 = 1000.0 / 1705.0, 500.0 / 1705.0
    shares = {smi: share for smi, _delta, share in peaks[1]["fragments"]}
    assert shares["CC[O]"] == pytest.approx((w1 * 0.3 + w2 * 0.9) / (w1 + w2))
    assert shares["[CH2]OC"] == pytest.approx((w1 * 0.7 + w2 * 0.1) / (w1 + w2))


def test_annotate_spectrum_rejects_atoms_that_cannot_build_the_ion(methoxyethane_row):
    mol = Chem.MolFromSmiles(METHOXYETHANE)
    bad = methoxyethane_row.copy()
    bad["pred_node_ccs"] = _ccs([(0, 1, 2, 3), (0, 1), (1, 2, 3), (3,)])  # C2H5 for C2H5O
    with pytest.raises(ValueError, match="do not build ion"):
        EXPORT.annotate_spectrum([bad], mol, "[M+H]+", 0.01)


def test_annotate_spectrum_keeps_sodium_in_delta():
    mol = Chem.MolFromSmiles(METHOXYETHANE)
    row = _pred_row(
        ints=[1000.0],
        formulas=["C2H5O"],
        adducts=["+Na"],
        joint=[(0, 0, 1.0)],
        atom_sets=[(1, 2, 3)],
    )
    (peak,) = EXPORT.annotate_spectrum([row], mol, "[M+Na]+", 0.01)
    assert peak["formula"] == "C2H5NaO+"
    assert peak["fragments"] == [("[CH2]OC", "+Na", pytest.approx(1.0))]


def test_main_writes_annotated_plain_and_table(tmp_path, monkeypatch, methoxyethane_row):
    pred_dp = tmp_path / "pred"
    pred_dp.mkdir()
    rows = []
    for spec_id in (0, 1):
        row = methoxyethane_row.copy()
        row["spec_id"] = spec_id
        rows.append(row)
    pd.DataFrame(rows).to_pickle(pred_dp / "predictions_chunk_0.pkl")
    mol = Chem.MolFromSmiles(METHOXYETHANE)
    pd.DataFrame(
        [
            {
                "mol_id": 0,
                "smiles": METHOXYETHANE,
                "mol": mol,
                "formula": "C3H8O",
                "input_mol_id": "methoxyethane",
            }
        ]
    ).to_pickle(tmp_path / "mol_df.pkl")
    prec_mz = 61.0648
    pd.DataFrame(
        [
            {
                "spec_id": 0,
                "mol_id": 0,
                "prec_type": "[M+H]+",
                "inst_type": "ORB",
                "ace": np.nan,
                "nce": 30.0,
                "prec_mz": prec_mz,
                "frag_mode": "CID",
            },
            {
                "spec_id": 1,
                "mol_id": 0,
                "prec_type": "[M+H]+",
                "inst_type": "QTOF",
                "ace": 20.0,
                "nce": np.nan,
                "prec_mz": prec_mz,
                "frag_mode": "CID",
            },
        ]
    ).to_pickle(tmp_path / "spec_df.pkl")
    out = tmp_path / "pred.mgf"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_annotated_mgf.py",
            "--pred_dp",
            str(pred_dp),
            "--mol_fp",
            str(tmp_path / "mol_df.pkl"),
            "--spec_fp",
            str(tmp_path / "spec_df.pkl"),
            "--output_fp",
            str(out),
        ],
    )
    EXPORT.main()

    text = out.read_text()
    assert "TITLE=methoxyethane_[M+H]+_ORB_NCE30.0" in text
    assert "FRAGMENTATION_MODE=HCD\nCOLLISION_ENERGY=30.0% NCE" in text
    assert "FRAGMENTATION_MODE=CID\nCOLLISION_ENERGY=20 eV" in text
    assert '101.0000 1000.0 "[CH2]OC +H"' in text
    assert '102.0000 200.0 "[CH3] -H3"' in text

    plain = list(mgf.read(str(tmp_path / "pred_plain.mgf")))
    assert len(plain) == 2
    np.testing.assert_allclose(plain[0]["m/z array"], [100.0, 101.0, 102.0])
    np.testing.assert_allclose(plain[0]["intensity array"], [500.0, 1000.0, 200.0])

    table = pd.read_csv(tmp_path / "pred_annotations.csv", keep_default_na=False)
    assert len(table) == 2 * 4  # two spectra x (1 + 2 + 1) fragments
    ethoxy = table[(table.spec_id == 0) & (table.mz == 101.0)]
    assert ethoxy.frag_smiles.tolist() == ["[CH2]OC", "CC[O]"]
    assert ethoxy.share.tolist() == pytest.approx([0.7, 0.3])
    assert set(table.ion_formula) == {"C3H9O+", "C2H6O+", "C+"}


def test_load_predictions_names_missing_annotation_columns(tmp_path, methoxyethane_row):
    row = methoxyethane_row.drop("pred_node_ccs")
    row["spec_id"] = 0
    pd.DataFrame([row]).to_pickle(tmp_path / "predictions_chunk_0.pkl")
    with pytest.raises(ValueError, match="pred_node_ccs"):
        EXPORT.load_predictions(str(tmp_path))
