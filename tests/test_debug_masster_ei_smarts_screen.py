"""Scientific regression checks for the EI SMARTS screening probe."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from fragnnet.models.masster.path_selection.smarts_templates import _is_connected, formed_bond_count
from scripts.debug.ei.debug_masster_ei_smarts_screen import (
    ChannelCalculator,
    atom_composition,
    channel_support,
    coverage_delta,
    draw_null,
    isotope_offsets,
    rule_products,
    summarize,
)
from scripts.debug.smarts.debug_masster_smarts_rule_gate import edge_tuple

FIXTURES = [
    (
        "Cc1ccc(S(=O)(=O)NC(=O)c2ccccc2)cc1",
        "ei_nh_benzoyl_arylsulfonamide_o_transfer",
        {"C": 7, "H": 8, "O": 1},
        1,
        108,
    ),
    (
        "COc1ccc(C(=O)F)cc1",
        "anisoyl_fluoride_co_f_transfer",
        {"C": 7, "H": 7, "O": 1, "F": 1},
        0,
        126,
    ),
    (
        "Oc1ccc(N=Nc2ccccc2)cc1Br",
        "ei_para_hydroxyaryl_azo_n2_loss",
        {"C": 12, "H": 9, "O": 1, "Br": 1},
        0,
        248,
    ),
]


@pytest.mark.parametrize("smiles,name,expected,shift,mass", FIXTURES)
def test_literature_product_formula_and_conservation(smiles, name, expected, shift, mass):
    mol = Chem.MolFromSmiles(smiles)
    sides = rule_products(mol)[name]
    assert len(sides) == 1
    side = sides[0]
    complement = tuple(sorted(set(range(mol.GetNumAtoms())) - set(side)))
    assert atom_composition(mol, side, shift) == Counter(expected)
    assert atom_composition(mol, side, shift) + atom_composition(
        mol, complement, -shift
    ) == atom_composition(mol, tuple(range(mol.GetNumAtoms())))
    assert formed_bond_count(side, edge_tuple(mol)) == 1
    mono, isotopes = channel_support(mol, sides, ChannelCalculator(("[M]+",)))
    assert mass in mono
    if "Br" in expected:
        assert mass + 2 in isotopes


@pytest.mark.parametrize(
    "smiles",
    [
        "CN(C(=O)c1ccccc1)S(=O)(=O)c1ccccc1",  # N-methyl favors the other branch.
        "CC(=O)NS(=O)(=O)c1ccccc1",  # Not the benzoyl series screened here.
        "O=C(F)c1ccccc1",  # No same-ring methoxy group.
        "COc1ccc(C(=O)Cl)cc1",  # Fluorine evidence cannot admit chloride.
        "COc1ccc(N=Nc2ccccc2)cc1",  # O-methyl, not para phenol.
        "Oc1ccccc1N=Nc1ccccc1",  # Ortho, not para phenol.
        "Oc1ccc(N=Nc2ccccn2)cc1",  # Heteroaryl is outside the phenyl series.
        "Oc1ccc(N=Nc2ccccc2)c2ccccc12",  # Fused aryl is outside the series.
        "Oc1ccc(NNc2ccccc2)cc1",  # Hydrazine, not azo.
        "COc1ccc(C(=O)F)cc1.[Na+]",  # No salt/adduct extrapolation.
    ],
)
def test_close_negative_controls(smiles):
    assert rule_products(Chem.MolFromSmiles(smiles)) == {}


@pytest.mark.parametrize("smiles", ["COc1ccccc1C(=O)F", "COc1cccc(C(=O)F)c1"])
def test_other_anisoyl_positions(smiles):
    assert "anisoyl_fluoride_co_f_transfer" in rule_products(Chem.MolFromSmiles(smiles))


def test_isotope_overlap_is_not_incremental_coverage():
    assert isotope_offsets(Counter({"Cl": 1})).tolist() == [0, 2]
    assert isotope_offsets(Counter({"F": 1})).tolist() == [0]
    mz, weights = np.array([100, 102, 104]), np.array([0.5, 0.3, 0.2])
    assert coverage_delta(mz, weights, {100, 102}, {102, 104}) == pytest.approx(0.2)
    assert coverage_delta(mz, weights, {100, 102, 104}, {102, 104}) == 0


def test_null_is_disconnected_size_matched_distinct_and_seeded():
    mol = Chem.MolFromSmiles(FIXTURES[0][0])
    sides = next(iter(rule_products(mol).values()))
    draws, missing = draw_null(mol, sides, np.random.default_rng(17))
    assert missing == 0
    assert [len(side) for side in draws] == [len(side) for side in sides]
    assert all(side not in sides and not _is_connected(side, edge_tuple(mol)) for side in draws)
    assert (draws, missing) == draw_null(mol, sides, np.random.default_rng(17))


@pytest.mark.parametrize("smiles,name,expected,shift,mass", FIXTURES)
def test_nominal_support_matches_production(smiles, name, expected, shift, mass):
    mol = Chem.MolFromSmiles(smiles)
    sides = rule_products(mol)[name]
    calc = ChannelCalculator(("[M]+",))
    mono, _ = channel_support(mol, sides, calc)
    calc.model.mass_mode = "nominal"
    bonds = list(zip(*edge_tuple(mol), strict=True))
    cuts = [sum((a in side) != (b in side) for a, b in bonds) for side in sides]
    mz, valid = calc(mol, sides, cuts, [1] * len(sides), 0)
    assert mono == set(mz[valid].astype(int))


def test_transferred_oxygen_is_the_carbonyl_oxygen():
    mol = Chem.MolFromSmiles(FIXTURES[0][0])
    side = rule_products(mol)[FIXTURES[0][1]][0]
    oxygens = [mol.GetAtomWithIdx(i) for i in side if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
    assert len(oxygens) == 1
    assert oxygens[0].GetNeighbors()[0].GetSymbol() == "C"
    assert (
        mol.GetBondBetweenAtoms(
            oxygens[0].GetIdx(), oxygens[0].GetNeighbors()[0].GetIdx()
        ).GetBondType()
        == Chem.BondType.DOUBLE
    )


def test_summary_weights_molecules_not_spectrum_replicates():
    rows = pd.DataFrame(
        {
            "rule": ["fixture"] * 5,
            "split": ["val"] * 5,
            "mol_id": [1, 1, 1, 2, 3],
            "delta_mono": [1.0, 1.0, 1.0, 0.0, 0.0],
            "delta": [1.0, 1.0, 1.0, 0.0, 0.0],
            "null": [0.0] * 5,
            "diff": [1.0, 1.0, 1.0, 0.0, 0.0],
            "n_products": [1] * 5,
            "n_null_missing": [0] * 5,
            "nominal_overlap": [0.0] * 5,
        }
    )
    summary = summarize(rows, 17).iloc[0]
    assert summary["delta"] == pytest.approx(1 / 3)
    assert summary["diff"] == pytest.approx(1 / 3)
    assert summary["molecules"] == 3
    assert summary["spectra"] == 5
    assert summary["molecules_ge1pct"] == 1
