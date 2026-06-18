"""Tests for bond SMARTS extraction helpers in 04_extract_bond_smarts.py.

Covers:
- ``atom_smarts``: correct SMARTS encoding for aromaticity and ring membership
- ``get_bond_smarts`` (radius 0): element + aromaticity + ring + bond type,
  canonical ordering (A-B == B-A), correct bond type symbols
- ``get_bond_smarts`` (radius >= 1): local subgraph expansion via
  FindAtomEnvironmentOfRadiusN + MolFragmentToSmarts
- ``get_frag_mw``: exact MW for known fragments including broken-bond
  dummy-atom correction for implicit H counts
- ``convert_mask``: packed h_formulae_idx → boolean h-delta validity mask
"""

import importlib.util
from pathlib import Path

import pytest
import torch as th
from rdkit import Chem
from rdkit.Chem import Descriptors

# ---------------------------------------------------------------------------
# Import helpers directly from the preproc script (not a package module)
# ---------------------------------------------------------------------------
_SCRIPT = (
    Path(__file__).parent.parent / "preproc_scripts" / "no_dag_exp" / "04_extract_bond_smarts.py"
)
_spec = importlib.util.spec_from_file_location("bond_smarts_script", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

atom_smarts = _mod.atom_smarts
get_bond_smarts = _mod.get_bond_smarts
get_frag_mw = _mod.get_frag_mw
convert_mask = _mod.convert_mask


# ---------------------------------------------------------------------------
# atom_smarts
# ---------------------------------------------------------------------------


class TestAtomSmarts:
    def test_aromatic_ring_carbon(self):
        # benzene — all atoms aromatic and in a ring
        mol = Chem.MolFromSmiles("c1ccccc1")
        c = mol.GetAtomWithIdx(0)
        assert atom_smarts(c) == "[#6;a;R]"

    def test_aliphatic_chain_carbon(self):
        mol = Chem.MolFromSmiles("CCO")
        c = mol.GetAtomWithIdx(0)
        assert atom_smarts(c) == "[#6;!a;!R]"

    def test_aliphatic_ring_carbon(self):
        mol = Chem.MolFromSmiles("C1CCCC1")  # cyclopentane
        c = mol.GetAtomWithIdx(0)
        assert atom_smarts(c) == "[#6;!a;R]"

    def test_chain_oxygen(self):
        mol = Chem.MolFromSmiles("CCO")
        o = mol.GetAtomWithIdx(2)
        assert atom_smarts(o) == "[#8;!a;!R]"

    def test_chain_nitrogen(self):
        mol = Chem.MolFromSmiles("CCN")
        n = mol.GetAtomWithIdx(2)
        assert atom_smarts(n) == "[#7;!a;!R]"


# ---------------------------------------------------------------------------
# get_bond_smarts — radius 0
# ---------------------------------------------------------------------------


class TestGetBondSmartsR0:
    def test_aromatic_bond(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        s = get_bond_smarts(mol, 0, 1, env_radius=0)
        assert s == "[#6;a;R]:[#6;a;R]"

    def test_single_bond_cc_chain(self):
        mol = Chem.MolFromSmiles("CC")
        s = get_bond_smarts(mol, 0, 1, env_radius=0)
        assert s == "[#6;!a;!R]-[#6;!a;!R]"

    def test_single_bond_co(self):
        mol = Chem.MolFromSmiles("CCO")
        # bond between C(idx=1) and O(idx=2)
        s = get_bond_smarts(mol, 1, 2, env_radius=0)
        assert s == "[#6;!a;!R]-[#8;!a;!R]"

    def test_double_bond(self):
        mol = Chem.MolFromSmiles("C=O")
        s = get_bond_smarts(mol, 0, 1, env_radius=0)
        assert s == "[#6;!a;!R]=[#8;!a;!R]"

    def test_triple_bond(self):
        mol = Chem.MolFromSmiles("C#N")
        s = get_bond_smarts(mol, 0, 1, env_radius=0)
        assert s == "[#6;!a;!R]#[#7;!a;!R]"

    def test_canonical_ordering_commutative(self):
        # get_bond_smarts(u, v) must equal get_bond_smarts(v, u)
        mol = Chem.MolFromSmiles("CCO")
        assert get_bond_smarts(mol, 1, 2, env_radius=0) == get_bond_smarts(mol, 2, 1, env_radius=0)
        assert get_bond_smarts(mol, 0, 1, env_radius=0) == get_bond_smarts(mol, 1, 0, env_radius=0)

    def test_ring_cc_single(self):
        mol = Chem.MolFromSmiles("C1CCCC1")  # cyclopentane
        s = get_bond_smarts(mol, 0, 1, env_radius=0)
        assert s == "[#6;!a;R]-[#6;!a;R]"

    def test_ring_vs_chain_distinguished(self):
        # ring C-C and chain C-C must produce different SMARTS
        ring_mol = Chem.MolFromSmiles("C1CCCC1")
        chain_mol = Chem.MolFromSmiles("CC")
        ring_smarts = get_bond_smarts(ring_mol, 0, 1, env_radius=0)
        chain_smarts = get_bond_smarts(chain_mol, 0, 1, env_radius=0)
        assert ring_smarts != chain_smarts


# ---------------------------------------------------------------------------
# get_bond_smarts — radius >= 1
# ---------------------------------------------------------------------------


class TestGetBondSmartsR1:
    def test_returns_non_empty_string(self):
        mol = Chem.MolFromSmiles("CCO")
        s = get_bond_smarts(mol, 0, 1, env_radius=1)
        assert isinstance(s, str) and len(s) > 0

    def test_parseable_smarts(self):
        # RDKit must be able to parse what it generated
        mol = Chem.MolFromSmiles("c1ccccc1CCO")
        for u, v in [(0, 1), (5, 6), (6, 7)]:
            s = get_bond_smarts(mol, u, v, env_radius=1)
            assert Chem.MolFromSmarts(s) is not None, f"Unparseable SMARTS: {s!r}"

    def test_r1_captures_more_atoms_than_r0(self):
        # radius-1 SMARTS for a non-terminal bond should include neighbors on both sides
        # → more atom tokens than radius-0 (which has exactly 2)
        mol = Chem.MolFromSmiles("CCCCC")
        # middle bond (idx 1–2) has one neighbor on each side
        s0 = get_bond_smarts(mol, 1, 2, env_radius=0)
        s1 = get_bond_smarts(mol, 1, 2, env_radius=1)
        # count '[' as a proxy for number of atom SMARTS tokens
        assert s1.count("[") > s0.count("["), (
            f"r1 SMARTS should have more atoms than r0; got r0={s0!r}, r1={s1!r}"
        )

    def test_r2_at_least_as_rich_as_r1(self):
        mol = Chem.MolFromSmiles("CCCCC")
        s1 = get_bond_smarts(mol, 1, 2, env_radius=1)
        s2 = get_bond_smarts(mol, 1, 2, env_radius=2)
        assert len(s2) >= len(s1)


# ---------------------------------------------------------------------------
# get_frag_mw
# ---------------------------------------------------------------------------


class TestGetFragMw:
    def _ethanol(self):
        return Chem.MolFromSmiles("CCO")

    def test_full_molecule_matches_rdkit(self):
        mol = self._ethanol()
        expected = Descriptors.ExactMolWt(Chem.AddHs(mol))
        mw = get_frag_mw(mol, [0, 1, 2])
        assert mw == pytest.approx(expected, abs=1e-3)

    def test_methyl_fragment_from_ethanol(self):
        # atom 0 (CH3 group) with 1 broken bond to atom 1
        mol = self._ethanol()
        mw = get_frag_mw(mol, [0])
        # C (12.000) + 3 H (3 * 1.00783) = 15.023 Da
        assert mw == pytest.approx(15.023, abs=1e-2)

    def test_hydroxyl_fragment_from_ethanol(self):
        # atom 2 (OH group) with 1 broken bond to atom 1
        mol = self._ethanol()
        mw = get_frag_mw(mol, [2])
        # O (15.9949) + 1 H (1.00783) = 17.003 Da
        assert mw == pytest.approx(17.003, abs=1e-2)

    def test_ethyl_fragment_from_ethanol(self):
        # atoms 0+1 with 1 broken bond to atom 2 (O)
        mol = self._ethanol()
        mw = get_frag_mw(mol, [0, 1])
        # C2H5 = 2*12 + 5*1.00783 = 29.039 Da
        assert mw == pytest.approx(29.039, abs=1e-2)

    def test_single_middle_atom_two_broken_bonds(self):
        # propane CCC — middle C (idx 1) has 2 broken bonds → CH2 fragment
        mol = Chem.MolFromSmiles("CCC")
        mw = get_frag_mw(mol, [1])
        # CH2: C (12.000) + 2 H (2 * 1.00783) = 14.016 Da
        assert mw == pytest.approx(14.016, abs=1e-2)


# ---------------------------------------------------------------------------
# convert_mask
# ---------------------------------------------------------------------------


class TestConvertMask:
    def _make(self, rows: list[list[int]]) -> th.Tensor:
        return th.tensor(rows, dtype=th.long)

    def test_all_zeros_only_center_is_one(self):
        # input shape (1, 9) for num_hs=4
        x = self._make([[0] * 9])
        out = convert_mask(x)
        assert out.shape == (1, 9)
        # only position 4 (num_h) stays 1; all others become 0
        assert out[0, 4].item() == 1
        assert out[0, :4].sum().item() == 0
        assert out[0, 5:].sum().item() == 0

    def test_all_nonzero_all_ones(self):
        x = self._make([[1] * 9])
        out = convert_mask(x)
        assert out.shape == (1, 9)
        assert out.sum().item() == 9

    def test_selective_positions(self):
        # odd positions [1,3] set → output positions 0,1 (first half) become 1
        # even positions [2,4] zero → output positions 5,6 remain 0
        # num_h = 9//2 = 4; odd=[1,3,5,7], even=[2,4,6,8]
        x = self._make([[0, 1, 0, 1, 0, 0, 0, 0, 0]])
        out = convert_mask(x)
        # positions 0,1 should be 1 (from odd[0]=v1=1, odd[1]=v3=1)
        assert out[0, 0].item() == 1
        assert out[0, 1].item() == 1
        # positions 2,3 should be 0 (from odd[2]=v5=0, odd[3]=v7=0)
        assert out[0, 2].item() == 0
        assert out[0, 3].item() == 0
        # center always 1
        assert out[0, 4].item() == 1
        # positions 5-8 all 0 (even positions all 0)
        assert out[0, 5:].sum().item() == 0

    def test_batch_independent(self):
        # two rows: first all-zero, second all-nonzero
        x = self._make([[0] * 9, [1] * 9])
        out = convert_mask(x)
        assert out.shape == (2, 9)
        assert out[0, 4].item() == 1
        assert out[0, :4].sum().item() == 0
        assert out[1].sum().item() == 9
