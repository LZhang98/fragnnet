"""Tests for H-cap / ve_arr correctness with formally charged atoms.

Covers the fix in extract_mol_info that adjusts ve_value for cationic atoms
with no H (e.g. [N+] in nitro, N-oxide) to prevent the H-cap from being
inflated by the formal charge.

Tests:
1. ve_arr for [N+] in nitro is 3 (neutral N valence), not 4.
2. H-cap for an isolated NO2 fragment (N-C bond cut) is 2, not 3 (pre-fix).
3. Full DAG for nitrobenzene does not contain the phantom H3NO2 formula.
4. Protonated amine ([NH3+] in zwitterion) is unaffected: ve remains 4.
5. N-oxide ([N+][O-]) N+ ve is adjusted from 4 to 3.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from fragnnet.frag.compute_frags import update_bonds
from fragnnet.utils.frag_utils import compute_cc_h_cap, extract_mol_info


# ---------------------------------------------------------------------------
# 1. ve_arr for [N+] in nitro is adjusted from 4 to 3
# ---------------------------------------------------------------------------


def test_nitro_nitrogen_ve_is_three():
    """[N+] in nitro group must have ve=3 after formal-charge adjustment.

    Nitrobenzene: without fix GetTotalValence()=4 → ve=4.
    With fix: FC=+1, Hs=0 → ve = 4 - 1 = 3.
    """
    smiles = "c1ccc([N+](=O)[O-])cc1"
    mol_d = extract_mol_info(smiles)

    mol = Chem.MolFromSmiles(smiles)
    nitro_n_idx = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "N" and a.GetFormalCharge() == 1
    )
    assert mol_d["ve_arr"][nitro_n_idx] == 3, (
        f"Expected ve=3 for [N+] in nitro, got {mol_d['ve_arr'][nitro_n_idx]}"
    )


# ---------------------------------------------------------------------------
# 2. H-cap for isolated NO2 fragment is 2 (reduced from pre-fix value of 3)
# ---------------------------------------------------------------------------


def test_h_cap_isolated_no2_fragment():
    """After cutting the N-C(ring) bond, the NO2 fragment H-cap must be 2.

    Fragment atoms: {N+, =O, O-}.  With fix:
      N+:  ve=3, sbond=2 → diff=1
      =O:  ve=2, sbond=1 → diff=1  (residual pi-bond excess, same as any C=O)
      O-:  ve=1, sbond=1 → diff=0
      cap = 1 + 1 + 0 = 2

    Before fix N+ contributed diff=2 (ve=4), giving cap=3.
    The remaining +1 from =O is an inherent property of the loose-cap design
    (shared with all double-bonded atoms, not specific to formal charges).
    """
    smiles = "c1ccc([N+](=O)[O-])cc1"
    mol_d = extract_mol_info(smiles)

    mol = Chem.MolFromSmiles(smiles)
    nitro_n_idx = next(
        a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetFormalCharge() == 1
    )
    o_double_idx = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "O" and a.GetFormalCharge() == 0
    )
    o_minus_idx = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "O" and a.GetFormalCharge() == -1
    )

    no2_atom_ids = np.array([nitro_n_idx, o_double_idx, o_minus_idx], dtype=np.int32)

    # Recompute sbond_arr for this fragment only (mirrors compute_dags internals)
    updated_sbond, _ = update_bonds(
        no2_atom_ids,
        mol_d["sbond_arr"],
        mol_d["bond_mask_arr"],
        mol_d["bonds"],
        mol_d["atoms_to_bonds"],
    )

    h_cap = compute_cc_h_cap(no2_atom_ids, mol_d["ve_arr"], updated_sbond, 0)
    assert h_cap == 2, (
        f"NO2 fragment H-cap should be 2 after fix (was 3 before); got {h_cap}"
    )


# ---------------------------------------------------------------------------
# 3. Full DAG for nitrobenzene has no phantom H3NO2 formula
# ---------------------------------------------------------------------------


def test_nitrobenzene_dag_no_h3no2():
    """Nitrobenzene DAG must not contain the phantom H3NO2 fragment formula.

    H3NO2 arises when the N+ valence inflation is not corrected (cap=3 → H=3
    for isolated NO2). After the fix cap=2, so only H=0,1,2 are valid and
    H3NO2 is absent.
    """
    from fragnnet.utils.frag_utils import compute_dags

    smiles = "c1ccc([N+](=O)[O-])cc1"
    mol_d = extract_mol_info(smiles)
    dag_d = compute_dags(
        mol_d,
        max_depth=3,
        h_prior=False,
        max_h_transfer=4,
        frag_max_time=30,
        isotopes=False,
        nb_isomorphic=True,
        wl_max_iterations=3,
    )

    all_formulas = set(dag_d["idx_to_formula"].values())
    assert "H3NO2" not in all_formulas, (
        "Phantom formula H3NO2 found in DAG — N+ ve adjustment not applied"
    )


# ---------------------------------------------------------------------------
# 4. Protonated amine in a zwitterion ([NH3+]) is unaffected: ve stays 4
# ---------------------------------------------------------------------------


def test_protonated_amine_ve_unaffected():
    """[NH3+] with 3 H atoms must NOT have its ve adjusted.

    Glycine zwitterion [NH3+]CC(=O)[O-] has net charge 0.
    The N+ has FC=+1 but Hs=3, so the guard (cur_num_hs == 0) blocks the
    adjustment → ve stays at GetTotalValence() = 4, correctly reflecting
    the 3 H bonds plus 1 heavy bond.
    """
    # Pass mol directly to skip ml_standardize (which would neutralize the zwitterion)
    mol = Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")
    mol_d = extract_mol_info(mol)

    n_idx = next(i for i, e in enumerate(mol_d["elems"]) if e == "N")
    assert mol_d["ve_arr"][n_idx] == 4, (
        f"[NH3+] ve must stay 4 (3H + 1 heavy bond = 4 total valence), got {mol_d['ve_arr'][n_idx]}"
    )


# ---------------------------------------------------------------------------
# 5. N-oxide ([N+][O-]) N+ ve is adjusted from 4 to 3
# ---------------------------------------------------------------------------


def test_n_oxide_nitrogen_ve_is_three():
    """Pyridine N-oxide: [n+]([O-]) nitrogen must have ve=3 after adjustment.

    The N+ has FC=+1 and no H.  GetTotalValence()=4 is reduced to 3 by the fix.
    Pass mol directly to bypass the standardizer.
    """
    mol = Chem.MolFromSmiles("c1cc[n+]([O-])cc1")
    assert mol is not None, "RDKit could not parse pyridine N-oxide SMILES"
    mol_d = extract_mol_info(mol)

    n_idx = next(i for i, e in enumerate(mol_d["elems"]) if e == "N")
    assert mol_d["ve_arr"][n_idx] == 3, (
        f"N-oxide [N+] ve must be 3, got {mol_d['ve_arr'][n_idx]}"
    )
