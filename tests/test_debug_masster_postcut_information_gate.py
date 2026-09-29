"""Tests for the post-cut information gate: descriptors and the listwise correction scores."""

import numpy as np
import pytest
import torch as th
from rdkit import Chem

import scripts.debug.debug_masster_base_peak_tables as TABLES
import scripts.debug.debug_masster_postcut_information_gate as GATE


@pytest.fixture
def phenethylamine() -> "GATE.PostCutLookups":
    """NCCc1ccccc1: atom 0 the amine N, 1-2 the chain, 3-8 the ring (3 bears the chain)."""
    mol = Chem.MolFromSmiles("NCCc1ccccc1")
    return GATE.PostCutLookups(mol, TABLES.MoleculeChemistry(mol))


def _descriptors(look: "GATE.PostCutLookups", atoms: tuple) -> dict:
    inside = np.zeros((1, look.n_atoms), dtype=bool)
    inside[0, list(atoms)] = True
    out = GATE.fragment_descriptors(inside, look, np.zeros(look.n_atoms))
    return {k: float(v[0]) for k, v in out.items()}


def test_benzyl_side_of_the_chain_cut(phenethylamine: "GATE.PostCutLookups") -> None:
    # Cutting C1-C2 keeps the benzyl group: one retained sp3 benzylic cut site, the lost cut site
    # is the carbon next to the amine, the ring is intact and the basic site is lost.
    d = _descriptors(phenethylamine, (2, 3, 4, 5, 6, 7, 8))
    assert d["n_cuts"] == 1 and d["bond_alkyl_CC"] == 1
    assert d["cut_in_C_sp3"] == 1 and d["cut_out_C_sp3"] == 1
    assert d["stab_in_benzylic"] == 1 and d["stab_out_alpha_hetero"] == 1
    assert d["kept_bonds"] == 7 and d["cyclomatic"] == 1
    assert d["rings_kept"] == 1 and d["aromatic_rings_kept"] == 1 and d["rings_broken"] == 0
    assert d["keeps_base"] == 0 and d["base_distance_missing"] == 1
    assert d["base_distance"] == GATE.DISTANCE_CAP


def test_amine_side_of_the_benzylic_cut(phenethylamine: "GATE.PostCutLookups") -> None:
    d = _descriptors(phenethylamine, (0, 1, 2))
    assert d["bond_benzylic_CC"] == 1 and d["cut_out_C_ar"] == 1
    assert d["keeps_base"] == 1 and d["base_kept_aliphatic_amine"] == 1
    # The retained cut site (C2) is two bonds from the amine N.
    assert d["base_distance"] == 2 and d["base_distance_missing"] == 0
    assert d["loses_aromatic"] == 1 and d["conj_kept"] == 0
    assert d["n_C"] == 2 and d["n_N"] == 1 and d["n_H"] == 6


def test_ring_opening_is_counted(phenethylamine: "GATE.PostCutLookups") -> None:
    d = _descriptors(phenethylamine, (0, 1, 2, 3, 4, 5))
    assert d["ring_bonds_cut"] == 2 and d["aromatic_bonds_cut"] == 2
    assert d["rings_broken"] == 1 and d["aromatic_rings_broken"] == 1 and d["breaks_aromatic"] == 1


def test_channel_rdbe_and_precursor(phenethylamine: "GATE.PostCutLookups") -> None:
    inside = np.zeros((2, 9), dtype=bool)
    inside[0, [0, 1, 2]] = True
    inside[1, :] = True
    fragment = GATE.fragment_descriptors(inside, phenethylamine, np.zeros(9))
    x = GATE.channel_descriptors(fragment, np.array([0, 1, 1]), np.array([1, 0, -1]), 9)
    columns = list(GATE.DESCRIPTORS)
    # Ethylamine (C2H7N, the +1 H channel) is saturated; the intact precursor is C8H11N.
    assert x[0, columns.index("rdbe")] == pytest.approx(0.0)
    assert x[1, columns.index("rdbe")] == pytest.approx(4.0)
    np.testing.assert_array_equal(x[:, columns.index("is_precursor")], [0.0, 1.0, 0.0])
    assert x.shape == (3, len(GATE.DESCRIPTORS))


def test_segment_scores_match_numpy() -> None:
    logits = th.tensor([0.2, -1.0, 0.5, 1.5, 0.1], dtype=th.float64)
    tgt = th.tensor([0.1, 0.0, 0.7, 0.2, 0.6], dtype=th.float64)
    segment = th.tensor([0, 0, 0, 1, 1])
    norm = th.tensor([0.8, 0.7], dtype=th.float64)
    ce, cos, agree = GATE.segment_scores(logits.float(), tgt.float(), segment, norm.float(), 2)
    for s, rows in enumerate(([0, 1, 2], [3, 4])):
        z = logits[rows].numpy()
        p = np.exp(z - z.max())
        p /= p.sum()
        t = tgt[rows].numpy()
        assert ce[s].item() == pytest.approx(-(t / t.sum() * np.log(p)).sum(), rel=1e-5)
        assert cos[s].item() == pytest.approx(
            p @ t / (np.linalg.norm(p) * norm[s].item()), rel=1e-5
        )
    np.testing.assert_array_equal(agree.numpy(), [1.0, 0.0])


@pytest.mark.parametrize("hidden", [0, 8])
def test_correction_starts_as_identity(hidden: int) -> None:
    model = GATE.Correction(width=4, hidden=hidden, dropout=0.0)
    log_pred = th.randn(6)
    assert th.equal(model(log_pred, th.randn(6, 4)), log_pred)
