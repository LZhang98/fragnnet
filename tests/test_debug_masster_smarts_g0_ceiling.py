"""Invariants of the G0 coverage-ceiling gate: unit conversion, channels and union."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch as th
from rdkit import Chem

import scripts.debug.smarts.debug_masster_smarts_g0_ceiling as G0
from fragnnet.models.masster.masster_templates import (
    build_masster_templates,
    filter_masster_templates,
    formed_bond_count,
)
from tests.test_masster_model import _build_batch, _build_model
from tests.test_masster_smarts_pathway import _static_graph

TOL = (1e-5, 200.0)
NEG = "[M-H]-"
NEG_INDEX = 1
PRIMARY_SULFONAMIDE = "NS(=O)(=O)c1ccccc1"
N_PHENYL_SULFONAMIDE = "O=S(=O)(Nc1ccccc1)c1ccccc1"
N_ACYL_SULFONAMIDE = "CC(=O)NS(=O)(=O)c1ccccc1"
BIS_PRIMARY_SULFONAMIDE = "NS(=O)(=O)c1ccc(S(N)(=O)=O)cc1"


def _templates(mol: Chem.Mol) -> tuple:
    return filter_masster_templates(
        build_masster_templates(mol), precursor_type=NEG, fragmentation_mode=None
    )


# 1. L2 conversion on a toy spectrum.


def test_eval_weights_follow_the_eval_target() -> None:
    peaks = [(300.0, 20.0), (100.0, 50.0), (1500.0, 40.0), (200.0, 0.0), (150.0, 30.0)]
    mzs, weights = G0.eval_weights(peaks)

    # mz >= 1500 and non-positive intensities are dropped; the rest is L1-normalised.
    assert mzs.tolist() == [100.0, 150.0, 300.0]
    assert weights.tolist() == pytest.approx([0.5, 0.3, 0.2])
    assert weights.sum() == pytest.approx(1.0)
    empty_mzs, empty_weights = G0.eval_weights([(2000.0, 5.0)])
    assert empty_mzs.size == empty_weights.size == 0


def test_covered_mask_uses_instrument_tolerance_with_floor() -> None:
    channels = np.array([100.0, 500.0])
    peaks = np.array([100.0015, 100.003, 500.004, 500.006])

    # 1e-5 * max(m/z, 200): 0.002 Da at 100, 0.005 Da at 500.
    assert G0.covered_mask(peaks, channels, TOL).tolist() == [True, False, True, False]
    assert G0.covered_mask(peaks, channels, (2e-5, 750.0)).tolist() == [True, True, True, True]
    assert G0.covered_mask(peaks, np.zeros(0), TOL).tolist() == [False] * 4


def test_l2_rise_on_a_toy_spectrum() -> None:
    weights = np.array([0.5, 0.3, 0.2])
    norm = float(np.sum(weights**2))  # 0.38
    covered = 0.5**2  # the base peak is in the pool
    new = 0.3**2  # the second peak is newly hit

    rise = G0.l2_rise(norm, covered, new)

    assert norm == pytest.approx(0.38)
    assert rise == pytest.approx(np.sqrt(0.34 / 0.38) - np.sqrt(0.25 / 0.38))
    # The pool-free bounds bracket the rise: dA / (2N) <= rise <= sqrt(dA / N).
    assert new / (2 * norm) <= rise <= np.sqrt(new / norm)
    assert G0.l2_rise(norm, covered, 0.0) == 0.0
    assert G0.l2_rise(0.0, 0.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="exceeds the norm"):
        G0.l2_rise(norm, covered, 0.5)


def test_whole_split_units_count_spectra_without_new_hits_as_zero() -> None:
    # Four spectra of two molecules; only one has a new hit.
    rows = pd.DataFrame(
        {
            "mol_id": [1, 1, 2, 2],
            "N": [0.38, 0.38, 0.5, 0.5],
            "A0": [0.25, 0.25, 0.4, np.nan],
            "dA_unique": [0.09, 0.0, 0.0, 0.0],
            "dA_raw": [0.09, 0.0, 0.0, 0.0],
            "dA_cap_upper": [0.09, 0.0, 0.0, 0.0],
            "dA_unique_opt": [0.09, 0.0, 0.0, 0.0],
            "dL1": [0.3, 0.0, 0.0, 0.0],
            "null_dA_unique": 0.0,
            "null_dA_raw": 0.0,
            "null_dL1": 0.0,
        }
    )

    units = G0.add_units(rows)

    expected = np.sqrt(0.34 / 0.38) - np.sqrt(0.25 / 0.38)
    assert units["rise"].tolist() == pytest.approx([expected, 0.0, 0.0, 0.0])
    assert units["bound_lower"].iloc[0] == pytest.approx(0.09 / 0.76)
    assert units["bound_upper"].iloc[0] == pytest.approx(np.sqrt(0.09 / 0.38))
    mean, lo, hi = G0.whole_split_mean(units, {"rise": units["rise"].to_numpy()}, 500, 0)["rise"]
    assert mean == pytest.approx(expected / 4)
    assert lo <= mean <= hi
    assert lo == 0.0  # molecule 2 alone has no hit


# 2. Products and the stand-in pool.


def test_pool_sets_of_propane_hold_every_component_with_its_boundary() -> None:
    pool = G0.pool_sets(Chem.MolFromSmiles("CCC"))

    assert pool == {(0, 1, 2): 0, (0,): 1, (1, 2): 1, (0, 1): 1, (2,): 1, (1,): 2}


def test_template_sides_keep_only_the_disconnected_side_with_production_counts() -> None:
    mol = Chem.MolFromSmiles(N_PHENYL_SULFONAMIDE)
    endpoints = G0.edge_tuple(mol)

    sides, conflicts = G0.template_sides(_templates(mol), endpoints)

    (side,) = sides
    assert side.atoms == tuple(range(3, 16))
    assert (side.cut, side.formed) == (2, 1)
    assert side.rules == {"aryl_sulfonamide_so2"}
    assert conflicts == 0


def test_null_sides_are_disconnected_and_size_matched() -> None:
    mol = Chem.MolFromSmiles(N_PHENYL_SULFONAMIDE)
    sides, _ = G0.template_sides(_templates(mol), G0.edge_tuple(mol))

    nulls, missing = G0.random_null_sides(mol, sides, np.random.default_rng(0))

    assert missing == 0
    assert [len(n.atoms) for n in nulls] == [len(sides[0].atoms)]
    assert formed_bond_count(nulls[0].atoms, G0.edge_tuple(mol)) == nulls[0].formed >= 1


# 3. Production-matched channels.


def _production_channels(smiles: str) -> dict[tuple[int, ...], set[tuple[int, float]]]:
    """(H shift, m/z) of every predicted channel per fragment atom set of the real model."""
    batch = _build_batch((smiles,))
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        max_masks=6,
        max_enumerated_masks=16,
        use_smarts_templates=True,
        include_cmf=False,
        max_h_transfer=4,
        h_transfer_bound="valence",
        precursor_types=G0.PRECURSOR_TYPES,
    ).eval()
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([NEG_INDEX]),
        spec_nce=th.full((1,), 30.0),
        static_graph=_static_graph(batch, (smiles,)),
        return_annotation=True,
    )
    memberships = output["fragment_atom_fragment_idxs"]
    atoms_of = {
        int(fragment): tuple(
            sorted(output["fragment_atom_indices"][memberships == fragment].tolist())
        )
        for fragment in memberships.unique()
    }
    channels: dict[tuple[int, ...], set[tuple[int, float]]] = {}
    for fragment, shift, mz in zip(
        output["pred_pathway_fragment_idxs"].tolist(),
        output["pred_pathway_h_shifts"].tolist(),
        output["pred_mzs"].double().tolist(),
    ):
        channels.setdefault(atoms_of[fragment], set()).add((int(shift), round(mz, 6)))
    return channels


@pytest.mark.parametrize("smiles", [PRIMARY_SULFONAMIDE, N_PHENYL_SULFONAMIDE, N_ACYL_SULFONAMIDE])
def test_side_channels_equal_what_the_model_builds_for_every_side(smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    sides, _ = G0.template_sides(_templates(mol), G0.edge_tuple(mol))
    calc = G0.ChannelCalculator()
    production = _production_channels(smiles)

    assert sides
    for side in sides:
        mzs, valid = calc(mol, [side.atoms], [side.cut], [side.formed], NEG_INDEX)
        ours = {
            (shift, round(float(mzs[0, k]), 6))
            for k, shift in enumerate(G0.H_SHIFTS)
            if valid[0, k]
        }
        assert ours == production[side.atoms]
        # Every assembled side is two precursor pieces, so it formed one bond.
        assert (side.cut, side.formed) == (2, 1)


def test_formed_bond_takes_back_valence_slots_through_the_calculator() -> None:
    # Pentane's terminal methyls as one assembled side: two cuts, one formed bond.
    mol = Chem.MolFromSmiles("CCCCC")
    calc = G0.ChannelCalculator()

    def top_shift(formed: int) -> int:
        _, valid = calc(mol, [(0, 4)], [2], [formed], NEG_INDEX)
        return max(shift for k, shift in enumerate(G0.H_SHIFTS) if valid[0, k])

    assert top_shift(0) == 2
    assert top_shift(1) == 0


# 4. Union across rules and twin exclusion.


def _spectrum(spec_id: int, peak_mzs: list[float], weights: list[float]) -> G0.SpectrumData:
    return G0.SpectrumData(spec_id, np.array(peak_mzs), np.array(weights), TOL)


def test_union_counts_a_peak_hit_by_two_sides_once() -> None:
    # The two primary sulfonamides of a symmetric bis-sulfonamide each lose their own SO2 and
    # give sides of the same composition, hence the same channel m/z.
    mol = Chem.MolFromSmiles(BIS_PRIMARY_SULFONAMIDE)
    templates = _templates(mol)
    assert [t.rule_name for t in templates] == ["primary_aryl_sulfonamide_so2"] * 2
    sides, _ = G0.template_sides(templates, G0.edge_tuple(mol))
    assert len(sides) == 2
    calc = G0.ChannelCalculator()
    mzs, valid = calc(
        mol, [s.atoms for s in sides], [s.cut for s in sides], [s.formed for s in sides], NEG_INDEX
    )
    assert valid[0, 4] and mzs[0, 4] == pytest.approx(mzs[1, 4], abs=1e-6)
    spectrum = _spectrum(7, [50.0, float(mzs[0, 4])], [0.6, 0.4])
    task = G0.MoleculeTask(1, BIS_PRIMARY_SULFONAMIDE, NEG, [spectrum], 200_000, 0)

    row = G0.process_molecule(task, calc)["spectra"][0]

    # The peak enters once in the union, and once in the rule's own mask; 0.4**2 not doubled.
    assert row["dA_unique"] == pytest.approx(0.16)
    assert row["dA_rule"] == {"primary_aryl_sulfonamide_so2": pytest.approx(0.16)}
    assert row["dL1"] == pytest.approx(0.4)
    assert row["dA_raw"] == pytest.approx(0.16)
    assert row["dA_unique_opt"] == pytest.approx(0.16)
    assert row["A0"] == pytest.approx(0.0)
    assert row["n_capped"] == 0


def test_connected_twin_makes_a_raw_hit_not_unique_and_a_cap_is_not_credited() -> None:
    # Butane's terminal carbons (C2 with two cuts and one formed bond) have connected C2
    # twins: raw coverage counts the peak, unique coverage does not.
    mol = Chem.MolFromSmiles("CCCC")
    side = G0.Side((0, 3), 2, 1, {"toy_rule"})
    calc = G0.ChannelCalculator()
    mzs, valid = calc(mol, [side.atoms], [side.cut], [side.formed], NEG_INDEX)
    assert valid[0, 4]
    spectrum = _spectrum(1, [float(mzs[0, 4])], [1.0])
    no_pool = [np.zeros(1, dtype=bool)]

    twin = G0.score_sides(calc, mol, [side], [spectrum], no_pool, NEG_INDEX, 200_000)
    capped = G0.score_sides(calc, mol, [side], [spectrum], no_pool, NEG_INDEX, 1)
    pooled = G0.score_sides(
        calc, mol, [side], [spectrum], [np.ones(1, dtype=bool)], NEG_INDEX, 200_000
    )

    assert twin.raw[0].tolist() == [True] and twin.unique[0].tolist() == [False]
    assert twin.capped_sides == [0]
    # A capped reach search is never credited, but is counted and can be bounded.
    assert capped.unique[0].tolist() == [False]
    assert capped.cap_credit[0].tolist() == [True]
    assert capped.capped_sides == [1] and capped.capped_search_sides == 1
    # A peak the pool already covers is not new.
    assert pooled.raw[0].tolist() == [False]
