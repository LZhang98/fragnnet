"""Tests for the post-cut information gate: descriptors, the listwise correction scores and the
rule-provenance marker, families and ceiling read."""

import argparse
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
import torch as th
from rdkit import Chem

import scripts.debug.masster_memorization.debug_masster_base_peak_tables as TABLES
import scripts.debug.masster_memorization.debug_masster_postcut_information_gate as GATE
from fragnnet.frag.masster_templates import MASSTER_RULE_NAMES
from scripts.debug.smarts.debug_masster_smarts_rule_gate import ChannelScorer
from tests.test_masster_model import _build_batch, _build_model


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


# Rule provenance: template sides, the marker, the families and the ceiling read.

SULFONAMIDE = "CNS(=O)(=O)c1ccccc1"
H_MASS = 1.00782503207
ROUND = 6


@pytest.fixture(scope="module")
def sulfonamide_channels() -> dict:
    """Sides of N-methylbenzenesulfonamide ([M+H]+) and their production channels."""
    batch = _build_batch((SULFONAMIDE,))
    model = _build_model(batch, include_cmf=False, max_h_transfer=4, h_transfer_bound="valence")
    sides = GATE.template_sides(Chem.MolFromSmiles(SULFONAMIDE), "[M+H]+")
    mzs, valid, shifts = GATE.side_channel_table(
        model, batch, sides, [0] * len(sides), th.tensor([0])
    )
    return {"batch": batch, "model": model, "sides": sides, "mzs": mzs, "valid": valid, "h": shifts}


def _side(rule: str, atoms: tuple, connected: bool) -> "GATE.TemplateSide":
    return GATE.TemplateSide(
        rule_name=rule,
        rule_position=MASSTER_RULE_NAMES.index(rule),
        atoms=atoms,
        connected=connected,
        cut_count=2,
        formed_bonds=0 if connected else 1,
    )


def test_template_sides_follow_the_builder(sulfonamide_channels: dict) -> None:
    disconnected, sulfonyl = sulfonamide_channels["sides"]
    assert disconnected.rule_name == sulfonyl.rule_name == "aryl_sulfonamide_so2"
    # The neutral loss SO2 is one connected piece (S and both O), the product is two pieces.
    assert sulfonyl.atoms == (2, 3, 4) and sulfonyl.connected and sulfonyl.formed_bonds == 0
    assert not disconnected.connected and disconnected.formed_bonds == 1
    # Each side carries ALL of the template's broken bonds (N-S and S-c), not its own crossings.
    assert disconnected.cut_count == sulfonyl.cut_count == 2
    # A primary sulfonamide has a polarity-guarded rule per adduct.
    primary = Chem.MolFromSmiles("NS(=O)(=O)c1ccccc1")
    assert {s.rule_name for s in GATE.template_sides(primary, "[M+H]+")} == {
        "primary_aryl_sulfonamide_o_transfer"
    }
    assert {s.rule_name for s in GATE.template_sides(primary, "[M-H]-")} == {
        "primary_aryl_sulfonamide_so2"
    }
    assert GATE.template_sides(Chem.MolFromSmiles("CCO"), "[M+H]+") == ()


def test_side_channels_match_the_production_scorer(sulfonamide_channels: dict) -> None:
    scorer = ChannelScorer()
    mol = Chem.MolFromSmiles(SULFONAMIDE)
    for i, side in enumerate(sulfonamide_channels["sides"]):
        # For a two-piece assembly the flat "reconnect" (minus 2) equals 2 per formed bond.
        mzs, valid = scorer(mol, [side.atoms], 0, reconnect=not side.connected)
        np.testing.assert_allclose(sulfonamide_channels["mzs"][i], mzs[0], atol=1e-9)
        np.testing.assert_array_equal(sulfonamide_channels["valid"][i], valid[0])


def test_channel_h_shift_range(sulfonamide_channels: dict) -> None:
    h, mzs, valid = (sulfonamide_channels[k] for k in ("h", "mzs", "valid"))
    np.testing.assert_array_equal(h, np.arange(-4, 5))
    # One H-shift step is one hydrogen mass, on every side.
    np.testing.assert_allclose(np.diff(mzs, axis=1), H_MASS, atol=1e-6)
    disconnected, sulfonyl = valid
    # SO2 has no hydrogen to lose below -1 (the ion's own proton is the only H that can leave).
    assert not sulfonyl[h < -1].any() and sulfonyl[(h >= -1) & (h <= 4)].all()
    # The complement of the product holds every H, so it can give but not take hydrogens.
    assert disconnected[h <= 0].all() and not disconnected[h > 0].any()
    # SO2 + H+ at shift 0, from element masses.
    table = Chem.GetPeriodicTable()
    so2 = sum(table.GetMostCommonIsotopeMass(a) for a in ("S", "O", "O"))
    assert mzs[1, list(h).index(0)] == pytest.approx(so2 + 1.007276, abs=2e-4)


def test_formed_bonds_tighten_the_valence_bound(sulfonamide_channels: dict) -> None:
    batch, model = sulfonamide_channels["batch"], sulfonamide_channels["model"]
    sulfonyl = sulfonamide_channels["sides"][1]
    _, plain, _ = GATE.side_channel_table(model, batch, [sulfonyl], [0], th.tensor([0]))
    heavy = dataclasses.replace(sulfonyl, formed_bonds=5)
    _, tight, _ = GATE.side_channel_table(model, batch, [heavy], [0], th.tensor([0]))
    # Five formed bonds take back ten valence slots: only the H losses stay valid.
    assert plain.sum() > tight.sum() and not tight[0, 5:].any()


def test_marker_on_a_sulfonamide_twin(sulfonamide_channels: dict) -> None:
    disconnected, sulfonyl = sulfonamide_channels["sides"]
    mzs, valid, h = (sulfonamide_channels[k] for k in ("mzs", "valid", "h"))
    zero = list(h).index(0)
    at_product, at_sulfonyl = round(mzs[0, zero], ROUND), round(mzs[1, zero], ROUND)
    cands = np.array(sorted([50.0, at_sulfonyl, at_product, 250.0]))
    cols = list(GATE.RULE_COLUMNS)
    out = GATE.rule_marker_columns(
        cands, mzs, valid, [disconnected, sulfonyl], tol_da=1e-5, rel=1e-5, min_mz=200.0
    )
    product_row, sulfonyl_row = (
        int(np.flatnonzero(cands == m)[0]) for m in (at_product, at_sulfonyl)
    )
    assert out[product_row, cols.index("rule_disc")] == 1
    assert out[product_row, cols.index("rule_conn")] == 0
    assert out[sulfonyl_row, cols.index("rule_conn")] == 1
    rule = cols.index("rule_aryl_sulfonamide_so2")
    np.testing.assert_array_equal(out[:, rule] > 0, np.isin(cands, [at_product, at_sulfonyl]))
    np.testing.assert_array_equal(out[:, cols.index("rule_any")], out[:, rule])
    # No other rule column and no atom-set share is set by the m/z match.
    others = [i for i, c in enumerate(cols) if c not in ("rule_any", "rule_disc", "rule_conn")]
    assert out[:, [i for i in others if i not in (rule, cols.index("rule_any_tol"))]].sum() == 0


def test_marker_is_multi_hot_and_tolerance_variant_is_wider() -> None:
    first = _side("aryl_sulfonamide_so2", (0,), connected=True)
    second = _side("aryl_sulfonate_so2", (1,), connected=False)
    cands = np.array([99.0, 100.0, 100.001, 101.0, 105.0])
    side_mzs = np.array([[100.0, 101.0], [100.0, 105.0]])
    valid = np.array([[True, True], [True, False]])
    out = GATE.rule_marker_columns(
        cands, side_mzs, valid, [first, second], tol_da=1e-5, rel=1e-5, min_mz=200.0
    )
    col = {c: out[:, i] for i, c in enumerate(GATE.RULE_COLUMNS)}
    # Candidate 100.0 sits at a channel of both rules: two flags, one rule_any. The invalid
    # 105.0 channel marks nothing.
    np.testing.assert_array_equal(col["rule_aryl_sulfonamide_so2"], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(col["rule_aryl_sulfonate_so2"], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(col["rule_any"], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(col["rule_conn"], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(col["rule_disc"], [0, 1, 0, 0, 0])
    # 100.001 is 1e-3 Da away: outside 1e-5 Da, inside 10 ppm at the 200 m/z floor (2e-3 Da).
    np.testing.assert_array_equal(col["rule_any_tol"], [0, 1, 1, 1, 0])
    assert (out[:, list(GATE.RULE_COLUMNS).index("rule_atomset")] == 0).all()


def test_atomset_identity_of_the_connected_side(sulfonamide_channels: dict) -> None:
    sides = sulfonamide_channels["sides"]
    inside = np.zeros((4, 11), dtype=bool)
    inside[0, [2, 3, 4]] = True  # exactly the SO2 side
    inside[1, :] = True  # the precursor
    inside[2, [2, 3]] = True  # a subset of the side
    inside[3, [0, 1, 5, 6, 7, 8, 9, 10]] = True  # the disconnected side's atoms
    np.testing.assert_array_equal(
        GATE.atomset_channel_match(inside, sides[1:], 11), [True, False, False, False]
    )
    np.testing.assert_array_equal(
        GATE.atomset_channel_match(inside, sides, 11), [True, False, False, True]
    )


def test_rule_families_are_separate_from_the_old_columns() -> None:
    assert set(GATE.DESCRIPTORS).isdisjoint(GATE.RULE_COLUMNS) and len(GATE.DESCRIPTORS) == 98
    assert GATE.RULE_COLUMNS[:5] == GATE.RULE_SUMMARY
    assert GATE.RULE_COLUMNS[5:] == tuple(f"rule_{n}" for n in MASSTER_RULE_NAMES)
    assert GATE.FAMILIES["all"] == GATE.DESCRIPTORS  # unchanged by the rule columns
    assert "rule_any_tol" not in GATE.FAMILIES["rules"]
    names, rules = list(GATE.DESCRIPTORS), list(GATE.RULE_COLUMNS)
    x_cols, rule_cols = GATE.resolve_family("rules+existing", names, rules)
    assert [names[i] for i in x_cols] == list(GATE.BASICS + GATE.EXISTING)
    assert [rules[i] for i in rule_cols] == list(GATE.RULE_FAMILY_COLUMNS)
    assert GATE.resolve_family("rules_permuted", names, rules) == GATE.resolve_family(
        "rules", names, rules
    )
    assert GATE.resolve_family("context", names, rules) == ([], [])
    # An old cache has no rule columns: the old families resolve, the rule families do not.
    assert GATE.available_families(names, []) == [
        "context",
        "existing",
        "postcut",
        "charge",
        "postcut+charge",
        "all",
    ]
    with pytest.raises(ValueError, match="absent from the cache"):
        GATE.resolve_family("rules", names, [])


def test_permutation_keeps_marker_counts_and_patterns() -> None:
    rng = np.random.default_rng(1)
    offsets = np.array([0, 5, 5, 30, 33, 60])
    rule_x = (rng.random((60, 6)) < 0.15).astype(np.float32)
    rule_x[33:60, :] *= 0  # a spectrum with no flags stays untouched
    rule_x[[0, 2, 7, 8, 20], 0] = 1
    rule_x[[7, 20], 3] = 1  # multi-hot rows
    out = GATE.permute_rule_columns(offsets, rule_x, seed=3)
    assert out.shape == rule_x.shape and out is not rule_x
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        np.testing.assert_array_equal(out[a:b].sum(axis=0), rule_x[a:b].sum(axis=0))
        rows_in = sorted(map(tuple, rule_x[a:b]))
        rows_out = sorted(map(tuple, out[a:b]))
        assert rows_in == rows_out  # whole rows move, so the multi-hot patterns survive
    np.testing.assert_array_equal(out[33:60], rule_x[33:60])
    assert not np.array_equal(out[5:30], rule_x[5:30])
    np.testing.assert_array_equal(out, GATE.permute_rule_columns(offsets, rule_x, seed=3))
    marked = GATE.marked_spectra(offsets, out, ["rule_any", "a", "b", "c", "d", "e"])
    np.testing.assert_array_equal(marked, [True, False, True, False, False])


def test_ceiling_on_a_toy_spectrum() -> None:
    pred = np.array([0.5, 0.3, 0.2])
    tgt = np.array([0.2, 0.5, 0.3])
    norm = float(np.linalg.norm(tgt))
    swap = np.array([True, False, False])
    # Literal: candidate 0 takes its target share 0.2, the rest keep 0.3 and 0.2.
    q = np.array([0.2, 0.3, 0.2])
    expected = q @ tgt / (np.linalg.norm(q) * norm)
    got = GATE.swap_ceiling(pred, tgt, swap, norm, mass_preserving=False)
    assert got == pytest.approx(expected)
    # The cosine ignores the scale, so the renormalisation of the plan text changes nothing.
    assert got == pytest.approx(GATE.cosine_with_target(q / q.sum(), tgt, norm))
    # A single marked candidate has no allocation to change: mass preserving is the raw cosine.
    raw = GATE.cosine_with_target(pred, tgt, norm)
    assert GATE.swap_ceiling(pred, tgt, swap, norm, mass_preserving=True) == pytest.approx(raw)
    # Two marked candidates keep their total predicted mass 0.8, split 2:5 like the target.
    both = np.array([True, True, False])
    q = np.array([0.8 * 0.2 / 0.7, 0.8 * 0.5 / 0.7, 0.2])
    assert GATE.swap_ceiling(pred, tgt, both, norm, mass_preserving=True) == pytest.approx(
        q @ tgt / (np.linalg.norm(q) * norm)
    )
    # Swapping everything to the target gives the covered-target cosine (1 when all covered).
    everything = np.ones(3, dtype=bool)
    assert GATE.swap_ceiling(pred, tgt, everything, norm, mass_preserving=False) == pytest.approx(
        1.0
    )
    # Uncovered target lowers the ceiling: the norm is the whole target's.
    assert GATE.swap_ceiling(
        pred, tgt, everything, norm * 2, mass_preserving=False
    ) == pytest.approx(0.5)
    # No marked peak: nothing changes.
    none = np.zeros(3, dtype=bool)
    assert GATE.swap_ceiling(pred, tgt, none, norm, mass_preserving=False) == pytest.approx(raw)
    # A marked set without target intensity: mass-preserving leaves the spectrum alone.
    assert GATE.swap_ceiling(
        pred, np.array([0.0, 0.6, 0.4]), swap, 0.7, mass_preserving=True
    ) == pytest.approx(GATE.cosine_with_target(pred, np.array([0.0, 0.6, 0.4]), 0.7))


def test_rank_classes_order_candidates_by_target_size() -> None:
    classes = GATE.rank_classes(np.array([0.5, 0.3, 0.1, 0.0, 0.0, 0.05]))
    # Ranks 1, 2, 3, (zeros), (zeros), 4 -> 1 + floor(log2(rank)); zeros one class below the last.
    np.testing.assert_array_equal(classes, [1, 2, 2, 4, 4, 3])
    np.testing.assert_array_equal(GATE.rank_classes(np.zeros(3)), [0, 0, 0])


def test_controls_are_unmarked_distinct_and_matched() -> None:
    tgt = np.array([0.5, 0.3, 0.1, 0.0, 0.0, 0.0])
    classes = GATE.rank_classes(tgt)
    marked = np.array([False, True, False, True, False, False])
    mz = np.array([100.0, 105.0, 110.0, 120.0, 125.0, 130.0])
    for seed in range(20):
        controls, tiers = GATE.match_controls(mz, classes, marked, np.random.default_rng(seed), 0.2)
        assert set(controls) <= {0, 2, 4, 5} and len(set(controls)) == 2
        assert (tiers == 0).all()  # band and class both matched
        assert 2 in controls  # the covered marked peak (class 2) can only take candidate 2
    # Out of the m/z band, the nearest class inside the band goes before the same class outside.
    far = np.array([100.0, 105.0, 400.0, 500.0, 510.0, 520.0])
    controls, tiers = GATE.match_controls(
        far, classes, np.array([0, 1, 0, 0, 0, 0], bool), np.random.default_rng(0), 0.2
    )
    assert tiers[0] == 1 and controls[0] == 0
    # More marked peaks than unmarked candidates: the surplus stays unmatched.
    controls, tiers = GATE.match_controls(
        mz[:3], classes[:3], np.array([True, True, False]), np.random.default_rng(0), 0.5
    )
    assert sorted(controls) == [-1, 2] and sorted(tiers)[0] == -1


def test_spectrum_headroom_is_zero_for_a_perfect_prediction_and_positive_for_a_miss() -> None:
    tgt = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    mz = np.array([100.0, 105.0, 110.0, 120.0, 125.0, 130.0])
    marked = np.array([True, False, False, False, False, False])
    norm = float(np.linalg.norm(tgt))
    rng = np.random.default_rng(0)
    perfect = GATE.spectrum_headroom(tgt.copy(), tgt, mz, marked, norm, rng, band=0.5, draws=5)
    assert perfect["headroom_literal"] == pytest.approx(0.0, abs=1e-12)
    assert perfect["headroom_mass_preserving"] == pytest.approx(0.0, abs=1e-12)
    # The marked base peak is under-predicted: fixing it gains more than fixing a zero-target peak.
    pred = np.array([0.1, 0.3, 0.2, 0.2, 0.1, 0.1])
    miss = GATE.spectrum_headroom(
        pred, tgt, mz, marked, norm, np.random.default_rng(0), band=0.5, draws=5
    )
    assert miss["rule_literal"] > 0 and miss["headroom_literal"] > 0
    assert miss["headroom_literal"] == pytest.approx(miss["rule_literal"] - miss["control_literal"])
    assert (
        miss["marked_peaks"] == 1
        and miss["tier0"] + miss["tier1"] + miss["tier2"] + miss["tier3"] == 1
    )


def test_marked_gain_units_and_decision_rule() -> None:
    values = np.array([0.1, 0.3, 0.0, 0.0, 0.5, 0.0])
    marked = np.array([True, True, False, False, False, False])
    mols = np.array([1, 2, 3, 4, 5, 6])
    row = GATE.marked_gain_row("g", values, marked, mols, reps=50)
    # Whole-split unit divides by all six spectra; subset unit by the two marked ones only.
    assert row["g_whole"] == pytest.approx(0.4 / 6) and row["g_subset"] == pytest.approx(0.2)
    assert row["g_whole_lo"] <= row["g_whole"] <= row["g_whole_hi"] or row["g_whole_lo"] >= 0
    assert GATE.g1_decision(0.004, 0.006, 0.001, 0.003) == "CONTINUE"
    assert GATE.g1_decision(0.004, 0.006, -0.001, 0.003) == "UNDERPOWERED"  # gain CI includes 0
    assert GATE.g1_decision(0.001, 0.002, -0.001, 0.002) == "STOP"
    assert GATE.g1_decision(0.001, 0.005, -0.001, 0.002) == "UNDERPOWERED"  # headroom CI wide
    assert GATE.g1_decision(0.001, 0.002, -0.004, 0.004) == "UNDERPOWERED"  # gain CI wide


def test_mol_id_parsing_and_spectrum_selection(tmp_path) -> None:
    np.testing.assert_array_equal(GATE.parse_mol_ids("3, 5,7"), [3, 5, 7])
    path = tmp_path / "ids.csv"
    path.write_text("mol_id\n4\n9\n")
    np.testing.assert_array_equal(GATE.parse_mol_ids(str(path)), [4, 9])
    with pytest.raises(ValueError, match="non-integer"):
        GATE.parse_mol_ids("3,x")
    specs = pd.DataFrame({"mol_id": [1, 1, 2, 3, 3, 4], "prec_type": ["[M+H]+"] * 6})
    bearing = {1, 3, 4}

    def sides_of(mol_id: int, precursor_type: str) -> tuple:
        return ("side",) if mol_id in bearing else ()

    def select(**kw) -> list:
        base = {
            "mol_ids": "",
            "template_bearing_only": False,
            "max_molecules": 0,
            "max_spectra": 0,
            "seed": 0,
            "split": "val",
        }
        return GATE.select_spectra(specs, sides_of, argparse.Namespace(**(base | kw)))

    assert select() == [0, 1, 2, 3, 4, 5]
    assert select(mol_ids="1,2") == [0, 1, 2]
    assert select(template_bearing_only=True) == [0, 1, 3, 4, 5]
    assert select(template_bearing_only=True, max_molecules=1, seed=1) in ([0, 1], [3, 4], [5])
    assert select(max_spectra=2) == [0, 1]
    with pytest.raises(ValueError, match="not in the val split"):
        select(mol_ids="99")
    with pytest.raises(ValueError, match="no spectra selected"):
        select(mol_ids="2", template_bearing_only=True)


def _synthetic_cache(path, spectra: int = 60, rows: int = 12) -> None:
    """A tiny cache in the extract layout, with rule flags on every third spectrum."""
    rng = np.random.default_rng(0)
    total = spectra * rows
    pred = rng.dirichlet(np.ones(rows), size=spectra).reshape(-1).astype(np.float32)
    tgt = (rng.random(total) < 0.4) * rng.random(total).astype(np.float32)
    rule_x = np.zeros((total, len(GATE.RULE_COLUMNS)), dtype=np.float32)
    columns = {c: i for i, c in enumerate(GATE.RULE_COLUMNS)}
    for s in range(0, spectra, 3):
        row = s * rows + int(rng.integers(rows))
        rule_x[
            row,
            [columns["rule_any"], columns["rule_any_tol"], columns["rule_aryl_sulfonamide_so2"]],
        ] = 1
    mz = np.concatenate([np.sort(rng.uniform(50, 400, rows)) for _ in range(spectra)])
    np.savez(
        path,
        spec_id=np.arange(spectra),
        mol_id=np.arange(spectra) // 2,
        offsets=np.arange(spectra + 1) * rows,
        pred=pred,
        tgt=tgt,
        x=rng.normal(size=(total, len(GATE.DESCRIPTORS))).astype(np.float32),
        target_norm=np.linalg.norm(tgt.reshape(spectra, rows), axis=1) + 0.1,
        context=rng.normal(size=(spectra, 3)).astype(np.float32),
        descriptors=np.asarray(GATE.DESCRIPTORS),
        context_names=np.asarray(GATE.CONTEXT),
        mz=mz,
        rule_x=rule_x,
        rule_columns=np.asarray(GATE.RULE_COLUMNS),
    )


def test_fit_and_ceiling_run_on_a_synthetic_cache(tmp_path) -> None:
    cache = tmp_path / "cache.npz"
    _synthetic_cache(cache)
    fit_csv = tmp_path / "fit.csv"
    GATE.main(
        ["fit", "--input", str(cache), "--output", str(fit_csv), "--device", "cpu"]
        + ["--epochs", "2", "--reps", "20", "--hidden", "4", "--batch_spectra", "16"]
        + ["--families", "context", "existing", "rules", "rules+existing", "rules_permuted"]
    )
    table = pd.read_csv(fit_csv).set_index("family")
    assert list(table.index) == ["context", "existing", "rules", "rules+existing", "rules_permuted"]
    assert table.loc["rules", "n_marked_spectra"] == 20
    for column in ("marked_gain_whole", "marked_gain_subset", "marked_over_context_subset_lo"):
        assert np.isfinite(table.loc["rules", column])
    # Whole-split units are the subset units scaled by marked / all spectra.
    assert table.loc["rules", "marked_gain_whole"] == pytest.approx(
        table.loc["rules", "marked_gain_subset"] * 20 / 60
    )
    ceiling_csv = tmp_path / "ceiling.csv"
    GATE.main(
        ["ceiling", "--input", str(cache), "--output", str(ceiling_csv), "--fit_csv", str(fit_csv)]
        + ["--reps", "20", "--draws", "3"]
    )
    result = pd.read_csv(ceiling_csv).set_index("marker")
    assert result.loc["rule_any", "marked_spectra"] == 20
    assert result.loc["rule_aryl_sulfonamide_so2", "marked_peaks"] == 20
    assert result.loc["rule_disc", "marked_spectra"] == 0
    assert result.loc["rule_any", "headroom_literal_whole"] == pytest.approx(
        result.loc["rule_any", "headroom_literal_subset"] * 20 / 60
    )
    provenance = json.loads(ceiling_csv.with_suffix(".provenance.json").read_text())
    decision = provenance["decisions"][str(fit_csv)]["decision"]
    assert decision in ("CONTINUE", "STOP", "UNDERPOWERED")
    assert provenance["cache_rule_columns"] == list(GATE.RULE_COLUMNS)
    assert provenance["masster_rule_names"] == list(MASSTER_RULE_NAMES)


def test_fit_still_reads_a_cache_without_rule_columns(tmp_path) -> None:
    cache = tmp_path / "cache.npz"
    _synthetic_cache(cache)
    loaded = np.load(cache)
    old = {k: loaded[k] for k in loaded.files if k not in ("rule_x", "rule_columns", "mz")}
    np.savez(tmp_path / "old.npz", **old)
    output = tmp_path / "fit.csv"
    args = ["fit", "--input", str(tmp_path / "old.npz"), "--output", str(output), "--device", "cpu"]
    GATE.main(args + ["--epochs", "1", "--reps", "10", "--hidden", "4", "--batch_spectra", "16"])
    table = pd.read_csv(output)
    # The default is every family the cache can serve: the six original ones, no rule columns.
    assert list(table["family"]) == [
        "context",
        "existing",
        "postcut",
        "charge",
        "postcut+charge",
        "all",
    ]
    assert "n_marked_spectra" not in table.columns
    with pytest.raises(ValueError, match="absent from the cache"):
        GATE.main(args + ["--families", "rules", "--epochs", "1", "--reps", "10"])
