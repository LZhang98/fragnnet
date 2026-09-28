"""Scientific invariants of the excise-then-cleave coverage diagnostic."""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from scripts.debug.debug_masster_excision_continuations import (
    Bridge,
    MoleculeCoverage,
    _alias_ft,
    continuation_sets,
    covered,
    enumerate_bridges,
    evaluate_spectrum,
    random_disconnected,
    summarize,
)
from scripts.debug.debug_masster_smarts_rule_gate import (
    ChannelScorer,
    _is_connected,
    edge_tuple,
    reachable_h,
)


@pytest.mark.parametrize("smiles", ["CCNCC", "CCS(=O)(=O)NC", "CCONC"])
def test_lift_matches_explicit_rewiring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    edges = edge_tuple(mol)
    pool = {
        s
        for n in range(1, mol.GetNumAtoms() + 1)
        for s in combinations(range(mol.GetNumAtoms()), n)
        if _is_connected(s, edges)
    }
    for bridge in enumerate_bridges(mol):
        remaining = sorted(set(range(mol.GetNumAtoms())) - bridge.removed)
        bonds = [(a, b) for a, b in zip(*edges) if a in remaining and b in remaining]
        bonds.append(tuple(sorted(bridge.endpoints)))
        rewired = tuple(zip(*bonds))
        expected = {
            s
            for n in range(1, len(remaining) + 1)
            for s in combinations(remaining, n)
            if bridge.endpoints.issubset(s) and _is_connected(s, rewired)
        }
        assert continuation_sets(pool, bridge) == expected
        # Fragments not spanning the new bond are ordinary connected sets.
        for n in range(1, len(remaining) + 1):
            for s in combinations(remaining, n):
                if not bridge.endpoints.issubset(s) and _is_connected(s, rewired):
                    assert s in pool


def test_ring_and_branching_deletions_are_excluded():
    assert enumerate_bridges(Chem.MolFromSmiles("C1CCCCC1")) == []
    mol = Chem.MolFromSmiles("CC(C)C")
    assert not any(b.removed == {1} for b in enumerate_bridges(mol))


def test_post_excision_cleavage_is_new_and_requires_both_endpoints():
    bridge = Bridge(frozenset({2, 3, 4}), frozenset({1, 5}), "SO2")
    pool = {(0, 1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4)}
    products = continuation_sets(pool, bridge)
    assert products == {(0, 1, 5, 6), (1, 5, 6)}


def test_environment_invariant_to_atom_order():
    mol = Chem.MolFromSmiles("CCS(=O)(=O)NCC")
    reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    assert Counter(b.environment for b in enumerate_bridges(mol)) == Counter(
        b.environment for b in enumerate_bridges(reverse)
    )


def test_connected_formula_twin_and_capped_search():
    mol = Chem.MolFromSmiles("CCOCC")
    assert reachable_h(mol, edge_tuple(mol), Counter(C=2), 2) == {5}
    assert reachable_h(mol, edge_tuple(mol), Counter(C=2), 2, cap=1) is None


def test_new_bond_consumes_two_valence_slots():
    mol = Chem.MolFromSmiles("CCNCC")
    bridge = next(
        b
        for b in enumerate_bridges(mol)
        if Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in b.removed) == Counter(N=1)
    )
    product = tuple(i for i in range(mol.GetNumAtoms()) if i not in bridge.removed)
    scorer = ChannelScorer()
    mass, allowed = scorer(mol, [product], 0)
    strict_mass, strict = scorer(mol, [product], 0, reconnect=True)
    np.testing.assert_array_equal(mass, strict_mass)
    assert np.all(~strict | allowed)
    assert allowed.sum() > strict.sum()


def test_random_null_matches_count_and_size():
    mol = Chem.MolFromSmiles("CCCCCCCC")
    products = {(0, 2, 3), (1, 3, 4), (1, 2, 5, 6)}
    nulls, missing = random_disconnected(mol, products, np.random.default_rng(0))
    assert missing == 0
    assert Counter(map(len, nulls)) == Counter(map(len, products))
    assert all(not _is_connected(s, edge_tuple(mol)) for s in nulls)


def test_no_bridge_covers_no_additional_peaks():
    mol = Chem.MolFromSmiles("CC")
    cache = MoleculeCoverage(mol, ChannelScorer(), cap=100)
    result = evaluate_spectrum(
        mol,
        {(0, 1)},
        np.array([31.0]),
        [],
        set(),
        np.array([31.0, 50.0]),
        np.array([0.8, 0.2]),
        (1e-5, 200),
        cache,
        np.random.default_rng(0),
    )
    assert len(result) == 1
    assert result[0]["pool_covered"] == pytest.approx(0.8)
    for name in ("template_added", "whole_added", "continuation_added", "unique", "null"):
        assert result[0][name] == 0


def test_tolerance_boundaries_and_empty_channels():
    np.testing.assert_array_equal(
        covered(np.array([100.001, 100.003]), np.array([100.0]), (1e-5, 200.0)), [True, False]
    )
    assert not covered(np.array([100.0]), np.array([]), (1e-5, 200.0)).any()


def test_bootstrap_uses_molecules_not_spectra():
    rows = []
    for molecule, repetitions, unique in [(1, 10, 0.1), (2, 1, 0.2), (3, 1, 0.3)]:
        for _ in range(repetitions):
            rows.append(
                {
                    "environment": "x",
                    "mol_id": molecule,
                    "products": 2,
                    "capped": 0,
                    "null_capped": 0,
                    "null_missing": 0,
                    "pool_covered": 0.8,
                    "template_added": 0,
                    "whole_added": 0,
                    "continuation_added": unique,
                    "unique": unique,
                    "null": 0,
                    "strict_unique": unique,
                    "strict_null": 0,
                }
            )
    summary = summarize(pd.DataFrame(rows)).iloc[0]
    assert summary.diff_pct == pytest.approx(20)
    assert summary.molecules == 3


def test_checkpoint_label_alias_preserves_width():
    config = {"spec_params": {"inst_types": ["FT"]}, "tolerance": {"FT": {"rel": 1e-5}}}
    aliased = _alias_ft(config)
    assert aliased["spec_params"]["inst_types"] == ["OB"]
    assert "OB" in aliased["tolerance"]
    assert config["spec_params"]["inst_types"] == ["FT"]


def test_continuation_peak_is_not_credited_to_whole_product():
    mol = Chem.MolFromSmiles("CCS(=O)(=O)NCC")
    bridge = next(b for b in enumerate_bridges(mol) if b.removed == frozenset({2, 3, 4}))
    whole = (0, 1, 5, 6, 7)
    child = (1, 5, 6, 7)
    pool = {tuple(range(8)), (1, 2, 3, 4, 5, 6, 7)}
    scorer = ChannelScorer()
    masses, _ = scorer(mol, [child], 0, reconnect=True)
    cache = MoleculeCoverage(mol, scorer, cap=10000)
    result = evaluate_spectrum(
        mol,
        pool,
        np.array([]),
        [bridge],
        {whole},
        np.array([masses[0, 4]]),
        np.array([1.0]),
        (1e-5, 200.0),
        cache,
        np.random.default_rng(3),
    )
    overall = next(r for r in result if r["environment"] == "__all__")
    assert overall["template_added"] == 0
    assert overall["whole_added"] == 0
    assert overall["continuation_added"] == 1
    assert overall["strict_unique"] == 1


def test_capped_query_never_counts_as_unique(monkeypatch):
    import scripts.debug.debug_masster_excision_continuations as probe

    mol = Chem.MolFromSmiles("CCOCC")
    cache = MoleculeCoverage(mol, ChannelScorer(), cap=1)
    atoms = (0, 3)
    cache.prepare({atoms}, False)
    mass = cache.channels[atoms, False][0][0]
    monkeypatch.setattr(probe, "reachable_h", lambda *a, **k: None)
    hit, capped = cache.hit({atoms}, np.array([mass]), (1e-5, 200.0), unique=True)
    assert not hit.any()
    assert capped == 1


def test_targeted_h_search_matches_exhaustive_answers():
    mol = Chem.MolFromSmiles("CC(O)CCNC")
    edges = edge_tuple(mol)
    composition = Counter(C=3, N=1)
    exhaustive = reachable_h(mol, edges, composition, 4, cap=100000)
    for target in range(4, 15):
        targeted = reachable_h(
            mol, edges, composition, 4, cap=100000, required_h=frozenset({target})
        )
        assert any(abs(h - target) <= 4 for h in targeted) == any(
            abs(h - target) <= 4 for h in exhaustive
        )


def test_carbon_only_bridge_can_have_no_connected_formula_twin():
    # Retaining both N atoms requires all four intervening C atoms in the precursor.
    # Deleting one internal CH2 gives C3N2, impossible for a connected precursor set.
    mol = Chem.MolFromSmiles("NCCCCN")
    bridge = next(b for b in enumerate_bridges(mol) if b.removed == {2})
    product = next(iter(continuation_sets({tuple(range(6))}, bridge)))
    composition = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in product)
    assert composition == Counter(C=3, N=2)
    assert reachable_h(mol, edge_tuple(mol), composition, len(product)) == set()
