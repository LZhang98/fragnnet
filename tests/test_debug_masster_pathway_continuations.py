"""Invariants of the model-free G2/G2b continuation gate (polarity, pool, motifs, graft, unit)."""

import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from fragnnet.frag.masster_templates import MASSTER_NL_RULES, formed_bond_count
from fragnnet.frag.smarts_prepass import _apply_nl_prepass
from scripts.debug import debug_masster_excision_continuations as ec
from scripts.debug.debug_masster_smarts_rule_gate import (
    ChannelScorer,
    _is_connected,
    edge_tuple,
)

SULFONAMIDE = "c1ccccc1S(=O)(=O)NC"


@pytest.fixture(scope="module")
def scorer() -> ChannelScorer:
    return ChannelScorer()


def _connected_sets(mol: Chem.Mol) -> set[tuple[int, ...]]:
    edges = edge_tuple(mol)
    n = mol.GetNumAtoms()
    return {
        s for k in range(1, n + 1) for s in combinations(range(n), k) if _is_connected(s, edges)
    }


def _config(**overrides) -> ec.UnitConfig:
    values = {
        "precursor_type": "[M-H]-",
        "precursor_index": 1,
        "operators": ("bridge", "graft"),
        "motifs": (),
        "families": None,
        "pool_cap": 4000,
        "graft_max_distance": 4,
        "graft_whole_max_cuts": 1,
        "reach_cap": 100_000,
    }
    return ec.UnitConfig(**{**values, **overrides})


# 1. Polarity plumbing -------------------------------------------------------------------


def test_adduct_index_selects_polarity_and_cache_is_keyed_by_it(scorer):
    mol = Chem.MolFromSmiles("NS(=O)(=O)c1ccccc1")
    whole = tuple(range(mol.GetNumAtoms()))
    positive = ec.MoleculeCoverage(mol, scorer, 100, precursor_index=0)
    negative = ec.MoleculeCoverage(mol, scorer, 100, precursor_index=1)
    assert np.isclose(positive.mz_array({whole}), 158.0270, atol=1e-3).any()
    assert np.isclose(negative.mz_array({whole}), 156.0125, atol=1e-3).any()
    assert not np.isclose(negative.mz_array({whole}), 158.0270, atol=1e-3).any()
    # One cache switched between polarities keeps both channel tables apart.
    shared = ec.MoleculeCoverage(mol, scorer, 100, precursor_index=0)
    first = shared.mz_array({whole})
    shared.precursor_index = 1
    second = shared.mz_array({whole})
    assert set(shared.by_polarity) == {0, 1}
    assert not np.array_equal(first, second)


def test_spectrum_polarity_must_match_the_adduct_index(scorer):
    mol = Chem.MolFromSmiles("NS(=O)(=O)c1ccccc1")
    cache = ec.MoleculeCoverage(mol, scorer, 100, precursor_index=1)
    cache.check_polarity("[M-H]-")
    with pytest.raises(ValueError, match=r"does not match adduct index 1"):
        cache.check_polarity("[M+H]+")
    with pytest.raises(ValueError, match="does not match adduct index"):
        ec.evaluate_spectrum(
            mol,
            {tuple(range(mol.GetNumAtoms()))},
            np.array([156.0125]),
            [],
            set(),
            np.array([156.0125]),
            np.array([1.0]),
            (1e-5, 200.0),
            cache,
            np.random.default_rng(0),
            prec_type="[M+H]+",
        )


def test_template_sides_use_the_builders_cut_and_formed_counts():
    mol = Chem.MolFromSmiles(SULFONAMIDE)
    if "aryl_sulfonamide_so2" not in {r.name for r in MASSTER_NL_RULES}:
        pytest.skip("aryl_sulfonamide_so2 is not registered")
    sides = ec.template_side_counts(mol, "[M-H]-")
    n = mol.GetNumAtoms()
    lost = (6, 7, 8)
    kept = tuple(a for a in range(n) if a not in lost)
    edges = edge_tuple(mol)
    # Both bonds of the template are broken for either side; only the kept one is two pieces.
    assert sides[lost] == (2, 0)
    assert sides[kept] == (2, formed_bond_count(kept, edges)) == (2, 1)
    assert ec.template_side_counts(mol, "[M+H]+").keys() == sides.keys()


def test_formed_bond_consumes_two_valence_slots_in_production_counts(scorer):
    mol = Chem.MolFromSmiles("CCCCC")
    side = (0, 4)
    cache = ec.MoleculeCoverage(mol, scorer, 100, precursor_index=1)
    optimistic = cache.mz_array({side}, counts={side: (2, 0)})
    strict = cache.mz_array({side}, counts={side: (2, 1)})
    assert strict.size < optimistic.size
    assert set(strict.round(6)) <= set(optimistic.round(6))
    # A formed-bond spec equals the legacy flat reconnect for one formed bond.
    np.testing.assert_allclose(cache.mz_array({side}, reconnect=True), strict)


def test_checkpoint_path_scores_template_sides_with_production_counts(scorer):
    mol = Chem.MolFromSmiles("CCCCC")
    side = (0, 4)  # two pieces, two cut bonds, one formed bond
    cache = ec.MoleculeCoverage(mol, scorer, 100, precursor_index=1)
    strict_counts, optimistic_counts = {side: (2, 1)}, {side: (2, 0)}
    only_optimistic = sorted(
        set(cache.mz_array({side}, counts=optimistic_counts).round(6))
        - set(cache.mz_array({side}, counts=strict_counts).round(6))
    )
    assert only_optimistic
    peak = np.array([only_optimistic[0]])

    def template_added(counts):
        (record,) = ec.evaluate_spectrum(
            mol,
            {tuple(range(5))},
            np.array([]),
            [],
            {side},
            peak,
            np.array([1.0]),
            (1e-5, 200.0),
            cache,
            np.random.default_rng(0),
            template_counts=counts,
            prec_type="[M-H]-",
        )
        return record["template_added"]

    assert template_added(strict_counts) == 0
    assert template_added(optimistic_counts) == 1
    assert template_added(None) == 1  # legacy: optimistic and crossing-bond cuts


def test_prefilter_agrees_with_the_exact_per_set_test(scorer):
    mol = Chem.MolFromSmiles(SULFONAMIDE)
    sets = _connected_sets(mol)
    cache = ec.MoleculeCoverage(mol, scorer, 20_000, precursor_index=1)
    rng = np.random.default_rng(3)
    channels = cache.mz_array(sets)
    peaks = np.concatenate([rng.choice(channels, 6), rng.uniform(50, 300, 6)])
    for unique in (False, True):
        expected = np.zeros(peaks.size, bool)
        capped = 0
        for atoms in sorted(sets):
            matches, was_capped = cache._hit_set(atoms, False, peaks, (1e-5, 200.0), unique)
            expected |= matches
            capped += was_capped
        hit, hit_capped = cache.hit(sets, peaks, (1e-5, 200.0), unique=unique)
        np.testing.assert_array_equal(hit, expected)
        assert hit_capped == capped


# 2. Model-free stand-in pool ------------------------------------------------------------


@pytest.mark.parametrize("smiles", ["CCOC", "C1CCOC1C", "NCC(=O)O"])
def test_standin_pool_equals_independent_fragmentation(smiles):
    mol = Chem.MolFromSmiles(smiles)
    expected = {tuple(range(mol.GetNumAtoms()))}
    for k in (1, 2):
        for cut in combinations(range(mol.GetNumBonds()), k):
            pieces = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, list(cut), addDummies=False))
            expected.update(tuple(sorted(piece)) for piece in pieces)
    # FragmentOnBonds keeps atom indices, so its pieces are the components of the cut mask.
    pool, stats = ec.standin_pool(mol, 10_000, np.random.default_rng(0))
    assert pool == expected
    assert stats == {"pool_raw": len(expected), "pool_kept": len(expected), "pool_truncated": 0}


def test_pool_cap_keeps_all_single_cut_sets_and_records_truncation():
    mol = Chem.MolFromSmiles("CCCCCCCC")
    full, _ = ec.standin_pool(mol, 10_000, np.random.default_rng(0))
    single = {tuple(range(8))}
    for bond in range(mol.GetNumBonds()):
        single.update(
            tuple(sorted(p))
            for p in Chem.GetMolFrags(Chem.FragmentOnBonds(mol, [bond], addDummies=False))
        )
    cap = len(single) + 3
    pool, stats = ec.standin_pool(mol, cap, np.random.default_rng(0))
    assert len(pool) == cap and single <= pool <= full
    assert stats["pool_truncated"] == 1 and stats["pool_raw"] == len(full)
    with pytest.raises(ValueError, match="below"):
        ec.standin_pool(mol, len(single) - 1, np.random.default_rng(0))


# 3. Motif restriction -------------------------------------------------------------------


def _motif_bridges(smiles: str, names: list[str]) -> list[ec.Bridge]:
    mol = Chem.MolFromSmiles(smiles)
    return ec.motif_bridges(mol, ec.enumerate_bridges(mol), names)


def test_aryl_sulfonyl_motif_selects_only_the_so2_bridge():
    (bridge,) = _motif_bridges(SULFONAMIDE, ["aryl_sulfonyl"])
    assert bridge.removed == {6, 7, 8} and bridge.endpoints == {5, 9}
    # Sulfonate esters and alkyl sulfonyls are not registered motifs.
    assert _motif_bridges("c1ccccc1S(=O)(=O)OC", ["aryl_sulfonyl"]) == []
    assert _motif_bridges("CCS(=O)(=O)NC", ["aryl_sulfonyl"]) == []
    (acid,) = _motif_bridges("c1ccccc1S(=O)(=O)[O-]", ["aryl_sulfonyl"])
    assert acid.removed == {6, 7, 8}


def test_sulfoxide_nitramine_and_nitro_motifs():
    (sulfoxide,) = _motif_bridges("CS(=O)C", ["sulfoxide"])
    assert sulfoxide.removed == {1, 2}
    assert _motif_bridges("O=S1CCCC1", ["sulfoxide"]) == []  # S is a ring atom
    (nitramine,) = _motif_bridges("CN[N+](=O)[O-]", ["nitramine"])
    assert nitramine.removed == {1, 2, 3} and nitramine.endpoints == {0, 4}
    nitro = _motif_bridges("c1ccccc1[N+](=O)[O-]", ["aryl_nitro"])
    assert {b.environment for b in nitro} == {
        "[O-:1][N+:1]([c:2])=[O:2]",
        "[O:1]=[N+:1]([c:2])[O-:2]",
    }
    assert all(len(b.removed) == 2 for b in nitro)
    assert _motif_bridges("CCOCC", list(ec.MOTIFS)) == []


@pytest.mark.parametrize(
    ("motif", "smiles", "rule"),
    [
        ("aryl_sulfonyl", SULFONAMIDE, "aryl_sulfonamide_so2"),
        ("sulfoxide", "CS(=O)C", "sulfoxide_so_loss"),
        ("nitramine", "CN[N+](=O)[O-]", "n_nitramine_n2o"),
    ],
)
def test_motif_removed_atoms_equal_the_registered_rules_lost_atoms(motif, smiles, rule):
    rules = {r.name: r for r in MASSTER_NL_RULES}
    if rule not in rules:
        pytest.skip(f"{rule} is not registered")
    assert rule in ec.MOTIFS[motif].rules
    mol = Chem.MolFromSmiles(smiles)
    everything = frozenset(range(mol.GetNumAtoms()))
    # The prepass returns one side of the split; the lost atoms are the other side.
    lost = {
        everything - frozenset(np.flatnonzero(mask).tolist())
        for mask, _ in _apply_nl_prepass([rules[rule]], mol, mol.GetNumAtoms())
    }
    assert lost == ec.motif_removed_sets(mol, [motif])


def test_resolve_motifs():
    assert ec.resolve_motifs("none") == ()
    assert ec.resolve_motifs("all") == tuple(ec.MOTIFS)
    assert ec.resolve_motifs("sulfoxide,aryl_nitro") == ("sulfoxide", "aryl_nitro")
    with pytest.raises(ValueError, match="unknown motifs"):
        ec.resolve_motifs("sulfoxide,bogus")


# 4. Graft operator ----------------------------------------------------------------------


def test_graft_needs_a_terminal_o_or_f_outside_and_not_adjacent_within_four_bonds():
    mol = Chem.MolFromSmiles("FCCCCCC")  # F0, C1..C6 on a chain
    pool = [(4, 5, 6), (5, 6), (2, 3), (1, 2), (3, 4, 5, 6), (0, 1, 2), tuple(range(7))]
    grafts = ec.enumerate_grafts(mol, pool, max_distance=4)
    found = {(g.base, g.terminal, g.anchor, g.distance) for g in grafts}
    assert found == {
        ((4, 5, 6), 0, 4, 4),  # exactly four bonds away
        ((2, 3), 0, 2, 2),
        ((3, 4, 5, 6), 0, 3, 3),
    }
    # (5, 6) is five bonds away, (1, 2) is adjacent to F, and the rest contain F.
    assert all(g.base != (5, 6) and g.base != (1, 2) for g in grafts)
    edges = edge_tuple(mol)
    for graft in grafts:
        assert graft.terminal not in graft.base
        assert not _is_connected(graft.product, edges)
        assert formed_bond_count(graft.product, edges) == 1
        assert graft.product == tuple(sorted((*graft.base, graft.terminal)))


def test_graft_terminals_are_single_heavy_neighbour_o_and_f_only():
    mol = Chem.MolFromSmiles("CC(=O)OCC(F)(F)CO")
    symbols = {i: mol.GetAtomWithIdx(i).GetSymbol() for i in ec.terminal_atoms(mol)}
    # Carbonyl O, both F and the hydroxyl O; the ester O has two neighbours.
    assert Counter(symbols.values()) == Counter(O=2, F=2)
    assert 4 not in symbols


def test_graft_anchor_is_the_nearest_atom_of_the_base_with_lowest_index_on_ties():
    mol = Chem.MolFromSmiles("CC(O)C(C)C")  # O2 is terminal on C1; C3 and C0 are 2 away
    grafts = ec.enumerate_grafts(mol, [(3, 4, 5), (0,), (3,)], max_distance=4)
    assert {(g.base, g.anchor, g.distance) for g in grafts} == {
        ((3, 4, 5), 3, 2),
        ((0,), 0, 2),
        ((3,), 3, 2),
    }
    assert ec.enumerate_grafts(mol, [tuple(range(mol.GetNumAtoms()))]) == []


def test_graft_families_are_invariant_to_atom_order():
    mol = Chem.MolFromSmiles("FCCC(=O)CCN")
    reverse = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))

    def families(m: Chem.Mol) -> Counter:
        return Counter(g.environment for g in ec.enumerate_grafts(m, _connected_sets(m)))

    assert families(mol) == families(reverse)
    assert all(key.count(":1") == 1 for key in families(mol))  # exactly one terminal atom


# 5. Ceiling math ------------------------------------------------------------------------


def test_coverage_rise_on_a_toy_spectrum_and_its_bounds():
    w = np.array([0.6, 0.3, 0.1])
    n = (w**2).sum()
    a0, da = w[0] ** 2, w[1] ** 2
    rise = float(ec.coverage_rise(a0, da, n))
    assert rise == pytest.approx(np.sqrt((a0 + da) / n) - np.sqrt(a0 / n))
    assert rise == pytest.approx(0.10441, abs=1e-4)
    assert da / (2 * n) <= rise <= np.sqrt(da / n)
    assert float(ec.coverage_rise(0.0, da, n)) == pytest.approx(np.sqrt(da / n))
    assert float(ec.coverage_rise(a0, 0.0, n)) == 0.0
    rng = np.random.default_rng(0)
    for _ in range(50):
        peaks = rng.dirichlet(np.ones(6))
        mask = rng.random(6) < 0.5
        a, d = (peaks[mask] ** 2).sum(), (peaks[~mask] ** 2).sum() * rng.random()
        total = (peaks**2).sum()
        assert d / (2 * total) - 1e-12 <= float(ec.coverage_rise(a, d, total)) <= np.sqrt(d / total)


def _ceiling_fixture():
    w = np.array([0.6, 0.3, 0.1], np.float32)
    spectra = pd.DataFrame(
        {
            "spec_id": [1, 2, 3],
            "mol_id": [10, 10, 11],
            "n": float((w.astype(float) ** 2).sum()),
            "weights": [w, w, w],
            **{f"a0_{b}_{v}": float(w[0]) ** 2 for b in "BG" for v in ("strict", "opt")},
        }
    )
    records = pd.DataFrame(
        {
            "spec_id": [1, 1, 2, 3],
            "environment": ["a", "b", "a", "a"],
            "baseline": "B",
            "hit_strict": [(1,), (1, 2), (), (2,)],
            "hit_opt": [(1,), (1, 2), (), (2,)],
        }
    )
    return spectra, records


def test_per_spectrum_ceiling_unions_families_and_counts_a_peak_once():
    spectra, records = _ceiling_fixture()
    per = ec.per_spectrum_ceiling(spectra, records, ["a", "b"]).set_index("spec_id")
    w = np.array([0.6, 0.3, 0.1])
    assert per.da[1] == pytest.approx(w[1] ** 2 + w[2] ** 2)  # peak 1 hit by both, once
    assert per.dl1[1] == pytest.approx(w[1] + w[2])
    assert per.da[2] == 0 and per.rise[2] == 0
    assert per.rise[1] == pytest.approx(
        float(ec.coverage_rise(w[0] ** 2, w[1] ** 2 + w[2] ** 2, (w**2).sum()))
    )
    assert (per.lower <= per.rise + 1e-12).all() and (per.rise <= per.upper + 1e-12).all()
    mixed = records.assign(baseline=["B", "G", "B", "B"])
    with pytest.raises(ValueError, match="different baselines"):
        ec.per_spectrum_ceiling(spectra, mixed, ["a", "b"])


def test_stratum_ceiling_scaling_census_versus_sample():
    spectra, records = _ceiling_fixture()
    per = ec.per_spectrum_ceiling(spectra, records, ["a"])
    universe = pd.DataFrame({"mol_id": [10, 11, 12], "n_spectra": [10, 4, 6]})
    means = per.groupby("mol_id").rise.mean()
    census = ec.stratum_ceiling(per, universe, census=True, reps=500, seed=0)
    sample = ec.stratum_ceiling(per, universe, census=False, reps=500, seed=0)
    assert census["spectra"] == 20 and sample["spectra"] == 14
    assert census["rise"] == pytest.approx((10 * means[10] + 4 * means[11]) / 20)
    assert sample["rise"] == pytest.approx((10 * means[10] + 4 * means[11]) / 14)
    assert census["evaluated_molecules"] == 2 and census["molecules"] == 3
    for key in ("rise", "lower", "upper", "l1"):
        assert census[key + "_lo"] <= census[key] <= census[key + "_hi"] + 1e-12
    assert census["lower"] <= census["rise"] <= census["upper"]
    with pytest.raises(ValueError, match="missing from the stratum universe"):
        ec.stratum_ceiling(per, universe[universe.mol_id != 10], census=False, reps=10)


# 6. Unit evaluation ---------------------------------------------------------------------


def test_unit_products_baselines_and_family_restriction(scorer):
    mol = Chem.MolFromSmiles("O=C(c1ccccc1)NS(=O)(=O)c1ccc(cc1)[N+](=O)[O-]")
    unit = ec.build_unit(
        mol, scorer, _config(motifs=ec.resolve_motifs("all")), np.random.default_rng(0)
    )
    bridge_keys = {
        k for k, op in unit.operator_of.items() if op == "bridge" and not k.startswith("__")
    }
    assert bridge_keys == {
        "[O:1]=[N+:1]([c:2])[O-:2]",
        "[O-:1][N+:1]([c:2])=[O:2]",
        "[O:1]=[S:1](=[O:1])([c:2])[NH:2]",
    }
    graft_keys = {k for k in unit.products if k.startswith(ec.GRAFT_PREFIX)}
    assert graft_keys and all(unit.operator_of[k] == "graft" for k in graft_keys)
    assert unit.products["__union__"] == unit.products["__all__"] | unit.products["__graft__"]
    # Whole-precursor products (all atoms minus X, or one cut plus one transfer) are baseline only.
    whole_bridge = {
        tuple(a for a in range(mol.GetNumAtoms()) if a not in b.removed)
        for b in ec.enumerate_bridges(mol)
    }
    assert not (whole_bridge & unit.products["__all__"])
    edges = edge_tuple(mol)
    for product in unit.products["__graft__"]:
        assert not _is_connected(product, edges)
    # Restricting to one family or a wildcard changes only the evaluated products.
    only = ec.build_unit(
        mol,
        scorer,
        _config(
            motifs=ec.resolve_motifs("all"),
            families=frozenset({"*graft", "[O:1]=[N+:1]([c:2])[O-:2]"}),
        ),
        np.random.default_rng(0),
    )
    assert {
        k for k, op in only.operator_of.items() if op == "bridge" and not k.startswith("__")
    } == {"[O:1]=[N+:1]([c:2])[O-:2]"}
    assert only.products["__graft__"] == unit.products["__graft__"]
    no_graft = ec.build_unit(mol, scorer, _config(operators=("bridge",)), np.random.default_rng(0))
    assert "__graft__" not in no_graft.products and "__union__" not in no_graft.products
    # Every family's load counts distinct (composition, H) formulas, never more than its sets.
    for key, sets in unit.products.items():
        formulas, new = unit.loads[key]
        assert new <= formulas <= len(sets)


def test_excised_then_cleaved_peak_is_credited_once_and_scored_both_ways(scorer):
    mol = Chem.MolFromSmiles(SULFONAMIDE)
    unit = ec.build_unit(mol, scorer, _config(operators=("bridge",)), np.random.default_rng(0))
    # Phenyl plus N: the pool set phenyl+SO2+N minus the SO2 bridge. N is not next to the
    # ring, so no connected set has this composition.
    product = (0, 1, 2, 3, 4, 5, 9)
    environment = "[O:1]=[S:1](=[O:1])([c:2])[NH:2]"
    assert product in unit.products[environment]
    counts = unit.real_counts["strict"][product]
    assert counts == (ec.crossing_bond_count(product, edge_tuple(mol)), 1)
    masses, allowed_h = unit.cache.channels[product, counts]
    peak = masses[np.flatnonzero(allowed_h == 5)[0]] if (allowed_h == 5).any() else masses[0]
    row, records = ec.evaluate_spectrum_unit(
        unit,
        np.array([peak, 30.0]),
        np.array([0.9, 0.1]),
        (1e-5, 200.0),
        np.random.default_rng(0),
        "[M-H]-",
    )
    family = next(r for r in records if r["environment"] == environment)
    assert row["n"] == pytest.approx(0.82)
    assert family["hit_strict"] == (0,) and family["da_strict"] == pytest.approx(0.81)
    assert family["strict_unique"] == pytest.approx(0.9)
    assert family["da_opt"] >= family["da_strict"]
    assert row["a0_B_strict"] == pytest.approx(0.0)
    per = ec.per_spectrum_ceiling(
        pd.DataFrame([{**row, "spec_id": 1, "mol_id": 1, "weights": np.array([0.9, 0.1])}]),
        pd.DataFrame([{**family, "spec_id": 1}]),
        [environment],
    )
    assert per.rise[0] == pytest.approx(np.sqrt(0.81 / 0.82))
    assert per.lower[0] <= per.rise[0] <= per.upper[0]
    with pytest.raises(ValueError, match="does not match adduct index"):
        ec.evaluate_spectrum_unit(
            unit,
            np.array([peak]),
            np.array([1.0]),
            (1e-5, 200.0),
            np.random.default_rng(0),
            "[M+H]+",
        )


def test_capped_twin_search_is_never_credited_in_a_unit(scorer, monkeypatch):
    mol = Chem.MolFromSmiles(SULFONAMIDE)
    unit = ec.build_unit(mol, scorer, _config(operators=("bridge",)), np.random.default_rng(0))
    product = (0, 1, 2, 3, 4, 5, 9)
    counts = unit.real_counts["strict"][product]
    peak = unit.cache.channels[product, counts][0][0]
    monkeypatch.setattr(ec, "reachable_h", lambda *a, **k: None)
    unit.cache.reaches.clear()
    _, records = ec.evaluate_spectrum_unit(
        unit, np.array([peak]), np.array([1.0]), (1e-5, 200.0), np.random.default_rng(0), "[M-H]-"
    )
    family = next(r for r in records if r["environment"] == "__all__")
    assert family["da_strict"] == 0 and family["capped_strict"] >= 1


# 7. Selection, confirmation and promotion -----------------------------------------------


def _family_rows(name, molecules, diff, *, load=5.0, caps=0, missing=0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mol in range(molecules):
        real = 0.01 + diff + rng.normal(0, diff / 20)
        rows.append(
            {
                "environment": name,
                "mol_id": mol,
                "strict_unique": real,
                "strict_null": 0.01,
                "unique": real,
                "null": 0.01,
                "capped": caps if mol == 0 else 0,
                "null_capped": 0,
                "capped_strict": 0,
                "null_capped_strict": 0,
                "null_missing": missing if mol == 0 else 0,
                "load_formulas": load,
                "load_new_formulas": load / 2,
                "products": 3,
            }
        )
    return rows


def test_confirmation_needs_ten_molecules_no_caps_and_a_positive_adjusted_limit():
    val = pd.DataFrame(
        _family_rows("good", 14, 0.02)
        + _family_rows("heavy", 14, 0.02, load=12.0)
        + _family_rows("capped", 14, 0.02, caps=1)
        + _family_rows("few", 6, 0.02)
        + _family_rows("null", 14, 0.0, seed=5)
    )
    table = ec.confirm_families(
        val, ["good", "heavy", "capped", "few", "null"], reps=4000, seed=1
    ).set_index("environment")
    assert table.confirmed.to_dict() == {
        "good": True,
        "heavy": True,
        "capped": False,
        "few": False,
        "null": False,
    }
    # Confirmed families are promoted only when the mean load is at most ten formulas.
    assert table.promoted.to_dict() == {
        "good": True,
        "heavy": False,
        "capped": False,
        "few": False,
        "null": False,
    }
    assert table.loc["good", "strict_diff_pct"] == pytest.approx(2.0, abs=0.2)
    # The Bonferroni limit widens with the number of families.
    wide = ec.confirm_families(val, ["good"] * 1, reps=4000, seed=1).adjusted_lo[0]
    assert wide >= table.loc["good", "adjusted_lo"]


def test_confirming_no_families_returns_an_empty_typed_table():
    val = pd.DataFrame(_family_rows("__all__", 3, 0.0))
    table = ec.confirm_families(val, [], reps=10)
    assert table.empty and list(table.columns) == list(ec.CONFIRMATION_COLUMNS)
    assert table[table.promoted].environment.tolist() == []


def test_family_selection_excludes_aggregates_caps_and_small_families():
    summary = pd.DataFrame(
        {
            "environment": ["x", "__all__", "capped", "small", "flat", "missing"],
            "molecules": [12, 40, 12, 9, 12, 12],
            "strict_lo": [0.1, 0.5, 0.1, 0.1, -0.1, 0.1],
            "capped": [0, 0, 1, 0, 0, 0],
            "null_capped": 0,
            "capped_strict": 0,
            "null_capped_strict": 0,
            "null_missing": [0, 0, 0, 0, 0, 1],
        }
    )
    assert ec.select_families(summary) == ["x"]


def test_operator_decision_needs_a_positive_union_and_a_small_load():
    val = pd.DataFrame(
        _family_rows("__all__", 15, 0.02, load=4.0)
        + _family_rows("__graft__", 15, 0.02, load=30.0, seed=2)
    )
    table = ec.decide_operators(val, reps=3000, seed=0).set_index("environment")
    assert bool(table.loc["__all__", "model_operator"]) is True
    assert bool(table.loc["__graft__", "model_operator"]) is False
    assert table.loc["__graft__", "strict_lo"] > 0  # beats the null but exceeds the load cap


# 8. End to end on a tiny synthetic stratum ----------------------------------------------


def _tiny_run_files(tmp_path):
    smiles = {1: SULFONAMIDE, 2: "c1ccccc1S(=O)(=O)NCC", 3: "CCCCO", 4: "c1ccccc1[N+](=O)[O-]"}
    pd.DataFrame({"mol_id": list(smiles), "smiles": list(smiles.values())}).to_pickle(
        tmp_path / "mol.pkl"
    )
    peaks = [(50.0, 10.0), (100.0, 100.0), (156.0125, 40.0)]
    rows = [
        {
            "spec_id": 10 * m + k,
            "mol_id": m,
            "prec_type": "[M-H]-",
            "inst_type": "OB",
            "dset": "nist20_hr",
            "peaks": peaks,
        }
        for m in smiles
        for k in range(3)
    ]
    rows.append(
        {
            "spec_id": 999,
            "mol_id": 1,
            "prec_type": "[M-H]-",
            "inst_type": "QTOF",
            "dset": "nist20_hr",
            "peaks": peaks,
        }
    )
    pd.DataFrame(rows).to_pickle(tmp_path / "slice.pkl")
    membership = [
        {"spec_id": r["spec_id"], "ma_mi_split_spec": "train" if r["mol_id"] in (1, 3) else "val"}
        for r in rows
    ]
    pd.DataFrame(membership).to_csv(tmp_path / "split.csv", index=False)
    pd.DataFrame({"mol_id": list(smiles), "ma_mi_split": ["train", "val", "train", "val"]}).to_csv(
        tmp_path / "split_map.csv", index=False
    )
    return tmp_path


def _tiny_args(tmp_path, split, output, *extra):
    return [
        "--mode", "model_free", "--precursor-type", "[M-H]-", "--split", split,
        "--spec-slice", str(tmp_path / "slice.pkl"), "--spec-split", str(tmp_path / "split.csv"),
        "--mol-df", str(tmp_path / "mol.pkl"), "--split-map", str(tmp_path / "split_map.csv"),
        "--output", str(tmp_path / output), "--select", "bearing", "--motifs", "all",
        "--operators", "bridge,graft", "--bootstrap-reps", "50", "--spectra-per-molecule", "2",
        *extra,
    ]  # fmt: skip


def test_model_free_run_selects_bearing_molecules_and_freezes_families(tmp_path):
    _tiny_run_files(tmp_path)
    ec.main(_tiny_args(tmp_path, "train", "train.pkl"))
    frame = pd.read_pickle(tmp_path / "train.pkl")
    universe = pd.read_csv(tmp_path / "train.universe.csv")
    # Molecule 3 has no motif; the QTOF spectrum is outside the OB/OB_F stratum.
    assert set(frame.mol_id) == {1}
    assert universe.set_index("mol_id").bearing.to_dict() == {1: True, 3: False}
    assert universe.set_index("mol_id").n_spectra.to_dict() == {1: 3, 3: 3}
    assert frame.groupby("spec_id").size().nunique() == 1
    provenance = json.loads((tmp_path / "train.provenance.json").read_text())
    assert provenance["status"] == "complete" and provenance["census"] is True
    assert provenance["split_map_disagreeing_molecules"] == 0
    assert provenance["MASSTER_RULE_NAMES"] and provenance["git_diff_sha256"]
    assert set(provenance["inputs_md5"]) >= {"spec_slice", "spec_split", "mol_df"}
    assert (tmp_path / "train.families.json").exists()
    assert (tmp_path / "train.ceiling.csv").exists() and (tmp_path / "train.molecules.csv").exists()
    # A rerun with another process count and the same seed gives identical records.
    ec.main(_tiny_args(tmp_path, "train", "again.pkl", "--processes", "2"))
    pd.testing.assert_frame_equal(frame, pd.read_pickle(tmp_path / "again.pkl"))


def test_validation_of_a_motif_run_requires_frozen_families_and_confirm_checks_the_chain(tmp_path):
    _tiny_run_files(tmp_path)
    ec.main(_tiny_args(tmp_path, "train", "train.pkl"))
    with pytest.raises(ValueError, match="frozen train-selected"):
        ec.main(_tiny_args(tmp_path, "val", "val.pkl"))
    family = "[O:1]=[S:1](=[O:1])([c:2])[NH:2]"
    (tmp_path / "train.families.json").write_text(json.dumps([family]))
    ec.main(
        _tiny_args(tmp_path, "val", "val.pkl", "--families", str(tmp_path / "train.families.json"))
    )
    confirm = [
        "--train", str(tmp_path / "train.pkl"), "--val", str(tmp_path / "val.pkl"),
        "--families", str(tmp_path / "train.families.json"), "--output", str(tmp_path / "c"),
        "--reps", "200", "--min-molecules", "1",
    ]  # fmt: skip
    # The families file was edited after train wrote it, so the frozen chain is broken.
    with pytest.raises(ValueError, match="not the one this train run wrote"):
        ec.confirm_main(confirm)
    provenance = json.loads((tmp_path / "train.provenance.json").read_text())
    provenance["families_sha256"] = ec._sha256(tmp_path / "train.families.json")
    (tmp_path / "train.provenance.json").write_text(json.dumps(provenance))
    ec.confirm_main(confirm)
    confirmation = pd.read_csv(tmp_path / "c.confirmation.csv")
    assert confirmation.environment.tolist() == [family]
    assert (tmp_path / "c.operators.csv").exists() and (tmp_path / "c.ceiling.csv").exists()
    assert isinstance(json.loads((tmp_path / "c.trackb_families.json").read_text()), list)


def test_split_map_disagreement_is_counted_in_provenance(tmp_path):
    _tiny_run_files(tmp_path)
    pd.DataFrame({"mol_id": [1, 2, 3, 4], "ma_mi_split": ["val", "val", "train", "val"]}).to_csv(
        tmp_path / "split_map.csv", index=False
    )
    ec.main(_tiny_args(tmp_path, "train", "train.pkl"))
    provenance = json.loads((tmp_path / "train.provenance.json").read_text())
    assert provenance["split_map_disagreeing_molecules"] == 1
