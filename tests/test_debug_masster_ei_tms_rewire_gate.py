"""Regression checks for the NIST23 EI TMS rewire gate."""

import numpy as np
import pytest
from rdkit import Chem

from fragnnet.models.masster.candidates.rewire_null import topology_null_smarts
from scripts.debug.ei.debug_masster_ei_nominal_coverage import pool_sets
from scripts.debug.ei.debug_masster_ei_tms_rewire_gate import (
    CANDIDATES,
    MAX_MATCHES,
    RewireChannelCalculator,
    bootstrap_ci,
    boundary_count,
    build_contexts,
    channel_table,
    composition_key,
    edited_products,
    exact_reach,
    gate_rules,
    isotope_consistency,
    isotope_support,
    m1_over_m,
    null_matches,
    own_hydrogens,
    prepare_spectrum,
    real_matches,
)

GLYCEROL = "C[Si](C)(C)OCC(O[Si](C)(C)C)CO[Si](C)(C)C"
HYDROXYBUTYRATE = "CC(O[Si](C)(C)C)CC(=O)O[Si](C)(C)C"
GLYCINE = "C[Si](C)(C)N(CC(=O)O[Si](C)(C)C)[Si](C)(C)C"
CHOLESTEROL = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O[Si](C)(C)C)C)C"
CONTEXTS = {c.rule.name: c for c in build_contexts(CANDIDATES)}
T1 = "tms_siloxy_silylium"


def bonds_of(mol):
    return [frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()]


def whole_products(mol, name):
    matches, _, _ = real_matches(mol, CONTEXTS[name])
    products = edited_products(
        [frozenset(range(mol.GetNumAtoms()))], matches[:MAX_MATCHES], bonds_of(mol)
    )
    return matches, products


def edges_of(mol):
    return tuple(zip(*[tuple(b) for b in bonds_of(mol)], strict=True))


@pytest.mark.parametrize("smiles", [GLYCEROL, GLYCINE])
def test_t1_product_is_c5h15osi2_with_valence_correction(smiles):
    mol = Chem.MolFromSmiles(smiles)
    _, products = whole_products(mol, T1)
    assert len(products) > 0
    for atoms, (n_formed, n_internal_cut) in products.items():
        ordered = tuple(sorted(atoms))
        assert dict(composition_key(mol, ordered)) == {"C": 5, "O": 1, "Si": 2}
        assert own_hydrogens(mol, ordered) == 15
        assert (n_formed, n_internal_cut) == (1, 0)


def test_t1_edit_distinct_match_counts_and_truncation():
    glycerol, _ = real_matches(Chem.MolFromSmiles(GLYCEROL), CONTEXTS[T1])[:2]
    two_tms, _ = real_matches(Chem.MolFromSmiles(HYDROXYBUTYRATE), CONTEXTS[T1])[:2]
    assert len(glycerol) == 18  # 3 acceptors x 2 donors x 3 methyls
    assert len(two_tms) == 6
    assert len(glycerol[:MAX_MATCHES]) == 8


def test_t1_nominal_147_channel_is_valid_at_shift_zero():
    mol = Chem.MolFromSmiles(GLYCEROL)
    _, products = whole_products(mol, T1)
    atoms = tuple(sorted(next(iter(products))))
    calc = RewireChannelCalculator(("[M]+",))
    table = channel_table(mol, calc, {atoms: -2}, bonds_of(mol))
    assert table.mass[atoms] == 147
    assert boundary_count(bonds_of(mol), frozenset(atoms)) == 3
    assert table.valid[atoms][4]  # shift 0
    assert 147 in isotope_support(mol, table, [atoms])
    assert 148 in isotope_support(mol, table, [atoms])


@pytest.mark.parametrize("smiles", [CHOLESTEROL, "CCO", "c1ccccc1"])
def test_no_t1_match_without_two_tms_groups(smiles):
    matches, capped, hits = real_matches(Chem.MolFromSmiles(smiles), CONTEXTS[T1])
    assert matches == []
    assert hits == []
    assert not capped


@pytest.mark.parametrize(
    "smiles,name,reachable",
    [
        (GLYCEROL, T1, False),
        (HYDROXYBUTYRATE, T1, False),
        (GLYCEROL, "vicinal_otms_geminal", True),
        (HYDROXYBUTYRATE, "silyl_to_ester_carbonyl", True),
    ],
)
def test_composition_reach_on_prototype_molecules(smiles, name, reachable):
    mol = Chem.MolFromSmiles(smiles)
    _, products = whole_products(mol, name)
    assert products
    verdicts = {exact_reach(mol, edges_of(mol), tuple(sorted(a)), 50_000) for a in products}
    assert verdicts == {reachable}


def test_t2_and_t3_products_have_literature_compositions():
    t2 = whole_products(Chem.MolFromSmiles(GLYCEROL), "vicinal_otms_geminal")[1]
    t3 = whole_products(Chem.MolFromSmiles(HYDROXYBUTYRATE), "silyl_to_ester_carbonyl")[1]
    mol2, mol3 = Chem.MolFromSmiles(GLYCEROL), Chem.MolFromSmiles(HYDROXYBUTYRATE)
    a2, a3 = (tuple(sorted(next(iter(p)))) for p in (t2, t3))
    assert dict(composition_key(mol2, a2)) == {"C": 7, "O": 2, "Si": 2}
    assert dict(composition_key(mol3, a3)) == {"C": 8, "O": 2, "Si": 2}


def test_null_is_two_component_same_shape_seeded_and_excludes_real_sites():
    rule = CANDIDATES[0]
    assert "." in topology_null_smarts(rule)
    mol = Chem.MolFromSmiles(GLYCEROL)
    ctx = CONTEXTS[T1]
    matches, _, hits = real_matches(mol, ctx)
    first, pool, _ = null_matches(mol, ctx, hits, len(matches), np.random.default_rng(1))
    second, _, _ = null_matches(mol, ctx, hits, len(matches), np.random.default_rng(1))
    assert first == second
    assert 0 < len(first) <= MAX_MATCHES
    assert pool > 0
    real_sites = {frozenset(h[k] for k in ctx.null_maps if k in h) for h in hits}
    assert all(frozenset(m.atoms) not in real_sites for m in first)
    # Same edit shape: every null edit cuts the rule's number of bonds and forms one.
    assert all(len(m.cut) == len(rule.cut) and len(m.form) == 1 for m in first)


def test_isotope_ratios_distinguish_si2_from_the_reachable_composition():
    assert m1_over_m((("C", 5), ("O", 1), ("Si", 2)), 15) == pytest.approx(0.158, abs=0.001)
    assert m1_over_m((("C", 6), ("O", 2), ("Si", 1)), 15) == pytest.approx(0.118, abs=0.001)


def test_isotope_consistency_prefers_si2_for_a_si2_like_ratio():
    mol = Chem.MolFromSmiles(GLYCEROL)
    _, products = whole_products(mol, T1)
    calc = RewireChannelCalculator(("[M]+",))
    sets = pool_sets(mol)
    base = {"sets": sets, "table": channel_table(mol, calc, dict.fromkeys(sets, 0), bonds_of(mol))}
    peaks = [[73.0, 100.0], [147.0, 50.0], [148.0, 50.0 * 0.16]]
    result = isotope_consistency(mol, prepare_spectrum(peaks), base, products)
    assert result["observed"] == pytest.approx(0.16)
    assert result["ratio_si2"] == pytest.approx(0.158, abs=0.001)
    assert result["si2_closer"]
    assert result["err_si2"] < result["err_competitor"]
    peaks = [[73.0, 100.0], [147.0, 50.0], [148.0, 50.0 * 0.118]]
    assert not isotope_consistency(mol, prepare_spectrum(peaks), base, products)["si2_closer"]


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    values = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.12, 0.22])
    first = bootstrap_ci(values, seed=7)
    assert first == bootstrap_ci(values, seed=7)
    assert first[0] < values.mean() < first[1]
    assert np.isnan(bootstrap_ci(values[:2], seed=7)).all()


def test_imported_rules_skip_candidate_names():
    names = [r.name for r in gate_rules()]
    assert len(names) == len(set(names))
    assert names[: len(CANDIDATES)] == [r.name for r in CANDIDATES]
    assert "sulfonyl_so2_extrusion" in names
