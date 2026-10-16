"""Topology null of the rewire rules (arm a0null): shape, determinism, excluded real sites, stage and PL routing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch as th
from rdkit import Chem

import fragnnet.models.masster.candidates.rewire_null as N
from fragnnet.models.masster import masster_pl as masster_pl_module
from fragnnet.models.masster.candidates.rewire_rules import (
    REWIRE_RULES,
    build_rewire_matches,
    raw_rule_matches,
    _PATTERNS,
)
from tests.test_masster_rewire_model import _pair, _run
from tests.test_masster_rewire_pl_export import _REWIRE_ON, _batch, _pl

ROOT = Path(__file__).parents[1]
GATE = ROOT / "scripts/debug/smarts/debug_masster_rewire_rule_gate.py"
SULFONE = "O=S(=O)(c1ccccc1)c1ccccc1"
MOLECULES = (
    SULFONE,
    "CS(=O)(=O)c1ccccc1",
    "O=S1(=O)c2ccccc2-c2ccccc12",
    "CC(C)(C)c1ccc(cc1)S(=O)(=O)Nc1ccccc1",
    "c1ccccc1N=Nc1ccccc1",
    "c1ccccc1Oc1ccccc1",
    "c1ccccc1Sc1ccccc1",
    "CCSCC",
    "COP(=O)(OC)SCCC(=O)OC",
    "O=[N+]([O-])c1ccc(O)c([N+](=O)[O-])c1",
    "CC(=O)Oc1ccccc1C(=O)O",
)


def _gate():
    spec = importlib.util.spec_from_file_location("gate_for_null_test", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rule_dict(rule) -> dict:
    return {
        "name": rule.name,
        "smarts": rule.smarts,
        "delete": list(rule.delete),
        "cut": [list(p) for p in rule.cut],
        "form": [list(p) for p in rule.form],
    }


def _terminal(atom: Chem.Atom) -> bool:
    # "!D1" is non-terminal; the gate's regex (used on REAL patterns) would also match it.
    text = atom.GetSmarts()
    return N._is_terminal_query(atom) and "!D1" not in text and "!X1" not in text


def test_auto_null_strings_equal_the_gate_for_all_rules() -> None:
    gate = _gate()
    for rule in REWIRE_RULES:
        assert N.topology_null_smarts(rule) == gate.topology_null_smarts(_rule_dict(rule)), rule.name


def test_null_smarts_overrides_are_used_verbatim() -> None:
    overridden = [r for r in REWIRE_RULES if r.null_smarts is not None]
    assert len(overridden) == 3
    for rule in overridden:
        assert N.null_pattern_smarts(rule) == rule.null_smarts
        index = REWIRE_RULES.index(rule)
        assert Chem.MolToSmarts(N.NULL_PATTERNS[index]) == Chem.MolToSmarts(Chem.MolFromSmarts(rule.null_smarts))


def test_expected_auto_nulls_of_the_spec() -> None:
    by_name = {r.name: N.null_pattern_smarts(r) for r in REWIRE_RULES}
    assert by_name["sulfonyl_so2_extrusion"] == "[*:1]~[!D1:2]~[*:5]"
    assert by_name["s_to_o_aromatic_C"] == "[*:1]~[!D1:2]~[D1:3]"
    assert by_name["nitro_nitrite_NO_loss"] == "[*:1]~[!D1:2]~[D1:4]"


@pytest.mark.parametrize("rule_name", ["sulfonyl_so2_extrusion", "s_to_o_aromatic_C", "diaryl_ether_o_excision"])
def test_gate_draw_parity_through_the_seed_key(rule_name) -> None:
    """The gate-parity function reproduces the reach-ceiling script's first null draw (same key construction)."""
    gate = _gate()
    index = [r.name for r in REWIRE_RULES].index(rule_name)
    rule = REWIRE_RULES[index]
    rd = _rule_dict(rule)
    rd["_null_pattern"] = Chem.MolFromSmarts(rule.null_smarts or gate.topology_null_smarts(rd))
    compared = 0
    for n, smiles in enumerate(MOLECULES):
        mol = Chem.MolFromSmiles(smiles)
        key = f"nist20:{n}"
        real = gate._matches(mol, Chem.MolFromSmarts(rule.smarts))
        reference = _reference_null(gate, mol, rd, real, key)
        ours = N.gate_null_matches(mol, index, key)
        assert [tuple(sorted(h.items())) for h in ours] == [tuple(sorted(h.items())) for h in reference]
        compared += bool(reference)
    assert compared >= 1


def _reference_null(gate, mol, rule, real, key):
    import zlib

    import numpy as np

    rng = np.random.default_rng(zlib.crc32(f"{key}|{rule['name']}".encode()))
    null_maps = [a.GetAtomMapNum() for a in rule["_null_pattern"].GetAtoms()]
    real_atoms = {frozenset(m[k] for k in null_maps if k in m) for m in real}
    pool = [m for m in gate._matches(mol, rule["_null_pattern"]) if frozenset(m.values()) not in real_atoms]
    if not real or not pool:
        return []
    return [pool[i] for i in rng.choice(len(pool), min(len(real), len(pool)), replace=False)]


def test_production_seed_is_reproducible_and_keyed_on_the_smiles() -> None:
    mol = Chem.MolFromSmiles("CCSCC(=O)Nc1ccc(Sc2ccccc2)cc1")
    first = N.build_null_rewire_matches(mol, max_matches_per_rule=8)
    assert first == N.build_null_rewire_matches(mol, max_matches_per_rule=8)
    assert N.null_seed("CCO", "a") == N.null_seed("CCO", "a") != N.null_seed("CCO", "b")


@pytest.mark.parametrize("smiles", MOLECULES)
def test_real_sites_excluded_counts_and_shape(smiles) -> None:
    mol = Chem.MolFromSmiles(smiles)
    real = build_rewire_matches(mol, max_matches_per_rule=10**9)
    null = N.build_null_rewire_matches(mol, max_matches_per_rule=10**9)
    for index, rule in enumerate(REWIRE_RULES):
        r = [m for m in real.matches if m.rule_index == index]
        n = [m for m in null.matches if m.rule_index == index]
        if not r:
            assert not n  # a rule with no real match yields no null
            continue
        assert len(n) <= len(r)
        real_sites = N._real_sites(mol, index)
        pattern = N.NULL_PATTERNS[index]
        maps = {a.GetAtomMapNum(): a for a in pattern.GetAtoms() if a.GetAtomMapNum()}
        for match in n:
            # the null site never equals a real site (compared on the null pattern's atoms)
            assert frozenset(match.motif_atoms) not in set(real_sites)
            for a, b in match.cut_pairs + match.form_pairs:
                assert mol.GetBondBetweenAtoms(a, b) is not None or (a, b) in match.form_pairs
            for a, b in match.cut_pairs:
                assert mol.GetBondBetweenAtoms(a, b) is not None
        pool_size = _pool_size(mol, index, real_sites)
        assert len(n) == min(len(r), pool_size)
        # shape: terminal pattern atoms are degree 1, others are not
        for hit in _null_hits(mol, index, real_sites):
            for m, idx in hit.items():
                deg = mol.GetAtomWithIdx(idx).GetDegree()
                if _terminal(maps[m]):
                    assert deg == 1
                elif "!D1" in maps[m].GetSmarts():
                    assert deg > 1


def _null_hits(mol, index, real_sites):
    return [h for h in raw_rule_matches(mol, N.NULL_PATTERNS[index]) if frozenset(h.values()) not in set(real_sites)]


def _pool_size(mol, index, real_sites) -> int:
    seen = set()
    for hit in _null_hits(mol, index, real_sites):
        m = N._to_match(mol, index, hit)
        seen.add(N._edit_key(m))
    return len(seen)


def test_empty_pool_yields_no_null_and_truncation_is_counted() -> None:
    # Methanesulfonyl fluoride-like cases are not needed: use a molecule whose real match has an empty pool.
    mol = Chem.MolFromSmiles("O=S1(=O)c2ccccc2-c2ccccc12")
    full = N.build_null_rewire_matches(mol, max_matches_per_rule=10**9)
    capped = N.build_null_rewire_matches(mol, max_matches_per_rule=1)
    per_rule = {}
    for m in full.matches:
        per_rule[m.rule_index] = per_rule.get(m.rule_index, 0) + 1
    assert len(capped.matches) == len(per_rule)
    assert capped.num_truncated == len(full.matches) - len(capped.matches)


def test_null_and_real_use_the_same_vocabulary_columns_and_comparable_counts() -> None:
    real_total = null_total = 0
    for smiles in MOLECULES:
        mol = Chem.MolFromSmiles(smiles)
        real = build_rewire_matches(mol, max_matches_per_rule=8)
        null = N.build_null_rewire_matches(mol, max_matches_per_rule=8)
        assert {m.rule_index for m in null.matches} <= {m.rule_index for m in real.matches}
        assert all(0 <= m.rule_index < len(REWIRE_RULES) for m in null.matches)
        real_total += len(real.matches)
        null_total += len(null.matches)
    assert 0 < null_total <= real_total


def test_null_edits_go_through_the_stage() -> None:
    mol = Chem.MolFromSmiles(SULFONE)
    null = N.build_null_rewire_matches(mol, max_matches_per_rule=8)
    real = build_rewire_matches(mol, max_matches_per_rule=8)
    assert null.matches
    batch, _, on = _pair((SULFONE,))
    from fragnnet.models.masster.candidates.builder import MassterStaticGraph

    def run(matches):
        graph = MassterStaticGraph(
            chemical_bond_index=batch.chemical_bond_index,
            atom_batch_indices=batch.batch,
            atom_ptr=batch.ptr,
            rewire_matches=(matches,),
        )
        with th.no_grad():
            return on(
                mol_pyg=batch,
                spec_prec_type=th.zeros(1, dtype=th.long),
                spec_nce=th.full((1,), 30.0),
                static_graph=graph,
            )

    out_real, out_null = run(real), run(null)
    assert th.isfinite(out_null["pred_logprobs"]).all()
    assert int(out_null["rewire_extra_count"].sum()) > 0
    # The edits themselves are different atom sets, and the stage's output follows them.
    assert {m.motif_atoms for m in null.matches}.isdisjoint({m.motif_atoms for m in real.matches})
    real_mzs = th.sort(out_real["pred_mzs"]).values
    null_mzs = th.sort(out_null["pred_mzs"]).values
    assert real_mzs.shape != null_mzs.shape or not th.equal(real_mzs, null_mzs)


def test_pl_null_flag_builds_a_null_static_graph_with_identical_state_dict_layout() -> None:
    masster_pl_module._cached_masster_rewire_matches.cache_clear()
    real_pl = _pl(**_REWIRE_ON)
    null_pl = _pl(**_REWIRE_ON, masster_rewire_null=True)
    batch = _batch(real_pl, (SULFONE, "CCO"))
    real_graph = real_pl._build_static_graph(batch)
    null_graph = null_pl._build_static_graph(batch)
    assert null_graph.rewire_matches[0].matches
    assert null_graph.rewire_matches[0] != real_graph.rewire_matches[0]
    assert len(null_graph.rewire_matches[1].matches) == 0
    r, n = real_pl.state_dict(), null_pl.state_dict()
    assert r.keys() == n.keys() and all(r[k].shape == n[k].shape for k in r)
    assert real_pl.hparams.masster_rewire_null is False and null_pl.hparams.masster_rewire_null is True
