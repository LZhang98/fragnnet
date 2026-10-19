"""Graph edits, nulls and reachability of the rewire rule gate on hand-checked molecules."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

import scripts.debug.smarts.debug_masster_rewire_rule_gate as G
from scripts.debug.smarts.debug_masster_smarts_rule_gate import edge_tuple

SO2_EXTRUSION = {
    "name": "sulfonyl_so2_extrusion",
    "smarts": "[#6:1][SX4:2](=[OX1:3])(=[OX1:4])[!#1:5]",
    "delete": [2, 3, 4],
    "cut": [],
    "form": [[1, 5]],
}
S_TO_O = {
    "name": "s_to_o_aromatic_C",
    "smarts": "[c:1][SX4:2](=[OX1:3])(=[OX1:4])[!#1:5]",
    "delete": [],
    "cut": [[1, 2], [2, 3]],
    "form": [[1, 3]],
}
ETHER = {"name": "diaryl_ether_o_excision", "smarts": "[c:1][OX2:2][c:3]", "delete": [2], "cut": [], "form": [[1, 3]]}
CATALOG = Path("output/rewire_rule_gate_2026_10_15/rule_catalog.json")


def _reach(smiles: str, rule: dict | None, composition: dict, h: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if rule is None:
        edge_sets = [edge_tuple(mol)]
    else:
        pattern = Chem.MolFromSmarts(rule["smarts"])
        edge_sets = [G._edit_edges(mol, m, rule) for m in G._matches(mol, pattern)]
    return G._reaches(mol, edge_sets, Counter(composition), h, sum(composition.values()))


def test_so2_extrusion_reaches_the_diarylamine_ion_only_after_the_edit() -> None:
    # PhSO2NHPh: [M+H-SO2]+ = C12H12N+, own H of C12N = 11.
    smiles = "O=S(=O)(Nc1ccccc1)c1ccccc1"
    assert _reach(smiles, None, {"C": 12, "N": 1}, 12) == "miss"
    assert _reach(smiles, SO2_EXTRUSION, {"C": 12, "N": 1}, 12) == "hit"


def test_s_to_o_migration_reaches_phenoxy_ions_with_continuation() -> None:
    # PhSO2NH2: Ar migrates to O; Ph-O (C6O) is a connected set only after the edit.
    smiles = "NS(=O)(=O)c1ccccc1"
    assert _reach(smiles, None, {"C": 6, "O": 1}, 6) == "miss"
    assert _reach(smiles, S_TO_O, {"C": 6, "O": 1}, 6) == "hit"
    # Continuation: a smaller piece containing the new C-O bond (C5O after ordinary cuts).
    assert _reach(smiles, S_TO_O, {"C": 5, "O": 1}, 4) == "hit"


def test_ether_excision_reaches_biphenyl() -> None:
    smiles = "c1ccc(Oc2ccccc2)cc1"
    assert _reach(smiles, None, {"C": 12}, 10) == "miss"
    assert _reach(smiles, ETHER, {"C": 12}, 10) == "hit"


def test_edit_drops_deleted_atoms_and_cut_bonds_and_adds_formed_bonds() -> None:
    mol = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)c1ccccc1")
    # _matches keeps one mapping per =O permutation; any S-N one gives the same edit.
    match = next(m for m in G._matches(mol, Chem.MolFromSmarts(SO2_EXTRUSION["smarts"])) if mol.GetAtomWithIdx(m[5]).GetSymbol() == "N")
    edges = set(zip(*G._edit_edges(mol, match, SO2_EXTRUSION)))
    deleted = {match[2], match[3], match[4]}
    assert not any(a in deleted or b in deleted for a, b in edges)
    assert (match[1], match[5]) in edges
    assert len(edges) == mol.GetNumBonds() - 4 + 1


@pytest.mark.parametrize(
    ("rule", "expected"),
    [(SO2_EXTRUSION, "[*:1]~[!D1:2]~[*:5]"), (S_TO_O, "[*:1]~[!D1:2]~[D1:3]"), (ETHER, "[*:1]~[!D1:2]~[*:3]")],
)
def test_null_keeps_the_edit_shape_with_any_elements(rule: dict, expected: str) -> None:
    assert G.topology_null_smarts(rule) == expected


def test_null_draws_are_reproducible_and_avoid_the_real_site() -> None:
    G._init([dict(SO2_EXTRUSION)], "[M+H]+")
    smiles = "CCOc1ccc(cc1)S(=O)(=O)NCc1ccccc1"
    rows_a = G._work(("t:1", smiles, ["C15H16NO"]))[1]
    rows_b = G._work(("t:1", smiles, ["C15H16NO"]))[1]
    assert rows_a == rows_b
    mol = Chem.MolFromSmiles(smiles)
    rule = G._RULES[0]
    real = {frozenset(m[k] for k in (1, 2, 5)) for m in G._matches(mol, rule["_pattern"])}
    pool = [m for m in G._matches(mol, rule["_null_pattern"]) if frozenset(m.values()) not in real]
    assert pool and all(frozenset(m.values()) not in real for m in pool)


@pytest.mark.skipif(not CATALOG.exists(), reason="rule catalog not generated on this machine")
def test_every_catalog_rule_is_a_valid_edit() -> None:
    rules = json.loads(CATALOG.read_text())["rules"]
    assert len({r["name"] for r in rules}) == len(rules)
    for rule in rules:
        pattern = Chem.MolFromSmarts(rule["smarts"])
        assert pattern is not None, rule["name"]
        maps = {a.GetAtomMapNum(): a.GetIdx() for a in pattern.GetAtoms() if a.GetAtomMapNum()}
        used = set(rule["delete"]) | {m for pair in rule["cut"] + rule["form"] for m in pair}
        assert used <= set(maps), rule["name"]
        for a, b in rule["cut"]:
            assert pattern.GetBondBetweenAtoms(maps[a], maps[b]) is not None, (rule["name"], a, b)
        for a, b in rule["form"]:
            assert pattern.GetBondBetweenAtoms(maps[a], maps[b]) is None, (rule["name"], a, b)
            assert a not in rule["delete"] and b not in rule["delete"], rule["name"]
        assert rule["status"] in {"admitted", "candidate", "unreviewed"}
        assert G.topology_null_smarts(rule) and Chem.MolFromSmarts(rule.get("null_smarts") or G.topology_null_smarts(rule))


def test_bootstrap_ci_brackets_a_clear_difference() -> None:
    import pandas as pd

    per_mol = pd.DataFrame({"real": np.full(30, 0.6), "null": np.full(30, 0.1), "w": np.ones(30)})
    lo, hi = G.bootstrap_diff(per_mol)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
