"""Topology null for the rewire rules (arm a0null): same edit shape at random sites, real sites excluded.

A port of the gate's ``topology_null_smarts`` / ``null_matches`` (scripts/debug/smarts/debug_masster_rewire_rule_gate.py,
debug_masster_rewire_reach_ceiling.py). Elements and bond orders are wildcarded, terminal atoms stay terminal, non-terminal
atoms stay non-terminal. The draw is deterministic per (canonical SMILES, rule name). Pure Python and RDKit.

Status: PROTOTYPE, a control arm only (a0null, MSGYM and NIST20). Used only with
``masster_rewire_null: true``, which needs ``masster_rewire_rules``. Never part of a production
model.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
from rdkit import Chem

from fragnnet.models.masster.candidates.rewire_rules import (
    _PATTERNS,
    REWIRE_RULES,
    RewireMatch,
    RewireMatchSet,
    RewireRule,
    _pairs,
    atom_map_numbers,
    build_rewire_matches,
    raw_rule_matches,
)


def _is_terminal_query(atom: Chem.Atom) -> bool:
    return bool(re.search(r"(X1|D1)(?!\d)", atom.GetSmarts()))


def topology_null_smarts(rule: RewireRule) -> str:
    """The edit's shape with any elements and bond orders (verbatim port of the gate's function)."""
    pattern = Chem.MolFromSmarts(rule.smarts)
    by_map = {a.GetAtomMapNum(): a for a in pattern.GetAtoms() if a.GetAtomMapNum()}
    touched = {m for pair in tuple(rule.form) + tuple(rule.cut) for m in pair}
    deleted = set(rule.delete)
    keep = set(touched)
    for m in deleted:
        atom = by_map[m]
        if not (_is_terminal_query(atom) and any(n.GetAtomMapNum() in deleted for n in atom.GetNeighbors())):
            keep.add(m)
    keep_idx = {by_map[m].GetIdx(): m for m in keep}
    pieces = {}
    for idx, m in keep_idx.items():
        atom = pattern.GetAtomWithIdx(idx)
        if _is_terminal_query(atom):
            pieces[idx] = f"[*;D1:{m}]"
        elif atom.GetDegree() > 1 or m in deleted:
            pieces[idx] = f"[*;!D1:{m}]"
        else:
            pieces[idx] = f"[*:{m}]"
    order = sorted(pieces)
    rw = Chem.RWMol(Chem.MolFromSmarts(".".join(pieces[i] for i in order)))
    local = {idx: k for k, idx in enumerate(order)}
    for bond in pattern.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in local and b in local:
            rw.AddBond(local[a], local[b], Chem.BondType.SINGLE)
    return Chem.MolToSmarts(rw).replace("-", "~")


def null_pattern_smarts(rule: RewireRule) -> str:
    return rule.null_smarts or topology_null_smarts(rule)


NULL_PATTERNS: tuple[Chem.Mol, ...] = tuple(Chem.MolFromSmarts(null_pattern_smarts(r)) for r in REWIRE_RULES)


def null_seed(key: str, rule_name: str) -> int:
    return zlib.crc32(f"{key}|{rule_name}".encode())


def _to_match(mol: Chem.Mol, rule_index: int, hit: dict[int, int]) -> RewireMatch:
    rule = REWIRE_RULES[rule_index]
    atoms = tuple(sorted(hit.values()))
    deleted = tuple(sorted(hit[m] for m in rule.delete if m in hit))
    cut = _pairs(rule.cut, hit)
    form = _pairs(rule.form, hit)
    numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms)
    heavy = sum(1 for i in deleted if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
    return RewireMatch(rule_index, atoms, numbers, deleted, cut, form, heavy)


def _edit_key(m: RewireMatch) -> tuple:
    return (m.motif_atoms, m.deleted, m.cut_pairs, m.form_pairs)


def _rank_key(m: RewireMatch, ranks: list[int]) -> tuple:
    """The match's edit expressed in canonical atom ranks (symmetry-equivalent matches tie)."""
    pairs = lambda ps: tuple(sorted(tuple(sorted(ranks[a] for a in p)) for p in ps))  # noqa: E731
    return (
        tuple(sorted(ranks[a] for a in m.motif_atoms)),
        tuple(sorted(ranks[a] for a in m.deleted)),
        pairs(m.cut_pairs),
        pairs(m.form_pairs),
    )


def _real_sites(mol: Chem.Mol, rule_index: int) -> list[frozenset]:
    maps = atom_map_numbers(NULL_PATTERNS[rule_index])
    return [frozenset(h[k] for k in maps if k in h) for h in raw_rule_matches(mol, _PATTERNS[rule_index])]


def gate_null_matches(mol: Chem.Mol, rule_index: int, key: str) -> list[dict[int, int]]:
    """The gate's FIRST null draw on raw lists with raw counts (parity reference only)."""
    real = _real_sites(mol, rule_index)
    if not real:
        return []
    real_set = set(real)
    pool = [h for h in raw_rule_matches(mol, NULL_PATTERNS[rule_index]) if frozenset(h.values()) not in real_set]
    if not pool:
        return []
    rng = np.random.default_rng(null_seed(key, REWIRE_RULES[rule_index].name))
    return [pool[i] for i in rng.choice(len(pool), min(len(real), len(pool)), replace=False)]


def build_null_rewire_matches(
    mol: Chem.Mol, *, max_matches_per_rule: int, seed_key: str | None = None
) -> RewireMatchSet:
    """Per rule with a real match: as many null matches as DEDUPED real matches (or the pool size when smaller)."""
    key = Chem.MolToSmiles(mol) if seed_key is None else seed_key
    real_counts: dict[int, int] = {}
    for m in build_rewire_matches(mol, max_matches_per_rule=10**9).matches:
        real_counts[m.rule_index] = real_counts.get(m.rule_index, 0) + 1
    out: list[RewireMatch] = []
    num_truncated = 0
    for rule_index, rule in enumerate(REWIRE_RULES):
        n_real = real_counts.get(rule_index, 0)
        if n_real == 0:
            continue
        real_set = set(_real_sites(mol, rule_index))
        pool: dict[tuple, RewireMatch] = {}
        for hit in raw_rule_matches(mol, NULL_PATTERNS[rule_index]):
            if frozenset(hit.values()) in real_set:
                continue
            match = _to_match(mol, rule_index, hit)
            pool.setdefault(_edit_key(match), match)
        if not pool:
            continue
        # Draw by position in an order that does not depend on the input atom numbering.
        ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
        pooled = sorted(pool.values(), key=lambda m: _rank_key(m, ranks))
        rng = np.random.default_rng(null_seed(key, rule.name))
        picked = sorted(
            (pooled[i] for i in rng.choice(len(pooled), min(n_real, len(pooled)), replace=False)), key=_edit_key
        )
        num_truncated += max(0, len(picked) - max_matches_per_rule)
        out.extend(picked[:max_matches_per_rule])
    return RewireMatchSet(tuple(out), num_truncated)
