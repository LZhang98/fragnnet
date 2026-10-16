"""Rewire rules for MASSter: admitted edit rules, vocabulary, spec hashes and per-molecule matches.

Each rule is a mapped SMARTS plus an edit (delete atoms, cut bonds, form bonds). The 17 rules are the ``admitted``
entries of the rule gate catalog (``output/rewire_rule_gate_2026_10_15/rule_catalog.json``, gitignored), copied here.
The vocabulary is separate from ``MASSTER_RULE_VOCABULARY`` and append-only. Pure Python and RDKit, no torch.

Status: PROTOTYPE. Applied only with ``masster_rewire_rules: true`` (off by default and in the
current recipe). The MSGYM screen ran 2026-10-16: on validation every arm is within noise of the
control and a0 equals a0null on cosine; retrieval and the NIST20 arms are prepared
(docs/plans/masster/masster_rewire_rules_training_plan.md). `REWIRE_RULE_VOCABULARY` is imported by
export code whether or not the rules are on.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, replace
from functools import cache

from rdkit import Chem

PRECURSOR_TYPES_ALLOWED = ("[M+H]+", "[M-H]-", "[M+NH4]+", "[M+Na]+")  # the only values a rule may list
_MAX_RAW_MATCHES = 5000  # the gate's GetSubstructMatches maxMatches

_P3 = ("[M+H]+", "[M-H]-", "[M+NH4]+")


@dataclass(frozen=True)
class RewireRule:
    name: str
    smarts: str  # mapped SMARTS, as in the gate
    delete: tuple[int, ...]  # map numbers; every bond of these atoms goes, they leave in the neutral
    cut: tuple[tuple[int, int], ...]  # map-number pairs, unordered
    form: tuple[tuple[int, int], ...]
    precursors: tuple[str, ...]  # allowed precursor types
    null_smarts: str | None = None  # overrides the automatic topology null


@dataclass(frozen=True)
class RewireMatch:
    rule_index: int  # 0-based position in REWIRE_RULE_VOCABULARY (= residual column)
    motif_atoms: tuple[int, ...]  # sorted molecule-local ids of EVERY mapped atom of the match
    motif_numbers: tuple[int, ...]  # atomic numbers aligned with motif_atoms (graph cross-check)
    deleted: tuple[int, ...]  # sorted local ids
    cut_pairs: tuple[tuple[int, int], ...]  # each (min, max), sorted
    form_pairs: tuple[tuple[int, int], ...]  # each (min, max), sorted
    n_deleted_heavy: int  # deleted atoms with atomic number > 1


@dataclass(frozen=True)
class RewireMatchSet:
    matches: tuple[RewireMatch, ...]  # deterministic order, truncated per rule
    num_truncated: int  # matches dropped by max_matches_per_rule (sum over rules)


REWIRE_RULES: tuple[RewireRule, ...] = (
    RewireRule(
        name='sulfonyl_so2_extrusion',
        smarts='[#6:1][SX4:2](=[OX1:3])(=[OX1:4])[!#1:5]',
        delete=(2, 3, 4),
        cut=(),
        form=((1, 5),),
        precursors=_P3,
    ),
    RewireRule(
        name='s_to_o_aromatic_C',
        smarts='[c:1][SX4:2](=[OX1:3])(=[OX1:4])[!#1:5]',
        delete=(),
        cut=((1, 2), (2, 3)),
        form=((1, 3),),
        precursors=_P3,
    ),
    RewireRule(
        name='thiono_thiolo_aryl',
        smarts='[c:1][OX2:2][PX4:3]=[SX1:4]',
        delete=(),
        cut=((1, 2), (3, 4)),
        form=((1, 4),),
        precursors=_P3,
    ),
    RewireRule(
        name='PS_O_transfer_6ring',
        smarts='[PX4:1][SX2:2][#6:3][*:4][#6,#16:5]=[OX1:6]',
        delete=(),
        cut=((1, 2),),
        form=((1, 6),),
        precursors=_P3,
    ),
    RewireRule(
        name='nitro_nitrite_NO_loss',
        smarts='[#6:1][N+:2](=[OX1:3])[OX1-:4]',
        delete=(2, 3),
        cut=(),
        form=((1, 4),),
        precursors=_P3,
    ),
    RewireRule(
        name='acyclic_thioether_s_extrusion',
        smarts='[#6:1][#16X2;!R:2][#6:3]',
        delete=(2,),
        cut=(),
        form=((1, 3),),
        precursors=_P3,
    ),
    RewireRule(
        name='cyclic_thioether_s_extrusion',
        smarts='[#6:1]@[#16X2;R;!a:2]@[#6:3]',
        delete=(2,),
        cut=(),
        form=((1, 3),),
        precursors=_P3,
    ),
    RewireRule(
        name='diaryl_sulfide_s_extrusion',
        smarts='[c:1][#16X2;!R:2][c:3]',
        delete=(2,),
        cut=(),
        form=((1, 3),),
        precursors=_P3,
    ),
    RewireRule(
        name='hetaryl_thio_alkyl_s_extrusion',
        smarts='[CX4:1][#16X2;!R:2][c:3]:[n]',
        delete=(2,),
        cut=(),
        form=((1, 3),),
        precursors=_P3,
    ),
    RewireRule(
        name='thiadiazole_cs2_extrusion',
        smarts='[#6:1][#16X2:2][#6:3]1~[#7:4]~[#7:5]~[#6:6]~[#16X2:7]1',
        delete=(2, 3, 7),
        cut=(),
        form=((1, 4),),
        precursors=_P3,
    ),
    RewireRule(
        name='diaryl_ether_o_excision',
        smarts='[c:1][OX2:2][c:3]',
        delete=(2,),
        cut=(),
        form=((1, 3),),
        precursors=_P3,
    ),
    RewireRule(
        name='carbamate_co2_extrusion',
        smarts='[#7:1][CX3:2](=[OX1:3])[OX2:4][#6:5]',
        delete=(2, 3, 4),
        cut=(),
        form=((1, 5),),
        precursors=_P3,
    ),
    RewireRule(
        name='co2_extrusion_aryl_Oside',
        smarts='[#6,#7,#8:1][CX3:2](=[OX1:3])[OX2:4][c:5]',
        delete=(2, 3, 4),
        cut=(),
        form=((1, 5),),
        precursors=_P3,
    ),
    RewireRule(
        name='azo_N2_extrusion',
        smarts='[#6:1][NX2:2]=[NX2:3][!#1;!X1:4]',
        delete=(2, 3),
        cut=(),
        form=((1, 4),),
        precursors=('[M-H]-',),
        null_smarts='[#6:1][!#1;!D1:2]~[!#1;!D1:3]~[!#1;!X1:4]',
    ),
    RewireRule(
        name='dinitrophenol_24_2NO',
        smarts='[OX2H1]c1[c:1]([N+:2](=[OX1:3])[OX1-:4])c[c:5]([N+:6](=[OX1:7])[OX1-:8])cc1',
        delete=(2, 3, 6, 7),
        cut=(),
        form=((1, 4), (5, 8)),
        precursors=('[M-H]-',),
        null_smarts='[#6:1][!#1;!D1:2](~[*;D1:3])~[*:4].[#6:5][!#1;!D1:6](~[*;D1:7])~[*:8]',
    ),
    RewireRule(
        name='acid_acyl_transfer_5',
        smarts='[OX2H1:1][CX3:2](=[OX1:3])[#6:4][NX3:5][CX3:6]=[OX1:7]',
        delete=(),
        cut=((2, 1), (5, 6)),
        form=((1, 6),),
        precursors=('[M+Na]+',),
        null_smarts='[*;D1:1]-[#6:2]-[*:4]-[*:5]-[*:6]',
    ),
    RewireRule(
        name='haloalkyl_sulfonyl_X_to_S',
        smarts='[SX4:1](=[OX1])(=[OX1])[CX4:2][F,Cl,Br;D1:3]',
        delete=(),
        cut=((2, 3),),
        form=((1, 3),),
        precursors=('[M-H]-',),
    ),
)

# Literal, append-only; checked against REWIRE_RULES at import (reordering or removing a rule fails).
REWIRE_RULE_VOCABULARY: tuple[str, ...] = (
    'sulfonyl_so2_extrusion',
    's_to_o_aromatic_C',
    'thiono_thiolo_aryl',
    'PS_O_transfer_6ring',
    'nitro_nitrite_NO_loss',
    'acyclic_thioether_s_extrusion',
    'cyclic_thioether_s_extrusion',
    'diaryl_sulfide_s_extrusion',
    'hetaryl_thio_alkyl_s_extrusion',
    'thiadiazole_cs2_extrusion',
    'diaryl_ether_o_excision',
    'carbamate_co2_extrusion',
    'co2_extrusion_aryl_Oside',
    'azo_N2_extrusion',
    'dinitrophenol_24_2NO',
    'acid_acyl_transfer_5',
    'haloalkyl_sulfonyl_X_to_S',
)
MASSTER_NUM_REWIRE_RULES = len(REWIRE_RULE_VOCABULARY)


def rewire_rule_spec(rule: RewireRule) -> str:
    """Canonical edit spec of a rule; ``null_smarts`` is deliberately not part of it."""
    cut = ";".join(f"{a}-{b}" for a, b in sorted(tuple(sorted(p)) for p in rule.cut))
    form = ";".join(f"{a}-{b}" for a, b in sorted(tuple(sorted(p)) for p in rule.form))
    delete = ",".join(map(str, sorted(rule.delete)))
    return f"{rule.name}|{rule.smarts}|{delete}|{cut}|{form}|{','.join(sorted(rule.precursors))}"


def _pattern_maps(pattern: Chem.Mol) -> dict[int, int]:
    return {a.GetAtomMapNum(): a.GetIdx() for a in pattern.GetAtoms() if a.GetAtomMapNum()}


def validate_rewire_rules(rules: tuple[RewireRule, ...], vocabulary: tuple[str, ...]) -> None:
    """Raise ValueError on any inconsistency; run at import so a rule typo never surfaces at batch time."""
    names = [r.name for r in rules]
    if len(set(names)) != len(names):
        raise ValueError("REWIRE_RULES names must be unique")
    if tuple(names) != tuple(vocabulary):
        raise ValueError("REWIRE_RULE_VOCABULARY must equal the rule names in order (append-only)")
    for rule in rules:
        pattern = Chem.MolFromSmarts(rule.smarts)
        if pattern is None:
            raise ValueError(f"{rule.name}: SMARTS does not parse")
        maps = _pattern_maps(pattern)
        for m in rule.delete:
            if m not in maps:
                raise ValueError(f"{rule.name}: delete map number {m} not in SMARTS")
        for kind, pairs in (("cut", rule.cut), ("form", rule.form)):
            for a, b in pairs:
                if a == b:
                    raise ValueError(f"{rule.name}: {kind} pair ({a}, {b}) has equal endpoints")
                if a not in maps or b not in maps:
                    raise ValueError(f"{rule.name}: {kind} pair ({a}, {b}) has a map number not in SMARTS")
        for a, b in rule.cut:
            if pattern.GetBondBetweenAtoms(maps[a], maps[b]) is None:
                raise ValueError(f"{rule.name}: cut pair ({a}, {b}) is not a bond of the SMARTS")
        if not rule.form:
            raise ValueError(f"{rule.name}: form must be non-empty")
        for a, b in rule.form:
            if a in rule.delete or b in rule.delete:
                raise ValueError(f"{rule.name}: form pair ({a}, {b}) has a deleted endpoint")
        if not rule.precursors:
            raise ValueError(f"{rule.name}: precursors must be non-empty")
        for p in rule.precursors:
            if p not in PRECURSOR_TYPES_ALLOWED:
                raise ValueError(f"{rule.name}: unknown precursor type {p!r}")
        if rule.null_smarts is not None:
            null = Chem.MolFromSmarts(rule.null_smarts)
            if null is None:
                raise ValueError(f"{rule.name}: null_smarts does not parse")
            if not set(_pattern_maps(null)) <= set(maps):
                raise ValueError(f"{rule.name}: null_smarts map numbers are not a subset of the rule's")


validate_rewire_rules(REWIRE_RULES, REWIRE_RULE_VOCABULARY)

REWIRE_RULE_SPEC_HASHES: tuple[int, ...] = tuple(
    zlib.crc32(rewire_rule_spec(r).encode("utf-8")) for r in REWIRE_RULES
)
_PATTERNS: tuple[Chem.Mol, ...] = tuple(Chem.MolFromSmarts(r.smarts) for r in REWIRE_RULES)


@cache
def atom_map_numbers(pattern: Chem.Mol) -> tuple[int, ...]:
    """The pattern's map number per query atom (0 when unmapped), once per pattern object.

    Rebuilding this per call cost ~0.17 ms per rule per molecule, about 40x the substructure search.
    Patterns are module constants and RDKit Mols hash by identity, so the cache stays small.
    """
    return tuple(a.GetAtomMapNum() for a in pattern.GetAtoms())


def raw_rule_matches(mol: Chem.Mol, pattern: Chem.Mol) -> list[dict[int, int]]:
    """One ``{map_number: atom_idx}`` per hit (unmapped atoms omitted), deduplicated, original order kept."""
    maps = atom_map_numbers(pattern)
    seen: set[tuple] = set()
    unique: list[dict[int, int]] = []
    for hit in mol.GetSubstructMatches(pattern, uniquify=False, maxMatches=_MAX_RAW_MATCHES):
        d = {m: idx for m, idx in zip(maps, hit) if m}
        key = tuple(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _pairs(pairs, hit: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(sorted((hit[a], hit[b]))) for a, b in pairs))


def build_rewire_matches(mol: Chem.Mol, *, max_matches_per_rule: int) -> RewireMatchSet:
    """All rule matches on the STANDARDIZED molecule, deduplicated on the edit, truncated per rule."""
    out: list[RewireMatch] = []
    num_truncated = 0
    for rule_index, (rule, pattern) in enumerate(zip(REWIRE_RULES, _PATTERNS)):
        seen: dict[tuple, RewireMatch] = {}
        for hit in raw_rule_matches(mol, pattern):
            atoms = tuple(sorted(hit.values()))
            deleted = tuple(sorted(hit[m] for m in rule.delete))
            cut = _pairs(rule.cut, hit)
            form = _pairs(rule.form, hit)
            key = (rule_index, atoms, deleted, cut, form)
            if key not in seen:
                numbers = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms)
                heavy = sum(1 for i in deleted if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
                seen[key] = RewireMatch(rule_index, atoms, numbers, deleted, cut, form, heavy)
        ordered = [seen[k] for k in sorted(seen)]
        num_truncated += max(0, len(ordered) - max_matches_per_rule)
        out.extend(ordered[:max_matches_per_rule])
    return RewireMatchSet(tuple(out), num_truncated)


def filter_rewire_matches(match_set: RewireMatchSet, *, precursor_type: str, num_known_rules: int) -> RewireMatchSet:
    """Keep matches whose rule lists ``precursor_type`` and whose index is below ``num_known_rules``."""
    kept = tuple(
        m
        for m in match_set.matches
        if m.rule_index < num_known_rules and precursor_type in REWIRE_RULES[m.rule_index].precursors
    )
    return replace(match_set, matches=kept)
