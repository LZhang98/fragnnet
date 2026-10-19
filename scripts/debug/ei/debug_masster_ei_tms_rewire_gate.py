"""Model-free NIST23 EI gate for candidate rewire rules (TMS migrations) and the ESI rewire rules under [M]+.

Research-only: no production registry change and no GPU. T1/T1b/T2/T3 are defined here as
candidates; every other entry of ``REWIRE_RULES`` is evaluated as if it listed ``[M]+``.
Peaks are read for train and val only; test molecules are used for structure counts and
composition reach only. Per rule and fold the gate measures:

(a) matches: molecules, spectra, edit-distinct matches per molecule, and truncation at
    ``max_matches_per_rule``;
(b) composition reach: does every product (depth 0, applied to the whole precursor) have a
    heavy composition with own-H within +-4 that no connected atom set of the precursor has?
    Exact connected-set enumeration up to ``EXACT_MAX_HEAVY`` heavy atoms, the 0-2-cut pool
    otherwise or when the exact search hits its cap (recorded per molecule);
(c) nominal added coverage with natural isotopes beyond the 0-2-cut pool, rule vs topology null;
(d) peak support at the product's nominal m/z (literature H shift; H shift 0 for ESI rules)
    and at the best valid channel within +-4 H, rule vs topology null. The null does not preserve
    Si, so for T1 (d) is descriptive: 147 is a generic ion in TMS spectra;
(e) T1 only: observed 148/147 against the M+1/M of C5H15OSi2 and of the closest competing
    composition at nominal 147 over EVERY connected atom set (H shift -4..+4), with the same
    statistic on one-TMS control spectra (147 >= 10% of base) that T1 cannot match. The 0-2-cut
    pool version is kept as an upper bound (``e_pool_*``); the headline is ``e_*`` (exact).

Products follow the production rewire stage: the rule is applied to a kept fragment F holding
every mapped atom, the product is a connected piece of the edited F that holds a formed bond,
its cut count is the boundary of the piece and its valence correction is
``-2 * (n_formed - n_internal_cut)``. F ranges over the whole precursor and every one-cut
fragment of the pool. Products already in the pool are ordinary candidates and are dropped
from (c) and (d). The baseline pool is the 0-2-cut stand-in of the closed 2026-10-01 screen,
not the trained model's selected pool. The decision (see ``decide``) is written to summary.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import random
import subprocess
import sys
import zlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger, rdBase
from torch_geometric.data import Batch

from fragnnet.models.masster.candidates.rewire_null import topology_null_smarts
from fragnnet.models.masster.candidates.rewire_rules import (
    _MAX_RAW_MATCHES,
    REWIRE_RULES,
    RewireRule,
    atom_map_numbers,
)
from fragnnet.models.masster.masster_nominal_isotopes import nominal_isotope_pattern
from fragnnet.utils.feat_utils import get_mol_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.debug.ei.debug_masster_ei_nominal_coverage import (  # noqa: E402
    H_SHIFTS,
    eval_target,
    nominal_atom_masses,
    pool_sets,
)
from scripts.debug.smarts.debug_masster_smarts_g0_ceiling import ChannelCalculator  # noqa: E402
from scripts.debug.smarts.debug_masster_smarts_rule_gate import reachable_h  # noqa: E402
from tests.test_masster_model import _build_model  # noqa: E402

LOGGER = logging.getLogger(__name__)
DATA = Path("data/proc/nist23_ei")
SPLIT = Path("data/split/nist23_ei_inchikey_hr_ei_excluded_2026_10_17")
FOLDS = ("train", "val", "test")
MAX_MATCHES = 8  # production masster_rewire_max_matches_per_rule
EXACT_MAX_HEAVY = 40
COMPETITOR_MZ = 147
COMPETITOR_CAP = 1_000_000  # connected-set visits per molecule before the exact (e) arm gives up
SI2 = ((("C", 5), ("O", 1), ("Si", 2)), 15)  # C5H15OSi2, the T1 product at H shift 0
TMS = "[Si]([CH3])([CH3])[CH3]"
TMS_GROUP = Chem.MolFromSmarts(f"[O,N,S]{TMS}")
PRECURSOR = ("[M]+",)

# 1. Rules -------------------------------------------------------------------------------------

CANDIDATES = (
    RewireRule(
        "tms_siloxy_silylium",
        f"[#6,#15:1][OX2:2]{TMS}.[#6,#15:4][O,N:5][Si:6]([CH3:7])([CH3])[CH3]",
        (),
        ((1, 2), (5, 6), (6, 7)),
        ((2, 6),),
        PRECURSOR,
    ),
    RewireRule(
        "tms_siloxy_silanol",
        "[#6,#15:1][OX2:2][Si:3]([CH3:8])([CH3])[CH3].[#6,#15:4][OX2:5][Si:6]([CH3:7])([CH3])[CH3]",
        (),
        ((1, 2), (4, 5), (6, 7), (3, 8)),
        ((2, 6),),
        PRECURSOR,
    ),
    RewireRule(
        "vicinal_otms_geminal",
        f"[CX4;D2:1]([OX2]{TMS})[CX4:3][OX2:4]{TMS}",
        (),
        ((3, 4), (1, 3)),
        ((1, 4),),
        PRECURSOR,
    ),
    RewireRule(
        "silyl_to_ester_carbonyl",
        f"[#6:9][CX4:8][CX3:1](=[OX1:2])[OX2]{TMS}."
        "[OX2;!$(OC=O);!$(O[CX4]C=O):5][Si:6]([CH3])([CH3])[CH3]",
        (),
        ((8, 9), (5, 6)),
        ((2, 6),),
        PRECURSOR,
    ),
)
LITERATURE_SHIFT = {
    "tms_siloxy_silylium": 0,
    "tms_siloxy_silanol": 1,
    "vicinal_otms_geminal": -1,
    "silyl_to_ester_carbonyl": 0,
}
T1 = "tms_siloxy_silylium"


def gate_rules() -> tuple[RewireRule, ...]:
    """Candidates plus every imported rule whose name is not a candidate name."""
    names = {rule.name for rule in CANDIDATES}
    return CANDIDATES + tuple(rule for rule in REWIRE_RULES if rule.name not in names)


class RuleContext(NamedTuple):
    rule: RewireRule
    pattern: Chem.Mol
    null_pattern: Chem.Mol
    maps: tuple[int, ...]
    null_maps: tuple[int, ...]
    shift: int


def build_contexts(rules: tuple[RewireRule, ...]) -> list[RuleContext]:
    contexts = []
    for rule in rules:
        pattern = Chem.MolFromSmarts(rule.smarts)
        null_pattern = Chem.MolFromSmarts(rule.null_smarts or topology_null_smarts(rule))
        contexts.append(
            RuleContext(
                rule,
                pattern,
                null_pattern,
                atom_map_numbers(pattern),
                atom_map_numbers(null_pattern),
                LITERATURE_SHIFT.get(rule.name, 0),
            )
        )
    return contexts


# 2. Matches and products ---------------------------------------------------------------------


class Match(NamedTuple):
    """One edit-distinct match, every field in molecule-local atom ids."""

    atoms: tuple[int, ...]  # sorted mapped atoms
    deleted: tuple[int, ...]
    cut: tuple[tuple[int, int], ...]
    form: tuple[tuple[int, int], ...]


def raw_matches(mol: Chem.Mol, pattern: Chem.Mol) -> tuple[list[dict[int, int]], bool]:
    """Deduplicated ``{map: atom}`` hits and whether the substructure search hit its cap."""
    maps = atom_map_numbers(pattern)
    hits = mol.GetSubstructMatches(pattern, uniquify=False, maxMatches=_MAX_RAW_MATCHES)
    seen, unique = set(), []
    for hit in hits:
        item = {m: i for m, i in zip(maps, hit, strict=True) if m}
        key = tuple(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique, len(hits) >= _MAX_RAW_MATCHES


def to_match(rule: RewireRule, hit: dict[int, int]) -> Match:
    pairs = lambda ps: tuple(sorted(tuple(sorted((hit[a], hit[b]))) for a, b in ps))  # noqa: E731
    return Match(
        tuple(sorted(hit.values())),
        tuple(sorted(hit[m] for m in rule.delete if m in hit)),
        pairs(rule.cut),
        pairs(rule.form),
    )


def edit_distinct(rule: RewireRule, hits: list[dict[int, int]]) -> list[Match]:
    """Matches deduplicated on the edit and sorted, as ``build_rewire_matches`` does."""
    return sorted({to_match(rule, hit) for hit in hits})


def real_matches(mol: Chem.Mol, ctx: RuleContext) -> tuple[list[Match], bool, list[dict[int, int]]]:
    hits, capped = raw_matches(mol, ctx.pattern)
    return edit_distinct(ctx.rule, hits), capped, hits


def null_matches(
    mol: Chem.Mol,
    ctx: RuleContext,
    real_hits: list[dict[int, int]],
    n_real: int,
    rng: np.random.Generator,
) -> tuple[list[Match], int, bool]:
    """Same edit shape at random sites: ``n_real`` distinct null edits, real sites excluded.

    Returns the draw truncated like production, the pool size and the raw-search cap flag.
    """
    real_sites = {frozenset(h[k] for k in ctx.null_maps if k in h) for h in real_hits}
    hits, capped = raw_matches(mol, ctx.null_pattern)
    pool = sorted(
        {to_match(ctx.rule, hit) for hit in hits if frozenset(hit.values()) not in real_sites}
    )
    if not pool:
        return [], 0, capped
    picked = rng.choice(len(pool), min(n_real, len(pool)), replace=False)
    return sorted(pool[i] for i in picked)[:MAX_MATCHES], len(pool), capped


def edited_products(
    fragments: list[frozenset[int]], matches: list[Match], bonds: list[frozenset[int]]
) -> dict[frozenset[int], tuple[int, int]]:
    """Connected pieces holding a formed bond of the edited fragments, deduplicated on atoms.

    A piece maps to ``(n_formed, n_internal_cut)`` of its first occurrence, as production does.
    """
    products: dict[frozenset[int], tuple[int, int]] = {}
    for match in matches:
        motif = frozenset(match.atoms)
        cut = {frozenset(pair) for pair in match.cut}
        formed = {frozenset(pair) for pair in match.form}
        deleted = set(match.deleted)
        for fragment in fragments:
            if not motif <= fragment:
                continue
            kept = fragment - deleted
            adjacency: dict[int, set[int]] = defaultdict(set)
            for bond in [b for b in bonds if b not in cut] + list(formed):
                a, b = tuple(bond)
                if a in kept and b in kept:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
            seen: set[int] = set()
            for start in sorted(kept):
                if start in seen:
                    continue
                piece, stack = set(), [start]
                while stack:
                    atom = stack.pop()
                    if atom not in piece:
                        piece.add(atom)
                        stack.extend(adjacency[atom] - piece)
                seen |= piece
                n_formed = sum(f <= piece for f in formed)
                if n_formed:
                    products.setdefault(frozenset(piece), (n_formed, sum(c <= piece for c in cut)))
    return products


# 3. Channels and nominal support -------------------------------------------------------------


class RewireChannelCalculator(ChannelCalculator):
    """Production channels for rewire products: cut counts plus a signed valence correction."""

    def channels(
        self, mol: Chem.Mol, sets: list[tuple[int, ...]], cuts: list[int], adjustments: list[int]
    ) -> np.ndarray:
        """Return the ``(n, 9)`` validity mask over H shifts -4..+4 for ``[M]+``."""
        if not sets:
            return np.zeros((0, len(H_SHIFTS)), dtype=bool)
        graph = Batch.from_data_list(
            [get_mol_graph(mol, ["a_onehot", "a_degree"], ["b_degree"], 0, True)]
        )
        if self.model is None:
            self.model = _build_model(
                graph,
                include_cmf=False,
                max_h_transfer=max(H_SHIFTS),
                h_transfer_bound="valence",
                precursor_types=self.precursor_types,
            ).eval()
        atoms = th.tensor([a for s in sets for a in s], dtype=th.long)
        fragments = th.tensor([i for i, s in enumerate(sets) for _ in s], dtype=th.long)
        with th.no_grad():
            _, valid = self.model._fragment_masses_and_valid_channels(
                graph,
                atoms,
                fragments,
                th.zeros(len(sets), dtype=th.long),
                len(sets),
                th.tensor([0]),
                th.tensor(cuts, dtype=th.long),
                None,
                th.tensor(adjustments, dtype=th.long),
            )
        return valid.numpy()[:, 0, :]


class Channels(NamedTuple):
    """Nominal channels of atom sets: monoisotopic nominal mass and validity per H shift."""

    mass: dict[tuple[int, ...], int]
    valid: dict[tuple[int, ...], np.ndarray]
    composition: dict[tuple[int, ...], tuple]


def composition_key(mol: Chem.Mol, atoms: tuple[int, ...]) -> tuple:
    """Sorted (element, count) pairs of the heavy atoms; H is carried separately."""
    return tuple(sorted(Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in atoms).items()))


def own_hydrogens(mol: Chem.Mol, atoms: tuple[int, ...]) -> int:
    return sum(mol.GetAtomWithIdx(i).GetTotalNumHs() for i in atoms)


@lru_cache(maxsize=200_000)
def isotope_pattern(heavy: tuple, hydrogens: int) -> tuple[float, ...]:
    """Production order-4 nominal pattern (offsets -4..+4) of a composition."""
    return nominal_isotope_pattern(tuple(sorted(heavy + (("H", hydrogens),))), 4)


@lru_cache(maxsize=200_000)
def isotope_offsets(heavy: tuple, hydrogens: int) -> tuple[int, ...]:
    """Offsets kept after crop normalization and the 0.001 cutoff (the screen's convention)."""
    pattern = np.asarray(isotope_pattern(heavy, hydrogens))
    pattern = pattern / pattern.sum()
    return tuple(int(o) for o in np.arange(-4, 5)[(pattern > 0) & (pattern >= 0.001)])


def m1_over_m(heavy: tuple, hydrogens: int) -> float:
    """M+1/M of a composition at its nominal monoisotopic mass."""
    pattern = isotope_pattern(heavy, hydrogens)
    return pattern[5] / pattern[4]


def boundary_count(bonds: list[frozenset[int]], atoms: frozenset[int]) -> int:
    return sum(len(bond & atoms) == 1 for bond in bonds)


def channel_table(
    mol: Chem.Mol,
    calc: RewireChannelCalculator,
    entries: dict[tuple[int, ...], int],
    bonds: list[frozenset[int]],
) -> Channels:
    """Channels of atom sets, given each set's valence adjustment; cuts are boundary bonds."""
    sets = list(entries)
    cuts = [boundary_count(bonds, frozenset(s)) for s in sets]
    valid = calc.channels(mol, sets, cuts, [entries[s] for s in sets])
    masses = nominal_atom_masses(mol)
    return Channels(
        {s: int(masses[list(s)].sum()) for s in sets},
        {s: valid[i] for i, s in enumerate(sets)},
        {s: composition_key(mol, s) for s in sets},
    )


def isotope_support(mol: Chem.Mol, table: Channels, sets: list[tuple[int, ...]]) -> set[int]:
    """Nominal masses (monoisotopic channel plus isotope envelope) of valid channels."""
    support: set[int] = set()
    for atoms in sets:
        own = own_hydrogens(mol, atoms)
        for shift in H_SHIFTS[table.valid[atoms]]:
            mass = table.mass[atoms] + int(shift)
            if mass <= 0:
                continue
            offsets = isotope_offsets(table.composition[atoms], own + int(shift))
            support.update(mass + o for o in offsets if mass + o > 0)
    return support


# 4. Spectra -----------------------------------------------------------------------------------


class Spectrum(NamedTuple):
    mz: np.ndarray  # integer m/z, one entry per peak
    weight: np.ndarray  # L1-normalized intensity per peak
    bins: dict[int, float]  # integer m/z -> summed L1 intensity
    base: float  # largest bin


def prepare_spectrum(peaks: object) -> Spectrum:
    mz, weight, _ = eval_target(peaks)
    bins: dict[int, float] = defaultdict(float)
    for m, w in zip(mz.tolist(), weight.tolist(), strict=True):
        bins[m] += w
    return Spectrum(mz, weight, dict(bins), max(bins.values()))


def coverage_delta(spectrum: Spectrum, base: set[int], added: set[int]) -> float:
    """L1 weight of observed integer m/z in ``added - base``."""
    return float(spectrum.weight[np.isin(spectrum.mz, list(added - base))].sum())


def peak_support(
    spectrum: Spectrum, table: Channels, products: list[tuple[int, ...]], shift: int
) -> tuple[float, float, float, float]:
    """Mean over products of L1 and >=10%-of-base at the literature shift and at the best channel."""
    lit_l1, lit_ge, best_l1, best_ge = [], [], [], []
    for atoms in products:
        lit = spectrum.bins.get(table.mass[atoms] + shift, 0.0)
        valid = table.valid[atoms]
        best = max(
            (spectrum.bins.get(table.mass[atoms] + int(s), 0.0) for s in H_SHIFTS[valid]),
            default=0.0,
        )
        lit_l1.append(lit)
        lit_ge.append(lit >= 0.1 * spectrum.base)
        best_l1.append(best)
        best_ge.append(best >= 0.1 * spectrum.base)
    return tuple(float(np.mean(v)) for v in (lit_l1, lit_ge, best_l1, best_ge))


# 5. Composition reach (b) --------------------------------------------------------------------


def composition_index(mol: Chem.Mol, sets: list[tuple[int, ...]]) -> dict[tuple, set[int]]:
    index: dict[tuple, set[int]] = defaultdict(set)
    for atoms in sets:
        index[composition_key(mol, atoms)].add(own_hydrogens(mol, atoms))
    return index


def exact_reach(mol: Chem.Mol, edges: tuple, atoms: tuple[int, ...], cap: int) -> bool | None:
    """Exact connected-set answer for small molecules; None when too large or the search hit its cap."""
    if mol.GetNumAtoms() > EXACT_MAX_HEAVY:
        return None
    key, own = composition_key(mol, atoms), own_hydrogens(mol, atoms)
    found = reachable_h(
        mol, edges, Counter(dict(key)), len(atoms), cap=cap, required_h=frozenset([own])
    )
    return None if found is None else any(abs(own - h) <= 4 for h in found)


def pool_reach(mol: Chem.Mol, atoms: tuple[int, ...], index: dict[tuple, set[int]]) -> bool:
    """Pool fallback: some 0-2-cut set has the composition with own-H within +-4."""
    own = own_hydrogens(mol, atoms)
    return any(abs(own - h) <= 4 for h in index.get(composition_key(mol, atoms), ()))


# 6. Per-molecule worker ----------------------------------------------------------------------

CALC: RewireChannelCalculator | None = None
CONTEXTS: dict[str, RuleContext] = {}


def init_worker(seed: int) -> None:
    global CALC, CONTEXTS
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- seed dependencies as well as local Generators
    th.manual_seed(seed)
    th.set_num_threads(1)
    RDLogger.DisableLog("rdApp.*")
    CALC = RewireChannelCalculator(PRECURSOR)
    CONTEXTS = {c.rule.name: c for c in build_contexts(gate_rules())}


def sampled(seed: int, mol_id: int, rule: str, fraction: float) -> bool:
    """Seeded per-(molecule, rule) sample; fraction 1 keeps everything."""
    if fraction >= 1.0:
        return True
    return (
        np.random.default_rng(
            np.random.SeedSequence([seed, mol_id, zlib.crc32(rule.encode())])
        ).random()
        < fraction
    )


def census_chunk(chunk: list[tuple[int, Chem.Mol]]) -> list[dict]:
    """Structure counts per (molecule, rule): edit-distinct matches and truncation."""
    rows = []
    for mol_id, mol in chunk:
        for name, ctx in CONTEXTS.items():
            matches, capped, _ = real_matches(mol, ctx)
            if matches:
                rows.append(
                    {
                        "mol_id": mol_id,
                        "rule": name,
                        "n_edit_distinct": len(matches),
                        "n_after_truncation": min(len(matches), MAX_MATCHES),
                        "raw_capped": capped,
                    }
                )
    return rows


def process_molecule(task: tuple) -> list[dict]:
    """Score one molecule for every rule it matches; one row per (molecule, rule)."""
    mol_id, mol, split, rules, spectra, seed, n_null, exact_cap, fraction = task
    bonds = [frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()]
    edges = tuple(zip(*[tuple(b) for b in bonds], strict=True)) if bonds else ((), ())
    spectra = [prepare_spectrum(peaks) for _, peaks in spectra]
    n_tms = len(mol.GetSubstructMatches(TMS_GROUP))
    heavy = mol.GetNumAtoms()
    pool_cache: dict | None = None

    def pool() -> dict:
        nonlocal pool_cache
        if pool_cache is None:
            sets = pool_sets(mol)
            table = channel_table(mol, CALC, dict.fromkeys(sets, 0), bonds)
            pool_cache = {
                "sets": sets,
                "table": table,
                "iso": isotope_support(mol, table, list(sets)),
            }
        return pool_cache

    rows = []
    for rule_name in rules:
        ctx = CONTEXTS[rule_name]
        matches, capped, hits = real_matches(mol, ctx)
        used = matches[:MAX_MATCHES]
        row = {
            "mol_id": mol_id,
            "split": split,
            "rule": rule_name,
            "heavy_atoms": heavy,
            "n_edit_distinct": len(matches),
            "n_after_truncation": len(used),
            "raw_capped": capped,
            "n_spectra": len(spectra),
        }
        if not sampled(seed, mol_id, rule_name, fraction):
            row["sampled"] = False
            rows.append(row)
            continue
        row["sampled"] = True

        # 6.1 Reach (b): products of the whole-precursor edit against connected sets.
        whole = frozenset(range(heavy))
        top = edited_products([whole], used, bonds)
        verdicts, methods = [], []
        for atoms in sorted(top, key=sorted):
            ordered = tuple(sorted(atoms))
            reached = exact_reach(mol, edges, ordered, exact_cap)
            if reached is None:
                reached = pool_reach(mol, ordered, composition_index(mol, list(pool()["sets"])))
                methods.append("pool")
            else:
                methods.append("exact")
            verdicts.append(reached)
        row.update(
            {
                "n_whole_products": len(top),
                "all_products_unreachable": bool(verdicts) and not any(verdicts),
                "any_product_unreachable": not all(verdicts) if verdicts else False,
                "reach_method": "pool" if "pool" in methods else "exact",
                "product_formulas": json.dumps(
                    sorted({str(composition_key(mol, tuple(sorted(a)))) for a in top})
                ),
            }
        )
        if not spectra:
            rows.append(row)
            continue

        # 6.2 Real and null products over the whole precursor and the one-cut fragments.
        base = pool()
        fragments = [frozenset(s) for s, cuts in base["sets"].items() if cuts <= 1]
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, mol_id, zlib.crc32(rule_name.encode())])
        )
        draws, null_pool, null_capped = [], 0, False
        for _ in range(n_null):
            drawn, null_pool, null_capped = null_matches(mol, ctx, hits, len(matches), rng)
            if drawn:
                draws.append(drawn)
        arms = [edited_products(fragments, used, bonds)] + [
            edited_products(fragments, drawn, bonds) for drawn in draws
        ]
        arms = [
            {a: adj for a, adj in arm.items() if tuple(sorted(a)) not in base["sets"]}
            for arm in arms
        ]
        entries = {
            tuple(sorted(a)): -2 * (n_formed - n_internal)
            for arm in arms
            for a, (n_formed, n_internal) in arm.items()
        }
        table = channel_table(mol, CALC, entries, bonds) if entries else None
        product_sets = [[tuple(sorted(a)) for a in arm] for arm in arms]
        supports = [isotope_support(mol, table, sets) if table else set() for sets in product_sets]
        null_ok = len(draws) == n_null
        row.update(
            {
                "n_extra_products": len(product_sets[0]),
                "null_pool": null_pool,
                "null_capped": null_capped,
                "null_draws": len(draws),
                "null_ok": null_ok,
            }
        )

        # 6.3 Per-spectrum metrics, averaged within the molecule.
        values = defaultdict(list)
        t1_rows = []
        competitors: tuple | None = None
        for spectrum in spectra:
            delta = coverage_delta(spectrum, base["iso"], supports[0])
            null_deltas = [coverage_delta(spectrum, base["iso"], s) for s in supports[1:]]
            values["delta"].append(delta)
            values["null"].append(np.mean(null_deltas) if null_ok else np.nan)
            names = ("lit_l1", "lit_ge10", "best_l1", "best_ge10")
            real_peak = (
                peak_support(spectrum, table, product_sets[0], ctx.shift)
                if product_sets[0]
                else None
            )
            null_peak = [
                peak_support(spectrum, table, sets, ctx.shift) for sets in product_sets[1:] if sets
            ]
            for index, name in enumerate(names):
                values[name].append(real_peak[index] if real_peak else np.nan)
                values[f"null_{name}"].append(
                    np.mean([p[index] for p in null_peak]) if null_peak and null_ok else np.nan
                )
            if (
                rule_name == T1
                and n_tms >= 2
                and spectrum.bins.get(147, 0.0) >= 0.1 * spectrum.base
            ):
                if competitors is None:
                    competitors = (
                        pool_competitors(base["table"], mol),
                        exact_competitors(mol),
                    )
                t1_rows.append(isotope_consistency(spectrum, *competitors))
        for name, vals in values.items():
            row[name] = float(np.nanmean(vals)) if not np.isnan(vals).all() else np.nan
        row["diff"] = row["delta"] - row["null"]
        for name in ("lit_l1", "lit_ge10", "best_l1", "best_ge10"):
            row[f"diff_{name}"] = row[name] - row[f"null_{name}"]
        row["e_spectra"] = len(t1_rows)
        row["e_exact_capped"] = bool(t1_rows) and competitors[1] is None
        row["e_detail"] = json.dumps(t1_rows)
        rows.append(row)
    return rows


def control_molecule(task: tuple) -> dict | None:
    """(e) control: a molecule with exactly one TMS group, which T1 cannot match, scored like T1."""
    mol_id, mol, split, spectra = task
    if len(mol.GetSubstructMatches(TMS_GROUP)) != 1:
        return None
    prepared = [prepare_spectrum(peaks) for _, peaks in spectra]
    prepared = [s for s in prepared if s.bins.get(147, 0.0) >= 0.1 * s.base]
    if not prepared:
        return None
    bonds = [frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()]
    sets = pool_sets(mol)
    table = channel_table(mol, CALC, dict.fromkeys(sets, 0), bonds)
    competitors = (pool_competitors(table, mol), exact_competitors(mol))
    detail = [isotope_consistency(spectrum, *competitors) for spectrum in prepared]
    return {
        "mol_id": mol_id,
        "split": split,
        "e_spectra": len(detail),
        "e_exact_capped": competitors[1] is None,
        "e_detail": json.dumps(detail),
    }


def competitor_ratios(mol: Chem.Mol, nominal: int, cap: int) -> dict[tuple, float] | None:
    """M+1/M of every composition some connected atom set has at ``nominal`` m/z, or None past ``cap``.

    Exhaustive (ESU) enumeration of connected sets whose heavy-atom nominal mass with own hydrogens
    is within the H-shift window of ``nominal``; every shift of -4..+4 is allowed, so the set
    contains every valid channel of the 0-2-cut pool and more. Keys are (heavy, hydrogens).
    """
    masses = nominal_atom_masses(mol).tolist()
    neighbors = [{n.GetIdx() for n in atom.GetNeighbors()} for atom in mol.GetAtoms()]
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    hydrogens = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]
    low, high = int(H_SHIFTS.min()), int(H_SHIFTS.max())
    found: dict[tuple, float] = {}
    visits = 0

    def extend(sub: frozenset[int], ext: list[int], closed: frozenset[int], mass: int) -> bool:
        nonlocal visits
        visits += 1
        if visits > cap:
            return False
        shift = nominal - mass
        if low <= shift <= high:
            hydrogen = sum(hydrogens[i] for i in sub) + shift
            if hydrogen >= 0:
                heavy = tuple(sorted(Counter(symbols[i] for i in sub).items()))
                if (heavy, hydrogen) not in found:
                    found[(heavy, hydrogen)] = m1_over_m(heavy, hydrogen)
        while ext:
            atom = ext.pop()
            if mass + masses[atom] > nominal - low:
                continue
            fresh = [u for u in neighbors[atom] if u > root and u not in closed]
            if not extend(sub | {atom}, ext + fresh, closed | neighbors[atom], mass + masses[atom]):
                return False
        return True

    for root in range(len(masses)):
        if masses[root] > nominal - low:
            continue
        closed = frozenset(neighbors[root] | {root})
        if not extend(
            frozenset([root]), [u for u in neighbors[root] if u > root], closed, masses[root]
        ):
            return None
    return found


def closest_ratio_error(competitors: dict[tuple, float], observed: float) -> float | None:
    """Absolute error of the competitor whose M+1/M is nearest the observed ratio; None if none."""
    if not competitors:
        return None
    return min(abs(observed - ratio) for ratio in competitors.values())


def exact_competitors(mol: Chem.Mol) -> dict[tuple, float] | None:
    """Exhaustive competitors at 147 excluding C5H15OSi2; None when the search hit its cap."""
    found = competitor_ratios(mol, COMPETITOR_MZ, COMPETITOR_CAP)
    if found is not None:
        found.pop(SI2, None)
    return found


def isotope_consistency(
    spectrum: Spectrum,
    pool_competitors: dict[tuple, float],
    exact_competitors: dict[tuple, float] | None,
) -> dict:
    """Observed 148/147 against the M+1/M of C5H15OSi2 and of the closest competitor at 147.

    Competitors exclude C5H15OSi2 itself. ``pool`` uses the 0-2-cut pool (a stand-in that
    under-counts competitors, so its share is an upper bound); ``exact`` uses every connected set
    and is the headline. An arm is None when it has no competitor or the exact search was capped.
    """
    observed = spectrum.bins.get(148, 0.0) / spectrum.bins[147]
    ratio_si2 = m1_over_m(*SI2)
    record = {
        "observed": observed,
        "ratio_si2": ratio_si2,
        "err_si2": abs(observed - ratio_si2),
    }
    for tag, competitors in (("pool", pool_competitors), ("exact", exact_competitors)):
        error = closest_ratio_error(competitors or {}, observed)
        record[f"err_{tag}"] = error
        record[f"si2_closer_{tag}"] = None if error is None else bool(record["err_si2"] < error)
    return record


def pool_competitors(table: Channels, mol: Chem.Mol) -> dict[tuple, float]:
    """Competitors at 147 among the valid channels of the 0-2-cut pool (excluding C5H15OSi2)."""
    found = {}
    for atoms, valid in table.valid.items():
        for shift in H_SHIFTS[valid]:
            if table.mass[atoms] + int(shift) == COMPETITOR_MZ:
                key = (table.composition[atoms], own_hydrogens(mol, atoms) + int(shift))
                if key != SI2:
                    found[key] = m1_over_m(*key)
    return found


# 7. Summary ----------------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, seed: int, reps: int = 2000) -> tuple[float, float]:
    """Percentile 95% interval of the mean over molecules (2000 resamples)."""
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(reps, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def paired(group: pd.DataFrame, rule_col: str, null_col: str, seed: int) -> dict:
    """Molecule-level rule and null means with a bootstrap interval on the paired difference."""
    sub = group.dropna(subset=[rule_col, null_col])
    diff = (sub[rule_col] - sub[null_col]).to_numpy()
    lo, hi = bootstrap_ci(diff, seed)
    return {
        "molecules": len(sub),
        "rule": float(sub[rule_col].mean()) if len(sub) else np.nan,
        "null": float(sub[null_col].mean()) if len(sub) else np.nan,
        "diff": float(diff.mean()) if len(sub) else np.nan,
        "lo": lo,
        "hi": hi,
    }


def isotope_stats(details: pd.Series, tag: str, seed: int) -> dict:
    """(e) statistics of one competitor arm; one JSON list of per-spectrum records per molecule.

    The share is the mean over molecules of the per-molecule fraction of spectra where C5H15OSi2
    is closer to the observed 148/147; spectra without that arm's competitor are skipped.
    """
    per_mol, spectra = [], []
    for blob in details:
        usable = [d for d in json.loads(blob) if d[f"si2_closer_{tag}"] is not None]
        if usable:
            per_mol.append(np.mean([d[f"si2_closer_{tag}"] for d in usable]))
            spectra.extend(usable)
    if not spectra:
        return {"molecules": 0, "spectra": 0}
    values = np.array(per_mol)
    lo, hi = bootstrap_ci(values, seed)
    return {
        "molecules": len(values),
        "spectra": len(spectra),
        "si2_closer_mean": float(values.mean()),
        "si2_closer_lo": lo,
        "si2_closer_hi": hi,
        "si2_closer_spectra": float(np.mean([d[f"si2_closer_{tag}"] for d in spectra])),
        "median_abs_err_si2": float(np.median([d["err_si2"] for d in spectra])),
        "median_abs_err_competitor": float(np.median([d[f"err_{tag}"] for d in spectra])),
    }


def summarize(
    rows: pd.DataFrame, census: pd.DataFrame, control: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Per rule and fold table of all metrics plus the pre-registered admission decision."""
    out = []
    for rule in [c.rule.name for c in build_contexts(gate_rules())]:
        for fold in (*FOLDS, "train+val"):
            folds = ("train", "val") if fold == "train+val" else (fold,)
            cen = census[(census.rule == rule) & census.split.isin(folds)]
            scored = rows[(rows.rule == rule) & rows.split.isin(folds) & rows.sampled]
            item = {
                "rule": rule,
                "fold": fold,
                "matched_molecules": len(cen),
                "matched_spectra": int(cen.n_spectra.sum()),
                "matches_mean": cen.n_edit_distinct.mean(),
                "matches_p95": cen.n_edit_distinct.quantile(0.95) if len(cen) else np.nan,
                "matches_max": cen.n_edit_distinct.max(),
                "matches_after_trunc_mean": cen.n_after_truncation.mean(),
                "molecules_truncated": int((cen.n_edit_distinct > MAX_MATCHES).sum()),
                "raw_capped_molecules": int(cen.raw_capped.sum()),
                "sampled_molecules": len(scored),
            }
            reach = scored[scored.n_whole_products > 0]
            item.update(
                {
                    "b_molecules": len(reach),
                    "b_all_unreachable": reach.all_products_unreachable.mean()
                    if len(reach)
                    else np.nan,
                    "b_any_unreachable": reach.any_product_unreachable.mean()
                    if len(reach)
                    else np.nan,
                    "b_exact_frac": (reach.reach_method == "exact").mean()
                    if len(reach)
                    else np.nan,
                }
            )
            peaks = scored[scored.n_spectra > 0]
            if fold != "test" and len(peaks):
                for tag, rule_col, null_col in (
                    ("c", "delta", "null"),
                    ("d_lit_l1", "lit_l1", "null_lit_l1"),
                    ("d_lit_ge10", "lit_ge10", "null_lit_ge10"),
                    ("d_best_l1", "best_l1", "null_best_l1"),
                    ("d_best_ge10", "best_ge10", "null_best_ge10"),
                ):
                    stat = paired(peaks, rule_col, null_col, seed)
                    item.update({f"{tag}_{k}": v for k, v in stat.items()})
                item["null_capped_molecules"] = int(peaks.null_capped.sum())
            if rule == T1 and fold != "test":
                e = scored[scored.e_spectra.fillna(0) > 0]
                item["e_exact_capped_molecules"] = int(e.e_exact_capped.sum())
                ctrl = control[control.split.isin(folds)]
                item["ctrl_exact_capped_molecules"] = int(ctrl.e_exact_capped.sum())
                for prefix, frame in (("e", e), ("ctrl", ctrl)):
                    for tag in ("exact", "pool"):
                        stats = isotope_stats(frame.e_detail, tag, seed)
                        name = prefix if tag == "exact" else f"{prefix}_pool"
                        item.update({f"{name}_{k}": v for k, v in stats.items()})
            out.append(item)
    summary = pd.DataFrame(out)
    decisions = [decide(row, summary) for _, row in summary.iterrows()]
    summary["admit_candidate"] = [d[0] for d in decisions]
    summary["admit_reason"] = [d[1] for d in decisions]
    return summary


def decide(row: pd.Series, summary: pd.DataFrame) -> tuple[str, str]:
    """Decision rule, read on train+val, with >=20 val molecules.

    Other rules: (b) AND ((c) OR (d)). T1: (b) AND ((c) OR (e)); its (d) is descriptive only
    because the topology null does not preserve Si (147 is a generic ion in TMS spectra). (e) needs
    the exhaustive-competitor share's lower bound above 0.5 and above the 1-TMS control's upper bound.
    """
    if row.fold != "train+val":
        return "", ""
    val = summary[(summary.rule == row.rule) & (summary.fold == "val")].iloc[0]
    n_val = int(val.matched_molecules)
    b = row.b_all_unreachable >= 0.95
    c = row.get("c_lo", np.nan) > 0
    d = row.get("d_lit_l1_lo", np.nan) > 0
    if row.rule == T1:
        e_lo = row.get("e_si2_closer_lo", np.nan)
        e = bool(e_lo > 0.5 and e_lo > row.get("ctrl_si2_closer_hi", np.nan))
        ok = bool(b and (c or e) and n_val >= 20)
        tail = f"d=descriptive ({d}); e={e} (ctrl hi {row.get('ctrl_si2_closer_hi', np.nan):.3f})"
    else:
        ok = bool(b and (c or d) and n_val >= 20)
        tail = f"d={d}"
    reason = (
        f"b={b} (all-unreachable {row.b_all_unreachable:.3f}); c={c}; {tail}; val_molecules={n_val}"
    )
    return ("yes" if ok else "no"), reason


# 8. Entry point ------------------------------------------------------------------------------


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 -- identity check, not security
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--null_reps", type=int, default=8)
    parser.add_argument("--exact_cap", type=int, default=50_000)
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--split_dp", type=Path, default=SPLIT)
    parser.add_argument("--out_dp", type=Path, default=Path("output/ei_tms_rewire_gate_2026_10_18"))
    args = parser.parse_args()
    if args.workers < 1 or args.null_reps < 1 or not 0 < args.sample_frac <= 1:
        parser.error("workers/null_reps must be positive and sample_frac in (0, 1]")
    args.out_dp.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    RDLogger.DisableLog("rdApp.*")

    # 1. Provenance before evaluation.
    paths = [DATA / "mol_df.pkl", DATA / "spec_df.pkl"]
    paths += [args.split_dp / f"{fold}_ids.csv" for fold in FOLDS]
    provenance = {
        "started_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "seed": args.seed,
        "null_reps": args.null_reps,
        "sample_frac": args.sample_frac,
        "exact_cap": args.exact_cap,
        "max_matches": MAX_MATCHES,
        "workers": args.workers,
        "python": sys.executable,
        "rdkit": rdBase.rdkitVersion,
        "torch": th.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs_md5": {str(p): md5(p) for p in paths},
        "source_md5": {str(Path(__file__)): md5(Path(__file__))},
        "rules": [
            {"name": r.name, "smarts": r.smarts, "delete": r.delete, "cut": r.cut, "form": r.form}
            for r in gate_rules()
        ],
    }
    (args.out_dp / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # 2. Load the e5 split; peaks of test spectra are dropped immediately.
    mol_df = pd.read_pickle(DATA / "mol_df.pkl").set_index("mol_id")
    spec_df = pd.read_pickle(DATA / "spec_df.pkl")
    ids = {f: set(pd.read_csv(args.split_dp / f"{f}_ids.csv").spec_id) for f in FOLDS}
    if any(ids[a] & ids[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("overlapping spectrum splits")
    labels = {i: f for f, s in ids.items() for i in s}
    spec_df["split"] = spec_df.spec_id.map(labels)
    spec_df = spec_df[spec_df.split.notna()]
    if not spec_df.prec_type.eq("[M]+").all():
        raise ValueError("gate requires [M]+ EI data")
    if spec_df.groupby("mol_id").split.nunique().max() != 1:
        raise ValueError("molecules cross splits")
    spec_df.loc[spec_df.split == "test", "peaks"] = None
    mol_split = spec_df.groupby("mol_id").split.first()
    mol_count = spec_df.groupby("mol_id").size()
    provenance["split_molecules"] = mol_split.value_counts().to_dict()
    provenance["split_spectra"] = spec_df.split.value_counts().to_dict()

    # 3. Structural census of every split molecule.
    started = datetime.now(UTC)
    mol_ids = sorted(mol_split.index)
    chunks = [
        [(int(i), mol_df.loc[i, "mol"]) for i in mol_ids[k : k + 1000]]
        for k in range(0, len(mol_ids), 1000)
    ]
    census = []
    with mp.Pool(args.workers, initializer=init_worker, initargs=(args.seed,)) as pool:
        for index, result in enumerate(pool.imap(census_chunk, chunks), 1):
            census.extend(result)
            if index % 20 == 0:
                LOGGER.info("Census %d/%d chunks", index, len(chunks))
        census = pd.DataFrame(census)
        census["split"] = census.mol_id.map(mol_split)
        census["n_spectra"] = census.mol_id.map(mol_count)
        census.to_csv(args.out_dp / "census.csv", index=False)
        LOGGER.info("Census:\n%s", census.groupby(["rule", "split"]).size().unstack().to_string())

        # 4. Score matched molecules; peaks go to train/val tasks only.
        peaks = spec_df.groupby("mol_id")
        rules_of = census.groupby("mol_id").rule.apply(list)
        tasks = []
        for mol_id, rules in rules_of.items():
            split = mol_split[mol_id]
            spectra = (
                []
                if split == "test"
                else list(
                    peaks.get_group(mol_id)[["spec_id", "peaks"]].itertuples(index=False, name=None)
                )
            )
            tasks.append(
                (
                    int(mol_id),
                    mol_df.loc[mol_id, "mol"],
                    split,
                    rules,
                    spectra,
                    args.seed,
                    args.null_reps,
                    args.exact_cap,
                    args.sample_frac,
                )
            )
        LOGGER.info("Scoring %d matched molecules (sample_frac=%s)", len(tasks), args.sample_frac)
        out = []
        for index, result in enumerate(
            pool.imap_unordered(process_molecule, tasks, chunksize=4), 1
        ):
            out.extend(result)
            if index % 250 == 0:
                LOGGER.info(
                    "Scored %d/%d molecules (%.1f min)",
                    index,
                    len(tasks),
                    (datetime.now(UTC) - started).total_seconds() / 60,
                )
        # 4b. (e) control: one-TMS molecules of train and val, scored like the T1 spectra.
        control_tasks = [
            (
                int(i),
                mol_df.loc[i, "mol"],
                mol_split[i],
                list(peaks.get_group(i)[["spec_id", "peaks"]].itertuples(index=False, name=None)),
            )
            for i in mol_ids
            if mol_split[i] != "test" and mol_df.loc[i, "mol"].HasSubstructMatch(TMS_GROUP)
        ]
        LOGGER.info("Scoring %d candidate control molecules", len(control_tasks))
        control = [
            r
            for r in pool.imap_unordered(control_molecule, control_tasks, chunksize=4)
            if r is not None
        ]
        rows = pd.DataFrame(out).sort_values(["rule", "mol_id"]).reset_index(drop=True)
    rows.drop(columns=["e_detail"]).to_csv(args.out_dp / "rows.csv", index=False)
    detail = [
        {"mol_id": r.mol_id, "split": r.split, **d}
        for r in rows[rows.e_spectra.fillna(0) > 0].itertuples()
        for d in json.loads(r.e_detail)
    ]
    pd.DataFrame(detail).to_csv(args.out_dp / "t1_isotope_spectra.csv", index=False)
    control = pd.DataFrame(control).sort_values("mol_id").reset_index(drop=True)
    pd.DataFrame(
        [
            {"mol_id": r.mol_id, "split": r.split, **d}
            for r in control.itertuples()
            for d in json.loads(r.e_detail)
        ]
    ).to_csv(args.out_dp / "t1_control_isotope_spectra.csv", index=False)

    # 5. Summary and decision.
    summary = summarize(rows, census, control, args.seed)
    summary.to_csv(args.out_dp / "summary.csv", index=False)
    provenance["finished_utc"] = datetime.now(UTC).isoformat()
    (args.out_dp / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    LOGGER.info("Done:\n%s", summary[summary.fold == "train+val"].T.to_string())


if __name__ == "__main__":
    main()
