"""Coverage gate for small-bridge excision and terminal-atom graft continuations.

Invoked by debug_masster_smarts_rule_gate.py continuations. This is an atom-set coverage
diagnostic, not a chemical mechanism simulator or a trained loss channel.

Two run modes share one evaluation core:

* ``--mode checkpoint`` (default, the 2026-09-28 gate): the candidate pool is read from a
  c0 checkpoint's annotated forward pass, ``[M+H]+`` only.
* ``--mode model_free`` (2026-09-29 G2/G2b): the pool is a stand-in, the connected components
  of every 0-, 1- and 2-cut mask of the precursor. It needs no checkpoint, GPU or split
  training data and works for either polarity.

Model-free operators:

* bridge: S minus X for a connected X of at most three heavy atoms with two non-ring external
  bonds to distinct endpoints, optionally restricted to registered bridge-shaped rule motifs
  (``--motifs``);
* graft: P plus t, with P a pool set and t a terminal O or F not adjacent to P within four
  bonds of it. The new bond joins t to the nearest boundary atom of P.

Valence is production-matched for the primary ("strict") variant: cut count as the builder
counts it and two valence slots per formed bond. The "optimistic" variant ignores formed bonds.

Subcommands, all through ``continuations``:
    continuations --mode model_free --select bearing --motifs all --operators bridge ...
    continuations confirm --train TRAIN.pkl --val VAL.pkl --families TRAIN.families.json ...
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import logging
import multiprocessing
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch

from fragnnet.frag.masster_templates import (
    MASSTER_RULE_NAMES,
    _is_connected,
    build_masster_templates,
    filter_masster_templates,
    formed_bond_count,
)
from fragnnet.runner import load_config
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_config_from_ckpt,
    init_model_from_ckpt,
)
from fragnnet.utils.setup_utils import get_dataset_cls

# Support both the script entry point and imports from pytest.
try:
    from ..masster_memorization.debug_masster_base_peak_tables import clustered_bootstrap
    from .debug_masster_smarts_rule_gate import TOLERANCE, ChannelScorer, edge_tuple, reachable_h
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "masster_memorization"))
    from debug_masster_base_peak_tables import clustered_bootstrap
    from debug_masster_smarts_rule_gate import TOLERANCE, ChannelScorer, edge_tuple, reachable_h

LOGGER = logging.getLogger(__name__)

# The two valence variants every score is reported under: "strict" is production-matched
# (cut count as the builder counts it, minus two slots per formed bond); "opt" ignores formed
# bonds.
VARIANTS = ("strict", "opt")
# Both operators form exactly one new bond: bridge joins the two endpoints, graft joins t to P.
OPERATOR_FORMED_BONDS = 1
GRAFT_PREFIX = "graft|"
# Family-file wildcards: admit every family of one operator (a "model operator" run).
WILDCARDS = {"*bridge": "bridge", "*graft": "graft"}
H_SHIFTS = np.arange(-4, 5)


@dataclass(frozen=True)
class Bridge:
    """Connected removed atoms with two acyclic attachment bonds."""

    removed: frozenset[int]
    endpoints: frozenset[int]
    environment: str


@dataclass(frozen=True)
class Graft:
    """A terminal O/F atom moved onto a candidate set across a short path."""

    base: tuple[int, ...]
    terminal: int
    anchor: int
    distance: int
    environment: str

    @property
    def product(self) -> tuple[int, ...]:
        """Atoms of the grafted ion: the base set plus the terminal atom, sorted."""
        return tuple(sorted((*self.base, self.terminal)))


@dataclass(frozen=True)
class Motif:
    """A registered rule's removed atoms, as bridge-shaped X, with explicit SMARTS.

    Attributes:
        name: Motif name.
        smarts: Query whose atom maps identify the removed atoms.
        removed_maps: Alternative removed-atom map-number tuples; each is one X.
        rules: Registered MASSter rules whose lost atoms the motif reproduces.
    """

    name: str
    smarts: str
    removed_maps: tuple[tuple[int, ...], ...]
    rules: tuple[str, ...]


# Bridge-shaped registered-rule motifs plus aromatic nitro NO. Patterns follow the registry
# (smarts_prepass.NL_RULES) except where stated. Every X is checked against
# enumerate_bridges, so a match whose removed atoms are not a two-bond acyclic bridge (for
# example a ring sulfoxide) is dropped.
MOTIFS: dict[str, Motif] = {
    m.name: m
    for m in (
        # SO2 between an aryl C and N or an acid/anion O: the union of aryl_sulfonamide_so2
        # ([NH]), primary/tertiary_aryl_sulfonamide_so2 ([NX3H2]/[NX3H0]) and
        # aryl_sulfonate_so2 ([O-,OH]). Sulfonate esters are not registered and stay out.
        Motif(
            "aryl_sulfonyl",
            "[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3,OX2H1,OX1-:5]",
            ((2, 3, 4),),
            (
                "aryl_sulfonamide_so2",
                "primary_aryl_sulfonamide_so2",
                "tertiary_aryl_sulfonamide_so2",
                "aryl_sulfonate_so2",
            ),
        ),
        # S=O between two carbons: sulfoxide_so_loss verbatim (X = S and its O).
        Motif(
            "sulfoxide",
            "[#6:1]-[SX3:2](~[OX1:3])-[#6:4]",
            ((2, 3),),
            ("sulfoxide_so_loss",),
        ),
        # N2O of a nitramine: n_nitramine_n2o verbatim (X = N+, =O and NH; the retained
        # endpoints are the anionic O and the carbon or nitrogen on NH).
        Motif(
            "nitramine",
            "[N+:1](=[O:2])([O-:3])-[NH:4]-[#6,#7:5]",
            ((1, 2, 4),),
            ("n_nitramine_n2o",),
        ),
        # Aromatic nitro NO: X is the N and either of its O atoms (no registered rule; the
        # confirmed [M+H]+ family of the 2026-09-28 excision gate).
        Motif(
            "aryl_nitro",
            "[c:1]-[N+:2](=[O:3])-[O-:4]",
            ((2, 3), (2, 4)),
            (),
        ),
    )
}


# 1. Bridges: enumeration and lift of pool sets ------------------------------------------


def enumerate_bridges(mol: Chem.Mol, max_atoms: int = 3) -> list[Bridge]:
    """Enumerate connected X with two non-ring external bonds and distinct endpoints.

    Group keys are canonical local SMILES with atom maps 1 for X and 2 for retained
    endpoints. Original hydrogen counts are retained; keys do not encode remote chemistry.
    """
    adjacency = [{neighbor.GetIdx() for neighbor in a.GetNeighbors()} for a in mol.GetAtoms()]
    current = {frozenset((a,)) for a in range(mol.GetNumAtoms())}
    bridges = []
    for _ in range(max_atoms):
        following = set()
        for removed in sorted(current, key=lambda x: tuple(sorted(x))):
            boundary = [(a, b) for a in sorted(removed) for b in sorted(adjacency[a] - removed)]
            endpoints = frozenset(b for _, b in boundary)
            if (
                len(boundary) == 2
                and len(endpoints) == 2
                and not any(mol.GetAtomWithIdx(a).IsInRing() for a in removed)
                and not any(mol.GetBondBetweenAtoms(a, b).IsInRing() for a, b in boundary)
            ):
                local = Chem.RWMol(mol)
                for atom in local.GetAtoms():
                    atom.SetAtomMapNum(1 if atom.GetIdx() in removed else 2)
                    atom.SetNumExplicitHs(mol.GetAtomWithIdx(atom.GetIdx()).GetTotalNumHs())
                    atom.SetNoImplicit(True)
                for atom in reversed(range(mol.GetNumAtoms())):
                    if atom not in removed | endpoints:
                        local.RemoveAtom(atom)
                environment = Chem.MolToSmiles(local, canonical=True, isomericSmiles=False)
                bridges.append(Bridge(removed, endpoints, environment))
            for atom in set().union(*(adjacency[a] for a in removed)) - removed:
                following.add(removed | {atom})
        current = following
    return bridges


def continuation_sets(pool: set[tuple[int, ...]], bridge: Bridge) -> set[tuple[int, ...]]:
    """Lift connected S containing X,a,b to the product S minus X."""
    required = bridge.removed | bridge.endpoints
    return {
        tuple(a for a in atoms if a not in bridge.removed)
        for atoms in pool
        if required.issubset(atoms)
    }


# 2. Registered-rule motifs --------------------------------------------------------------


@functools.cache
def _motif_query(name: str) -> tuple[Chem.Mol, dict[int, int]]:
    """Compile a motif query and map its atom-map numbers to query atom indices."""
    query = Chem.MolFromSmarts(MOTIFS[name].smarts)
    if query is None:
        raise ValueError(f"invalid motif SMARTS for {name}")
    return query, {a.GetAtomMapNum(): a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum()}


def resolve_motifs(spec: str) -> tuple[str, ...]:
    """Parse ``--motifs``: ``none``, ``all`` or a comma-separated list of motif names."""
    if spec == "none":
        return ()
    if spec == "all":
        return tuple(MOTIFS)
    names = tuple(part for part in spec.split(",") if part)
    unknown = sorted(set(names) - MOTIFS.keys())
    if unknown or not names:
        raise ValueError(f"unknown motifs {unknown}; choose from {sorted(MOTIFS)}, all or none")
    return names


def motif_removed_sets(mol: Chem.Mol, names: Sequence[str]) -> set[frozenset[int]]:
    """Return the atom sets X that the named motifs remove from ``mol``."""
    found: set[frozenset[int]] = set()
    for name in names:
        query, maps = _motif_query(name)
        for match in mol.GetSubstructMatches(query, uniquify=True, maxMatches=10_000):
            for removed_maps in MOTIFS[name].removed_maps:
                found.add(frozenset(match[maps[m]] for m in removed_maps))
    return found


def motif_bridges(mol: Chem.Mol, bridges: Iterable[Bridge], names: Sequence[str]) -> list[Bridge]:
    """Keep the bridges whose removed atoms are exactly a named motif's X."""
    if not names:
        return list(bridges)
    removed = motif_removed_sets(mol, names)
    return [bridge for bridge in bridges if bridge.removed in removed]


# 3. Graft operator ----------------------------------------------------------------------


def _local_environment(mol: Chem.Mol, maps: Mapping[int, int], keep: Iterable[int]) -> str:
    """Canonical SMILES of the induced subgraph on ``keep`` with atom maps and explicit H."""
    keep = set(keep)
    local = Chem.RWMol(mol)
    for atom in local.GetAtoms():
        atom.SetAtomMapNum(maps.get(atom.GetIdx(), 3))
        atom.SetNumExplicitHs(mol.GetAtomWithIdx(atom.GetIdx()).GetTotalNumHs())
        atom.SetNoImplicit(True)
    for atom in reversed(range(mol.GetNumAtoms())):
        if atom not in keep:
            local.RemoveAtom(atom)
    return Chem.MolToSmiles(local, canonical=True, isomericSmiles=False)


def graft_environment(mol: Chem.Mol, distances: np.ndarray, terminal: int, anchor: int) -> str:
    """Family key of a graft: the atoms on any shortest t-to-anchor path, with t=1, anchor=2.

    Path atoms carry map 3 and keep their hydrogen counts. The key is invariant to atom order
    because it uses every shortest-path atom, not one chosen path.
    """
    span = distances[terminal, anchor]
    path = [
        v for v in range(mol.GetNumAtoms()) if distances[terminal, v] + distances[v, anchor] == span
    ]
    return _local_environment(mol, {terminal: 1, anchor: 2}, path)


def terminal_atoms(mol: Chem.Mol, symbols: Sequence[str] = ("O", "F")) -> list[int]:
    """Atoms of the given elements with exactly one heavy neighbour."""
    return [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() in symbols and sum(n.GetAtomicNum() > 1 for n in a.GetNeighbors()) == 1
    ]


def enumerate_grafts(
    mol: Chem.Mol, pool: Iterable[tuple[int, ...]], max_distance: int = 4
) -> list[Graft]:
    """Enumerate P plus t for every pool set P and terminal O/F atom t.

    t must lie outside P and not be adjacent to it, with topological distance from P at most
    ``max_distance``. The new bond joins t to the atom of P nearest to t (lowest index on
    ties), which is necessarily a boundary atom of P.

    Args:
        mol: Precursor molecule.
        pool: Connected candidate atom sets, as sorted tuples.
        max_distance: Largest allowed bond distance from t to P.

    Returns:
        One Graft per (P, t) pair, ordered by P then t.
    """
    terminals = terminal_atoms(mol)
    if not terminals:
        return []
    distances = Chem.GetDistanceMatrix(mol)
    neighbor = {
        t: next(n.GetIdx() for n in mol.GetAtomWithIdx(t).GetNeighbors() if n.GetAtomicNum() > 1)
        for t in terminals
    }
    keys: dict[tuple[int, int], str] = {}
    grafts = []
    for base in sorted(pool):
        members = set(base)
        indices = np.asarray(base)
        for t in terminals:
            if t in members or neighbor[t] in members:
                continue
            row = distances[t, indices]
            nearest = int(row.argmin())
            if row[nearest] > max_distance:
                continue
            anchor = int(indices[nearest])
            if (t, anchor) not in keys:
                keys[t, anchor] = graft_environment(mol, distances, t, anchor)
            grafts.append(Graft(base, t, anchor, int(row[nearest]), keys[t, anchor]))
    return grafts


# 4. Stand-in candidate pool -------------------------------------------------------------


def standin_pool(
    mol: Chem.Mol, cap: int, rng: np.random.Generator
) -> tuple[set[tuple[int, ...]], dict[str, int]]:
    """Connected components of every 0-, 1- and 2-cut mask of the precursor.

    Cutting any one or two bonds and keeping every resulting piece is a superset of what the
    trained model retains, so this pool over-covers a real one. When the unique sets exceed
    ``cap``, every 0- and 1-cut set is kept and a seeded random sample of the sets that only
    arise from 2-cut masks fills the rest.

    Args:
        mol: Precursor molecule.
        cap: Maximum pool size.
        rng: Generator used only to sample under the cap.

    Returns:
        The pool and ``pool_raw`` / ``pool_kept`` / ``pool_truncated`` counts.

    Raises:
        ValueError: If the 0- and 1-cut sets alone exceed ``cap``.
    """
    n = mol.GetNumAtoms()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for index, (left, right) in enumerate(bonds):
        adjacency[left].append((right, index))
        adjacency[right].append((left, index))

    def components(cut: tuple[int, ...]) -> list[tuple[int, ...]]:
        seen = [False] * n
        pieces = []
        for start in range(n):
            if seen[start]:
                continue
            seen[start] = True
            stack, piece = [start], []
            while stack:
                atom = stack.pop()
                piece.append(atom)
                for other, index in adjacency[atom]:
                    if index not in cut and not seen[other]:
                        seen[other] = True
                        stack.append(other)
            pieces.append(tuple(sorted(piece)))
        return pieces

    single = {tuple(range(n))}
    for bond in range(len(bonds)):
        single.update(components((bond,)))
    double: set[tuple[int, ...]] = set()
    for pair in combinations(range(len(bonds)), 2):
        double.update(components(pair))
    double -= single
    raw = len(single) + len(double)
    if raw > cap:
        if len(single) > cap:
            raise ValueError(f"pool cap {cap} is below the {len(single)} 0- and 1-cut sets")
        ordered = sorted(double)
        keep = rng.choice(len(ordered), cap - len(single), replace=False)
        double = {ordered[i] for i in sorted(keep.tolist())}
    pool = single | double
    return pool, {"pool_raw": raw, "pool_kept": len(pool), "pool_truncated": int(raw > cap)}


# 5. Valence: production-matched cut and formed-bond counts ------------------------------


def crossing_bond_count(atoms: Iterable[int], edges: tuple[tuple[int, ...], ...]) -> int:
    """Number of bonds with exactly one endpoint in ``atoms``."""
    members = set(atoms)
    return sum((a in members) != (b in members) for a, b in zip(*edges))


def template_side_counts(
    mol: Chem.Mol, precursor_type: str
) -> dict[tuple[int, ...], tuple[int, int]]:
    """Cut and formed-bond counts of every eligible template side, as the builder assigns them.

    The builder gives each side ``len(template.broken_bond_indices)`` cuts (all bonds of the
    template, not only the side's crossing bonds) and ``formed_bond_count(side)`` formed
    bonds. A side shared by two templates keeps the more permissive counts.

    Args:
        mol: Precursor molecule the templates match.
        precursor_type: Adduct that gates template eligibility.

    Returns:
        Map from a side's atoms to ``(cut_count, formed_bonds)``.
    """
    edges = edge_tuple(mol)
    templates = filter_masster_templates(
        build_masster_templates(mol), precursor_type=precursor_type, fragmentation_mode=None
    )
    sides: dict[tuple[int, ...], tuple[int, int]] = {}
    for template in templates:
        for component in template.components:
            atoms = tuple(int(a) for a in component)
            counts = (len(template.broken_bond_indices), formed_bond_count(atoms, edges))
            previous = sides.get(atoms)
            if previous is None or previous[0] - 2 * previous[1] < counts[0] - 2 * counts[1]:
                sides[atoms] = counts
    return sides


def variant_counts(
    sets: Iterable[tuple[int, ...]],
    edges: tuple[tuple[int, ...], ...],
    variant: str,
    formed: int | None = None,
) -> dict[tuple[int, ...], tuple[int, int]]:
    """Cut and formed-bond counts for ``sets`` under one valence variant.

    The cut count is the set's own crossing-bond count. ``strict`` adds ``formed`` formed bonds
    (default: pieces minus one, as ``formed_bond_count``); ``opt`` adds none.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown valence variant {variant}")
    return {
        atoms: (
            crossing_bond_count(atoms, edges),
            0
            if variant == "opt"
            else (formed_bond_count(atoms, edges) if formed is None else formed),
        )
        for atoms in sets
    }


def _mol_graph(mol: Chem.Mol) -> Batch:
    return Batch.from_data_list(
        [get_mol_graph(mol, ["a_onehot", "a_degree"], ["b_degree"], 0, True)]
    )


def ensure_model(scorer: ChannelScorer, mol: Chem.Mol, precursor_index: int) -> Any:
    """Return the scorer's MASSter model, building it with a trivial query if needed."""
    if scorer.model is None:
        scorer(mol, [tuple(range(mol.GetNumAtoms()))], precursor_index)
    return scorer.model


def production_channels(
    scorer: ChannelScorer,
    mol: Chem.Mol,
    sets: Sequence[tuple[int, ...]],
    precursor_index: int,
    cut_counts: Sequence[int],
    formed_counts: Sequence[int],
    graph: Batch | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MASSter's ion channels for atom sets with explicit cut and formed-bond counts.

    This is the call the model makes for template sides
    (``_fragment_masses_and_valid_channels`` with ``fragment_cut_counts`` and
    ``fragment_formed_bond_counts``), on CPU. ``ChannelScorer`` derives cuts from the set's own
    crossing bonds and cannot express a formed bond except as a flat ``reconnect``.

    Args:
        scorer: Scorer whose model is reused (built on first use).
        mol: Precursor molecule.
        sets: Atom sets.
        precursor_index: Adduct index in the model's ``precursor_types``.
        cut_counts: Severed bonds per set.
        formed_counts: Bonds formed inside each set.
        graph: Prebuilt precursor graph; built when omitted.

    Returns:
        ``(mz, valid)``, both shaped ``(len(sets), 9)`` over H shifts -4..+4.
    """
    if not len(sets) == len(cut_counts) == len(formed_counts):
        raise ValueError("sets, cut_counts and formed_counts must have equal length")
    if not sets:
        return np.zeros((0, 9)), np.zeros((0, 9), bool)
    model = ensure_model(scorer, mol, precursor_index)
    graph = _mol_graph(mol) if graph is None else graph
    atoms = th.tensor([a for s in sets for a in s])
    fragments = th.tensor([i for i, s in enumerate(sets) for _ in s])
    with th.no_grad():
        mzs, valid = model._fragment_masses_and_valid_channels(
            graph,
            atoms,
            fragments,
            th.zeros(len(sets), dtype=th.long),
            len(sets),
            th.tensor([precursor_index]),
            th.tensor(list(cut_counts)),
            th.tensor(list(formed_counts)),
        )
    return mzs.double().numpy()[:, 0, :], valid.numpy()[:, 0, :]


def covered(peaks: np.ndarray, channels: np.ndarray, tolerance: tuple[float, float]) -> np.ndarray:
    """Match peaks to sorted channels with the instrument's relative/floor tolerance."""
    if not channels.size:
        return np.zeros(peaks.size, dtype=bool)
    channels = np.sort(channels)
    relative, floor = tolerance
    upper = np.clip(np.searchsorted(channels, peaks), 0, channels.size - 1)
    lower = np.maximum(upper - 1, 0)
    return (abs(peaks - channels[upper]) <= np.maximum(channels[upper], floor) * relative) | (
        abs(peaks - channels[lower]) <= np.maximum(channels[lower], floor) * relative
    )


# A valence specification is False (optimistic: no formed bond), True (legacy strict: one
# formed bond, cuts must be at least two) or an explicit (cut_count, formed_bonds) pair.
Valence = bool | tuple[int, int]


class MoleculeCoverage:
    """Cache the channel and exhaustive same-composition queries for one molecule.

    Channels are cached per polarity (``precursor_index``); the reach search depends only on
    composition and hydrogen counts, so it is shared.
    """

    def __init__(
        self, mol: Chem.Mol, scorer: ChannelScorer, cap: int, precursor_index: int = 0
    ) -> None:
        self.mol, self.scorer, self.cap = mol, scorer, cap
        self.precursor_index = precursor_index
        self.edges = edge_tuple(mol)
        self.symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        self.hydrogens = [a.GetTotalNumHs() for a in mol.GetAtoms()]
        self.by_polarity: dict[int, dict] = defaultdict(dict)
        self.reaches: dict = {}
        self._graph: Batch | None = None

    @property
    def channels(self) -> dict:
        """Channel cache of the current polarity, keyed by (atoms, valence spec)."""
        return self.by_polarity[self.precursor_index]

    def check_polarity(self, prec_type: str) -> None:
        """Raise unless the cache's adduct index is the spectrum's precursor type."""
        model = ensure_model(self.scorer, self.mol, self.precursor_index)
        expected = model.precursor_types[self.precursor_index]
        if expected != prec_type:
            raise ValueError(
                f"spectrum precursor type {prec_type} does not match adduct index "
                f"{self.precursor_index} ({expected})"
            )

    def _resolve(self, atoms: tuple[int, ...], spec: Valence) -> tuple[int, int]:
        if isinstance(spec, tuple):
            return spec
        cuts = crossing_bond_count(atoms, self.edges)
        if not spec:
            return cuts, 0
        if cuts < 2:
            raise ValueError("a new bond requires two freed boundary valence slots")
        return cuts, 1

    def _spec(
        self,
        atoms: tuple[int, ...],
        reconnect: bool,
        counts: Mapping[tuple[int, ...], tuple[int, int]] | None,
    ) -> Valence:
        return counts[atoms] if counts is not None and atoms in counts else reconnect

    def prepare(
        self,
        sets: Iterable[tuple[int, ...]],
        reconnect: bool = False,
        counts: Mapping[tuple[int, ...], tuple[int, int]] | None = None,
    ) -> None:
        """Compute and cache the valid channels of every set not yet cached.

        Args:
            sets: Atom sets.
            reconnect: Valence spec for sets without an entry in ``counts``.
            counts: Explicit ``(cut_count, formed_bonds)`` per set.
        """
        store = self.channels
        missing = sorted(s for s in sets if (s, self._spec(s, reconnect, counts)) not in store)
        if not missing:
            return
        specs = [self._spec(s, reconnect, counts) for s in missing]
        resolved = [self._resolve(s, spec) for s, spec in zip(missing, specs)]
        if self._graph is None:
            self._graph = _mol_graph(self.mol)
        mzs, valid = production_channels(
            self.scorer,
            self.mol,
            missing,
            self.precursor_index,
            [c for c, _ in resolved],
            [f for _, f in resolved],
            self._graph,
        )
        for atoms, spec, mass, allowed in zip(missing, specs, mzs, valid, strict=True):
            own_h = sum(self.hydrogens[a] for a in atoms)
            store[atoms, spec] = (mass[allowed], H_SHIFTS[allowed] + own_h)

    def mz_array(
        self,
        sets: Iterable[tuple[int, ...]],
        reconnect: bool = False,
        counts: Mapping[tuple[int, ...], tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """Sorted valid channel m/z of all ``sets`` concatenated."""
        ordered = sorted(sets)
        self.prepare(ordered, reconnect, counts)
        arrays = [self.channels[s, self._spec(s, reconnect, counts)][0] for s in ordered]
        return np.sort(np.concatenate(arrays)) if arrays else np.zeros(0)

    def _hit_set(
        self,
        atoms: tuple[int, ...],
        spec: Valence,
        peaks: np.ndarray,
        tolerance: tuple,
        unique: bool,
    ) -> tuple[np.ndarray, int]:
        masses, counts_h = self.channels[atoms, spec]
        matches = covered(peaks, masses, tolerance)
        if not matches.any() or not unique:
            return matches, 0
        composition = Counter(self.symbols[a] for a in atoms)
        matched_channels = np.array(
            [covered(peaks, np.array([mass]), tolerance).any() for mass in masses]
        )
        masses, counts_h = masses[matched_channels], counts_h[matched_channels]
        targets_h = frozenset(int(h) for h in counts_h)
        key = (tuple(sorted(composition.items())), targets_h)
        if key not in self.reaches:
            self.reaches[key] = reachable_h(
                self.mol,
                self.edges,
                composition,
                len(atoms),
                cap=self.cap,
                required_h=targets_h,
            )
        reach = self.reaches[key]
        if reach is None:
            return np.zeros(peaks.size, bool), 1
        # Conservative: any connected twin within +-4 H excludes the channel,
        # even if its own valence filter would reject that particular H shift.
        keep = [not any(abs(h - r) <= 4 for r in reach) for h in counts_h]
        return covered(peaks, masses[np.asarray(keep, bool)], tolerance), 0

    def hit(
        self,
        sets: Iterable[tuple[int, ...]],
        peaks: np.ndarray,
        tolerance: tuple,
        *,
        reconnect: bool = False,
        unique: bool = False,
        counts: Mapping[tuple[int, ...], tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Peaks matched by the sets' valid channels, and the number of capped sets.

        Args:
            sets: Atom sets.
            peaks: Peak m/z values.
            tolerance: ``(relative, floor)`` instrument tolerance.
            reconnect: Legacy valence flag for sets without an entry in ``counts``.
            unique: Drop channels with a connected same-composition twin within +-4 H, and
                credit nothing for a set whose twin search hit the cap.
            counts: Explicit ``(cut_count, formed_bonds)`` per set.

        Returns:
            Boolean per peak, and the count of capped sets (never credited).
        """
        ordered = sorted(sets)
        self.prepare(ordered, reconnect, counts)
        hit = np.zeros(peaks.size, bool)
        if not ordered or not peaks.size:
            return hit, 0
        specs = [self._spec(s, reconnect, counts) for s in ordered]
        arrays = [self.channels[s, spec][0] for s, spec in zip(ordered, specs)]
        lengths = np.fromiter((a.size for a in arrays), dtype=np.int64, count=len(arrays))
        if not lengths.sum():
            return hit, 0
        # Vectorised prefilter: only sets with a channel near some peak need the exact test.
        merged = np.concatenate(arrays)
        owner = np.repeat(np.arange(len(ordered)), lengths)
        relative, floor = tolerance
        ordered_peaks = np.sort(peaks)
        upper = np.clip(np.searchsorted(ordered_peaks, merged), 0, peaks.size - 1)
        lower = np.maximum(upper - 1, 0)
        window = np.maximum(merged, floor) * relative * 1.01
        near = (abs(merged - ordered_peaks[upper]) <= window) | (
            abs(merged - ordered_peaks[lower]) <= window
        )
        capped = 0
        for index in np.unique(owner[near]):
            matches, was_capped = self._hit_set(
                ordered[index], specs[index], peaks, tolerance, unique
            )
            hit |= matches
            capped += was_capped
        return hit, capped


def random_disconnected(
    mol: Chem.Mol, products: set[tuple[int, ...]], rng: np.random.Generator
) -> tuple[set[tuple[int, ...]], int]:
    """Match product count and size with distinct random disconnected atom sets."""
    edges = edge_tuple(mol)
    nulls = set()
    for product in sorted(products):
        for _ in range(1000):
            draw = tuple(
                sorted(rng.choice(mol.GetNumAtoms(), len(product), replace=False).tolist())
            )
            if draw not in nulls and not _is_connected(draw, edges):
                nulls.add(draw)
                break
    return nulls, len(products) - len(nulls)


# 6. Legacy checkpoint-pool spectrum evaluation ------------------------------------------


def evaluate_spectrum(
    mol: Chem.Mol,
    pool: set[tuple[int, ...]],
    pool_channels: np.ndarray,
    bridges: list[Bridge],
    templates: set[tuple[int, ...]],
    peaks: np.ndarray,
    weights: np.ndarray,
    tolerance: tuple,
    cache: MoleculeCoverage,
    rng: np.random.Generator,
    families: set[str] | None = None,
    *,
    template_counts: Mapping[tuple[int, ...], tuple[int, int]] | None = None,
    prec_type: str | None = None,
) -> list[dict]:
    """Compare continuation products to the pool, templates, whole products, and null.

    ``template_counts`` gives template sides their production cut and formed-bond counts;
    without it they are scored optimistically, as before 2026-09-29. ``prec_type`` raises if it
    disagrees with the cache's adduct index.
    """
    if prec_type is not None:
        cache.check_polarity(prec_type)
    ordinary = covered(peaks, pool_channels, tolerance)
    template_hit, _ = cache.hit(templates, peaks, tolerance, counts=template_counts)
    baseline = ordinary | template_hit
    whole = {tuple(a for a in range(mol.GetNumAtoms()) if a not in b.removed) for b in bridges}
    whole_hit, _ = cache.hit(whole, peaks, tolerance)
    # All generic whole products join the baseline to isolate post-excision cleavage.
    continuation_baseline = baseline | whole_hit
    by_environment = defaultdict(set)
    for bridge in bridges:
        if families is not None and bridge.environment not in families:
            continue
        by_environment[bridge.environment] |= continuation_sets(pool, bridge) - whole
    by_environment["__all__"] = set().union(*by_environment.values())
    records = []
    for environment, products in sorted(by_environment.items()):
        nulls, missing = random_disconnected(mol, products, rng)
        raw, _ = cache.hit(products, peaks, tolerance)
        remaining_peaks = peaks[~continuation_baseline]
        remaining_weights = weights[~continuation_baseline]
        unique, capped = cache.hit(products, remaining_peaks, tolerance, unique=True)
        strict, _ = cache.hit(products, remaining_peaks, tolerance, reconnect=True, unique=True)
        null, null_capped = cache.hit(nulls, remaining_peaks, tolerance, unique=True)
        strict_null, _ = cache.hit(nulls, remaining_peaks, tolerance, reconnect=True, unique=True)
        records.append(
            {
                "environment": environment,
                "pool_count": len(pool),
                "bridges": len(bridges),
                "products": len(products),
                "null_count": len(nulls),
                "null_missing": missing,
                "pool_covered": float(weights[ordinary].sum()),
                "template_added": float(weights[template_hit & ~ordinary].sum()),
                "whole_added": float(weights[whole_hit & ~baseline].sum()),
                "continuation_added": float(weights[raw & ~continuation_baseline].sum()),
                "unique": float(remaining_weights[unique].sum()),
                "strict_unique": float(remaining_weights[strict].sum()),
                "null": float(remaining_weights[null].sum()) if not missing else np.nan,
                "strict_null": float(remaining_weights[strict_null].sum())
                if not missing
                else np.nan,
                "capped": capped,
                "null_capped": null_capped,
            }
        )
    return records


# 7. Model-free molecule units and the unit-based spectrum evaluation --------------------


@dataclass(frozen=True)
class UnitConfig:
    """Model-free evaluation settings shared by every molecule of a run."""

    precursor_type: str
    precursor_index: int
    operators: tuple[str, ...]
    motifs: tuple[str, ...]
    families: frozenset[str] | None
    pool_cap: int
    graft_max_distance: int
    graft_whole_max_cuts: int
    reach_cap: int


@dataclass
class MoleculeUnit:
    """Everything about one molecule that does not depend on the spectrum.

    Attributes:
        products: Continuation atom sets per environment (family) name.
        operator_of: ``bridge``, ``graft`` or ``union`` per environment.
        loads: ``(distinct (composition, H) formulas, those not in the baseline)`` per
            environment.
        arrays: Sorted baseline channel m/z per named baseline component.
        real_counts: Cut and formed-bond counts per variant for every product set.
        stats: Molecule-level counts for the run log.
    """

    mol: Chem.Mol
    cache: MoleculeCoverage
    pool: set[tuple[int, ...]]
    products: dict[str, set[tuple[int, ...]]]
    operator_of: dict[str, str]
    loads: dict[str, tuple[int, int]]
    arrays: dict[str, np.ndarray]
    real_counts: dict[str, dict[tuple[int, ...], tuple[int, int]]]
    stats: dict[str, Any]


def family_allowed(environment: str, operator: str, families: frozenset[str] | None) -> bool:
    """Whether a family passes the frozen family list (``*bridge``/``*graft`` admit all)."""
    return (
        families is None
        or environment in families
        or any(WILDCARDS[w] == operator for w in families if w in WILDCARDS)
    )


def _formula_key(cache: MoleculeCoverage, atoms: tuple[int, ...]) -> tuple:
    composition = Counter(cache.symbols[a] for a in atoms)
    return tuple(sorted(composition.items())), sum(cache.hydrogens[a] for a in atoms)


def build_unit(
    mol: Chem.Mol, scorer: ChannelScorer, config: UnitConfig, rng: np.random.Generator
) -> MoleculeUnit:
    """Build the stand-in pool, operators' products, baselines and load counts of a molecule.

    Args:
        mol: Precursor molecule (parsed from its SMILES, as the training builder does).
        scorer: Shared channel scorer.
        config: Run settings.
        rng: Generator for pool truncation.

    Returns:
        The molecule's unit.
    """
    unknown = sorted(set(config.operators) - {"bridge", "graft"})
    if unknown or not config.operators:
        raise ValueError(f"operators must be a non-empty subset of bridge, graft; got {unknown}")
    cache = MoleculeCoverage(mol, scorer, config.reach_cap, config.precursor_index)
    edges = cache.edges

    # 1. Stand-in pool, registered template sides and generic whole-precursor bridge products.
    pool, pool_stats = standin_pool(mol, config.pool_cap, rng)
    bridges = enumerate_bridges(mol)
    whole_bridge = {
        tuple(a for a in range(mol.GetNumAtoms()) if a not in b.removed) for b in bridges
    }
    templates = template_side_counts(mol, config.precursor_type)

    # 2. Grafts. Whole-precursor grafts (one cut plus one transfer) join the baseline.
    grafts = (
        enumerate_grafts(mol, pool, config.graft_max_distance)
        if "graft" in config.operators
        else []
    )
    whole_graft = {
        g.product
        for g in grafts
        if crossing_bond_count(g.base, edges) <= config.graft_whole_max_cuts
    }

    # 3. Products per family.
    products: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    operator_of: dict[str, str] = {}
    evaluated = motif_bridges(mol, bridges, config.motifs) if "bridge" in config.operators else []
    for bridge in evaluated:
        if family_allowed(bridge.environment, "bridge", config.families):
            products[bridge.environment] |= continuation_sets(pool, bridge) - whole_bridge
            operator_of[bridge.environment] = "bridge"
    if "bridge" in config.operators:
        products["__all__"] = set().union(*products.values())
        operator_of["__all__"] = "bridge"
    graft_all: set[tuple[int, ...]] = set()
    for graft in grafts:
        key = GRAFT_PREFIX + graft.environment
        if graft.product in whole_graft or not family_allowed(key, "graft", config.families):
            continue
        products[key].add(graft.product)
        operator_of[key] = "graft"
        graft_all.add(graft.product)
    if "graft" in config.operators:
        products["__graft__"] = graft_all
        operator_of["__graft__"] = "graft"
    if len(config.operators) == 2:
        products["__union__"] = set(products["__all__"]) | graft_all
        operator_of["__union__"] = "union"

    # 4. Baseline channel arrays, both valence variants.
    arrays = {"pool": cache.mz_array(pool)}
    for variant in VARIANTS:
        arrays[f"template_{variant}"] = cache.mz_array(
            templates,
            counts=templates
            if variant == "strict"
            else {a: (c, 0) for a, (c, _) in templates.items()},
        )
        arrays[f"whole_bridge_{variant}"] = cache.mz_array(
            whole_bridge, counts=variant_counts(whole_bridge, edges, variant)
        )
        arrays[f"whole_graft_{variant}"] = cache.mz_array(
            whole_graft, counts=variant_counts(whole_graft, edges, variant)
        )
    every = set().union(*products.values()) if products else set()
    real_counts = {v: variant_counts(every, edges, v) for v in VARIANTS}
    for variant in VARIANTS:
        cache.prepare(every, counts=real_counts[variant])

    # 5. Load: distinct (composition, H) formulas per family, and those beyond the baseline.
    known = {
        _formula_key(cache, atoms) for atoms in (*pool, *templates, *whole_bridge, *whole_graft)
    }
    loads = {}
    for environment, sets in products.items():
        formulas = {_formula_key(cache, atoms) for atoms in sets}
        loads[environment] = (len(formulas), len(formulas - known))

    stats = {
        "n_atoms": mol.GetNumAtoms(),
        **pool_stats,
        "bridges": len(bridges),
        "bridges_evaluated": len(evaluated),
        "templates": len(templates),
        "grafts": len(grafts),
        "graft_products": len(graft_all),
        "whole_bridge": len(whole_bridge),
        "whole_graft": len(whole_graft),
        "products": {k: len(v) for k, v in products.items() if k.startswith("__")},
    }
    return MoleculeUnit(
        mol, cache, pool, dict(products), operator_of, loads, arrays, real_counts, stats
    )


def coverage_rise(a0: np.ndarray | float, da: np.ndarray | float, n: np.ndarray | float):
    """Rise of the coverage ceiling ``sqrt(A/N)`` when ``da`` more squared intensity is covered.

    Args:
        a0: Sum of squared L1-normalised intensity of baseline-covered peaks.
        da: Sum of squared intensity of newly covered peaks.
        n: Sum of squared intensity of all peaks.

    Returns:
        ``sqrt((a0 + da) / n) - sqrt(a0 / n)``, which lies between ``da / (2 n)`` and
        ``sqrt(da / n)``.
    """
    return np.sqrt((np.asarray(a0) + da) / n) - np.sqrt(np.asarray(a0) / n)


def evaluate_spectrum_unit(
    unit: MoleculeUnit,
    peaks: np.ndarray,
    weights: np.ndarray,
    tolerance: tuple[float, float],
    rng: np.random.Generator,
    prec_type: str,
) -> tuple[dict, list[dict]]:
    """Score every family of a molecule against one spectrum, in both valence variants.

    The baseline is the stand-in pool, registered template sides and generic whole-precursor
    bridge products (``B``); graft families add the whole-precursor graft products (``G``).
    Newly hit peaks are those on no baseline channel and with no connected same-composition
    twin within +-4 H; capped twin searches credit nothing.

    Args:
        unit: The molecule's unit.
        peaks: Peak m/z, shape ``(P,)``.
        weights: L1-normalised linear intensities, shape ``(P,)``.
        tolerance: ``(relative, floor)`` instrument tolerance.
        rng: Generator for the random null.
        prec_type: The spectrum's precursor type; raises if it disagrees with the cache.

    Returns:
        The spectrum row (``n = sum w^2``, baseline ``a0`` per baseline and variant) and one
        record per family.
    """
    cache = unit.cache
    cache.check_polarity(prec_type)
    if peaks.shape != weights.shape:
        raise ValueError("peaks and weights must have the same shape")
    w2 = weights**2
    total = float(w2.sum())
    if not total > 0:
        raise ValueError("spectrum carries no intensity")

    # 1. Baseline coverage of the spectrum, per variant and per operator baseline.
    cover = {name: covered(peaks, array, tolerance) for name, array in unit.arrays.items()}
    base = {}
    for variant in VARIANTS:
        plain = cover["pool"] | cover[f"template_{variant}"] | cover[f"whole_bridge_{variant}"]
        base["B", variant] = plain
        base["G", variant] = plain | cover[f"whole_graft_{variant}"]
    row = {
        "n_peaks": int(peaks.size),
        "n": total,
        **{f"a0_{b}_{v}": float(w2[base[b, v]].sum()) for b in "BG" for v in VARIANTS},
        "pool_covered": float(weights[cover["pool"]].sum()),
        "template_added": float(weights[cover["template_opt"] & ~cover["pool"]].sum()),
        "whole_added": float(
            weights[cover["whole_bridge_opt"] & ~(cover["pool"] | cover["template_opt"])].sum()
        ),
        "graft_whole_added": float(weights[cover["whole_graft_opt"] & ~base["B", "opt"]].sum()),
    }

    # 2. Draw every family's null and price all of them in one batch per variant.
    nulls = {env: random_disconnected(cache.mol, sets, rng) for env, sets in unit.products.items()}
    all_null = set().union(*(n for n, _ in nulls.values())) if nulls else set()
    null_counts = {
        v: variant_counts(all_null, cache.edges, v, OPERATOR_FORMED_BONDS) for v in VARIANTS
    }
    for variant in VARIANTS:
        cache.prepare(all_null, counts=null_counts[variant])

    # 3. Real and null hits per family. Hit indices are positions in ``peaks``.
    records = []
    for environment, products in sorted(unit.products.items()):
        operator = unit.operator_of[environment]
        baseline_name = "B" if operator == "bridge" else "G"
        null_sets, missing = nulls[environment]
        record = {
            "environment": environment,
            "operator": operator,
            "baseline": baseline_name,
            "pool_count": len(unit.pool),
            "products": len(products),
            "null_count": len(null_sets),
            "null_missing": missing,
            "load_formulas": unit.loads[environment][0],
            "load_new_formulas": unit.loads[environment][1],
            **{
                k: row[k]
                for k in ("pool_covered", "template_added", "whole_added", "graft_whole_added")
            },
        }
        raw_mask = ~base[baseline_name, "opt"]
        raw, _ = cache.hit(products, peaks, tolerance, counts=unit.real_counts["opt"])
        record["continuation_added"] = float(weights[raw & raw_mask].sum())
        for variant in VARIANTS:
            keep = ~base[baseline_name, variant]
            remaining = peaks[keep]
            real, capped = cache.hit(
                products, remaining, tolerance, unique=True, counts=unit.real_counts[variant]
            )
            null, null_capped = cache.hit(
                null_sets, remaining, tolerance, unique=True, counts=null_counts[variant]
            )
            full = np.zeros(peaks.size, bool)
            full[keep] = real
            null_full = np.zeros(peaks.size, bool)
            null_full[keep] = null
            prefix = "" if variant == "opt" else "strict_"
            tag = "" if variant == "opt" else "_strict"
            record[prefix + "unique"] = float(weights[full].sum())
            record[prefix + "null"] = float(weights[null_full].sum()) if not missing else np.nan
            record["capped" + tag] = capped
            record["null_capped" + tag] = null_capped
            record[f"da_{variant}"] = float(w2[full].sum())
            record[f"da_{variant}_null"] = float(w2[null_full].sum()) if not missing else np.nan
            record[f"hit_{variant}"] = tuple(int(i) for i in np.flatnonzero(full))
        records.append(record)
    return row, records


# 8. Coverage ceiling in the pre-registered unit -----------------------------------------


def per_spectrum_ceiling(
    spectra: pd.DataFrame,
    records: pd.DataFrame,
    environments: Sequence[str],
    *,
    variant: str = "strict",
) -> pd.DataFrame:
    """Ceiling terms per evaluated spectrum for the union of the given families' new hits.

    Args:
        spectra: One row per evaluated spectrum (``spec_id``, ``mol_id``, ``n``, ``weights``
            and the ``a0_*`` columns).
        records: Family records with ``hit_<variant>`` index tuples.
        environments: Families whose newly hit peaks are unioned per spectrum. They must share
            one baseline.
        variant: ``strict`` (production valence) or ``opt``.

    Returns:
        Per spectrum: ``n``, ``a0``, ``da`` (squared intensity of newly hit peaks), ``dl1``
        (their intensity), ``rise``, ``lower = da / (2 n)`` and ``upper = sqrt(da / n)``.

    Raises:
        ValueError: If the families use different baselines.
    """
    chosen = records[records.environment.isin(list(environments))]
    baselines = set(chosen.baseline)
    if len(baselines) > 1:
        raise ValueError(f"families use different baselines {sorted(baselines)}; use __union__")
    baseline = baselines.pop() if baselines else "B"
    hits: dict[Any, set[int]] = defaultdict(set)
    for spec_id, indices in zip(chosen.spec_id, chosen[f"hit_{variant}"]):
        hits[spec_id].update(indices)
    rows = []
    for spec in spectra.itertuples():
        weights = np.asarray(spec.weights, float)
        indices = sorted(hits.get(spec.spec_id, ()))
        da = float((weights[indices] ** 2).sum()) if indices else 0.0
        dl1 = float(weights[indices].sum()) if indices else 0.0
        a0 = getattr(spec, f"a0_{baseline}_{variant}")
        rows.append(
            {
                "spec_id": spec.spec_id,
                "mol_id": spec.mol_id,
                "n": spec.n,
                "a0": a0,
                "da": da,
                "dl1": dl1,
                "rise": float(coverage_rise(a0, da, spec.n)),
                "lower": da / (2 * spec.n),
                "upper": float(np.sqrt(da / spec.n)),
            }
        )
    return pd.DataFrame(rows)


def stratum_ceiling(
    per_spectrum: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    census: bool,
    reps: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Whole-stratum mean ceiling rise with a molecule-clustered bootstrap interval.

    Every stratum spectrum weighs equally. A molecule contributes its stratum spectrum count
    times the mean rise of its evaluated spectra, so a molecule sampled with at most four
    spectra stands for all of its own. With ``census`` every stratum molecule is in the
    denominator and an unevaluated one contributes zero (right when molecules without the
    operator's motif cannot contribute); otherwise only the evaluated molecules are, which
    estimates the stratum mean from a random sample of molecules.

    Args:
        per_spectrum: Output of ``per_spectrum_ceiling``.
        universe: One row per stratum molecule: ``mol_id``, ``n_spectra``.
        census: Whether unevaluated stratum molecules count as zeros.
        reps: Bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        For each of ``rise``, ``lower``, ``upper`` and ``l1`` (percentage points of intensity)
        the estimate and ``_lo``/``_hi`` 95% limits, plus molecule and spectrum counts.

    Raises:
        ValueError: If an evaluated molecule is missing from the universe.
    """
    counts = universe.set_index("mol_id").n_spectra.astype(float)
    means = per_spectrum.groupby("mol_id")[["rise", "lower", "upper", "dl1"]].mean()
    if not means.index.isin(counts.index).all():
        raise ValueError("evaluated molecules missing from the stratum universe")
    index = counts.index if census else means.index
    weights = counts.reindex(index).to_numpy()
    values = {
        name: (means[name].reindex(index).fillna(0.0).to_numpy() * weights)
        * (100.0 if name == "dl1" else 1.0)
        for name in ("rise", "lower", "upper", "dl1")
    }
    denominator = weights.sum()
    draws = clustered_bootstrap(
        np.arange(len(index)), {**values, "n": weights}, reps=reps, seed=seed
    )
    names = [*values, "n"]
    result: dict[str, float] = {
        "molecules": int(len(index)),
        "evaluated_molecules": int(len(means)),
        "spectra": int(denominator),
        "census": bool(census),
    }
    for column, name in enumerate(names[:-1]):
        ratios = draws[:, column] / draws[:, -1]
        key = "l1" if name == "dl1" else name
        result[key] = float(values[name].sum() / denominator)
        result[key + "_lo"], result[key + "_hi"] = (
            float(x) for x in np.quantile(ratios, [0.025, 0.975])
        )
    return result


# 9. Family summaries, frozen-family selection and confirmation --------------------------


def summarize(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Bootstrap paired differences over molecule means, keeping incomplete cases visible."""
    rows = []
    rng = np.random.default_rng(seed)
    for environment, group in frame.groupby("environment"):
        means = group.groupby("mol_id")[["unique", "null", "strict_unique", "strict_null"]].mean()
        paired = means.dropna()
        row = {
            "environment": environment,
            "molecules": len(means),
            "spectra": len(group),
            "products_mean": group.products.mean(),
            "products_max": group.products.max(),
            "capped": int(group.capped.sum()),
            "null_capped": int(group.null_capped.sum()),
            "null_missing": int(group.null_missing.sum()),
        }
        for column in (
            "pool_covered",
            "template_added",
            "whole_added",
            "continuation_added",
            "unique",
            "null",
            "strict_unique",
            "strict_null",
        ):
            row[column + "_pct"] = 100 * group.groupby("mol_id")[column].mean().mean()
        for prefix, real, null in (
            ("", "unique", "null"),
            ("strict_", "strict_unique", "strict_null"),
        ):
            diff = (paired[real] - paired[null]).to_numpy() * 100
            row[prefix + "diff_pct"] = diff.mean() if diff.size else np.nan
            if diff.size >= 3:
                draws = rng.choice(diff, (2000, diff.size), replace=True).mean(axis=1)
                row[prefix + "lo"], row[prefix + "hi"] = np.quantile(draws, [0.025, 0.975])
            else:
                row[prefix + "lo"] = row[prefix + "hi"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


CAP_COLUMNS = ("capped", "null_capped", "capped_strict", "null_capped_strict")


def summarize_unit(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """``summarize`` plus cap rates, load and squared-intensity columns of the unit records."""
    base = summarize(frame, seed)
    grouped = frame.groupby("environment")
    caps = frame[list(CAP_COLUMNS)].sum(axis=1)
    extra = pd.DataFrame(
        {
            "operator": grouped.operator.first(),
            "capped_strict": grouped.capped_strict.sum(),
            "null_capped_strict": grouped.null_capped_strict.sum(),
            "spectra_any_cap_pct": 100 * (caps > 0).groupby(frame.environment).mean(),
            "spectra_null_missing_pct": 100
            * (frame.null_missing > 0).groupby(frame.environment).mean(),
            "load_formulas_mean": grouped.load_formulas.mean(),
            "load_formulas_max": grouped.load_formulas.max(),
            "load_new_formulas_mean": grouped.load_new_formulas.mean(),
            "da_strict_mean": grouped.da_strict.mean(),
        }
    ).reset_index()
    return base.merge(extra, on="environment")


def select_families(summary: pd.DataFrame, min_molecules: int = 10) -> list[str]:
    """Frozen train selection: at least ``min_molecules``, strict CI above zero, nothing capped.

    Families named ``__*`` are operator aggregates and are never selected.
    """
    cap_sum = summary[["capped", "null_capped", "capped_strict", "null_capped_strict"]].sum(axis=1)
    passing = summary[
        ~summary.environment.str.startswith("__")
        & (summary.molecules >= min_molecules)
        & (summary.strict_lo > 0)
        & (cap_sum == 0)
        & (summary.null_missing == 0)
    ]
    return passing.environment.tolist()


def _paired_strict_diff(group: pd.DataFrame, variant: str = "strict") -> np.ndarray:
    real, null = ("strict_unique", "strict_null") if variant == "strict" else ("unique", "null")
    means = group.groupby("mol_id")[[real, null]].mean().dropna()
    return (means[real] - means[null]).to_numpy() * 100


CONFIRMATION_COLUMNS = (
    "environment",
    "molecules",
    "strict_diff_pct",
    "adjusted_lo",
    "adjusted_hi",
    "caps",
    "null_missing",
    "load_formulas_mean",
    "load_new_formulas_mean",
    "confirmed",
    "promoted",
)


def confirm_families(
    val: pd.DataFrame,
    families: Sequence[str],
    *,
    alpha: float = 0.05,
    min_molecules: int = 10,
    reps: int = 20000,
    seed: int = 20260928,
    load_limit: float = 10.0,
) -> pd.DataFrame:
    """Frozen validation of train-selected families, with promotion on the candidate load.

    A family confirms iff it has at least ``min_molecules`` paired molecules, its
    Bonferroni-adjusted strict paired-bootstrap lower limit is above zero, and none of its real
    or null searches was capped or missing. It is promoted iff it confirms and adds at most
    ``load_limit`` distinct (composition, H) formulas per bearing spectrum on average.

    Args:
        val: Validation family records.
        families: Families selected on train (the family count sets the correction).
        alpha: Family-wise level; each limit is ``alpha / (2 K)`` and ``1 - alpha / (2 K)``.
        min_molecules: Minimum paired molecules.
        reps: Bootstrap draws.
        seed: Bootstrap seed.
        load_limit: Largest mean load of a promoted family.

    Returns:
        One row per family.
    """
    rng = np.random.default_rng(seed)
    low, high = alpha / (2 * max(len(families), 1)), 1 - alpha / (2 * max(len(families), 1))
    rows = []
    for family in families:
        group = val[val.environment == family]
        diffs = _paired_strict_diff(group) if len(group) else np.zeros(0)
        if diffs.size >= 3:
            draws = rng.choice(diffs, (reps, diffs.size), replace=True).mean(axis=1)
            lo, hi = np.quantile(draws, [low, high])
        else:
            lo = hi = np.nan
        caps = int(group[list(CAP_COLUMNS)].to_numpy().sum()) if len(group) else 0
        missing = int(group.null_missing.sum()) if len(group) else 0
        confirmed = bool(diffs.size >= min_molecules and lo > 0 and caps == 0 and missing == 0)
        load = float(group.load_formulas.mean()) if len(group) else np.nan
        rows.append(
            {
                "environment": family,
                "molecules": int(diffs.size),
                "strict_diff_pct": float(diffs.mean()) if diffs.size else np.nan,
                "adjusted_lo": lo,
                "adjusted_hi": hi,
                "caps": caps,
                "null_missing": missing,
                "load_formulas_mean": load,
                "load_new_formulas_mean": float(group.load_new_formulas.mean())
                if len(group)
                else np.nan,
                "confirmed": confirmed,
                "promoted": bool(confirmed and load <= load_limit),
            }
        )
    table = pd.DataFrame(rows, columns=CONFIRMATION_COLUMNS)
    return table.astype({"confirmed": bool, "promoted": bool})


def decide_operators(
    val: pd.DataFrame,
    *,
    reps: int = 20000,
    seed: int = 20260928,
    load_limit: float = 10.0,
) -> pd.DataFrame:
    """Model-operator decision for each whole-operator family on validation.

    An operator becomes a model operator iff its whole union beats the random null (paired
    molecule-bootstrap 95% lower limit above zero, strict variant) and its mean load is at most
    ``load_limit`` formulas per bearing spectrum. Caps and missing nulls are reported, not
    silently dropped.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for environment in ("__all__", "__graft__"):
        group = val[val.environment == environment]
        if not len(group):
            continue
        row: dict[str, Any] = {"environment": environment, "spectra": len(group)}
        for variant in VARIANTS:
            diffs = _paired_strict_diff(group, variant)
            if diffs.size >= 3:
                draws = rng.choice(diffs, (reps, diffs.size), replace=True).mean(axis=1)
                lo, hi = np.quantile(draws, [0.025, 0.975])
            else:
                lo = hi = np.nan
            row[f"{variant}_molecules"] = int(diffs.size)
            row[f"{variant}_diff_pct"] = float(diffs.mean()) if diffs.size else np.nan
            row[f"{variant}_lo"], row[f"{variant}_hi"] = lo, hi
        row["capped_spectra_pct"] = float(100 * (group[list(CAP_COLUMNS)].sum(axis=1) > 0).mean())
        row["null_missing_spectra_pct"] = float(100 * (group.null_missing > 0).mean())
        row["load_formulas_mean"] = float(group.load_formulas.mean())
        row["load_new_formulas_mean"] = float(group.load_new_formulas.mean())
        row["model_operator"] = bool(
            row["strict_lo"] > 0 and row["load_formulas_mean"] <= load_limit
        )
        rows.append(row)
    return pd.DataFrame(rows)


# 10. Run bookkeeping --------------------------------------------------------------------


def _alias_ft(value):
    """Rename the historical single label without changing the learned embedding width."""
    if isinstance(value, dict):
        return {("ORB" if k == "FT" else k): _alias_ft(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_alias_ft(v) for v in value]
    return "ORB" if isinstance(value, str) and value == "FT" else value


def _identity(path: str) -> dict:
    item = Path(path)
    stat = item.stat()
    return {"path": str(item.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _md5(path: str | Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - provenance checksum, not security
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance_base(args: argparse.Namespace, started: float) -> dict:
    """Provenance every run records: rules, git state, command line and timestamps."""
    return {
        "args": vars(args),
        "command_line": [sys.executable, *sys.argv],
        "started_unix": started,
        "started_utc": datetime.fromtimestamp(started, UTC).isoformat(),
        "torch": th.__version__,
        "rdkit": Chem.rdBase.rdkitVersion,
        "MASSTER_RULE_NAMES": list(MASSTER_RULE_NAMES),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("debug_masster_smarts_rule_gate.py"))
        },
    }


def _write_provenance(output: Path, provenance: dict) -> None:
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))


# 11. Model-free driver ------------------------------------------------------------------

_WORKER_SCORER: ChannelScorer | None = None


def _worker_scorer() -> ChannelScorer:
    global _WORKER_SCORER  # noqa: PLW0603 - one model per worker process
    if _WORKER_SCORER is None:
        _WORKER_SCORER = ChannelScorer()
    return _WORKER_SCORER


def _init_worker() -> None:
    th.set_num_threads(1)


def process_molecule(task: dict) -> dict:
    """Score all sampled spectra of one molecule (a worker-process entry point).

    Args:
        task: ``mol_id``, ``smiles``, ``config`` (UnitConfig), ``seed``, ``split`` and
            ``spectra`` (dicts with ``spec_id``, ``inst_type``, ``prec_type``, ``peaks``).

    Returns:
        ``records``, ``spectra`` and ``stats`` (with wall-clock seconds) for the molecule.
    """
    started = time.perf_counter()
    mol = Chem.MolFromSmiles(task["smiles"])
    if mol is None:
        raise ValueError(f"unparseable SMILES for molecule {task['mol_id']}")
    rng = np.random.default_rng([task["seed"], int(task["mol_id"])])
    unit = build_unit(mol, _worker_scorer(), task["config"], rng)
    built = time.perf_counter()
    records, spectra = [], []
    for spec in task["spectra"]:
        peaks = np.asarray(spec["peaks"], float).reshape(-1, 2)
        weights = peaks[:, 1] / peaks[:, 1].sum()
        tick = time.perf_counter()
        row, result = evaluate_spectrum_unit(
            unit,
            peaks[:, 0],
            weights,
            TOLERANCE[spec["inst_type"]],
            rng,
            spec["prec_type"],
        )
        identity = {
            "spec_id": spec["spec_id"],
            "mol_id": task["mol_id"],
            "split": task["split"],
            "inst_type": spec["inst_type"],
        }
        spectra.append(
            {
                **identity,
                "prec_type": spec["prec_type"],
                **row,
                "weights": weights.astype(np.float32),
                "seconds": time.perf_counter() - tick,
            }
        )
        records.extend({**identity, **record} for record in result)
    stats = {
        "mol_id": task["mol_id"],
        "spectra": len(task["spectra"]),
        "seconds_build": built - started,
        "seconds_total": time.perf_counter() - started,
        **{k: v for k, v in unit.stats.items() if k != "products"},
        **{f"products_{k}": v for k, v in unit.stats["products"].items()},
    }
    return {"records": records, "spectra": spectra, "stats": stats}


def load_stratum(args: argparse.Namespace) -> pd.DataFrame:
    """Read the model-free spectra: NIST20 rows of the run's adduct and instruments and split.

    Raises:
        ValueError: If the slice holds another adduct's rows or a split has no spectra.
    """
    spec = pd.read_pickle(args.spec_slice)
    membership = pd.read_csv(args.spec_split)[["spec_id", "ma_mi_split_spec"]]
    stratum = spec[
        (spec.dset == "nist20_hr")
        & (spec.prec_type == args.precursor_type)
        & spec.inst_type.isin(args.inst_types)
    ].merge(membership, on="spec_id", how="left", validate="one_to_one")
    stratum = stratum[stratum.ma_mi_split_spec == args.split].reset_index(drop=True)
    if stratum.empty:
        raise ValueError(
            f"no {args.precursor_type} {args.inst_types} spectra in split {args.split}"
        )
    return stratum


def _bearing_molecules(smiles: pd.Series, motifs: Sequence[str]) -> set[int]:
    """Molecules with at least one bridge whose removed atoms are a named motif's X."""
    bearing = set()
    for mol_id, text in smiles.items():
        mol = Chem.MolFromSmiles(text)
        if (
            mol is not None
            and motif_removed_sets(mol, motifs)
            and motif_bridges(mol, enumerate_bridges(mol), motifs)
        ):
            bearing.add(int(mol_id))
    return bearing


def run_model_free(args: argparse.Namespace) -> None:
    """Run the model-free gate on one split (stand-in pool, no checkpoint, CPU only)."""
    motifs = resolve_motifs(args.motifs)
    operators = tuple(dict.fromkeys(part for part in args.operators.split(",") if part))
    if args.select == "bearing" and not motifs:
        raise ValueError("--select bearing needs --motifs other than none")
    if args.split == "val" and args.select == "bearing" and not args.families:
        raise ValueError("validation of a motif run requires a frozen train-selected --families")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    families = None if not args.families else frozenset(json.loads(Path(args.families).read_text()))

    # 1. Stratum, molecule universe and selection, before any target is inspected.
    stratum = load_stratum(args)
    smiles = pd.read_pickle(args.mol_df).set_index("mol_id")["smiles"]
    universe = stratum.groupby("mol_id").size().rename("n_spectra").reset_index()
    universe["bearing"] = np.nan
    ids = np.sort(universe.mol_id.to_numpy())
    rng = np.random.default_rng(args.seed)
    if args.select == "bearing":
        bearing = _bearing_molecules(smiles.loc[ids], motifs)
        universe["bearing"] = universe.mol_id.isin(bearing)
        pool_ids = np.asarray(sorted(bearing))
        count = len(pool_ids) if args.molecules is None else min(args.molecules, len(pool_ids))
    else:
        pool_ids = ids
        count = 512 if args.molecules is None else min(args.molecules, len(pool_ids))
    selected = (
        np.sort(rng.choice(pool_ids, count, replace=False)) if count < len(pool_ids) else pool_ids
    )
    positions_by_mol = stratum.groupby("mol_id").indices
    keep = []
    for mol_id in selected:
        positions = positions_by_mol[mol_id]
        keep.extend(
            rng.choice(positions, min(args.spectra_per_molecule, len(positions)), replace=False)
        )
    keep = sorted(int(k) for k in keep)
    sample = stratum.iloc[keep]
    sample[["spec_id", "mol_id", "inst_type", "prec_type"]].to_csv(
        output.with_suffix(".sample.csv"), index=False
    )
    universe["selected"] = universe.mol_id.isin(set(selected.tolist()))
    universe.to_csv(output.with_suffix(".universe.csv"), index=False)
    census = bool(args.select == "bearing" and count == len(pool_ids))
    # The molecule-level map must agree with the spectrum-level membership it was derived from.
    disagreeing = 0
    if args.split_map:
        molecule_split = pd.read_csv(args.split_map).set_index("mol_id").ma_mi_split
        disagreeing = int((molecule_split.reindex(universe.mol_id).to_numpy() != args.split).sum())
        if disagreeing:
            LOGGER.warning(
                "%d stratum molecules are not %s in %s", disagreeing, args.split, args.split_map
            )

    provenance = _provenance_base(args, started)
    provenance.update(
        {
            "mode": "model_free",
            "motifs": {name: MOTIFS[name].smarts for name in motifs},
            "families_sha256": _sha256(args.families) if args.families else None,
            "inputs_md5": {
                "spec_slice": _md5(args.spec_slice),
                "spec_split": _md5(args.spec_split),
                "mol_df": _md5(args.mol_df),
                **({"split_map": _md5(args.split_map)} if args.split_map else {}),
            },
            "selected_molecules": int(len(selected)),
            "selected_spectra": len(keep),
            "stratum_molecules": int(len(universe)),
            "stratum_spectra": int(len(stratum)),
            "census": census,
            "split_map_disagreeing_molecules": disagreeing,
            "status": "running",
        }
    )
    _write_provenance(output, provenance)
    LOGGER.info(
        "Split %s: %d molecules / %d spectra in stratum; selected %d molecules, %d spectra",
        args.split,
        len(universe),
        len(stratum),
        len(selected),
        len(keep),
    )

    # 2. Score molecules, largest first, in worker processes with per-molecule seeds.
    config = UnitConfig(
        precursor_type=args.precursor_type,
        precursor_index=args.precursor_index,
        operators=operators,
        motifs=motifs,
        families=families,
        pool_cap=args.pool_cap,
        graft_max_distance=args.graft_max_distance,
        graft_whole_max_cuts=args.graft_whole_max_cuts,
        reach_cap=args.reach_cap,
    )
    by_mol = sample.groupby("mol_id")
    tasks = []
    for mol_id, group in by_mol:
        tasks.append(
            {
                "mol_id": int(mol_id),
                "smiles": smiles.loc[mol_id],
                "config": config,
                "seed": args.seed,
                "split": args.split,
                "spectra": [
                    {
                        "spec_id": int(r.spec_id),
                        "inst_type": r.inst_type,
                        "prec_type": r.prec_type,
                        "peaks": r.peaks,
                    }
                    for r in group.itertuples()
                ],
            }
        )
    tasks.sort(key=lambda t: -Chem.MolFromSmiles(t["smiles"]).GetNumAtoms())
    records, spectra, stats = [], [], []
    if args.processes == 1:
        results = map(process_molecule, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            args.processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
        )
        results = executor.map(process_molecule, tasks, chunksize=1)
    try:
        for done, result in enumerate(results, start=1):
            records.extend(result["records"])
            spectra.extend(result["spectra"])
            stats.append(result["stats"])
            if done % 8 == 0 or done == len(tasks):
                LOGGER.info(
                    "Scored %d/%d molecules in %.1fs", done, len(tasks), time.time() - started
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    # 3. Write outputs, summaries and, on train, the frozen family list.
    frame = pd.DataFrame(records).sort_values(
        ["mol_id", "spec_id", "environment"], ignore_index=True
    )
    spectra_frame = pd.DataFrame(spectra).sort_values(["mol_id", "spec_id"], ignore_index=True)
    stats_frame = pd.DataFrame(stats).sort_values("mol_id", ignore_index=True)
    frame.to_pickle(output)
    spectra_frame.to_pickle(output.with_suffix(".spectra.pkl"))
    stats_frame.to_csv(output.with_suffix(".molecules.csv"), index=False)
    summary = summarize_unit(frame, args.seed)
    summary.to_csv(output.with_suffix(".summary.csv"), index=False)
    ceilings = []
    operator_names = [
        e for e in ("__all__", "__graft__", "__union__") if e in set(frame.environment)
    ]
    for environment in operator_names:
        for variant in VARIANTS:
            per = per_spectrum_ceiling(spectra_frame, frame, [environment], variant=variant)
            ceilings.append(
                {
                    "environment": environment,
                    "variant": variant,
                    **stratum_ceiling(
                        per, universe, census=census, reps=args.bootstrap_reps, seed=args.seed
                    ),
                }
            )
    pd.DataFrame(ceilings).to_csv(output.with_suffix(".ceiling.csv"), index=False)
    if args.split == "train":
        chosen = select_families(summary)
        output.with_suffix(".families.json").write_text(json.dumps(chosen, indent=2))
        provenance["families_sha256"] = _sha256(output.with_suffix(".families.json"))
        provenance["families_selected"] = len(chosen)
    provenance["elapsed_seconds"] = time.time() - started
    provenance["finished_utc"] = datetime.now(UTC).isoformat()
    provenance["status"] = "complete"
    _write_provenance(output, provenance)
    LOGGER.info(
        "Summary:\n%s",
        summary.sort_values("strict_diff_pct", ascending=False).head(15).to_string(index=False),
    )
    LOGGER.info("DONE %s", output)


# 12. Checkpoint driver (the 2026-09-28 [M+H]+ gate) -------------------------------------


def run_checkpoint(args: argparse.Namespace) -> None:
    """Run the bounded c0 coverage gate, with optional frozen val families."""
    if args.split == "val" and not args.families:
        raise ValueError("validation requires a frozen train-selected --families file")
    molecules_requested = 256 if args.molecules is None else args.molecules
    th.set_num_threads(2)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002 - seed legacy dependencies as well
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Restore checkpoint semantics, aliasing only its obsolete instrument label.
    saved = get_config_from_ckpt(args.checkpoint)
    prec_types = saved["spec_params"]["prec_types"]
    if saved["spec_params"]["inst_types"] != ["FT"] or prec_types != [args.precursor_type]:
        raise ValueError(
            f"this diagnostic is scoped to c0's historical {args.precursor_type}/FT checkpoint"
        )
    # The YAML only completes keys the checkpoint predates; the checkpoint's own hparams override
    # every key it holds. c0_control.yml was removed with the FT configs (Git 8c52c6d3); i1 is c0
    # plus int_embedder, which the checkpoint holds.
    config = load_config(
        "config", "config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml", []
    )
    config.update(_alias_ft(saved))
    config = apply_inference_config(
        config,
        disable_preproc=True,
        num_workers=0,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )
    for key, expected in (
        ("masster_smarts_templates", False),
        ("masster_include_cmf", False),
        ("masster_max_h_transfer", 4),
        ("masster_h_transfer_bound", "valence"),
        ("masster_unique_fragment_mode", True),
    ):
        if config[key] != expected:
            raise ValueError(f"expected {key}={expected}")
    device = th.device(args.device)
    if device.type == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    model = init_model_from_ckpt(args.checkpoint, config, device)
    precursor_index = list(model.precursor_types).index(args.precursor_type)
    dataset_config = copy.deepcopy(config)
    dataset_config["spec_params"]["inst_types"] = ["ORB", "ORB_F"]
    dataset = get_dataset_cls(config["model_type"])(split=args.split, **dataset_config)
    specs = dataset.spec_df.reset_index(drop=True)
    molecules = pd.read_pickle(config["mol_fp"]).set_index("mol_id")["mol"]
    allowed = None if not args.families else set(json.loads(Path(args.families).read_text()))

    # 2. Random molecule sampling precedes target inspection; retain zero-bridge molecules.
    ids = np.sort(specs.mol_id.unique())
    selected = rng.choice(ids, min(molecules_requested, len(ids)), replace=False)
    keep, bridges_by_mol, templates_by_mol, counts_by_mol = [], {}, {}, {}
    for mol_id in selected:
        positions = np.flatnonzero(specs.mol_id.to_numpy() == mol_id)
        keep.extend(
            rng.choice(positions, min(args.spectra_per_molecule, len(positions)), replace=False)
        )
        mol = molecules.loc[mol_id]
        bridges = enumerate_bridges(mol)
        bridges_by_mol[mol_id] = bridges
        edges = edge_tuple(mol)
        sides = template_side_counts(mol, args.precursor_type)
        templates_by_mol[mol_id] = {s for s in sides if not _is_connected(s, edges)}
        counts_by_mol[mol_id] = (
            None
            if args.legacy_template_valence
            else {s: sides[s] for s in templates_by_mol[mol_id]}
        )
    keep = sorted(keep)
    specs.iloc[keep][["spec_id", "mol_id", "inst_type", "prec_type"]].to_csv(
        output.with_suffix(".sample.csv"), index=False
    )
    provenance = _provenance_base(args, started)
    provenance.update(
        {
            "mode": "checkpoint",
            "device": str(device),
            "gpu": th.cuda.get_device_name(device) if device.type == "cuda" else None,
            "inputs": [
                _identity(p) for p in [args.checkpoint, config["spec_fp"], config["mol_fp"]]
            ],
            "split_files": [
                _identity(str(p))
                for p in sorted(Path(config["split_dp"]).rglob("*"))
                if p.is_file()
            ],
            "config": config,
            "label_alias": "FT -> ORB; both observed ORB/ORB_F input indices set to zero",
        }
    )
    _write_provenance(output, provenance)
    LOGGER.info(
        "Selected %d molecules, %d spectra; %d molecules have eligible bridges",
        len(selected),
        len(keep),
        sum(bool(v) for v in bridges_by_mol.values()),
    )
    loader = DataLoader(
        Subset(dataset, keep),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )
    scorer = ChannelScorer()
    caches, records = {}, []
    cursor = 0

    # 3. Read final retained memberships and chemical channels from annotation outputs.
    with th.inference_mode():
        for batch in loader:
            rows = specs.iloc[keep[cursor : cursor + int(batch["batch_size"])]]
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            batch["spec_inst_type"].zero_()
            out = model(**batch, return_explainability_tensors=True)
            atom_ptr = batch["mol_pyg"].ptr.cpu().numpy()
            atoms = out["fragment_atom_indices"].cpu().numpy()
            memberships = out["fragment_atom_fragment_idxs"].cpu().numpy()
            views = out["ion_h_fragment_batch_idxs"].cpu().numpy()
            masses = out["ion_h_dense_mzs"].double().cpu().numpy()
            valid = out["ion_h_chemical_valid_channels"].cpu().numpy()
            pools = defaultdict(set)
            order = np.argsort(memberships, kind="stable")
            fragment_ids, starts = np.unique(memberships[order], return_index=True)
            for fragment, members in zip(
                fragment_ids, np.split(atoms[order], starts[1:]), strict=True
            ):
                view = views[fragment]
                pools[view].add(tuple(sorted((members - atom_ptr[view]).tolist())))
            for view, row in enumerate(rows.itertuples()):
                mol = molecules.loc[row.mol_id]
                if row.mol_id not in caches:
                    caches[row.mol_id] = MoleculeCoverage(
                        mol, scorer, args.reach_cap, precursor_index
                    )
                # Raw stored intensities: deliberately no new target filtering or sqrt scaling.
                peaks = np.asarray(row.peaks, float).reshape(-1, 2)
                weights = peaks[:, 1] / peaks[:, 1].sum()
                result = evaluate_spectrum(
                    mol,
                    pools[view],
                    masses[views == view][valid[views == view]],
                    bridges_by_mol[row.mol_id],
                    templates_by_mol[row.mol_id],
                    peaks[:, 0],
                    weights,
                    TOLERANCE[row.inst_type],
                    caches[row.mol_id],
                    rng,
                    families=allowed,
                    template_counts=counts_by_mol[row.mol_id],
                    prec_type=row.prec_type,
                )
                for record in result:
                    record.update(
                        spec_id=row.spec_id,
                        mol_id=row.mol_id,
                        split=args.split,
                        inst_type=row.inst_type,
                    )
                records.extend(result)
            cursor += len(rows)
            if cursor % (args.batch_size * 8) == 0 or cursor == len(keep):
                pd.DataFrame(records).to_pickle(output)
                LOGGER.info(
                    "Scored %d/%d spectra in %.1fs", cursor, len(keep), time.time() - started
                )

    # 4. Freeze exploratory train selections before any validation targets are inspected.
    frame = pd.DataFrame(records)
    frame.to_pickle(output)
    summary = summarize(frame, args.seed)
    summary.to_csv(output.with_suffix(".summary.csv"), index=False)
    if args.split == "train":
        passing = summary[
            (summary.environment != "__all__")
            & (summary.molecules >= 10)
            & (summary.strict_lo > 0)
            & (summary.capped == 0)
            & (summary.null_capped == 0)
            & (summary.null_missing == 0)
        ]
        output.with_suffix(".families.json").write_text(
            json.dumps(passing.environment.tolist(), indent=2)
        )
    provenance["elapsed_seconds"] = time.time() - started
    provenance["status"] = "complete"
    _write_provenance(output, provenance)
    LOGGER.info(
        "Summary:\n%s",
        summary.sort_values("strict_diff_pct", ascending=False).head(15).to_string(index=False),
    )


# 13. Confirmation entry point and dispatch ----------------------------------------------


def confirm_main(argv: list[str]) -> None:
    """Confirm train-selected families on validation and decide promotion and operators."""
    parser = argparse.ArgumentParser(prog="confirm", description=confirm_main.__doc__)
    parser.add_argument("--train", required=True, help="train records pickle")
    parser.add_argument("--val", required=True, help="validation records pickle")
    parser.add_argument("--families", required=True, help="train-selected families JSON")
    parser.add_argument("--output", required=True, help="prefix for the confirmation tables")
    parser.add_argument("--reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--min-molecules", type=int, default=10)
    parser.add_argument("--load-limit", type=float, default=10.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_path, val_path = Path(args.train), Path(args.val)
    families = json.loads(Path(args.families).read_text())

    # 1. Verify the frozen chain: train finished, wrote this families file, val used it.
    train_prov = json.loads(train_path.with_suffix(".provenance.json").read_text())
    val_prov = json.loads(val_path.with_suffix(".provenance.json").read_text())
    if train_prov["status"] != "complete" or val_prov["status"] != "complete":
        raise ValueError("train and validation runs must both be complete")
    if train_prov["args"]["split"] != "train" or val_prov["args"]["split"] != "val":
        raise ValueError("--train must be a train run and --val a val run")
    fingerprint = _sha256(args.families)
    if train_prov["families_sha256"] != fingerprint:
        raise ValueError("families file is not the one this train run wrote")
    if val_prov.get("families_sha256") not in (None, fingerprint):
        raise ValueError("the validation run used a different families file")

    # 2. Confirm families, decide operators, and report the validation ceilings.
    val = pd.read_pickle(val_path)
    confirmation = confirm_families(
        val,
        families,
        min_molecules=args.min_molecules,
        reps=args.reps,
        seed=args.seed,
        load_limit=args.load_limit,
    )
    confirmation.to_csv(f"{args.output}.confirmation.csv", index=False)
    operators = decide_operators(val, reps=args.reps, seed=args.seed, load_limit=args.load_limit)
    operators.to_csv(f"{args.output}.operators.csv", index=False)
    spectra = pd.read_pickle(val_path.with_suffix(".spectra.pkl"))
    universe = pd.read_csv(val_path.with_suffix(".universe.csv"))
    census = bool(val_prov["census"])
    promoted = confirmation[confirmation.promoted].environment.tolist()
    # Track B's arm gate needs promoted families and model operators unioned per spectrum on
    # shared molecules; this file configures that run (--families, both operators).
    wildcard = {"__all__": "*bridge", "__graft__": "*graft"}
    model_operators = operators[operators.model_operator].environment.tolist()
    Path(f"{args.output}.trackb_families.json").write_text(
        json.dumps(sorted({*promoted, *(wildcard[e] for e in model_operators)}), indent=2)
    )
    ceilings = []
    groups = {"promoted_families": promoted, **{e: [e] for e in operators.environment}}
    for name, environments in groups.items():
        if not environments:
            continue
        for variant in VARIANTS:
            per = per_spectrum_ceiling(spectra, val, environments, variant=variant)
            ceilings.append(
                {
                    "group": name,
                    "variant": variant,
                    **stratum_ceiling(per, universe, census=census, reps=args.reps, seed=args.seed),
                }
            )
    pd.DataFrame(ceilings).to_csv(f"{args.output}.ceiling.csv", index=False)
    LOGGER.info("Confirmation:\n%s", confirmation.to_string(index=False))
    LOGGER.info("Operators:\n%s", operators.to_string(index=False))
    LOGGER.info("Ceilings:\n%s", pd.DataFrame(ceilings).to_string(index=False))
    LOGGER.info("DONE %s", args.output)


def main(argv: list[str] | None = None) -> None:
    """Run a bounded coverage gate (checkpoint or model-free) or the ``confirm`` step."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "confirm":
        confirm_main(argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("checkpoint", "model_free"), default="checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--families", help="train-selected JSON; required for val motif runs")
    parser.add_argument(
        "--molecules",
        type=int,
        help="molecules to sample (checkpoint 256, random 512, bearing all)",
    )
    parser.add_argument("--spectra-per-molecule", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reach-cap", type=int, default=200000)
    parser.add_argument("--precursor-type", default=None)
    parser.add_argument("--processes", type=int, default=1)
    # Checkpoint mode.
    parser.add_argument("--checkpoint", default="saved_ckpts/c0_control_vcpho5qm.ckpt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--legacy-template-valence",
        action="store_true",
        help="score template sides optimistically, as before 2026-09-29",
    )
    # Model-free mode.
    parser.add_argument(
        "--spec-slice", default="output/smarts_pathway_gates_2026_09_29/mh_spec.pkl"
    )
    parser.add_argument(
        "--spec-split", default="output/smarts_pathway_gates_2026_09_29/spec_split.csv"
    )
    parser.add_argument(
        "--split-map", default="output/smarts_pathway_gates_2026_09_29/split_map.csv"
    )
    parser.add_argument("--mol-df", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--inst-types", default="ORB,ORB_F")
    parser.add_argument("--select", choices=("bearing", "random"), default="random")
    parser.add_argument("--motifs", default="none", help="none, all or comma-separated motif names")
    parser.add_argument("--operators", default="bridge", help="comma-separated: bridge, graft")
    parser.add_argument("--pool-cap", type=int, default=4000)
    parser.add_argument("--graft-max-distance", type=int, default=4)
    parser.add_argument(
        "--graft-whole-max-cuts",
        type=int,
        default=1,
        help="graft baseline: pool sets with at most this many cuts (0 removes it)",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args(argv)
    if min(args.spectra_per_molecule, args.batch_size, args.reach_cap, args.processes) < 1:
        parser.error("sample sizes, reach cap and processes must be positive")
    if args.molecules is not None and args.molecules < 1:
        parser.error("--molecules must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode == "checkpoint":
        args.precursor_type = args.precursor_type or "[M+H]+"
        run_checkpoint(args)
        return
    if args.precursor_type is None:
        parser.error("--precursor-type is required in model_free mode")
    args.inst_types = [part for part in args.inst_types.split(",") if part]
    th.set_num_threads(1)
    # The model's adduct order; check_polarity verifies it against the built model.
    args.precursor_index = {"[M+H]+": 0, "[M-H]-": 1}[args.precursor_type]
    run_model_free(args)


if __name__ == "__main__":
    main()
