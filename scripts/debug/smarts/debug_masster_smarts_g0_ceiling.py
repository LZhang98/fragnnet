"""G0 coverage ceiling of the registered MASSter SMARTS templates on a whole split.

Plan: docs/plans/masster/masster_smarts_pathway_plan.md (section 4, gate G0). For every
spectrum of a precursor type (``[M-H]-`` by default) this measures how much the
disconnected sides of the eligible templates could raise the candidate-space cosine ceiling
of the base-peak decomposition, ``oracle = ||T_covered|| / ||T||``.

Unit, per spectrum:

- ``w``: eval-convention target intensities (rel filter 0.0, ``mz < mz_max``, intensity > 0,
  L1-normalised, linear, unbinned, precursor peak kept). ``N = sum(w^2)``.
- ``A0 = sum w^2`` over peaks within the instrument tolerance of a valid ion channel of a
  stand-in pool (connected components of every 0-, 1- and 2-cut mask). A real pool covers
  less, so ``A0`` over-covers and the rise is conservative.
- ``dA = sum w^2`` over peaks newly hit by a disconnected template side and not covered by
  the pool. Only unique coverage counts: no connected atom set of the same composition
  reaches the channel within +-4 H. A capped reach search is not credited.
- ``rise = sqrt((A0 + dA) / N) - sqrt(A0 / N)``; the whole-split value is the mean over ALL
  spectra of a stratum, so spectra without a new hit contribute 0.

Template sides are scored with the production channel call: ``fragment_cut_counts =
len(template.broken_bond_indices)`` and ``fragment_formed_bond_counts = formed_bond_count``
(see ``MassterModel._fragment_masses_and_valid_channels``). A random disconnected null of the
same product sizes is scored identically, for reference only.

Run from the repository root:

    python scripts/debug/smarts/debug_masster_smarts_g0_ceiling.py --workers 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Batch

from fragnnet.models.masster.candidates.smarts_templates import (
    _MASSTER_PRECURSOR_TYPES,
    MASSTER_NUM_RULES,
    MASSTER_RULE_NAMES,
    MassterTemplate,
    _is_connected,
    build_masster_templates,
    filter_masster_templates,
    formed_bond_count,
)
from fragnnet.utils.feat_utils import get_mol_graph

# Support both the script entry point and imports from pytest.
try:
    from .debug_masster_smarts_rule_gate import TOLERANCE, edge_tuple, reachable_h
except ImportError:
    from debug_masster_smarts_rule_gate import TOLERANCE, edge_tuple, reachable_h

# The gate module puts tests/ on sys.path, which provides the model factory.
from test_masster_model import _build_model  # noqa: E402

try:
    from ..masster_memorization.debug_masster_base_peak_tables import clustered_bootstrap
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "masster_memorization"))
    from debug_masster_base_peak_tables import clustered_bootstrap

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger("g0")

OUT_DEFAULT = Path("output/smarts_pathway_gates_2026_09_29")
PROTECTED_DIRS = (
    Path("output/smarts_reach_2026_09_27"),
    Path("output/excision_continuations_2026_09_28"),
    Path("output/postcut_gate_2026_09_29"),
)
PRECURSOR_TYPES = ("[M+H]+", "[M-H]-")
H_SHIFTS = tuple(range(-4, 5))
TWIN_WINDOW = 4
MZ_MAX = 1500.0  # config/template.yml `mz_max`: the eval target drops peaks at or above it.
HEADLINE_THRESHOLD = 0.003
POOL_MAX_CUTS = 2
POOL_CHUNK = 4000


# 1. Unit: eval-convention weights, tolerance coverage and the L2 conversion.


def eval_weights(peaks: object, mz_max: float = MZ_MAX) -> tuple[np.ndarray, np.ndarray]:
    """Return the eval-convention target of one spectrum.

    Mirrors ``preproc_spec(train=False, pred=False)`` at ``target_rel_ints_thresh = 0``:
    peaks with non-positive intensity or ``mz >= mz_max`` are dropped and the remaining
    intensities are L1-normalised. No binning, no intensity transform, precursor kept.

    Args:
        peaks: Sequence of ``(m/z, intensity)`` pairs, shape ``(T, 2)``.
        mz_max: Peaks at or above this m/z are dropped.

    Returns:
        m/z sorted ascending ``(T',)`` and L1-normalised weights ``(T',)``; both empty when
        no peak survives.
    """
    array = np.asarray(peaks, dtype=np.float64).reshape(-1, 2)
    keep = (array[:, 1] > 0.0) & (array[:, 0] < mz_max)
    array = array[keep]
    if array.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    order = np.argsort(array[:, 0], kind="stable")
    mzs = array[order, 0]
    ints = array[order, 1]
    return mzs, ints / ints.sum()


def covered_mask(
    peak_mzs: np.ndarray, channel_mzs: np.ndarray, tolerance: tuple[float, float]
) -> np.ndarray:
    """Return which peaks lie within tolerance of a sorted channel m/z array.

    The tolerance is ``rel * max(peak m/z, floor)``, as in ``map_targets`` of the base-peak
    decomposition.

    Args:
        peak_mzs: Peak m/z, shape ``(T,)``.
        channel_mzs: Ascending channel m/z, shape ``(K,)``.
        tolerance: ``(relative, floor)`` of the instrument.

    Returns:
        Boolean array ``(T,)``.
    """
    if channel_mzs.size == 0 or peak_mzs.size == 0:
        return np.zeros(peak_mzs.size, dtype=bool)
    relative, floor = tolerance
    upper = np.clip(np.searchsorted(channel_mzs, peak_mzs), 0, channel_mzs.size - 1)
    lower = np.maximum(upper - 1, 0)
    nearest = np.minimum(
        np.abs(peak_mzs - channel_mzs[lower]), np.abs(peak_mzs - channel_mzs[upper])
    )
    return nearest <= relative * np.maximum(peak_mzs, floor)


def l2_rise(norm: float, covered_l2: float, new_l2: float) -> float:
    """Return ``sqrt((A0 + dA) / N) - sqrt(A0 / N)`` for one spectrum.

    Args:
        norm: ``N = sum(w^2)`` of the whole target.
        covered_l2: ``A0``, ``sum(w^2)`` on peaks the baseline pool covers.
        new_l2: ``dA``, ``sum(w^2)`` on peaks newly covered by the new products.

    Returns:
        The rise of the coverage ceiling; 0 when ``norm`` is 0.

    Raises:
        ValueError: If the covered mass exceeds the norm.
    """
    if norm <= 0.0:
        return 0.0
    if covered_l2 + new_l2 > norm * (1.0 + 1e-9):
        raise ValueError(f"covered mass {covered_l2 + new_l2} exceeds the norm {norm}")
    return float(np.sqrt((covered_l2 + new_l2) / norm) - np.sqrt(covered_l2 / norm))


# 2. Production-matched ion channels.


class ChannelCalculator:
    """MASSter's own ion channels for arbitrary atom sets of one molecule."""

    def __init__(self, precursor_types: tuple[str, ...] = PRECURSOR_TYPES) -> None:
        self.precursor_types = precursor_types
        self.model = None

    def precursor_index(self, prec_type: str) -> int:
        """Return the model's adduct index for ``prec_type``."""
        if prec_type not in self.precursor_types:
            raise ValueError(f"unsupported precursor type {prec_type!r}")
        return self.precursor_types.index(prec_type)

    def __call__(
        self,
        mol: Chem.Mol,
        atom_sets: list[tuple[int, ...]],
        cut_counts: list[int],
        formed_counts: list[int],
        precursor_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return valid ion channels of atom sets, with the production valence bound.

        Args:
            mol: Precursor molecule.
            atom_sets: Molecule-local atoms of each fragment.
            cut_counts: ``fragment_cut_counts`` per fragment.
            formed_counts: ``fragment_formed_bond_counts`` per fragment.
            precursor_index: Adduct index into ``precursor_types``.

        Returns:
            ``(m/z, valid)``, both ``(len(atom_sets), 9)`` over H shifts -4..+4 (one ion
            mode, no CMF).
        """
        if not atom_sets:
            return np.zeros((0, len(H_SHIFTS))), np.zeros((0, len(H_SHIFTS)), dtype=bool)
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
        atoms = th.tensor([a for s in atom_sets for a in s], dtype=th.long)
        fragments = th.tensor([i for i, s in enumerate(atom_sets) for _ in s], dtype=th.long)
        with th.no_grad():
            mzs, valid = self.model._fragment_masses_and_valid_channels(
                graph,
                atoms,
                fragments,
                th.zeros(len(atom_sets), dtype=th.long),
                len(atom_sets),
                th.tensor([precursor_index]),
                th.tensor(cut_counts, dtype=th.long),
                th.tensor(formed_counts, dtype=th.long),
            )
        return mzs.double().numpy()[:, 0, :], valid.numpy()[:, 0, :]


# 3. Products: template sides and the stand-in pool.


@dataclass
class Side:
    """One disconnected template side with the counts production gives it."""

    atoms: tuple[int, ...]
    cut: int
    formed: int
    rules: set[str] = field(default_factory=set)


def template_sides(
    templates: tuple[MassterTemplate, ...], endpoints: tuple[tuple[int, ...], ...]
) -> tuple[list[Side], int]:
    """Return the deduplicated disconnected sides of templates.

    Production gives every side ``cut = len(template.broken_bond_indices)`` and
    ``formed = formed_bond_count(side)``. Templates that share an atom set and agree on both
    counts merge their rules onto one side.

    Args:
        templates: Eligible templates of one molecule.
        endpoints: Molecule-local bond endpoints as two parallel sequences.

    Returns:
        The sides and the number of atom sets that appear with more than one
        ``(cut, formed)`` pair (production keeps the first producer's counts).
    """
    sides: dict[tuple[tuple[int, ...], int, int], Side] = {}
    for template in templates:
        for component in template.components:
            atoms = tuple(int(a) for a in component)
            if _is_connected(atoms, endpoints):
                continue
            key = (atoms, len(template.broken_bond_indices), formed_bond_count(atoms, endpoints))
            side = sides.setdefault(key, Side(atoms, key[1], key[2]))
            side.rules.add(template.rule_name)
    by_atoms = Counter(key[0] for key in sides)
    conflicts = sum(count > 1 for count in by_atoms.values())
    return list(sides.values()), conflicts


def pool_sets(mol: Chem.Mol, max_cuts: int = POOL_MAX_CUTS) -> dict[tuple[int, ...], int]:
    """Return connected components of every 0- to ``max_cuts``-cut mask with their cut counts.

    Args:
        mol: Precursor molecule.
        max_cuts: Largest number of simultaneously severed bonds.

    Returns:
        Map from sorted atom tuple to the number of severed bonds crossing its boundary. A
        connected set's boundary is fixed by the molecule, so the count does not depend on
        the mask that produced it.
    """
    n_atoms = mol.GetNumAtoms()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    left = np.asarray([b[0] for b in bonds], dtype=np.int64)
    right = np.asarray([b[1] for b in bonds], dtype=np.int64)
    result: dict[tuple[int, ...], int] = {}
    masks: list[tuple[int, ...]] = [()]
    if max_cuts >= 1:
        masks += [(i,) for i in range(len(bonds))]
    if max_cuts >= 2:
        masks += [(i, j) for i in range(len(bonds)) for j in range(i + 1, len(bonds))]
    if max_cuts > 2:
        raise ValueError("the stand-in pool supports at most two cuts")
    for cut in masks:
        keep = np.ones(len(bonds), dtype=bool)
        keep[list(cut)] = False
        graph = coo_matrix(
            (np.ones(int(keep.sum())), (left[keep], right[keep])), shape=(n_atoms, n_atoms)
        )
        n_components, labels = connected_components(graph, directed=False)
        if n_components == 1:
            result.setdefault(tuple(range(n_atoms)), 0)
            continue
        for label in range(n_components):
            atoms = tuple(np.flatnonzero(labels == label).tolist())
            if atoms in result:
                continue
            inside = set(atoms)
            result[atoms] = sum((bonds[i][0] in inside) != (bonds[i][1] in inside) for i in cut)
    return result


def channel_mzs(
    calc: ChannelCalculator,
    mol: Chem.Mol,
    sets: list[tuple[int, ...]],
    cuts: list[int],
    formed: list[int],
    precursor_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Call the channel calculator in bounded chunks."""
    mzs, valid = [], []
    for start in range(0, len(sets), POOL_CHUNK):
        part = slice(start, start + POOL_CHUNK)
        mz, ok = calc(mol, sets[part], cuts[part], formed[part], precursor_index)
        mzs.append(mz)
        valid.append(ok)
    if not sets:
        return calc(mol, [], [], [], precursor_index)
    return np.concatenate(mzs), np.concatenate(valid)


def random_null_sides(
    mol: Chem.Mol, sides: list[Side], rng: np.random.Generator, tries: int = 1000
) -> tuple[list[Side], int]:
    """Draw one distinct random disconnected atom set per product, matched in size.

    Args:
        mol: Precursor molecule.
        sides: The real disconnected sides.
        rng: Random generator.
        tries: Draw attempts per product.

    Returns:
        Null sides (cut counts from their own boundary, formed = pieces - 1) and the number
        of products that found no distinct disconnected set.
    """
    edges = edge_tuple(mol)
    bonds = list(zip(*edges))
    nulls: dict[tuple[int, ...], Side] = {}
    missing = 0
    for side in sorted(sides, key=lambda s: s.atoms):
        for _ in range(tries):
            draw = tuple(
                sorted(
                    int(a) for a in rng.choice(mol.GetNumAtoms(), len(side.atoms), replace=False)
                )
            )
            if draw in nulls or _is_connected(draw, edges):
                continue
            inside = set(draw)
            cut = sum((a in inside) != (b in inside) for a, b in bonds)
            nulls[draw] = Side(draw, cut, formed_bond_count(draw, edges))
            break
        else:
            missing += 1
    return list(nulls.values()), missing


# 4. Scoring products against spectra.


@dataclass
class SpectrumData:
    """One spectrum's eval target and instrument tolerance."""

    spec_id: int
    peak_mzs: np.ndarray
    weights: np.ndarray
    tolerance: tuple[float, float]


@dataclass
class SideScores:
    """Per-spectrum peak masks hit by a set of sides."""

    raw: list[np.ndarray]
    unique: list[np.ndarray]
    cap_credit: list[np.ndarray]
    capped_sides: list[int]
    by_rule: list[dict[str, np.ndarray]]
    capped_search_sides: int
    n_sides: int


def score_sides(
    calc: ChannelCalculator,
    mol: Chem.Mol,
    sides: list[Side],
    spectra: list[SpectrumData],
    pool_covered: list[np.ndarray],
    precursor_index: int,
    reach_cap: int,
) -> SideScores:
    """Find which peaks the sides newly hit, with and without twin exclusion.

    A peak is *new* when a valid channel of a side is within tolerance and the stand-in pool
    does not cover it. It is *unique* when, in addition, the reach search finished and no
    connected atom set of the side's composition has an own-H count within +-4 of the
    channel's. The reach search is asked only about channels that hit a new peak.

    Args:
        calc: Channel calculator.
        mol: Precursor molecule.
        sides: Disconnected sides (real or null).
        spectra: Spectra of the molecule.
        pool_covered: Per spectrum, peaks the pool covers.
        precursor_index: Adduct index.
        reach_cap: Step cap of ``reachable_h``.

    Returns:
        Per-spectrum masks over peaks: ``raw`` (no twin exclusion), ``unique``, and
        ``cap_credit`` (unique, or on a side whose reach search capped), the number of
        capped sides that hit a new peak, and the per-rule unique masks.
    """
    n_spec = len(spectra)
    raw = [np.zeros(s.peak_mzs.size, dtype=bool) for s in spectra]
    unique = [m.copy() for m in raw]
    cap_credit = [m.copy() for m in raw]
    by_rule: list[dict[str, np.ndarray]] = [{} for _ in range(n_spec)]
    capped_sides = [0] * n_spec
    if not sides:
        return SideScores(raw, unique, cap_credit, capped_sides, by_rule, 0, 0)

    # 1. Channels of every side and the peaks each hits outside the pool.
    mzs, valid = channel_mzs(
        calc,
        mol,
        [s.atoms for s in sides],
        [s.cut for s in sides],
        [s.formed for s in sides],
        precursor_index,
    )
    new_hit: list[np.ndarray] = []  # per spectrum: (sides, peaks, shifts)
    for j, spec in enumerate(spectra):
        relative, floor = spec.tolerance
        window = relative * np.maximum(spec.peak_mzs, floor)[None, :, None]
        distance = np.abs(spec.peak_mzs[None, :, None] - mzs[:, None, :])
        hit = (distance <= window) & valid[:, None, :]
        new_hit.append(hit & ~pool_covered[j][None, :, None])

    # 2. One reach search per side, asked only about channels that hit a new peak.
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    hydrogens = [a.GetTotalNumHs() for a in mol.GetAtoms()]
    edges = edge_tuple(mol)
    unique_channel = np.zeros_like(valid)
    capped = np.zeros(len(sides), dtype=bool)
    for i, side in enumerate(sides):
        wanted = np.zeros(len(H_SHIFTS), dtype=bool)
        for hit in new_hit:
            wanted |= hit[i].any(axis=0)
        if not wanted.any():
            continue
        own = sum(hydrogens[a] for a in side.atoms)
        required = frozenset(own + H_SHIFTS[k] for k in np.flatnonzero(wanted))
        reach = reachable_h(
            mol,
            edges,
            Counter(symbols[a] for a in side.atoms),
            len(side.atoms),
            reach_cap,
            required,
        )
        if reach is None:
            capped[i] = True
            continue
        for k in np.flatnonzero(wanted):
            target = own + H_SHIFTS[k]
            unique_channel[i, k] = not any(abs(target - h) <= TWIN_WINDOW for h in reach)

    # 3. Union over sides and channels, per spectrum and per rule.
    for j in range(n_spec):
        hit = new_hit[j]  # (sides, peaks, shifts)
        raw[j] = hit.any(axis=(0, 2))
        unique[j] = (hit & unique_channel[:, None, :]).any(axis=(0, 2))
        cap_credit[j] = unique[j] | (hit & capped[:, None, None]).any(axis=(0, 2))
        capped_sides[j] = int((capped & hit.any(axis=(1, 2))).sum())
        for rule in sorted({r for s in sides for r in s.rules}):
            members = np.asarray([rule in s.rules for s in sides])
            by_rule[j][rule] = (hit[members] & unique_channel[members][:, None, :]).any(axis=(0, 2))
    return SideScores(raw, unique, cap_credit, capped_sides, by_rule, int(capped.sum()), len(sides))


def summed(mask: np.ndarray, values: np.ndarray) -> float:
    """Return ``values[mask].sum()``."""
    return float(values[mask].sum())


# 5. One molecule.


@dataclass
class MoleculeTask:
    """Everything a worker needs for one template-bearing molecule."""

    mol_id: int
    smiles: str
    prec_type: str
    spectra: list[SpectrumData]
    reach_cap: int
    null_seed: int


def process_molecule(task: MoleculeTask, calc: ChannelCalculator) -> dict:
    """Score one molecule's spectra against its template sides and a matched null.

    Args:
        task: The molecule, its spectra and the search settings.
        calc: Channel calculator (one model per process).

    Returns:
        ``{"mol_id", "meta", "spectra"}`` where ``spectra`` holds one row dict per spectrum.

    Raises:
        ValueError: If the SMILES cannot be parsed.
    """
    started = time.time()
    mol = Chem.MolFromSmiles(task.smiles)
    if mol is None:
        raise ValueError(f"molecule {task.mol_id}: cannot parse {task.smiles!r}")
    edges = edge_tuple(mol)
    index = calc.precursor_index(task.prec_type)

    # 1. Eligible templates and their disconnected sides.
    templates = filter_masster_templates(
        build_masster_templates(mol), precursor_type=task.prec_type, fragmentation_mode=None
    )
    sides, conflicts = template_sides(templates, edges)

    # 2. Stand-in pool: connected components of all 0-, 1- and 2-cut masks.
    pool = pool_sets(mol)
    pool_atoms = list(pool)
    pool_cuts = [pool[a] for a in pool_atoms]
    pool_mzs, pool_valid = channel_mzs(
        calc, mol, pool_atoms, pool_cuts, [0] * len(pool_atoms), index
    )
    pool_channels = np.unique(pool_mzs[pool_valid])
    pool_covered = [covered_mask(s.peak_mzs, pool_channels, s.tolerance) for s in task.spectra]

    # 3. Real sides, then the matched random disconnected null.
    real = score_sides(calc, mol, sides, task.spectra, pool_covered, index, task.reach_cap)
    # Sensitivity: the optimistic valence bound of the gate probes (no formed-bond correction).
    optimistic_sides = [Side(s.atoms, s.cut, 0, s.rules) for s in sides]
    optimistic = score_sides(
        calc, mol, optimistic_sides, task.spectra, pool_covered, index, task.reach_cap
    )
    rng = np.random.default_rng(task.null_seed + task.mol_id)
    null_sides, null_missing = random_null_sides(mol, sides, rng)
    null = score_sides(calc, mol, null_sides, task.spectra, pool_covered, index, task.reach_cap)

    rows = []
    for j, spec in enumerate(task.spectra):
        w2 = spec.weights**2
        rows.append(
            {
                "spec_id": spec.spec_id,
                "A0": summed(pool_covered[j], w2),
                "dA_unique": summed(real.unique[j], w2),
                "dA_raw": summed(real.raw[j], w2),
                "dA_cap_upper": summed(real.cap_credit[j], w2),
                "dA_unique_opt": summed(optimistic.unique[j], w2),
                "dL1": summed(real.unique[j], spec.weights),
                "n_capped": real.capped_sides[j],
                "dA_rule": {r: summed(m, w2) for r, m in real.by_rule[j].items()},
                "dL1_rule": {r: summed(m, spec.weights) for r, m in real.by_rule[j].items()},
                "null_dA_unique": summed(null.unique[j], w2),
                "null_dA_raw": summed(null.raw[j], w2),
                "null_dA_cap_upper": summed(null.cap_credit[j], w2),
                "null_dL1": summed(null.unique[j], spec.weights),
                "null_n_capped": null.capped_sides[j],
            }
        )
    meta = {
        "mol_id": task.mol_id,
        "n_atoms": mol.GetNumAtoms(),
        "n_templates": len(templates),
        "rules": sorted({t.rule_name for t in templates}),
        "n_sides": len(sides),
        "side_conflicts": conflicts,
        "n_pool_sets": len(pool_atoms),
        "n_null_sides": len(null_sides),
        "null_missing": null_missing,
        "capped_search_sides": real.capped_search_sides,
        "null_capped_search_sides": null.capped_search_sides,
        "seconds": time.time() - started,
    }
    return {"mol_id": task.mol_id, "meta": meta, "spectra": rows}


_WORKER_CALC: ChannelCalculator | None = None


def _init_worker() -> None:
    """Pin torch to one thread and build the per-process channel calculator."""
    global _WORKER_CALC
    th.set_num_threads(1)
    _WORKER_CALC = ChannelCalculator()


def _run_task(task: MoleculeTask) -> dict:
    if _WORKER_CALC is None:
        raise RuntimeError("worker was not initialised")
    try:
        return process_molecule(task, _WORKER_CALC)
    except Exception as error:  # re-raised in the parent with the molecule id
        raise RuntimeError(f"molecule {task.mol_id} failed: {error!r}") from error


# 6. Summaries.


def whole_split_mean(
    frame: pd.DataFrame, columns: dict[str, np.ndarray], reps: int, seed: int
) -> dict[str, tuple[float, float, float]]:
    """Return mean, and a molecule-clustered 95% CI, of per-spectrum columns.

    The bootstrap resamples molecules and divides the resampled sums by the resampled
    spectrum count, so every spectrum of the stratum is in the denominator.

    Args:
        frame: Rows of one stratum, with ``mol_id``.
        columns: Name -> per-row values.
        reps: Bootstrap replicates.
        seed: RNG seed.

    Returns:
        Name -> ``(mean, lo, hi)``; NaN when the stratum is empty.
    """
    if len(frame) == 0:
        return dict.fromkeys(columns, (np.nan, np.nan, np.nan))
    values = {"__n": np.ones(len(frame)), **columns}
    sums = clustered_bootstrap(frame["mol_id"].to_numpy(), values, reps, seed)
    out = {}
    for k, name in enumerate(values):
        if name == "__n":
            continue
        ratio = sums[:, k] / sums[:, 0]
        out[name] = (
            float(values[name].mean()),
            float(np.quantile(ratio, 0.025)),
            float(np.quantile(ratio, 0.975)),
        )
    return out


def add_units(rows: pd.DataFrame) -> pd.DataFrame:
    """Add rise, pool-free bounds and null rise columns to per-spectrum rows."""
    rows = rows.copy()
    n = rows["N"].to_numpy()
    a0 = rows["A0"].fillna(0.0).to_numpy()

    def rise(delta: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.sqrt((a0 + delta) / n) - np.sqrt(a0 / n)
        return np.where(delta > 0, value, 0.0)

    for tag, column in (
        ("rise", "dA_unique"),
        ("rise_raw", "dA_raw"),
        ("rise_cap_upper", "dA_cap_upper"),
        ("rise_opt", "dA_unique_opt"),
        ("null_rise", "null_dA_unique"),
        ("null_rise_raw", "null_dA_raw"),
    ):
        rows[tag] = rise(rows[column].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        rows["bound_lower"] = np.where(n > 0, rows["dA_unique"] / (2 * n), 0.0)
        rows["bound_upper"] = np.where(n > 0, np.sqrt(rows["dA_unique"] / n), 0.0)
    rows["dL1_pp"] = 100.0 * rows["dL1"]
    rows["null_dL1_pp"] = 100.0 * rows["null_dL1"]
    rows["rise_minus_null"] = rows["rise"] - rows["null_rise"]
    return rows


def strata(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the boolean row mask of every reported stratum.

    The headline is NIST20 (``nist20_hr``) ORB/ORB_F spectra whose molecule has a
    ``nist20_ma_mi_inchikey`` assignment, train, val and test pooled.
    """
    nist = rows["dset"] == "nist20_hr"
    obs = rows["inst"].isin(["ORB", "ORB_F"])
    assigned = rows["split"].isin(["train", "val", "test"])
    headline = nist & obs & assigned
    masks = {"headline_nist20_ob_train_val_test": headline}
    for split in ("train", "val", "test"):
        masks[f"nist20_ob_{split}"] = nist & obs & (rows["split"] == split)
    masks["nist20_ob_unassigned"] = nist & obs & ~assigned
    masks["nist20_qtof_assigned"] = nist & (rows["inst"] == "QTOF") & assigned
    masks["nist20_qtof_unassigned"] = nist & (rows["inst"] == "QTOF") & ~assigned
    mona = rows["dset"] == "mona23"
    masks["mona23_all_tolerance_instruments"] = mona
    for inst in ("ORB", "ORB_F", "QTOF"):
        masks[f"mona23_{inst}"] = mona & (rows["inst"] == inst)
    return masks


def summarize_strata(rows: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Return one summary row per stratum, whole-split and template-bearing subset."""
    out = []
    rise_columns = [
        "rise",
        "rise_raw",
        "rise_cap_upper",
        "rise_opt",
        "bound_lower",
        "bound_upper",
        "dL1_pp",
        "null_rise",
        "null_rise_raw",
        "null_dL1_pp",
        "rise_minus_null",
    ]
    for name, mask in strata(rows).items():
        frame = rows[mask]
        bearing = frame[frame["bearing"]]
        row: dict = {
            "stratum": name,
            "n_spectra": len(frame),
            "n_molecules": frame["mol_id"].nunique(),
            "n_bearing_spectra": len(bearing),
            "n_bearing_molecules": bearing["mol_id"].nunique(),
            "frac_spectra_dA_positive": float((frame["dA_unique"] > 0).mean())
            if len(frame)
            else np.nan,
            "n_spectra_with_capped_side": int((frame["n_capped"] > 0).sum()),
            "frac_bearing_with_capped_side": (
                float((bearing["n_capped"] > 0).mean()) if len(bearing) else np.nan
            ),
        }
        whole = whole_split_mean(frame, {c: frame[c].to_numpy() for c in rise_columns}, reps, seed)
        for column, (mean, lo, hi) in whole.items():
            row[f"{column}_whole"] = mean
            row[f"{column}_whole_lo"] = lo
            row[f"{column}_whole_hi"] = hi
        subset = whole_split_mean(
            bearing, {c: bearing[c].to_numpy() for c in ("rise", "rise_raw", "dL1_pp")}, reps, seed
        )
        for column, (mean, lo, hi) in subset.items():
            row[f"{column}_bearing"] = mean
            row[f"{column}_bearing_lo"] = lo
            row[f"{column}_bearing_hi"] = hi
        out.append(row)
    return pd.DataFrame(out)


def per_rule_table(rows: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Return the rule-alone rise per rule and stratum.

    A rule's rise uses only that rule's disconnected sides (unique coverage), against the
    same pool. Rules overlap on molecules, so the rows do not sum to the union.
    """
    rules = sorted({r for d in rows["dA_rule"] for r in d})
    out = []
    n = rows["N"].to_numpy()
    a0 = rows["A0"].fillna(0.0).to_numpy()
    for name, mask in strata(rows).items():
        frame = rows[mask]
        if len(frame) == 0:
            continue
        for rule in rules:
            has_rule = frame["dA_rule"].map(lambda d, r=rule: r in d).to_numpy()
            delta = frame["dA_rule"].map(lambda d, r=rule: d.get(r, 0.0)).to_numpy()
            l1 = frame["dL1_rule"].map(lambda d, r=rule: d.get(r, 0.0)).to_numpy()
            idx = np.flatnonzero(mask.to_numpy())
            with np.errstate(divide="ignore", invalid="ignore"):
                value = np.sqrt((a0[idx] + delta) / n[idx]) - np.sqrt(a0[idx] / n[idx])
            rise = np.where(delta > 0, value, 0.0)
            whole = whole_split_mean(frame, {"rise": rise, "dL1_pp": 100.0 * l1}, reps, seed)
            sub = frame[has_rule]
            out.append(
                {
                    "stratum": name,
                    "rule": rule,
                    "n_spectra_with_rule": int(has_rule.sum()),
                    "n_molecules_with_rule": int(sub["mol_id"].nunique()),
                    "rise_whole": whole["rise"][0],
                    "rise_whole_lo": whole["rise"][1],
                    "rise_whole_hi": whole["rise"][2],
                    "dL1_pp_whole": whole["dL1_pp"][0],
                    "rise_on_rule_spectra": float(rise[has_rule].mean())
                    if has_rule.any()
                    else np.nan,
                    "dL1_pp_on_rule_spectra": (
                        float(100.0 * l1[has_rule].mean()) if has_rule.any() else np.nan
                    ),
                    "frac_rule_spectra_dA_positive": (
                        float((delta[has_rule] > 0).mean()) if has_rule.any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(out)


def decide(summary: pd.DataFrame) -> dict:
    """Apply the pre-registered G0 decision to the headline stratum.

    Passes iff the headline whole-split mean rise is at least ``HEADLINE_THRESHOLD``.
    """
    head = summary[summary["stratum"] == "headline_nist20_ob_train_val_test"].iloc[0]
    return {
        "headline_rise": float(head["rise_whole"]),
        "headline_ci": [float(head["rise_whole_lo"]), float(head["rise_whole_hi"])],
        "threshold": HEADLINE_THRESHOLD,
        "passes": bool(head["rise_whole"] >= HEADLINE_THRESHOLD),
    }


# 7. Provenance, report and driver.


def md5_file(path: Path) -> str:
    """Return the md5 of a file (content identity, not security)."""
    digest = hashlib.md5()  # noqa: S324
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 26):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(args: argparse.Namespace, inputs: list[Path]) -> dict:
    """Collect the rule registry, git state, input hashes and command line."""
    diff = subprocess.check_output(["git", "diff"])
    return {
        "MASSTER_RULE_NAMES": list(MASSTER_RULE_NAMES),
        "MASSTER_NUM_RULES": MASSTER_NUM_RULES,
        "rule_precursor_types": {
            name: _MASSTER_PRECURSOR_TYPES.get(name) for name in MASSTER_RULE_NAMES
        },
        "git_rev": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "input_md5": {str(path): md5_file(path) for path in inputs},
        "command": " ".join(sys.argv),
        "args": {k: str(v) for k, v in vars(args).items()},
        "started_utc": datetime.now(UTC).isoformat(),
    }


def build_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, list[MoleculeTask], dict]:
    """Load the slice, weight every spectrum and prepare tasks for template-bearing molecules.

    Returns:
        Per-spectrum rows (all spectra of the requested instruments, dA not yet filled), the
        worker tasks, and census facts.
    """
    spec = pd.read_pickle(args.spec)
    mol_df = pd.read_pickle(args.mol_df)[["mol_id", "smiles"]].set_index("mol_id")
    split_map = pd.read_csv(args.split_map)[["mol_id", "ma_mi_split"]].set_index("mol_id")
    census: dict = {"n_slice_rows": len(spec)}
    spec = spec[spec["prec_type"] == args.prec_type]
    census["n_prec_type_rows"] = len(spec)
    census["dropped_no_tolerance_instruments"] = (
        spec.loc[~spec["inst_type"].isin(args.inst_types), "inst_type"].value_counts().to_dict()
    )
    spec = spec[spec["inst_type"].isin(args.inst_types)]
    for inst in args.inst_types:
        if inst not in TOLERANCE:
            raise ValueError(f"no tolerance for instrument {inst!r}")

    # 1. Eval-convention weights per spectrum; empty targets are not evaluated.
    records = []
    data: dict[int, SpectrumData] = {}
    empty = 0
    for r in spec.itertuples():
        mzs, w = eval_weights(r.peaks)
        if w.size == 0:
            empty += 1
            continue
        data[int(r.spec_id)] = SpectrumData(int(r.spec_id), mzs, w, TOLERANCE[r.inst_type])
        records.append(
            {
                "spec_id": int(r.spec_id),
                "mol_id": int(r.mol_id),
                "prec_type": r.prec_type,
                "dset": r.dset,
                "inst": r.inst_type,
                "split": split_map["ma_mi_split"].get(int(r.mol_id), "unassigned"),
                "N": float(np.sum(w**2)),
                "n_peaks": int(w.size),
            }
        )
    census["empty_target_spectra_dropped"] = empty
    rows = pd.DataFrame(records)
    del spec

    # 2. Pre-filter molecules: build templates once per molecule, never loop rules over all.
    started = time.time()
    bearing: dict[int, int] = {}
    for mol_id in rows["mol_id"].unique():
        mol = Chem.MolFromSmiles(mol_df.at[int(mol_id), "smiles"])
        templates = filter_masster_templates(
            build_masster_templates(mol), precursor_type=args.prec_type, fragmentation_mode=None
        )
        if templates:
            bearing[int(mol_id)] = mol.GetNumAtoms()
    census["template_prefilter_seconds"] = time.time() - started
    census["n_molecules_total"] = int(rows["mol_id"].nunique())
    census["n_bearing_molecules_total"] = len(bearing)
    ids = sorted(bearing)
    if args.max_molecules is not None:
        rng = np.random.default_rng(args.seed)
        ids = sorted(
            rng.choice(ids, size=min(args.max_molecules, len(ids)), replace=False).tolist()
        )
    census["n_bearing_molecules_run"] = len(ids)
    rows["bearing"] = rows["mol_id"].isin(bearing)
    rows["processed"] = rows["mol_id"].isin(ids)

    tasks = []
    for mol_id in sorted(ids, key=lambda m: -bearing[m]):
        specs = [data[s] for s in rows.loc[rows["mol_id"] == mol_id, "spec_id"]]
        tasks.append(
            MoleculeTask(
                mol_id,
                mol_df.at[mol_id, "smiles"],
                args.prec_type,
                specs,
                args.reach_cap,
                args.seed,
            )
        )
    return rows, tasks, census


def run_tasks(tasks: list[MoleculeTask], workers: int) -> list[dict]:
    """Run molecule tasks in worker processes (in-process when ``workers`` is 1)."""
    results: list[dict] = []
    started = time.time()
    if workers <= 1:
        _init_worker()
        iterator = map(_run_task, tasks)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(workers, initializer=_init_worker)
        iterator = pool.imap_unordered(_run_task, tasks, chunksize=1)
    try:
        for done, result in enumerate(iterator, 1):
            results.append(result)
            if done % 10 == 0 or done == len(tasks):
                LOGGER.info(
                    "%d/%d molecules, %.0f s elapsed (last mol %d in %.1f s)",
                    done,
                    len(tasks),
                    time.time() - started,
                    result["mol_id"],
                    result["meta"]["seconds"],
                )
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return results


def merge_results(rows: pd.DataFrame, results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach worker outputs to the per-spectrum rows; unprocessed spectra get zeros."""
    spectra = pd.DataFrame([s for r in results for s in r["spectra"]]).set_index("spec_id")
    metas = pd.DataFrame([r["meta"] for r in results])
    rows = rows.join(spectra, on="spec_id")
    zero = [
        "dA_unique",
        "dA_raw",
        "dA_cap_upper",
        "dA_unique_opt",
        "dL1",
        "n_capped",
        "null_dA_unique",
        "null_dA_raw",
        "null_dA_cap_upper",
        "null_dL1",
        "null_n_capped",
    ]
    rows[zero] = rows[zero].fillna(0.0)
    rows["n_capped"] = rows["n_capped"].astype(int)
    rows["null_n_capped"] = rows["null_n_capped"].astype(int)
    for column in ("dA_rule", "dL1_rule"):
        rows[column] = rows[column].map(lambda v: v if isinstance(v, dict) else {})
    return rows, metas


def fmt(value: float, digits: int = 5) -> str:
    """Format a float for the report."""
    return "nan" if pd.isna(value) else f"{value:.{digits}f}"


def write_report(
    path: Path,
    summary: pd.DataFrame,
    per_rule: pd.DataFrame,
    metas: pd.DataFrame,
    decision: dict,
    census: dict,
    prov: dict,
    partial: bool,
    elapsed: float,
) -> None:
    """Write the markdown report of one G0 run."""
    lines = ["# G0 coverage ceiling, registered templates", ""]
    if partial:
        lines += [
            "PARTIAL RUN: only a random subset of template-bearing molecules was processed.",
            "",
        ]
    lines += [
        f"Run {prov['started_utc']}, git {prov['git_rev'][:8]} (diff sha256 "
        f"{prov['git_diff_sha256'][:12]}), {elapsed:.0f} s. Precursor type "
        f"{prov['args']['prec_type']}. Rules read at run time: {len(prov['MASSTER_RULE_NAMES'])}.",
        "",
        "## Decision",
        "",
        f"Headline whole-split mean rise = {fmt(decision['headline_rise'])} "
        f"(95% CI {fmt(decision['headline_ci'][0])} to {fmt(decision['headline_ci'][1])}); "
        f"threshold {decision['threshold']}. **G0 {'PASSES' if decision['passes'] else 'FAILS'}**.",
        "",
        "## Strata",
        "",
        "| stratum | spectra | molecules | bearing spectra | bearing mol | rise whole [95% CI] | "
        "rise on bearing (D2) | L1 pp whole | bound lo | bound hi | null rise | rise raw (no twin excl.) | rise optimistic valence | "
        "rise if capped credited |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.stratum} | {r.n_spectra} | {r.n_molecules} | {r.n_bearing_spectra} | "
            f"{r.n_bearing_molecules} | {fmt(r.rise_whole)} [{fmt(r.rise_whole_lo)}, "
            f"{fmt(r.rise_whole_hi)}] | {fmt(r.rise_bearing)} | {fmt(r.dL1_pp_whole)} | "
            f"{fmt(r.bound_lower_whole)} | {fmt(r.bound_upper_whole)} | {fmt(r.null_rise_whole)} | "
            f"{fmt(r.rise_raw_whole)} | {fmt(r.rise_opt_whole)} | {fmt(r.rise_cap_upper_whole)} |"
        )
    head = "headline_nist20_ob_train_val_test"
    lines += [
        "",
        f"## Per rule, {head}",
        "",
        "| rule | spectra | molecules | rise whole [95% CI] | "
        "rise on rule spectra | L1 pp on rule spectra | share with dA>0 |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in per_rule[per_rule["stratum"] == head].itertuples():
        lines.append(
            f"| {r.rule} | {r.n_spectra_with_rule} | {r.n_molecules_with_rule} | "
            f"{fmt(r.rise_whole)} [{fmt(r.rise_whole_lo)}, {fmt(r.rise_whole_hi)}] | "
            f"{fmt(r.rise_on_rule_spectra)} | {fmt(r.dL1_pp_on_rule_spectra)} | "
            f"{fmt(r.frac_rule_spectra_dA_positive, 3)} |"
        )
    capped = summary.set_index("stratum")
    lines += [
        "",
        "## Caps and census",
        "",
        f"- Reach cap {prov['args']['reach_cap']}. Headline bearing spectra with a capped side "
        f"(a side whose channel hit a new peak but whose twin search hit the cap, not credited): "
        f"{fmt(capped.loc[head, 'frac_bearing_with_capped_side'], 4)}.",
        f"- Molecules processed {len(metas)}; sides {int(metas['n_sides'].sum())}; capped real "
        f"searches {int(metas['capped_search_sides'].sum())}; capped null searches "
        f"{int(metas['null_capped_search_sides'].sum())}; null products with no distinct "
        f"disconnected draw {int(metas['null_missing'].sum())}; atom sets with conflicting "
        f"(cut, formed) {int(metas['side_conflicts'].sum())}.",
        f"- Census: {json.dumps(census, default=str)}",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prec_type", default="[M-H]-", choices=PRECURSOR_TYPES)
    parser.add_argument("--spec", type=Path, default=OUT_DEFAULT / "mh_spec.pkl")
    parser.add_argument("--split_map", type=Path, default=OUT_DEFAULT / "split_map.csv")
    parser.add_argument("--mol_df", type=Path, default=Path("data/proc/nist20/mol_df.pkl"))
    parser.add_argument("--inst_types", nargs="+", default=["ORB", "ORB_F", "QTOF"])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max_molecules", type=int, default=None)
    parser.add_argument("--reach_cap", type=int, default=200_000)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 5:
        raise ValueError("--workers must be between 1 and 5")
    resolved = args.out.resolve()
    for protected in PROTECTED_DIRS:
        if resolved == protected.resolve():
            raise ValueError(f"refusing to write into {protected}")
    return args


def main(argv: list[str] | None = None) -> None:
    """Run G0: score, summarise and write rows, summary and report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    prov = provenance(args, [args.spec, args.split_map, args.mol_df])

    # 1. Rows for every spectrum, tasks for template-bearing molecules.
    rows, tasks, census = build_rows(args)
    LOGGER.info("census: %s", census)

    # 2. Score the template-bearing molecules.
    results = run_tasks(tasks, args.workers)
    rows, metas = merge_results(rows, results)

    # 3. Units, summaries, decision.
    rows = add_units(rows)
    summary = summarize_strata(rows, args.reps, args.seed)
    per_rule = per_rule_table(rows, args.reps, args.seed)
    decision = decide(summary)
    elapsed = time.time() - started
    prov["finished_utc"] = datetime.now(UTC).isoformat()
    prov["seconds"] = elapsed
    prov["census"] = census
    prov["decision"] = decision
    prov["partial"] = args.max_molecules is not None

    rows.to_pickle(args.out / "g0_rows.pkl")
    summary.to_csv(args.out / "g0_summary.csv", index=False)
    per_rule.to_csv(args.out / "g0_per_rule.csv", index=False)
    metas.to_csv(args.out / "g0_molecules.csv", index=False)
    (args.out / "g0_provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    write_report(
        args.out / "g0_report.md",
        summary,
        per_rule,
        metas,
        decision,
        census,
        prov,
        args.max_molecules is not None,
        elapsed,
    )
    LOGGER.info("decision: %s", decision)
    LOGGER.info("DONE")


if __name__ == "__main__":
    main()
