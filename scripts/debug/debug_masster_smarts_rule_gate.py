"""Per-class gate for candidate and existing MASSter SMARTS rules on NIST20.

For every NIST20 [M+H]+/[M-H]- spectrum (FT or QTOF) of a molecule a rule matches, in the
rule's admitted polarity, measure the target intensity its template-only (disconnected)
product explains through MASSter's own ion channels: valence bound, +-4 H, no CMF, at the
spectrum's instrument tolerance. Only peaks that no connected atom set of the same
heavy-atom composition can reach (within 4 H) count. Null: random disconnected atom sets of
the same size, scored the same way. Summaries give molecule-bootstrap 95% CIs.

The candidate list records the 2026-09-27 admissions; admitted rules now live in
``masster_templates.py``. Uses the MASSter test factory for the channel function.

Run from the repository root:
    python scripts/debug/debug_masster_smarts_rule_gate.py
"""

import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import RDLogger
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from test_masster_model import _build_model  # noqa: E402

from fragnnet.frag.masster_templates import (  # noqa: E402
    _MASSTER_PRECURSOR_TYPES,
    MASSTER_FRAG_RULES,
    MASSTER_NL_RULES,
    _is_connected,
)
from fragnnet.frag.smarts_prepass import (  # noqa: E402
    NeutralLossRule,
    SmartsFragRule,
    _apply_nl_prepass,
    _apply_smarts_prepass,
)
from fragnnet.utils.feat_utils import get_mol_graph  # noqa: E402

RDLogger.DisableLog("rdApp.*")
BOTH = ("[M+H]+", "[M-H]-")
POS, NEG = ("[M+H]+",), ("[M-H]-",)
SULFONAMIDE = "[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3{}:5]"
CANDIDATES = [
    (NeutralLossRule("primary_aryl_sulfonamide_so2", SULFONAMIDE.format("H2"), [2, 3, 4], 63.9619, 8), BOTH),
    (NeutralLossRule("tertiary_aryl_sulfonamide_so2", SULFONAMIDE.format("H0"), [2, 3, 4], 63.9619, 8), BOTH),
    (NeutralLossRule("primary_aryl_sulfonamide_o_transfer", SULFONAMIDE.format("H2"), [2, 4, 5], None, 8), POS),
    (
        NeutralLossRule(
            "phenylalanine_carboxyl_o_transfer",
            "[c:1]-[CH2:2]-[CH:3](-[NX3;H2:4])-[C:5](=[O:6])[OX2H1:7]",
            [3, 4, 5, 6],
            None,
            10,
        ),
        POS,
    ),
    (
        NeutralLossRule(
            "hydroxamic_acid_nh_excision",
            "[#6]-[CX3:1](~[OX1,OX2H1:2])~[NX2,NX3H1:3]-[OX2H1:4]",
            [3],
            15.0109,
            5,
        ),
        POS,
    ),
    (
        NeutralLossRule(
            "pfas_sulfonate_f_transfer",
            "[F:6]-[CX4:1](-F)-[SX4:2](=[O:3])(=[O:4])-[OX2H1,OX1-:5]",
            [],
            None,
            7,
            product_map_nums=[2, 3, 4, 5, 6],
        ),
        NEG,
    ),
    (
        NeutralLossRule(
            "pfas_sulfinate_f_transfer",
            "[F:6]-[CX4:1](-F)-[SX3:2](=[O:3])-[OX2H1,OX1-:5]",
            [],
            None,
            6,
            product_map_nums=[2, 3, 5, 6],
        ),
        NEG,
    ),
    (
        SmartsFragRule(
            "n_acyl_aryl_sulfonamide_carbonyl_o_transfer",
            "[c:1]-[SX4:2](=O)(=O)-[#7:3]~[CX3:4](~[OX1,OX2H1:5])-[#6:6]",
            [(1, 2), (4, 5)],
            [1, 5],
            min_num_atoms=10,
        ),
        NEG,
    ),
    (
        SmartsFragRule(
            "n_methylanilinoethanol_o_smiles",
            "[c:1]-[NX3;H0;!R:2](-[CX4])-[CH2:3]-[CH2:4]-[OX2H1:5]",
            [(1, 2), (4, 5)],
            [1, 5],
            min_num_atoms=10,
        ),
        NEG,
    ),
    (
        SmartsFragRule(
            "diacylhydrazine_carboxylate_o_transfer",
            "[#6]-[CX3:1](=[OX1])-[NX3:2](-[CX4]([CH3])([CH3])[CH3])-[#7]"
            "~[CX3:3](~[OX1,OX2H1:4])-[#6]",
            [(1, 2), (3, 4)],
            [1, 4],
            min_num_atoms=10,
        ),
        NEG,
    ),
]
EXISTING = [
    (rule, _MASSTER_PRECURSOR_TYPES.get(rule.name, BOTH))
    for rule in (*MASSTER_FRAG_RULES, *MASSTER_NL_RULES)
]
TOLERANCE = {"FT": (1e-5, 200.0), "QTOF": (2e-5, 750.0)}


def edge_tuple(mol) -> tuple[tuple[int, ...], ...]:
    return tuple(zip(*[(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]))


def products(rule, mol, edges) -> list[tuple[int, ...]]:
    """Disconnected template sides of ``rule`` on ``mol``."""
    n = mol.GetNumAtoms()
    if isinstance(rule, NeutralLossRule):
        sides = [tuple(np.flatnonzero(m).tolist()) for m, _ in _apply_nl_prepass([rule], mol, n)]
        sides += [tuple(sorted(set(range(n)) - set(side))) for side in sides]
    else:
        bonds = np.asarray(
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], np.int32
        )
        pairs = _apply_smarts_prepass(
            [rule], mol, np.ones(len(bonds), np.uint8), bonds, n, len(bonds)
        )
        sides = [tuple(np.flatnonzero(m).tolist()) for a, b, _ in pairs for m in (a, b)]
    return sorted({s for s in sides if s and len(s) < n and not _is_connected(s, edges)})


def reachable_h(mol, edges, composition: Counter, size: int, cap: int = 200_000):
    """Own-H counts of connected atom sets with ``composition``, or None past ``cap``."""
    n = mol.GetNumAtoms()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    hydrogens = [a.GetTotalNumHs() for a in mol.GetAtoms()]
    found: set[int] = set()
    checks = 0
    if size <= n - size:
        # 1. Grow connected sets from their smallest atom under the composition bound.
        adjacency = defaultdict(list)
        for left, right in zip(*edges):
            adjacency[left].append(right)
            adjacency[right].append(left)
        seen: set[frozenset[int]] = set()

        def grow(current: set[int], frontier: set[int], counts: Counter) -> None:
            nonlocal checks
            checks += 1
            if checks > cap:
                return
            if len(current) == size:
                found.add(sum(hydrogens[i] for i in current))
                return
            for atom in sorted(frontier):
                symbol = symbols[atom]
                if atom < min(current) or counts[symbol] >= composition.get(symbol, 0):
                    continue
                grown = current | {atom}
                key = frozenset(grown)
                if key in seen:
                    continue
                seen.add(key)
                counts[symbol] += 1
                grow(grown, (frontier | set(adjacency[atom])) - grown, counts)
                counts[symbol] -= 1

        for start in range(n):
            if composition.get(symbols[start], 0):
                grow({start}, set(adjacency[start]), Counter({symbols[start]: 1}))
    else:
        # 2. Enumerate the small removed set and test the remainder's connectivity.
        removed = Counter(symbols) - composition
        pools = {el: [i for i in range(n) if symbols[i] == el] for el in removed}

        def choose(elements: list[str], chosen: set[int]) -> None:
            nonlocal checks
            if checks > cap:
                return
            if not elements:
                checks += 1
                keep = [i for i in range(n) if i not in chosen]
                if _is_connected(keep, edges):
                    found.add(sum(hydrogens[i] for i in keep))
                return
            for picked in combinations(pools[elements[0]], removed[elements[0]]):
                choose(elements[1:], chosen | set(picked))

        choose([el for el in removed if removed[el]], set())
    return None if checks > cap else found


class ChannelScorer:
    """MASSter's ion channels for arbitrary atom sets of one molecule."""

    def __init__(self) -> None:
        self.model = None

    def __call__(self, mol, sets, precursor_index: int) -> tuple[np.ndarray, np.ndarray]:
        graph = Batch.from_data_list(
            [get_mol_graph(mol, ["a_onehot", "a_degree"], ["b_degree"], 0, True)]
        )
        if self.model is None:
            self.model = _build_model(
                graph, include_cmf=False, max_h_transfer=4, h_transfer_bound="valence"
            ).eval()
        edges = edge_tuple(mol)
        atoms = th.tensor([a for s in sets for a in s])
        fragments = th.tensor([i for i, s in enumerate(sets) for _ in s])
        cuts = th.tensor([sum((a in s) != (b in s) for a, b in zip(*edges)) for s in map(set, sets)])
        with th.no_grad():
            mzs, valid = self.model._fragment_masses_and_valid_channels(
                graph,
                atoms,
                fragments,
                th.zeros(len(sets), dtype=th.long),
                len(sets),
                th.tensor([precursor_index]),
                cuts,
            )
        # One ion mode without CMF: (sets, H shifts -4..+4).
        return mzs.double().numpy()[:, 0, :], valid.numpy()[:, 0, :]


def unique_fraction(scorer, mol, sets, reaches, row) -> float:
    """Spectrum intensity the sets explain on peaks no same-composition connected set reaches."""
    if not sets:
        return np.nan
    hydrogens = [a.GetTotalNumHs() for a in mol.GetAtoms()]
    peaks = np.asarray(row.peaks, float).reshape(-1, 2)
    weights = peaks[:, 1] / peaks[:, 1].sum()
    relative, floor = TOLERANCE[row.inst_type]
    mzs, valid = scorer(mol, sets, 0 if row.prec_type == "[M+H]+" else 1)
    hit = np.zeros(len(peaks), bool)
    for i, atoms in enumerate(sets):
        own = sum(hydrogens[a] for a in atoms)
        for k, shift in enumerate(range(-4, 5)):
            if not valid[i, k]:
                continue
            if reaches[i] is None or any(abs(own + shift - h) <= 4 for h in reaches[i]):
                continue
            hit |= np.abs(peaks[:, 0] - mzs[i, k]) <= np.maximum(mzs[i, k], floor) * relative
    return float(weights[hit].sum())


def bootstrap(frame: pd.DataFrame) -> pd.Series:
    by_molecule = frame.groupby("mol_id")[["unique", "null"]].mean().dropna()
    if len(by_molecule) < 3:
        return pd.Series({"diff_pct": np.nan, "lo": np.nan, "hi": np.nan})
    diffs = []
    for seed in range(1000):
        draw = by_molecule.sample(len(by_molecule), replace=True, random_state=seed)
        diffs.append(draw.unique.mean() - draw.null.mean())
    return pd.Series(
        {
            "diff_pct": 100 * (by_molecule.unique.mean() - by_molecule.null.mean()),
            "lo": 100 * np.quantile(diffs, 0.025),
            "hi": 100 * np.quantile(diffs, 0.975),
        }
    )


def summarize(d: pd.DataFrame) -> pd.Series:
    counts = pd.Series(
        {
            "mols": d.mol_id.nunique(),
            "spectra": len(d),
            "unique_pct": 100 * d.unique.mean(),
            "null_pct": 100 * d.null.mean(),
            "ge1_pct": 100 * (d.unique >= 0.01).mean(),
            "null_ge1_pct": 100 * (d.null >= 0.01).mean(),
            "capped": d.capped.mean(),
        }
    )
    return pd.concat([counts, bootstrap(d)])


def main() -> None:
    # 1. Load NIST20 molecules and the eligible spectra.
    mol_df = pd.read_pickle("data/proc/nist20/mol_df.pkl").set_index("mol_id")
    spec = pd.read_pickle("data/proc/nist20/spec_df.pkl")
    spec = spec[spec.prec_type.isin(BOTH) & spec.inst_type.isin(list(TOLERANCE))]
    spec_by_mol = dict(tuple(spec.groupby("mol_id")))
    scorer = ChannelScorer()
    rng = np.random.default_rng(0)
    rows = []

    # 2. Score each rule's products and a random disconnected null on every spectrum.
    for kind, group in (("candidate", CANDIDATES), ("existing", EXISTING)):
        for rule, modes in group:
            for mol_id, mol in mol_df["mol"].items():
                if mol_id not in spec_by_mol:
                    continue
                edges = edge_tuple(mol)
                prods = products(rule, mol, edges)
                if not prods:
                    continue
                symbols = [a.GetSymbol() for a in mol.GetAtoms()]
                reaches = [
                    reachable_h(mol, edges, Counter(symbols[i] for i in p), len(p)) for p in prods
                ]
                nulls = []
                for p in prods:
                    for _ in range(20):
                        draw = sorted(rng.choice(mol.GetNumAtoms(), len(p), replace=False))
                        draw = tuple(int(a) for a in draw)
                        if not _is_connected(draw, edges):
                            nulls.append(draw)
                            break
                null_reaches = [
                    reachable_h(mol, edges, Counter(symbols[i] for i in r), len(r)) for r in nulls
                ]
                for row in spec_by_mol[mol_id].itertuples():
                    if row.prec_type not in modes:
                        continue
                    rows.append(
                        {
                            "kind": kind,
                            "rule": rule.name,
                            "mol_id": mol_id,
                            "mode": row.prec_type,
                            "inst": row.inst_type,
                            "unique": unique_fraction(scorer, mol, prods, reaches, row),
                            "null": unique_fraction(scorer, mol, nulls, null_reaches, row),
                            "capped": any(r is None for r in reaches),
                        }
                    )
            print(rule.name, "done", flush=True)

    # 3. Summarize per rule and polarity.
    frame = pd.DataFrame(rows)
    frame.to_pickle("output/smarts_reach_2026_09_27/rule_gate.pkl")
    summary = frame.groupby(["kind", "rule", "mode"]).apply(summarize)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
