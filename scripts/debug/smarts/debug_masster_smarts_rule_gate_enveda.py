"""Per-class SMARTS gate on Enveda-180 (Bruker timsTOF, QTOF/CID).

The measure is the NIST20 gate's (``debug_masster_smarts_rule_gate.py``, whose helpers this
reuses). For each [M+H]+/[M-H]- spectrum of a matched molecule: the target intensity the rule's
disconnected product explains through MASSter's ion channels (valence bound, +-4 H, no CMF),
counting only peaks that no connected atom set of the same heavy-atom composition reaches
within 4 H. Null: random disconnected atom sets of the same size. Summaries average per
molecule and give molecule-bootstrap 95% CIs.

Differences from the NIST20 run:
- Every SMARTS rule in the code is scored, in both polarities: the MASSter registry, the rules
  disabled on 2026-09-29 (their definitions are copied below), the rules dropped on
  2026-09-27, the shared rules MASSter never used, and the probe's other candidates.
- The census counts every matched molecule; the gate scores up to ``--max_molecules`` of them
  per rule, sampled with a fixed seed.
- Channels are computed once per molecule and polarity, and molecules run in parallel.
- Tolerance is 10 ppm with a 200 m/z floor, the Enveda configs' QTOF loss tolerance.

Run from the repository root:
    python scripts/debug/smarts/debug_masster_smarts_rule_gate_enveda.py --workers 8
"""

import argparse
import sys
import time
import zlib
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_masster_smarts_rule_gate import (  # noqa: E402
    CANDIDATES,
    ChannelScorer,
    edge_tuple,
    products,
    reachable_h,
    summarize,
)

from fragnnet.fragmentation.smarts_prepass import (  # noqa: E402
    FRAG_RULES,
    NL_RULES,
    NeutralLossRule,
    SmartsFragRule,
)
from fragnnet.models.masster.candidates.smarts_templates import (  # noqa: E402
    _MASSTER_PRECURSOR_TYPES,
    MASSTER_FRAG_RULES,
    MASSTER_NL_RULES,
    _is_connected,
)

RDLogger.DisableLog("rdApp.*")
BOTH = ("[M+H]+", "[M-H]-")
RELATIVE_TOLERANCE, MZ_FLOOR = 1e-5, 200.0

# MASSter-only rules disabled on 2026-09-29, copied from the commented-out definitions in
# candidates/smarts_templates.py with their former precursor guards.
DISABLED_MASSTER_ONLY = [
    (
        SmartsFragRule(
            name="pyridyl_carbamate_alkyl_transfer",
            reactant_smarts=(
                "[$(c1ncccc1),$(c1cnccc1),$(c1ccncc1):1]-[OX2:2]-"
                "[CX3:3](=[O:6])-[NX3:4]-"
                "[$([CH3]),$([CH2][CH3]),$([CH2][CH2][CH2][CH3]):5]"
            ),
            cut_bond_map_pairs=[(2, 3), (4, 5)],
            merge_outer_map_nums=[2, 5],
            min_num_atoms=11,
        ),
        ("[M+H]+",),
    ),
    (
        NeutralLossRule(
            name="methyl_benzoate_co2_transfer",
            reactant_smarts="[c:1]-[CX3;!R:2](=[O:3])-[OX2:4]-[CH3:5]",
            nl_map_nums=[2, 3, 4],
            nl_mass=43.9898,
            min_num_atoms=10,
        ),
        ("[M+H]+",),
    ),
    (
        NeutralLossRule(
            name="anisoyl_fluoride_co_f_transfer",
            reactant_smarts=(
                "[$(c1c([OX2][CH3])cccc1),$(c1cc([OX2][CH3])ccc1),"
                "$(c1ccc([OX2][CH3])cc1):1]-[CX3;!R:2](=[O:3])-[F:4]"
            ),
            nl_map_nums=[2, 3],
            nl_mass=27.9949,
            min_num_atoms=10,
        ),
        ("[M]+",),
    ),
    (
        NeutralLossRule(
            name="short_haloalkyl_ammonium_c3_transfer",
            reactant_smarts=(
                "[$([NX3;+0;!$(N-C=O)]),$([NX4;+1]):1]-[CH2:2]-[CH2:3]-[CH2:4]-[Cl,Br,I:5]"
            ),
            nl_map_nums=[2, 3, 4],
            nl_mass=42.0470,
            min_num_atoms=5,
        ),
        ("[M+H]+", "[M]+"),
    ),
    (
        NeutralLossRule(
            name="n_fluoro_sulfonamide_f_transfer",
            reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3H1:5]-[F:6]",
            nl_map_nums=[],
            nl_mass=None,
            min_num_atoms=10,
            product_map_nums=[2, 3, 4, 6],
        ),
        ("[M-H]-",),
    ),
]
DISABLED_SHARED = ("diarylvinyl_c2h2_loss", "diaryl_ketone_co_loss", "aminobenzothiazole_hcn_loss")
DROPPED_2026_09_27 = (
    "anthraquinone_bis_co_loss",
    "xanthone_ring_cleavage",
    "phenoxazine_ring_cleavage",
    "furanocoumarin_ring_cleavage",
    "benzopyranone_ring_cleavage",
    "halogenated_quinoline_ring_cleavage",
    "benzopyranone_bis_ch_loss",
    "anthraquinone_ring_c4_plus_ch_loss",
    "morphinan_amine_arm_loss",
    "desoxymorphinan_ring_cleavage",
    "desoxymorphinan_aromatic_core_release",
    "aryl_sulfonate_so3",
)

# Filled before the worker pool forks, so workers read them without pickling.
RULES: dict[str, tuple[str, object, tuple[str, ...]]] = {}
PATTERNS: dict[str, Chem.Mol] = {}
MOLS: dict[str, Chem.Mol] = {}
SPECTRA: dict[str, list[tuple[str, np.ndarray]]] = {}
_SCORER: ChannelScorer | None = None


def _collect_rules() -> dict[str, tuple[str, object, tuple[str, ...]]]:
    """Every SMARTS rule in the code: name -> (group, rule, admitted precursor types)."""
    rules: dict[str, tuple[str, object, tuple[str, ...]]] = {}
    for rule in (*MASSTER_FRAG_RULES, *MASSTER_NL_RULES):
        rules[rule.name] = ("registered", rule, _MASSTER_PRECURSOR_TYPES.get(rule.name, BOTH))
    shared = {rule.name: rule for rule in (*FRAG_RULES, *NL_RULES)}
    for name in DISABLED_SHARED:
        rules[name] = ("disabled_2026_09_29", shared[name], BOTH)
    for rule, modes in DISABLED_MASSTER_ONLY:
        rules[rule.name] = ("disabled_2026_09_29", rule, modes)
    for name in DROPPED_2026_09_27:
        rules[name] = ("dropped_2026_09_27", shared[name], BOTH)
    for name, rule in shared.items():
        rules.setdefault(name, ("shared_never_masster", rule, BOTH))
    for rule, modes in CANDIDATES:
        rules.setdefault(rule.name, ("candidate_not_admitted", rule, modes))
    return rules


def _census_chunk(mol_ids: list[str]) -> list[tuple[str, str, int]]:
    """(rule, mol_id, disconnected products) for every SMARTS match in the chunk."""
    found = []
    for mol_id in mol_ids:
        mol = MOLS[mol_id]
        edges = None
        for name, pattern in PATTERNS.items():
            if not mol.HasSubstructMatch(pattern):
                continue
            edges = edge_tuple(mol) if edges is None else edges
            found.append((name, mol_id, len(products(RULES[name][1], mol, edges))))
    return found


def _worker_init() -> None:
    global _SCORER
    th.set_num_threads(1)
    _SCORER = ChannelScorer()


def _unique_channel_mzs(mol, sets, reaches, precursor_index: int) -> np.ndarray | None:
    """m/z of the channels that no same-composition connected set reaches (probe semantics)."""
    if not sets:
        return None
    hydrogens = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]
    mzs, valid = _SCORER(mol, sets, precursor_index)
    kept = []
    for i, atoms in enumerate(sets):
        if reaches[i] is None:
            continue
        own = sum(hydrogens[a] for a in atoms)
        for k, shift in enumerate(range(-4, 5)):
            if valid[i, k] and not any(abs(own + shift - h) <= 4 for h in reaches[i]):
                kept.append(mzs[i, k])
    return np.asarray(kept, float)


def _explained(channel_mzs: np.ndarray | None, peaks: np.ndarray) -> float:
    if channel_mzs is None:
        return np.nan
    weights = peaks[:, 1] / peaks[:, 1].sum()
    if channel_mzs.size == 0:
        return 0.0
    tolerance = np.maximum(channel_mzs, MZ_FLOOR) * RELATIVE_TOLERANCE
    hit = (np.abs(peaks[:, :1] - channel_mzs[None, :]) <= tolerance[None, :]).any(axis=1)
    return float(weights[hit].sum())


def _score(task: tuple[str, str]) -> list[dict]:
    """Gate rows for one (rule, molecule): every [M+H]+/[M-H]- spectrum it has."""
    name, mol_id = task
    group, rule, admitted = RULES[name]
    mol = MOLS[mol_id]
    edges = edge_tuple(mol)
    prods = products(rule, mol, edges)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    reaches = [reachable_h(mol, edges, Counter(symbols[i] for i in p), len(p)) for p in prods]
    rng = np.random.default_rng(zlib.crc32(f"{name}|{mol_id}".encode()))
    nulls = []
    for p in prods:
        for _ in range(20):
            draw = tuple(
                int(a) for a in sorted(rng.choice(mol.GetNumAtoms(), len(p), replace=False))
            )
            if not _is_connected(draw, edges):
                nulls.append(draw)
                break
    null_reaches = [reachable_h(mol, edges, Counter(symbols[i] for i in r), len(r)) for r in nulls]
    rows = []
    for precursor_index, mode in enumerate(BOTH):
        spectra = [peaks for prec_type, peaks in SPECTRA[mol_id] if prec_type == mode]
        if not spectra or not prods:
            continue
        unique_mzs = _unique_channel_mzs(mol, prods, reaches, precursor_index)
        null_mzs = _unique_channel_mzs(mol, nulls, null_reaches, precursor_index)
        for peaks in spectra:
            rows.append(
                {
                    "group": group,
                    "rule": name,
                    "mol_id": mol_id,
                    "mode": mode,
                    "admitted": mode in admitted,
                    "unique": _explained(unique_mzs, peaks),
                    "null": _explained(null_mzs, peaks),
                    "capped": any(r is None for r in reaches),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--proc_dp", default="data/proc/enveda_180")
    parser.add_argument("--out_dp", default="output/smarts_gate_enveda_2026_09_29")
    parser.add_argument("--max_molecules", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit_molecules", type=int, default=0, help="pilot: first N molecules")
    args = parser.parse_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # 1. Load molecules and their [M+H]+/[M-H]- QTOF spectra.
    RULES.update(_collect_rules())
    PATTERNS.update(
        {name: Chem.MolFromSmarts(rule.reactant_smarts) for name, (_, rule, _) in RULES.items()}
    )
    mol_df = pd.read_pickle(Path(args.proc_dp) / "mol_df.pkl").set_index("mol_id")
    spec = pd.read_pickle(Path(args.proc_dp) / "spec_df.pkl")
    spec = spec.loc[
        spec.prec_type.isin(BOTH) & (spec.inst_type == "QTOF"), ["mol_id", "prec_type", "peaks"]
    ]
    for mol_id, prec_type, peaks in spec.itertuples(index=False):
        SPECTRA.setdefault(mol_id, []).append((prec_type, np.asarray(peaks, float).reshape(-1, 2)))
    del spec
    mol_ids = [
        mol_id
        for mol_id in mol_df.index
        if mol_id in SPECTRA and mol_df.at[mol_id, "mol"] is not None
    ]
    if args.limit_molecules:
        mol_ids = mol_ids[: args.limit_molecules]
    MOLS.update({mol_id: mol_df.at[mol_id, "mol"] for mol_id in mol_ids})
    del mol_df
    print(
        f"{len(RULES)} rules, {len(mol_ids)} molecules with spectra ({time.time() - start:.0f} s)",
        flush=True,
    )

    context = get_context("fork")
    # 2. Census: every matched molecule and its disconnected-product count.
    chunks = [mol_ids[i : i + 2000] for i in range(0, len(mol_ids), 2000)]
    with context.Pool(args.workers) as pool:
        census = pd.DataFrame(
            [row for chunk in pool.imap_unordered(_census_chunk, chunks) for row in chunk],
            columns=["rule", "mol_id", "products"],
        )
    census.to_pickle(out_dp / "census.pkl")
    print(f"census done ({time.time() - start:.0f} s)", flush=True)

    # 3. Gate: up to max_molecules molecules with a disconnected product per rule.
    tasks = []
    for name, group in census[census.products > 0].groupby("rule"):
        chosen = group.mol_id.to_numpy()
        if len(chosen) > args.max_molecules:
            rng = np.random.default_rng(zlib.crc32(name.encode()))
            chosen = rng.choice(chosen, args.max_molecules, replace=False)
        tasks += [(name, mol_id) for mol_id in chosen]
    rows = []
    with context.Pool(args.workers, initializer=_worker_init) as pool:
        for i, task_rows in enumerate(pool.imap_unordered(_score, tasks, chunksize=4), start=1):
            rows += task_rows
            if i % 1000 == 0:
                print(f"  {i}/{len(tasks)} tasks ({time.time() - start:.0f} s)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_pickle(out_dp / "rule_gate.pkl")

    # 4. Summarize per rule and polarity, next to the census counts.
    matched = census.groupby("rule").agg(
        matched=("mol_id", "nunique"), with_product=("products", lambda p: int((p > 0).sum()))
    )
    summary = frame.groupby(["group", "rule", "mode", "admitted"]).apply(
        summarize, include_groups=False
    )
    summary = summary.reset_index().merge(matched, left_on="rule", right_index=True, how="left")
    summary.to_csv(out_dp / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print(summary.round(3).to_string(), flush=True)
    print(matched.to_string(), flush=True)
    print(f"done ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
