"""MassSpecGym reach of the registered MASSter SMARTS rules.

Applies the 2026-09-27 per-class gate (``debug_masster_smarts_rule_gate.py``) to the
MassSpecGym benchmark: which molecules each [M+H]+-eligible rule matches, by split, and the
target intensity its disconnected products explain on peaks no same-composition connected
atom set reaches within 4 H (``unique``), against a random disconnected null.

Run from the repository root:
    python scripts/debug/smarts/debug_masster_smarts_msgym_reach.py [--workers 10]
"""

import argparse
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_masster_smarts_rule_gate import (  # noqa: E402
    EXISTING,
    ChannelScorer,
    edge_tuple,
    products,
    reachable_h,
    unique_fraction,
)

from fragnnet.models.masster.candidates.smarts_templates import _is_connected  # noqa: E402

DATA = Path("data/proc/ms_gym_benchmark_1_5")
SPLIT = Path("data/split/ms_gym_benchmark_1_5")
OUT = Path("output/msgym_smarts_reach_2026_10_14")
MODE = "[M+H]+"
RULES = [rule for rule, modes in EXISTING if MODE in modes]
_SCORER = None


def _score_molecule(task) -> list[dict]:
    global _SCORER
    if _SCORER is None:
        RDLogger.DisableLog("rdApp.*")
        _SCORER = ChannelScorer()
    mol_id, mol, spectra = task
    rng = np.random.default_rng(int(mol_id))
    edges = edge_tuple(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    rows = []
    for rule in RULES:
        prods = products(rule, mol, edges)
        if not prods:
            continue
        reaches = [reachable_h(mol, edges, Counter(symbols[i] for i in p), len(p)) for p in prods]
        nulls = []
        for p in prods:
            for _ in range(20):
                draw = tuple(int(a) for a in sorted(rng.choice(mol.GetNumAtoms(), len(p), replace=False)))
                if not _is_connected(draw, edges):
                    nulls.append(draw)
                    break
        null_reaches = [
            reachable_h(mol, edges, Counter(symbols[i] for i in r), len(r)) for r in nulls
        ]
        for row in spectra.itertuples():
            rows.append(
                {
                    "rule": rule.name,
                    "mol_id": mol_id,
                    "spec_id": row.spec_id,
                    "split": row.split,
                    "inst": row.inst_type,
                    "unique": unique_fraction(_SCORER, mol, prods, reaches, row),
                    "null": unique_fraction(_SCORER, mol, nulls, null_reaches, row) if nulls else 0.0,
                    "capped": any(r is None for r in reaches),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Spectra with their split; every benchmark spectrum is [M+H]+.
    mol_df = pd.read_pickle(DATA / "mol_df.pkl").set_index("mol_id")
    spec = pd.read_pickle(DATA / "spec_df.pkl")
    split = pd.concat(
        [pd.read_csv(SPLIT / f"{name}_ids.csv").assign(split=name) for name in ("train", "val", "test")]
    )
    spec = spec.merge(split[["spec_id", "split"]], on="spec_id", how="inner")
    spec = spec[spec.prec_type == MODE]
    print(f"{len(spec)} spectra, {spec.mol_id.nunique()} molecules, {len(RULES)} rules", flush=True)

    # 2. Score each molecule in parallel.
    tasks = [(mol_id, mol_df.at[mol_id, "mol"], group) for mol_id, group in spec.groupby("mol_id")]
    rows = []
    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_score_molecule, tasks, chunksize=8)):
            rows.extend(result)
            if i % 2000 == 0:
                print(f"{i}/{len(tasks)} molecules", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_pickle(OUT / "rows.pkl")

    # 3. Molecule and spectrum counts per split.
    totals = spec.groupby("split").agg(spectra=("spec_id", "size"), mols=("mol_id", "nunique"))
    print("\nTotals\n" + totals.to_string())
    if frame.empty:
        return
    per_spec = frame.groupby(["spec_id", "split", "mol_id"])[["unique", "null"]].max().reset_index()
    lines = []
    for name, d in [("all", per_spec), *per_spec.groupby("split")]:
        lines.append(
            {
                "split": name,
                "matched_mols": d.mol_id.nunique(),
                "matched_spectra": len(d),
                "spec_unique_gt0": int((d.unique > 0).sum()),
                "mols_unique_gt0": d.loc[d.unique > 0, "mol_id"].nunique(),
                "spec_ge10pct": int((d.unique >= 0.10).sum()),
                "mols_ge10pct": d.loc[d.unique >= 0.10, "mol_id"].nunique(),
                "spec_ge25pct": int((d.unique >= 0.25).sum()),
                "spec_ge50pct": int((d.unique >= 0.50).sum()),
                "null_ge10pct": int((d.null >= 0.10).sum()),
                "mean_unique_pct_matched": 100 * d.unique.mean(),
            }
        )
    print("\nAny rule (max over rules per spectrum)\n" + pd.DataFrame(lines).round(3).to_string(index=False))
    rule_table = frame.groupby("rule").apply(
        lambda d: pd.Series(
            {
                "mols": d.mol_id.nunique(),
                "test_mols": d.loc[d.split == "test", "mol_id"].nunique(),
                "spectra": d.spec_id.nunique(),
                "unique_pct": 100 * d.unique.mean(),
                "null_pct": 100 * d.null.mean(),
                "spec_ge10pct": int((d.unique >= 0.10).sum()),
                "capped_frac": d.capped.mean(),
            }
        )
    )
    print("\nPer rule\n" + rule_table.sort_values("mols", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
