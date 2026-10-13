"""Screen three literature EI assemblies against nominal connected-cut/isotope support.

Research-only definitions: no production registry or dataset changes. Census all split
structures, evaluate train/val only, and retain molecule-clustered bootstrap intervals.
The baseline is all 0–2-cut connected components, not the trained model's selected pool
or an exhaustive search over all connected subsets. A positive screen is not admission.
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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger, rdBase

from fragnnet.fragmentation.smarts_prepass import NeutralLossRule, SmartsFragRule
from fragnnet.models.masster.candidates.smarts_templates import _is_connected, formed_bond_count
from fragnnet.models.masster.nominal_isotopes import nominal_isotope_pattern

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.debug.ei.debug_masster_ei_nominal_coverage import (  # noqa: E402
    H_SHIFTS,
    eval_target,
    nominal_atom_masses,
    pool_sets,
)
from scripts.debug.smarts.debug_masster_smarts_g0_ceiling import ChannelCalculator  # noqa: E402
from scripts.debug.smarts.debug_masster_smarts_rule_gate import edge_tuple, products  # noqa: E402

LOGGER = logging.getLogger(__name__)
DATA = Path("data/proc/nist23_ei")
SPLIT = Path("data/split/nist23_ei_inchikey")
RULES = (
    # 70-eV EI, carbonyl 18O tracing: doi:10.1002/rcm.6850.
    # The phenol branch requires N-H; N-alkyl derivatives favor nitrilium instead.
    SmartsFragRule(
        "ei_nh_benzoyl_arylsulfonamide_o_transfer",
        "[c:1]-[SX4:2](=O)(=O)-[NX3H1:3]-[CX3:4](=[OX1:5])-[c:6]",
        [(1, 2), (4, 5)],
        [1, 5],
        min_num_atoms=10,
    ),
    # Existing disabled definition, doi:10.1002/oms.1210261122.
    NeutralLossRule(
        "anisoyl_fluoride_co_f_transfer",
        "[$(c1c([OX2][CH3])cccc1),$(c1cc([OX2][CH3])ccc1),"
        "$(c1ccc([OX2][CH3])cc1):1]-[CX3;!R:2](=[O:3])-[F:4]",
        [2, 3],
        27.9949,
        10,
    ),
    # Moanta 2012, Rev. Chim. 63:7–9, Scheme 5. Narrow the survey's azo family
    # to a para-hydroxyphenyl side; screen substituents, do not assume all react.
    NeutralLossRule(
        "ei_para_hydroxyaryl_azo_n2_loss",
        "[$([cR1]1[cR1][cR1][cR1][cR1][cR1]1):1]-[NX2;!R:2]=[NX2;!R:3]-"
        "[cR1:4]1[cR1][cR1][cR1]([OX2H1])[cR1][cR1]1",
        [2, 3],
        28.0061,
        15,
    ),
)
QUERIES = tuple(Chem.MolFromSmarts(rule.reactant_smarts) for rule in RULES)
CALC = None


def rule_products(mol: Chem.Mol) -> dict[str, list[tuple[int, ...]]]:
    """Return disconnected retained assemblies of neutral, single-component precursors."""
    if Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
        return {}
    edges = edge_tuple(mol)
    result = {}
    for rule, query in zip(RULES, QUERIES, strict=True):
        if mol.HasSubstructMatch(query):
            sides = products(rule, mol, edges)
            if sides:
                result[rule.name] = sides
    return result


def atom_composition(mol: Chem.Mol, atoms: tuple[int, ...], shift: int = 0) -> Counter:
    """Element counts of the retained precursor atoms plus a channel's H transfer."""
    composition = Counter(mol.GetAtomWithIdx(i).GetSymbol() for i in atoms)
    composition["H"] += sum(mol.GetAtomWithIdx(i).GetTotalNumHs() for i in atoms) + shift
    return composition


def isotope_offsets(composition: Counter) -> np.ndarray:
    """Production order-4 support after crop normalization and the 0.001 cutoff."""
    if any(value < 0 for value in composition.values()):
        raise ValueError("negative channel composition")
    pattern = np.asarray(nominal_isotope_pattern(tuple(sorted(composition.items())), 4))
    pattern /= pattern.sum()
    return np.arange(-4, 5)[(pattern > 0) & (pattern >= 0.001)]


def channel_support(
    mol: Chem.Mol, sets: list[tuple[int, ...]], calc: ChannelCalculator
) -> tuple[set[int], set[int]]:
    """Nominal monoisotopic/isotope support with production H and formed-bond bounds."""
    if not sets:
        return set(), set()
    edges = edge_tuple(mol)
    bonds = list(zip(*edges, strict=True))
    cuts = [sum((a in side) != (b in side) for a, b in bonds) for side in map(set, sets)]
    formed = [formed_bond_count(side, edges) for side in sets]
    _, valid = calc(mol, sets, cuts, formed, 0)
    masses = nominal_atom_masses(mol)
    mono, isotopes = set(), set()
    for index, side in enumerate(sets):
        base = int(masses[list(side)].sum())
        for shift in H_SHIFTS[valid[index]]:
            mass = base + int(shift)
            if mass <= 0:
                continue
            mono.add(mass)
            offsets = isotope_offsets(atom_composition(mol, side, int(shift)))
            isotopes.update(int(mass + offset) for offset in offsets if mass + offset > 0)
    return mono, isotopes


def draw_null(
    mol: Chem.Mol, sides: list[tuple[int, ...]], rng: np.random.Generator
) -> tuple[list[tuple[int, ...]], int]:
    """One distinct, size-matched disconnected set per product; exclude all real products."""
    edges = edge_tuple(mol)
    chosen, forbidden = set(), set(sides)
    missing = 0
    for side in sides:
        for _ in range(2000):
            draw = tuple(sorted(rng.choice(mol.GetNumAtoms(), len(side), replace=False).tolist()))
            if draw not in chosen and draw not in forbidden and not _is_connected(draw, edges):
                chosen.add(draw)
                break
        else:
            missing += 1
    return sorted(chosen), missing


def coverage_delta(mz: np.ndarray, weights: np.ndarray, base: set, added: set) -> float:
    """L1-normalized observed intensity newly supported by an assembly's channels."""
    return float(weights[np.isin(mz, list(added - base))].sum())


def init_worker(seed: int) -> None:
    global CALC
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- seed dependencies as well as local Generators
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.*")
    CALC = ChannelCalculator(("[M]+",))


def process_molecule(task: tuple) -> list[dict]:
    mol_id, mol, spectra, seed, n_null = task
    pool = pool_sets(mol)
    base_mono, base_iso = channel_support(mol, list(pool), CALC)
    rows = []
    matched = rule_products(mol)
    for rule_index, rule in enumerate(RULES):
        sides = matched.get(rule.name, [])
        if not sides:
            continue
        mono, iso = channel_support(mol, sides, CALC)
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(mol_id), rule_index]))
        null_support = []
        missing = 0
        for _ in range(n_null):
            null_sides, absent = draw_null(mol, sides, rng)
            missing += absent
            if not absent:
                null_support.append(channel_support(mol, null_sides, CALC)[1])
        for spec_id, split, peaks in spectra:
            mz, weights, _ = eval_target(peaks)
            delta = coverage_delta(mz, weights, base_iso, iso)
            null = (
                np.mean(
                    [coverage_delta(mz, weights, base_iso, support) for support in null_support]
                )
                if len(null_support) == n_null
                else np.nan
            )
            rows.append(
                {
                    "mol_id": mol_id,
                    "spec_id": spec_id,
                    "split": split,
                    "rule": rule.name,
                    "n_products": len(sides),
                    "n_pool": len(pool),
                    "n_null_missing": missing,
                    "base_l1": coverage_delta(mz, weights, set(), base_iso),
                    "delta_mono": coverage_delta(mz, weights, base_mono, mono),
                    "delta": delta,
                    "null": null,
                    "diff": delta - null,
                    "n_rule_masses": len(iso),
                    "n_new_masses": len(iso - base_iso),
                    "nominal_overlap": len(iso & base_iso) / len(iso) if iso else np.nan,
                    "product_atoms": json.dumps(sides),
                    "new_observed_mz": json.dumps(sorted(set(mz.tolist()) & (iso - base_iso))),
                }
            )
    return rows


def summarize(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Molecule-weighted means and paired molecule-bootstrap 95% intervals, 2000 draws."""
    summaries = []
    for (rule, split), group in rows.groupby(["rule", "split"]):
        molecule = group.groupby("mol_id")[["delta_mono", "delta", "null", "diff"]].mean()
        item = {"rule": rule, "split": split, "molecules": len(molecule), "spectra": len(group)}
        item.update(molecule.mean().to_dict())
        item["molecules_ge1pct"] = int((molecule.delta >= 0.01).sum())
        item["products_max"] = int(group.n_products.max())
        item["null_missing"] = int(group.n_null_missing.sum())
        item["nominal_overlap"] = float(group.groupby("mol_id").nominal_overlap.mean().mean())
        if len(molecule) >= 3 and molecule.notna().all().all():
            rng = np.random.default_rng(seed)
            values = molecule["diff"].to_numpy()
            means = np.mean(values[rng.integers(len(values), size=(2000, len(values)))], axis=1)
            item["lo"], item["hi"] = np.quantile(means, [0.025, 0.975])
        else:
            item["lo"], item["hi"] = np.nan, np.nan
        summaries.append(item)
    return pd.DataFrame(summaries)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--null_reps", type=int, default=32)
    parser.add_argument("--out_dp", type=Path, default=Path("output/masster_ei_smarts_2026_10_01"))
    args = parser.parse_args()
    if args.workers < 1 or args.null_reps < 1:
        parser.error("workers and null_reps must be positive")
    args.out_dp.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    RDLogger.DisableLog("rdApp.*")

    # 1. Record source/input identity before evaluation. Never score test spectra.
    paths = [DATA / "mol_df.pkl", DATA / "spec_df.pkl"]
    paths += [SPLIT / f"{split}_ids.csv" for split in ("train", "val", "test")]
    source_paths = [
        Path(__file__),
        Path("scripts/debug/smarts/debug_masster_smarts_g0_ceiling.py"),
        Path("src/fragnnet/models/masster/masster_model.py"),
        Path("src/fragnnet/utils/formula_utils.py"),
    ]
    provenance = {
        "started_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "seed": args.seed,
        "null_reps": args.null_reps,
        "workers": args.workers,
        "python": sys.executable,
        "rdkit": rdBase.rdkitVersion,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs_sha256": {str(path): sha256(path) for path in paths},
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "rules": [{"name": rule.name, "smarts": rule.reactant_smarts} for rule in RULES],
    }
    (args.out_dp / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.out_dp / "starting.diff").write_bytes(subprocess.check_output(["git", "diff"]))
    mol_df = pd.read_pickle(DATA / "mol_df.pkl").set_index("mol_id")
    spec_df = pd.read_pickle(DATA / "spec_df.pkl")
    split_ids = {
        split: set(pd.read_csv(SPLIT / f"{split}_ids.csv").spec_id)
        for split in ("train", "val", "test")
    }
    if any(
        split_ids[a] & split_ids[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("overlapping spectrum splits")
    labels = {identifier: split for split, ids in split_ids.items() for identifier in ids}
    spec_df["split"] = spec_df.spec_id.map(labels)
    relevant = spec_df[spec_df.split.notna()]
    if not relevant.prec_type.eq("[M]+").all():
        raise ValueError("screen requires [M]+ EI data")
    if relevant.groupby("mol_id").split.nunique().max() != 1:
        raise ValueError("molecules cross splits")

    # 2. Full structural census, including the test structures but not their peak values.
    census = []
    for mol_id in sorted(relevant.mol_id.unique()):
        mol = mol_df.loc[mol_id, "mol"]
        for rule, sides in rule_products(mol).items():
            census.append(
                {
                    "mol_id": mol_id,
                    "rule": rule,
                    "n_products": len(sides),
                    "smiles": Chem.MolToSmiles(mol),
                }
            )
    census_df = pd.DataFrame(census, columns=["mol_id", "rule", "n_products", "smiles"])
    census_df = census_df.merge(relevant[["mol_id", "split"]].drop_duplicates(), on="mol_id")
    census_df.to_csv(args.out_dp / "census.csv", index=False)
    LOGGER.info("Census:\n%s", census_df.groupby(["rule", "split"]).size().to_string())
    selected = relevant[
        relevant.mol_id.isin(census_df.mol_id) & relevant.split.isin(["train", "val"])
    ]
    tasks = [
        (
            int(mol_id),
            mol_df.loc[mol_id, "mol"],
            list(group[["spec_id", "split", "peaks"]].itertuples(index=False, name=None)),
            args.seed,
            args.null_reps,
        )
        for mol_id, group in selected.groupby("mol_id")
    ]

    # 3. Evaluate every matching train/val spectrum with per-molecule seeded controls.
    output = []
    with mp.Pool(args.workers, initializer=init_worker, initargs=(args.seed,)) as pool:
        for index, result in enumerate(pool.imap(process_molecule, tasks), 1):
            output.extend(result)
            if index % 25 == 0:
                LOGGER.info("Scored %d/%d molecules", index, len(tasks))
    rows = pd.DataFrame(output)
    rows.to_csv(args.out_dp / "rows.csv", index=False)
    summary = summarize(rows, args.seed) if len(rows) else pd.DataFrame()
    summary.to_csv(args.out_dp / "summary.csv", index=False)
    provenance["finished_utc"] = datetime.now(UTC).isoformat()
    provenance["split_molecules"] = relevant.groupby("split").mol_id.nunique().to_dict()
    provenance["split_spectra"] = relevant.groupby("split").size().to_dict()
    (args.out_dp / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    LOGGER.info("Screen complete:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
