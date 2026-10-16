"""Copy MSGYM retrieval shards with each candidate's SMILES aligned to its graph atom order.

Checkpoints that match rules on SMILES (``masster_smarts_templates``, rewire rules) re-parse
``mol_df.smiles`` and index the dataset graph, built from ``mol_df.mol``, with the result. The
retrieval prepare stage stores the raw candidate SMILES next to the ML-standardized mol, and for
about 1.2% of formula-track candidates (mostly multi-component entries) standardization renumbers
or changes the atoms; rewire expansion then stops with "RDKit and graph atom order disagree".
Training data never has this: its SMILES are already standardized.

For each mismatched molecule this writes ``smiles`` = SMILES of the standardized mol and renumbers
the mol into that SMILES's atom order, so the graph's chemistry is unchanged and re-parsing agrees
with it. Every other row is copied as is. Truth labels (candidates.csv, by InChIKey) are untouched.

Usage: python scripts/ms2c/align_retrieval_shards.py <src_run_dp> <dst_run_dp>
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def _atoms_and_bonds(mol: Chem.Mol) -> tuple[list[int], set[tuple[int, int, float]]]:
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    bonds = {
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
         b.GetBondTypeAsDouble())
        for b in mol.GetBonds()
    }  # fmt: skip
    return atoms, bonds


def align_mol_df(mol_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (aligned copy, number of rows changed); raises if a row cannot be aligned."""
    out = mol_df.copy()
    n_changed = 0
    for i, (smiles, mol) in enumerate(zip(mol_df["smiles"], mol_df["mol"])):
        parsed = Chem.MolFromSmiles(smiles)
        if parsed is not None and _atoms_and_bonds(parsed)[0] == _atoms_and_bonds(mol)[0]:
            continue
        new_smiles = Chem.MolToSmiles(mol)
        order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
        new_mol = Chem.RenumberAtoms(mol, order)
        reparsed = Chem.MolFromSmiles(new_smiles)
        if reparsed is None or _atoms_and_bonds(reparsed) != _atoms_and_bonds(new_mol):
            raise ValueError(f"cannot align mol_id {mol_df['mol_id'].iloc[i]}: {smiles}")
        out.iat[i, out.columns.get_loc("smiles")] = new_smiles
        out.iat[i, out.columns.get_loc("mol")] = new_mol
        n_changed += 1
    return out, n_changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_run_dp", type=Path)
    parser.add_argument("dst_run_dp", type=Path)
    args = parser.parse_args()
    if not (args.src_run_dp / "queries.csv").is_file():
        raise FileNotFoundError(f"{args.src_run_dp} has no queries.csv; run the prepare stage first")
    if (args.dst_run_dp / "queries.csv").exists():
        raise FileExistsError(f"{args.dst_run_dp} already holds aligned shards")
    args.dst_run_dp.mkdir(parents=True, exist_ok=True)
    total = 0
    for src_sd in sorted(args.src_run_dp.glob("shard_*")):
        dst_sd = args.dst_run_dp / src_sd.name
        (dst_sd / "split").mkdir(parents=True, exist_ok=True)
        for name in ("spec_df.pkl", "candidates.csv", "split/test_ids.csv"):
            shutil.copy2(src_sd / name, dst_sd / name)
        mol_df, n_changed = align_mol_df(pd.read_pickle(src_sd / "mol_df.pkl"))
        mol_df.to_pickle(dst_sd / "mol_df.pkl")
        total += n_changed
        print(f"{src_sd.name}: {len(mol_df)} molecules, {n_changed} aligned", flush=True)
    shutil.copy2(args.src_run_dp / "excluded_candidates.csv", args.dst_run_dp / "excluded_candidates.csv")
    shutil.copy2(args.src_run_dp / "queries.csv", args.dst_run_dp / "queries.csv")  # completion marker, last
    print(f"done: {total} molecules aligned -> {args.dst_run_dp}")


if __name__ == "__main__":
    main()
