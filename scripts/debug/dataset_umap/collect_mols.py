"""Collect the unique molecules of each MS/MS dataset for the joint chemical-space UMAP.

Writes one row per (dataset, molecule): dataset, 2D InChIKey block (RDKit, recomputed for every
source), SMILES, spectrum count, and Br/Cl/I flags. Runs in ``FRAGNNET-GPU`` because the
processed ``mol_df.pkl`` files hold RDKit objects; ``plot_umap.py`` runs in ``umap-viz``.

Datasets: NIST20 HR and NIST23 HR (``dset`` prefix ``nist20_hr``/``nist23_hr``), MassSpecGym
benchmark 1.5, Spectraverse, and from CASMI 2026 train MSnLib (``pluskal_ms2``), Enveda-180 and
the natural-product proxy ``enveda-np-examples``.

Usage (from the repository root):
    PYTHONPATH=src python scripts/debug/dataset_umap/collect_mols.py \
        --out_fp output/dataset_umap_2026_10_16/mols.parquet
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROC = {
    "NIST20 HR": ("data/proc/nist20", "nist20_hr"),
    "NIST23 HR": ("data/proc/nist23", "nist23_hr"),
    "MassSpecGym": ("data/proc/ms_gym_benchmark_1_5", "ms_gym_benchmark_1_5"),
    "Spectraverse": ("data/proc/spectraverse", "spectraverse"),
}
CASMI = {"MSnLib": "pluskal_ms2", "Enveda-180": "enveda-180", "CASMI NP proxy": "enveda-np-examples"}


def mol_info(smiles: str) -> tuple:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None, False, False, False
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    return smiles, Chem.MolToInchiKey(mol)[:14] or None, "Br" in symbols, "Cl" in symbols, "I" in symbols


def proc_counts(proc_dp: str, prefix: str) -> pd.DataFrame:
    spec = pd.read_pickle(Path(proc_dp) / "spec_df.pkl")
    spec = spec.loc[spec["dset"].astype(str).str.startswith(prefix), ["mol_id"]]
    smiles = pd.read_pickle(Path(proc_dp) / "mol_df.pkl").set_index("mol_id")["smiles"]
    return spec["mol_id"].map(smiles).value_counts().rename_axis("smiles").reset_index(name="n_spectra")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casmi_train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    ap.add_argument("--out_fp", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    frames = []
    for name, (proc_dp, prefix) in PROC.items():
        frames.append(proc_counts(proc_dp, prefix).assign(dataset=name))
    casmi = pq.read_table(args.casmi_train_fp, columns=["ingest_lib", "normalized_smiles"]).to_pandas()
    for name, lib in CASMI.items():
        counts = casmi.loc[casmi["ingest_lib"] == lib, "normalized_smiles"].value_counts()
        frames.append(counts.rename_axis("smiles").reset_index(name="n_spectra").assign(dataset=name))
    df = pd.concat(frames, ignore_index=True)

    with Pool(args.workers) as pool:
        info = pool.map(mol_info, df["smiles"].unique().tolist(), chunksize=2000)
    info = pd.DataFrame(info, columns=["smiles", "ik14", "Br", "Cl", "I"])
    df = df.merge(info, on="smiles").dropna(subset=["ik14"])
    # One row per (dataset, 2D key): sum spectra over stereo/tautomer SMILES variants.
    df = df.groupby(["dataset", "ik14"], as_index=False).agg(
        smiles=("smiles", "first"), n_spectra=("n_spectra", "sum"), Br=("Br", "max"), Cl=("Cl", "max"), I=("I", "max")
    )
    args.out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_fp, index=False)
    print(df.groupby("dataset").agg(molecules=("ik14", "size"), spectra=("n_spectra", "sum"),
                                    Br=("Br", "sum"), Cl=("Cl", "sum"), I=("I", "sum")).to_string())


if __name__ == "__main__":
    main()
