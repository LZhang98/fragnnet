"""Collect the unique molecules of each MS/MS dataset for the joint chemical-space UMAP.

Writes one row per (dataset, molecule): dataset, 2D InChIKey block (RDKit, recomputed for every
source), SMILES, spectrum count, and Br/Cl/I flags. Runs in ``FRAGNNET-GPU`` because the
processed ``mol_df.pkl`` files hold RDKit objects; the later steps run in ``umap-viz`` (see
``common.py``).

Datasets: NIST20 HR and NIST23 HR (``dset`` prefix ``nist20_hr``/``nist23_hr``), MassSpecGym
benchmark 1.5, Spectraverse, and from CASMI 2026 train MSnLib (``pluskal_ms2``), Enveda-180 and
the natural-product proxy ``enveda-np-examples``. Optionally HMDB (``--hmdb_fp``: the PubChem
property CSV of HMDB-deposited CIDs, columns CID/SMILES/InChIKey; hmdb.ca blocks scripted
downloads, and the PubChem copy has no detected/expected status). HMDB has no spectra. Other
structure-only sets via ``--extra "NAME=PATH:SMILES_COLUMN"`` (``.tsv`` read tab-separated), e.g.
``--extra "NP Atlas=data/raw/npatlas/NPAtlas_download.tsv:compound_smiles"``.

``--spectraverse_sources_fp`` also writes each Spectraverse molecule's library sources (mol_id,
smiles, source from ``spec_source.csv``), which ``combo_coverage.py`` reads to drop MS-DIAL.

Usage (from the repository root):
    PYTHONPATH=src python scripts/debug/dataset_umap/collect_mols.py \
        --out_fp output/dataset_umap_2026_10_16/mols.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import mol_info_frame  # noqa: E402

PROC = {
    "NIST20 HR": ("data/proc/nist20", "nist20_hr"),
    "NIST23 HR": ("data/proc/nist23", "nist23_hr"),
    "MassSpecGym": ("data/proc/ms_gym_benchmark_1_5", "ms_gym_benchmark_1_5"),
    "Spectraverse": ("data/proc/spectraverse", "spectraverse"),
}
CASMI = {"MSnLib": "pluskal_ms2", "Enveda-180": "enveda-180", "CASMI NP proxy": "enveda-np-examples"}


def proc_counts(proc_dp: str, prefix: str) -> pd.DataFrame:
    spec = pd.read_pickle(Path(proc_dp) / "spec_df.pkl")
    spec = spec.loc[spec["dset"].astype(str).str.startswith(prefix), ["mol_id"]]
    smiles = pd.read_pickle(Path(proc_dp) / "mol_df.pkl").set_index("mol_id")["smiles"]
    return spec["mol_id"].map(smiles).value_counts().rename_axis("smiles").reset_index(name="n_spectra")


def spectraverse_sources(proc_dp: Path) -> pd.DataFrame:
    """One row per (Spectraverse molecule, library source)."""
    spec = pd.read_pickle(proc_dp / "spec_df.pkl")[["spec_id", "mol_id"]]
    smiles = pd.read_pickle(proc_dp / "mol_df.pkl").set_index("mol_id")["smiles"]
    src = spec.merge(pd.read_csv(proc_dp / "spec_source.csv"), on="spec_id")[["mol_id", "source"]]
    src = src.drop_duplicates()
    src["smiles"] = src["mol_id"].map(smiles)
    return src[["mol_id", "smiles", "source"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casmi_train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    ap.add_argument("--out_fp", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hmdb_fp", type=Path, default=None)
    ap.add_argument("--extra", action="append", default=[], help='"NAME=PATH:SMILES_COLUMN", repeatable')
    ap.add_argument("--spectraverse_sources_fp", type=Path, default=None)
    args = ap.parse_args()

    frames = []
    for name, (proc_dp, prefix) in PROC.items():
        frames.append(proc_counts(proc_dp, prefix).assign(dataset=name))
    casmi = pq.read_table(args.casmi_train_fp, columns=["ingest_lib", "normalized_smiles"]).to_pandas()
    for name, lib in CASMI.items():
        counts = casmi.loc[casmi["ingest_lib"] == lib, "normalized_smiles"].value_counts()
        frames.append(counts.rename_axis("smiles").reset_index(name="n_spectra").assign(dataset=name))
    if args.hmdb_fp is not None:
        hmdb = pd.read_csv(args.hmdb_fp, usecols=["SMILES"]).dropna().drop_duplicates()
        frames.append(pd.DataFrame({"smiles": hmdb["SMILES"], "n_spectra": 0, "dataset": "HMDB"}))
    for spec in args.extra:
        name, rest = spec.split("=", 1)
        path, col = rest.rsplit(":", 1)
        extra = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",", usecols=[col])[col]
        extra = extra.dropna().drop_duplicates()
        frames.append(pd.DataFrame({"smiles": extra, "n_spectra": 0, "dataset": name}))
    df = pd.concat(frames, ignore_index=True)

    info = mol_info_frame(df["smiles"].tolist(), args.workers)
    df = df.merge(info, on="smiles").dropna(subset=["ik14"])
    # One row per (dataset, 2D key): sum spectra over stereo/tautomer SMILES variants.
    df = df.groupby(["dataset", "ik14"], as_index=False).agg(
        smiles=("smiles", "first"), n_spectra=("n_spectra", "sum"), Br=("Br", "max"), Cl=("Cl", "max"), I=("I", "max")
    )
    args.out_fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_fp, index=False)
    print(df.groupby("dataset").agg(molecules=("ik14", "size"), spectra=("n_spectra", "sum"),
                                    Br=("Br", "sum"), Cl=("Cl", "sum"), I=("I", "sum")).to_string())
    if args.spectraverse_sources_fp is not None:
        spectraverse_sources(Path(PROC["Spectraverse"][0])).to_parquet(args.spectraverse_sources_fp)


if __name__ == "__main__":
    main()
