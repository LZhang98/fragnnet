"""MSnLib vs NIST20/NIST23 HR MS/MS: molecule, scaffold and acquisition coverage.

MSnLib appears twice in our data: as CASMI 2026's ``pluskal_ms2`` library and as Spectraverse
v1.0.2's ``MSnLib`` source (``data/proc/spectraverse/spec_source.csv``). Molecules are compared by
the 2D InChIKey first block, recomputed with RDKit from each source's SMILES so every set uses the
same key (CASMI's own ``inchikey14`` equals RDKit's on a 3,000-SMILES sample). NIST rows are the
HR MS/MS datasets only (``nist20_hr``; ``nist23_hr*``), never the MoNA rows stored beside them.

Writes to ``--out_dp``: ``molecule_sets.parquet`` (one row per InChIKey with set membership and
descriptors), ``overlap_counts.csv``, ``overlap_coverage.csv`` (share of the row set found in the
column set), ``scaffold_coverage.csv``, ``acquisition.json`` and ``summary.json``.

Usage:
    PYTHONPATH=src python scripts/debug/casmi_enveda/msnlib_nist_coverage.py \
        --out_dp output/msnlib_nist_coverage_2026_10_06 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

HALOGENS = ("F", "Cl", "Br", "I")


def describe(smiles: str) -> dict:
    """RDKit 2D InChIKey block, Murcko scaffold, heavy atoms, exact mass and element flags."""
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "ik14": None}
    symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except (ValueError, RuntimeError):
        scaffold = None
    return {
        "smiles": smiles,
        "ik14": Chem.MolToInchiKey(mol)[:14] or None,
        "scaffold": scaffold,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "exact_mass": Descriptors.ExactMolWt(mol),
        "has_halogen": bool(symbols & set(HALOGENS)),
        "has_S": "S" in symbols,
        "has_P": "P" in symbols,
        "charged": Chem.GetFormalCharge(mol) != 0,
    }


def load_nist(proc_dp: Path, dset_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HR MS/MS spectra (dset starting with ``dset_prefix``) and their molecules."""
    spec = pd.read_pickle(proc_dp / "spec_df.pkl")
    keep = spec["dset"].astype(str).str.startswith(dset_prefix)
    logger.info("%s dsets: %s; keeping %s", proc_dp, spec["dset"].value_counts().to_dict(), dset_prefix)
    spec = spec.loc[keep, ["mol_id", "dset", "inst_type", "prec_type", "frag_mode", "ion_mode", "nce", "ace"]]
    mol = pd.read_pickle(proc_dp / "mol_df.pkl")[["mol_id", "smiles"]]
    return spec.merge(mol, on="mol_id", how="left"), mol[mol["mol_id"].isin(spec["mol_id"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--casmi_train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    parser.add_argument("--spv_proc_dp", type=Path, default=Path("data/proc/spectraverse"))
    parser.add_argument("--nist20_proc_dp", type=Path, default=Path("data/proc/nist20"))
    parser.add_argument("--nist23_proc_dp", type=Path, default=Path("data/proc/nist23"))
    parser.add_argument("--out_dp", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dp.mkdir(parents=True, exist_ok=True)

    # 1. Spectra per source, each with a SMILES column.
    casmi = pd.read_parquet(
        args.casmi_train_fp,
        columns=["ingest_lib", "normalized_smiles", "adduct", "ionization_mode", "instrument_type",
                 "collision_energy_orig", "collision_energy_orig_units", "collision_energy_ev"],
    ).rename(columns={"normalized_smiles": "smiles"})
    spv_spec = pd.read_pickle(args.spv_proc_dp / "spec_df.pkl")[
        ["spec_id", "mol_id", "inst_type", "prec_type", "ace", "ion_mode"]
    ]
    source = pd.read_csv(args.spv_proc_dp / "spec_source.csv")
    if len(source) != len(spv_spec) or not source["spec_id"].isin(spv_spec["spec_id"]).all():
        raise ValueError("spec_source.csv does not cover spec_df.pkl")
    spv_spec = spv_spec.merge(source, on="spec_id").merge(
        pd.read_pickle(args.spv_proc_dp / "mol_df.pkl")[["mol_id", "smiles"]], on="mol_id"
    )
    spv_msnlib = spv_spec[spv_spec["source"] == "MSnLib"]
    nist20_spec, _ = load_nist(args.nist20_proc_dp, "nist20_hr")
    nist23_spec, _ = load_nist(args.nist23_proc_dp, "nist23_hr")

    # 2. One RDKit description per distinct SMILES across all sources.
    all_smiles = pd.unique(
        pd.concat([casmi["smiles"], spv_spec["smiles"], nist20_spec["smiles"], nist23_spec["smiles"]])
    )
    with Pool(args.workers) as pool:
        desc = pd.DataFrame(pool.map(describe, all_smiles, chunksize=2000)).set_index("smiles")
    logger.info("described %d SMILES (%d unparsable)", len(desc), desc["ik14"].isna().sum())
    for frame in (casmi, spv_spec, spv_msnlib, nist20_spec, nist23_spec):
        frame["ik14"] = frame["smiles"].map(desc["ik14"])

    sets = {
        "MSnLib (CASMI pluskal_ms2)": casmi.loc[casmi.ingest_lib == "pluskal_ms2", "ik14"],
        "MSnLib (Spectraverse)": spv_msnlib["ik14"],
        "NIST20 HR": nist20_spec["ik14"],
        "NIST23 HR": nist23_spec["ik14"],
        "CASMI train, all": casmi["ik14"],
        "CASMI train, not MSnLib": casmi.loc[casmi.ingest_lib != "pluskal_ms2", "ik14"],
        "CASMI enveda-180": casmi.loc[casmi.ingest_lib == "enveda-180", "ik14"],
        "CASMI enveda-np-examples": casmi.loc[casmi.ingest_lib == "enveda-np-examples", "ik14"],
        "Spectraverse, all": spv_spec["ik14"],
    }
    sets = {name: set(values.dropna()) for name, values in sets.items()}
    sets["NIST20 or NIST23 HR"] = sets["NIST20 HR"] | sets["NIST23 HR"]
    names = list(sets)
    counts = pd.DataFrame([[len(sets[a] & sets[b]) for b in names] for a in names], index=names, columns=names)
    coverage = counts.div(pd.Series({n: len(sets[n]) for n in names}), axis=0)
    counts.to_csv(args.out_dp / "overlap_counts.csv")
    coverage.round(4).to_csv(args.out_dp / "overlap_coverage.csv")

    # 3. Scaffolds (Murcko, from the same descriptions; acyclic molecules give "").
    ik_to_scaffold = desc.dropna(subset=["ik14"]).drop_duplicates("ik14").set_index("ik14")["scaffold"]
    scaf = {name: set(ik_to_scaffold.reindex(list(s)).dropna()) - {""} for name, s in sets.items()}
    scaf_names = ["MSnLib (CASMI pluskal_ms2)", "NIST20 HR", "NIST23 HR", "NIST20 or NIST23 HR",
                  "CASMI enveda-np-examples", "CASMI train, not MSnLib"]
    scaffold_cov = pd.DataFrame(
        [[len(scaf[a] & scaf[b]) / max(len(scaf[a]), 1) for b in scaf_names] for a in scaf_names],
        index=scaf_names, columns=scaf_names,
    )
    scaffold_cov["n_scaffolds"] = [len(scaf[a]) for a in scaf_names]
    scaffold_cov.round(4).to_csv(args.out_dp / "scaffold_coverage.csv")

    # 4. Per-molecule table with membership and descriptors.
    union = sorted(set().union(*sets.values()))
    mol_table = ik_to_scaffold.to_frame().join(
        desc.dropna(subset=["ik14"]).drop_duplicates("ik14").set_index("ik14")[
            ["heavy_atoms", "exact_mass", "has_halogen", "has_S", "has_P", "charged"]
        ]
    ).reindex(union)
    for name in names:
        mol_table[name] = mol_table.index.isin(sets[name])
    mol_table.rename_axis("ik14").reset_index().to_parquet(args.out_dp / "molecule_sets.parquet", index=False)

    # 5. Acquisition conditions per source, and for MSnLib-NIST shared molecules.
    msn = casmi[casmi.ingest_lib == "pluskal_ms2"]
    shared = sets["MSnLib (CASMI pluskal_ms2)"] & sets["NIST20 or NIST23 HR"]
    nist_all = pd.concat([nist20_spec.assign(src="NIST20 HR"), nist23_spec.assign(src="NIST23 HR")])
    msn_pairs = set(zip(msn["ik14"], msn["adduct"]))
    nist_pairs = set(zip(nist_all["ik14"], nist_all["prec_type"]))
    acquisition = {
        "msnlib_casmi": {
            "spectra": len(msn),
            "instrument_type": msn["instrument_type"].fillna("NA").value_counts().head(8).to_dict(),
            "adduct": msn["adduct"].value_counts().head(10).to_dict(),
            "ionization_mode": msn["ionization_mode"].value_counts().to_dict(),
            "ce_units": msn["collision_energy_orig_units"].fillna("NA").value_counts().to_dict(),
            "n_energies": msn["collision_energy_ev"].map(lambda x: 0 if x is None else len(x)).value_counts().to_dict(),
            "ce_orig_top": msn["collision_energy_orig"].astype(str).value_counts().head(10).to_dict(),
            "spectra_per_molecule_median": float(msn.groupby("ik14").size().median()),
        },
        "msnlib_spectraverse": {
            "spectra": len(spv_msnlib),
            "inst_type": spv_msnlib["inst_type"].value_counts().to_dict(),
            "prec_type": spv_msnlib["prec_type"].value_counts().head(10).to_dict(),
        },
    }
    for name, spec in (("NIST20 HR", nist20_spec), ("NIST23 HR", nist23_spec)):
        acquisition[name] = {
            "spectra": len(spec),
            "dset": spec["dset"].value_counts().to_dict(),
            "inst_type": spec["inst_type"].value_counts().to_dict(),
            "frag_mode": spec["frag_mode"].astype(str).value_counts().to_dict(),
            "prec_type": spec["prec_type"].value_counts().head(10).to_dict(),
            "ion_mode": spec["ion_mode"].value_counts().to_dict(),
            "spectra_per_molecule_median": float(spec.groupby("ik14").size().median()),
        }
    acquisition["shared_msnlib_nist"] = {
        "molecules": len(shared),
        "molecule_adduct_pairs_msnlib": len({p for p in msn_pairs if p[0] in shared}),
        "of_which_in_nist_same_adduct_string": len({p for p in msn_pairs if p[0] in shared} & nist_pairs),
        "msnlib_spectra_on_shared_molecules": int(msn["ik14"].isin(shared).sum()),
    }
    (args.out_dp / "acquisition.json").write_text(json.dumps(acquisition, indent=2, default=str))

    # 6. Descriptor contrast: what each side has that the other lacks.
    def profile(mask: pd.Series) -> dict:
        sub = mol_table[mask]
        return {
            "molecules": int(len(sub)),
            "heavy_atoms_median": float(sub["heavy_atoms"].median()),
            "exact_mass_p05_p50_p95": [round(float(q), 1) for q in sub["exact_mass"].quantile([0.05, 0.5, 0.95])],
            "halogen_share": round(float(sub["has_halogen"].mean()), 4),
            "S_share": round(float(sub["has_S"].mean()), 4),
            "P_share": round(float(sub["has_P"].mean()), 4),
            "charged_share": round(float(sub["charged"].mean()), 4),
        }
    m, n = mol_table["MSnLib (CASMI pluskal_ms2)"], mol_table["NIST20 or NIST23 HR"]
    summary = {
        "set_sizes": {name: len(s) for name, s in sets.items()},
        "profiles": {
            "MSnLib only": profile(m & ~n),
            "NIST20/23 only": profile(n & ~m),
            "shared": profile(m & n),
        },
        "unparsable_smiles": int(desc["ik14"].isna().sum()),
    }
    (args.out_dp / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("done: %s", json.dumps(summary["set_sizes"]))


if __name__ == "__main__":
    main()
