"""Audit downloaded HR-EI peak tables and supplied structures without modifying raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors

NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def read_records(path: Path) -> list[tuple[dict, list]]:
    """Read explicit MSP pairs or uncompressed JCAMP XY peak tables, preserving metadata."""
    records = []
    metadata, peaks = {}, []
    in_peaks = False
    for line in path.read_text(errors="strict").splitlines():
        line = line.strip()
        if path.suffix == ".jdx":
            if line.startswith("##TITLE="):
                metadata, peaks = {}, []
            if line.startswith("##"):
                key, value = line[2:].split("=", 1)
                metadata[key.lower()] = value.strip()
                if key == "END":
                    records.append((metadata, peaks))
                    in_peaks = False
                elif key == "PEAK TABLE":
                    if value != "(XY..XY)":
                        raise ValueError(f"Unsupported JCAMP table: {value}")
                    in_peaks = True
                continue
            if in_peaks and line:
                pairs = re.findall(rf"({NUMBER}),({NUMBER})", line)
                if not pairs:
                    raise ValueError(f"Invalid JCAMP peak row: {line}")
                peaks.extend((float(x), float(y)) for x, y in pairs)
        else:
            if line.lower().startswith("name:"):
                if metadata:
                    records.append((metadata, peaks))
                metadata, peaks, in_peaks = {}, [], False
            if in_peaks and line:
                match = re.match(rf"^({NUMBER})\s+({NUMBER})(?:\s|$)", line)
                if match is None:
                    raise ValueError(f"Invalid MSP peak row: {line}")
                peaks.append(tuple(map(float, match.groups())))
            elif ":" in line:
                key, value = line.split(":", 1)
                metadata[key.lower()] = value.strip()
                in_peaks = key.lower().replace(" ", "") == "numpeaks"
    if path.suffix != ".jdx" and metadata:
        records.append((metadata, peaks))
    return records


def formula_counts(formula: str) -> dict:
    """Compare elemental formulas regardless of element ordering (isotopes excluded)."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if "".join(e + n for e, n in tokens) != formula:
        return {}
    return {e: int(n or 1) for e, n in tokens}


def main() -> None:
    """Write record-level audit, source summary, SDF inventory, and input hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hr_ei_ms"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    RDLogger.DisableLog("rdApp.*")

    # 1. Load the existing nominal EI split, using its stored connectivity identifiers.
    mol_path = Path("data/proc/nist23_ei/mol_df.pkl")
    train_path = Path("data/split/nist23_ei_inchikey/train_ids.csv")
    train_keys = None
    if mol_path.exists() and train_path.exists():
        mols = pd.read_pickle(mol_path)
        train = pd.read_csv(train_path)
        train_keys = set(mols.loc[mols.mol_id.isin(train.mol_id), "inchikey_s"])

    # 2. Inspect each primary library once; subset MSPs duplicate HREI-MSDB records.
    sources = {
        "hrei_msdb": "hrei_msdb/all_spectra.msp",
        "recetox_exposome": "recetox_exposome/recetox_exposome_gc-ei-ms_20201028.msp",
        "recetox_metabolome": "recetox_metabolome/rcx_gc-orbitrap_metabolites_20210817.msp",
        "castro_gc_qtof": "castro_gc_qtof/ChromChem_USC_EIQTOF_library_JCAMP.jdx",
    }
    rows, inputs = [], []
    for source, relative in sources.items():
        path = args.raw_dir / relative
        inputs.append(path)
        for index, (meta, peaks) in enumerate(read_records(path)):
            array = np.array(peaks, dtype=float).reshape(-1, 2)
            if path.suffix == ".jdx":
                array[:, 0] *= float(meta["xfactor"])
                array[:, 1] *= float(meta["yfactor"])
            name = meta.get("name", meta.get("title", ""))
            smiles = meta.get("smiles", "")
            if not smiles:
                match = re.search(r'"SMILES=([^\"]+)"', meta.get("comments", ""))
                smiles = match[1] if match else ""
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            key = Chem.MolToInchiKey(mol) if mol is not None else ""
            formula = rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""
            supplied_formula = meta.get("formula", meta.get("molform", ""))
            labelled = mol is not None and any(a.GetIsotope() for a in mol.GetAtoms())
            expected = int(meta.get("num peaks", meta.get("npoints", "-1")))
            rows.append(
                {
                    "source": source,
                    "record": index,
                    "name": name,
                    "smiles": smiles,
                    "structure_valid": mol is not None,
                    "computed_inchikey": key,
                    "connectivity_key": key.split("-")[0],
                    "formula": supplied_formula,
                    "computed_formula": formula,
                    "isotope_labelled": labelled,
                    "formula_match": (formula_counts(formula) == formula_counts(supplied_formula))
                    if mol is not None and not labelled
                    else None,
                    "supplied_key": meta.get("inchikey", ""),
                    "key_match": key == meta["inchikey"] if key and meta.get("inchikey") else None,
                    "connectivity_match": key.split("-")[0] == meta["inchikey"].split("-")[0]
                    if key and meta.get("inchikey")
                    else None,
                    "derivative_name": bool(re.search(r"TMS|trimethylsilyl|methoxim", name, re.I)),
                    "silicon_structure": mol is not None
                    and any(a.GetAtomicNum() == 14 for a in mol.GetAtoms()),
                    "peak_count": len(peaks),
                    "expected_peaks": expected,
                    "peak_count_match": len(peaks) == expected,
                    "finite_positive_peaks": bool(
                        len(array) and np.isfinite(array).all() and (array > 0).all()
                    ),
                    "fractional_peak_count": int(
                        (np.abs(array[:, 0] - np.rint(array[:, 0])) > 1e-6).sum()
                    ),
                    "min_mz": float(array[:, 0].min()) if len(array) else None,
                    "max_mz": float(array[:, 0].max()) if len(array) else None,
                    "nominal_train_overlap": key.split("-")[0] in train_keys
                    if key and train_keys is not None
                    else None,
                }
            )

    # 3. Inventory SDF parsing separately; never substitute native SDFs for derivative SMILES.
    sdf_rows = []
    for path in sorted(args.raw_dir.glob("*/*.sdf")):
        inputs.append(path)
        molecules = list(Chem.SDMolSupplier(str(path)))
        for index, mol in enumerate(molecules):
            sdf_rows.append(
                {
                    "file": str(path),
                    "record": index,
                    "valid": mol is not None,
                    "name": mol.GetProp("_Name") if mol is not None else "",
                    "formula": rdMolDescriptors.CalcMolFormula(mol) if mol is not None else "",
                }
            )
    records = pd.DataFrame(rows)
    records["mapping_consistent"] = (
        records.structure_valid
        & records.formula_match.eq(True)
        & records.connectivity_match.eq(True)
        & ~records.isotope_labelled
    )
    records.to_csv(args.output_dir / "records.csv", index=False)
    pd.DataFrame(sdf_rows).to_csv(args.output_dir / "sdf_records.csv", index=False)
    summary = records.groupby("source").agg(
        spectra=("record", "size"),
        valid_structures=("structure_valid", "sum"),
        fractional_spectra=("fractional_peak_count", lambda s: int((s > 0).sum())),
        peaks=("peak_count", "sum"),
        peak_count_matches=("peak_count_match", "sum"),
        valid_peak_tables=("finite_positive_peaks", "sum"),
        formula_matches=("formula_match", "sum"),
        key_matches=("key_match", "sum"),
        connectivity_matches=("connectivity_match", "sum"),
        mapping_consistent=("mapping_consistent", "sum"),
        isotope_labelled=("isotope_labelled", "sum"),
        derivative_names=("derivative_name", "sum"),
        silicon_structures=("silicon_structure", "sum"),
        nominal_train_overlap=("nominal_train_overlap", "sum"),
        overlap_assessable=("nominal_train_overlap", "count"),
    )
    summary.to_csv(args.output_dir / "summary.csv")
    mapped = records.loc[records.structure_valid]
    cross = mapped.groupby("connectivity_key").filter(lambda g: g.source.nunique() > 1)
    cross.to_csv(args.output_dir / "cross_library_overlap.csv", index=False)
    inputs += [p for p in [mol_path, train_path] if p.exists()]
    provenance = {
        "rdkit_version": rdBase.rdkitVersion,
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs
        ],
        "nominal_train_keys": len(train_keys) if train_keys is not None else None,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string())
    print("Cross-library shared connectivity keys:", cross.connectivity_key.nunique())


if __name__ == "__main__":
    main()
