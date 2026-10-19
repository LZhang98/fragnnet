"""Create a new NIST23 EI split excluding resolvable downloaded HR-EI identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


def exclude_rows(frame: pd.DataFrame, molecules: pd.DataFrame, keys: set[str]) -> tuple:
    """Partition original split rows by connectivity identity without reassigning survivors."""
    lookup = molecules.set_index("mol_id")["inchikey_s"]
    identifiers = frame.mol_id.map(lookup)
    if identifiers.isna().any():
        raise ValueError("Split contains unmapped molecule IDs")
    remove = identifiers.isin(keys)
    return frame.loc[~remove].copy(), frame.loc[remove].copy()


def main() -> None:
    """Write split CSVs, exclusion provenance, removed rows, and an audit summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path, default=Path("output/hr_ei_raw_audit_2026_10_17/verified/records.csv")
    )
    parser.add_argument("--source", type=Path, default=Path("data/split/nist23_ei_inchikey"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    RDLogger.DisableLog("rdApp.*")

    # 1. Preserve both computed and supplied identities, even for flagged reference records.
    audit = pd.read_csv(args.audit).fillna("")
    identities = []
    for row in audit.itertuples():
        for field in ["computed_inchikey", "supplied_key"]:
            key = getattr(row, field).split("-")[0]
            if re.fullmatch("[A-Z]{14}", key):
                identities.append(
                    {
                        "connectivity_key": key,
                        "source": row.source,
                        "record": str(row.record),
                        "identity_field": field,
                        "name": row.name,
                    }
                )
    # The native structures are additional conservative exclusions, not derivative substitutes.
    sdf_paths = sorted(Path("data/raw/hr_ei_ms").glob("recetox*/*.sdf"))
    for path in sdf_paths:
        for index, mol in enumerate(Chem.SDMolSupplier(str(path))):
            if mol is not None:
                key = Chem.MolToInchiKey(mol).split("-")[0]
                identities.append(
                    {
                        "connectivity_key": key,
                        "source": path.parent.name,
                        "record": f"sdf:{index}",
                        "identity_field": "sdf_native_or_reference",
                        "name": mol.GetProp("_Name"),
                    }
                )
    identity_frame = pd.DataFrame(identities).drop_duplicates()
    keys = set(identity_frame.connectivity_key)
    identity_frame.to_csv(args.output / "excluded_identities.csv", index=False)

    # 2. Filter every partition in place in its ordering; never move a survivor to another split.
    mol_path = Path("data/proc/nist23_ei/mol_df.pkl")
    molecules = pd.read_pickle(mol_path)
    removed, summary, input_paths = [], {}, [args.audit, mol_path, *sdf_paths]
    for name in ["train", "val", "test", "secondary"]:
        path = args.source / f"{name}_ids.csv"
        frame = pd.read_csv(path)
        kept, dropped = exclude_rows(frame, molecules, keys)
        kept.to_csv(args.output / path.name, index=False)
        dropped["partition"] = name
        removed.append(dropped)
        summary[name] = {
            "original_spectra": len(frame),
            "retained_spectra": len(kept),
            "removed_spectra": len(dropped),
            "retained_molecules": int(kept.mol_id.nunique()),
        }
        input_paths.append(path)
        retained_keys = set(molecules.loc[molecules.mol_id.isin(kept.mol_id), "inchikey_s"])
        if retained_keys & keys:
            raise RuntimeError(f"HR-EI overlap remains in {name}")
    pd.concat(removed, ignore_index=True).to_csv(args.output / "removed_rows.csv", index=False)
    original_meta = args.source / "meta.json"
    meta = json.loads(original_meta.read_text())
    meta.update(
        split_dp=str(args.output),
        source_split=str(args.source),
        exclusion_policy="HR-EI computed/supplied connectivity keys + RECETOX SDF parents",
        excluded_connectivity_keys=len(keys),
        partitions=summary,
        unresolved_sources=["castro_gc_qtof"],
        input_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*input_paths, original_meta]
        },
        original_split_assignments_preserved=True,
        all_retained_partitions_overlap=0,
    )
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"excluded_connectivity_keys": len(keys), "partitions": summary}, indent=2))


if __name__ == "__main__":
    main()
