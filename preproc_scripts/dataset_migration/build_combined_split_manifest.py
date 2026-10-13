"""Build a shared molecule-disjoint split across processed spectrum sources."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build(source_dirs: list[Path], output_dir: Path, seed: int = 42) -> None:
    """Write per-source spectrum ID lists under one global InChIKey split."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    molecule_tables = [pd.read_pickle(path / "mol_df.pkl") for path in source_dirs]
    keys = pd.concat([table["inchikey_s"] for table in molecule_tables], ignore_index=True)
    if keys.isna().any():
        raise ValueError("Missing molecule InChIKey first block")
    unique_keys = np.sort(keys.unique())
    shuffled = np.random.default_rng(seed).permutation(unique_keys)
    n_test = int(np.ceil(len(shuffled) * 0.1))
    n_val = int(np.ceil(len(shuffled) * 0.1))
    assignments = pd.Series("train", index=unique_keys)
    assignments.loc[shuffled[:n_test]] = "test"
    assignments.loc[shuffled[n_test : n_test + n_val]] = "val"

    output_dir.mkdir(parents=True)
    counts = {}
    for path, molecules in zip(source_dirs, molecule_tables, strict=True):
        name = path.name
        molecule_split = pd.Series(
            molecules["inchikey_s"].map(assignments).to_numpy(), index=molecules["mol_id"]
        )
        spectra = pd.read_pickle(path / "spec_df.pkl")
        spectra_split = spectra["mol_id"].map(molecule_split)
        if spectra_split.isna().any():
            raise ValueError(f"Unmapped spectrum molecule in {path}")
        source_output = output_dir / name
        source_output.mkdir()
        counts[name] = {}
        for split in ("train", "val", "test"):
            ids = spectra.loc[spectra_split == split, ["spec_id", "mol_id", "group_id"]]
            ids.to_csv(source_output / f"{split}_ids.csv", index=False)
            counts[name][split] = len(ids)
        del spectra

    manifest = {
        "sources": [str(path.resolve()) for path in source_dirs],
        "split_key": "inchikey_s",
        "seed": seed,
        "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "molecule_keys": len(unique_keys),
        "spectrum_counts": counts,
        "format": "Per-source IDs refer to their source proc tables; IDs are not globally unique.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    build(arguments.sources, arguments.output, arguments.seed)
