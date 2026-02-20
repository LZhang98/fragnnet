"""Build a molecule-level ClassyFire annotation dataframe.

Merges ClassyFire chemical taxonomy (kingdom → superklass → klass → subklass →
direct_parent) onto the project's mol_df via SMILES canonicalization, producing
a lookup pickle keyed by mol_id.

Usage:
    python preproc_scripts/08_prepare_classyfire.py \\
        --mol_df data/proc/nist20/mol_df.pkl \\
        --cf_zip data/classyfire/20231201_nist20mona23_1.zip \\
        --output data/classyfire/nist20_mol_classyfire.pkl
"""

import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd

from fragnnet.utils.data_utils import rdkit_import

logger = logging.getLogger(__name__)

# Columns to keep from the ClassyFire TSV
CF_COLS = [
    "kingdom",
    "superklass",
    "klass",
    "subklass",
    "direct_parent",
    "alternative_parents",
    "substituents",
]

# Output columns (mol_df identity cols + CF taxonomy cols)
OUTPUT_COLS = ["mol_id", "smiles", "inchikey"] + CF_COLS


def load_classyfire_tsv(zip_path: str) -> pd.DataFrame:
    """Load the ClassyFire TSV file from inside a zip archive.

    Expects exactly one TSV file inside the zip.

    Args:
        zip_path: Path to the zip archive containing the ClassyFire TSV.

    Returns:
        DataFrame with columns: sid, smiles, kingdom, superklass, klass,
        subklass, direct_parent, geometric_descriptor, alternative_parents,
        substituents.

    Raises:
        FileNotFoundError: If no TSV file is found inside the zip.
        ValueError: If more than one TSV file is found inside the zip.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
        if len(tsv_names) == 0:
            raise FileNotFoundError(f"No .tsv file found inside {zip_path}")
        if len(tsv_names) > 1:
            raise ValueError(f"Expected 1 .tsv file in {zip_path}, found {len(tsv_names)}: {tsv_names}")
        with zf.open(tsv_names[0]) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), sep="\t", dtype=str)
    logger.info("Loaded %d ClassyFire records from %s", len(df), zip_path)
    return df


def canonicalize_smiles_series(smiles_series: pd.Series) -> pd.Series:
    """Canonicalize a Series of SMILES strings using RDKit.

    Invalid SMILES (unparseable by RDKit) are returned as None.

    Args:
        smiles_series: Series of SMILES strings.

    Returns:
        Series of canonical SMILES strings (same index), None for failures.
    """
    (Chem,) = rdkit_import("rdkit.Chem")

    def _canonicalize(smi: str) -> str | None:
        if not isinstance(smi, str) or not smi.strip():
            return None
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    return smiles_series.map(_canonicalize)


def parse_semicolon_list(value: str | float) -> list[str]:
    """Parse a semicolon-separated ClassyFire string into a list.

    Args:
        value: Semicolon-separated string, or NaN/None if absent.

    Returns:
        List of stripped, non-empty strings. Empty list if value is NaN/None.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def build_mol_classyfire_df(
    mol_df: pd.DataFrame,
    cf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ClassyFire annotations onto mol_df via canonical SMILES.

    Args:
        mol_df: Molecule dataframe with columns: mol_id, smiles, inchikey_s.
        cf_df: Combined ClassyFire dataframe with columns: smiles + CF_COLS.
            May contain duplicates; first occurrence per canonical SMILES is kept.

    Returns:
        DataFrame with columns defined by OUTPUT_COLS. Contains only molecules
        that matched at least one ClassyFire record (inner join). Indexed 0..N-1.
    """
    logger.info("Canonicalizing mol_df SMILES (%d rows)...", len(mol_df))
    mol_df = mol_df.copy()
    mol_df["canonical_smiles"] = canonicalize_smiles_series(mol_df["smiles"])

    logger.info("Canonicalizing ClassyFire SMILES (%d rows)...", len(cf_df))
    cf_df = cf_df.copy()
    cf_df["canonical_smiles"] = canonicalize_smiles_series(cf_df["smiles"])

    # Drop rows where canonicalization failed
    cf_df = cf_df.dropna(subset=["canonical_smiles"])
    mol_df_valid = mol_df.dropna(subset=["canonical_smiles"])

    # Deduplicate ClassyFire by canonical SMILES (keep first occurrence)
    cf_df = cf_df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    logger.info("ClassyFire unique canonical SMILES: %d", len(cf_df))

    # Select only needed CF columns for the merge
    cf_merge = cf_df[["canonical_smiles"] + CF_COLS]

    # Inner join: only molecules with ClassyFire annotations
    merged = mol_df_valid.merge(cf_merge, on="canonical_smiles", how="inner")

    # Parse semicolon-separated columns to Python lists
    for col in ("alternative_parents", "substituents"):
        merged[col] = merged[col].map(parse_semicolon_list)

    # Rename inchikey_s → inchikey for output
    if "inchikey_s" in merged.columns and "inchikey" not in merged.columns:
        merged = merged.rename(columns={"inchikey_s": "inchikey"})

    # Coverage report
    n_total = len(mol_df)
    n_matched = len(merged)
    logger.info(
        "ClassyFire coverage: %d / %d mol_df rows matched (%.1f%%)",
        n_matched,
        n_total,
        100.0 * n_matched / n_total if n_total > 0 else 0.0,
    )

    return merged[OUTPUT_COLS].reset_index(drop=True)


def _load_mol_df(mol_df_path: str) -> pd.DataFrame:
    """Load mol_df from pickle based on file extension.

    Args:
        mol_df_path: Path to mol_df file (.pkl or .pickle).

    Returns:
        Loaded mol_df DataFrame.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(mol_df_path)
    if path.suffix in (".pkl", ".pickle"):
        return pd.read_pickle(mol_df_path)
    raise ValueError(f"Unsupported mol_df format: {path.suffix}. Expected .pkl or .pickle")


def main() -> None:
    """CLI entry point: load data, build ClassyFire dataframe, save pickle."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mol_df",
        default="data/proc/nist20/mol_df.pkl",
        help="Path to mol_df (.pkl or .pickle, default: data/proc/nist20/mol_df.pkl)",
    )
    parser.add_argument("--cf_zip", required=True, help="Path to ClassyFire zip file")
    parser.add_argument(
        "--output",
        default="data/classyfire/mol_classyfire.pkl",
        help="Output pickle path (default: data/classyfire/mol_classyfire.pkl)",
    )
    args = parser.parse_args()

    # Load inputs
    logger.info("Loading mol_df from %s", args.mol_df)
    mol_df = _load_mol_df(args.mol_df)

    logger.info("Loading ClassyFire TSV from %s", args.cf_zip)
    cf_df = load_classyfire_tsv(args.cf_zip)

    # Build output dataframe
    result_df = build_mol_classyfire_df(mol_df, cf_df)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_pickle(output_path)
    logger.info("Saved %d rows to %s", len(result_df), output_path)

    # Summary stats
    for col in ("kingdom", "superklass", "klass"):
        n_unique = result_df[col].nunique()
        logger.info("Unique %s values: %d", col, n_unique)


if __name__ == "__main__":
    main()
