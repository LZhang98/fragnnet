"""
Create mol_df from candidate SMILES JSON file for MS2C Spectraverse dataset.

This script converts the JSON file containing query SMILES and candidate SMILES
into a mol_df dataframe compatible with fragnnet inference pipelines.

The JSON structure is:
{
    "query_smiles_1": ["candidate_1", "candidate_2", ...],
    "query_smiles_2": ["candidate_1", "candidate_2", ...],
    ...
}

Output:
- mol_df.pkl: DataFrame with columns matching fragnnet standard format:
    - mol_id: integer id
    - smiles: canonicalized SMILES
    - mol: RDKit Mol object
    - inchikey_s: first 14 chars of InChIKey
    - scaffold: Murcko scaffold SMILES
    - formula: molecular formula
    - inchi: InChI string
    - mw/exact_mw: molecular weights (average/exact)
    - num_atoms/num_bonds/num_radicals: structural counts
    - charge: formal charge
    - single_mol: flag for single connected component
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from rdkit import RDLogger

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_per_query_mol_dfs(candidates_dict: dict, mol_df: pd.DataFrame, output_dp: str) -> None:
    """
    Save separate mol_df for each query SMILES.

    Args:
        candidates_dict: Dictionary mapping query SMILES to candidate SMILES lists
        mol_df: Full mol_df DataFrame with all molecules
        output_dp: Output directory path
    """
    per_query_dp = os.path.join(output_dp, "per_query_mol_dfs")
    os.makedirs(per_query_dp, exist_ok=True)

    for idx, (query_smiles, candidate_list) in enumerate(candidates_dict.items()):
        # Get all SMILES for this query (query + candidates)
        query_smiles_set = {query_smiles}
        query_smiles_set.update(candidate_list)

        # Filter mol_df to only include these SMILES
        query_mol_df = mol_df[mol_df["smiles"].isin(query_smiles_set)].copy()

        # Reset mol_id to be sequential for this subset
        query_mol_df = query_mol_df.reset_index(drop=True)
        query_mol_df.loc[:, "mol_id"] = np.arange(len(query_mol_df))

        # Create filename based on index (sanitize query_smiles for filename)
        filename = f"mol_df_query_{idx:05d}.pkl"
        output_fp = os.path.join(per_query_dp, filename)
        query_mol_df.to_pickle(output_fp)

        if (idx + 1) % 100 == 0 or idx == 0 or idx == len(candidates_dict) - 1:
            logger.info(
                f"Saved {idx + 1}/{len(candidates_dict)}: {filename} "
                f"({len(query_mol_df)} molecules)"
            )

    logger.info(f"Saved {len(candidates_dict)} per-query mol_df files to {per_query_dp}")


def create_mol_df_from_json(
    json_fp: str, output_dp: str, save_per_query: bool = False
) -> pd.DataFrame:
    """
    Create mol_df from candidate SMILES JSON file.

    Args:
        json_fp: Path to JSON file with structure {query_smiles: [candidate_smiles_list]}
        output_dp: Output directory path
        save_per_query: If True, save a separate mol_df for each query SMILES

    Returns:
        mol_df: DataFrame with molecule information
    """
    # Load JSON
    logger.info(f"Loading JSON from {json_fp}")
    with open(json_fp) as f:
        candidates_dict = json.load(f)

    # Collect all unique SMILES
    all_smiles = set()
    for query_smiles, candidate_list in candidates_dict.items():
        all_smiles.add(query_smiles)
        all_smiles.update(candidate_list)

    all_smiles = sorted(list(all_smiles))
    logger.info(f"Total unique SMILES: {len(all_smiles)}")

    # Create initial DataFrame
    mol_df = pd.DataFrame({"smiles": all_smiles})

    # Convert SMILES to RDKit molecules
    logger.info("Converting SMILES to RDKit molecules...")
    mol_df.loc[:, "mol"] = par_apply_series(mol_df["smiles"], data_utils.mol_from_smiles)

    # Remove invalid molecules
    initial_count = len(mol_df)
    mol_df = mol_df[mol_df["mol"].notna()]
    valid_count = len(mol_df)
    if initial_count != valid_count:
        logger.warning(f"Removed {initial_count - valid_count} invalid SMILES")

    # Generate molecular properties using data_utils functions
    logger.info("Computing molecular properties...")
    mol_df.loc[:, "inchikey_s"] = par_apply_series(mol_df["mol"], data_utils.mol_to_inchikey_s)
    mol_df.loc[:, "scaffold"] = par_apply_series(mol_df["mol"], data_utils.get_murcko_scaffold)
    mol_df.loc[:, "formula"] = par_apply_series(mol_df["mol"], data_utils.mol_to_formula)
    mol_df.loc[:, "inchi"] = par_apply_series(mol_df["mol"], data_utils.mol_to_inchi)
    mol_df.loc[:, "mw"] = par_apply_series(
        mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=False)
    )
    mol_df.loc[:, "exact_mw"] = par_apply_series(
        mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=True)
    )
    mol_df.loc[:, "num_atoms"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_atoms)
    mol_df.loc[:, "num_bonds"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_bonds)
    mol_df.loc[:, "num_radicals"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_radicals)
    mol_df.loc[:, "charge"] = par_apply_series(mol_df["mol"], data_utils.mol_to_charge)
    mol_df.loc[:, "single_mol"] = par_apply_series(mol_df["mol"], data_utils.check_single_mol)

    # Assign mol_id
    mol_df = mol_df.reset_index(drop=True)
    mol_df.loc[:, "mol_id"] = np.arange(len(mol_df))

    # Reorder columns to match fragnnet standard
    column_order = [
        "mol_id",
        "smiles",
        "mol",
        "inchikey_s",
        "scaffold",
        "formula",
        "inchi",
        "mw",
        "exact_mw",
        "num_atoms",
        "num_bonds",
        "num_radicals",
        "charge",
        "single_mol",
    ]
    mol_df = mol_df[column_order]

    # Save
    os.makedirs(output_dp, exist_ok=True)
    output_fp = os.path.join(output_dp, "mol_df.pkl")
    mol_df.to_pickle(output_fp)
    logger.info(f"Saved mol_df to {output_fp}")

    # Save per-query mol_df files if requested
    if save_per_query:
        logger.info("\n=== Saving per-query mol_df files ===")
        save_per_query_mol_dfs(candidates_dict, mol_df, output_dp)
    else:
        # Save compressed version
        output_fp_gz = os.path.join(output_dp, "mol_df.pkl.gz")
        mol_df.to_pickle(output_fp_gz)
        logger.info(f"Saved mol_df (compressed) to {output_fp_gz}")
    # Print summary
    logger.info("\n=== mol_df Summary ===")
    logger.info(f"Total molecules: {len(mol_df)}")
    logger.info(f"Columns: {list(mol_df.columns)}")
    logger.info("\nFirst 5 rows:")
    logger.info(mol_df[["mol_id", "smiles", "formula", "mw"]].head())

    return mol_df


def main():
    parser = argparse.ArgumentParser(
        description="Create mol_df from candidate SMILES JSON for MS2C Spectraverse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Basic usage
            python 01_create_ms2c_df.py --json_fp data/raw/spectraverse/nps_candidates/nps_nist23_clm_formula_example.json --output_dp data/ms2c/spectraverse

            # With custom output directory
            python 01_create_ms2c_df.py --json_fp candidates.json --output_dp output/ms2c_data
        """,
    )
    parser.add_argument(
        "--json_fp",
        type=str,
        required=True,
        help="Path to JSON file with candidate SMILES {query_smiles: [candidate_list]}",
    )
    parser.add_argument(
        "--output_dp",
        type=str,
        default="data/ms2c/spectraverse",
        help="Output directory path for mol_df.pkl and mol_df.pkl.gz",
    )
    parser.add_argument(
        "--save_per_query",
        action="store_true",
        help="Save a separate mol_df for each query SMILES in per_query_mol_dfs/ subdirectory",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger.setLevel(getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("MS2C Spectraverse - Create mol_df from JSON")
    logger.info("=" * 80)
    logger.info(f"JSON file: {args.json_fp}")
    logger.info(f"Output directory: {args.output_dp}")
    logger.info(f"Save per-query mol_dfs: {args.save_per_query}")
    logger.info(f"Log level: {args.log_level}")
    logger.info("=" * 80)

    # Create mol_df
    create_mol_df_from_json(args.json_fp, args.output_dp, args.save_per_query)

    logger.info("\n" + "=" * 80)
    logger.info("Done!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
