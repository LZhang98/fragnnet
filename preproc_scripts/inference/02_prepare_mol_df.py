"""
Prepare mol_df for FragGNN inference from SMILES input.

This script takes a list of SMILES strings and creates a mol_df file
that contains molecular properties and metadata.

Usage:
    python 02_prepare_mol_df.py --smiles_list "CCO" "CC" --output_dir output/
    python 02_prepare_mol_df.py --smiles_file smiles.txt --output_dir output/
"""

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

import fragnnet.utils.formula_utils as formula_utils
import fragnnet.utils.frag_utils as frag_utils
from fragnnet.frag.compute_frags import MAX_NUM_NODES
from fragnnet.utils import data_utils
from fragnnet.utils.data_utils import par_apply_series

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_mol_df(smiles_list, ids_list, max_heavy_atoms):
    """
    Create mol_df from a list of SMILES strings.

    Args:
        smiles_list (list): List of SMILES strings
        ids_list (list, optional): List of molecule IDs
        max_heavy_atoms (int): Maximum number of heavy atoms allowed

    Returns:
        pd.DataFrame: mol_df with required columns
    """
    logger.info(f"Creating mol_df for {len(smiles_list)} SMILES...")

    # Create initial DataFrame
    mol_df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "mol_id": range(len(smiles_list)),
            "input_mol_id": np.array([str(i) for i in range(len(smiles_list))])
            if ids_list is None
            else np.where(pd.isna(ids_list), np.nan, [str(x) for x in ids_list]),
        }
    )

    # Convert SMILES to RDKit molecules
    logger.info("Converting SMILES to RDKit molecules...")
    mol_df.loc[:, "mol"] = par_apply_series(mol_df["smiles"], data_utils.mol_from_smiles)

    # Remove invalid molecules
    initial_count = len(mol_df)
    mol_df = mol_df[mol_df["mol"].notna()]
    valid_count = len(mol_df)
    if initial_count != valid_count:
        logger.warning(f"Removed {initial_count - valid_count} invalid SMILES")

    # Generate molecular properties
    logger.info("Computing molecular properties...")
    mol_df.loc[:, "inchikey_s"] = par_apply_series(mol_df["mol"], data_utils.mol_to_inchikey_s)
    mol_df.loc[:, "formula"] = par_apply_series(mol_df["mol"], data_utils.mol_to_formula)
    mol_df.loc[:, "mw"] = par_apply_series(
        mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=False)
    )
    mol_df.loc[:, "exact_mw"] = par_apply_series(
        mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=True)
    )
    mol_df.loc[:, "num_atoms"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_atoms)
    mol_df.loc[:, "num_bonds"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_bonds)
    mol_df.loc[:, "charge"] = par_apply_series(mol_df["mol"], data_utils.mol_to_charge)
    mol_df.loc[:, "single_mol"] = par_apply_series(mol_df["mol"], data_utils.check_single_mol)
    mol_df.loc[:, "num_radicals"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_radicals)
    # Heavy atoms = non-hydrogen atoms
    mol_df.loc[:, "num_heavy_atoms"] = par_apply_series(
        mol_df["mol"], lambda m: int(m.GetNumHeavyAtoms()) if m is not None else np.nan
    )
    mol_df.loc[:, "num_bonds"] = par_apply_series(
        mol_df["mol"], lambda m: int(m.GetNumBonds()) if m is not None else np.nan
    )

    # filter by out_set_elements
    allowed_elements = set(frag_utils.ELEMENTS)
    mol_df.loc[:, "num_out_set_elements"] = mol_df["formula"].apply(
        lambda x: len(set(list(formula_utils.parse_formula(x).keys())) - allowed_elements)
    )

    # Filter out molecules based on quality criteria
    logger.info("Filtering molecules based on quality criteria...")
    initial_count_filter = len(mol_df)

    # Drop molecules that are not single molecules, have radicals, or have charges
    mol_df = mol_df[
        (mol_df["single_mol"] == True)
        & (mol_df["num_radicals"] == 0)
        & (mol_df["charge"] == 0)
        & (mol_df["num_atoms"] <= max_heavy_atoms)
        & (mol_df["num_out_set_elements"] == 0)
    ]

    filtered_count = len(mol_df)
    if initial_count_filter != filtered_count:
        logger.warning(
            f"Filtered out {initial_count_filter - filtered_count} molecules due to quality criteria"
        )
        logger.info("  - Kept only single molecules with no radicals and no charge")

    # Reset mol_id after filtering
    mol_df = mol_df.reset_index(drop=True)
    mol_df.loc[:, "mol_id"] = list(range(len(mol_df)))

    logger.info(f"Created mol_df with {len(mol_df)} molecules")
    return mol_df


def validate_inputs(smiles_list, max_heavy_atoms):
    """
    Validate input parameters.

    Args:
        smiles_list (list): List of SMILES strings
        max_heavy_atoms (int): Maximum number of heavy atoms
    """
    if not smiles_list:
        raise ValueError("SMILES list cannot be empty")

    if max_heavy_atoms < 1 or max_heavy_atoms > MAX_NUM_NODES:
        raise ValueError(f"max_heavy_atoms must be between 1 and {MAX_NUM_NODES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare mol_df for FragGNN inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Single molecule
            python 02_prepare_mol_df.py --smiles_list "CCO" --output_dir output/

            # Multiple molecules
            python 02_prepare_mol_df.py --smiles_list "CCO" "CC(=O)O" "CCN" --output_dir output/

            # From file
            python 02_prepare_mol_df.py --smiles_file smiles.txt --output_dir output/

            # From CSV with IDs
            python 02_prepare_mol_df.py --smiles_file data.csv --smiles_colname smiles --id_colname mol_id --output_dir output/
                """,
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles_list", nargs="+", help="List of SMILES strings")
    group.add_argument(
        "--smiles_file", type=str, help="File containing SMILES strings (one per line, or CSV/TSV)"
    )
    parser.add_argument("--smiles_colname", type=str, help="Column name containing SMILES strings")
    parser.add_argument("--id_colname", type=str, help="Column name containing mol IDs")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for mol_df.pkl.gz and config.json",
    )
    parser.add_argument(
        "--max_heavy_atoms",
        type=int,
        default=MAX_NUM_NODES,
        help="Maximum number of heavy atoms in a molecule",
    )
    args = parser.parse_args()

    # Read SMILES
    ids_list = None
    if args.smiles_list:
        smiles_list = args.smiles_list
    else:
        logger.info(f"Reading SMILES from {args.smiles_file}")
        if (
            args.smiles_file.endswith(".smi")
            or args.smiles_file.endswith(".smiles")
            or args.smiles_file.endswith(".txt")
        ):
            with open(args.smiles_file) as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        elif (
            args.smiles_file.endswith(".csv")
            or args.smiles_file.endswith(".tsv")
            or args.smiles_file.endswith(".csv.gz")
            or args.smiles_file.endswith(".tsv.gz")
        ):
            sep = (
                ","
                if args.smiles_file.endswith(".csv") or args.smiles_file.endswith(".csv.gz")
                else "\t"
            )
            smiles_df = pd.read_csv(args.smiles_file, sep=sep)
            if not args.smiles_colname or args.smiles_colname not in smiles_df.columns:
                raise ValueError(
                    f"Column name {args.smiles_colname} for SMILES not specified or not found in CSV columns: {smiles_df.columns.tolist()}"
                )
            smiles_list = smiles_df[args.smiles_colname].astype(str).tolist()
            if args.id_colname and args.id_colname in smiles_df.columns:
                ids_list = smiles_df[args.id_colname].astype(str).tolist()
                if len(ids_list) != len(smiles_list):
                    raise ValueError(
                        f"Length of IDs list ({len(ids_list)}) does not match length of SMILES list ({len(smiles_list)})"
                    )
            elif args.id_colname not in smiles_df.columns:
                logger.warning(
                    f"ID column name '{args.id_colname}' not found in CSV columns: {smiles_df.columns.tolist()}"
                )

    # Validate inputs
    validate_inputs(smiles_list, args.max_heavy_atoms)

    logger.info("Input summary:")
    logger.info(f"  - {len(smiles_list)} SMILES")
    logger.info(f"  - Max heavy atoms: {args.max_heavy_atoms}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create mol_df
    mol_df = create_mol_df(smiles_list, ids_list, args.max_heavy_atoms)

    # Save files
    mol_df_path = os.path.join(args.output_dir, "mol_df.pkl.gz")

    logger.info(f"Saving mol_df to {mol_df_path}")
    mol_df.to_pickle(mol_df_path)

    # Save metadata config
    config_metadata = {
        "timestamp": datetime.now().isoformat(),
        "input_parameters": {
            "num_smiles": len(smiles_list),
            "smiles_file": args.smiles_file
            if hasattr(args, "smiles_file") and args.smiles_file
            else None,
            "max_heavy_atoms": args.max_heavy_atoms,
        },
        "output_files": {
            "mol_df": mol_df_path,
        },
    }

    config_path = os.path.join(args.output_dir, "mol_df_config.json")
    with open(config_path, "w") as f:
        json.dump(config_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Configuration saved to {config_path}")

    # Print summary
    logger.info("Summary:")
    logger.info(f"  - mol_df: {len(mol_df)} molecules")
    logger.info(f"  - Output directory: {args.output_dir}")
    logger.info(f"  - Configuration saved: {config_path}")

    # Print first few entries for verification
    logger.info("\nFirst 5 molecules in mol_df:")
    print(mol_df.head(5))

    # Heavy atom statistics
    if "num_heavy_atoms" in mol_df.columns and len(mol_df) > 0:
        stats = mol_df["num_heavy_atoms"].describe()
        logger.info("\nHeavy atom count stats (num_heavy_atoms):")
        logger.info(f"  count: {int(stats['count'])}")
        logger.info(f"  mean:  {stats['mean']:.2f}")
        logger.info(f"  std:   {stats['std']:.2f}")
        logger.info(f"  min:   {int(stats['min'])}")
        logger.info(f"  25%:   {stats['25%']:.2f}")
        logger.info(f"  50%:   {stats['50%']:.2f}")
        logger.info(f"  75%:   {stats['75%']:.2f}")
        logger.info(f"  max:   {int(stats['max'])}")

    # num_bonds statistics
    if "num_bonds" in mol_df.columns and len(mol_df) > 0:
        stats_bonds = mol_df["num_bonds"].describe()
        logger.info("\nBond count stats (num_bonds):")
        logger.info(f"  count: {int(stats_bonds['count'])}")
        logger.info(f"  mean:  {stats_bonds['mean']:.2f}")
        logger.info(f"  std:   {stats_bonds['std']:.2f}")
        logger.info(f"  min:   {int(stats_bonds['min'])}")
        logger.info(f"  25%:   {stats_bonds['25%']:.2f}")
        logger.info(f"  50%:   {stats_bonds['50%']:.2f}")
        logger.info(f"  75%:   {stats_bonds['75%']:.2f}")
        logger.info(f"  max:   {int(stats_bonds['max'])}")
