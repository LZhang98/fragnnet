import argparse
import json
import logging

import pandas as pd

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Create candidate SMILES set from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:
            python create_candidate_set.py --json_file candidates.json
        """,
    )

    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to JSON file containing query SMILES and candidate SMILES lists",
    )

    parser.add_argument(
        "--spec_df_path",
        type=str,
        required=True,
        help="path to spec_df.pkl.gz file created by 02_prepare_spec_df.py which has specid and metadata",
    )
    parser.add_argument(
        "--mol_df_path",
        type=str,
        required=True,
        help="path to mol_df.pkl.gz file created by 02_prepare_spec_df.py which has molid and metadata",
    )

    args = parser.parse_args()

    # Load JSON file

    logger.info(f"Loading JSON from {args.json_file}")
    with open(args.json_file) as f:
        candidates_dict = json.load(f)

    # logger.info(f"Total unique SMILES collected: {len(all_smiles)}")
    query_smiles_list = list(candidates_dict.keys())
    logger.info(f"Total query SMILES collected: {len(query_smiles_list)}")

    mol_df = pd.read_pickle(args.mol_df_path)
    spec_df = pd.read_pickle(args.spec_df_path)
    logger.info(f"Loaded mol_df with {len(mol_df)} molecules")
    spec_mol_df = spec_df.merge(mol_df, on="mol_id")
    logger.info(f"spec_df merged with mol_df has {len(spec_mol_df)} entries")
