import argparse
import json
import logging
import tarfile
from typing import Any

import pandas as pd

REQUIRED_SPEC_COLS = ["spec_id", "mol_id", "peaks"]
REQUIRED_MOL_COLS = ["mol_id", "smiles"]


def load_candidates_dict(json_file: str, logger: logging.Logger) -> dict[str, list[str]]:
    """Load a candidates dict mapping query SMILES to candidate SMILES lists."""
    logger.info("Loading JSON from %s", json_file)
    if json_file.endswith(".tar.gz"):
        with tarfile.open(json_file, "r:gz") as tar:
            json_members = [member for member in tar.getmembers() if member.name.endswith(".json")]
            if not json_members:
                raise ValueError(f"No .json file found in archive: {json_file}")
            if len(json_members) > 1:
                logger.warning(
                    "Multiple .json files found in archive %s. Using first: %s",
                    json_file,
                    json_members[0].name,
                )
            json_file_obj = tar.extractfile(json_members[0])
            if json_file_obj is None:
                raise ValueError(f"Failed to extract {json_members[0].name} from {json_file}")
            with json_file_obj:
                candidates_dict = json.load(json_file_obj)
    else:
        with open(json_file) as f:
            candidates_dict = json.load(f)

    if not isinstance(candidates_dict, dict):
        raise ValueError("JSON root must be a dictionary of query_smiles -> candidate_smiles_list")

    return candidates_dict


def validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    """Validate required columns exist in a DataFrame."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")


def build_candidate_df(
    candidates_dict: dict[str, list[str]],
    spec_df: pd.DataFrame,
    mol_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Build candidate_df compatible with ms2c_inference candidate_to_query_sets."""
    validate_columns(spec_df, REQUIRED_SPEC_COLS, "spec_df")
    validate_columns(mol_df, REQUIRED_MOL_COLS, "mol_df")

    mol_smiles_to_id: dict[str, int] = {}
    for _, row in mol_df.iterrows():
        smiles = row["smiles"]
        mol_id = row["mol_id"]
        if smiles in mol_smiles_to_id:
            continue
        mol_smiles_to_id[smiles] = int(mol_id)

    records: list[dict[str, Any]] = []
    for query_smiles, candidate_smiles_list in candidates_dict.items():
        
        if not isinstance(candidate_smiles_list, list):
            raise ValueError(f"Expected list of candidates for {query_smiles}")
        if query_smiles not in mol_smiles_to_id:
            raise ValueError(f"Query SMILES not found in mol_df: {query_smiles}")

        query_mol_id = mol_smiles_to_id[query_smiles]
        query_spec_rows = spec_df[spec_df["mol_id"] == query_mol_id]
        if query_spec_rows.empty:
            raise ValueError(f"No spec_df rows found for query mol_id={query_mol_id}")

        candidate_mol_id_list: list[int | None] = []
        missing_candidates = 0
        for candidate_smiles in candidate_smiles_list:
            if candidate_smiles in mol_smiles_to_id:
                candidate_mol_id_list.append(mol_smiles_to_id[candidate_smiles])
            else:
                candidate_mol_id_list.append(None)
                missing_candidates += 1

        if missing_candidates:
            logger.warning(
                "%d candidate SMILES missing from mol_df for query_smiles=%s",
                missing_candidates,
                query_smiles,
            )

        for _, spec_row in query_spec_rows.iterrows():
            records.append(
                {
                    "query_spec_id": spec_row["spec_id"],
                    "query_mol_id": spec_row["mol_id"],
                    "query_peaks": spec_row["peaks"],
                    "candidate_smiles_list": candidate_smiles_list,
                    "candidate_mol_id_list": candidate_mol_id_list,
                }
            )

    return pd.DataFrame(records)

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
        #required=True,
        default="data/ms2c/nps/nps_candidates/nps_nist23/nps_nist23_clm_formula.json",
        help="Path to JSON file containing query SMILES and candidate SMILES lists",
    )

    parser.add_argument(
        "--spec_df_path",
        type=str,
        #required=True,
        default="data/proc/nps_nist23/spec_df.pkl",
        help="path to spec_df.pkl.gz file created by 02_prepare_spec_df.py which has specid and metadata",
    )
    parser.add_argument(
        "--mol_df_path",
        type=str,
        #required=True,
        default="data/proc/nps_nist23/mol_df.pkl",
        help="path to mol_df.pkl.gz file created by 02_prepare_spec_df.py which has molid and metadata",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        #required=True,
        default="data/preproc/ms2c_inference/nps_nist23_candidate_df.pkl.gz",
        help="Path to output candidate_df.pkl.gz",
    )

    args = parser.parse_args()

    # Load JSON file

    candidates_dict = load_candidates_dict(args.json_file, logger)

    # logger.info(f"Total unique SMILES collected: {len(all_smiles)}")
    query_smiles_list = list(candidates_dict.keys())
    logger.info(f"Total query SMILES collected: {len(query_smiles_list)}")

    mol_df = pd.read_pickle(args.mol_df_path)
    logger.info("Loaded mol_df with %d molecules", len(mol_df))
    spec_df = pd.read_pickle(args.spec_df_path)
    logger.info("Loaded spec_df with %d spectra", len(spec_df))

    candidate_df = build_candidate_df(candidates_dict, spec_df, mol_df, logger)
    logger.info("Built candidate_df with %d rows", len(candidate_df))
    candidate_df.to_pickle(args.output_path)
    logger.info("Saved candidate_df to %s", args.output_path)
