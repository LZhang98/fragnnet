import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import (
    par_apply_series,
)

REQUIRED_CANDIDATE_COLS = [
    "spec_id",
    "mol_id",
    "peaks",
    "prec_type",
    "inst_type",
    "ace",
    "nce",
    "ace_extra_1",
    "nce_extra_1",
    "ace_extra_2",
    "nce_extra_2",
    "candidate_smiles_list",
]


def build_query_spec_df(row: pd.Series, size: int) -> pd.DataFrame:
    """Build a per-query spec_df from a single candidate_df row."""
    records: list[dict[str, Any]] = []

    for i in range(size):
        records.append(
            {
                "spec_id": i,
                "mol_id": i,
                "peaks": row["peaks"],
                "prec_type": row["prec_type"],
                "inst_type": row["inst_type"],
                "ace": row["ace"],
                "nce": row["nce"],
                "ace_extra_1": row["ace_extra_1"],
                "nce_extra_1": row["nce_extra_1"],
                "ace_extra_2": row["ace_extra_2"],
                "nce_extra_2": row["nce_extra_2"],
            }
        )
    spec_df = pd.DataFrame(records)
    return spec_df


def build_query_mol_df(candidate_smiles_list: list[str]) -> pd.DataFrame:
    """Build a per-query mol_df from candidate_smiles_list."""
    # de-duplicate smiles
    candidate_smiles_list = list(set(candidate_smiles_list))
    mol_df = pd.DataFrame(
        zip(sorted(candidate_smiles_list), list(range(len(candidate_smiles_list)))),
        columns=["smiles", "mol_id"],
    )
    mol_df.loc[:, "mol"] = par_apply_series(mol_df["smiles"], data_utils.mol_from_smiles)
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
    mol_df.loc[:, "charge"] = par_apply_series(mol_df["mol"], data_utils.mol_to_charge)
    mol_df.loc[:, "single_mol"] = par_apply_series(mol_df["mol"], data_utils.check_single_mol)
    mol_df.loc[:, "num_radicals"] = par_apply_series(mol_df["mol"], data_utils.mol_to_num_radicals)
    if (mol_df["smiles"] == "").any():
        raise ValueError("Empty SMILES found in mol_df")
    if (mol_df["formula"] == "").any():
        raise ValueError("Empty formula found in mol_df")
    if (mol_df["exact_mw"] == 0).any():
        raise ValueError("Zero exact_mw found in mol_df")
    return mol_df


def create_query_mol_spec_df(candidate_df: pd.DataFrame, output_dir: Path) -> None:
    """Create per-query spec_df and mol_df files from candidate_df.

    For each row in candidate_df, creates:
    - A spec_df with candidate molecule IDs
    - A mol_df with candidate SMILES and properties (cached if same mol_id)
    - Records in a query_index.json for reference

    Args:
        candidate_df: DataFrame with columns including candidate_smiles_list
        output_dir: Directory to write query files and index
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in REQUIRED_CANDIDATE_COLS:
        if col not in candidate_df.columns:
            raise ValueError(f"Missing required column in candidate_df: {col}")

    processed_mol_d = {}
    index_records = []

    for i, row in candidate_df.iterrows():
        spec_query_id = row["spec_id"]
        mol_query_id = row["mol_id"]
        # if pd.isna(spec_query_id):
        #    spec_query_id = f"no_spec_{i}"

        if row["mol_id"] in processed_mol_d:
            candidates_size, mol_df_path = processed_mol_d[row["mol_id"]]
        else:
            mol_df = build_query_mol_df(row["candidate_smiles_list"])
            mol_df_path = output_dir / f"mol_df_query_{mol_query_id}.pkl.gz"
            mol_df.to_pickle(mol_df_path)
            candidates_size = len(mol_df)
            processed_mol_d[row["mol_id"]] = (candidates_size, mol_df_path)

        spec_df = build_query_spec_df(row, size=candidates_size)
        query_spec_path = output_dir / f"spec_df_query_{spec_query_id}.pkl.gz"
        spec_df.to_pickle(query_spec_path)

        index_records.append(
            {
                "query_id": spec_query_id,
                "mol_query_id": mol_query_id,
                "spec_df_path": str(query_spec_path),
                "mol_df_path": str(mol_df_path),
            }
        )

        if (i + 1) % 10 == 0:
            logging.getLogger(__name__).info(f"Processed {i + 1}/{len(candidate_df)} candidates")

    index_path = output_dir / "query_index.json"
    with index_path.open("w") as f:
        json.dump(index_records, f, indent=2)

    logger = logging.getLogger(__name__)
    logger.info(f"Processed all {len(candidate_df)} candidates successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Convert candidate_df to per-query spec_df and shared mol_df subsets.",
    )
    parser.add_argument(
        "--candidate_df_path",
        type=str,
        default="data/ms2c/candidates/nps_nist23_candidate_df.pkl.gz",
        required=False,
        help="Path to candidate_df.pkl.gz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="data/ms2c/proc/nps_nist23",
        help="Output directory for query files",
    )

    args = parser.parse_args()

    candidate_df = pd.read_pickle(args.candidate_df_path)

    logger.info(f"Loaded candidate_df with {len(candidate_df)} rows")

    create_query_mol_spec_df(candidate_df, Path(args.output_dir))
    logger.info(f"Saved query outputs to {args.output_dir}")
