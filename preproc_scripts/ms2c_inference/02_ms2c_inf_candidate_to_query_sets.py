import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

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


def build_query_spec_df(row: pd.Series, size) -> pd.DataFrame:
    """Build a per-query spec_df from a single candidate_df row."""
    pass


def build_query_mol_df(candidate_smiles_list: list[str]) -> pd.DataFrame:
    """Build a per-query mol_df from candidate_smiles_list."""

    pass


def create_query_mol_spec_df(candidate_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for col in REQUIRED_CANDIDATE_COLS:
        if col not in candidate_df.columns:
            raise ValueError(f"Missing required column in candidate_df: {col}")

    mol_cache: dict[tuple[int, ...], str] = {}
    index_records: list[dict[str, Any]] = []

    for i, row in candidate_df.iterrows():
        query_id = row["query_spec_id"]
        if pd.isna(query_id):
            query_id = f"no_spec_{i}"

        query_spec_df = build_query_spec_df(row)
        query_spec_path = output_dir / f"spec_df_query_{query_id}.pkl.gz"
        query_spec_df.to_pickle(query_spec_path)

        # mol_ids = [mid for mid in row["candidate_mol_id_list"] if mid is not None]
        # if pd.notna(row["query_mol_id"]):
        #    mol_ids.append(int(row["query_mol_id"]))
        # mol_ids = sorted(set(mol_ids))

        if not mol_ids:
            raise ValueError(f"No mol_id values found for query_id={query_id}")

        mol_key = tuple(mol_ids)
        if mol_key in mol_cache:
            mol_path = mol_cache[mol_key]
        else:
            mol_subset = get_mol_subset(mol_df, mol_ids)
            mol_path = str(output_dir / f"mol_df_set_{len(mol_cache)}.pkl.gz")
            mol_subset.to_pickle(mol_path)
            mol_cache[mol_key] = mol_path

        index_records.append(
            {
                "query_id": query_id,
                "spec_df_path": str(query_spec_path),
                "mol_df_path": mol_path,
            }
        )

    index_path = output_dir / "query_index.json"
    with index_path.open("w") as f:
        json.dump(index_records, f, indent=2)


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
