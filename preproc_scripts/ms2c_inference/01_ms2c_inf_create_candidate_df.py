import argparse
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd
import fragnnet.utils.data_utils as data_utils
import numpy as np
import os
from multiprocessing import Pool
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


def load_split(path):
    splits=os.listdir(path)
    splits=sorted(splits, key=lambda x: int(x.replace("cv", "")))
    print("splits founded:", splits)
    ret_ids=[]
    for sp in splits:
        test_mol_id=set(pd.read_csv(os.path.join(path, sp, "test_ids.csv"))["mol_id"])
        ret_ids.append(test_mol_id)
    return ret_ids


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
    output_path:str,
    candidate_mol_ids:set
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
        if mol_id not in candidate_mol_ids:
            continue
        mol_smiles_to_id[smiles] = int(mol_id)

    spec_records: list[dict[str, Any]] = []
    mol_records: list[dict[str, Any]] = []
    map_smi_to_mol_id={}
    map_test_spec_spec_id={}
    for query_smiles in candidates_dict:
        if query_smiles not in mol_smiles_to_id:
            continue
        candidate_smiles_list=candidates_dict[query_smiles]
        assert query_smiles in candidate_smiles_list
        assert len(candidate_smiles_list)==len(set(candidate_smiles_list))
        if not isinstance(candidate_smiles_list, list):
            raise ValueError(f"Expected list of candidates for {query_smiles}")
        if query_smiles not in mol_smiles_to_id:
            raise ValueError(f"Query SMILES not found in mol_df: {query_smiles}")

        query_mol_id = mol_smiles_to_id[query_smiles]
        query_spec_rows = spec_df[spec_df["mol_id"] == query_mol_id]
        if query_spec_rows.empty:
            raise ValueError(f"No spec_df rows found for query mol_id={query_mol_id}")

        num_candidate_smiles = len(candidate_smiles_list)
        logger.info(
            f"Processing query_smiles: {query_smiles} with {num_candidate_smiles} candidates"
        )
        
        for smi in candidate_smiles_list:
            if smi in map_smi_to_mol_id:
                continue
            mol=data_utils.mol_from_smiles(smi)
            assert not mol is None, smi
            mol_records.append(
                {
                    "mol_id":len(mol_records),
                    "mol":mol,
                    "inchikey_s":data_utils.mol_to_inchikey_s(mol),
                    "scaffold":data_utils.get_murcko_scaffold(mol),
                    "formula":data_utils.mol_to_formula(mol),
                    "inchi":data_utils.mol_to_inchi(mol),
                    "mw":data_utils.mol_to_mol_weight(mol),
                    "exact_mw":data_utils.mol_to_mol_weight(mol),
                    "num_atoms":data_utils.mol_to_num_atoms(mol),
                    "num_bonds":data_utils.mol_to_num_bonds(mol),
                    "charge":data_utils.mol_to_charge(mol),
                    "single_mol":data_utils.check_single_mol(mol),
                    "num_radicals":data_utils.mol_to_num_radicals(mol),
                }
            )
            assert smi not in map_smi_to_mol_id
            map_smi_to_mol_id[smi]=mol_records[-1]["mol_id"]
        for _, spec_row in query_spec_rows.iterrows():
            if np.isnan(spec_row["nce"]) and np.isnan(spec_row["ace"]):
                continue
            for smi in candidate_smiles_list:
                k=f'{smi}_{spec_row["prec_type"]}_{spec_row["inst_type"]}_{spec_row["ace"]}_{spec_row["nce"]}_{spec_row["ace_extra_1"]}_{spec_row["nce_extra_1"]}_{spec_row["ace_extra_2"]}'
                if k in map_test_spec_spec_id:
                    spec_records[map_test_spec_spec_id[k]]["gt_spec_id"].append(spec_row["spec_id"])
                else:
                    spec_records.append(
                        {
                            "spec_id": len(spec_records),
                            "mol_id": map_smi_to_mol_id[smi],
                            "peaks": spec_row["peaks"],
                            "prec_type": spec_row["prec_type"],
                            "inst_type": spec_row["inst_type"],
                            "ace": spec_row["ace"],
                            "nce": spec_row["nce"],
                            "ace_extra_1": spec_row["ace_extra_1"],
                            "nce_extra_1": spec_row["nce_extra_1"],
                            "ace_extra_2": spec_row["ace_extra_2"],
                            "nce_extra_2": spec_row["nce_extra_2"],
                            "gt_spec_id": [spec_row["spec_id"]],
                            "spec_type":spec_row["spec_type"],
                            "dset":spec_row["dset"],
                            "ion_mode":spec_row["ion_mode"],
                            "group_id": len(spec_records)
                        }
                        )
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_spec_records=pd.DataFrame(spec_records)
    save_spec_records.to_pickle(os.path.join(output_path, "spec_df.pkl"))
    pd.DataFrame(mol_records).to_pickle(os.path.join(output_path, "mol_df.pkl"))
    print(f"saving {len(spec_records)} spec_records to {output_path}")

    # generate split
    pd.DataFrame.from_dict({
        "mol_id":save_spec_records["mol_id"],
        "spec_id":save_spec_records["spec_id"],
        "group_id":save_spec_records["spec_id"],
    }).to_csv(os.path.join(output_path, "test_ids.csv"), index=False)


if __name__ == "__main__":
    # The following code reformat the json file to similar format as training
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
        # required=True,
        default="data/ms2c/nps/nps_candidates/nps_nist23/nps_nist23_clm_formula.json",
        help="Path to JSON file containing query SMILES and candidate SMILES lists",
    )

    parser.add_argument(
        "--spec_df_path",
        type=str,
        # required=True,
        default="data/proc/nps_nist23/spec_df.pkl",
        help="path to spec_df.pkl.gz file created by 02_prepare_spec_df.py which has specid and metadata",
    )
    parser.add_argument(
        "--mol_df_path",
        type=str,
        # required=True,
        default="data/proc/nps_nist23/mol_df.pkl",
        help="path to mol_df.pkl.gz file created by 02_prepare_spec_df.py which has molid and metadata",
    )
    parser.add_argument(
        "--split_path",
        type=str,
        required=True,
        help="path to training splits",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        # required=True,
        default="data/ms2c/candidates/nps_nist23_candidate_df.pkl.gz",
        help="Path to output candidate_df.pkl.gz",
    )
    args = parser.parse_args()

    # Load JSON file
    candidates_dict = load_candidates_dict(args.json_file, logger)

    # Load split file, to consistant with cross-validation
    splits=load_split(args.split_path)
    
    # logger.info(f"Total unique SMILES collected: {len(all_smiles)}")
    query_smiles_list = list(candidates_dict.keys())
    logger.info(f"Total query SMILES collected: {len(query_smiles_list)}")

    mol_df = pd.read_pickle(args.mol_df_path)
    logger.info("Loaded mol_df with %d molecules", len(mol_df))
    spec_df = pd.read_pickle(args.spec_df_path)
    logger.info("Loaded spec_df with %d spectra", len(spec_df))
   
    inp=[]
    for i,s in enumerate(splits):
        inp.append([candidates_dict, spec_df, mol_df, logger, os.path.join(args.output_path, f"{i}"), s])
        
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    pool=Pool(10)
    pool.starmap(build_candidate_df, inp)

    # Create output directory if it doesn't exist
    logger.info("Saved candidate_df to %s", args.output_path)
