import argparse
import os

import h5py
import pandas as pd

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import compute_mol_df_props, par_apply_series
from fragnnet.utils.formula_utils import ELEMENTS, parse_formula

# from scipy.spatial.distance import squareform


def create_mol_df(smiles_to_mol_id):
    mol_df = pd.DataFrame()
    mol_df["smiles_org"] = smiles_to_mol_id.keys()
    mol_df["mol_id"] = smiles_to_mol_id.values()

    mol_df.loc[:, "mol"] = par_apply_series(
        mol_df["smiles_org"], lambda s: data_utils.mol_from_smiles(s, ml_standardize=False)
    )
    mol_df.loc[:, "smiles"] = par_apply_series(mol_df["mol"], data_utils.mol_to_smiles)
    compute_mol_df_props(mol_df)
    mol_df.loc[:, "num_out_set_elements"] = par_apply_series(
        mol_df["formula"], (lambda x: len(set(list(parse_formula(x).keys())) - set(ELEMENTS)))
    )

    mol_df.dropna(subset=["smiles", "inchikey_s", "scaffold", "formula", "inchi"])

    return mol_df


def process_single_h5_file(data_fp):
    """Process a single h5 file and return UMAP projection data or error info"""

    print(f"Starting to process {data_fp}")

    # Read the h5 file and convert to MCES format
    with h5py.File(data_fp, "r") as f:
        # Check if required datasets exist
        if "mces" not in f:
            raise FileNotFoundError(f"Missing 'mces' dataset in {data_fp}")

        if "smiles" not in f:
            raise FileNotFoundError(f"Missing 'smiles' dataset in {data_fp}")

        if "computation_indices" not in f:
            raise FileNotFoundError(f"Missing 'computation_indices' dataset in {data_fp}")

        # Check dataset shapes to avoid memory issues

        smiles_shape = f["smiles"].shape
        mces_shape = f["mces"].shape
        comp_shape = f["computation_indices"].shape
        print(
            f"Dataset shapes for {data_fp}: smiles={smiles_shape}, mces={mces_shape}, comp_indices={comp_shape}"
        )

        # Extract data from h5 file with memory-efficient approach
        smiles_data = f["smiles"][:]
        mces_data = f["mces"][:]
        computation_indices = f["computation_indices"][:]

        smiles_list_left = []
        smiles_list_right = []
        mces_values = []

        for compute_setup, mces in zip(computation_indices, mces_data):
            smiles_list_left.append(smiles_data[compute_setup[1]].decode())
            smiles_list_right.append(smiles_data[compute_setup[2]].decode())
            mces_values.append(mces)

        # Create DataFrame with the extracted data
        h5_distances_df = pd.DataFrame(
            {"smiles1": smiles_list_left, "smiles2": smiles_list_right, "mces": mces_values}
        )
    return h5_distances_df


def get_mces_dfs(data_dp):
    h5_files = [os.path.join(data_dp, f) for f in os.listdir(data_dp) if f.endswith(".h5")]
    h5_distances_df = pd.concat([process_single_h5_file(f) for f in h5_files], ignore_index=True)
    smiles = h5_distances_df["smiles1"].tolist() + h5_distances_df["smiles2"].tolist()
    unique_smiles = list(set(smiles))
    smiles_to_mol_id = {smiles: idx for idx, smiles in enumerate(unique_smiles)}
    h5_distances_df["mol_id_1"] = h5_distances_df["smiles1"].map(smiles_to_mol_id)
    h5_distances_df["mol_id_2"] = h5_distances_df["smiles2"].map(smiles_to_mol_id)
    mol_df = create_mol_df(smiles_to_mol_id)

    return h5_distances_df, mol_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter MCES data, save index pairs and mol_df with properties."
    )
    parser.add_argument(
        "--data_dp",
        type=str,
        default="./data/mces-data/biostructures_sampled_mces",
        help="Path to the MCES .h5 file",
    )
    parser.add_argument(
        "--mces_proc",
        type=str,
        default="./data/proc/mces",  # Changed to pkl extension
        help="Output .pkl file for filtered MCES index pairs",
    )

    args = parser.parse_args()

    # mol_df = pd.read_pickle(args.mol_df_fp)
    h5_distances_df, mol_df = get_mces_dfs(args.data_dp)
    print(f"Total MCES pairs before filtering: {len(h5_distances_df)}")
    h5_distances_df.to_pickle(os.path.join(args.mces_proc, "mces_pairs.pkl"))
    print(f"Saving mol_df to {os.path.join(args.mces_proc, 'mol_df.pkl')}")
    mol_df.to_pickle(os.path.join(args.mces_proc, "mol_df.pkl"))
