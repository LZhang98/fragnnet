import argparse
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import compute_mol_df_props, par_apply_series
from fragnnet.utils.formula_utils import ELEMENTS, parse_formula


def process_mol_df(data_fp):
    mol_df = pd.read_csv(data_fp)
    mol_df["smiles_org"] = mol_df["smiles"]

    mol_df.loc[:, "mol"] = par_apply_series(
        mol_df["smiles_org"], lambda s: data_utils.mol_from_smiles(s, ml_standardize=False)
    )
    mol_df = mol_df[mol_df["mol"].notna()]
    print(f"> Filtered mol_df shape after mol_from_smiles: {mol_df.shape}")
    mol_df.loc[:, "smiles"] = par_apply_series(mol_df["mol"], data_utils.mol_to_smiles)
    compute_mol_df_props(mol_df)
    mol_df.loc[:, "num_out_set_elements"] = par_apply_series(
        mol_df["formula"], (lambda x: len(set(list(parse_formula(x).keys())) - set(ELEMENTS)))
    )

    mol_df.drop(columns=["mol"], inplace=True)
    mol_df.dropna(subset=["smiles", "smiles_org", "inchikey_s", "scaffold", "formula", "inchi"])

    mol_df = mol_df[mol_df["single_mol"] == True]
    print(f"> Filtered mol_df shape after single mol: {mol_df.shape}")
    mol_df = mol_df[mol_df["num_radicals"] == False]
    print(f"> Filtered mol_df shape after num_radicals = 1: {mol_df.shape}")
    mol_df = mol_df[mol_df["charge"] == 0]
    print(f"> Filtered mol_df shape after charge = 0: {mol_df.shape}")
    mol_df = mol_df[mol_df["num_out_set_elements"] == 0]
    print(f"> Filtered mol_df shape after num_out_set_elements = 0: {mol_df.shape}")
    mol_df = mol_df[mol_df["exact_mw"] <= 1600]
    print(f"> Filtered mol_df shape after exact_mw <= 1600 Da: {mol_df.shape}")
    mol_df = mol_df[(mol_df["num_atoms"] > 1) & (mol_df["num_atoms"] < 100)]
    print(f"> Filtered mol_df shape after 1 < num_atom < 100: {mol_df.shape}")
    mol_df = mol_df.reset_index(drop=True)
    mol_df["mol_id"] = mol_df.index
    return mol_df


def sample_pairs(
    mol_df, num_pairs, random_seed=42, batch_size=100000, max_atom_diff=10, max_bond_diff=10
):
    """
    Randomly sample pairs of row indices from mol_df and deduplicate them.
    Uses batch sampling for much better performance.

    Args:
        mol_df (pd.DataFrame): The DataFrame containing molecular data.
        num_pairs (int): The number of unique pairs to sample.
        random_seed (int): Random seed for reproducibility.
        batch_size (int): Number of candidate pairs to generate per batch.
        max_atom_diff (int): Maximum allowed difference in number of atoms between paired molecules.
        max_bond_diff (int): Maximum allowed difference in number of bonds between paired molecules.

    Returns:
        list: A list of unique (row_idx1, row_idx2) pairs.
    """
    num_rows = len(mol_df)
    if num_rows < 2:
        raise ValueError("Not enough rows in mol_df to create pairs.")

    print(f"Sampling {num_pairs} pairs from {num_rows} molecules with seed={random_seed}")
    print(f"Filter constraints: max_atom_diff={max_atom_diff}, max_bond_diff={max_bond_diff}")

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Precompute arrays for fast lookup
    num_atoms_arr = mol_df["num_atoms"].values
    num_bonds_arr = mol_df["num_bonds"].values

    sampled_pairs = set()
    total_attempts = 0

    # Use tqdm for progress bar
    with tqdm(total=num_pairs, desc="Sampling pairs", unit="pairs") as pbar:
        while len(sampled_pairs) < num_pairs:
            # Generate a batch of candidate pairs all at once (MUCH faster)
            idx1_batch = np.random.randint(0, num_rows, size=batch_size)
            idx2_batch = np.random.randint(0, num_rows, size=batch_size)

            # Filter out pairs where idx1 == idx2
            valid_mask = idx1_batch != idx2_batch
            idx1_batch = idx1_batch[valid_mask]
            idx2_batch = idx2_batch[valid_mask]

            # Apply atom and bond filters using vectorized operations (MUCH faster)
            atom_diff = np.abs(num_atoms_arr[idx1_batch] - num_atoms_arr[idx2_batch])
            bond_diff = np.abs(num_bonds_arr[idx1_batch] - num_bonds_arr[idx2_batch])
            filter_mask = (atom_diff < max_atom_diff) & (bond_diff < max_bond_diff)

            idx1_filtered = idx1_batch[filter_mask]
            idx2_filtered = idx2_batch[filter_mask]

            # Ensure idx1 < idx2 for consistent ordering (vectorized)
            min_idx = np.minimum(idx1_filtered, idx2_filtered)
            max_idx = np.maximum(idx1_filtered, idx2_filtered)

            # Add pairs to set (this deduplicates automatically)
            new_pairs = set(zip(min_idx, max_idx))
            before_count = len(sampled_pairs)
            sampled_pairs.update(new_pairs)
            new_count = len(sampled_pairs) - before_count

            total_attempts += batch_size
            pbar.update(new_count)

            # Update stats
            if len(sampled_pairs) >= num_pairs:
                break

            if total_attempts > 10 * num_pairs:
                print(
                    "Warning: High rejection rate, consider adjusting filters or increasing dataset size."
                )
                break

            rejection_rate = (total_attempts - len(sampled_pairs)) / total_attempts * 100
            pbar.set_postfix(
                {"attempts": f"{total_attempts:,}", "reject%": f"{rejection_rate:.1f}"}
            )

    rejection_rate = (total_attempts - len(sampled_pairs)) / total_attempts * 100
    print(
        f"Sampled {len(sampled_pairs)} unique pairs after {total_attempts:,} attempts (rejection rate: {rejection_rate:.1f}%)"
    )

    # Return exactly num_pairs (trim if we overshot)
    return list(sampled_pairs)[:num_pairs]


def add_computation_indices(sampled_pairs):
    """
    Add indices to the sampled pairs.
    Args:
        sampled_pairs (_type_): _description_

    Returns:
        list: _description_
    """

    rev_list = []
    for idx, pair in enumerate(sampled_pairs):
        rev_list.append([idx, pair[0], pair[1]])
    return rev_list


def create_hdf5_dataset(output_fp, smiles, computation_indices):
    """
    Create an HDF5 dataset with SMILES strings and computation indices.

    Args:
        output_fp (str): Path to the output HDF5 file.
        smiles (list of str): List of SMILES strings.
        computation_indices (list of lists): List of computation indices, where each sublist
                                             represents [row_idx1, row_idx2, computation_idx].
    """
    with h5py.File(output_fp, "w") as h5f:
        # Create a dataset for SMILES strings
        h5f.create_dataset(
            "smiles", data=[s.encode("utf-8") for s in smiles], dtype=h5py.string_dtype()
        )

        # Create a dataset for computation indices
        h5f.create_dataset("computation_indices", data=computation_indices, dtype="int32")

    print(f"HDF5 dataset created at {output_fp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter MCES data, save index pairs and mol_df with properties."
    )
    parser.add_argument(
        "--input_csv_fp",
        type=str,
        default="./data/mces-data/biostructures.csv",
        help="Output .pkl file for mol_df",
    )
    parser.add_argument(
        "--mol_df_cache_fp",
        type=str,
        default="./data/mces-data/biostructures_mol_df.pkl",
        help="preprocessed mol_df file",
    )
    parser.add_argument(
        "--update_cache",
        action="store_true",
        default=False,
        help="Whether to update the mol_df cache",
    )
    parser.add_argument(
        "--hdf5_output_fp",
        type=str,
        default="./data/mces-data/biostructures_sampled.h5",
        help="Output .pkl file for mol_df",
    )
    parser.add_argument(
        "--num_pairs", type=int, default=1_0000_000, help="Number of unique pairs to sample"
    )
    parser.add_argument(
        "--pair_per_file", type=int, default=1_000_000, help="Number of unique pairs to sample"
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_atom_diff",
        type=int,
        default=10,
        help="Maximum allowed difference in number of atoms between paired molecules",
    )
    parser.add_argument(
        "--max_bond_diff",
        type=int,
        default=10,
        help="Maximum allowed difference in number of bonds between paired molecules",
    )
    args = parser.parse_args()

    # Validate input file exists if we need to process it
    if not os.path.exists(args.mol_df_cache_fp) or args.update_cache:
        if not os.path.exists(args.input_csv_fp):
            raise FileNotFoundError(f"Input CSV file not found: {args.input_csv_fp}")
        print(f"Processing molecules from {args.input_csv_fp}...")
        mol_df = process_mol_df(args.input_csv_fp)
        print(f"Saving mol_df cache to {args.mol_df_cache_fp}...")
        mol_df.to_pickle(args.mol_df_cache_fp)
    else:
        print(f"Loading mol_df from cache: {args.mol_df_cache_fp}...")
        mol_df = pd.read_pickle(args.mol_df_cache_fp)

    # Randomly sample and deduplicate pairs
    sampled_pairs = sample_pairs(
        mol_df,
        args.num_pairs,
        random_seed=args.random_seed,
        max_atom_diff=args.max_atom_diff,
        max_bond_diff=args.max_bond_diff,
    )
    print(f"Sampled {len(sampled_pairs)} unique pairs of indices.")
    output_fp = args.hdf5_output_fp
    output_fp, ext = output_fp.rsplit(".", 1)

    print(f"num of smiles: {len(mol_df['smiles'])}")
    for i in range(0, len(sampled_pairs), args.pair_per_file):
        print(f"Processing pairs {i} to {min(i + args.pair_per_file, len(sampled_pairs))}")
        sampled_pairs_batch = sampled_pairs[i : i + args.pair_per_file]
        sampled_pairs_batch = add_computation_indices(sampled_pairs_batch)

        # Create HDF5 dataset for the current batch
        create_hdf5_dataset(
            output_fp + f"_{i // args.pair_per_file + 1}.{ext}",
            mol_df["smiles"].tolist(),
            sampled_pairs_batch,
        )
