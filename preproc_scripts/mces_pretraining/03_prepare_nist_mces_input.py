import argparse
import pandas as pd
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series
from fragnnet.utils.formula_utils import parse_formula
from fragnnet.utils.frag_utils import ELEMENTS
import h5py
import random
from tqdm import tqdm

def sample_pairs(mol_df, num_pairs, max_attempts=None):
    """
    Randomly sample pairs of row indices from mol_df and deduplicate them.

    Args:
        mol_df (pd.DataFrame): The DataFrame containing molecular data.
        num_pairs (int): The number of unique pairs to sample.
        max_attempts (int, optional): Maximum number of attempts to sample pairs.

    Returns:
        list: A list of unique (row_idx1, row_idx2) pairs.
    """
    num_rows = len(mol_df)
    if num_rows < 2:
        raise ValueError("Not enough rows in mol_df to create pairs.")

    sampled_pairs = set()  # Use a set to ensure uniqueness
    attempts = 0  # Track the number of attempts

    # Use tqdm to show progress
    with tqdm(total=num_pairs, desc="Sampling pairs") as pbar:
        while len(sampled_pairs) < num_pairs and (max_attempts is None or attempts < max_attempts):
            # Randomly sample two distinct indices
            idx1, idx2 = random.sample(range(num_rows), 2)
            attempts += 1

            # Ignore pairs that are too different in terms of number of atoms and bonds
            if abs(mol_df.iloc[idx1]['num_atoms'] - mol_df.iloc[idx2]['num_atoms']) >= 10:
                continue
            if abs(mol_df.iloc[idx1]['num_bonds'] - mol_df.iloc[idx2]['num_bonds']) >= 10:
                continue

            # Add the pair in a sorted order to avoid duplicates like (1, 2) and (2, 1)
            if tuple(sorted((idx1, idx2))) not in sampled_pairs:
                sampled_pairs.add(tuple(sorted((idx1, idx2))))
                pbar.update(1)  # Update the progress bar

    return list(sampled_pairs)

def add_computation_indices(sampled_pairs):
	"""
	Add indices to the sampled pairs.
	Args:
		sampled_pairs (_type_): _description_

	Returns:
		list: _description_
	"""
 
	rev_list = []
	for idx,pair in enumerate(sampled_pairs):
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
	with h5py.File(output_fp, 'w') as h5f:
		# Create a dataset for SMILES strings
		h5f.create_dataset('smiles', data=[s.encode('utf-8') for s in smiles], dtype=h5py.string_dtype())
		
  		# Create a dataset for computation indices
		h5f.create_dataset('computation_indices', data=computation_indices, dtype='int32')
	
	print(f"HDF5 dataset created at {output_fp}")
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Filter MCES data, save index pairs and mol_df with properties.")
	parser.add_argument(
		"--bio_20k_csv_fp",
		type=str,
		default="./data/mces-data/biostructures_20k.csv",
		help="Path to the bio_20k CSV file, this is the one from https://github.com/boecker-lab/myopic-mces-data/blob/main/biostructures_20k.csv "
	)
	parser.add_argument(
		"--mol_df_fp",
		type=str,
		default="./data/mces-data/biostructures_mol_df.pkl",
		help="preprocessed mol_df file"
	)
	parser.add_argument(
		"--hdf5_output_fp",
		type=str,
		default="./data/mces-data/biostructures_sampled.h5",
		help="Output .pkl file for mol_df"
	)
	parser.add_argument(
		"--pair_per_file",
		type=int,
		default=10_000_000,
		help="Number of unique pairs to sample"
	)
	args = parser.parse_args()
	
	mol_df = pd.read_pickle(args.mol_df_fp)
  
	# Randomly sample and deduplicate pairs
	sampled_pairs = sample_pairs(mol_df, args.num_pairs, max_attempts=args.num_pairs * 5)
	print(f"Sampled {len(sampled_pairs)} unique pairs of indices.")
	output_fp = args.hdf5_output_fp
	output_fp, ext = output_fp.rsplit('.', 1)
 
	print(f"num of smiles: {len(mol_df['smiles'])}")
	for i in range(0, len(sampled_pairs), args.pair_per_file):
		print(f"Processing pairs {i} to {min(i + args.pair_per_file, len(sampled_pairs))}")
		sampled_pairs_batch = sampled_pairs[i:i + args.pair_per_file]
		sampled_pairs_batch = add_computation_indices(sampled_pairs_batch)
		
		# Create HDF5 dataset for the current batch
		create_hdf5_dataset( output_fp + f"_{i // args.pair_per_file + 1}.{ext}",
			mol_df['smiles'].tolist(), 
			sampled_pairs_batch
		)