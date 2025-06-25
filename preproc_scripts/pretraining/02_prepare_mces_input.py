import argparse
import pandas as pd
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series
from fragnnet.utils.formula_utils import parse_formula
from fragnnet.utils.frag_utils import ELEMENTS
import h5py
import random
from tqdm import tqdm

def process_mol_df(data_fp):
	mol_df = pd.read_csv(data_fp)
	mol_df['smiles_org'] = mol_df['smiles']

	mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles_org"], lambda s: data_utils.mol_from_smiles(s, ml_standardize=False))
	mol_df = mol_df[mol_df["mol"].notna()]
	print(f"> Filtered mol_df shape after mol_from_smiles: {mol_df.shape}")
	mol_df.loc[:,"smiles"] = par_apply_series(mol_df["mol"], data_utils.mol_to_smiles)
	mol_df.loc[:,"inchikey_s"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchikey_s)
	mol_df.loc[:,"scaffold"] = par_apply_series(mol_df["mol"],data_utils.get_murcko_scaffold)
	mol_df.loc[:,"formula"] = par_apply_series(mol_df["mol"],data_utils.mol_to_formula)
	mol_df.loc[:,"inchi"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchi)
	mol_df.loc[:,"mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=False))
	mol_df.loc[:,"exact_mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True))
	mol_df.loc[:,"num_atoms"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_atoms)
	mol_df.loc[:,"num_bonds"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_bonds)
	mol_df.loc[:,"charge"] = par_apply_series(mol_df["mol"],data_utils.mol_to_charge)
	mol_df.loc[:,"single_mol"] = par_apply_series(mol_df["mol"],data_utils.check_single_mol)
	mol_df.loc[:,"num_radicals"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_radicals)
	mol_df.loc[:,'num_out_set_elements'] = par_apply_series(mol_df['formula'], (lambda x: len(set(list(parse_formula(x).keys())) - set(ELEMENTS))))

	mol_df.drop(columns=["mol"], inplace=True)
	mol_df.dropna(subset=["smiles", "smiles_org", "inchikey_s", "scaffold", "formula", "inchi"])

	mol_df = mol_df[mol_df['single_mol'] == True]
	print(f"> Filtered mol_df shape after single mol: {mol_df.shape}")
	mol_df = mol_df[mol_df['num_radicals'] == False]
	print(f"> Filtered mol_df shape after num_radicals = 1: {mol_df.shape}")
	mol_df = mol_df[mol_df['charge'] == 0]
	print(f"> Filtered mol_df shape after charge = 0: {mol_df.shape}")
	mol_df = mol_df[mol_df['num_out_set_elements'] == 0]
	print(f"> Filtered mol_df shape after num_out_set_elements = 0: {mol_df.shape}")
	mol_df = mol_df[mol_df['exact_mw'] <= 1600]
	print(f"> Filtered mol_df shape after exact_mw <= 1600 Da: {mol_df.shape}")
	mol_df = mol_df[(mol_df['num_atoms'] > 1 ) & (mol_df['num_atoms'] < 100)]
	print(f"> Filtered mol_df shape after 1 < num_atom < 100: {mol_df.shape}")
	mol_df = mol_df.reset_index(drop=True)
 
	return mol_df

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
		"--input_csv_fp",
		type=str,
		default="./data/mces-data/biostructures.csv",
		help="Output .pkl file for mol_df"
	)
	parser.add_argument(
		"--mol_df_cache_fp",
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
		"--num_pairs",
		type=int,
		default=10,
		help="Number of unique pairs to sample"
	)
	parser.add_argument(
		"--pair_per_file",
		type=int,
		default=10,
		help="Number of unique pairs to sample"
	)
	args = parser.parse_args()
	
	if args.mol_df_cache_fp is None:
		mol_df = process_mol_df(args.input_csv_fp)
	else:
		mol_df = pd.read_pickle(args.mol_df_cache_fp)
  
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