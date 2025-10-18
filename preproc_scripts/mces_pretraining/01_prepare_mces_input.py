import argparse
import os
import pandas as pd
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series
from fragnnet.utils.formula_utils import parse_formula
from fragnnet.utils.frag_utils import ELEMENTS
import h5py
import random
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
import multiprocessing

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
	mol_df['mol_id'] = mol_df.index
	return mol_df

def sample_pairs_chunk(all_indices, num_atoms_arr, num_bonds_arr, num_pairs_target, seed):
	"""
	Sample pairs in a single worker process.
	
	Args:
		all_indices: List of all molecule indices to sample from
		num_atoms_arr: Array of num_atoms for quick lookup
		num_bonds_arr: Array of num_bonds for quick lookup
		num_pairs_target: Target number of pairs for this chunk
		seed: Random seed for reproducibility (ensures different workers sample different pairs)
		
	Returns:
		set: Set of sampled pairs (as tuples)
	"""
	random.seed(seed)
	np.random.seed(seed)
	
	num_rows = len(all_indices)
	if num_rows < 2:
		return set()
	
	sampled_pairs = set()
	attempts = 0
	max_attempts = num_pairs_target * 5  # Limit attempts per chunk
	
	while len(sampled_pairs) < num_pairs_target and attempts < max_attempts:
		# Randomly sample two distinct indices
		idx1, idx2 = random.sample(all_indices, 2)
		attempts += 1
		
		# Filter by atom and bond differences
		if abs(num_atoms_arr[idx1] - num_atoms_arr[idx2]) >= 10:
			continue
		if abs(num_bonds_arr[idx1] - num_bonds_arr[idx2]) >= 10:
			continue
		
		# Add the pair in sorted order
		# make sure idx 1 < idx 2
		if idx1 < idx2:
			pair = (idx1, idx2)
		else:
			pair = (idx2, idx1)
		sampled_pairs.add(pair)
	#print(f"Sampled pairs: {attempts} / {max_attempts}", end='\r')
	return sampled_pairs

def sample_pairs(mol_df, num_pairs, n_jobs=-1):
	"""
	Randomly sample pairs of row indices from mol_df in parallel and deduplicate them.

	Args:
		mol_df (pd.DataFrame): The DataFrame containing molecular data.
		num_pairs (int): The number of unique pairs to sample.
		n_jobs (int): Number of parallel jobs (-1 uses all cores).
		max_attempts (int, optional): Maximum number of attempts to sample pairs (deprecated, use multiplier internally).

	Returns:
		list: A list of unique (row_idx1, row_idx2) pairs.
	"""
	num_rows = len(mol_df)
	if num_rows < 2:
		raise ValueError("Not enough rows in mol_df to create pairs.")

	# Determine number of chunks
	n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
	pairs_per_chunk = num_pairs // n_jobs
 
	print(f"Sampling {num_pairs} pairs from {num_rows} molecules using {n_jobs} jobs...")
	
	# Precompute arrays for fast lookup
	num_atoms_arr = mol_df['num_atoms'].values
	num_bonds_arr = mol_df['num_bonds'].values
	all_indices = list(range(num_rows))
	
	print(f"Splitting work into {n_jobs} chunks ({pairs_per_chunk} pairs per chunk)...")
	# Run parallel sampling
	results = Parallel(n_jobs=n_jobs, backend='loky', verbose=51)(
		delayed(sample_pairs_chunk)(
			all_indices,
			num_atoms_arr,
			num_bonds_arr,
			pairs_per_chunk,
			seed = 42 * i  # Different seed per chunk
		)
		for i in range(n_jobs)
	)
	
	# Merge all sampled pairs and deduplicate
	print("Merging and deduplicating results from all chunks...")
	sampled_pairs = set()
	total_pairs_before_dedup = 0
	for chunk_pairs in tqdm(results, desc="Merging chunks"):
		total_pairs_before_dedup += len(chunk_pairs)
		sampled_pairs.update(chunk_pairs)
	
	num_duplicates = total_pairs_before_dedup - len(sampled_pairs)
	print(f"Total pairs from all chunks: {total_pairs_before_dedup}")
	print(f"Unique pairs after deduplication: {len(sampled_pairs)}")
	print(f"Duplicates removed: {num_duplicates} ({100 * num_duplicates / total_pairs_before_dedup:.2f}%)")
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
		"--update_cache",
		action='store_true',
		default=False,
		help="Whether to update the mol_df cache"
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
		default=1_0000_000,
		help="Number of unique pairs to sample"
	)
	parser.add_argument(
		"--pair_per_file",
		type=int,
		default=1_000_000,
		help="Number of unique pairs to sample"
	)
	parser.add_argument(
		"--n_jobs",
		type=int,
		default=-1,
		help="Number of parallel jobs for sampling (-1 uses all cores)"
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
	sampled_pairs = sample_pairs(mol_df, args.num_pairs, n_jobs=args.n_jobs)
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
