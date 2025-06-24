import argparse
import pandas as pd
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series
from fragnnet.utils.formula_utils import parse_formula
from fragnnet.utils.frag_utils import ELEMENTS
import h5py
import random

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

def sample_and_deduplicate_pairs(mol_df, num_pairs, include_idx = True):
	"""
	Randomly sample pairs of row indices from mol_df and deduplicate them.

	Args:
		mol_df (pd.DataFrame): The DataFrame containing molecular data.
		num_pairs (int): The number of unique pairs to sample.

	Returns:
		list: A list of unique (row_idx1, row_idx2) pairs.
	"""
	num_rows = len(mol_df)
	if num_rows < 2:
		raise ValueError("Not enough rows in mol_df to create pairs.")

	sampled_pairs = set()  # Use a set to ensure uniqueness
	while len(sampled_pairs) < num_pairs:
		# Randomly sample two distinct indices
		idx1, idx2 = random.sample(range(num_rows), 2)
		# Add the pair in a sorted order to avoid duplicates like (1, 2) and (2, 1)
		sampled_pairs.add(tuple(sorted((idx1, idx2))))
  
	rev_list = []
	if include_idx:
		for idx,pair in enumerate(sampled_pairs):
			rev_list.append([idx, pair[0], pair[1]])
	else:
		rev_list = list(sampled_pairs)

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
	#parser.add_argument(
	#	"--mol_df_fp",
	#	type=str,
	#	default="./data/mces-data/biostructures_mol_df.pkl",
	#	help="Output .pkl file for mol_df"
	#)
	parser.add_argument(
		"--hdf5_output_fp",
		type=str,
		default="./data/mces-data/biostructures_sampled.hdf5",
		help="Output .pkl file for mol_df"
	)
	parser.add_argument(
		"--num_pairs",
		type=int,
		default=100_000_000,
		help="Number of unique pairs to sample"
	)
	args = parser.parse_args()
	
	mol_df = process_mol_df(args.input_csv_fp)
	#mol_df.to_pickle(args.mol_df_fp)
	#print(mol_df.head())

	# Randomly sample and deduplicate pairs
	sampled_pairs = sample_and_deduplicate_pairs(mol_df, args.num_pairs, include_idx=True)
	print(f"Sampled {len(sampled_pairs)} unique pairs of indices.")
	print(sampled_pairs[:10])  # Print the first 10 pairs as a preview
 
	create_hdf5_dataset(args.hdf5_output_fp, 
		mol_df['smiles'].tolist(), 
		sampled_pairs
	)