import argparse
import os
import pandas as pd
import h5py
from tqdm import tqdm

def create_input_data(query_smiles, base_smiles):
	"""
	Create input data for the model by sampling pairs from mol_df based on bio_20k_df.

	Args:
		mol_df (pd.DataFrame): The DataFrame containing molecular data.
		bio_20k_df (pd.DataFrame): The DataFrame containing biological data.

	Returns:
		list: A list of unique (row_idx1, row_idx2) pairs.
	"""
	indices = []
	num_base_smiles = len(base_smiles)
	num_query_smiles = len(query_smiles)
	#print(f"Number of base SMILES: {num_base_smiles}, Number of query SMILES: {num_query_smiles}")
	for i in range(num_query_smiles):
		for j in range(num_base_smiles):
			indices.append((i + num_base_smiles, j))
		#indices.extend([(i + num_base_smiles, j) for j in range(num_base_smiles)])

	return indices

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
	print(f"Creating HDF5 dataset at {output_fp} with {len(smiles)} SMILES and {len(computation_indices)} computation indices.")
	with h5py.File(output_fp, 'w') as h5f:
		# Create a dataset for SMILES strings
		h5f.create_dataset('smiles', data=[s.encode('utf-8') for s in smiles], dtype=h5py.string_dtype())
		# Create a dataset for computation indices
		h5f.create_dataset('computation_indices', data=computation_indices, dtype='int32')

	# print(f"HDF5 dataset created at {output_fp}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Filter MCES data, save index pairs and mol_df with properties.")
	parser.add_argument(
		"--bio_20k_csv_fp",
		type=str,
		default="./data/mces-data/biostructures_20k.csv",
		help="Path to the bio_20k CSV file, this is the one from https://github.com/boecker-lab/myopic-mces-data/blob/main/biostructures_20k.csv "
	)
	parser.add_argument(
		"--outlier_indices_fp",
		type=str,
		default="./data/mces-data/biostructures_20k_outlier_indices.txt",
		help="Path to the outlier indices file"
	)
	parser.add_argument(
		"--mol_df_fp",
		type=str,
		default="./data/proc/nist23/mol_df.pkl",
		help="preprocessed fragnnet mol_df file"
	)
	parser.add_argument(
		"--hdf5_output_dp",
		type=str,
		default="./data/mces-data/nist23_inputs",
		help="Output directory for HDF5 files"
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=2000,
		help="Batch size for processing query SMILES (default: 2000)"
	)

	args = parser.parse_args()
	print(f"Loading bio_20k CSV file {args.bio_20k_csv_fp}")
	bio_20k_df = pd.read_csv(args.bio_20k_csv_fp)	
	print("remove outlier biomolecular structures")
	# for biomolecular structures, do not consider outlier clusters
	outlier_indices = [int(l.strip()) for l in open(args.outlier_indices_fp)]
	base_smiles = [l.strip() for i, l in enumerate(bio_20k_df.smiles.tolist()) 
						if i not in outlier_indices]
	print(f"Removed {len(outlier_indices)} outlier biomolecular structures")
	print(f"Remaining biomolecular structures: {len(base_smiles)}")

	mol_df = pd.read_pickle(args.mol_df_fp)
	query_smiles = mol_df['smiles'].tolist()
	print(f"Loaded {len(query_smiles)} query SMILES from {args.mol_df_fp}")
 
	os.makedirs(args.hdf5_output_dp, exist_ok=True)
	batch_size = args.batch_size
	for i in tqdm(range(0, len(query_smiles), batch_size), desc=f"Processing batches (size {batch_size})"):

		compute_indices = create_input_data(query_smiles[i:i + batch_size], base_smiles)
		sampled_pairs_batch = add_computation_indices(compute_indices)
		output_fp = os.path.join(args.hdf5_output_dp, f"mces_input_{i // batch_size + 1}.h5")
		#print(len(sampled_pairs_batch), len(base_smiles), len(query_smiles[i:i + batch_size]))
		# Create HDF5 dataset for the current batch
		dataset_smiles = base_smiles + query_smiles[i:i + batch_size]
		#print(query_smiles[i:i + batch_size])
		#print(f"Creating dataset with {len(dataset_smiles)} SMILES")
		create_hdf5_dataset( output_fp,
			dataset_smiles,
			sampled_pairs_batch
		)