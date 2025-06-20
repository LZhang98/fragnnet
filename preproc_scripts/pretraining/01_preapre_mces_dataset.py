import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import h5py
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series
from scipy.spatial.distance import squareform

def get_np_memory_usage(array):
	memory_usage = array.nbytes
	# Convert to megabytes
	memory_usage_mb = memory_usage / (1024 ** 2)
	print(f"Memory usage: {memory_usage} bytes ({memory_usage_mb:.2f} MB)")

def process_mol_df(data_fp, mol_df_fp):
	# Existing function unchanged
	with np.load(data_fp, allow_pickle=True) as mces_data:
		smiles = mces_data['smiles']
  
		# Create mol_df from smiles using data_utils functions
		mol_df = pd.DataFrame({'smiles_org': smiles})
		mol_df.loc[:,"idx"] = range(len(mol_df))

		mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles_org"],data_utils.mol_from_smiles)
		mol_df.loc[:,"smiles"] = par_apply_series(mol_df["smiles_org"],data_utils.mol_to_smiles)
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
		mol_df.drop(columns=["mol"], inplace=True)
		mol_df.dropna(subset=["smiles", "smiles_org", "inchikey_s", "scaffold", "formula", "inchi"])
		mol_df = mol_df.reset_index(drop=True)
		mol_df.to_pickle(mol_df_fp)

def process_mces(data_fp, mces_out_fp):
	"""
	Process MCES data and save as HDF5 file with dataset of shape [N, 3] where
	each row is [mol_idx_1, mol_idx_2, mces_value] with mces_value as float16.
	"""
	with np.load(data_fp, allow_pickle=True) as mces_data, h5py.File(mces_out_fp, 'a') as h5f:
		mces_values = mces_data['mces']
		num_smiles = len(mces_data['smiles'])
	
	# Create or open HDF5 file with append mode
	#with h5py.File(mces_out_fp, 'a+') as h5f:
		# Check if dataset exists, create if not
		if 'mces_pairs' not in h5f:
			# Create dataset with unlimited first dimension (can grow)
			# and 3 columns for [mol_idx_1, mol_idx_2, mces_value]
			dt = np.dtype([('mol_idx_1', np.int32), ('mol_idx_2', np.int32), ('mces_value', np.float16)])
			dataset = h5f.create_dataset('mces_pairs', 
										 compression='gzip',
										 shape=(0,), 
										 maxshape=(None,),
										 dtype=dt)
		else:
			dataset = h5f['mces_pairs']
		
		# Process MCES values and append to dataset in batches
		batch_size = 1000  # Adjust based on memory constraints
		batch_data = []
		
		mces_idx = 0
		total_saved = 0
		
		for i in tqdm(range(0, num_smiles)):
			for j in tqdm(range(i+1, num_smiles), leave=False):
				if mces_values[mces_idx] > 0:
					# Add to batch
					batch_data.append((i, j, np.float16(mces_values[mces_idx])))
					
					# If batch is full, append to HDF5
					if len(batch_data) >= batch_size:
						# Convert to structured array
						batch_array = np.array(batch_data, dtype=dt)
						
						# Resize dataset to accommodate new data
						current_size = dataset.shape[0]
						new_size = current_size + len(batch_array)
						dataset.resize((new_size,))
						
						# Write batch to dataset
						dataset[current_size:new_size] = batch_array
						
						total_saved += len(batch_data)
						batch_data = []
				
				mces_idx += 1
		
		# Save any remaining data in the last batch
		if batch_data:
			batch_array = np.array(batch_data, dtype=dt)
			current_size = dataset.shape[0]
			new_size = current_size + len(batch_array)
			dataset.resize((new_size,))
			dataset[current_size:new_size] = batch_array
			total_saved += len(batch_data)
		
		print(f"Total MCES pairs saved: {total_saved}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Filter MCES data, save index pairs and mol_df with properties.")
	parser.add_argument(
		"--data_fp",
		type=str,
		default="./data/mces-data/mces_distances.npz",
		help="Path to the MCES .npz file"
	)
	parser.add_argument(
		"--mces_out_fp",
		type=str,
		default="./data/mces-data/mces_pairs.h5",  # Changed to h5 extension
		help="Output .h5 file for filtered MCES index pairs"
	)
	parser.add_argument(
		"--mol_df_fp",
		type=str,
		default="./data/mces-data/mol_df.pkl",
		help="Output .pkl file for mol_df"
	)
	args = parser.parse_args()
	# process_mces(args.data_fp, args.mces_out_fp)

	# process_mol_df(args.data_fp, args.mol_df_fp)
	# mol_df = pd.read_pickle(args.mol_df_fp)
	# print(mol_df.head())