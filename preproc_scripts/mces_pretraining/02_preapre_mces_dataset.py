import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series
from fragnnet.utils.formula_utils import parse_formula
from fragnnet.utils.frag_utils import ELEMENTS
#from scipy.spatial.distance import squareform

def get_np_memory_usage(array):
	memory_usage = array.nbytes
	# Convert to megabytes
	memory_usage_mb = memory_usage / (1024 ** 2)
	print(f"Memory usage: {memory_usage} bytes ({memory_usage_mb:.2f} MB)")

def get_mol_df(data_fp):

	with np.load(data_fp, allow_pickle=True) as mces_data:
		smiles = mces_data['smiles']
  
		# Create mol_df from smiles using data_utils functions
		mol_df = pd.DataFrame({'smiles_org': smiles})
		mol_df.loc[:,"mol_id"] = range(len(mol_df))

		mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles_org"], lambda s: data_utils.mol_from_smiles(s, ml_standardize=False))
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

		mol_df = mol_df[(mol_df['single_mol'] == True )]
		print(f"> Filtered mol_df shape after single mol: {mol_df.shape}")
		mol_df = mol_df[(mol_df['num_radicals'] == 0 )]
		print(f"> Filtered mol_df shape after num_radicals = 0: {mol_df.shape}")
		mol_df = mol_df[mol_df['charge'] == 0]
		print(f"> Filtered mol_df shape after charge = 0: {mol_df.shape}")
		mol_df = mol_df[mol_df['num_out_set_elements'] == 0]
		print(f"> Filtered mol_df shape after num_out_set_elements = 0: {mol_df.shape}")
		mol_df = mol_df[mol_df['exact_mw'] <= 1600]
		print(f"> Filtered mol_df shape after exact_mw <= 1600 Da: {mol_df.shape}")
		mol_df = mol_df[(mol_df['num_atoms'] > 3 ) & (mol_df['num_atoms'] < 100)]
		print(f"> Filtered mol_df shape after 3 < num_atom < 100: {mol_df.shape}")
		mol_df = mol_df.reset_index(drop=True)

		return mol_df
  
def get_mces_df(data_fp, mol_df, mce_min=0, mce_max=None):
	"""
	Process MCES data and return a DataFrame with columns:
	- 'mol_id_1': first molecule mol_id
	- 'mol_id_2': second molecule mol_id  
	- 'mces': MCES value as float16
	"""

	mol_ids = mol_df['mol_id'].values.tolist()
	smiles_idx_mapping = dict(zip(org_idx, mol_ids))
 
	with np.load(data_fp, allow_pickle=True) as mces_data:
		mces_values = mces_data['mces']
		num_smiles = len(mces_data['smiles'])
	
		# Collect all filtered data in lists
		mol_id_1_list = []
		mol_id_2_list = []
		mces_list = []
		
		mces_idx = 0
		
		print("Processing MCES pairs...")
		for i in tqdm(range(0, num_smiles)):
			for j in tqdm(range(i+1, num_smiles), leave=False):
				if mces_values[mces_idx] >= mce_min and (mce_max is None or mces_values[mces_idx] <= mce_max):
					# Check if i and j in smiles_idx_mapping
					if i not in smiles_idx_mapping or j not in smiles_idx_mapping:
						mces_idx += 1
						continue

					# Map original indices to new indices
					mapped_i = smiles_idx_mapping[i]
					mapped_j = smiles_idx_mapping[j]

					# Add to lists
					mol_id_1_list.append(mapped_i)
					mol_id_2_list.append(mapped_j)
					mces_list.append(mces_values[mces_idx])

				mces_idx += 1
		
		# Create DataFrame
		print("Creating DataFrame...")
		mces_df = pd.DataFrame({
			'mol_id_1': mol_id_1_list,
			'mol_id_2': mol_id_2_list,
			'mces': mces_list
		})
		
		# Convert mces to float16 for memory efficiency
		mces_df['mces'] = mces_df['mces'].astype(np.float16)
				
		print(f"Total MCES pairs saved: {len(mces_df)}")
		print(f"DataFrame shape: {mces_df.shape}")
		print(f"DataFrame memory usage: {mces_df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
		return mces_df

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Filter MCES data, save index pairs and mol_df with properties.")
	parser.add_argument(
		"--data_fp",
		type=str,
		default="./data/mces-data/mces_distances.npz",
		help="Path to the MCES .npz file"
	)
	parser.add_argument(
		"--mce_min",
		type=int,
		default=0,  # Changed to h5 extension
		help="maximum value of MCES pairs to process (default: None, process all pairs)"
	)
	parser.add_argument(
		"--mce_max",
		type=int,
		default=10,  # 
		help="maximum value of MCES pairs to process (default: None, process all pairs)"
	)
	parser.add_argument(
		"--mces_out_fp",
		type=str,
		default="./data/proc/mces/mces_pairs.pkl",  # Changed to pkl extension
		help="Output .pkl file for filtered MCES index pairs"
	)
	parser.add_argument(
		"--mol_df_fp",
		type=str,
		default="./data/proc/mces/mol_df.pkl",
		help="Output .pkl file for mol_df"
	)
	args = parser.parse_args()
	
	mol_df = get_mol_df(args.data_fp)
	mol_df.to_pickle(args.mol_df_fp)
	#mol_df = pd.read_pickle(args.mol_df_fp)
	mces_df = get_mces_df(args.data_fp, mol_df, args.mce_min, args.mce_max)
 
	# Save as pickle
	print(f"Saving {len(mces_df)} MCES pairs to {args.mces_out_fp}...")
	mces_df.to_pickle(args.mces_out_fp)