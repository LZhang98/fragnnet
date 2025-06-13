import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series


def get_np_memory_usage(array):
	memory_usage = array.nbytes
	# Convert to megabytes
	memory_usage_mb = memory_usage / (1024 ** 2)
	print(f"Memory usage: {memory_usage} bytes ({memory_usage_mb:.2f} MB)")

def process_mol_df(data_fp, mol_df_fp):
	with np.load(data_fp, allow_pickle=True) as mces_data:
		smiles = mces_data['smiles']
  
		# Create mol_df from smiles using data_utils functions
		mol_df = pd.DataFrame({'smiles_org': smiles})
		mol_df.loc[:,"idx"] = range(len(mol_df))

		#mol_df['mol'] = par_apply_series(mol_df['smiles'], data_utils.mol_from_smiles)
		#mol_df['smiles_std'] = par_apply_series(mol_df['mol'], data_utils.mol_to_smiles)
		
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
		mol_df = mol_df.dropna(subset=["smiles", "smiles_org", "inchikey_s", "scaffold", "formula", "inchi"])
		mol_df = mol_df.reset_index(drop=True)
		mol_df.to_pickle(mol_df_fp)
  
def process_mces(data_fp, out_fp, mol_df_fp):
	

	with np.load(data_fp, allow_pickle=True) as mces_data:
		mces = mces_data['mces']
		print(len(mces))
		num_mces = len(mces)
		num_smiles = len(mces_data['smiles'])
		get_np_memory_usage(mces)
		#get_np_memory_usage(mces_data['smiles'])
		#smiles = mces_data['smiles']

		# Remove all pairs where mces == -1
		#valid_mask = mces != -1
		#valid_indices = np.where(valid_mask)[0]

		#N = len(smiles)

		def _upper_triangular_index_to_pair(idx, N):
			"""
			Inverse mapping: flat index -> (i, j) for upper-triangular matrix storage.
			Used to recover the original molecule pair from a flat MCES index.
			See: https://stackoverflow.com/a/27086195
			"""
			#N = self.num_samples
			i = N - 2 - int(np.floor(np.sqrt(-8*idx + 4*N*(N-1)-7)/2.0 - 0.5))
			j = idx + i + 1 - N*(N-1)//2 + (N-i)*((N-i)-1)//2
			return i, j

		#for i in range(num_smiles + 1):
		#	print(_upper_triangular_index_to_pair(i, num_mces))
		# Prepare output array: (smiles_idx1, smiles_idx2, mces)
		#filtered_mces_data = np.empty((len(valid_indices), 3), dtype=np.int64)
		#for k, idx in enumerate(tqdm(valid_indices, desc="Processing valid pairs")):
		#	i, j = unravel_indices(idx, N)
		#	filtered_mces_data[k, 0] = i
		#	filtered_mces_data[k, 1] = j
		#	filtered_mces_data[k, 2] = mces[idx]


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Filter MCES data, save index pairs and mol_df with properties.")
	parser.add_argument(
		"--data_fp",
		type=str,
		default="./data/mces-data/mces_distances.npz",
		help="Path to the MCES .npz file"
	)
	parser.add_argument(
		"--out_fp",
		type=str,
		default="./data/mces-data/mces_pairs_idx.npy",
		help="Output .npy file for filtered MCES index pairs"
	)
	parser.add_argument(
		"--mol_df_fp",
		type=str,
		default="./data/mces-data/mol_df.pkl",
		help="Output .pkl file for mol_df"
	)
	args = parser.parse_args()
	process_mces(args.data_fp, args.out_fp, args.mol_df_fp)