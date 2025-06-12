import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import fragnnet.utils.data_utils as data_utils
from fragnnet.utils.data_utils import par_apply_series

def main(data_fp, out_fp, mol_df_fp):
	# Load MCES data into memory (allow_pickle=True for string arrays)
	with np.load(data_fp, allow_pickle=True) as mces_data:
		mces = mces_data['mces']
		smiles = mces_data['smiles'].copy()

		# Remove all pairs where mces == -1
		valid_mask = mces != -1
		valid_indices = np.where(valid_mask)[0]

		N = len(smiles)
		def unravel_indices(idx, N):
			i = idx // N
			j = idx % N
			return i, j

		# Prepare output array: (smiles_idx1, smiles_idx2, mces)
		filtered_mces_data = np.empty((len(valid_indices), 3), dtype=np.int64)
		for k, idx in enumerate(tqdm(valid_indices, desc="Processing valid pairs")):
			i, j = unravel_indices(idx, N)
			filtered_mces_data[k, 0] = i
			filtered_mces_data[k, 1] = j
			filtered_mces_data[k, 2] = mces[idx]

	# Create mol_df from smiles using data_utils functions
	mol_df = pd.DataFrame({'smiles': smiles})
	#mol_df['mol'] = par_apply_series(mol_df['smiles'], data_utils.mol_from_smiles)
	#mol_df['smiles_std'] = par_apply_series(mol_df['mol'], data_utils.mol_to_smiles)
	
	mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles"],data_utils.mol_from_smiles)
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
	
	# Save as npy file
	#np.save(out_fp, result)
	#print(f"Saved filtered MCES index pairs to {out_fp}")
	# Save mol_df as pickle
	#mol_df.to_pickle(mol_df_fp)
	#print(f"Saved mol_df to {mol_df_fp}")

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
	main(args.data_fp, args.out_fp, args.mol_df_fp)