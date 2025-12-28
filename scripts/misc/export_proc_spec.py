# post process predicted spectra
import pandas as pd
import os
import numpy as np
import argparse
from tqdm import tqdm

def not_nan_none(val):
	return val is not None and not (isinstance(val, float) and np.isnan(val))
			
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--proc_dp", default="data/proc/nps_nist23", type=str, help="Processed data directory")
	parser.add_argument("--output_fp", default="nps_nist23.mgf", type=str, help="Destination pickle file path")
	
	args = parser.parse_args()
	proc_dp = args.proc_dp
	mgf_fp = args.output_fp

	print("Loading reference data...")
	mol_df_path_pkl = os.path.join(proc_dp, "mol_df.pkl")
	mol_df = pd.read_pickle(mol_df_path_pkl)
	print(f"Loaded mol_df with {len(mol_df)} molecules from {mol_df_path_pkl}")

	spec_df_path_pkl = os.path.join(proc_dp, "spec_df.pkl")
	spec_df = pd.read_pickle(spec_df_path_pkl)
	print(f"Loaded spec_df with {len(spec_df)} spectra from {spec_df_path_pkl}")

	merged_df = spec_df.merge(mol_df, how='left', left_on='mol_id', right_on='mol_id') #, suffixes=('_spec', '_mol'))
	print(f"Merged spec_df and mol_df into merged_df with {len(merged_df)} entries")
	#print(merged_df.columns.tolist())
	# Ensure output directories exist
	if os.path.dirname(mgf_fp) != '':
		os.makedirs(os.path.dirname(mgf_fp), exist_ok=True)

	with open(mgf_fp, 'w') as mgf_file:
		for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
			mgf_file.write('BEGIN IONS\n')
			mgf_file.write(f"TITLE={row['spec_id']}\n")

			if 'mol_id' in row and not_nan_none(row['mol_id']):
				mgf_file.write(f"MOL_ID={row['mol_id']}\n")
			if 'smiles' in row and not_nan_none(row['smiles']):
				mgf_file.write(f"SMILES={row['smiles']}\n")
			if 'inchikey_s' in row and not_nan_none(row['inchikey_s']):
				mgf_file.write(f"INCHIKEY_S={row['inchikey_s']}\n")
			if 'prec_type' in row and not_nan_none(row['prec_type']):
				mgf_file.write(f"PRECURSORTYPE={row['prec_type']}\n")
			if 'nce' in row and not_nan_none(row['nce']):
				mgf_file.write(f"COLLISIONENERGY={row['nce']}% NCE\n")
			if 'ace' in row and not_nan_none(row['ace']):
				mgf_file.write(f"COLLISIONENERGY={row['ace']}eV\n")
			if 'formula' in row:
				mgf_file.write(f"FORMULA={row['formula']}\n")
			if 'exact_mass' in row:
				mgf_file.write(f"EXACTMASS={row['exact_mass']}\n")
			if 'dset' in row and not_nan_none(row['dset']):
				mgf_file.write(f"DATASET={row['dset']}\n")
			if 'dset_spec_id' in row and not_nan_none(row['dset_spec_id']):
				mgf_file.write(f"DATASET_SPEC_ID={row['dset_spec_id']}\n")
			if 'prec_mz' in row and not_nan_none(row['prec_mz']):
				mgf_file.write(f"PEPMASS={row['prec_mz']}\n")
    
			# Write peaks, filter out intensity < 1
			peaks = row['peaks']
			for mz, intensity in peaks:
				mgf_file.write(f"{mz:.6f} {intensity:.1f}\n")
			mgf_file.write('END IONS\n\n')