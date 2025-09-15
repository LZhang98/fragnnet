# Example: stack pred_mz and pred_int as pred_peaks
# pred_peaks = np.stack([pred_mz, pred_int], axis=1)
import pandas as pd
import os
import numpy as np


def not_nan_none(val):
	return val is not None and not (isinstance(val, float) and np.isnan(val))
			
if __name__ == "__main__":

	intensity_threshold = 1.0
	
	src_fp = "output/inference_outputs/mas_2064277/ikohsc16/chunks/chunk_0.pkl"
	proc_dp = "./data/other/mas_2064277/proc"
	mol_df = pd.read_pickle(os.path.join(proc_dp, "mol_df.pkl.gz"))
	print(f"Loaded mol_df with {len(mol_df)} molecules")
	spec_df = pd.read_pickle(os.path.join(proc_dp, "spec_df.pkl.gz"))
	print(f"Loaded spec_df with {len(spec_df)} spectra")
	
	spec_df = spec_df.merge(mol_df[['mol_id', 'smiles', 'inchikey_s']], on='mol_id', how='left')
	spec_df = spec_df.drop(columns=['peaks'])
	#print(spec_df.head(10))
	#df = pd.read_pickle(src_fp)
	#print(df.head(10))
	pred_df = pd.read_pickle(src_fp)
	print(f"Loaded pred_df with {len(pred_df)} predictions")
	pred_df = pred_df.merge(spec_df, on='spec_id', how='left')
	print(f"Merged pred_df now has {len(pred_df)} rows")

	pred_df.loc[:, "pred_peaks"] = pred_df.apply(lambda x: np.stack([np.array(x['pred_mzs']), np.array(x['pred_ints'])], axis=1), axis=1)
	#print(pred_df.head(10))
	#print(pred_df.columns)
	# Convert predictions to MGF format
	mgf_fp = src_fp.replace('.pkl', '.mgf')
	with open(mgf_fp, 'w') as mgf_file:
		for idx, row in pred_df.iterrows():
			#print(row)
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
				mgf_file.write(f"ALT_COLLISIONENERGY={row['ace']}eV\n")
			if 'formula' in row:
				mgf_file.write(f"FORMULA={row['formula']}\n")
			if 'exact_mass' in row:
				mgf_file.write(f"EXACTMASS={row['exact_mass']}\n")
    
    		# Write peaks, filter out intensity < 1
			peaks = row['pred_peaks']
			for mz, intensity in peaks:
				if intensity >= intensity_threshold:
					mgf_file.write(f"{mz:.6f} {intensity:.6f}\n")
			mgf_file.write('END IONS\n\n')
	print(f"Saved MGF to {mgf_fp}")