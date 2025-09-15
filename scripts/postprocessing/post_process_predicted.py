# Example: stack pred_mz and pred_int as pred_peaks
# pred_peaks = np.stack([pred_mz, pred_int], axis=1)
import pandas as pd
import os
import numpy as np
import argparse


def not_nan_none(val):
	return val is not None and not (isinstance(val, float) and np.isnan(val))
			
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--intensity_threshold", type=float, default=1.0, help="Minimum intensity threshold for peaks")
	parser.add_argument("--proc_dp", type=str, default="./data/other/mas_2064277/proc", help="Processed data directory")
	# Add a mutually exclusive group
	src_group = parser.add_mutually_exclusive_group()
	src_group.add_argument("--src_fp", type=str, help="Source pickle file path")
	src_group.add_argument("--src_dp", type=str, help="Source directory path to process all pkl/pkl.gz files")

	# Add a mutually exclusive group
	dst_group = parser.add_mutually_exclusive_group()
	dst_group.add_argument("--output_fp", type=str, help="Destination pickle file path")
	dst_group.add_argument("--output_dp", type=str, help="Destination directory path to process all pkl/pkl.gz files")

	# Whether to export to MGF format
	parser.add_argument("--export_mgf", action="store_true", help="Export predictions to MGF format")
	parser.add_argument("--export_pkl", action="store_true", help="Export predictions to PKL format")

	args = parser.parse_args()
	# Check that either src_fp or src_dp is provided
	if not args.src_fp and not args.src_dp:
		parser.error("Either --src_fp or --src_dp must be provided")

	if not args.output_fp and not args.output_dp:
		parser.error("Either --output_fp or --output_dp must be provided")

	intensity_threshold = args.intensity_threshold
	proc_dp = args.proc_dp
	
	# Load reference data once
	mol_df = pd.read_pickle(os.path.join(proc_dp, "mol_df.pkl.gz"))
	print(f"Loaded mol_df with {len(mol_df)} molecules")
	spec_df = pd.read_pickle(os.path.join(proc_dp, "spec_df.pkl.gz"))
	print(f"Loaded spec_df with {len(spec_df)} spectra")
	
	spec_df = spec_df.merge(mol_df[['mol_id', 'smiles', 'inchikey_s']], on='mol_id', how='left')
	spec_df = spec_df.drop(columns=['peaks'])

	# Get list of files to process
	if args.src_fp:
		files_to_process = [args.src_fp]
	else:
		# Process all pkl and pkl.gz files in src_dp
		import glob
		pkl_files = glob.glob(os.path.join(args.src_dp, "*.pkl"))
		pkl_gz_files = glob.glob(os.path.join(args.src_dp, "*.pkl.gz"))
		files_to_process = pkl_files + pkl_gz_files
		print(f"Found {len(files_to_process)} pickle files to process in {args.src_dp}")

	# Process each file
	for file_idx, src_fp in enumerate(files_to_process):
		print(f"\nProcessing file {file_idx + 1}/{len(files_to_process)}: {src_fp}")
		
		try:
			pred_df = pd.read_pickle(src_fp)
			print(f"Loaded pred_df with {len(pred_df)} predictions")
			pred_df = pred_df.merge(spec_df, on='spec_id', how='left')
			print(f"Merged pred_df now has {len(pred_df)} rows")

			pred_df.loc[:, "pred_peaks"] = pred_df.apply(lambda x: np.stack([np.array(x['pred_mzs']), np.array(x['pred_ints'])], axis=1), axis=1)
			
			if args.export_mgf:
				# Convert predictions to MGF format
				mgf_fp = src_fp.replace('.pkl', '.mgf').replace('.gz', '')
				with open(mgf_fp, 'w') as mgf_file:
					for idx, row in pred_df.iterrows():
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
			
			if args.export_pkl:
				# Save processed predictions to PKL format
				if args.output_fp:
					pkl_fp = args.output_fp
				elif args.output_dp:
					basename = os.path.basename(src_fp)
					pkl_fp = os.path.join(args.output_dp, f"processed_{basename}")
				else:
					# Default: save in same directory with 'processed_' prefix
					dirname = os.path.dirname(src_fp)
					basename = os.path.basename(src_fp)
					pkl_fp = os.path.join(dirname, f"processed_{basename}")
				
				# Ensure output directory exists
				os.makedirs(os.path.dirname(pkl_fp), exist_ok=True)
				
				# Save the processed dataframe
				pred_df.to_pickle(pkl_fp)
				print(f"Saved processed PKL to {pkl_fp}")
			
			print(f"Completed processing {os.path.basename(src_fp)}")
			
		except Exception as e:
			print(f"Error processing {src_fp}: {str(e)}")
			continue

	print(f"\nFinished processing all files!")