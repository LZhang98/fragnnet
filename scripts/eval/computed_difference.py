# post process predicted spectra
import pandas as pd
import os
import numpy as np
import argparse
from fragnnet.utils.spec_utils import calculate_match_mzs

def not_nan_none(val):
	return val is not None and not (isinstance(val, float) and np.isnan(val))

def get_miss_matched(ground_truth_mzs, predicted_mzs, match_tolerance=1e-5):
	# Calculate matched m/z using the utility function
	"""_summary_

	Args:
		ground_truth_mzs (_type_): _description_
		predicted_mzs (_type_): _description_
		match_tolerance (_type_, optional): _description_. Defaults to 1e-5. (10 ppm)
		match_relative (bool, optional): _description_. Defaults to True.
		match_tolerance_min_mz (float, optional): _description_. Defaults to 0.1.
	Returns:
		_type_: _description_
	"""
 
	match_mzs = calculate_match_mzs(
		true_mzs=ground_truth_mzs,
		pred_mzs=predicted_mzs,
		tolerance=match_tolerance
	)
	
	# Identify unmatched predicted m/z values
	unmatched_predicted_mzs = predicted_mzs[~np.any(match_mzs, axis=0)]
	unmatched_groundtruth_mzs = ground_truth_mzs[~np.any(match_mzs, axis=1)]
	print(unmatched_predicted_mzs, unmatched_groundtruth_mzs)
	return unmatched_predicted_mzs

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--intensity_threshold", type=float, default=1.0, help="Minimum intensity threshold for peaks")
	#parser.add_argument("--proc_dp", type=str, required=True, help="Processed data directory")
	parser.add_argument("--src_fp", type=str, default="output/inference_eval/ikohsc16/predictions_test_with_gt.pkl", help="Predicted pickle file path")
	#parser.add_argument("--output_fp", type=str, help="Destination pickle file path")
	args = parser.parse_args()
	intensity_threshold = args.intensity_threshold
  
	"""proc_dp = args.proc_dp
	
	# Load reference data once - try .pkl.gz first, then .pkl, then error
	mol_df_path_gz = os.path.join(proc_dp, "mol_df.pkl.gz")
	mol_df_path_pkl = os.path.join(proc_dp, "mol_df.pkl")
	
	if os.path.exists(mol_df_path_gz):
		mol_df = pd.read_pickle(mol_df_path_gz)
		print(f"Loaded mol_df with {len(mol_df)} molecules from {mol_df_path_gz}")
	elif os.path.exists(mol_df_path_pkl):
		mol_df = pd.read_pickle(mol_df_path_pkl)
		print(f"Loaded mol_df with {len(mol_df)} molecules from {mol_df_path_pkl}")
	else:
		raise FileNotFoundError(f"Could not find mol_df file in {proc_dp}. Looked for both mol_df.pkl.gz and mol_df.pkl")
	
	spec_df_path_gz = os.path.join(proc_dp, "spec_df.pkl.gz")
	spec_df_path_pkl = os.path.join(proc_dp, "spec_df.pkl")
	
	if os.path.exists(spec_df_path_gz):
		spec_df = pd.read_pickle(spec_df_path_gz)
		print(f"Loaded spec_df with {len(spec_df)} spectra from {spec_df_path_gz}")
	elif os.path.exists(spec_df_path_pkl):
		spec_df = pd.read_pickle(spec_df_path_pkl)
		print(f"Loaded spec_df with {len(spec_df)} spectra from {spec_df_path_pkl}")
	else:
		raise FileNotFoundError(f"Could not find spec_df file in {proc_dp}. Looked for both spec_df.pkl.gz and spec_df.pkl")
	"""
	# Process each file
	pred_df = pd.read_pickle(args.src_fp)
	print(f"Loaded pred_df with {len(pred_df)} predictions")

	#pred_df = pred_df.merge(spec_df, on='spec_id', how='left')
	print(f"assemble predicted peaks {len(pred_df)} rows")
	pred_df.loc[:, "pred_peaks"] = pred_df.apply(lambda x: np.stack([np.array(x['pred_mzs']), np.array(x['pred_ints'])], axis=1), axis=1)
	
	if intensity_threshold > 0:
		# Apply intensity threshold filtering
		print(f"Applied intensity threshold of {intensity_threshold}, now pred_df has peaks with intensities >= {intensity_threshold}")
		pred_df.loc[:, "pred_peaks"] = pred_df["pred_peaks"].apply(lambda peaks: peaks[peaks[:, 1] >= intensity_threshold])

	# Round m/z values to 0.001 for pred_peaks
	print("Rounding m/z values to nearest 0.01 for predicted peaks")
	pred_df.loc[:, "pred_peaks"] = pred_df["pred_peaks"].apply(lambda peaks: np.column_stack([
		np.ceil(peaks[:, 0] / 0.01) * 0.01,  # Round up m/z values to nearest 0.01
		np.ceil(peaks[:, 1] / 0.01) * 0.01,  # Round up intensity values to nearest 0.01
	]))
	pred_df.loc[:, "pred_mzs"] = pred_df["pred_peaks"].apply(lambda peaks: peaks[:, 0] if peaks is not None and len(peaks) > 0 else np.array([]))
	pred_df.loc[:, "pred_ints"] = pred_df["pred_peaks"].apply(lambda peaks: peaks[:, 1] if peaks is not None and len(peaks) > 0 else np.array([]))

	# Convert peaks column to numpy arrays
	print("Converting peaks column to numpy arrays")
	pred_df.loc[:, "peaks"] = pred_df["peaks"].apply(lambda peaks: np.array(peaks) if peaks is not None and len(peaks) > 0 else np.array([]).reshape(0, 2))
	pred_df.loc[:, "mzs"] = pred_df["peaks"].apply(lambda peaks: np.array(peaks)[:, 0] if peaks is not None and len(peaks) > 0 else np.array([]))
	
 	#pred_df.drop(columns=['pred_mzs', 'pred_ints'], inplace=True)
	#print(pred_df.columns)
	#print(pred_df[['peaks','pred_peaks','cos_sim_0.01']].head(10))
	#pred_df.loc[:, "missed_mz"] = pred_df.apply(lambda row: get_miss_matched_mzs(row), axis=1)
	get_miss_matched(pred_df.loc[0,'mzs'], pred_df.loc[0,'pred_mzs'])