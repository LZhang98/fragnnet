import pandas as pd
import numpy as np
import argparse
from fragnnet.utils.spec_utils import calculate_match_mzs

def not_nan_none(val):
	return val is not None and not (isinstance(val, float) and np.isnan(val))

def get_miss_matched(ground_truth_peaks, predicted_peaks, match_tolerance=1e-5):
	# Calculate matched m/z using the utility function
	"""_summary_

	Args:
		ground_truth_mzs (_type_): _description_
		predicted_mzs (_type_): _description_
		match_tolerance (_type_, optional): _description_. Defaults to 1e-5. (10 ppm)
	Returns:
		_type_: _description_
	"""
 
	ground_truth_mzs = ground_truth_peaks[:,0] if ground_truth_peaks is not None and len(ground_truth_peaks) > 0 else np.array([])
	predicted_mzs = predicted_peaks[:,0] if predicted_peaks is not None and len(predicted_peaks) > 0 else np.array([])
	match_mzs = calculate_match_mzs(
		true_mzs=ground_truth_mzs,
		pred_mzs=predicted_mzs,
		tolerance=match_tolerance
	)
	
	# Identify unmatched predicted m/z values	
	unmatched_predicted_peaks = predicted_peaks[~np.any(match_mzs, axis=0)]
	unmatched_groundtruth_peaks = ground_truth_peaks[~np.any(match_mzs, axis=1)]

	return unmatched_predicted_peaks, unmatched_groundtruth_peaks

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--intensity_threshold", type=float, default=1.0, help="Minimum intensity threshold for peaks")
	#parser.add_argument("--proc_dp", type=str, required=True, help="Processed data directory")
	parser.add_argument("--src_fp", type=str, default="output/inference_eval/ikohsc16/predictions_test_with_gt.pkl", help="Predicted pickle file path")
	#parser.add_argument("--output_fp", type=str, help="Destination pickle file path")
	args = parser.parse_args()
	intensity_threshold = args.intensity_threshold
 
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
 
	# Convert peaks column to numpy arrays
	print("Converting peaks column to numpy arrays")
	pred_df.loc[:, "peaks"] = pred_df["peaks"].apply(lambda peaks: np.array(peaks) if peaks is not None and len(peaks) > 0 else np.array([]).reshape(0, 2))
	
	# Calculate false positives and true negatives
	print("Calculating unmatched peaks...")
	results = pred_df.apply(lambda row: get_miss_matched(row['peaks'], row['pred_peaks']), axis=1)
	pred_df.loc[:, 'false_pos'] = results.apply(lambda x: x[0])  # unmatched predicted peaks
	pred_df.loc[:, 'false_neg'] = results.apply(lambda x: x[1])   # unmatched ground truth peaks

	# Find top 100 false positive m/z values by occurrence frequency
	print("\nAnalyzing false positive m/z frequencies...")
	
	# Group analysis by precursor type and instrument type
	grouping_columns = []
	if 'prec_type' in pred_df.columns:
		grouping_columns.append('prec_type')
	if 'inst_type' in pred_df.columns:
		grouping_columns.append('inst_type')
	
	if len(grouping_columns) > 0:
		print(f"Grouping analysis by: {', '.join(grouping_columns)}")
		
		# Get all unique combinations of grouping variables
		if len(grouping_columns) == 1:
			groups = pred_df[grouping_columns[0]].unique()
			group_combinations = [(g,) for g in groups]
		else:
			group_combinations = pred_df[grouping_columns].drop_duplicates().values
		
		for group_vals in group_combinations:
			if len(grouping_columns) == 1:
				group_filter = pred_df[grouping_columns[0]] == group_vals[0]
				group_label = f"{grouping_columns[0]}={group_vals[0]}"
				file_suffix = f"_{grouping_columns[0]}_{group_vals[0]}"
			else:
				group_filter = True
				group_parts = []
				file_parts = []
				for i, col in enumerate(grouping_columns):
					group_filter &= (pred_df[col] == group_vals[i])
					group_parts.append(f"{col}={group_vals[i]}")
					file_parts.append(f"{col}_{group_vals[i]}")
				group_label = ", ".join(group_parts)
				file_suffix = "_" + "_".join(file_parts)
			
			print(f"\n=== Group: {group_label} ===")
			
			# Filter data for this group
			group_df = pred_df[group_filter]
			
			# Collect all false positive m/z values for this group
			all_false_pos_mzs = []
			for false_pos_peaks in group_df['false_pos']:
				if len(false_pos_peaks) > 0:
					# Extract m/z values (first column) from false positive peaks
					false_pos_mzs = false_pos_peaks[:, 0]
					all_false_pos_mzs.extend(false_pos_mzs)
			
			# Collect all false negative m/z values for this group
			all_false_neg_mzs = []
			for false_neg_peaks in group_df['false_neg']:  # true_neg contains unmatched ground truth peaks (false negatives)
				if len(false_neg_peaks) > 0:
					# Extract m/z values (first column) from false negative peaks
					false_neg_mzs = false_neg_peaks[:, 0]
					all_false_neg_mzs.extend(false_neg_mzs)
			
			# Count occurrences of each m/z value
			if len(all_false_pos_mzs) > 0:
				all_false_pos_mzs = np.array(all_false_pos_mzs)
				unique_mzs, counts = np.unique(all_false_pos_mzs, return_counts=True)
				
				# Sort by frequency (descending)
				sorted_indices = np.argsort(counts)[::-1]
				top_100_mzs = unique_mzs[sorted_indices][:100]
				top_100_counts = counts[sorted_indices][:100]
				
				print(f"Found {len(unique_mzs)} unique false positive m/z values")
				print(f"Total false positive occurrences: {len(all_false_pos_mzs)}")
				print(f"Number of spectra: {len(group_df)}")
				print(f"\nTop 20 false positive m/z values for {group_label}:")
				print("Rank\tm/z\tCount")
				print("-" * 25)
				for i, (mz, count) in enumerate(zip(top_100_mzs[:20], top_100_counts[:20]), 1):
					print(f"{i}\t{mz:.3f}\t{count}")
				
				# Find examples for each top false positive m/z
				print(f"\nFinding examples for top 10 false positive peaks...")
				examples_data = []
				
				for rank, target_mz in enumerate(top_100_mzs[:10], 1):
					examples_found = 0
					print(f"\nExamples for m/z {target_mz:.3f} (rank {rank}):")
					
					for idx, (spec_id, false_pos_peaks) in enumerate(zip(group_df['spec_id'], group_df['false_pos'])):
						if examples_found >= 10:  # Limit to 10 examples per m/z
							break
							
						if len(false_pos_peaks) > 0:
							# Check if this target m/z is in the false positive peaks for this spectrum
							false_pos_mzs = false_pos_peaks[:, 0]
							if target_mz in false_pos_mzs:
								# Find the intensity for this m/z
								mz_idx = np.where(false_pos_mzs == target_mz)[0][0]
								intensity = false_pos_peaks[mz_idx, 1]
								
								examples_found += 1
								example_row = group_df.iloc[idx]
								
								print(f"  {examples_found}. spec_id: {spec_id}, intensity: {intensity:.3f}")
								
								# Collect data for CSV
								example_data = {
									'false_pos_mz': target_mz,
									'rank': rank,
									'example_num': examples_found,
									'spec_id': spec_id,
									'intensity': intensity
								}
								
								# Add grouping columns
								for col in grouping_columns:
									example_data[col] = example_row[col]
								
								# Add other relevant columns if available
								for col in ['mol_id_x', 'mol_id_y', 'prec_mz', 'cos_sim_0.01', 'cos_hun']:
									if col in example_row:
										example_data[col] = example_row[col]
								
								examples_data.append(example_data)
				
				# Save examples to separate file
				if examples_data:
					examples_df = pd.DataFrame(examples_data)
					examples_file = args.src_fp.replace('.pkl', f'_false_pos_examples{file_suffix}.csv')
					examples_df.to_csv(examples_file, index=False)
					print(f"\nSaved examples to: {examples_file}")
				
				# Save results to file
				results_data = {
					'rank': range(1, len(top_100_mzs) + 1),
					'mz': top_100_mzs,
					'count': top_100_counts
				}
				
				# Add grouping columns to results
				for i, col in enumerate(grouping_columns):
					results_data[col] = group_vals[i]
				
				results_df = pd.DataFrame(results_data)
				output_file = args.src_fp.replace('.pkl', f'_top100_false_pos_mzs{file_suffix}.csv')
				results_df.to_csv(output_file, index=False)
				print(f"Saved results to: {output_file}")
			else:
				print(f"No false positive peaks found for {group_label}")
			
			# === FALSE NEGATIVE ANALYSIS ===
			print(f"\n--- False Negative Analysis for {group_label} ---")
			
			# Count occurrences of each false negative m/z value
			if len(all_false_neg_mzs) > 0:
				all_false_neg_mzs = np.array(all_false_neg_mzs)
				unique_neg_mzs, neg_counts = np.unique(all_false_neg_mzs, return_counts=True)
				
				# Sort by frequency (descending)
				sorted_neg_indices = np.argsort(neg_counts)[::-1]
				top_100_neg_mzs = unique_neg_mzs[sorted_neg_indices][:100]
				top_100_neg_counts = neg_counts[sorted_neg_indices][:100]
				
				print(f"Found {len(unique_neg_mzs)} unique false negative m/z values")
				print(f"Total false negative occurrences: {len(all_false_neg_mzs)}")
				print(f"Number of spectra: {len(group_df)}")
				print(f"\nTop 20 false negative m/z values for {group_label}:")
				print("Rank\tm/z\tCount")
				print("-" * 25)
				for i, (mz, count) in enumerate(zip(top_100_neg_mzs[:20], top_100_neg_counts[:20]), 1):
					print(f"{i}\t{mz:.3f}\t{count}")
				
				# Find examples for each top false negative m/z
				print(f"\nFinding examples for top 10 false negative peaks...")
				neg_examples_data = []
				
				for rank, target_mz in enumerate(top_100_neg_mzs[:10], 1):
					examples_found = 0
					print(f"\nExamples for m/z {target_mz:.3f} (rank {rank}):")
					
					for idx, (spec_id, false_neg_peaks) in enumerate(zip(group_df['spec_id'], group_df['false_neg'])):
						if examples_found >= 10:  # Limit to 10 examples per m/z
							break
							
						if len(false_neg_peaks) > 0:
							# Check if this target m/z is in the false negative peaks for this spectrum
							false_neg_mzs = false_neg_peaks[:, 0]
							if target_mz in false_neg_mzs:
								# Find the intensity for this m/z
								mz_idx = np.where(false_neg_mzs == target_mz)[0][0]
								intensity = false_neg_peaks[mz_idx, 1]
								
								examples_found += 1
								example_row = group_df.iloc[idx]
								
								print(f"  {examples_found}. spec_id: {spec_id}, intensity: {intensity:.3f}")
								
								# Collect data for CSV
								example_data = {
									'false_neg_mz': target_mz,
									'rank': rank,
									'example_num': examples_found,
									'spec_id': spec_id,
									'intensity': intensity
								}
								
								# Add grouping columns
								for col in grouping_columns:
									example_data[col] = example_row[col]
								
								# Add other relevant columns if available
								for col in ['mol_id_x', 'mol_id_y', 'prec_mz', 'cos_sim_0.01', 'cos_hun']:
									if col in example_row:
										example_data[col] = example_row[col]
								
								neg_examples_data.append(example_data)
				
				# Save false negative examples to separate file
				if neg_examples_data:
					neg_examples_df = pd.DataFrame(neg_examples_data)
					neg_examples_file = args.src_fp.replace('.pkl', f'_false_neg_examples{file_suffix}.csv')
					neg_examples_df.to_csv(neg_examples_file, index=False)
					print(f"\nSaved false negative examples to: {neg_examples_file}")
				
				# Save false negative results to file
				neg_results_data = {
					'rank': range(1, len(top_100_neg_mzs) + 1),
					'mz': top_100_neg_mzs,
					'count': top_100_neg_counts
				}
				
				# Add grouping columns to results
				for i, col in enumerate(grouping_columns):
					neg_results_data[col] = group_vals[i]
				
				neg_results_df = pd.DataFrame(neg_results_data)
				neg_output_file = args.src_fp.replace('.pkl', f'_top100_false_neg_mzs{file_suffix}.csv')
				neg_results_df.to_csv(neg_output_file, index=False)
				print(f"Saved false negative results to: {neg_output_file}")
			else:
				print(f"No false negative peaks found for {group_label}")
	else:
		print("No grouping columns found (prec_type, inst_type), performing overall analysis...")
		# Collect all false positive m/z values
		all_false_pos_mzs = []
		for false_pos_peaks in pred_df['false_pos']:
			if len(false_pos_peaks) > 0:
				# Extract m/z values (first column) from false positive peaks
				false_pos_mzs = false_pos_peaks[:, 0]
				all_false_pos_mzs.extend(false_pos_mzs)
		
		# Count occurrences of each m/z value
		if len(all_false_pos_mzs) > 0:
			all_false_pos_mzs = np.array(all_false_pos_mzs)
			unique_mzs, counts = np.unique(all_false_pos_mzs, return_counts=True)
			
			# Sort by frequency (descending)
			sorted_indices = np.argsort(counts)[::-1]
			top_100_mzs = unique_mzs[sorted_indices][:100]
			top_100_counts = counts[sorted_indices][:100]
			
			print(f"Found {len(unique_mzs)} unique false positive m/z values")
			print(f"Total false positive occurrences: {len(all_false_pos_mzs)}")
			print("\nTop 100 false positive m/z values by occurrence:")
			print("Rank\tm/z\tCount")
			print("-" * 25)
			for i, (mz, count) in enumerate(zip(top_100_mzs, top_100_counts), 1):
				print(f"{i}\t{mz:.3f}\t{count}")
			
			# Save results to file
			results_df = pd.DataFrame({
				'rank': range(1, len(top_100_mzs) + 1),
				'mz': top_100_mzs,
				'count': top_100_counts
			})
			output_file = args.src_fp.replace('.pkl', '_top100_false_pos_mzs.csv')
			results_df.to_csv(output_file, index=False)
			print(f"\nSaved top 100 false positive m/z results to: {output_file}")
		else:
			print("No false positive peaks found!")
	
	print(f"\nFinished processing all files!")