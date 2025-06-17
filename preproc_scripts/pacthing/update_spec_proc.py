import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

def add_missing_columns(df):
	"""
	Add missing columns to a DataFrame with default values using .loc for better performance:
	- 'ramped' and 'stepped': False if missing
	- 'col_energy_extra_1' and 'col_energy_extra_2': np.nan if missing
	
	Args:
		df (pd.DataFrame): Input DataFrame
		
	Returns:
		pd.DataFrame: DataFrame with added columns if they were missing
	"""
	# Add 'ramped' column if it doesn't exist
	if 'ramped' not in df.columns:
		df.loc[:, 'ramped'] = False
	
	# Add 'stepped' column if it doesn't exist
	if 'stepped' not in df.columns:
		df.loc[:, 'stepped'] = False
	
	# Add 'col_energy_extra_1' column if it doesn't exist
	if 'col_energy_extra_1' not in df.columns:
		df.loc[:, 'col_energy_extra_1'] = np.nan
	
	# Add 'col_energy_extra_2' column if it doesn't exist
	if 'col_energy_extra_2' not in df.columns:
		df.loc[:, 'col_energy_extra_2'] = np.nan
	
	return df

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Add missing columns to a Pickle file.")
	parser.add_argument(
		"--input_fp",
		type=str,
		required=True,
		help="Input Pickle (.pkl) file path"
	)
	parser.add_argument(
		"--output_fp",
		type=str,
		default=None,
		help="Output Pickle file path. If not specified, original file will be overwritten."
	)
	
	args = parser.parse_args()
	
	# Main processing logic
	input_fp = args.input_fp
	
	# Check if the input file exists
	if not os.path.exists(input_fp):
		print(f"Error: File {input_fp} does not exist.")
		exit(1)
	
	print(f"Processing: {input_fp}")
	
	# Read the Pickle file
	df = pd.read_pickle(input_fp)
	
	# Add missing columns
	df = add_missing_columns(df)
	
	# Determine output path
	if args.output_fp:
		output_path = args.output_fp
		
		# Create parent directory if it doesn't exist
		os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
	else:
		# Overwrite the original file
		output_path = input_fp
	
	# Save the modified DataFrame
	df.to_pickle(output_path)
	
	print(f"Successfully processed file and saved to {output_path}")
