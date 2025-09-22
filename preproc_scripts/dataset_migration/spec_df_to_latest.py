import numpy as np
import os
import pandas as pd
import argparse

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--spec_df_fp", type=str, required=True)

	args = parser.parse_args()
	print(f"> Loading spec_df from: {args.spec_df_fp}")
	spec_df = pd.read_pickle(args.spec_df_fp)
	print(f"> Loaded spec_df with shape: {spec_df.shape}")
	print(f"> Original columns: {list(spec_df.columns)}")

	print(f"> Checking for missing collision energy columns...")
	for name in ['ace_extra_1', 'nce_extra_1', 'ace_extra_2', 'nce_extra_2']:
		if name not in spec_df.columns:
			print(f"> adding missing column {name} to spec_df")
			spec_df.loc[:, name] = np.nan
		else:
			print(f"> column {name} already exists")

	print(f"> Checking for missing boolean columns...")
	for name in ['ramped','stepped']:
		if name not in spec_df.columns:
			print(f"> adding missing column {name} to spec_df")
			spec_df.loc[:, name] = False
		else:
			print(f"> column {name} already exists")

	if 'has_ace' not in spec_df.columns:
		spec_df.loc[:,'has_ace'] = ~spec_df["ace"].isna()
	
	if 'has_nce' not in spec_df.columns:
		spec_df.loc[:,'has_nce'] = ~spec_df["nce"].isna()
  
	print(f"> Updated spec_df shape: {spec_df.shape}")
	print(f"> Updated columns: {list(spec_df.columns)}")

	# Create backup of original spec_df as spec_df_orig
	spec_df_orig_fp = args.spec_df_fp.replace("spec_df", "spec_df_orig")
	print(f"> Creating backup of original spec_df...")
	print(f"> saving original spec_df as {spec_df_orig_fp}")
	spec_df_orig = pd.read_pickle(args.spec_df_fp)  # Read original again to avoid modifications
	spec_df_orig.to_pickle(spec_df_orig_fp)
	print(f"> Backup saved successfully")
	
	# Save updated spec_df with same filename (replacing original)
	print(f"> Saving updated spec_df to replace original...")
	print(f"> saving updated spec_df as {args.spec_df_fp}")
	spec_df.to_pickle(args.spec_df_fp)
	print(f"> Updated spec_df saved successfully")
	
	print("> ==========================================")
	print("> Migration complete!")
	print(f"> Original backup: {spec_df_orig_fp}")
	print(f"> Updated file: {args.spec_df_fp}")
	print("> ============================================")
 