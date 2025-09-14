import argparse
import os
import pandas as pd
import h5py
from tqdm import tqdm

if __name__ == "__main__":
	mol_df_fp = "data/other/mimedb/proc/mol_df.pkl.gz"
	spec_df_fp = "data/other/mimedb/proc/spec_df.pkl.gz"
 
	spec_df = pd.read_pickle(spec_df_fp)
	print('Unique prec_type values:')
	print(spec_df['prec_type'].unique())
	