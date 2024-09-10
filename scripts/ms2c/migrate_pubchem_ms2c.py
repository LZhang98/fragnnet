import pickle
import os
import pandas as pd


if __name__ == "__main__":
	proc_dp = 'data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50'
	mol_df_fp = os.path.join(proc_dp, 'mol_df.pkl.gz')
	spec_df_fp = os.path.join(proc_dp, 'spec_df.pkl.gz')
	mol_df = pd.read_pickle(mol_df_fp)
	spec_df = pd.read_pickle(spec_df_fp)	
	
	print(len(spec_df))
	#pickle.loads('')