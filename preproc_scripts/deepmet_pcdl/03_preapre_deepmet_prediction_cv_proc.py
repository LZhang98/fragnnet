
import os
import pandas as pd
import argparse

import frag_gnn.utils.data_utils as data_utils
import numpy as np
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows


def create_spec_df(mol_df):
	spec_dfs_l = []
	for prec_type in ["[M+H]+","[M-H]-"]:
		spec_df_prec = pd.DataFrame()
		spec_df_prec["mol_id"] = mol_df["mol_id"]
		spec_df_prec["exact_mw"] = mol_df["exact_mw"]
		spec_df_prec['spec_type'] = 'MS2'
		spec_df_prec['dset'] = "deepmet"
		spec_df_prec['prec_mz'] = np.nan
		spec_df_prec['prec_type'] = prec_type
		spec_df_prec['nce'] = np.nan
		spec_df_prec['ion_mode'] = 'N' if prec_type == '[M-H]-' else 'P'
		spec_df_prec.loc[:,"prec_mz"] =  par_apply_df_rows(spec_df_prec,data_utils.infer_prec_mz)
		spec_df_prec = spec_df_prec.drop(columns=["exact_mw"])
  
		for ace in [10,20,40]:
			spec_df_ace = spec_df_prec.copy()
			spec_df_ace['ace'] = ace
			spec_dfs_l.append(spec_df_ace)
	
	# 'spec_id', 'mol_id', 'prec_type', 'inst_type', 'frag_mode',
 	# 'spec_type','ion_mode', 'dset', 'dset_spec_id', 'col_gas', 'res', 'ace', 'nce',
	# 'prec_mz', 'peaks', 'ri', 'group_id'
	spec_df_prec = pd.concat(spec_dfs_l,axis=0)
	spec_df_prec['frag_mode'] = 'CID'
	spec_df_prec['inst_type'] = 'QTOF'
	spec_df_prec = spec_df_prec.reset_index(drop=True)
	spec_df_prec['spec_id'] = spec_df_prec.index
	spec_df_prec['dset_spec_id'] = spec_df_prec['spec_id']
	spec_df_prec['group_id'] = spec_df_prec['spec_id']
	# avoid dataset crash
	spec_df_prec['peaks'] = np.empty((len(spec_df_prec),0)).tolist()

	return spec_df_prec

def get_args():
	parser = argparse.ArgumentParser()
	
	# frag feats configs
	parser.add_argument("--proc_dp",type=str,default="data/deepmet/proc")
	parser.add_argument("--deepmet_csv", type=str,default="data/deepmet/DeepMet-DMIDs.zip", help="")
	parser.add_argument("--chunk_size", type=int,default=100000, help="")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	output_dp = './data/deepmet/proc'
	split_dp = './data/split/pcdl'
	prc_dp = './data/proc/pcdl'
	mol_df_fp = os.path.join(prc_dp, 'mol_df.pkl')
	mol_df = pd.read_pickle(mol_df_fp)
	
	cvs = [f for f in os.listdir(split_dp) if f.startswith('cv')]
	for cv_fold in cvs:
		if cv_fold == 'cv_5':
			continue
		cv_dp = os.path.join(split_dp, cv_fold)
		#print(cv_dp)
		test_split_fp = os.path.join(cv_dp, 'test_ids.csv')
		test_split_df = pd.read_csv(test_split_fp)
		#print(test_split_df)
		split_mol_df = mol_df[mol_df['mol_id'].isin(test_split_df['mol_id'])]
		split_spec_df = create_spec_df(split_mol_df)
		split_spec_df.to_pickle(os.path.join(output_dp, f'{cv_fold}_spec_df.pkl'))