
import pandas as pd
from frag_gnn.utils.misc_utils import booltype
from tqdm import tqdm
import argparse
import os

import frag_gnn.utils.frag_utils as frag_utils
from frag_gnn.utils.data_utils import par_apply
from frag_gnn.utils.misc_utils import booltype


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--deepmet_fp", type=str,default="data/deepmet/deepmet_fragnnet_cv.csv", help="")
	parser.add_argument("--proc_dp", type=str,default="data/proc/nist20mona23/", help="")
	parser.add_argument("--dset", type=str,default="nist20_hr", help="")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	print(f'> Loading data from {args.deepmet_fp}')
	deepmet_df = pd.read_csv(args.deepmet_fp)
	#print(deepmet_df)
	proc_moldf_fp = args.proc_dp + 'mol_df.pkl'
	print(f'> Loading data from {proc_moldf_fp}')
	proc_moldf = pd.read_pickle(proc_moldf_fp)
 
	proc_specdf_fp = args.proc_dp + 'spec_df.pkl'
	print(f'> Loading data from {proc_specdf_fp}')
	proc_specdf = pd.read_pickle(proc_specdf_fp)

	#inchikeys = deepmet_df['inchikey_s'].tolist()
	mol_ids = deepmet_df['mol_id'].tolist()
	moldf = proc_moldf[proc_moldf['mol_id'].isin(mol_ids) & \
     					(proc_moldf['charge'] == 0) & \
          				(proc_moldf['num_atoms'] <= frag_utils.MAX_NUM_NODES)]
	#print(len(moldf))
	proc_specdf = proc_specdf[proc_specdf['mol_id'].isin(moldf['mol_id'].to_list()) & \
                           	  (proc_specdf['prec_type'] == '[M+H]+') & \
                              (proc_specdf['frag_mode'] == 'HCD') & \
                              (proc_specdf['inst_type'] == 'FT') & 
                              (proc_specdf['dset'] == args.dset)]
	
	proc_specdf_nce = proc_specdf.drop_duplicates(subset=['mol_id','nce'])
	#print(proc_specdf_nce['nce'].value_counts())
	#nce_df = proc_specdf.groupby('mol_id')['nce'].unique()
	#nce_df[1:] = nce_df[1:].apply(lambda x: set(x))
	#print(nce_df[1:].value_counts())
	
	nce_count_df = proc_specdf_nce['nce'].value_counts()
	top_ten_nce = nce_count_df.index.tolist()[0:10]
	
	proc_specdf_top_ten_nce = proc_specdf_nce[proc_specdf_nce['nce'].isin(top_ten_nce)]
	nce_df = proc_specdf_top_ten_nce.groupby('mol_id')['nce'].unique()
	nce_df[1:] = nce_df[1:].apply(lambda x: set(x))
	print(nce_df[1:].value_counts())
	
 	#print(nce_df)
	#print(len(proc_specdf['mol_id'].unique()))
	#proc_specdf['nce'].unique()
	#df = proc_specdf['nce'].value_counts()
	#print(df)
	#print(proc_specdf['mol_id'].value_counts())
	#print(proc_specdf[proc_specdf['mol_id']==34153]['nce'].value_counts())
	#print(proc_specdf[(proc_specdf['mol_id']==34153) & (proc_specdf['nce'] == 10.0)])