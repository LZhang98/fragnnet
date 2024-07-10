
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
	parser.add_argument("--deepmet_moldf_dp", type=str,default="data/deepmet/proc/mol_df_1.pkl.gz", help="")
	parser.add_argument("--frag_dp",type=str,default="data/frag/debug")
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--max_time",type=int,default=150)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--allowed_elements",type=str,nargs="+",default=frag_utils.ELEMENTS)
	parser.add_argument("--nb_isomorphic",type=booltype,default=False)
	parser.add_argument("--max_iterations",type=int,default=-1)
	parser.add_argument("--isotopes",type=booltype,default=False)
	parser.add_argument("--allow_cached", type=booltype,default=False, help="")
	parser.add_argument("--compressed", type=booltype,default=True, help="")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	args = get_args()

	mol_df = pd.read_pickle(args.deepmet_moldf_dp)
	os.makedirs(args.frag_dp, exist_ok=True)
	dag_feat_inputs_l = []
 
	for _, row in tqdm(mol_df.iterrows(), total=len(mol_df), desc="Prepare Data"):
		# put here to prevent unnecessary frag computation
		# we may support this in the future
		# todo updat mol_df to include this
		if row['charge'] != 0:
			continue
		if row['num_atoms'] > frag_utils.MAX_NUM_NODES:
			continue
			
		dag_feat_inputs_l.append([
								row['mol'], 
								row['mol_id'], 
								args.max_depth,
								True, # h_prior
								args.max_h_transfer,
								args.max_time,
								args.isotopes, 
								args.nb_isomorphic,
								args.max_iterations,
								args.frag_dp,
								args.allow_cached,
								args.compressed
								])
	
	frag_results = par_apply(iter(dag_feat_inputs_l), frag_utils.timed_get_dags,True, return_as_generator=True)
	for res in tqdm(frag_results, total = len(dag_feat_inputs_l), desc = "Compute Frags"):
		continue