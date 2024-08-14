
import shutil
import pandas as pd
from frag_gnn.utils.misc_utils import booltype
from tqdm import tqdm
import argparse
import os

import frag_gnn.utils.frag_utils as frag_utils
from frag_gnn.utils.data_utils import par_apply
from frag_gnn.utils.misc_utils import booltype
import tempfile
import json

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--msgym_moldf_dp", type=str,default="data/ms2c/msgym/proc/retrieval_candidates_formula", help="")
	parser.add_argument("--frag_dp",type=str,default="data/frag/debug")
	parser.add_argument("--max_depth",type=int,default=3)
	parser.add_argument("--max_time",type=int,default=150)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--allowed_elements",type=str,nargs="+",default=frag_utils.ELEMENTS)
	parser.add_argument("--nb_isomorphic",type=booltype,default=False)
	parser.add_argument("--max_iterations",type=int,default=-1)
	parser.add_argument("--isotopes",type=booltype,default=False)
	parser.add_argument("--allow_cached", type=booltype,default=False, help="")
	parser.add_argument("--compressed", type=booltype,default=True, help="")
	parser.add_argument("--frag_dp_archive", type=str,default="zip", choices=["tar", "zip", "bztar", "gztar"],  help="")
	parser.add_argument("--star_mol_df_id", type=int, default=0)
	parser.add_argument("--num_ids", type=int, default=1)
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	args = get_args()

	os.makedirs(args.frag_dp, exist_ok=True)
 
	for mod_df_id in tqdm(range(args.star_mol_df_id, args.star_mol_df_id+args.num_ids)):
		mol_df_fp = os.path.join(args.msgym_moldf_dp, f"mol_df_{mod_df_id}.pkl")
		if not os.path.isfile(mol_df_fp):
			print(f"{mol_df_fp} is not a file")
			continue
		mol_df = pd.read_pickle(mol_df_fp)
		dag_feat_inputs_l = []
		with tempfile.TemporaryDirectory() as tmp_dp:

			for _, row in tqdm(mol_df.iterrows(), total=len(mol_df), desc="Prepare Data", leave=None):
				# put here to prevent unnecessary frag computation
				# we may support this in the future
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
										tmp_dp,
										args.allow_cached,
										args.compressed
										])
			
			frag_results = par_apply(iter(dag_feat_inputs_l), frag_utils.timed_get_dags,True, return_as_generator=True)
			for res in tqdm(frag_results, total = len(dag_feat_inputs_l), desc = "Compute Frags", leave=None):
				continue
			
			# num_fags_compunted = len([name for name in os.listdir(tmp_dp) if os.path.isfile(os.path.join(tmp_dp, name))])
			# print(mol_df_fp,len(dag_feat_inputs_l), num_fags_compunted)
			frag_dest = os.path.join(args.frag_dp, f"mol_df_{mod_df_id}")
			if args.frag_dp_archive in ["tar", "zip", "bztar", "gztar"]:
				shutil.make_archive(base_name = frag_dest, format= args.frag_dp_archive, root_dir=tmp_dp, base_dir="./")
			else:
				shutil.copytree(tmp_dp, frag_dest, dirs_exist_ok = True)