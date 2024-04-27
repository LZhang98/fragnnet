import os
import pandas as pd
import argparse

from rdkit import RDLogger
import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.data_utils import par_apply_series
from frag_gnn.utils.formula_utils import get_elements_set, parse_formula
from frag_gnn.utils.frag_utils import ELEMENTS
from frag_gnn.utils.spec_utils import prec_type_mass_diff, PREC_TYPES

from frag_gnn.frag.compute_frags import (
	MAX_NUM_NODES
)

def get_args():
	
	parser = argparse.ArgumentParser()

 	#gen params
	parser.add_argument("--proc_dp", type=str,default="data/proc/nist20mona23", help="")
	parser.add_argument("--split_dp", type=str, default="./data/split/nist20_d4_wr00_inchikey")
	parser.add_argument("--save_dp", type=str,default="data/benchmark/speed_benchmark/5000", help="")
	parser.add_argument("--random_seed", type=int,default=42, help="")
	parser.add_argument("--num_sampled", type=int,default=5000, help="")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	RDLogger.DisableLog("rdApp.*")
	args = get_args()
	os.makedirs(args.save_dp, exist_ok=True)
	split_csvs = [
		"test_ids.csv",
		"val_ids.csv",
		"train_ids.csv"
	]

	mol_df_fp = os.path.join(args.proc_dp,"mol_df.pkl")
	print(f"> Read mol df from {mol_df_fp}")
	mol_df = pd.read_pickle(mol_df_fp)

	split_id_dfs_l = []
	for split_csv_filename in split_csvs:
		split_id_fp = os.path.join(args.split_dp, split_csv_filename)
		sp_mol_ids = pd.read_csv(split_id_fp)['mol_id'].drop_duplicates().to_frame()
		split_id_dfs_l.append(sp_mol_ids)
	id_df = pd.concat(split_id_dfs_l)
 
	mol_df = mol_df[mol_df['mol_id'].isin(id_df['mol_id'])]
	mol_df = mol_df.sample(n=args.num_sampled, random_state=args.random_seed)
	mol_df.reset_index()
 
	spec_df = mol_df.copy()
	spec_df['spec_id'] = spec_df['mol_id']
	spec_df['group_id'] = spec_df['mol_id']
	spec_df['prec_type'] = '[M+H]+'
	
	spec_df['nce'] = [[10,20,30] for _ in range(len(spec_df))]
	spec_df['ace'] = [[10,20,30] for _ in range(len(spec_df))]
	spec_df["prec_mz"] = spec_df['exact_mw'] + prec_type_mass_diff('[M+H]+')

	print(f"> selected {len(mol_df)} mol after filter")
	print("> save mol df")
	mol_df = mol_df[['mol_id','mol','smiles','inchi']]
	mol_df.to_pickle(os.path.join(args.save_dp, "mol_df.pkl.gz"))
	print("> save spec df")
	spec_df = spec_df[['spec_id', 'group_id', 'mol_id','prec_type','nce', 'ace', 'prec_mz']]
	spec_df.to_pickle(os.path.join(args.save_dp, "spec_df.pkl.gz"))