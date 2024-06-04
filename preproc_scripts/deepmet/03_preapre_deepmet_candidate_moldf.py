
import os
import pandas as pd
import argparse

import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.data_utils import par_apply_series

def process_mol(mol_df):

	# get mol df
	mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles"],data_utils.mol_from_smiles)
	mol_df.loc[:,"inchikey"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchikey)
	mol_df.loc[:,"scaffold"] = par_apply_series(mol_df["mol"],data_utils.get_murcko_scaffold)
	mol_df.loc[:,"formula"] = par_apply_series(mol_df["mol"],data_utils.mol_to_formula)
	mol_df.loc[:,"inchi"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchi)
	mol_df.loc[:,"mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=False))
	mol_df.loc[:,"exact_mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True))
	mol_df.loc[:,"num_atoms"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_atoms)
	mol_df.loc[:,"num_bonds"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_bonds)
	mol_df.loc[:,"charge"] = par_apply_series(mol_df["mol"],data_utils.mol_to_charge)
	mol_df.loc[:,"single_mol"] = par_apply_series(mol_df["mol"],data_utils.check_single_mol)
	mol_df.loc[:,"num_radicals"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_radicals)
	mol_df["inchikey_s"] = mol_df["inchikey"].apply(lambda x: x[:14])

 	# reset indices
	mol_df = mol_df.reset_index(drop=True)

	return mol_df

def get_args():
	parser = argparse.ArgumentParser()
	
	# frag feats configs
	parser.add_argument("--proc_dp",type=str,default="data/deepmet/proc")
	parser.add_argument("--deepmet_csv", type=str,default="data/deepmet/DeepMet-DMIDs.zip", help="")
	parser.add_argument("--chunk_size", type=int,default=100000, help="")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	mol_df = pd.read_csv(args.deepmet_csv)
	mol_df = mol_df[['SMILES', 'DMID']]
	mol_df.rename(columns={"SMILES": "smiles", "DMID": "mol_id"}, inplace=True)
 
	os.makedirs(args.proc_dp, exist_ok=True)
	chunk_size = args.chunk_size
	if chunk_size < 0 or chunk_size > len(mol_df):
		mol_df = process_mol(mol_df)
		mol_df_fp = os.path.join(args.proc_dp,"mol_df.pkl.gz")
		mol_df.to_pickle(mol_df_fp)
	else:
		for i in range(0, len(mol_df), chunk_size):
			print("> chunk {} to {}".format(i, i+chunk_size))
			mol_df_chunk = mol_df[i:i+chunk_size].copy()
			print(len(mol_df_chunk))
			mol_df_chunk = process_mol(mol_df_chunk)
			end = min(i+chunk_size, len(mol_df))
			mol_df_fp = os.path.join(args.proc_dp,f"mol_df_{i//chunk_size+1}.pkl.gz")
			mol_df_chunk.to_pickle(mol_df_fp)
			print(f"Saved {i} to {end} to file {mol_df_fp}")
			del mol_df_chunk