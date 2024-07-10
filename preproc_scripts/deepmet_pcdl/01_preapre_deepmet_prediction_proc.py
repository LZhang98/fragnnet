
import os
import pandas as pd
import argparse

import frag_gnn.utils.data_utils as data_utils
import numpy as np
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows

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
	parser.add_argument("--proc_dp",type=str,default="data/deepmet/add_carbon_proc")
	parser.add_argument("--deepmet_csv", type=str,default="data/deepmet/AddCarbon-HMDB4.tsv.gz", help="")
	parser.add_argument("--smiles_col", type=str,default="mutated_smiles", help="")
	parser.add_argument("--id_col", type=str,default="mol_id", help="")
	parser.add_argument("--chunk_size", type=int,default=100000, help="")
	parser.add_argument("--is_tsv", type=bool, action=argparse.BooleanOptionalAction, default=False)
	#parser.set_defaults("--is_tsv", type=bool, action=argparse.BooleanOptionalAction)
	args = parser.parse_args()
	return args

if __name__ == "__main__":


	"""
	for mol_file in os.listdir('data/deepmet/proc/'):
		if mol_file.startswith("mol_df_"):
			mol_df = pd.read_pickle(os.path.join(f'data/deepmet/proc/{mol_file}'))
			spec_df = create_spec_df(mol_df)
			spec_df.to_pickle(os.path.join(f'data/deepmet/proc/{mol_file.replace("mol","spec")}'))
	exit()
	"""
 
	args = get_args()
	mol_df = pd.read_csv(args.deepmet_csv, sep="\t" if args.is_tsv else ",")
	smile_col = args.smiles_col
	id_col = args.id_col
	if id_col not in mol_df.columns:
		mol_df['mol_id'] = mol_df.index

	mol_df = mol_df[[smile_col, id_col]]
	mol_df.rename(columns={smile_col: "smiles", id_col: "mol_id"}, inplace=True)
 
	os.makedirs(args.proc_dp, exist_ok=True)
	chunk_size = args.chunk_size
	if chunk_size < 0 or chunk_size > len(mol_df):
		mol_df = process_mol(mol_df)
		mol_df_fp = os.path.join(args.proc_dp,"mol_df.pkl.gz")
		mol_df.to_pickle(mol_df_fp)
		spec_df = create_spec_df(mol_df)
		spec_df.to_pickle(os.path.join(args.proc_dp,"spec_df.pkl.gz"))
	else:
		for i in range(0, len(mol_df), chunk_size):
			print("> chunk {} to {}".format(i, i+chunk_size))
			mol_df_chunk = mol_df[i:i+chunk_size].copy()
			print(len(mol_df_chunk))
			mol_df_chunk = process_mol(mol_df_chunk)
			end = min(i+chunk_size, len(mol_df))
			mol_df_fp = os.path.join(args.proc_dp,f"mol_df_{i//chunk_size+1}.pkl.gz")
			mol_df_chunk.to_pickle(mol_df_fp)
			spec_df_chunk = create_spec_df(mol_df_chunk)
			spec_df_chunk.to_pickle(os.path.join(args.proc_dp,f"spec_df_{i//chunk_size+1}.pkl.gz"))
			print(f"Saved {i} to {end} to file {mol_df_fp}")
			del mol_df_chunk