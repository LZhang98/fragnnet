import json

import os
import pandas as pd
import argparse

import frag_gnn.utils.data_utils as data_utils
import numpy as np
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows
from tqdm import tqdm

def process_mol(mol_df, drop_duplicate = True, tqdm_leave=True):

	# get mol df
	mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles"],data_utils.mol_from_smiles, tqdm_leave)
	mol_df = mol_df.dropna(subset=["mol"])
	mol_df.loc[:,"inchikey"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchikey, tqdm_leave)
	mol_df["inchikey_s"] = mol_df["inchikey"].apply(lambda x: x[:14])
 
	if drop_duplicate:
		mol_df = mol_df.drop_duplicates(subset=['inchikey'])
 
	mol_df.loc[:,"scaffold"] = par_apply_series(mol_df["mol"],data_utils.get_murcko_scaffold, tqdm_leave)
	mol_df.loc[:,"formula"] = par_apply_series(mol_df["mol"],data_utils.mol_to_formula, tqdm_leave)
	mol_df.loc[:,"inchi"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchi, tqdm_leave)
	mol_df.loc[:,"mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=False), tqdm_leave)
	mol_df.loc[:,"exact_mw"] = par_apply_series(mol_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True), tqdm_leave)
	mol_df.loc[:,"num_atoms"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_atoms, tqdm_leave)
	mol_df.loc[:,"num_bonds"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_bonds, tqdm_leave)
	mol_df.loc[:,"charge"] = par_apply_series(mol_df["mol"],data_utils.mol_to_charge, tqdm_leave)
	mol_df.loc[:,"single_mol"] = par_apply_series(mol_df["mol"],data_utils.check_single_mol, tqdm_leave)
	mol_df.loc[:,"num_radicals"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_radicals, tqdm_leave)

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
		spec_df_prec['ace'] = np.nan
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
	# msg_candidates_mass_json = './data/raw/msgym/MassSpecGym_retrieval_candidates_mass.json'
	
	parser.add_argument("--proc_dp",type=str,default="data/msgym/proc/retrieval_candidates_formula")
	parser.add_argument("--json_fp",type=str,default="./data/raw/msgym/MassSpecGym_retrieval_candidates_formula.json")
	parser.add_argument("--tsv_fp",type=str,default="./data/raw/msgym/MassSpecGym.tsv")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	# read tsv first, we only need to do testing
	print(f"> read {args.tsv_fp}")
	msg_spectra_df = pd.read_csv(args.tsv_fp, sep="\t")
	msg_spectra_df = msg_spectra_df[msg_spectra_df["fold"] == "test"]
	
	msg_candidates_dict = json.load(open(args.json_fp))
	os.makedirs(args.proc_dp, exist_ok=True)
	print(f"> save to {args.proc_dp}")
	msg_test_candidate_dict = {}
 
	for smiles in msg_spectra_df['smiles'].to_list():
		if smiles in msg_candidates_dict:
			msg_test_candidate_dict[smiles] = msg_candidates_dict[smiles]
	print(len(msg_test_candidate_dict))
	for idx, input_smiles in tqdm(enumerate(msg_test_candidate_dict), total=len(msg_test_candidate_dict)):
		try:
			candidates_smiles = msg_test_candidate_dict[input_smiles]
			mol_df = pd.DataFrame({'smiles': candidates_smiles})
			mol_df['mol_id'] = mol_df.index
			mol_df = process_mol(mol_df, True, None)
			#print(len(mol_df))
			mol_df_fp = os.path.join(args.proc_dp,f"mol_df_{idx}.pkl")
			mol_df.to_pickle(mol_df_fp)
			#spec_df = create_spec_df(mol_df)
			#spec_df.to_pickle(os.path.join(args.proc_dp,"spec_df.pkl"))
			del mol_df
		except Exception as e:
			print(e)
