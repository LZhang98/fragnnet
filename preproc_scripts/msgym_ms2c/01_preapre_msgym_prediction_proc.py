import json

import os
import pandas as pd
import argparse

import frag_gnn.utils.data_utils as data_utils
import numpy as np
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows, parse_inst_info, seq_apply_df_rows
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
	mol_df.loc[:,"exact_mw"] = par_apply_series(mol_df["mol"
													],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True), tqdm_leave)
	mol_df.loc[:,"num_atoms"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_atoms, tqdm_leave)
	mol_df.loc[:,"num_bonds"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_bonds, tqdm_leave)
	mol_df.loc[:,"charge"] = par_apply_series(mol_df["mol"],data_utils.mol_to_charge, tqdm_leave)
	mol_df.loc[:,"single_mol"] = par_apply_series(mol_df["mol"],data_utils.check_single_mol, tqdm_leave)
	mol_df.loc[:,"num_radicals"] = par_apply_series(mol_df["mol"],data_utils.mol_to_num_radicals, tqdm_leave)

 	# reset indices
	mol_df = mol_df.reset_index(drop=True)
 
	return mol_df

def create_spec_df(mol_df, prec_type, prec_mz, inst_type, frag_mode, nce, ace):
	spec_df_prec = pd.DataFrame()
	spec_df_prec["mol_id"] = mol_df["mol_id"]
	spec_df_prec["exact_mw"] = mol_df["exact_mw"]
	spec_df_prec['spec_type'] = 'MS2'
	spec_df_prec['dset'] = "msgym"
	spec_df_prec['prec_mz'] = np.nan
	spec_df_prec['prec_type'] = prec_type
	spec_df_prec['ace'] = ace
	spec_df_prec['nce'] = nce # nce should be NaN
	spec_df_prec['ion_mode'] = 'N' if prec_type == '[M-H]-' else 'P'
	spec_df_prec.loc[:,"prec_mz"] =  prec_mz #par_apply_df_rows(spec_df_prec,data_utils.infer_prec_mz)
	spec_df_prec = spec_df_prec.drop(columns=["exact_mw"])

	# 'spec_id', 'mol_id', 'prec_type', 'inst_type', 'frag_mode',
 	# 'spec_type','ion_mode', 'dset', 'dset_spec_id', 'col_gas', 'res', 'ace', 'nce',
	# 'prec_mz', 'peaks', 'ri', 'group_id'
	spec_df_prec['frag_mode'] = frag_mode
	spec_df_prec['inst_type'] = inst_type
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
	parser.add_argument("--proc_dp",type=str,default="./data/ms2c/msgym/proc/retrieval_candidates_formula")
	parser.add_argument("--json_fp",type=str,default="./data/raw/msgym/MassSpecGym_retrieval_candidates_formula_with_test.json")
	parser.add_argument("--tsv_fp",type=str,default="./data/raw/msgym/MassSpecGym_with_test.tsv")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	# read tsv first, we only need to do testing
	print(f"> read {args.tsv_fp}")
	msg_spectra_df = pd.read_csv(args.tsv_fp, sep="\t")
	# NOTE: only use simulation_challenge
	msg_spectra_df = msg_spectra_df[msg_spectra_df["simulation_challenge"]]
	msg_spectra_test_df = msg_spectra_df[msg_spectra_df["fold"] == "test"].copy()
	msg_spectra_test_df.rename(columns={"instrument_type": "inst_type", "precursor_mz":"prec_mz", "adduct":"prec_type","collision_energy":"ace"}, inplace=True)
	
	inst_type, frag_mode = seq_apply_df_rows(msg_spectra_test_df,parse_inst_info)
	msg_spectra_test_df.loc[:,"inst_type"] = inst_type
	msg_spectra_test_df.loc[:,"frag_mode"] = frag_mode
 
	msg_spectra_test_df["peaks"] = msg_spectra_test_df.apply(lambda x: "\n".join([f"{m} {i}" for m,i in zip(x["mzs"].split(","),x["intensities"].split(","))]),axis=1)
	msg_spectra_test_df = msg_spectra_test_df.drop(columns=["mzs","intensities"])
	

	print(f"> read {args.json_fp}")
	msg_candidates_dict = json.load(open(args.json_fp))
	os.makedirs(args.proc_dp, exist_ok=True)
	print(f"> save to {args.proc_dp}")


	unique_smiles = msg_spectra_test_df['smiles'].unique().tolist()
	
	msg_spectra_test_df['mol_df'] = ''
	msg_spectra_test_df['spec_df'] = ''
    
	for mol_idx, input_smiles in tqdm(enumerate(unique_smiles), total=len(unique_smiles)):

		mol_df_fname = f"mol_df_{mol_idx}.pkl"
		try:
			candidates_smiles = msg_candidates_dict[input_smiles]
			mol_df_fp = os.path.join(args.proc_dp, mol_df_fname)
			#if os.path.isfile(mol_df_fp):
			#	mol_df = pd.read_pickle(mol_df_fp)
			#else:
			mol_df = pd.DataFrame({'smiles': candidates_smiles})
			mol_df['mol_id'] = mol_df.index
			mol_df = process_mol(mol_df, True, None)
			mol_df.to_pickle(mol_df_fp)

			for spectra_idx, row in msg_spectra_test_df[msg_spectra_test_df["smiles"] == input_smiles].iterrows():

				prec_type = row['prec_type']
				prec_mz = row['prec_mz']
				inst_type = row['inst_type']
				frag_mode = row['frag_mode']
				ace = row['ace']
				nce = np.nan
				spec_df = create_spec_df(mol_df, prec_type, prec_mz, inst_type, frag_mode, nce, ace)
	
				identifier = row['identifier']
				spec_df_fname =  f"spec_df_{identifier}.pkl"
				spec_df_fp = os.path.join(args.proc_dp, spec_df_fname)
				spec_df.to_pickle(spec_df_fp)
				msg_spectra_test_df.loc[spectra_idx,'mol_df'] = mol_df_fname
				msg_spectra_test_df.loc[spectra_idx,'spec_df'] = spec_df_fname

				del spec_df
				
			del mol_df
		except Exception as e:
			print("failed", e)
	
	msg_spectra_test_df.to_csv(os.path.join(args.proc_dp, "msg_id_map.csv"))