import numpy as np
import pandas as pd

import argparse
from frag_gnn.utils import formula_utils, frag_utils
import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows

from frag_gnn.frag.compute_frags import (
	MAX_NUM_NODES
)

def process_mol(mol_df, drop_duplicate = False, tqdm_leave=True):

	# get mol df
	mol_df.loc[:,"mol"] = par_apply_series(mol_df["inchi"],data_utils.mol_from_inchi, tqdm_leave)
	mol_df = mol_df.dropna(subset=["mol"])
	mol_df.loc[:,"smiles"] = par_apply_series(mol_df["mol"],data_utils.mol_to_smiles, tqdm_leave)
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

def create_spec_df(mol_df, prec_type, nce):

	#for prec_type in ["[M+H]+","[M-H]-"]:
	spec_df = pd.DataFrame()
	spec_df["mol_id"] = mol_df["mol_id"]
	spec_df["target_group_id"] = mol_df["target_group_id"]
	spec_df["target_inchikey_s"] = mol_df["target_inchikey_s"]
	spec_df["exact_mw"] = mol_df["exact_mw"]
	spec_df['spec_type'] = 'MS2'
	spec_df['dset'] = "casmi2016"
	spec_df['prec_mz'] = np.nan
	spec_df['prec_type'] = prec_type
	spec_df['nce'] = float(nce)
	spec_df['ace'] = np.nan
	spec_df['ion_mode'] = 'N' if prec_type == '[M-H]-' else 'P'
	spec_df.loc[:,"prec_mz"] =  par_apply_df_rows(spec_df,data_utils.infer_prec_mz)
	spec_df = spec_df.drop(columns=["exact_mw"])
	
	# 'spec_id', 'mol_id', 'prec_type', 'inst_type', 'frag_mode',
 	# 'spec_type','ion_mode', 'dset', 'dset_spec_id', 'col_gas', 'res', 'ace', 'nce',
	# 'prec_mz', 'peaks', 'ri', 'group_id'
	spec_df['frag_mode'] = 'HCD'
	spec_df['inst_type'] = 'FT'
	spec_df = spec_df.reset_index(drop=True)
	spec_df['spec_id'] = spec_df.index
	spec_df['dset_spec_id'] = spec_df['spec_id']
	spec_df['group_id'] = spec_df['spec_id']
 
	# avoid dataset crash
	#spec_df['peaks'] = np.empty((len(spec_df),0)).tolist()
	return spec_df

def get_args():
	parser = argparse.ArgumentParser()
	
	# frag feats configs
	parser.add_argument("--proc_dp",type=str,default="data/proc/nist20mona23v3")
	parser.add_argument("--split_dp",type=str,default="data/split/nist20mona23v3_d4_wr00_inchikey")
	parser.add_argument("--split_out_dp",type=str,default="data/split/nist20mona23v3_d4_wr00_inchikey_casmi2016")
	parser.add_argument("--casmi_csv",type=str,default = "data/raw/casmi2016/solutions_casmi2016_cat2and3.csv")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = get_args()
 
	print(f'> Read casmi 2016 csv df from {args.casmi_csv}')
	casmi_df = pd.read_csv(args.casmi_csv)
	#print(casmi_df)
 
	casmi_df = casmi_df[casmi_df['ION_MODE'] == ' POSITIVE']
	print(len(casmi_df))
	casmi_mol_data = []
	#nce = [20,35,50]
	casmi_spec_df = []

	casmi_spec_data = []
	for idx,row in casmi_df.iterrows():
		#print(row)
		c_name = row['ChallengeName']
		c_id = c_name.split('-')[-1]
		c_id = int(c_id)
  
		c_smiles =row['SMILES']
		c_inchi =row['INCHI']
		c_inchikey =row['INCHIKEY']
		c_mol = data_utils.mol_from_inchi(c_inchi)
		c_inchikey_s = data_utils.mol_to_inchikey_s(c_mol)
		filename = f"data/raw/casmi2016/cfm4_outputs/id_output_{c_name}.log"
		with open(filename, 'r') as file_in:
			for line in file_in:
				inchi = line.split(' ')[-1]
				casmi_mol_data.append((c_id,inchi,c_inchikey_s))
   
		# read specfile
		spec_filename = f"data/raw/casmi2016/peaklists/{c_name}.txt"
  
		peaks = []
		with open(spec_filename) as file_in:
			for line in file_in:
				#peak_data = (float(x) for x in )
				x,y = line.rstrip().split("\t")
				peaks.append((float(x),float(y)))
		casmi_spec_data.append((c_id,peaks))
  
	mol_df = pd.DataFrame(data=casmi_mol_data, columns=["target_group_id", "inchi","target_inchikey_s"])
	#print(mol_df)
	#print(mol_df)
	mol_df = process_mol(mol_df)
	mol_df = mol_df[mol_df['single_mol']]
	mol_df = mol_df[mol_df['charge'] == 0]
 	# filter by heavy atom account
	allowed_elements = set(frag_utils.ELEMENTS)
	#mol_df = mol_df[mol_df['num_heavy_atoms'] <= MAX_NUM_NODES]
	# filter by radicals
	mol_df['num_radicals'] = mol_df['mol'].apply(lambda x: sum([atom.GetNumRadicalElectrons() for atom in x.GetAtoms()]))
	mol_df = mol_df[mol_df['num_radicals'] == 0]
	# filter by out_set_elements
	mol_df['num_out_set_elements'] = mol_df['formula'].apply(lambda x: len(set(list(formula_utils.parse_formula(x).keys())) - allowed_elements))
	mol_df = mol_df[mol_df['num_out_set_elements'] == 0]
	mol_df = mol_df.reset_index(drop=True)
	mol_df['mol_id'] = mol_df.index + 1

	mol_df.to_pickle("./data/ms2c/casmi2016/proc/mol_df.pkl.gz")

	spec_df = create_spec_df(mol_df, "[M+H]+", 50)
	#spec_df = create_spec_df(mol_df, "[M+H]+", 35)
	#spec_df["peaks"]
	casmi_spec_df = pd.DataFrame(data=casmi_spec_data, columns=['target_group_id','peaks'])
	spec_df = pd.merge(spec_df,casmi_spec_df,how="inner",on="target_group_id")
	spec_df.to_pickle("./data/ms2c/casmi2016/proc/spec_df.pkl.gz")
	#print(spec_df['peaks'].head())