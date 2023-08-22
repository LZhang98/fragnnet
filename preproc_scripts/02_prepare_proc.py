import numpy as np
import os
import pandas as pd
import ast
import json
from pprint import pprint, pformat
import joblib
from tqdm import tqdm
import argparse

import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows, seq_apply_series, seq_apply_df_rows
from frag_gnn.utils.misc_utils import list_str2float, booltype


def load_df(df_dp,dsets,num_entries):
	
	dfs = []
	for dset in dsets:
		dset_fp = os.path.join(df_dp,f"{dset}_df.json")
		dset_df = pd.read_json(open(dset_fp,"r",encoding="utf8"))
		dset_df.loc[:,"dset"] = dset
		dfs.append(dset_df)
	if num_entries > 0:
		dfs = [df.sample(n=num_entries,replace=False,random_state=420) for df in dfs]
	if len(dfs) > 1:
		all_df = pd.concat(dfs,ignore_index=True)
	else:
		all_df = dfs[0]
	all_df = all_df.reset_index(drop=True)

	return all_df


def preprocess_spec(spec_df):

	# convert smiles to mol and back (for standardization/stereochemistry)
	spec_df.loc[:,"mol"] = par_apply_series(spec_df["smiles"],data_utils.mol_from_smiles)
	spec_df.loc[:,"smiles"] = par_apply_series(spec_df["mol"],data_utils.mol_to_smiles)
	spec_df = spec_df.dropna(subset=["mol","smiles"])
	# enumerate smiles to create molecule ids
	smiles_set = set(spec_df["smiles"])
	print("> num_smiles", len(smiles_set))
	smiles_to_mid = {smiles:i for i,smiles in enumerate(sorted(smiles_set))}
	spec_df.loc[:,"mol_id"] = spec_df["smiles"].replace(smiles_to_mid)

	# extract peak info (still represented as str)
	spec_df.loc[:,"peaks"] = par_apply_series(spec_df["peaks"],data_utils.parse_peaks_str)
	# get mz resolution
	spec_df.loc[:,"res"] = par_apply_series(spec_df["peaks"],data_utils.get_res)
	# standardize the instrument type and frag_mode
	inst_type, frag_mode = seq_apply_df_rows(spec_df,data_utils.parse_inst_info)
	spec_df.loc[:,"inst_type"] = inst_type
	spec_df.loc[:,"frag_mode"] = frag_mode
	# standardize ce
	spec_df.loc[:,"ace"] = par_apply_series(spec_df["col_energy"],data_utils.parse_ace_str)
	spec_df.loc[:,"nce"] = par_apply_series(spec_df["col_energy"],data_utils.parse_nce_str)
	spec_df = spec_df.drop(columns=["col_energy"])
	# standardise prec_type
	spec_df.loc[:,"prec_type"] = par_apply_series(spec_df["prec_type"],data_utils.parse_prec_type_str)
	# convert prec_mz
	spec_df.loc[:,"prec_mz"] = pd.to_numeric(spec_df["prec_mz"],errors="coerce")
	# convert ion_mode
	spec_df.loc[:,"ion_mode"] = par_apply_series(spec_df["ion_mode"],data_utils.parse_ion_mode_str)
	# convert peaks to float
	spec_df.loc[:,"peaks"] = par_apply_series(spec_df["peaks"],data_utils.convert_peaks_to_float)
	# get retention index
	spec_df.loc[:,"ri"] = par_apply_series(spec_df["ri"],data_utils.parse_ri_str)

	# remove columns from spec_df
	spec_df = spec_df[["spec_id","mol_id","prec_type","inst_type","frag_mode","spec_type","ion_mode","dset","dset_spec_id","col_gas","res","ace","nce","prec_mz","peaks","ri"]]
	# relabel spec_id (this is to make it unique across datasets)
	spec_df.loc[:,"spec_id"] = np.arange(spec_df.shape[0])
	# set group_id (same compound and precursor)
	group_df = spec_df.drop(columns=["spec_id","dset_spec_id","peaks","nce","ace","res","prec_mz","ri","col_gas"]).drop_duplicates()
	group_df.loc[:,"group_id"] = np.arange(group_df.shape[0])
	spec_df = spec_df.merge(group_df,how="inner")

	# get mol df
	mol_df = pd.DataFrame(zip(sorted(smiles_set),list(range(len(smiles_set)))),columns=["smiles","mol_id"])
	mol_df.loc[:,"mol"] = par_apply_series(mol_df["smiles"],data_utils.mol_from_smiles)
	mol_df.loc[:,"inchikey_s"] = par_apply_series(mol_df["mol"],data_utils.mol_to_inchikey_s)
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

	# remove invalid mols and corresponding spectra
	all_mol_id = set(mol_df["mol_id"])
	mol_df = mol_df.dropna(subset=["mol"],axis=0)
	bad_mol_id = all_mol_id - set(mol_df["mol_id"])
	print("> bad_mol_id",len(bad_mol_id))
	spec_df = spec_df[~spec_df["mol_id"].isin(bad_mol_id)]

	# reset indices
	spec_df = spec_df.reset_index(drop=True)
	mol_df = mol_df.reset_index(drop=True)

	return spec_df, mol_df


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--df_dp", type=str, default="data/df")
	parser.add_argument("--proc_dp", type=str, default="data/proc")
	parser.add_argument("--num_entries", type=int, default=-1)
	parser.add_argument("--ow_spec_mol", type=booltype, default=False)
	parser.add_argument("--dset", type=str, nargs="+", default=["nist20_hr","nist20_lr","mona23"])
	flags = parser.parse_args()

	assert os.path.isdir(flags.proc_dp), flags.proc_dp
	data_dp = flags.proc_dp

	spec_df_fp = os.path.join(data_dp,"spec_df.pkl")
	mol_df_fp = os.path.join(data_dp,"mol_df.pkl")

	if not flags.ow_spec_mol and os.path.isfile(spec_df_fp) and os.path.isfile(mol_df_fp):

		print("> loading previous spec_df, mol_df")
		spec_df = pd.read_pickle(spec_df_fp)
		mol_df = pd.read_pickle(mol_df_fp)

	else:

		print("> creating new spec_df, mol_df")
		assert os.path.isdir(flags.df_dp), flags.df_dp
		all_df = load_df(flags.df_dp,flags.num_entries)

		spec_df, mol_df = preprocess_spec(all_df)

		# save everything to file
		spec_df.to_pickle(spec_df_fp)
		mol_df.to_pickle(mol_df_fp)

	print(spec_df.shape)
	print(spec_df.isna().sum())
	print(mol_df.shape)
	print(mol_df.isna().sum())

