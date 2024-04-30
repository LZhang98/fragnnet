
import os
import pickle
import signal
import sys
import traceback
from threading import Event
import argparse
import os
import io

import tarfile
import bz2
import numpy as np

from tqdm import tqdm
import pandas as pd
from frag_gnn.utils.ms2c_utils import MolCandidateDB
from frag_gnn.utils.misc_utils import booltype, PPM
import frag_gnn.utils.formula_utils as formula_utils
import frag_gnn.utils.frag_utils as frag_utils
import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.misc_utils import booltype
from frag_gnn.frag.compute_frags import (
	MAX_NUM_NODES
)

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from copy import deepcopy
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows, seq_apply_series, seq_apply_df_rows

def get_args():
	parser = argparse.ArgumentParser()
	
	# frag candidates configs
	parser.add_argument("--db_file", type=str, default= "./data/ms2c/databases/hmdb5.sqlite")
	parser.add_argument("--tolerance", type=float, default= 10, help="Default 10ppm")
	parser.add_argument("--use_ppm", type=booltype, default= True)
	parser.add_argument("--split_type", type=str,default="test", help="")
	parser.add_argument("--candidate_dp",type=str,default='./data/ms2c/casmi16/hmdb5_candidates')
	parser.add_argument("--use_cached_candidate",type=booltype,default=False)
	
	args = parser.parse_args()
	return args

def filter_candidates(db_results, target_smiles, target_mol_id):

	chem, rdmolops, rd_moldesc, rd_logger = data_utils.rdkit_import("rdkit.Chem",
																			   "rdkit.Chem.rdmolops", 
																			   "rdkit.Chem.rdMolDescriptors", 
																			   "rdkit.RDLogger")
	
	rd_logger.DisableLog('rdApp.*') 
	t_mol = chem.MolFromSmiles(target_smiles)

	# make sure target in there
	#print(targert_smiles)
	db_results.append((-1, data_utils.mol_to_inchikey(t_mol),
					target_smiles, 
					data_utils.mol_to_formula(t_mol),
					data_utils.mol_to_mol_weight(t_mol,exact=True)))
	
	case_cdf = pd.DataFrame(db_results,columns=['mol_id','inchikey','smiles','formula','mw'])
	#print(case_cdf.tail)
	# drop not single mol rows
	case_cdf = case_cdf[~(case_cdf['smiles'].str.contains('.', regex=False))]
	case_cdf[["inchikey_s", "inchikey_a", "inchikey_flag"]] = case_cdf["inchikey"].str.split('-', expand=True)
	# just normal inchkey 
	case_cdf = case_cdf[case_cdf['inchikey_flag'] == 'N']
	# drop duplicate by incheky and always keep self
	case_cdf = case_cdf.drop_duplicates(subset="inchikey_s", keep="last")
	case_cdf["mol"] = None
	case_cdf["tanimoto"] = -1.0

	#print(case_cdf)
	allowed_elements = set(frag_utils.ELEMENTS)
	for idx, case_row in case_cdf.iterrows():

		mol = chem.MolFromSmiles(case_row["smiles"])

		if mol is None:
			continue
		
		#mol = data_utils.mol_from_smiles(case_row["smiles"]) 
		#if mol is np.nan:
		#	continue

		#check number of atoms for frag
		if rd_moldesc.CalcNumHeavyAtoms(mol) > MAX_NUM_NODES:
			continue

		# remove charged or has radical
		num_radicals = sum([atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()])
		num_charges = rdmolops.GetFormalCharge(mol)
		if num_charges != 0 or num_radicals != 0:
			continue
		
		# remove not allowed element
		elements_set = set(list(formula_utils.parse_formula(case_row["formula"]).keys()))
		if len(elements_set - allowed_elements) > 0:
			continue

		# remove Stereochemistry
		# we should  never see case_cdf['inchikey_flag'] == 'N'
		if case_row['inchikey_a'] != 'UHFFFAOYSA':
			chem.RemoveStereochemistry(mol)

		# skip ones we can not Kekulize, do this after RemoveStereochemistry
		try:
			chem.Kekulize(deepcopy(mol),clearAromaticFlags=True)
		except:
			continue
		
		case_cdf.at[idx,"smiles"] = data_utils.mol_to_smiles(mol)

	
	return target_mol_id, target_smiles, case_cdf

def fetch_db(db_file, mw, mass_tol ):
	with MolCandidateDB(db_file) as db:
		db_results = db.get_compounds_by_exact_mass_range(mw -mass_tol, mw + mass_tol)
	return db_results

if __name__ == "__main__":

	RDLogger.DisableLog('rdApp.*') 

	args = get_args()

	#casmi_cat1_df = pd.read_csv('data/misc/casmi2016/solutions_casmi2016_cat1.csv')
	casmi_cat23_df = pd.read_csv('data/misc/casmi2016/solutions_casmi2016_cat2and3.csv')
	casmi_df = pd.concat([casmi_cat23_df])
	
	casmi_df.loc[:,"mol"] = par_apply_series(casmi_df["SMILES"],data_utils.mol_from_smiles)
	casmi_df.loc[:,"smiles"] = par_apply_series(casmi_df["mol"],data_utils.mol_to_smiles)
	casmi_df.loc[:,"exact_mw"] = par_apply_series(casmi_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True))
	casmi_df['mol_id'] = casmi_df['ChallengeName']
	
	db_file = args.db_file
	db_query_input = []
	filter_candidates_input_l = []
	
	for _, row in tqdm(casmi_df.iterrows(), total = len(casmi_df), desc = "Prepare pubchem database input"):
		# we can frag them, just ignore	
		if Chem.rdMolDescriptors.CalcNumHeavyAtoms(row['mol']) > MAX_NUM_NODES:
			continue

		mw = row['exact_mw']
		if args.use_ppm:
			mass_tol = mw * args.tolerance * PPM
		else:
			mass_tol = args.tolerance
		db_query_input.append([db_file,mw,mass_tol])
			#db_results = db.get_compounds_by_exact_mass_range(mw -mass_tol, mw + mass_tol)
		filter_candidates_input_l.append([None, row['smiles'],row['mol_id']])

	db_results = data_utils.par_apply(iter(db_query_input),fetch_db, True, return_as_generator=True)	
	for idx, res in enumerate(tqdm(db_results, total =  len(db_query_input), desc = "Fetch hmdb database ")):
		filter_candidates_input_l[idx][0] = res
	
	candidate_data_df_l = []
	canidate_results = data_utils.par_apply(iter(filter_candidates_input_l),filter_candidates, True, return_as_generator=True)	
	for res in tqdm(canidate_results,  total =  len(filter_candidates_input_l), desc = "Filter candidates"):
		target_mol_id, targert_smiles, case_df = res

		case_df['target_mol_id'] = target_mol_id
		case_df['targert_smiles'] = targert_smiles
		case_df = case_df[['target_mol_id','targert_smiles','inchikey_s','smiles','tanimoto']] 
		candidate_data_df_l.append(case_df)

	candidates_df = pd.concat(candidate_data_df_l)
	candidates_df['mol_id'] = candidates_df['inchikey_s']
	candidates_df.loc[:,"mol"] = par_apply_series(candidates_df["smiles"],data_utils.mol_from_smiles)
	candidates_df.loc[:,"smiles"] = par_apply_series(candidates_df["mol"],data_utils.mol_to_smiles)
	candidates_df.loc[:,"mol"] = par_apply_series(candidates_df["smiles"],data_utils.mol_from_smiles)
	
 	#print(f"> Save to {candidates_fp}")
	candidate_name = 'casmi16_cat3'
	os.makedirs(args.candidate_dp, exist_ok=True)
	args_fp = os.path.join(args.candidate_dp, f'{candidate_name}.yaml')
	candidates_fp = os.path.join(args.candidate_dp,f'{candidate_name}.pkl.gz')

	candidates_df.reset_index(drop=True, inplace=True)
	candidates_df.to_pickle(candidates_fp)