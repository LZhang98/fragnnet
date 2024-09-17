
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
from copy import deepcopy
import yaml
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows, seq_apply_series, seq_apply_df_rows

def get_args():
	parser = argparse.ArgumentParser()
	
	# frag candidates configs
	parser.add_argument("--proc_dp", type=str, default="./data/proc/nist20mona23")
	parser.add_argument("--split_dp", type=str, default="./data/split/nist20_d4_wr00_inchikey")
	parser.add_argument("--db_file", type=str, default= "./data/ms2c/pubchem/pubchem.sqlite")
	parser.add_argument("--tolerance", type=float, default= 10, help="Default 10ppm")
	parser.add_argument("--use_ppm", type=booltype, default= True)
	parser.add_argument("--max_candidates", type=int, default= 50, help="max canidates include target itself")
	parser.add_argument("--subset_size", type=int, default=-1)
	parser.add_argument("--fp_type", type=str, default='MORGAN-R2', help="")
	parser.add_argument("--split_type", type=str,default="test", help="")
	parser.add_argument("--candidate_dp",type=str,default='./data/ms2c/pubchem/candidates')
	
	args = parser.parse_args()
	return args

def filter_candidates(db_results, target_smiles, target_mol_id, top_k = None, morgen_radius = 3):

	chem, rd_fpgen, rdmolops, rd_ds, rd_moldesc, rd_logger = data_utils.rdkit_import("rdkit.Chem",
																			   "rdkit.Chem.rdFingerprintGenerator", 
																			   "rdkit.Chem.rdmolops", 
																			   "rdkit.DataStructs", 
																			   "rdkit.Chem.rdMolDescriptors",
																			   "rdkit.RDLogger")
	
	rd_logger.DisableLog('rdApp.*') 

	mfpgen = rd_fpgen.GetMorganGenerator(radius=morgen_radius)
	t_mol = chem.MolFromSmiles(target_smiles)
	t_fp = mfpgen.GetFingerprint(t_mol)

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
		#tanimoto  distance
		case_cdf.at[idx,"tanimoto"] = rd_ds.TanimotoSimilarity(t_fp, mfpgen.GetFingerprint(mol))

	case_cdf = case_cdf[case_cdf['tanimoto'] >= 0.0]
	case_cdf = case_cdf.sort_values('tanimoto',ascending = False)
	
	if top_k is not None:
		case_cdf = case_cdf[:top_k]

	return target_mol_id, target_smiles, case_cdf

def fetch_db(db_file, mw, mass_tol ):
	with MolCandidateDB(db_file) as db:
		db_results = db.get_compounds_by_exact_mass_range(mw -mass_tol, mw + mass_tol)
	return db_results

if __name__ == "__main__":

	RDLogger.DisableLog('rdApp.*') 

	args = get_args()


	split_configs_d = {
		"test":"test_ids.csv",
		"val":"val_ids.csv",
		"train":"train_ids.csv"
	}

	task_configs = []
	#for split in args.included_splits:
	if args.split_type not in ["test", "val", "train"]:
		print("Error, unknown split name")
		exit(0)

	# disable rdkit warning
	if args.fp_type == 'MORGAN-R2':
		morgen_radius = 2
	elif args.fp_type == 'MORGAN-R3':
		morgen_radius = 3
	else:
		print("Not Supported Fingerprint")

	split_basename = os.path.basename(os.path.normpath(args.split_dp))
	split_type = args.split_type
	split_csv_filename = split_configs_d[split_type]
	tol_type = f"{args.tolerance}Da" if not args.use_ppm else f"{args.tolerance}ppm" 
	candidate_name = f'pubchem_{split_basename}_{split_type}_{tol_type}_{args.fp_type}_{args.max_candidates}'
	if args.subset_size > 0:
		candidate_name += f'_s{args.subset_size}'
	#candidate_dp = os.path.join(args.candidate_dp, split_basename)
	os.makedirs(args.candidate_dp, exist_ok= True)
	args_fp = os.path.join(args.candidate_dp, f'{candidate_name}.yaml')

	candidates_fp = os.path.join(args.candidate_dp,f'{candidate_name}.pkl.gz')

	mol_df_fp = os.path.join(args.proc_dp,"mol_df.pkl")
	print(f"> Read mol df from {mol_df_fp}")
	mol_df = pd.read_pickle(mol_df_fp)

	id_fp = os.path.join(args.split_dp, split_csv_filename)
	print(f"> Processing {split_type} split, read ids from {id_fp}")
	#val_sp_fp = os.path.join(args.split_dp,"val_ids.csv")
	sp_mol_ids = pd.read_csv(id_fp)['mol_id'].drop_duplicates().to_frame()
	sp_mol_df = sp_mol_ids.merge(mol_df, how="left", on="mol_id")

	db_file = args.db_file
	db_query_input = []
	filter_candidates_input_l = []
	if args.subset_size > 0:
		sp_mol_df = sp_mol_df[:args.subset_size]
	for _, row in tqdm(sp_mol_df.iterrows(), total = len(sp_mol_df), desc = "Prepare pubchem database input"):
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
		filter_candidates_input_l.append([None, row['smiles'],row['mol_id'], args.max_candidates, morgen_radius])

	db_results = data_utils.par_apply(iter(db_query_input),fetch_db, True, return_as_generator=True)	
	for idx, res in enumerate(tqdm(db_results, total =  len(db_query_input), desc = "Fetch pubchem database ")):
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
	print(f"> Save to {candidates_fp}")
	candidates_df.reset_index(drop=True, inplace=True)
	candidates_df.to_pickle(candidates_fp)