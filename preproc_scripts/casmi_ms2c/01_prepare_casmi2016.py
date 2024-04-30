
import os
import pickle
import re
import signal
import sys
import traceback
from threading import Event
import argparse
import os
import io

import tarfile
import bz2
import zipfile
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


def parse_casmi16_mgf_str(mgf_text_list):
	peaks = []
	for row in mgf_text_list:
		row = row.strip()
		if re.match(r"^(\d+(?:\.\d+)?)\s(\d+(?:\.\d+)?)", row):
			mz_value,int_value = [float(i) for i in row.split()]
			peaks.append((mz_value,int_value))
	return  peaks

if __name__ == "__main__":

	RDLogger.DisableLog('rdApp.*') 

	casmi_cat1_df = pd.read_csv('data/misc/casmi2016/solutions_casmi2016_cat1.csv')
	casmi_cat23_df = pd.read_csv('data/misc/casmi2016/solutions_casmi2016_cat2and3.csv')
	casmi_cat1_df['cat'] = 1
	casmi_cat23_df['cat'] = 2
	casmil_df = pd.concat([casmi_cat1_df,casmi_cat23_df])

	casmil_df.loc[:,"mol"] = par_apply_series(casmil_df["SMILES"],data_utils.mol_from_smiles)
	casmil_df.loc[:,"smiles"] = par_apply_series(casmil_df["mol"],data_utils.mol_to_smiles)
	casmil_df.loc[:,"inchikey_s"] = par_apply_series(casmil_df["mol"],data_utils.mol_to_inchikey_s)
	casmil_df.loc[:,"scaffold"] = par_apply_series(casmil_df["mol"],data_utils.get_murcko_scaffold)
	casmil_df.loc[:,"formula"] = par_apply_series(casmil_df["mol"],data_utils.mol_to_formula)
	casmil_df.loc[:,"inchi"] = par_apply_series(casmil_df["mol"],data_utils.mol_to_inchi)
	casmil_df.loc[:,"mw"] = par_apply_series(casmil_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=False))
	casmil_df.loc[:,"exact_mw"] = par_apply_series(casmil_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True))
	casmil_df.loc[:,"num_atoms"] = par_apply_series(casmil_df["mol"],data_utils.mol_to_num_atoms)
	casmil_df.loc[:,"num_bonds"] = par_apply_series(casmil_df["mol"],data_utils.mol_to_num_bonds)
	casmil_df.loc[:,"charge"] = par_apply_series(casmil_df["mol"],data_utils.mol_to_charge)
	casmil_df.loc[:,"single_mol"] = par_apply_series(casmil_df["mol"],data_utils.check_single_mol)
	casmil_df.loc[:,"num_radicals"] = par_apply_series(casmil_df["mol"],data_utils.mol_to_num_radicals)
	
	proc_dp = "data/ms2c/casmi2016/proc"
	casmil_df.to_pickle(os.path.join(proc_dp,'mol_df.pkl'))
	
	cat23_neg_zip_fp = "data/misc/casmi2016/CASMI2016_Cat2and3_Challenge_negative_mgf.zip"
	cat23_pos_zip_fp = "data/misc/casmi2016/CASMI2016_Cat2and3_Challenge_positive_mgf.zip"
	
	print(f">read mgf from {cat23_pos_zip_fp}")
	archive = zipfile.ZipFile(cat23_pos_zip_fp,  'r')
	spec_data = []
	for file in tqdm(archive.namelist()):
		if file.endswith('mgf'):
			#print(file)
			text_list = archive.read(file).decode('utf-8').split('\n')
			#print(text_list)
			peaks = parse_casmi16_mgf_str(text_list)
			mol_id = file.split('.')[0]
			print(file)
			spec_data.append([mol_id,])
			#print(mz_array, intensity_array)
			exit()
			#spectra_list = parse_mgf_text_list(text_list, id_prefix = file.split('.')[0])
			#spectra_task_inputs.append([spectra_list,f"{file.split('.')[0]}.pickle.gz", sqlite_db_path])
			#basename = os.path.basename(file)
			#spectra_inputs.append(spectra_list)
			#case_results_fp_inputs.append(join(results_dir, f"{basename.split('.')[0]}.pickle.gz"))
			#id_spectrum_list_hb(spectra_list, case_results_fp_inputs[-1], sqlite_db_path)