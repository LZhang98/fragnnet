from io import StringIO
import urllib.request
import os
from datetime import date
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from tqdm import tqdm
from rdkit import Chem

import gzip
import re
import pandas as pd
import multiprocessing
import concurrent.futures
import signal
import hashlib
from threading import Event
from frag_gnn.utils.ms2c_utils import MolCandidateDB
import numpy as np
import frag_gnn.utils.data_utils as data_utils

if __name__ == "__main__":

	data_dir = "./data/ms2c/hmdb"
	print(f"> Data dir: {data_dir}")

	hmdb_csv_fp = 'data/ms2c/hmdb/hmdb5_mols.csv'
	"""
	hmdb_df = pd.read_csv(hmdb_csv_fp)
	hmdb_df.loc[:,"mol"] = data_utils.par_apply_series(hmdb_df["smiles"],data_utils.mol_from_smiles)
	hmdb_df.loc[:,"exact_mass"] = data_utils.par_apply_series(hmdb_df["mol"],lambda mol: data_utils.mol_to_mol_weight(mol,exact=True))
	hmdb_df.loc[:,"formula"] = data_utils.par_apply_series(hmdb_df["mol"], data_utils.mol_to_formula)
	#hmdb_df = hmdb_df.drop(columns=["mol"])
	hmdb_df.to_csv(hmdb_csv_fp, index=None)
	"""
	hmdb_df = pd.read_csv(hmdb_csv_fp)
	hmdb_df.loc[:,"cid"] = data_utils.par_apply_series(hmdb_df["hmdb_id"], lambda x: int(x[4:]))
 
	df = hmdb_df[['cid','inchikey','exact_mass','formula','smiles']]
	df = df.dropna()

	db_file  = f"{data_dir}/hmdb5.sqlite"
	with MolCandidateDB(db_file) as db:
		db._go_fast_at_all_cost()
		db.create_tables()
  
		df = df.astype({'cid':int, 'inchikey':'string', 'exact_mass':float, 'formula':'string','smiles':'string'})
		db.add_compounds_from_df(df)