import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from pprint import pformat
import torch as th
import yaml

import shutil

if __name__ == "__main__":
	spec_df_dp = './data/proc/pcdl/spec_df.pkl'
	spec_df = pd.read_pickle(spec_df_dp)
	last_group_id = spec_df['group_id'].max()
	#print(spec_df)
	
	nl_spec_df = spec_df.copy()
	nl_spec_df = nl_spec_df.drop("group_id",axis = 1)
	nl_spec_df["prec_type"] = nl_spec_df["prec_type"] + "_NL"
	nl_spec_df["peaks"] = nl_spec_df.apply(lambda x : [(x['prec_mz']-i,j) for i,j in x["peaks"] if abs(x['prec_mz']-i) > 1], axis =1)
	nl_spec_df = nl_spec_df[nl_spec_df['peaks'].map(lambda x: len(x)) > 0]

	# set group_id (same compound and precursor)
	nl_group_df = nl_spec_df[["mol_id","prec_type"]].copy().drop_duplicates()
	nl_group_df.loc[:,"group_id"] = np.arange(nl_group_df.shape[0])
	nl_spec_df = nl_spec_df.merge(nl_group_df,how="inner")

	nl_spec_df['group_id'] += last_group_id + 1
	print(nl_spec_df["prec_type"].value_counts())
	print(nl_spec_df["inst_type"].unique())

	extended_spectra_df = pd.concat([spec_df, nl_spec_df])
	extended_spectra_df.reset_index(inplace=True,drop=True)
	extended_spectra_df['spec_id'] = extended_spectra_df.index
	extended_spectra_df.to_pickle('./data/proc/pcdl_nl/spec_df.pkl')

	print(extended_spectra_df["prec_type"].value_counts())
	print(extended_spectra_df["inst_type"].unique())

	src_dst_list = [
		["./data/split/pcdl", "./data/split/pcdl_nl"],
		["./data/split/pcdl/cv0", "./data/split/pcdl_nl/cv0"],
		["./data/split/pcdl/cv1", "./data/split/pcdl_nl/cv1"],
		["./data/split/pcdl/cv2", "./data/split/pcdl_nl/cv2"],
		["./data/split/pcdl/cv3", "./data/split/pcdl_nl/cv3"],
		["./data/split/pcdl/cv4", "./data/split/pcdl_nl/cv4"],
	]
	for src_split_dir, dst_split_dir in src_dst_list:
		src_split_dir = "./data/split/pcdl"
		dst_split_dir = "./data/split/pcdl_nl"

		train_ids_df = pd.read_csv(src_split_dir + "/train_ids.csv")
		val_ids_df = pd.read_csv(src_split_dir + "/val_ids.csv")
		test_ids_df = pd.read_csv(src_split_dir + "/test_ids.csv")

		
		ext_train_ids_df = extended_spectra_df[extended_spectra_df['mol_id'].isin(train_ids_df['mol_id'].tolist())][['spec_id','mol_id','group_id']]
		ext_val_ids_df = extended_spectra_df[extended_spectra_df['mol_id'].isin(val_ids_df['mol_id'].tolist())][['spec_id','mol_id','group_id']]
		ext_test_ids_df = extended_spectra_df[extended_spectra_df['mol_id'].isin(test_ids_df['mol_id'].tolist())][['spec_id','mol_id','group_id']]
		
		ext_train_ids_df.to_csv(dst_split_dir + "/train_ids.csv",index=False)
		ext_val_ids_df.to_csv(dst_split_dir + "/val_ids.csv",index=False)
		ext_test_ids_df.to_csv(dst_split_dir + "/test_ids.csv",index=False)

		#print(train_ids)
