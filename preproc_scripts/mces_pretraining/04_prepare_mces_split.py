import argparse
import os
import pandas as pd
import numpy as np
import json

from fragnnet.utils import frag_utils
from fragnnet.utils.misc_utils import np_temp_seed

def make_splits(args):

	mol_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl"))
	mces_data = pd.read_hdf(os.path.join(args.proc_dp,"mces_data.h5"))

	print("> Spectrum filters")

	# Apply spectrum filters
	print("> create split(s)")
	split_data_list = []
	
 	# split based on molecule
	# primary split
	primary_split_mol_id = np.unique(primary_split_df["mol_id"])
	split_keys = primary_split_df[primary_split_df["mol_id"].isin(primary_split_mol_id)][args.split_key]
	split_keys = np.unique(split_keys)
	if args.total_frac == 1.0:
		total_num = split_keys.shape[0]
	else:
		total_num = int(np.ceil(split_keys.shape[0]*args.total_frac))
	
	if args.split_type == "random":
		test_num = int(np.ceil(total_num*args.test_frac))
		val_num = int(np.ceil(total_num*args.val_frac))
		with np_temp_seed(args.meta_rseed):
			if args.total_frac < 1.0:
				split_keys = np.random.choice(split_keys,size=total_num,replace=False)
			test_keys = np.random.choice(split_keys,size=test_num,replace=False)
			train_val_keys = np.setdiff1d(split_keys,test_keys)
			val_keys = np.random.choice(train_val_keys,size=val_num,replace=False)
			train_keys = np.setdiff1d(train_val_keys,val_keys)
		
		train_df = primary_split_df[primary_split_df[args.split_key].isin(train_keys)][["spec_id","mol_id","group_id"]]
		val_df = primary_split_df[primary_split_df[args.split_key].isin(val_keys)][["spec_id","mol_id","group_id"]]
		test_df = primary_split_df[primary_split_df[args.split_key].isin(test_keys)][["spec_id","mol_id","group_id"]]

		# secondary split
		secondary_df = secondary_split_df[~secondary_split_df[args.split_key].isin(train_val_keys)][["spec_id","mol_id","group_id"]]
		split_data = { 
			"split_dp": args.split_dp,
			"train_df": train_df,
			"val_df": val_df,
			"test_df": test_df,
			"secondary_df": secondary_df
		}
		split_data_list.append(split_data)

	split_data = { 
			"split_dp": args.split_dp,
			"train_df": train_df,
			"val_df": val_df,
			"test_df": test_df,
			"secondary_df": secondary_df
	}
	split_data_list.append(split_data)

	else:
		raise ValueError(f"{args.split_type} is not supported")

	# create split directory
	for split_data in split_data_list:
		os.makedirs(split_data['split_dp'],exist_ok=True)
		print(f"> Save split to {split_data['split_dp']}")

		print("-" * 16)
		for split_type in ["train_df","val_df","test_df","secondary_df"]:
			print(">", split_type)
			print("> number of unique mols", split_data[split_type]['mol_id'].nunique())
			print("> number of unique spec", split_data[split_type]['spec_id'].nunique())
			print("> number of unique groups", split_data[split_type]['group_id'].nunique())
			print("-" * 16)
		
		# save ids
		train_fp = os.path.join(split_data['split_dp'],"train_ids.csv")
		val_fp = os.path.join(split_data['split_dp'],"val_ids.csv")
		test_fp = os.path.join(split_data['split_dp'],"test_ids.csv")
		secondary_fp = os.path.join(split_data['split_dp'],"secondary_ids.csv")
		split_data['train_df'].to_csv(train_fp,index=False)
		split_data['val_df'].to_csv(val_fp,index=False)
		split_data['test_df'].to_csv(test_fp,index=False)
		split_data['secondary_df'].to_csv(secondary_fp,index=False)
		
		# save split metadata
		meta_d = vars(args)
		meta_fp = os.path.join(split_data['split_dp'],"meta.json")
		with open(meta_fp,"w") as f:
			json.dump(meta_d,f,indent=4)

	return

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--split_type",type=str,choices=["random"],default="random")
	parser.add_argument("--split_key",type=str,choices=["inchikey_s","scaffold"],default="inchikey_s")
	parser.add_argument("--elements",type=str,nargs="+",default=frag_utils.ELEMENTS)

	# non-filtering args
	parser.add_argument("--meta_rseed",type=int,default=420420)
	parser.add_argument("--total_frac",type=float,default=1.0)
	parser.add_argument("--test_frac",type=float,default=0.2)
	parser.add_argument("--val_frac",type=float,default=0.2)
	parser.add_argument("--proc_dp",type=str,required=True)
	parser.add_argument("--split_dp",type=str,required=True)
	args = parser.parse_args()

	make_splits(args)

