import argparse
import os
import pandas as pd

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

	casmi_df['inchikey_s'] = casmi_df['INCHIKEY'].apply(lambda x: x.split('-')[0])
	inchikey_s_l = casmi_df['inchikey_s'].to_list()
	#print(inchikey_s_l)

	mol_df_fp = os.path.join(args.proc_dp,'mol_df.pkl')
	#spec_df_fp = os.path.join(args.proc_dp,'spec_df.pkl')
	print(f'> Read mol df from {mol_df_fp}')
	mol_df = pd.read_pickle(mol_df_fp)

	in_casmi = mol_df[mol_df['inchikey_s'].isin(inchikey_s_l)]
	os.makedirs(args.split_out_dp, exist_ok=True)
	print(f"> from to {args.split_dp}")
	print(f"> output to {args.split_out_dp}")
 
	# update train split
	train_split_df = pd.read_csv(os.path.join(args.split_dp, "train_ids.csv"))
	updated_train_split_df = train_split_df[~train_split_df['mol_id'].isin(in_casmi['mol_id'].to_list())]
	train_to_test_split_df = train_split_df[train_split_df['mol_id'].isin(in_casmi['mol_id'].to_list())]
	print(f"train: {len(train_split_df)} > {len(updated_train_split_df)}" )
	train_to_test_split_df.to_csv(os.path.join(args.split_out_dp, "train_ids.csv"))
 
	# update val split
	val_split_df = pd.read_csv(os.path.join(args.split_dp, "val_ids.csv"))
	updated_val_split_df = val_split_df[~val_split_df['mol_id'].isin(in_casmi['mol_id'].to_list())]
	val_to_test_split_df = val_split_df[val_split_df['mol_id'].isin(in_casmi['mol_id'].to_list())]
	print(f"val: {len(val_split_df)} > {len(updated_val_split_df)}" )
	val_to_test_split_df.to_csv(os.path.join(args.split_out_dp, "val_ids.csv"))
  
	# update test split
	test_split_df = pd.read_csv( os.path.join(args.split_dp, "test_ids.csv"))
	updated_test_split_df = pd.concat([test_split_df,val_to_test_split_df,train_to_test_split_df])
		
	print(f"test: {len(test_split_df)} > {len(updated_test_split_df)}" )
	updated_test_split_df.to_csv(os.path.join(args.split_out_dp, "test_ids.csv"))