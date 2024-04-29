import pandas as pd
import argparse
import os

def get_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--proc_dp", type=str, default="data/proc/nist20mona23")
	parser.add_argument("--dsets", type=str, nargs="+", default=["nist20_hr"])
	parser.add_argument("--prec_types", type=str, nargs="+", default=["[M+H]+"])
	parser.add_argument("--inst_types", type=str, nargs="+", default=["FT"])
	parser.add_argument("--base_split_dp", type=str, default="data/split/nist20_mona23_d4_wr00_inchikey")
	parser.add_argument("--dst_split_dp", type=str, default="data/split/nist20_mona23_d4_wr00_inchikey_casmi16") 
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	casmsi_cat1_df = pd.read_csv('data/misc/casmi2016/solutions_casmi2016_cat1.csv')
	casmsi_cat23_df = pd.read_csv('data/misc/casmi2016/solutions_casmi2016_cat2and3.csv')
	casmsi_inchikeys= list(casmsi_cat1_df['INCHIKEY']) + list(casmsi_cat23_df['INCHIKEY'])
	casmsi_inchikeys_s = [x.split('-')[0] for x in casmsi_inchikeys]
 	#print(casmsi_cat1_df)
	#print(casmsi_cat23_df)
	
	args = get_args()
	mol_df_fp = os.path.join(args.proc_dp,"mol_df.pkl")
	mol_df = pd.read_pickle(mol_df_fp)
	mol_id_list = mol_df[mol_df['inchikey_s'].isin(casmsi_inchikeys_s)]['mol_id'].values.tolist()
 
	spec_df_fp = os.path.join(args.proc_dp,"spec_df.pkl")
	spec_df = pd.read_pickle(spec_df_fp)
	#print(spec_df[spec_df['mol_id'].isin(mol_id_list)])
	selected_spec_df = spec_df[spec_df['mol_id'].isin(mol_id_list)]
	#selected_spec_df = selected_spec_df[selected_spec_df['dset']]
	#print(selected_spec_df.columns)
	
	selected_spec_df = selected_spec_df[selected_spec_df['dset'].isin(args.dsets)]
	selected_spec_df = selected_spec_df[selected_spec_df['prec_type'].isin(args.prec_types)]
	selected_spec_df = selected_spec_df[selected_spec_df['inst_type'].isin(args.inst_types)]
	#print(selected_spec_df)
	#selected_spec_df = selected_spec_df[selected_spec_df['dset'].isin(args.dsets)]
	
	train_split_df = pd.read_csv(os.path.join(args.base_split_dp, 'train_ids.csv'))
	val_split_df = pd.read_csv(os.path.join(args.base_split_dp, 'val_ids.csv'))
	test_split_df = pd.read_csv(os.path.join(args.base_split_dp, 'test_ids.csv'))
 
	move_from_train = train_split_df[train_split_df['group_id'].isin(selected_spec_df['group_id'])]
	move_from_val = val_split_df[val_split_df['group_id'].isin(selected_spec_df['group_id'])]
 
	train_split_df = train_split_df[~train_split_df['group_id'].isin(selected_spec_df['group_id'])]
	val_split_df = val_split_df[~val_split_df['group_id'].isin(selected_spec_df['group_id'])]
	
	os.makedirs(args.dst_split_dp, exist_ok=True)
	train_split_df.to_csv(os.path.join(args.dst_split_dp, 'train_ids.csv'), index=False)
	val_split_df.to_csv(os.path.join(args.dst_split_dp, 'val_ids.csv'), index=False)
	
	test_split_df = pd.concat([test_split_df, move_from_train, move_from_val])
	test_split_df.to_csv(os.path.join(args.dst_split_dp, 'test_ids.csv'), index=False)
	#test_split_df = test_split_df[test_split_df['spec_id'].isin(selected_spec_df['spec_id'])]