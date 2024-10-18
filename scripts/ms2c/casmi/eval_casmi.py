
import os
import pandas as pd
import argparse

from tqdm import tqdm
import glob


LAST_RANK = 51
LAST_SCORE = 0.0

def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--predicted_fp", type=str,default="./data/ms2c/casmi2016/predicted/pos.pkl", help="")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/casmi2016/proc/", help="")
	parser.add_argument("--frag_dp", type=str,default="./data/ms2c/casmi2016/frag", help="frag dp only used in fragnnet")
	args = parser.parse_args()
	return args

if __name__ == "__main__":

	args = get_args()
	#predicted_spec_df = pd.read_pickle(args.predicted_fp)
	#print(predicted_spec_df.head())
	#os.makedirs(args.save_dp, exist_ok=True)
	print(f"> {args.proc_dp}")
	print(f"> {args.predicted_fp}")
	#print(f"> {args.save_dp}")

	print("> load target_spec_df")
	spec_input_df = pd.read_pickle(os.path.join(args.proc_dp,"spec_df.pkl.gz"))
	mol_input_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl.gz"))
	mol_input_df = mol_input_df[['mol_id','inchikey_s','smiles']]
	spec_input_df = spec_input_df.merge(mol_input_df, left_on='mol_id', right_on='mol_id')
	#print(spec_input_df.columns)
	spec_input_df = spec_input_df[['target_group_id', 'target_inchikey_s', 'smiles', 'inchikey_s', 'spec_id']]
	#print(len(spec_input_df))
	#spec_input_df = spec_input_df.drop_duplicates(subset=['group_id'])
	# Get a list of all .gz files in the directory

	predicted_spec_df = pd.read_pickle(args.predicted_fp)
 
	#predicted_spec_df['group_id'] =  predicted_spec_df['spec_id']
	predicted_spec_df = predicted_spec_df.merge(spec_input_df, left_on='spec_id', right_on='spec_id')
	predicted_spec_df['cos_sim_0.01_rank'] = predicted_spec_df.groupby('target_group_id')['cos_sim_0.01'].rank(ascending=False, method='max')
	#predicted_spec_df['cos_hun_rank'] = predicted_spec_df.groupby('target_group_id')['cos_hun'].rank(ascending=False, method='max')
	total_tasks =  predicted_spec_df['target_group_id'].nunique()
 
	for idx,grp in predicted_spec_df.groupby('target_group_id'):
		print(grp[['cos_sim_0.01','cos_sim_0.01_rank']])
	print(total_tasks)
	stats_df = predicted_spec_df.groupby('target_group_id').apply(lambda x: x[x['target_inchikey_s'] == x['inchikey_s']].nsmallest(1, 'cos_sim_0.01_rank'), include_groups=False).reset_index(drop=True)
	print(len(stats_df))
	for rank_type in ['cos_sim_0.01']:
		top_stats = [rank_type]
		for k in [1,3,5,10]:
			print(len(stats_df[stats_df[rank_type+'_rank'] <= k]), len(stats_df[stats_df[rank_type+'_rank'] <= k])/total_tasks)
		#top_stats.append(stats_df[rank_type].mean())
		#top_stats.append(stats_df[rank_type].std())

