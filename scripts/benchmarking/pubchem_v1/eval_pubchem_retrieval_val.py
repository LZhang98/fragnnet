
import os
import pandas as pd
import argparse

from tqdm import tqdm
import glob


LAST_RANK = 51
LAST_SCORE = 0.0

def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_dp",type=str,default="./data/ms2c/pubchem/benchmark/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50")
	parser.add_argument("--predicted_dp", type=str,default="/home/feiw/gitRepo/frag-gnn/data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50")
 
	args = parser.parse_args()
	return args


if __name__ == "__main__":


	args = get_args()
	#predicted_spec_df = pd.read_pickle(args.predicted_fp)
	#print(predicted_spec_df.head())
	os.makedirs(args.save_dp, exist_ok=True)
	print(f"> {args.proc_dp}")
	print(f"> {args.predicted_dp}")
	print(f"> {args.save_dp}")

	print("> load target_spec_df")
	spec_input_df = pd.read_pickle(os.path.join(args.proc_dp,"spec_df.pkl.gz"))
	mol_input_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl.gz"))
	mol_input_df = mol_input_df[['mol_id','inchikey_s','smiles']]
	spec_input_df = spec_input_df.merge(mol_input_df, left_on='mol_id', right_on='mol_id')
	#print(spec_input_df.columns)
	spec_input_df = spec_input_df[['group_id', 'target_mol_id', 'target_mol_smiles', 'target_inchikey_s', 'smiles', 'inchikey_s', 'tanimoto']]
	#print(len(spec_input_df))
	spec_input_df = spec_input_df.drop_duplicates(subset=['group_id'])
	# Get a list of all .gz files in the directory
	gz_files = glob.glob(os.path.join(args.predicted_dp, '*.gz'))

	# Extract and print the basenames (without the .gz extension)
	overall_stats_rows = []
	for gz_file in tqdm(gz_files):
		basename = os.path.basename(gz_file).replace('.pkl.gz', '')
		#print(basename)
		predicted_spec_df = pd.read_pickle(gz_file)
		#predicted_spec_df['group_id'] =  predicted_spec_df['spec_id']
		predicted_spec_df = predicted_spec_df.merge(spec_input_df, left_on='spec_id', right_on='group_id')
		predicted_spec_df['cos_sim_0.01_rank'] = predicted_spec_df.groupby('target_mol_id')['cos_sim_0.01'].rank(ascending=False)
		predicted_spec_df['cos_hun_rank'] = predicted_spec_df.groupby('target_mol_id')['cos_hun'].rank(ascending=False)
		stats_df = predicted_spec_df.groupby('target_mol_id').apply(lambda x: x[x['target_mol_smiles'] == x['smiles']], include_groups=False)

		for rank_type in ['cos_sim_0.01_rank', 'cos_hun_rank']:
			top_stats = [basename,rank_type]
			for k in [1,3,5,10]:
				top_stats.append(len(stats_df[stats_df[rank_type] <= k]))
				top_stats.append(len(stats_df[stats_df[rank_type] <= k])/len(stats_df))
			overall_stats_rows.append(top_stats)
		stats_df = stats_df[['group_id','smiles', 'inchikey_s', 'tanimoto','cos_sim_0.01','cos_hun','cos_sim_0.01_rank','cos_hun_rank']]
		stats_df.to_csv(os.path.join(args.save_dp, f"{basename}_summary.csv"),index = False)

	overall_stats_df = pd.DataFrame(overall_stats_rows, \
    	columns = ['model','ranking method', 'top-1','top-1 ratio','top-3','top-3 ratio','top-5','top-5 ratio','top-10','top-10 ratio'])
	overall_stats_df = overall_stats_df[['model','ranking method', 'top-1','top-3','top-5','top-10','top-1 ratio','top-3 ratio','top-5 ratio','top-10 ratio']]
	overall_stats_df = overall_stats_df.sort_values(by='model', ascending=True)

 	#result_df = pd.DataFrame(result_l,columns = ['target_id','rank', 'score'])
	overall_stats_df.to_csv(os.path.join(args.save_dp, "overall_stats_summary.csv"),index = False)
 
