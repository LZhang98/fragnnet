
import pandas as pd
from tqdm import tqdm
import argparse
import os

def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--per_grp_dir",type=str,default="./data/ms2c/pubchem/results/fraggnn_d4_pubchem_nist20_d4_wr00_inchikey_test")
	parser.add_argument("--result_csv",type=str,default="./data/benchmark/icml_ms2c/inchikey/fragnnet_d4.csv")
	#parser.add_argument("--per_grp_dir",type=str,default="./data/ms2c/pubchem/results/iceberg_pubchem_nist20_d4_wr00_inchikey_test")
	#parser.add_argument("--result_csv",type=str,default="./data/benchmark/icml_ms2c/inchikey/iceberg_inchikey.csv")
	
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = get_args()
	per_grp_dir =  args.per_grp_dir
	pkls = [f for f in os.listdir(per_grp_dir) if f.endswith(".pkl") or f.endswith(".pkl.gz")]

	results_l = []
	ranking_methods = ['cos_sim_hun', 'cos_sim_bin']
	
	for pkl in tqdm(pkls):
		pkl_fp = os.path.join(per_grp_dir, pkl)
		try:
			grp = pd.read_pickle(pkl_fp)
		except Exception as e:
			print(pkl_fp, e)
		if len(grp) == 0:
			print(pkl_fp)
			continue
		grp_id = pkl.split('.')[0]
		target_spec_id = grp['spec_id'][0]
		results_row_l = [grp_id]
		for ranking_method in ranking_methods:
			grp = grp.sort_values(ranking_method, ascending = False)
			grp[f'{ranking_method}_rank'] = grp[ranking_method].rank(ascending=False, method='max')
			rank = grp[grp['spec_id'] == target_spec_id][f'{ranking_method}_rank'][0]
			results_row_l.append(rank)
			#score = grp[grp['spec_id'] == target_spec_id][ranking_method][0]
			#if score == grp[ranking_method].max() and rank != 1.0:
			#	print(score, rank)
			#	print(grp[grp[ranking_method] == score])
		results_l.append(results_row_l)
  
	df = pd.DataFrame(results_l, columns=['grp_id'] + ranking_methods)
	df = df.sort_values(by = 'grp_id') 
	df.to_csv(args.result_csv, index = False )
	print(df)
	for ranking_method in ranking_methods:
		print(ranking_method)
		for k in [1,2,3,5,10]:
			print(k, len(df[df[ranking_method] <= k]), len(df[df[ranking_method] <= k])/ len(df))