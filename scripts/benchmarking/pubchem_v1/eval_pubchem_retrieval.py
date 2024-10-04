
import os
import pandas as pd
import json
import time
import argparse
import numpy as np
import torch as th
from tqdm import tqdm

from frag_gnn.utils.spec_utils import calculate_sparse_cosine_similarity, mf1000_normalize_np
from frag_gnn.utils.data_utils import par_apply

LAST_RANK = 50
LAST_SCORE = 0.0

def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_dp",type=str,default="./data/ms2c/pubchem/benchmark/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_fraggnet_d3")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/pubchem/predicted_input/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50")
	parser.add_argument("--predicted_dp", type=str,default="./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_fraggnet_d3")
 
	args = parser.parse_args()
	return args


if __name__ == "__main__":


	args = get_args()
 
	os.makedirs(args.save_dp, exist_ok=True)
	print(f"> {args.proc_dp}")
	print(f"> {args.predicted_dp}")
	print(f"> {args.save_dp}")


	def get_rank(true_mzs, true_ints, true_smiles, predict_spec_mol_df):
		#identifier = row['identifier']
		#true_smiles = row['smiles']

		# nothing predicted last rank
		#if len(predict_spec_mol_df) == 0:
		#	return (identifier,LAST_RANK, LAST_SCORE, None)

		true_mzs = th.tensor(true_mzs)
		true_ints = th.tensor(true_ints)

		predict_spec_mol_df['cos_sim_0.01'] = predict_spec_mol_df.apply(lambda x: calculate_sparse_cosine_similarity(true_mzs,true_ints,
                                                                              th.zeros(len(true_mzs), dtype=th.long), 
																				th.tensor(x['pred_mzs']), 
																				th.tensor(x['pred_ints']), 
																				th.zeros(len(x['pred_mzs']), dtype=th.long))[0].item() ,axis=1)

		predict_spec_mol_df['cos_sim_0.01'] = predict_spec_mol_df['cos_sim_0.01']
		predict_spec_mol_df['rank'] = predict_spec_mol_df['cos_sim_0.01'].rank(ascending = False)
		
		ranks = predict_spec_mol_df[predict_spec_mol_df['smiles'] == true_smiles]['rank'].values
		scores = predict_spec_mol_df[predict_spec_mol_df['smiles'] == true_smiles]['cos_sim_0.01'].values
		rank_value = ranks[0] if len(ranks) else LAST_RANK
		score_value = scores[0] if len(ranks) else LAST_SCORE


		#spec_df['identifier'] = identifier
		#spec_df = spec_df[['identifier', 'spec_id', 'mol_id', 'smiles', \
      	#	'inchikey', 'formula', 'exact_mw', 'cos_sim_0.01', 'rank']]
		#
		# print(rank_value, score_value)
		# print(predict_spec_mol_df.head())
		# print(predict_spec_mol_df[predict_spec_mol_df['smiles'] == true_smiles])
		return rank_value, score_value, predict_spec_mol_df

	print("> load mol_df")
	mol_df = pd.read_pickle(os.path.join(args.proc_dp,'mol_df.pkl.gz'))

	#print("> load spec_fp")
	#spec_df = pd.read_pickle(os.path.join(args.proc_dp,'spec_df.pkl.gz'))
 
	print("> load target_spec_df")
	target_spec_df = pd.read_pickle(os.path.join(args.proc_dp,"target_spec_df.pkl.gz"))

	#for target_id in tqdm(target_spec_df[""]):
	result_l = []
	for index, row in tqdm(target_spec_df.iterrows(), len=len(target_spec_df)):
		#print(row)
		target_id = row["target_mol_id"]
		true_mzs = row['spec_mzs'] #[float(x) for x in row['spec_mzs'].split(",")]
		true_ints = mf1000_normalize_np(row['spec_ints'])
		true_smiles = row['target_mol_smiles']
		save_fp = os.path.join(args.save_dp, f"{target_id}.pkl.gz")

		if os.path.isfile(save_fp) and args.allow_cached:
			continue

		#spec_df_grp = spec_df[ spec_df['target_mol_id'] == target_id]

		
		predicted_gz_fp = os.path.join(args.predicted_dp,f"{target_id}.pkl.gz")
		predict_spec_mol_df = pd.read_pickle(predicted_gz_fp)
		mol_df_grp = mol_df[ mol_df['mol_id'].isin(predict_spec_mol_df['mol_id'])]
		predict_spec_mol_df = predict_spec_mol_df.merge(mol_df_grp, on = 'mol_id', how = 'left')
		rank_value, score_value, predict_spec_mol_df = get_rank(true_mzs, true_ints, true_smiles, predict_spec_mol_df)
		result_l.append((target_id, rank_value, score_value))

	#for idx, row in tqdm(msg_id_map_df.iterrows(),total=len(msg_id_map_df), desc="prepare ranking inputs"):
	#	input_l.append(row)

	#result_l = []
	#details_l = []
	#rank_results = par_apply(iter(input_l), get_rank, False, return_as_generator=True)
	#for rank_results in tqdm(rank_results, total = len(input_l), desc = "ranking", leave=None):
	#	identifier, rank_value, score_value, spec_df = rank_results
	#	result_l.append((identifier, rank_value, score_value))
	#	details_l.append(spec_df)
  
	result_df = pd.DataFrame(result_l,columns = ['target_id','rank', 'score'])
	result_df.to_csv(os.path.join(args.save_dp, "summary.csv"),index = False)
 
	for k in [1,3,5,10]:
			print(k, len(result_df[result_df["rank"] <= k]), len(result_df[result_df["rank"] <= k])/len(result_df))
  
	#details_df = pd.concat(details_l)
	#details_df.to_csv(os.path.join(args.save_dp, "details.csv"),index = False)