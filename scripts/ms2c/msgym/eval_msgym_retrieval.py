
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

LAST_RANK = 257
LAST_SCORE = 0.0

def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_dp",type=str,default="./data/ms2c/msgym/benchmark/retrieval_candidates_formula")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/msgym/proc/retrieval_candidates_formula")
	parser.add_argument("--predicted_dp", type=str,default="./data/ms2c/msgym/fragnnet_predicted/retrieval_candidates_formula")
 
	args = parser.parse_args()
	return args


if __name__ == "__main__":


	args = get_args()
 
	os.makedirs(args.save_dp, exist_ok=True)
	print(f"> {args.proc_dp}")
	print(f"> {args.predicted_dp}")
	print(f"> {args.save_dp}")
	for i in range (3):
		print("This uses a this  a random noise tie break at 1-e5 * np.random.randn")
 
	msg_id_map_fp = os.path.join(args.proc_dp,"msg_id_map.csv")
	msg_id_map_df = pd.read_csv(msg_id_map_fp)

	def get_rank(row):
		identifier = row['identifier']
		true_smiles = row['smiles']

		mol_fp = os.path.join(args.proc_dp, row['mol_df'])
		spec_fp = os.path.join(args.predicted_dp, row['spec_df'])
		
		has_spec_fp = os.path.exists(spec_fp)
		has_spec_fp_gz = os.path.exists(spec_fp+'.gz')
  
		# nothing predicted last rank
		if not has_spec_fp and not has_spec_fp_gz:
			return (identifier,LAST_RANK, LAST_SCORE, None)
		elif not has_spec_fp:
			spec_fp  = spec_fp+'.gz'
   
		if not os.path.exists(mol_fp):
			return (identifier,LAST_RANK, LAST_SCORE, None)

		mol_df = pd.read_pickle(mol_fp)
		spec_df = pd.read_pickle(spec_fp)

		# nothing predicted last rank
		if len(spec_df) == 0:
			return (identifier,LAST_RANK, LAST_SCORE, None)
			
		spec_df = spec_df.merge(mol_df, on = 'mol_id', how = 'left')
  
		true_mzs = [float(x) for x in row['mz'].split(",")]
		true_ints = mf1000_normalize_np(np.array([float(x) for x in row['int'].split(",")]))

		true_mzs = th.tensor(true_mzs)
		true_ints = th.tensor(true_ints)

		spec_df['cos_sim_0.01'] = spec_df.apply(lambda x: calculate_sparse_cosine_similarity(true_mzs,true_ints,
                                                                              th.zeros(len(true_mzs), dtype=th.long), 
																				th.tensor(x['pred_mzs']), 
																				th.tensor(x['pred_ints']), 
																				th.zeros(len(x['pred_mzs']), dtype=th.long))[0].item() ,axis=1)
		# NOTE: this has a random tie break
		# NOTE: this is for msgym paper only
		# NOTE: change this for next project if needed
		spec_df['cos_sim_0.01'] = spec_df['cos_sim_0.01']+ np.random.randn(len(spec_df)) * 1e-5
		spec_df['rank'] = spec_df['cos_sim_0.01'].rank(ascending = False)
		
		ranks = spec_df[spec_df['smiles'] == true_smiles]['rank'].values
		scores = spec_df[spec_df['smiles'] == true_smiles]['cos_sim_0.01'].values
		rank_value = ranks[0] if len(ranks) else LAST_RANK
		score_value = scores[0] if len(ranks) else LAST_SCORE


		spec_df['identifier'] = identifier
		spec_df = spec_df[['identifier', 'spec_id', 'mol_id', 'smiles', \
      		'inchikey', 'formula', 'exact_mw', 'cos_sim_0.01', 'rank']]

		return identifier, rank_value, score_value, spec_df
  
	input_l = []

	for idx, row in tqdm(msg_id_map_df.iterrows(),total=len(msg_id_map_df), desc="prepare ranking inputs"):
		input_l.append(row)

	result_l = []
	details_l = []
	rank_results = par_apply(iter(input_l), get_rank, False, return_as_generator=True)
	for rank_results in tqdm(rank_results, total = len(input_l), desc = "ranking", leave=None):
		identifier, rank_value, score_value, spec_df = rank_results
		result_l.append((identifier, rank_value, score_value))
		details_l.append(spec_df)
  
	result_df = pd.DataFrame(result_l,columns = ['identifier','rank', 'score'])
	result_df.to_csv(os.path.join(args.save_dp, "summary.csv"),index = False)
 
	for k in range(1,21):
		print(k, len(result_df[result_df["rank"] <= k]), len(result_df[result_df["rank"] <= k])/len(result_df))
  
	details_df = pd.concat(details_l)
	details_df.to_csv(os.path.join(args.save_dp, "details.csv"),index = False)