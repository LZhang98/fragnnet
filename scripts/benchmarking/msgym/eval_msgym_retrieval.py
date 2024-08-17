
import os
import pandas as pd
import json
import time
import argparse
import numpy as np
import torch as th
from tqdm import tqdm

from frag_gnn.utils.spec_utils import calculate_sparse_cosine_similarity
from frag_gnn.utils.data_utils import par_apply

LAST_RANK = 257
LAST_SCORE = -1

def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_fp",type=str,default="./data/ms2c/msgym/benchmark/retrieval_candidates_formula.csv")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/msgym/proc/retrieval_candidates_formula")
	parser.add_argument("--predicted_dp", type=str,default="./data/ms2c/msgym/fragnnet_predicted/retrieval_candidates_formula")
 
	args = parser.parse_args()
	return args


if __name__ == "__main__":


	args = get_args()
	save_dp = os.path.dirname(args.save_fp)
	os.makedirs(save_dp, exist_ok=True)

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
			return (identifier,LAST_RANK, LAST_SCORE)
		elif not has_spec_fp:
			spec_fp  = spec_fp+'.gz'
   
		if not os.path.exists(mol_fp):
			return (identifier,LAST_RANK, LAST_SCORE)

		mol_df = pd.read_pickle(mol_fp)
		spec_df = pd.read_pickle(spec_fp)

		# nothing predicted last rank
		if len(spec_df) == 0:
			return (identifier,LAST_RANK, LAST_SCORE)
			
		spec_df = spec_df.merge(mol_df, on = 'mol_id', how = 'left')
  
		true_mzs = [float(x) for x in row['mz'].split(",")]
		true_ints = [float(x)*1000 for x in row['int'].split(",")]

		true_mzs = th.tensor(true_mzs)
		true_ints = th.tensor(true_ints)

		spec_df['cos_sim_0.01'] = spec_df.apply(lambda x: calculate_sparse_cosine_similarity(true_mzs,true_ints,
                                                                              th.zeros(len(true_mzs), dtype=th.long), 
																				th.tensor(x['pred_mzs']), 
																				th.tensor(x['pred_ints']), 
																				th.zeros(len(x['pred_mzs']), dtype=th.long))[0].item() ,axis=1)
		spec_df['rank'] = spec_df['cos_sim_0.01'].rank(ascending = False)
		
		ranks = spec_df[spec_df['smiles'] == true_smiles]['rank'].values
		scores = spec_df[spec_df['smiles'] == true_smiles]['cos_sim_0.01'].values
		rank_value = ranks[0] if len(ranks) else LAST_RANK
		score_value = scores[0] if len(ranks) else LAST_SCORE
  
		return identifier, rank_value, score_value
  
	input_l = []
	result_l = []
 
	for idx, row in tqdm(msg_id_map_df.iterrows(),total=len(msg_id_map_df)):


		input_l.append(row)

	frag_results = par_apply(iter(input_l), get_rank, False, return_as_generator=True)
	for rank_results in tqdm(frag_results, total = len(input_l), desc = "ranking", leave=None):
		result_l.append(rank_results)
  
	result_df = pd.DataFrame(result_l,columns = ['identifier','rank', 'score'])
	result_df.to_csv(args.save_fp,index = False)
 
	for k in range(1,20):
		print(k, len(result_df[result_df["rank"] <= k]))