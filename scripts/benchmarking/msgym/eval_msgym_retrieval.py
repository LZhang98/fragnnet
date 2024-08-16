from frag_gnn.utils.spec_utils import calculate_sparse_cosine_similarity
from frag_gnn.utils.misc_utils import booltype, get_ckpt_from_wandb

import os
import pandas as pd
import json
import time
import argparse
import numpy as np
import torch as th
from tqdm import tqdm


def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_fp",type=str,default="./data/ms2c/msgym/benchmark/retrieval_candidates_formula.csv")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/msgym/proc/retrieval_candidates_formula")
	parser.add_argument("--predicted_dp", type=str,default="./data/ms2c/msgym/fragnnet_predicted/retrieval_candidates_formula")
 
	args = parser.parse_args()
	return args


if __name__ == "__main__":


	args = get_args()
	timestr = time.strftime("%Y%m%d-%H%M%S")
	save_dp = os.path.dirname(args.save_fp)
	os.makedirs(save_dp, exist_ok=True)
	#json.dump(vars(args), open(f"{args.save_dp}/args_{timestr}.json", "w"), indent=2)

	# parser.add_argument("--msg_id_map", type=str,default="data/ms2c/msgym/proc/retrieval_candidates_formula/msg_id_map.csv")
	msg_id_map_fp = os.path.join(args.proc_dp,"msg_id_map.csv")
	msg_id_map_df = pd.read_csv(msg_id_map_fp)
	print(msg_id_map_df)
 
	for idx, row in tqdm(msg_id_map_df.iterrows(),len(msg_id_map_df)):
		#print(row)
		identifier = row['identifier']
		mol_fp = os.path.join(args.proc_dp, row['mol_df'])
		spec_fp = os.path.join(args.predicted_dp, row['spec_df'])

		mol_df = pd.read_pickle(mol_fp)
		spec_fp = pd.read_pickle(spec_fp)

	#target_proc_fp = "data/proc/ms_gymv2"
	#target_spec_df =  pd.read_pickle(os.path.join(target_proc_fp,"spec_df.pkl"))
	#target_mol_df = pd.read_pickle(os.path.join(target_proc_fp,"mol_df.pkl"))
	#test_split_csv = "data/split/ms_gym_d3_h4_benchmarkv2/test_ids.csv"
	#split_csv_df = pd.read_csv(test_split_csv)
	#print(split_csv_df)
	#target_spec_df = target_spec_df[target_spec_df['spec_id'].isin(split_csv_df['spec_id'].to_list())]
	#target_spec_df["peaks"] = target_spec_df.apply(lambda x: "\n".join([f"{m} {i}" for m,i in zip(x["mzs"].split(","),x["intensities"].split(","))]),axis=1)
	#target_spec_df = target_spec_df.drop(columns=["mzs","intensities"])
	#print(target_spec_df.head())
	#print(target_spec_df.columns)
 	#print(spec_df)
	#predicted_mss = pd.read_pickle(frag_gnn_predicted_fp)
	#spec_df = spec_df.merge(fraggnn_ms, on = 'spec_id', how = 'left')
	exit()
	resuts_l = []
	for grp_name, grp in tqdm(spec_df.groupby("target_mol_id")):
		#print(grp.head())
		#try:
			spec_mzs = target_spec_df[target_spec_df["target_mol_id"] == grp_name]["spec_mzs"].values
			spec_ints = target_spec_df[target_spec_df["target_mol_id"] == grp_name]["spec_ints"].values
			target_inchikey_s = target_spec_df[target_spec_df["target_mol_id"] == grp_name]["target_inchikey_s"].values
			#target_inchikey_s = str(target_inchikey_s)
			#print(target_inchikey_s, grp_name)

			if target_inchikey_s not in grp['mol_id'].values.tolist():
				continue

			spec_mzs = th.from_numpy(spec_mzs[0])
			spec_ints = th.from_numpy(spec_ints[0])
			grp['pred_mzs_t'] = grp['pred_mzs'].apply(lambda x: th.from_numpy(x) if isinstance(x, np.ndarray) else th.tensor([x]))
			grp['pred_int_t'] = grp['pred_ints'].apply(lambda x: th.from_numpy(x) if isinstance(x, np.ndarray) else th.tensor([x]))
			
			#print( target_spec_df[target_spec_df["target_mol_id"]  == grp_name])
			#grp['cos'] = grp.apply(lambda x: calculate_sparse_cosine_hungarian(spec_mzs,spec_ints,th.zeros_like(spec_mzs, dtype=th.long), 
			#															   x['pred_mzs_t'], 
			#															  x['pred_int_t'], th.zeros_like( x['pred_mzs_t'], dtype=th.long))[0],axis=1)
			
			grp['cos'] = grp.apply(lambda x: calculate_sparse_cosine_similarity(spec_mzs,spec_ints,th.zeros_like(spec_mzs, dtype=th.long), 
																		x['pred_mzs_t'], 
																		x['pred_int_t'], th.zeros_like( x['pred_mzs_t'], dtype=th.long))[0].item() ,axis=1)
																		#if x['pred_mzs_t'] != None and x['pred_int_t'] != None else 0,axis=1)

			
			grp = grp.sort_values('cos', ascending = False)
			#print(grp[['mol_id','cos','tanimoto']])
			mod_ids = grp['mol_id'].values.tolist()
		
			#print(mod_ids.index(target_inchikey_s) + 1)
		#except:
			#continue
		#else:
			save_grp = grp[['mol_id','cos','tanimoto']]
			#print(save_grp)
			#exit()
			target_inchikey_s = target_inchikey_s.item()
			save_grp.to_csv(f"./data/ms2c_results/frag_gnn_d4_sc/frag_gnn_{str(target_inchikey_s)}.csv", index = False )
			resuts_l.append((target_inchikey_s, mod_ids.index(target_inchikey_s) + 1, save_grp.iloc[mod_ids.index(target_inchikey_s)]['cos']))
		
	df = pd.DataFrame(resuts_l, columns=['inchikey','ranking','cos'])
	df.to_csv(f"./data/ms2c_results/frag_gnn_d4_sc/0_frag_gnn_all.csv", index = False )

	for k in [1,2,3,5,10]:
		print(k, len(df[df["ranking"] <= k]))