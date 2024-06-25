
from frag_gnn.runner import load_config
from frag_gnn.dataset import SpecMolDataset
import pandas as pd
from rdkit import Chem
from frag_gnn.utils.misc_utils import booltype
from frag_gnn.utils.spec_utils import prec_type_mass_diff, PREC_TYPES
from tqdm import tqdm
import torch as th
from frag_gnn.utils.plot_utils import plot_spectra_sparse
from frag_gnn.utils.spec_utils import calculate_sparse_cosine_hungarian
from frag_gnn.utils.spec_utils import calculate_sparse_cosine_similarity
from frag_gnn.utils.spec_utils import round_aggregate_peaks

if __name__ == "__main__":

	#frag_gnn_predicted_fp = "data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/kme8fry7_fraggnn_d4_nodemlp_mnce_nocp.pkl.gz"
	icebgern_predicted_fp ="data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/f8p1azsm_iceberg_inten_100.pkl.gz"
	spec_df_fp = "data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/spec_df.pkl.gz"
	target_spec_fp = "data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/target_spec_df.pkl.gz" 

	spec_df = pd.read_pickle(spec_df_fp)
	target_spec_df =  pd.read_pickle(target_spec_fp)

	#print(spec_df)
	icebger_ms = pd.read_pickle(icebgern_predicted_fp)
	spec_df = spec_df.merge(icebger_ms, on = 'spec_id', how = 'left')

	resuts_l = []
	for grp_name, grp in tqdm(spec_df.groupby("target_mol_id")):
		#print(grp.head())
		spec_mzs = target_spec_df[target_spec_df["target_mol_id"] == grp_name]["spec_mzs"].values
		spec_ints = target_spec_df[target_spec_df["target_mol_id"] == grp_name]["spec_ints"].values
		target_inchikey_s = target_spec_df[target_spec_df["target_mol_id"] == grp_name]["target_inchikey_s"].values
		#target_inchikey_s = str(target_inchikey_s)
		#print(target_inchikey_s, grp_name)
		grp = grp.dropna()
		if target_inchikey_s not in grp['mol_id'].values.tolist():
			continue
		
		spec_mzs = th.from_numpy(spec_mzs[0])
		spec_ints = th.from_numpy(spec_ints[0])
		grp['pred_mzs_t'] = grp['pred_mzs'].apply(lambda x: th.from_numpy(x))
		grp['pred_int_t'] = grp['pred_ints'].apply(lambda x: th.from_numpy(x))
		
		#print( target_spec_df[target_spec_df["target_mol_id"]  == grp_name])
		grp['cos'] = grp.apply(lambda x: calculate_sparse_cosine_hungarian(spec_mzs,spec_ints,th.zeros_like(spec_mzs, dtype=th.long), 
																	   x['pred_mzs_t'], 
																	  x['pred_int_t'], th.zeros_like( x['pred_mzs_t'], dtype=th.long))[0].item(),axis=1)
		grp = grp.sort_values('cos', ascending = False)
		#print(grp[['mol_id','cos','tanimoto']])
		mod_ids = grp['mol_id'].values.tolist()
		#print(mod_ids.index(target_inchikey_s) + 1)

		save_grp = grp[['mol_id','cos','tanimoto']]
		target_inchikey_s = target_inchikey_s.item()
		save_grp.to_csv(f"./data/ms2c_results/iceberg/iceberg_{target_inchikey_s}.csv", index = False )
		resuts_l.append((target_inchikey_s, mod_ids.index(target_inchikey_s) + 1, save_grp.iloc[mod_ids.index(target_inchikey_s)]['cos']))
		
	df = pd.DataFrame(resuts_l, columns=['inchikey','ranking','cos'])
	df.to_csv(f"./data/ms2c_results/iceberg/0_iceberg_all.csv", index = False )

	#print(len())
	for k in [1,2,3,5,10]:
		print(k, len(df[df["ranking"] <= k]))
