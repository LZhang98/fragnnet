
from frag_gnn.runner import load_config
from frag_gnn.dataset import SpecMolDataset
import pandas as pd
from rdkit import Chem
from frag_gnn.utils.misc_utils import booltype
from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from frag_gnn.utils.data_utils import mol_from_smiles, mol_to_smiles
from tqdm import tqdm
import argparse
import os
import numpy as np
import torch as th
import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.data_utils import par_apply_series, par_apply_df_rows, seq_apply_series, seq_apply_df_rows
from frag_gnn.utils.misc_utils import list_str2float, booltype
from frag_gnn.utils.plot_utils import plot_spectra_sparse

def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--split_type", type=str,default="test", help="")
	parser.add_argument("--candidate_pkl_fp",type=str,default='./data/ms2c/pubchem/candidates/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.pkl.gz')
	parser.add_argument("--ms2c_data_dp",type=str, default='./data/ms2c/pubchem/proc]')
	
	parser.add_argument("--include_sanity",type=booltype,default=True)
	parser.add_argument("--output_per_grp",type=booltype,default=False)
	parser.add_argument("--template_fp",type=str,default="config/template.yml")
	parser.add_argument("--custom_fp",type=str,default="config/nist20v3_d4_wr00_inchikey/fraggnn_d4/s0.yml")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	args = get_args()
	print("> Loading configs")
	print(f"> template fp: {args.template_fp}")
	print(f"> custom fp: {args.custom_fp}")
	config_d = load_config(args.template_fp, args.custom_fp)
	prec_types_d = config_d['spec_params']['prec_types']

	print(f"> Loading data for {args.split_type} split")
	ds = SpecMolDataset(split=args.split_type, **config_d)

	print("> Getting Spectra setup")
	candidate_spec_df_rows = []
	target_spec_df_rows = []
	
	target_spec_df_cols = ['target_mol_id','prec_type','nce', 'ace', 'prec_mz','spec_mzs','spec_ints', 'target_mol_smiles','target_inchikey_s']
	candidate_spec_df_cols = ['target_mol_id','prec_type','nce', 'ace', 'prec_mz','target_mol_smiles','target_inchikey_s']
	spec_df_export_cols = ['spec_id', 'group_id', 'mol_id','prec_type','nce', 'ace', 'prec_mz', 
						'target_mol_id','tanimoto', 'spec_mzs','spec_ints','dset','dset_spec_id','peaks', 'inst_type']

	for d in tqdm(ds):
		inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(d['mol_smiles'][0])).split('-')[0]
		target_spec_df_rows.append(
						[d['mol_id'],
	  					prec_types_d[d['spec_prec_type'].numpy()[0]],
						d['spec_ce'].tolist(),  #merged nce
						np.nan,	#d['spec_ce'].tolist(), 
						d['spec_prec_mz'].tolist()[0],
						d['spec_mzs'].numpy(),
						d['spec_ints'].numpy(),
						d['mol_smiles'][0],
						inchikey])

		for spec_ce in d['spec_ce'].tolist():
			candidate_spec_df_rows.append([d['mol_id'],
						prec_types_d[d['spec_prec_type'].numpy()[0]],
						spec_ce, #d['spec_ce'].tolist(),  
						np.nan,	#d['spec_ce'].tolist(), 
						d['spec_prec_mz'].tolist()[0],
						d['mol_smiles'][0],
						inchikey])
   
	target_spec_df = pd.DataFrame(target_spec_df_rows,columns=target_spec_df_cols)

	candidate_basename = os.path.basename(args.candidate_pkl_fp).split('.')[0]
	save_dp = os.path.join(args.ms2c_data_dp, candidate_basename)
	os.makedirs(save_dp, exist_ok=True)
	
	target_spec_fp = os.path.join(save_dp, f"target_spec_df.pkl.gz")
	#spec_df = spec_df[spec_df_export_cols]
	print(f"> Save spec df {target_spec_fp}")
	target_spec_df.to_pickle(target_spec_fp)

	print("> Process mol and spec df")	
	print("> Read candidate df")
	candidates_df =  pd.read_pickle(args.candidate_pkl_fp)	 
	candidates_df = candidates_df.dropna()
 
	spec_df = pd.DataFrame(candidate_spec_df_rows,columns=candidate_spec_df_cols)
	spec_df = candidates_df.merge(target_spec_df, on = 'target_mol_id', how = 'inner')
	spec_df['prec_mz'] = spec_df.apply(lambda x: Chem.Descriptors.ExactMolWt(x['mol']) + PREC_TYPE_TO_MASS_DIFF['[M+H]+'], axis=1)
	spec_df.reset_index(drop=True, inplace=True)
	spec_df['spec_id'] = spec_df.index
	spec_df['dset'] = "pubchem"
	spec_df['dset_spec_id'] = spec_df['spec_id']
	# note fragnnet in this paper only support one adduct type M+H
	spec_df['group_id'] = spec_df.groupby(['target_mol_id', 'mol_id']).ngroup()
	# avoid dataset crash
	spec_df['peaks'] = np.empty((len(spec_df),0)).tolist()
	#spec_df['frag_mode'] = frag_mode
	spec_df['inst_type'] = "FT"
 
	# create sanity check version
	if args.include_sanity:
		sanity_spec_df = spec_df.loc[spec_df['mol_id'] == spec_df['target_inchikey_s']]
		sanity_spec_df = sanity_spec_df[spec_df_export_cols]

	print(f"> Save spec df and split")
	if not args.output_per_grp:
		candidate_spec_fp = os.path.join(save_dp, f"spec_df.pkl.gz")
		spec_df_export = spec_df[spec_df_export_cols]
		spec_df_export.to_pickle(candidate_spec_fp)

	else:
		for grp_name, grp_spec_df in spec_df.groupby("target_mol_id"):
			grp_spec_df_export = grp_spec_df[spec_df_export_cols]
			grp_name_str = str(grp_name)
			os.makedirs(os.path.join(save_dp, grp_name_str), exist_ok=True)
			grp_candidate_spec_fp = os.path.join(save_dp, grp_name_str,f"spec_df.pkl.gz")
			grp_spec_df_export.to_pickle(grp_candidate_spec_fp)

	print(f"> Save mol df")
	if not args.output_per_grp:
		# create mol df
		candidates_df.drop_duplicates(subset='mol_id', keep="last", inplace=True)
		candidates_df.reset_index(drop=True, inplace=True)
		candidate_mol_fp = os.path.join(save_dp, f"mol_df.pkl.gz")
		print(candidate_mol_fp)
		candidates_df.to_pickle(candidate_mol_fp)
	else:
		for grp_name, grp_mol_df in candidates_df.groupby("target_mol_id"):
			grp_name_str = str(grp_name)
			os.makedirs(os.path.join(save_dp, grp_name_str), exist_ok=True)
			grp_candidate_mol_fp = os.path.join(save_dp, grp_name_str,f"mol_df.pkl.gz")
			grp_mol_df.to_pickle(grp_candidate_mol_fp)

	if args.include_sanity:
		sanity_dp = os.path.join(args.ms2c_data_dp, f"{candidate_basename}_sanity")
		os.makedirs(sanity_dp, exist_ok=True)
		sanity_spec_fp = os.path.join(sanity_dp, f"spec_df.pkl.gz")
		print(f"> Sanity spec {sanity_spec_fp}")
		sanity_spec_df.to_pickle(sanity_spec_fp)
		
		#sanity_split_fp = os.path.join(sanity_dp, f"{args.split_type}_ids.csv")
		#print(f"> Sanity split {sanity_split_fp}")
		#sanity_split_df.to_csv(sanity_split_fp, index=False)

		sanity_mol_df = candidates_df[candidates_df['mol_id'].isin(sanity_spec_df['mol_id'])]
		sanity_mol_fp = os.path.join(sanity_dp, f"mol_df.pkl.gz")
		print(f"> Sanity mol {sanity_mol_fp}")
		sanity_mol_df.to_pickle(sanity_mol_fp)
		