
from frag_gnn.runner import load_config
from frag_gnn.dataset import SpecMolDataset
import pandas as pd
from rdkit import Chem
from frag_gnn.utils.misc_utils import booltype
from frag_gnn.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from tqdm import tqdm
import argparse
import os
import numpy as np
from frag_gnn.utils.misc_utils import booltype

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--proc_dp', type=str, default='./data/proc/nist20mona23v3')
	parser.add_argument('--split_dp', type=str, default='./data/split/nist20mona23v3_d4_wr00_scaffold')
	parser.add_argument('--split_type', type=str,default='test', help='')
	parser.add_argument('--candidate_pkl_fp',type=str,default='./data/ms2c/pubchem/candidates/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.pkl.gz')
	parser.add_argument('--ms2c_data_dp',type=str, default='./data/ms2c/pubchem/proc')
	
	parser.add_argument('--include_sanity',type=booltype,default=True)
	parser.add_argument('--output_per_grp',type=booltype,default=False)
	#parser.add_argument('--template_fp',type=str,default='config/template.yml')
	#parser.add_argument('--custom_fp',type=str,default='config/nist20v3_d4_wr00_inchikey/fraggnn_d4/s0.yml')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	
	args = get_args()
	print('> Loading configs')
	#config_d = load_config(args.template_fp, args.custom_fp)
	#prec_types_d = config_d['spec_params']['prec_types']

	#print(f'> Loading data for {args.split_type} split')
	#ds = SpecMolDataset(split=args.split_type, **config_d)
 
	split_configs_d = {
		'test':'test_ids.csv',
		'val':'val_ids.csv',
		'train':'train_ids.csv'
	}

	mol_df_fp = os.path.join(args.proc_dp,'mol_df.pkl')
	spec_df_fp = os.path.join(args.proc_dp,'spec_df.pkl')
	print(f'> Read mol df from {mol_df_fp}')
	mol_df = pd.read_pickle(mol_df_fp)
	print(f'> Read spec df from {spec_df_fp}')
	spec_df = pd.read_pickle(spec_df_fp)
	
	split_type = args.split_type
	split_csv_filename = split_configs_d[split_type]
	split__fp = os.path.join(args.split_dp, split_csv_filename)
	print(f'> Processing {split_type} split, read split from {split__fp}')
	split_df = pd.read_csv(split__fp)
 	# select spectra
	spec_df = spec_df[spec_df['spec_id'].isin(split_df['spec_id'])]
	# select molecules
	mol_df = mol_df[mol_df['mol_id'].isin(split_df['mol_id'])]
	target_spec_df = spec_df.merge(mol_df[['mol_id','smiles','inchikey_s']],on='mol_id',how='inner')
	

	print('> Getting Spectra setup')
	target_spec_df = target_spec_df.rename(columns={'mol_id': 'target_mol_id', 'smiles': 'target_mol_smiles', 'inchikey_s':'target_inchikey_s'})
	target_spec_df_cols = ['target_mol_id', 'target_mol_smiles','target_inchikey_s','prec_type','nce', 'ace', 'peaks','prec_mz', 'inst_type', 'frag_mode']
	
	spec_df_export_cols = ['spec_id', 'group_id', 'mol_id','prec_type','nce', 'ace', 'prec_mz', 
						'target_mol_id','target_mol_smiles','target_inchikey_s', 'tanimoto','dset','dset_spec_id','peaks', 'inst_type']

	candidate_basename = os.path.basename(args.candidate_pkl_fp).split('.')[0]
	save_dp = os.path.join(args.ms2c_data_dp, candidate_basename)
	os.makedirs(save_dp, exist_ok=True)
	
	target_spec_fp = os.path.join(save_dp, f'target_spec_df.pkl.gz')
	print(f'> Save target spec df df {target_spec_fp}')
	target_spec_df_export = target_spec_df[target_spec_df_cols]
	target_spec_df_export.to_pickle(target_spec_fp)

	print('> Process mol and spec df')	
	print('> Read candidate df')
	candidates_df =  pd.read_pickle(args.candidate_pkl_fp)	 
	candidates_df = candidates_df.dropna()
	candidate_spec_df = target_spec_df_export
	candidate_spec_df = candidate_spec_df.merge(candidates_df, on = 'target_mol_id', how = 'inner')
	candidate_spec_df.reset_index(drop=True, inplace=True)
	candidate_spec_df['spec_id'] = candidate_spec_df.index
	candidate_spec_df['dset'] = 'pubchem'
	candidate_spec_df['dset_spec_id'] = candidate_spec_df['spec_id']

	# note fragnnet in this paper only support one adduct type M+H
	candidate_spec_df['group_id'] = candidate_spec_df.groupby(['target_mol_id', 'mol_id']).ngroup()
	# avoid dataset crash
	# candidate_spec_df['peaks'] = np.empty((len(candidate_spec_df),0)).tolist()
	# spec_df['frag_mode'] = frag_mode
	# candidate_spec_df['inst_type'] = 'FT'
 
	# create sanity check version
	if args.include_sanity:
		sanity_spec_df = candidate_spec_df.loc[candidate_spec_df['mol_id'] == candidate_spec_df['target_inchikey_s']]
		sanity_spec_df = sanity_spec_df[spec_df_export_cols]

	print(f'> Save spec df and split')
	if not args.output_per_grp:
		candidate_spec_fp = os.path.join(save_dp, f'spec_df.pkl.gz')
		spec_df_export = candidate_spec_df[spec_df_export_cols]
		spec_df_export.to_pickle(candidate_spec_fp)
	else:
		for grp_name, grp_spec_df in candidate_spec_df.groupby('target_mol_id'):
			grp_spec_df_export = grp_spec_df[spec_df_export_cols]
			grp_name_str = str(grp_name)
			os.makedirs(os.path.join(save_dp, grp_name_str), exist_ok=True)
			grp_candidate_spec_fp = os.path.join(save_dp, grp_name_str,f'spec_df.pkl.gz')
			grp_spec_df_export.to_pickle(grp_candidate_spec_fp)

	print(f'> Save mol df')
	if not args.output_per_grp:
		# create mol df
		candidates_df.drop_duplicates(subset='mol_id', keep='last', inplace=True)
		candidates_df.reset_index(drop=True, inplace=True)
		candidate_mol_fp = os.path.join(save_dp, f'mol_df.pkl.gz')
		print(candidate_mol_fp)
		candidates_df.to_pickle(candidate_mol_fp)
	else:
		for grp_name, grp_mol_df in candidates_df.groupby('target_mol_id'):
			grp_name_str = str(grp_name)
			os.makedirs(os.path.join(save_dp, grp_name_str), exist_ok=True)
			grp_candidate_mol_fp = os.path.join(save_dp, grp_name_str,f'mol_df.pkl.gz')
			grp_mol_df.to_pickle(grp_candidate_mol_fp)

	if args.include_sanity:
		sanity_dp = os.path.join(args.ms2c_data_dp, f'{candidate_basename}_sanity')
		os.makedirs(sanity_dp, exist_ok=True)
		sanity_spec_fp = os.path.join(sanity_dp, f'spec_df.pkl.gz')
		print(f'> Sanity spec {sanity_spec_fp}')
		sanity_spec_df.to_pickle(sanity_spec_fp)
		
		#sanity_split_fp = os.path.join(sanity_dp, f'{args.split_type}_ids.csv')
		#print(f'> Sanity split {sanity_split_fp}')
		#sanity_split_df.to_csv(sanity_split_fp, index=False)

		sanity_mol_df = candidates_df[candidates_df['mol_id'].isin(sanity_spec_df['mol_id'])]
		sanity_mol_fp = os.path.join(sanity_dp, f'mol_df.pkl.gz')
		print(f'> Sanity mol {sanity_mol_fp}')
		sanity_mol_df.to_pickle(sanity_mol_fp)
		