import pandas as pd
import os

old_fp = "data/ms2c/pubchem/candidates/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_old.pkl.gz"
old_df = pd.read_pickle(old_fp)
old_df = old_df.drop(['mol', 'targert_smiles'], axis=1)
#print(old_df.head())

new_fp = "data/ms2c/pubchem/candidates/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.pkl.gz"
#fp = os.path.join(v2_dp,"15.pkl.gz")
newdf = pd.read_pickle(new_fp)
newdf = newdf.drop(['targert_smiles'], axis=1)
#print(newdf.head())

target_mol_ids = old_df['target_mol_id'].unique()
#print(target_mol_ids)
for target_mol_id in target_mol_ids:
    #old_grp_df = old_df[old_df['target_mol_id'] == target_mol_id]
    new_grp_df = newdf[newdf['target_mol_id'] == target_mol_id]
    if  len(new_grp_df) - new_grp_df['inchikey_s'].nunique() != 0 or len(new_grp_df) - new_grp_df['smiles'].nunique() != 0:
        print(target_mol_id)
    #print(old_grp_df.head())
    #print(new_grp_df.head())
    #differences = new_grp_df - old_grp_df
    #print(old_grp_df.compare(new_grp_df))
    #old_grp_df = old_grp_df.drop_duplicates(['smiles'])
    #old_grp_df = old_grp_df.merge(new_grp_df, left_on = 'smiles', right_on = 'smiles', how = 'outer')
    #old_grp_df = old_grp_df[old_grp_df['smiles_x'] != old_grp_df['smiles_y']]
    #old_grp_df = old_grp_df[['smiles','inchikey_s_x','tanimoto_x','inchikey_s_y','tanimoto_y']]
    #print(old_grp_df)
    #exit()