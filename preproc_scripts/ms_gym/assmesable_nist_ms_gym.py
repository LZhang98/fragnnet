import pandas as pd
import argparse
import os
import pandas as pd
import numpy as np
import json

from frag_gnn.utils import frag_utils
from frag_gnn.utils.misc_utils import booltype, np_temp_seed
from frag_gnn.utils.frag_utils import ELEMENTS
from frag_gnn.utils.proc_utils import filter_spec_mol
import shutil
from tqdm import tqdm
import os

ms_gym_mol_df = pd.read_pickle("data/proc/ms_gym/mol_df.pkl")
ms_gym_spec_df = pd.read_pickle("data/proc/ms_gym/spec_df.pkl")
ms_gym_test_df = pd.read_csv("data/df/ms_gym_fold.csv")

test_fold = ms_gym_test_df[ms_gym_test_df["fold"] == "test"]["dset_spec_id"].to_list()
val_fold = ms_gym_test_df[ms_gym_test_df["fold"] == "val"]["dset_spec_id"].to_list()

ms_gym_spec_val_df = ms_gym_spec_df[ms_gym_spec_df['dset_spec_id'].isin(val_fold)].copy()
ms_gym_spec_test_df = ms_gym_spec_df[ms_gym_spec_df['dset_spec_id'].isin(test_fold)].copy()

ms_gym_spec_val_df['dset'] = "ms_gym_val"
ms_gym_spec_test_df['dset'] = "ms_gym_test"

ms_gym_mol_val_df = ms_gym_mol_df[ms_gym_mol_df['mol_id'].isin(ms_gym_spec_val_df['mol_id'])].copy()
ms_gym_mol_test_df  = ms_gym_mol_df[ms_gym_mol_df['mol_id'].isin(ms_gym_spec_test_df['mol_id'])].copy()

ms_gym_mol_val_df['dset'] = "ms_gym_val"
ms_gym_mol_test_df['dset'] = "ms_gym_test"

print(">ms_gym_spec_val_df",ms_gym_spec_val_df.shape)
print(">ms_gym_spec_test_df",ms_gym_spec_test_df.shape)

print(">ms_gym_mol_val_df",ms_gym_mol_val_df.shape)
print(">ms_gym_mol_test_df",ms_gym_mol_test_df.shape)

#print(ms_gym_mol_df.columns)
#print(ms_gym_spec_df.columns)
#print(ms_gym_test_df.head())


# nist df
#mol_df = "data/proc/nist20mona23v2/mol_df.pkl"
print(">read nist_mol_df")
nist_mol_df = pd.read_pickle("data/proc/nist20mona23v2/mol_df.pkl")
nist_mol_df['dset'] = 'nist20mona23'

#duplicated_df = nist_mol_df[nist_mol_df.duplicated(subset=["inchikey_s"], keep=False)]
#grps = duplicated_df.groupby("inchikey_s")
#for grp_df in grps:
#    print(grp_df)
#print(nist_mol_df[nist_mol_df.duplicated(subset="inchikey_s")]["inchikey_s"])

print(">nist_mol_df",nist_mol_df.shape)
merged_mol_df = pd.concat([nist_mol_df,ms_gym_mol_val_df,ms_gym_mol_test_df])
print(">mol_df size ",merged_mol_df.shape)

merged_mol_df.rename(columns={"mol_id":"org_mol_id"}, inplace=True)
merged_mol_df.drop_duplicates(subset=["smiles"], inplace=True)
merged_mol_df.reset_index(inplace=True, drop=True)
merged_mol_df["mol_id"] = merged_mol_df.index
print(">mol_df size after duplicate ",merged_mol_df.shape)

print(">> update mol_id")
mol_ids = merged_mol_df[merged_mol_df["dset"].isin(["ms_gym_val","ms_gym_test"])]['mol_id'].to_list()
org_mol_ids = merged_mol_df[merged_mol_df["dset"].isin(["ms_gym_val","ms_gym_test"])]['org_mol_id'].to_list()
update_mol_id_dict = {org_mol_ids[i]: mol_ids[i] for i in range(len(org_mol_ids))}

print("> update_mol_id_dict size",len(update_mol_id_dict))
ms_gym_spec_val_df["mol_id"]  = ms_gym_spec_val_df["mol_id"].replace(update_mol_id_dict)
ms_gym_spec_test_df["mol_id"] = ms_gym_spec_test_df["mol_id"].replace(update_mol_id_dict)

#assert len(mol_df[(mol_df["mol_id"] != mol_df["org_mol_id"]) & (mol_df["dset"] == "nist20mona23")]) == 0

print(">> read nist spec df")
nist_spec_df = pd.read_pickle("data/proc/nist20mona23v2/spec_df.pkl")
print(">> create merged spec df")
merged_spec_df = pd.concat([nist_spec_df,ms_gym_spec_val_df,ms_gym_spec_test_df])

print(f"> merged_mol_df {merged_mol_df.shape}")
print(f"> merged_spec_df {merged_spec_df.shape}")
print("merged_mol_df",merged_mol_df['dset'].unique())
print("merged_spec_df",merged_spec_df['dset'].unique())

merged_mol_df.to_pickle("data/proc/nist20mona23ms_gym_valtest/mol_df.pkl")
merged_spec_df.to_pickle("data/proc/nist20mona23ms_gym_valtest/spec_df.pkl")
 
ms_gym_frag_dp = "./data/frag/ms_gym_d3_h4"
new_frag_dp = "./data/frag/nist20mona23ms_gym_valtest_d3_h4"

for org_mol_id in tqdm(update_mol_id_dict):

    mol_id = update_mol_id_dict[org_mol_id]
    src = f"{ms_gym_frag_dp}/dags/{org_mol_id}.pickle.bz2"
    if os.path.isfile(src):
        dst = f"{ms_gym_frag_dp}/dags/{mol_id}.pickle.bz2"
        shutil.copyfile(src, dst)