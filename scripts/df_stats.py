import pandas as pd

from frag_gnn.utils.data_utils import par_apply_series, parse_prec_type_str

spec_df = pd.read_pickle("data/proc_all/spec_df.pkl")
mol_df = pd.read_pickle("data/proc_all/mol_df.pkl")

dsets = ["nist20_hr","nist20_lr","mona23"]

# some preliminary selection

spec_df = spec_df[spec_df["spec_type"]=="MS2"]
spec_df = spec_df.dropna(subset=["prec_mz","prec_type"],axis=0,how="all")

# count number of spectra per dataset

print(">>> number of spectra per dataset")
print(spec_df["dset"].value_counts()[:10])

# check number of molecules per dataset

print(">>> number of molecules per dataset")
dset_mol_ids = {}
all_mol_ids = set()
for dset in dsets:
    dset_mol_ids[dset] = set(spec_df[spec_df["dset"]==dset]["mol_id"])
    print(dset,len(dset_mol_ids[dset]))
    all_mol_ids.update(dset_mol_ids[dset])

print(">>> number of common molecules between datasets")
for dset1 in dsets:
    for dset2 in dsets:
        if dset1 != dset2:
            print(dset1,dset2,len(dset_mol_ids[dset1].intersection(dset_mol_ids[dset2])))

print(">>> number of unique molecules per dataset")
for dset1 in dsets:
    mol_ids = dset_mol_ids[dset1].copy()
    for dset2 in dsets:
        if dset1 != dset2:
            mol_ids = mol_ids.difference(dset_mol_ids[dset2])
    print(dset1,len(mol_ids))
print("all",len(all_mol_ids))

# count the number of instrument types per dataset

print(">>> number of instrument types per dataset")
for dset in dsets:
    dset_df = spec_df[spec_df["dset"]==dset]
    print(dset)
    print(dset_df["inst_type"].value_counts()[:10])

# count the number of precursor adducts per dataset

print(">>> number of precursor adducts per dataset")
for dset in dsets:
    dset_df = spec_df[spec_df["dset"]==dset]
    print(dset)
    print(dset_df["prec_type"].value_counts()[:10])

import pdb; pdb.set_trace()
