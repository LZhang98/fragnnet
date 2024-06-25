import pandas as pd
import re
from math import prod

from frag_gnn.utils.frag_utils import ELEMENTS
from frag_gnn.utils.proc_utils import filter_spec_mol, merge_spec_df

# read in spec_df and mol_df
spec_df = pd.read_pickle("data/proc/nist20mona23/spec_df.pkl")
mol_df = pd.read_pickle("data/proc/nist20mona23/mol_df.pkl")

# perform selection
spec_df, mol_df = filter_spec_mol(
    spec_df,
    mol_df,
    allowed_elements=ELEMENTS,
    dsets=["nist20_hr","mona23"],
    prec_types=["[M+H]+"],
    num_entries=-1)

nist_spec_df = spec_df[spec_df["dset"] == "nist20_hr"]
mona_spec_df = spec_df[spec_df["dset"] == "mona23"]

# ###

# def count_num_subformuale(formula_str):
#     # Regular expression to match elements and their counts
#     element_regex = r'([A-Z][a-z]*)(\d*)'
#     # Dictionary to store element counts
#     element_counts = {}
#     # Find all matches in the formula
#     matches = re.findall(element_regex, formula_str)
#     for element, count in matches:
#         # Convert count to integer, defaulting to 1 if empty
#         count = int(count) if count else 1
#         # Add count to the element's total in the dictionary
#         if element in element_counts:
#             element_counts[element] += count
#         else:
#             element_counts[element] = count
#     # calculate product
#     num_subformulae = prod((v+1) for v in element_counts.values())
#     return num_subformulae

# nist_mol_df = mol_df[mol_df["mol_id"].isin(nist_spec_df["mol_id"])]
# mona_mol_df = mol_df[mol_df["mol_id"].isin(mona_spec_df["mol_id"])]

# import pdb; pdb.set_trace()

# nist_mol_df.loc[:,"num_subformulae"] = nist_mol_df["formula"].apply(count_num_subformuale)
# mona_mol_df.loc[:,"num_subformulae"] = mona_mol_df["formula"].apply(count_num_subformuale)

# import pdb; pdb.set_trace()

# ###

# read in frag stats file
m_spec_stats_df = pd.read_pickle("data/frag/nist20mona23_d4_h4_isoFalse_nb2/m_spec_stats_df.pkl")

# find cases where the recall is 0
zero_wrecall = m_spec_stats_df[m_spec_stats_df["wrecall_10ppm_h4"] == 0.]
zero_wrecall_spec = spec_df[spec_df["group_id"].isin(zero_wrecall["group_id"])]
print(zero_wrecall_spec)

# find cases where nist and mona are the same
# check: mol_id, inst_type, frag_mode, prec_type, nce

both_cols = ["mol_id","inst_type","frag_mode","prec_type","nce"]
both_df = nist_spec_df[both_cols+["spec_id"]].merge(
    mona_spec_df[both_cols+["spec_id"]],
    on=both_cols,
    how="inner",
    suffixes=["_nist","_mona"])
nist_both_spec_df = nist_spec_df[nist_spec_df["spec_id"].isin(both_df["spec_id_nist"])]
mona_both_spec_df = mona_spec_df[mona_spec_df["spec_id"].isin(both_df["spec_id_mona"])]
print(nist_both_spec_df.shape)
print(mona_both_spec_df.shape)

nist_both_m_spec_df = merge_spec_df(nist_both_spec_df)
mona_both_m_spec_df = merge_spec_df(mona_both_spec_df)

print(nist_both_m_spec_df.shape)
print(mona_both_m_spec_df.shape)

# deal with dups

nist_dups = both_df[["spec_id_nist","spec_id_mona"]].groupby("spec_id_mona").count()
mona_dups = both_df[["spec_id_nist","spec_id_mona"]].groupby("spec_id_nist").count()

print(nist_dups["spec_id_nist"].value_counts())
print(mona_dups["spec_id_mona"].value_counts())

nist_mona_nodups = both_df[both_df["spec_id_mona"].isin(nist_dups[nist_dups["spec_id_nist"] == 1].index) & both_df["spec_id_nist"].isin(mona_dups[mona_dups["spec_id_mona"] == 1].index)]
assert nist_mona_nodups.shape[0] == nist_mona_nodups["spec_id_nist"].nunique() == nist_mona_nodups["spec_id_mona"].nunique()

nist_nodups_spec_df = nist_spec_df[nist_spec_df["spec_id"].isin(nist_mona_nodups["spec_id_nist"])]
mona_nodups_spec_df = mona_spec_df[mona_spec_df["spec_id"].isin(nist_mona_nodups["spec_id_mona"])]

nist_nodups_m_spec_df = merge_spec_df(nist_nodups_spec_df)
mona_nodups_m_spec_df = merge_spec_df(mona_nodups_spec_df)

import pdb; pdb.set_trace()

# remove Q Exactive

mona_raw_df = pd.read_json("data/df/mona23_df.json")
mona_both_raw_df = mona_raw_df[mona_raw_df["dset_spec_id"].isin(mona_both_spec_df["dset_spec_id"])]
# mona_nodups_raw_df = mona_raw_df[mona_raw_df["dset_spec_id"].isin(mona_nodups_spec_df["dset_spec_id"])]
mona_noq_raw_df = mona_both_raw_df[~(mona_both_raw_df["inst"].str.contains("Q Exactive")|mona_both_raw_df["inst"].str.contains("Q-Exactive"))]

mona_noq_spec_df = mona_both_spec_df[mona_both_spec_df["dset_spec_id"].isin(mona_noq_raw_df["dset_spec_id"])]
both_noq_df = both_df[both_df["spec_id_mona"].isin(mona_noq_spec_df["spec_id"])]

nist_noq_spec_df = nist_both_spec_df[nist_both_spec_df["spec_id"].isin(both_noq_df["spec_id_nist"])]
mona_noq_spec_df = mona_both_spec_df[mona_both_spec_df["spec_id"].isin(both_noq_df["spec_id_mona"])]

nist_noq_m_spec_df = merge_spec_df(nist_noq_spec_df)
mona_noq_m_spec_df = merge_spec_df(mona_noq_spec_df)

import pdb; pdb.set_trace()

# setup splits (all)

nist_split_df = nist_both_spec_df[nist_both_spec_df["group_id"].isin(nist_both_m_spec_df["group_id"])][["spec_id","mol_id","group_id"]]
mona_split_df = mona_both_spec_df[mona_both_spec_df["group_id"].isin(mona_both_m_spec_df["group_id"])][["spec_id","mol_id","group_id"]]

nist_split_fp = "data/split/debug_nist_mona/nist_ids.csv"
mona_split_fp = "data/split/debug_nist_mona/mona_ids.csv"

nist_split_df.to_csv(nist_split_fp,index=False)
mona_split_df.to_csv(mona_split_fp,index=False)

# setup split (nodups)

nist_nodups_split_df = nist_nodups_spec_df[nist_nodups_spec_df["group_id"].isin(nist_nodups_m_spec_df["group_id"])][["spec_id","mol_id","group_id"]]
mona_nodups_split_df = mona_nodups_spec_df[mona_nodups_spec_df["group_id"].isin(mona_nodups_m_spec_df["group_id"])][["spec_id","mol_id","group_id"]]

nist_nodups_split_fp = "data/split/debug_nist_mona/nist_nodups_ids.csv"
mona_nodups_split_fp = "data/split/debug_nist_mona/mona_nodups_ids.csv"

nist_nodups_split_df.to_csv(nist_nodups_split_fp,index=False)
mona_nodups_split_df.to_csv(mona_nodups_split_fp,index=False)

# setup split (no Q Exactive)

nist_noq_split_df = nist_noq_spec_df[nist_noq_spec_df["group_id"].isin(nist_noq_m_spec_df["group_id"])][["spec_id","mol_id","group_id"]]
mona_noq_split_df = mona_noq_spec_df[mona_noq_spec_df["group_id"].isin(mona_noq_m_spec_df["group_id"])][["spec_id","mol_id","group_id"]]

nist_noq_split_fp = "data/split/debug_nist_mona/nist_noq_ids.csv"
mona_noq_split_fp = "data/split/debug_nist_mona/mona_noq_ids.csv"

nist_noq_split_df.to_csv(nist_noq_split_fp,index=False)
mona_noq_split_df.to_csv(mona_noq_split_fp,index=False)
