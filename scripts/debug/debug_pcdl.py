import pandas as pd
import numpy as np
mol_df = pd.read_pickle("data/proc/pcdl_nl/mol_df.pkl")
#mol_df = pd.read_pickle("data/proc/ms_gym/mol_df.pkl")
spec_df = pd.read_pickle("data/proc/pcdl_nl/spec_df.pkl")

#df = pd.read_csv("data/df/pcdl_df.csv")
#print(df.head())
#print(df.columns)
print(spec_df["prec_type"].value_counts())
print(spec_df["inst_type"].unique())