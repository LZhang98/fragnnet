import pandas as pd
import numpy as np
#mol_df = pd.read_pickle("data/proc/ms_gym/mol_df.pkl")
mol_df = pd.read_pickle("data/proc/ms_gym/mol_df.pkl")
spec_df = pd.read_pickle("data/proc/ms_gym/spec_df.pkl")

#mol_id_a = mol_df["mol_id"]
#mol_id_b = mol_df["mol_id"]

mol_df_id_set = set(mol_df["mol_id"].to_list())
spec_df_id_set = set(spec_df["mol_id"].to_list())

#print(np.all(np.unique(mol_df["mol_id"]) == np.unique(spec_df["mol_id"])))
#print(len(mol_df_id_set))
#print(len(spec_df_id_set))

#print(mol_df_id_set - spec_df_id_set)
#print(spec_df_id_set - mol_df_id_set)
print(mol_df[mol_df["mol_id"] == 46]['inchikey_s'])