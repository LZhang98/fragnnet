import pandas as pd
import numpy as np
#mol_df = pd.read_pickle("data/proc/ms_gym/mol_df.pkl")
mol_df = pd.read_pickle("data/proc/ms_gym/mol_df.pkl")
spec_df = pd.read_pickle("data/proc/ms_gym/spec_df.pkl")

for split in ['train','val','test']:
    id_df = pd.read_csv(f"data/split/ms_gym_d3_h4_benchmark/{split}_ids.csv")
    print("id_df", id_df.shape)
    spec_ids_list = id_df['spec_id'].to_list()
    split_spec_df = spec_df[spec_df["spec_id"].isin(spec_ids_list)]
    print("split_spec_df",split_spec_df['inst_type'].unique())
    print("split_spec_df",split_spec_df.shape)
    print("split_spec_df max_prec_mz",split_spec_df["prec_mz"].max())
    #split_mol_id_set = set(mol_ids_list)
    #print(split_mol_id_set -total_mol_ids_set )
