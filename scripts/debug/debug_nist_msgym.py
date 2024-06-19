import pandas as pd
import numpy as np
#mol_df = pd.read_pickle("data/proc/ms_gym/mol_df.pkl")
mol_df = pd.read_pickle("data/proc/nist20mona23ms_gym_valtest/mol_df.pkl")
#spec_df = pd.read_pickle("data/proc/nist20mona23ms_gym_valtest/spec_df.pkl")
total_mol_ids_set= set(mol_df['mol_id'].to_list())
for split in ['train','val','test']:
    train_df = pd.read_csv("data/split/nist20mona23ms_gym_valtest_d3_h4_inchikey/train_ids.csv")
    mol_ids_list = train_df['mol_id'].to_list()
    split_mol_id_set = set(mol_ids_list)
    print(split_mol_id_set -total_mol_ids_set )
