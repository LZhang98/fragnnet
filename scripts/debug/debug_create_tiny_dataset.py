import os
import argparse
import pandas as pd

if __name__ == "__main__":
    
    #project_dp = os.path.join("data","debug")
    src_proc_dp = "data/proc/nist20"
    mol_df_fp = os.path.join(src_proc_dp,"mol_df.pkl")
    #print(f"read in mol from {mol_df_fp}")
    mol_df = pd.read_pickle(mol_df_fp)
    print(mol_df.shape)
 
    mol_df = mol_df.sample(128)
    mol_ids = mol_df['mol_id'].to_list()
    print(mol_df.shape)
  
    sped_df_fp = os.path.join(src_proc_dp,"spec_df.pkl")
    print(f"read in spec from {sped_df_fp}")
    spec_df = pd.read_pickle(sped_df_fp)
    print(spec_df.shape)
    spec_df = spec_df[spec_df['mol_id'].isin(mol_ids)]
    spec_df = spec_df[spec_df['prec_type'].isin(['[M+H]+','[M+Na]+','[M-H]-'])]
    print(spec_df.shape)

    dst_proc_dp = "data/proc/debug_tinyv4"
    if not os.path.exists(dst_proc_dp):
        os.makedirs(dst_proc_dp)
    mol_df_fp = os.path.join(dst_proc_dp,"mol_df.pkl")
    mol_df.to_pickle(mol_df_fp)
    spec_df_fp = os.path.join(dst_proc_dp,"spec_df.pkl")
    spec_df.to_pickle(spec_df_fp)
