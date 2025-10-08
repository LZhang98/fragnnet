import pandas as pd
import numpy as np

if __name__ == "__main__":
    spec_df = pd.read_pickle('data/proc/spectraverse/spec_df.pkl')
    mol_df = pd.read_pickle('data/proc/spectraverse/mol_df.pkl')
    print(f"Loaded {len(spec_df)} spectra")
    print(spec_df.head(5))
    print(spec_df.columns)

    print(f"Unique spec_df['mol_id'].nunique(): {spec_df['mol_id'].nunique()}")
    print(f"Loaded {len(mol_df)} molecules")
    
    print(spec_df[spec_df['spec_id'] == 53028][['spec_id', 'mol_id', 'nce', 'nce_extra_1', 'nce_extra_2', 'ace', 'ace_extra_1', 'ace_extra_2']])
    #spec_mol_df = spec_df.merge(mol_df, on='mol_id')
    #print(spec_mol_df.columns)
    #colnames = ['spec_id', 'mol_id', 'inchikey_s', 'nce', 'nce_extra_1', 'nce_extra_2', 'ace', 'ace_extra_1', 'ace_extra_2']
    #print(spec_df[spec_df['nce'] >= spec_df['nce_extra_1']])
    #print(spec_mol_df[spec_mol_df['ace'] >= spec_mol_df['ace_extra_1']][colnames])
    #print(spec_mol_df[spec_mol_df['nce'] >= spec_mol_df['nce_extra_1'] & pd.notna(spec_mol_df['nce_extra_1']) &  ~pd.notna(spec_mol_df['nce_extra_2'])][colnames])
    #print(spec_mol_df[spec_mol_df['nce'] == spec_mol_df['nce_extra_1']][colnames])