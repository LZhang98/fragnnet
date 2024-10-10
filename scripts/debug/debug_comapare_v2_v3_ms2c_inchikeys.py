
import os
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

print("> scaffold ")
#old_mol_fp = "data/ms2c/pubchem/proc/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/mol_df.pkl.gz"
#new_mol_fp = "data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/mol_df.pkl.gz"

#old_mol_df = pd.read_pickle(old_mol_fp)
#old_inchikey_s = old_mol_df['inchikey_s'].to_list()
#print(old_inchikey_s)
#new_mol_df = pd.read_pickle(new_mol_fp)

#new_inchikey_s = new_mol_df['inchikey_s'].to_list()
#print("In old no in new", len(set(old_inchikey_s) - set(new_inchikey_s)))
#print("In new no in old",  len(set(new_inchikey_s) - set(old_inchikey_s)))

print("> inchikey ")
old_mol_fp = "data/ms2c/pubchem/proc/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/mol_df.pkl.gz"
new_mol_fp = "data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/mol_df.pkl.gz"

old_mol_df = pd.read_pickle(old_mol_fp)
new_mol_df = pd.read_pickle(new_mol_fp)

old_inchikey_s = old_mol_df['inchikey_s'].to_list()
new_inchikey_s = new_mol_df['inchikey_s'].to_list()
print("In old no in new", len(set(old_inchikey_s) - set(new_inchikey_s)))
print("In new no in old", len(set(new_inchikey_s) - set(old_inchikey_s)))

old_inchikey_s = old_mol_df['smiles'].to_list()
new_inchikey_s = new_mol_df['smiles'].to_list()
print("smiles in old no in new", len(set(old_inchikey_s) - set(new_inchikey_s)))
print("smiles in new no in old", len(set(new_inchikey_s) - set(old_inchikey_s)))