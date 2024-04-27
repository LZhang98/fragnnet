#!/bin/bash
# inchikey
python preproc_scripts/09_perpare_ms2c_mol_and_spec_df.py \
    --candidate_pkl_fp=./data/ms2c/pubchem/candidates/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.pkl.gz \
    --custom_fp="config/nist20_d4_wr00_inchikey/fraggnn_d4_nodemlp_mnce_nocp.yml" 
    
# scaffold
python preproc_scripts/09_perpare_ms2c_mol_and_spec_df.py\
    --candidate_pkl_fp=data/ms2c/pubchem/candidates/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.pkl.gz \
    --custom_fp="config/nist20_d4_wr00_scaffold/fraggnn_d4_nodemlp_mnce_nocp.yml" 
