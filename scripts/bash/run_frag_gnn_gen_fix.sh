#!/bin/bash
ulimit -n 2048
# inchikey d4
python ./scripts/ms2c/fix_icml_ms2c_candidates.py \
	--candidate_pkl_fp='data/ms2c/pubchem/candidates/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.pkl.gz'

# scaffold d4
python ./scripts/ms2c/fix_icml_ms2c_candidates.py \
	--candidate_pkl_fp='data/ms2c/pubchem/candidates/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.pkl.gz'
