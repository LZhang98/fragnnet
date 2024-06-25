#!/bin/bash

# inchikey
python scripts/ms2c/run_iceberg_gen.py \
    --iceberg_gen_ckpt_fp=saved_ckpts/019gu0eq_iceberg_gen/ckpt/model-epoch=041.ckpt \
    --task_dp=data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
    --magma_dp=data/ms2c/pubchem/magma/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_magma

# scaffold
python scripts/ms2c/run_iceberg_gen.py \
    --iceberg_gen_ckpt_fp=saved_ckpts/dizgfqmp_iceberg_gen/ckpt/model-epoch=040.ckpt \
    --task_dp=data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
    --magma_dp=data/ms2c/pubchem/magma/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50_magma