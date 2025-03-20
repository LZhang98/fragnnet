#!/bin/bash

# inchikey
python scripts/ms2c/run_iceberg_inten.py \
    --iceberg_inten_ckpt_fp=saved_ckpts/f8p1azsm_iceberg_inten_100/ckpt/model-epoch=079.ckpt \
    --task_dp=data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
    --magma_dp=data/ms2c/pubchem/magma/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_magma \
    --save_filename=f8p1azsm_iceberg_inten_100.pkl.gz \
    --batch_size=32

# scaffold
python scripts/ms2c/run_iceberg_inten.py \
    --iceberg_inten_ckpt_fp=saved_ckpts/xnmchi3a_iceberg_inten_100/ckpt/model-epoch=078.ckpt \
    --task_dp=data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
    --magma_dp=data/ms2c/pubchem/magma/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50_magma \
    --save_filename=xnmchi3a_iceberg_inten_100.pkl.gz \
    --batch_size=32