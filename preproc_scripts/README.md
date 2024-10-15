# How to prepare the data

## Step 1: process raw data into csv

This assumes that the NIST20 data is in a folder `data/raw/nist_20`, containing two files: `hr_nist_msms.MSP` (the MSP file with the spectrum information) and `hr_nist_msms.MOL` (the MOL file with the molecule structures).

This will create a new directory `data/dfv2`

`python preproc_scripts/01_prepare_df.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/dfv2 --output_name nist20_hr_df`

`python preproc_scripts/01_prepare_df.py --msp_file mona_23/mona_23.msp --input_format msp --output_format csv --output_dp data/dfv2 --output_name mona23_df`

## Step 2: convert into pickle file

`python preproc_scripts/02_prepare_proc.py --df_dp data/dfv2 --dsets nist20_hr mona23`

## Step 3: generate DAGs

Frag configuration: depth 4, nb isomorphism (3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 4 --frag_dp data/frag/nist20mona23v3_d4_h4_isoFalse_nb3 --proc_dp data/proc/nist20mona23v3 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode online --wandb_run_name nist20mona23v3_d4_h4_isoFalse_nb3
```

## Step 4: prepare the split

TODO

## Step 5: set up a training run

```bash
python scripts/setup_job_vv.py --gpu_type rtx6000 --qos m --env FRAG-GNN-GPU4 --time -1 --job_id -1 --python_script_fp scripts/run_pl_model_fit.py --script_kwargs custom_fp=config/nist20v3_d4_wr00_inchikey/neims/s0.yml --job_name debug_v3_neims
```

## Step 6: prepare the MAGMa DAGs (for the ICEBERG models)

```bash
python preproc_scripts/05_prepare_magma_feats.py --magma_dp data/magma/gen/nist20mona23v3 --proc_dp data/proc/nist20mona23v3 --dsets nist20_hr mona23
```

## Step 7: predict MAGMa DAGs (using a trained ICEBERG model)

For each seed `sN` repeat the following:

```bash
python preproc_scripts/06_predict_magma_dags.py --magma_dp data/magma/inten/nist20mona23v3_d4_wr00_inchikey_100_sN --proc_dp data/proc/nist20mona23v3 --gen_ckpt_fp data/magma/ckpt/nist20v3_d4_wr00_inchikey_sN.ckpt --dsets nist20_hr mona23
```

```bash
python scripts/setup_job_vv.py --gpu_type rtx6000 --qos m --env FRAG-GNN-GPU4 --time -1 --python_script_fp preproc_scripts/06_predict_magma_dags.py --script_kwargs magma_dp=data/magma/inten/nist20mona23v3_d4_wr00_inchikey_100_sN proc_dp=data/proc/nist20mona23v3 gen_ckpt_fp=ata/magma/ckpt/nist20v3_d4_wr00_inchikey_sN.ckpt dsets="nist20_hr mona23"
```

Don't forget to symlink the formula directory:

`ln -s $HOME/frag-gnn/data/magma/gen/nist20mona23v3/magma_formula $HOME/frag-gnn/data/magma/inten/nist20mona23v3_d4_wr00_inchikey_100_sN/magma_formula`

## Step 8: set up a training run

For example, NEIMS:

`python scripts/setup_job_vv.py --gpu_type rtx6000 --qos m --env FRAG-GNN-GPU4 --time -1 --job_id -1 --python_script_fp scripts/run_pl_model_fit.py --script_kwargs custom_fp=config/nist20v3_d4_wr00_inchikey/neims/s0.yml --job_name debug_v3_neims`

## Seeds

s0 - 397623
s1 - 634340
s2 - 444303
s3 - 103976
s4 - 132548

# How Prepare MSGym Data

```bash
# for sim challenege
python preproc_scripts/01_prepare_df.py --input_format=ms_gym --output_format=csv --output_name=ms_gym  --ms_gym_tsv=msgym/MassSpecGym_with_test.tsv

# for extra data in msgym
python preproc_scripts/01_prepare_df.py --input_format=ms_gym_extra --output_format=csv --output_name=ms_gym_extra --ms_gym_tsv=msgym/MassSpecGym_with_test.tsv
 
# for all data in msgym
python preproc_scripts/02_prepare_proc.py \
 --proc_dp=data/proc/ms_gym_all \
 --dsets ms_gym ms_gym_extra
 
# Run DAG Gen
python preproc_scripts/03_prepare_dag_feats.py \
 --wandb_mode=off \
 --max_depth=3 \
 --wandb_run_name=foo\
 --frag_dp=data/frag/ms_gym_d3_h4v2 \
 --proc_dp=data/proc/ms_gym_all \
 --dsets ms_gym ms_gym_extra \
 --prec_types [M+H]+ [M+Na]+

# Benchmark split (sim chanllenge)
python preproc_scripts/04_prepare_split.py \
  --proc_dp=data/proc/ms_gym_all \
  --ces ace \
  --split_dp=./data/split/ms_gym_d3_h4_benchmarkv2 \
  --frag_dp=./data/frag/ms_gym_d3_h4v2 \
  --primary_dsets ms_gym \
  --max_prec_mz=1000.0 \
  --inst_types FT QTOF \
  --split_type=predefined_dsetid_csv \
  --predefined_dsetid_csv=data/df/ms_gym_fold.csv
  
# Benchmark split all
# this is not for the paper
# just for fun
python preproc_scripts/04_prepare_split.py \
  --proc_dp=data/proc/ms_gym_all \
  --ces ace \
  --split_dp=./data/split/ms_gym_d3_h4_benchmark_all \
  --frag_dp=./data/frag/ms_gym_d3_h4v2 \
  --primary_dsets ms_gym ms_gym_extra \
  --max_prec_mz=1000.0 \
  --inst_types FT QTOF \
  --split_type=predefined_dsetid_csv \
  --predefined_dsetid_csv data/df/ms_gym_fold.csv data/df/ms_gym_extra_fold.csv
```

# How Prepare Pubchem MS2C Data

## Create Pubchem MS2C Candidates Set

```bash
# inchikey
python preproc_scripts/pubchem_ms2c/02_prepare_ms2c_candidates.py  \
 --split_dp="./data/split/nist20mona23v3_d4_wr00_inchikey" \
 --proc_dp="./data/proc/nist20mona23v3" \
 --split_type=test

# scaffold
python preproc_scripts/pubchem_ms2c/02_prepare_ms2c_candidates.py  \
 --split_dp="./data/split/nist20mona23v3_d4_wr00_scaffold" \
 --proc_dp="./data/proc/nist20mona23v3" \
 --split_type=test
```

## Create Pubchem MS2C Proc

```bash
# pubchem ms2c for scaffold split
python preproc_scripts/pubchem_ms2c/03_perpare_ms2c_mol_and_spec_df.py \
 --candidate_pkl_fp=./data/ms2c/pubchem/candidates/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.pkl.gz \
 --proc_dp=./data/proc/nist20mona23v3 \
 --split_dp=./data/split/nist20mona23v3_d4_wr00_scaffold \
 --split_type=test \
 --ms2c_data_dp=./data/ms2c/pubchem/proc
```

## Create Pubchem MS2C DAGs

```bash
python preproc_scripts/pubchem_ms2c/04_run_ms2c_frag_gen.py \
 --frag_dp=pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50  \
 --proc_dp=data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
 --max_depth=3
```
