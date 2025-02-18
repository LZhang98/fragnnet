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

Inchikey:

`python preproc_scripts/04_prepare_split.py --split_key inchikey_s --primary_dsets nist20_hr --proc_dp data/proc/nist20mona23v3 --frag_dp data/frag/nist20mona23v3_d4_h4_isoFalse_nb3 --split_dp data/split/nist20mona23v3_d4_wr00_inchikey`

Scaffold:

`python preproc_scripts/04_prepare_split.py --split_key scaffold --primary_dsets nist20_hr --proc_dp data/proc/nist20mona23v3 --frag_dp data/frag/nist20mona23v3_d4_h4_isoFalse_nb3 --split_dp data/split/nist20mona23v3_d4_wr00_scaffold`

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

## How Run MSGym MS2C

### How Prepare MSGym Data

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

# How Run Pubchem MS2C Tasks

PubChem MS2C will use validation functions instead of testing functions. The idea is to supply ground truth spectra to every validation prediction call (with different molecules). This would save some code by running similarity calculations separately. Current setup uses `cos_sim@0.01` and `cos_hun`. * NOTE current setup only works with merged spectrum and single adduct type, it also relies on this assumption to work. Code change is required when we extend this to multiple adduct and engeries * .

## How Prepare Pubchem MS2C Data

This is only required if you does not have all the files before run ms2c prediction.

### Create Pubchem MS2C Pubchem Database

This script will download *latest* sdf from pubchem and process it  into a sqlite, only run this when you don't have an existing pubchem sqlite file.

``` bash
python preproc_scripts/pubchem_ms2c/01_prepare_pubchem.py  
```

### Create Pubchem MS2C Candidates Set

This script is used to create ms2c_candidates from a given split. We are using `split_type=test` for all our experiments. Set `include_sanity=True` to have a sanity check set, sanity check set is roughly equal to `split_type` only set. This is useful to check if there are miss-aligenment between data and/or there is an issue with libaray. It is also very useful to debug compute related issues such as numeric issue on  compute canada system libaray.  

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

### Create Pubchem MS2C Proc

From canidates, we will create proc for `c2ms` prediction.

```bash
# pubchem ms2c for scaffold split
python preproc_scripts/pubchem_ms2c/03_perpare_ms2c_mol_and_spec_df.py \
 --candidate_pkl_fp=./data/ms2c/pubchem/candidates/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.pkl.gz \
 --proc_dp=./data/proc/nist20mona23v3 \
 --split_dp=./data/split/nist20mona23v3_d4_wr00_inchikey \
 --split_type=test \
 --ms2c_data_dp=./data/ms2c/pubchem/proc

# pubchem ms2c for scaffold split
python preproc_scripts/pubchem_ms2c/03_perpare_ms2c_mol_and_spec_df.py \
 --candidate_pkl_fp=./data/ms2c/pubchem/candidates/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.pkl.gz \
 --proc_dp=./data/proc/nist20mona23v3 \
 --split_dp=./data/split/nist20mona23v3_d4_wr00_scaffold \
 --split_type=test \
 --ms2c_data_dp=./data/ms2c/pubchem/proc
```

## Run C2MS prediction for MS2C tasks

### Create Pubchem MS2C DAGs for FraGNNet

```bash
python preproc_scripts/pubchem_ms2c/04_run_ms2c_frag_gen.py \
 --frag_dp=pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50  \
 --proc_dp=data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
 --max_depth=3
```

### Run Prediction for Fragnnet and other models

If you wish to run `fragnnet`,`nemis`,`massformer` predict and provide wandb run id, following script will download best checkpoint from given run id and perform prediction. Model type is determined from checkpoint automatically.

```bash
python scripts/ms2c/pubchem_v1/run_ms2c_predict_pubchem.py \
     --device=cpu\
    --save_fp=save_fp \
    --magma_dp=magma_dp \
    --iceberg_gen_wandb_run_id=i7k9gc3d \
    --iceberg_intern_wandb_run_id=c8iwu7il \
    --proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
```

In case that wandb is not accessaible, you can also supply following script wil checkpoint file.

```bash
python scripts/ms2c/pubchem_v1/run_ms2c_predict_pubchem.py \
    --device=cpu\
    --save_fp=save_fp\
    --magma_dp=magma_dp\
    --iceberg_gen_ckpt_fp=./saved_ckpts/nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.ckpt\
    --iceberg_intern_ckpt_fp=./saved_ckpts/nist20v3_inchikey_iceberg_inten_100_s0_c8iwu7il.ckpt\
    --proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
```

### Run Prediction for Iceberg models

If you wish to run `iceberg gen` and `iceberg predict` and provide wandb run id, following script will download best checkpoint from given run id and perform prediction.  

```bash
  python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
    --device=cpu \
    --save_fp=save_fp \
    --magma_dp=magma_dp \
    --iceberg_gen_wandb_run_id=i7k9gc3d \
    --iceberg_intern_wandb_run_id=c8iwu7il \
    --proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
```

In case that wandb is not accessaible, you can also supply following script wil checkpoint file.

```bash
  python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
    --device=cpu\
    --save_fp=save_fp\
    --magma_dp=magma_dp\
    --iceberg_gen_ckpt_fp=./saved_ckpts/nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.ckpt\
    --iceberg_intern_ckpt_fp=./saved_ckpts/nist20v3_inchikey_iceberg_inten_100_s0_c8iwu7il.ckpt\
    --proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
```

## Run Evalution

### Create Rank only pkl [Optional]

This is use to reduce size of pkl to read, in case we don't care about predicted ms/ms

```bash
    python scripts/ms2c/pubchem_v1/create_ranking_only_pkl.py \
      --save_dp=./data/ms2c/pubchem/run_240925/benchmark/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50_rank_only
      --predicted_dp="./data/ms2c/pubchem/run_240925/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
```

### Create Ranking Stats

```bash
    python scripts/ms2c/pubchem_v1/eval_pubchem_retrieval_val.py \
      --save_dp=./data/ms2c/pubchem/run_240925/benchmark/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
      --proc_dp=./data/ms2c/pubchem/run_240925/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
      --predicted_dp="./data/ms2c/pubchem/run_240925/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
```

### Create NDCG Stats

```bash
    python scripts/ms2c/pubchem_v1/eval_ndcg.py \
      --save_dp=./data/ms2c/pubchem/run_240925/benchmark/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
      --proc_dp=./data/ms2c/pubchem/run_240925/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
      --predicted_dp="./data/ms2c/pubchem/run_240925/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
```
