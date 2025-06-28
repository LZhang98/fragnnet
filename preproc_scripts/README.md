# Data Preprocessing

## Step 0: Exporting the NIST 2020 MS/MS library

All of the data used in the paper is based on the NIST 2020 MS/MS library. Since this library is not publicly available, we are unfortunately unable to provide access to it.

The raw NIST data can be exported using the instructions [here](https://github.com/Roestlab/massformer).

## Step 1: Processing raw data into CSV

### NIST 20
This assumes that the NIST20 data is in a folder `data/raw/nist_20`, containing two files: `hr_nist_msms.MSP` (the MSP file with the spectrum information) and `hr_nist_msms.MOL` (the MOL file with the molecule structures).

This will create a new directory `data/df`

```bash
python preproc_scripts/01_prepare_df.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/df --output_name nist20_hr
```

### MassSpecGym
This assumes that the Spectraverse data is in `data/raw/msgym/MassSpecGym_with_test.tsv`, this will use `ms_gym`,`ms_gym_extra` as dataset name for each part of MassSpecGym

```bash
# for sim challenege
python preproc_scripts/01_prepare_df.py --input_format=ms_gym --output_format=csv --output_name=ms_gym  --ms_gym_tsv=msgym/MassSpecGym_with_test.tsv

# for extra data in msgym
python preproc_scripts/01_prepare_df.py --input_format=ms_gym_extra --output_format=csv --output_name=ms_gym_extra --ms_gym_tsv=msgym/MassSpecGym_with_test.tsv
```

### Spectraverse
This assumes that the Spectraverse data is in `data/raw/spectraverse/spectraverse-0.0.1.mgf`

```bash
python preproc_scripts/01_prepare_df.py \
	--input_format=spectraverse \
    --output_format=csv \
    --output_name=spectraverse \
    --mgf_file=spectraverse/spectraverse-0.0.1.mgf
```

## Step 2: Processing CSV into Pickle Files

This step converts the data into binary format (pickle) for training and evaluation.

`spec_df.pkl` contains spectrum information.
`mol_df.pkl` contains molecule information.
`ann_df.pkl` contains annotation information.

### NIST 20
```bash
python preproc_scripts/02_prepare_proc.py \
    --df_dp data/df \
    --dsets nist20_hr \
    --proc_dp data/proc/nist20
```

### MSGYM
```bash
# for all data in msgym
python preproc_scripts/02_prepare_proc.py \
    --proc_dp=data/proc/ms_gym_all \
    --dsets ms_gym ms_gym_extra
```
### Spectraverse
```bash
python preproc_scripts/02_prepare_proc.py \
	--proc_dp=data/proc/spectraverse \
    --dsets spectraverse
```
## Step 3: Generating Fragmentation DAGs

### NIST 20
This step generates fragmentation DAGs that are used for the FraGNNet models. Since fragmentation can be slow, we recommend using a compute with many cores (by default, the script will use all available cores). 

Frag configuration: depth 3 (used for FraGNNet-D3), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 3 --frag_dp data/frag/nist20_d3 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode disabled
```

Frag configuration: depth 4 (used for FraGNNet-D4), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 4 --frag_dp data/frag/nist20_d4 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --dsets nist20_hr--wandb_mode disabled
```

### MassSpecGym

```bash
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
```

### Spectraverse

```bash
# Benchmark split (sim chanllenge)
python preproc_scripts/04_prepare_split.py \
  --proc_dp=data/proc/ms_gym_all \
  --split_dp=./data/split/spectraverse \
  --frag_dp=./data/frag/spectraverse_d3_h4 \
  --primary_dsets spectraverse \
  --split_type=predefined_dsetid_csv \
  --predefined_dsetid_csv=data/df/spectraverse_fold.csv
```

## Step 4: Preparing Splits

This step splits the data into training, validation, and test sets. To exactly reproduce the splits used in the paper, download the split id files ([inchikey](), [scaffold]()) and place them in `data/split/nist20_inchikey/` and `data/split/nist20_scaffold/` respectively. Then, run the following scripts to remap the ids to work with the processed dataset from the previous step:

Inchikey split:

```bash
python preproc_scripts/04_prepare_split.py --split_type predefined --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --dag_filtering False --proc_dp data/proc/nist20 --predefined_id_type dset_spec_id --predefined_train_id_fp data/split/nist20_inchikey/train_dset_ids.csv --predefined_val_id_fp data/split/nist20_inchikey/val_dset_ids.csv --predefined_test_id_fp data/split/nist20_inchikey/test_dset_ids.csv --split_dp data/split/nist20_inchikey
```

Scaffold split:

```bash
python preproc_scripts/04_prepare_split.py --split_type predefined --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --dag_filtering False --proc_dp data/proc/nist20 --predefined_id_type dset_spec_id --predefined_train_id_fp data/split/nist20_scaffold/train_dset_ids.csv --predefined_val_id_fp data/split/nist20_scaffold/val_dset_ids.csv --predefined_test_id_fp data/split/nist20_scaffold/test_dset_ids.csv --split_dp data/split/nist20_scaffold
```

If you don't want to use the exact splits that we used (just ones with similar statistics), you can use the same script to generate random splits:

Inchikey split:

```bash
python preproc_scripts/04_prepare_split.py --split_type random --split_key inchikey_s --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_inchikey
```

Scaffold split:

```bash
python preproc_scripts/04_prepare_split.py --split_type random --split_key scaffold --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_scaffold
```

## Step 5: Preparing Optimal MAGMa DAGs (ICEBERG)

This step generates optimal MAGMa DAGs for training/evaluating the ICEBERG generator model optimal ICEBERG intensity model.

```bash
python preproc_scripts/05_prepare_magma_feats.py --magma_dp data/magma/gen/nist20 --proc_dp data/proc/nist20 --dsets nist20_hr
```

## Step 6: Preparing Approximate MAGMa DAGs (ICEBERG)

This step generates approximate MAGMa DAGs using a trained ICEBERG generator model, for training and evaluating the ICEBERG intensity model.

For each seed `${SEED}` (see [config README](../config/README.md)) repeat the following:

```bash
python preproc_scripts/06_predict_magma_dags.py --magma_dp data/magma/inten/nist20_inchikey_s$SEED --proc_dp data/proc/nist20 --gen_ckpt_fp data/magma/ckpt/nist20_inchikey_s$SEED.ckpt --dsets nist20_hr
```

Don't forget to symlink the formula directory:

```bash
ln -s $HOME/frag-gnn/data/magma/gen/nist20/magma_formula $HOME/frag-gnn/data/magma/inten/nist20_inchikey_s$SEED/magma_formula
```
