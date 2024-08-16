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

`python preproc_scripts/03_prepare_dag_feats.py --max_depth 4 --frag_dp data/frag/nist20mona23v3_d4_h4_isoFalse_nb3 --proc_dp data/proc/nist20mona23v3 --prec_types "[M+H]+" --inst_types FT --frag_modes HCD --ion_modes P --wandb_mode online --wandb_run_name nist20mona23v3_d4_h4_isoFalse_nb3`

## Step 4: prepare the split

TODO

## Step 5: prepare the MAGMa DAGs (for the ICEBERG models)

TODO

## Step 6: predict MAGMa DAGs (using an ICEBERG model)

TODO

## Step 7: set up a training run

`python scripts/setup_job_vv.py --gpu_type rtx6000 --qos m --env FRAG-GNN-GPU4 --time -1 --job_id -1 --python_script_fp scripts/run_pl_model_fit.py --script_kwargs custom_fp=config/nist20v3_d4_wr00_inchikey/neims/s0.yml --job_name debug_v3_neims`
