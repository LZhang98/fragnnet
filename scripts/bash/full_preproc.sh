#!/bin/bash
set -euo pipefail

# perform preproc and splitting pipeline on all datasets of interest
# all dags should be depth 3 and the dag dir should be <dset_name>_d3

D=data_v2
DAG_FLAGS=(--max_depth 3 --max_h_transfer 4 --isotopes False --nb_isomorphic False --max_time -1 --wandb_mode disabled)

# NIST20: starts with proc already done.
# generate dag features and splits (both scaffold and inchikey)

uv run python preproc_scripts/03_prepare_dag_feats.py \
  --frag_dp $D/frag/nist20_d3 \
  --proc_dp $D/proc/nist20 \
  --dsets nist20_hr \
  --prec_types "[M+H]+" \
  --inst_types ORB ORB_F \
  --frag_modes HCD \
  --ion_modes P \
  "${DAG_FLAGS[@]}"

for SPLIT_KEY in inchikey_s scaffold; do
  SPLIT_NAME=${SPLIT_KEY%_s}
  uv run python preproc_scripts/04_prepare_split.py \
    --split_type random \
    --split_key $SPLIT_KEY \
    --primary_dsets nist20_hr \
    --prec_types "[M+H]+" \
    --inst_types ORB ORB_F \
    --frag_modes HCD \
    --ion_modes P \
    --proc_dp $D/proc/nist20 \
    --frag_dp $D/frag/nist20_d3 \
    --split_dp $D/split/nist20_${SPLIT_NAME}
done

# Enveda-180. starts with raw/enveda-180/enveda-180-filtered.jsonl.gz
# generate csv, then proc, then dags, then splits (both scaffold and inchikey)

ENVEDA_PREC=("[M+H]+" "[M-H]-" "[M+Na]+" "[M+NH4]+" "[M+Cl]-" "[M+K]+")

mkdir -p $D/df/enveda_180

uv run python preproc_scripts/01_prepare_df.py \
  --input_format enveda \
  --output_format csv \
  --output_name enveda_180 \
  --raw_data_dp $D/raw \
  --output_dp $D/df/enveda_180 \
  --jsonl_file enveda-180/enveda-180-filtered.jsonl.gz

uv run python preproc_scripts/02_prepare_proc.py \
  --df_dp $D/df/enveda_180 \
  --dsets enveda_180 \
  --proc_dp $D/proc/enveda_180

uv run python preproc_scripts/03_prepare_dag_feats.py \
  --frag_dp $D/frag/enveda_180_d3 \
  --proc_dp $D/proc/enveda_180 \
  --dsets enveda_180 \
  --prec_types "${ENVEDA_PREC[@]}" \
  --inst_types QTOF \
  --frag_modes CID \
  --ion_modes P N \
  "${DAG_FLAGS[@]}"

for SPLIT_KEY in inchikey_s scaffold; do
  SPLIT_NAME=${SPLIT_KEY%_s}
  uv run python preproc_scripts/04_prepare_split.py \
    --split_type random \
    --split_key $SPLIT_KEY \
    --test_frac 0.1 \
    --val_frac 0.1 \
    --meta_rseed 42 \
    --primary_dsets enveda_180 \
    --prec_types "${ENVEDA_PREC[@]}" \
    --ce_types stepped single \
    --proc_dp $D/proc/enveda_180 \
    --frag_dp $D/frag/enveda_180_d3 \
    --split_dp $D/split/enveda_180_${SPLIT_NAME}
done

# MSGym: starts with raw/massspecgym/MassSpecGym1.5.tsv.
# generate csv, then proc, then dags, then splits (predefined, scaffold, and inchikey)

mkdir -p $D/df/ms_gym

uv run python preproc_scripts/01_prepare_df.py \
  --input_format ms_gym \
  --output_format csv \
  --output_name ms_gym \
  --raw_data_dp $D/raw \
  --output_dp $D/df/ms_gym \
  --ms_gym_tsv massspecgym/MassSpecGym1.5.tsv

uv run python preproc_scripts/02_prepare_proc.py \
  --df_dp $D/df/ms_gym \
  --dsets ms_gym \
  --proc_dp $D/proc/ms_gym

uv run python preproc_scripts/03_prepare_dag_feats.py \
  --frag_dp $D/frag/ms_gym_d3 \
  --proc_dp $D/proc/ms_gym \
  --dsets ms_gym \
  --prec_types "[M+H]+" \
  --inst_types ORB QTOF \
  --frag_modes CID \
  --ion_modes P \
  "${DAG_FLAGS[@]}"

# predefined: split the official MassSpecGym folds (ms_gym_fold.csv: dset_spec_id,fold) into id files
uv run python - <<EOF
import pandas as pd
dp = "$D/df/ms_gym"
fold_df = pd.read_csv(f"{dp}/ms_gym_fold.csv")
for fold in ["train", "val", "test"]:
    fold_df.loc[fold_df["fold"] == fold, ["dset_spec_id"]].to_csv(
        f"{dp}/ms_gym_{fold}_ids.csv", index=False
    )
EOF

uv run python preproc_scripts/04_prepare_split.py \
  --split_dp $D/split/ms_gym_predefined \
  --split_type predefined \
  --predefined_id_type dset_spec_id \
  --predefined_train_id_fp $D/df/ms_gym/ms_gym_train_ids.csv \
  --predefined_val_id_fp $D/df/ms_gym/ms_gym_val_ids.csv \
  --predefined_test_id_fp $D/df/ms_gym/ms_gym_test_ids.csv \
  --primary_dsets ms_gym \
  --max_prec_mz 1000.0 \
  --prec_types "[M+H]+" \
  --inst_types ORB QTOF \
  --frag_modes CID \
  --ion_modes P \
  --proc_dp $D/proc/ms_gym \
  --frag_dp $D/frag/ms_gym_d3

for SPLIT_KEY in inchikey_s scaffold; do
  SPLIT_NAME=${SPLIT_KEY%_s}
  uv run python preproc_scripts/04_prepare_split.py \
    --split_type random \
    --split_key $SPLIT_KEY \
    --primary_dsets ms_gym \
    --max_prec_mz 1000.0 \
    --prec_types "[M+H]+" \
    --inst_types ORB QTOF \
    --frag_modes CID \
    --ion_modes P \
    --proc_dp $D/proc/ms_gym \
    --frag_dp $D/frag/ms_gym_d3 \
    --split_dp $D/split/ms_gym_${SPLIT_NAME}
done
