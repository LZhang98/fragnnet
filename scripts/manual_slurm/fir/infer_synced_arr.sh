#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=infer_synced_arr
# Resources are the fir:h100-3g.40gb preset from GPU_PROFILES in scripts/setup_job_cc.py
# (python scripts/setup_job_cc.py --list_gpu_profiles) - the same MIG slice
# infer_synced_arr_hdf5.sh uses, at half the price of a whole H100. Keep the three lines
# below as one unit when switching presets: the cpu/mem values are that slice's
# proportional share, so a mismatched pair bills at the higher ratio. Inference runs with
# num_workers=0, so the cpu count only covers the shard staging copy, not dataloading.
#SBATCH --mem=140G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --time=0-12:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/infer_synced_arr_%A_%a.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/infer_synced_arr_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=0-14

# SYNCED inference, tar (pickle folder) frag backend, as an array job.
#
# Consumes the per-shard tars written by preproc_infer_synced_dag_arr.sh:
#   ${DEST_FRAG_DP}/synced_d3_h4/synced_d3_h4_<task>.tar
# Task i predicts mol_df rows [i*CHUNK_SIZE, (i+1)*CHUNK_SIZE) - the same slice that
# built shard i - so no tar merging is needed. Keep --array and CHUNK_SIZE in sync
# with the preprocessing array (289,249 molecules / 20000 -> 15 tasks).
#
# Output: ${OUT_DP}/shard_<task>/predictions_chunk_*.pkl

DATASET_NAME=synced
MAX_DEPTH=3
MAX_H_TRANSFER=4
SAVE_NAME=${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}
FRAG_BACKEND=tar
CHUNK_SIZE=20000

# --- MODEL: set one of these two ------------------------------------------
# Local checkpoint. Relative paths resolve against FRAGNNET_DP.
MODEL_CKPT=${MODEL_CKPT:-saved_ckpts/deepmet/nist20_inchikey_fraggnn_d3_ma_mi_amolc_cv_all_hifszbst.ckpt}
# ...or a W&B run id, used only when MODEL_CKPT is empty. The download works from the
# compute nodes, but warm the cache once before submitting so the tasks do not race
# each other writing the same cached .ckpt:
#   python -c "from fragnnet.utils.misc_utils import get_best_ckpt_from_wandb; \
#              print(get_best_ckpt_from_wandb('saved_ckpts', '<run_id>'))"
WANDB_RUN_ID=${WANDB_RUN_ID:-}
# Training config for that checkpoint, if it needs one (relative to FRAGNNET_DP).
CUSTOM_FP=${CUSTOM_FP:-}

BATCH_SIZE=${BATCH_SIZE:-16}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
if [ -f "${SCRIPT_DIR}/infer_synced_arr_common.sh" ]; then
    source "${SCRIPT_DIR}/infer_synced_arr_common.sh"
else
    source "${FRAGNNET_DP}/scripts/manual_slurm/fir/infer_synced_arr_common.sh"
fi
