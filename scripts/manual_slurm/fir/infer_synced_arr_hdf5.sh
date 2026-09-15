#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=infer_synced_arr_hdf5
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/infer_synced_arr_hdf5_%A_%a.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/infer_synced_arr_hdf5_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=0-14

# SYNCED inference, packed HDF5 frag backend, as an array job.
#
# Consumes the per-shard HDF5 written by preproc_infer_synced_dag_arr_hdf5.sh:
#   ${DEST_FRAG_DP}/synced_d3_h4_hdf5/shard_<task>/{dags.h5,meta_info.json}
# Task i predicts mol_df rows [i*CHUNK_SIZE, (i+1)*CHUNK_SIZE) - the same slice that
# built shard i - so merge_dag_h5_shards.py is NOT needed here; merge only when you
# want to run inference as a single job. Keep --array and CHUNK_SIZE in sync with the
# preprocessing array (289,480 molecules / 20000 -> 15 tasks).
#
# Output: ${OUT_DP}/shard_<frag_shard>_<sub>/predictions_chunk_*.parquet

DATASET_NAME=synced
MAX_DEPTH=3
MAX_H_TRANSFER=4
SAVE_NAME=${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_hdf5
FRAG_BACKEND=hdf5
CHUNK_SIZE=20000

# One array task per frag shard by default. Raise SUB_SHARDS to split each shard
# across several tasks (shorter per-task wall time, same DAGs) and widen --array to
# 0-(15*SUB_SHARDS - 1) to match; tasks past the end of mol_df exit cleanly.
SUB_SHARDS=${SUB_SHARDS:-1}

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
if [ -f "${SCRIPT_DIR}/infer_synced_arr_hdf5_common.sh" ]; then
    source "${SCRIPT_DIR}/infer_synced_arr_hdf5_common.sh"
else
    source "${FRAGNNET_DP}/scripts/manual_slurm/fir/infer_synced_arr_hdf5_common.sh"
fi
