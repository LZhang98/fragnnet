#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=preproc_infer_synced_dag_arr
#SBATCH --mem=180G
#SBATCH --cpus-per-task=96
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_synced_dag_arr_%A_%a.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_synced_dag_arr_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=0-14

# SYNCED inference preprocessing, STEP 03 ONLY (DAG feature generation), as an array
# job -> one bz2 pickle per molecule under dags/, shipped as one .tar per shard.
#
# Array sizing: 289,480 molecules / CHUNK_SIZE=20000 -> 15 tasks (0-14). Keep
# --array and CHUNK_SIZE in sync when either changes.

DATASET_NAME=synced
MAX_DEPTH=3
MAX_H_TRANSFER=4
MAX_TIME=150
USE_HDF5=False
USE_LMDB=False
SKIP_STATS=True
SAVE_NAME=${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}
CHUNK_SIZE=20000

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
if [ -f "${SCRIPT_DIR}/preproc_infer_synced_dag_arr_common.sh" ]; then
    source "${SCRIPT_DIR}/preproc_infer_synced_dag_arr_common.sh"
else
    source "${FRAGNNET_DP}/scripts/manual_slurm/fir/preproc_infer_synced_dag_arr_common.sh"
fi
