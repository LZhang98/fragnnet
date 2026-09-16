#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=preproc_infer_synced_dag_arr_hdf5
#SBATCH --mem=180G
#SBATCH --cpus-per-task=96
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_synced_dag_arr_hdf5_%A_%a.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_synced_dag_arr_hdf5_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=0-14

# SYNCED inference preprocessing, STEP 03 ONLY (DAG feature generation), as an array
# job -> one packed dags.h5 per shard. Uses the --use_hdf5 True + --skip_stats True
# fast path in 03_prepare_dag_feats.py.
#
# Array sizing: 289,249 molecules / CHUNK_SIZE=20000 -> 15 tasks (0-14). Keep
# --array and CHUNK_SIZE in sync when either changes.
#
# Each task writes ${DEST_FRAG_DP}/${SAVE_NAME}/shard_<task>/{dags.h5,meta_info.json}.
# The dataset loader takes a SINGLE .h5 path, so merge the shards once the array
# finishes:
#   python preproc_scripts/inference/merge_dag_h5_shards.py \
#       --shard_dp /scratch/feiw/frag-gnn-data/inference/frags/synced_d3_h4_hdf5 \
#       --output_dp /scratch/feiw/frag-gnn-data/inference/frags/synced_d3_h4_hdf5_merged
# Molecule ids are global (the shards slice mol_df, they do not renumber it), so the
# merged packed index stays aligned with mol_df.

DATASET_NAME=synced
MAX_DEPTH=3
MAX_H_TRANSFER=4
MAX_TIME=150
USE_HDF5=True
SKIP_STATS=True
SAVE_NAME=${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_hdf5
CHUNK_SIZE=20000

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
if [ -f "${SCRIPT_DIR}/preproc_infer_synced_dag_arr_common.sh" ]; then
    source "${SCRIPT_DIR}/preproc_infer_synced_dag_arr_common.sh"
else
    source "${FRAGNNET_DP}/scripts/manual_slurm/fir/preproc_infer_synced_dag_arr_common.sh"
fi
