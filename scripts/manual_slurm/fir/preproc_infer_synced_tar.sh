#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=preproc_infer_synced_tar
#SBATCH --mem=180G
#SBATCH --cpus-per-task=96
#SBATCH --time=0-24:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_synced_tar_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_synced_tar_%j.err
#SBATCH --open-mode=append

# SYNCED inference preprocessing -> one bz2 pickle per molecule under dags/,
# shipped to scratch as a single .tar (the frag_gen_*.sh convention).

DATASET_NAME=synced
MAX_DEPTH=3
MAX_H_TRANSFER=4
MAX_TIME=150
USE_HDF5=False
SKIP_STATS=True
SAVE_NAME=${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
if [ -f "${SCRIPT_DIR}/preproc_infer_synced_common.sh" ]; then
    source "${SCRIPT_DIR}/preproc_infer_synced_common.sh"
else
    source "${FRAGNNET_DP}/scripts/manual_slurm/fir/preproc_infer_synced_common.sh"
fi
