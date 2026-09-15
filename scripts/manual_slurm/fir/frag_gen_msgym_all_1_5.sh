#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_msgym_all_1_5
#SBATCH --mem=256G
#SBATCH --cpus-per-task=128
#SBATCH --time=0-3:00:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_msgym_all_1_5_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_msgym_all_1_5_%j.err
#SBATCH --open-mode=append

DATASET_NAME=ms_gym_all_1_5
MAX_DEPTH=3
MAX_H_TRANSFER=4
MAX_TIME=-1
NB_FLAG=False
ISO_FLAG=False
SMARTS_PREPASS=False
USE_HDF5=False
SAVE_NAME=${DATASET_NAME}_ma_mi_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_iso${ISO_FLAG}_smp${SMARTS_PREPASS}
DSETS='ms_gym_benchmark_1_5 ms_gym_extra_1_5'
PREC='[M+H]+ [M+Na]+'
INST='FT QTOF NA'
FRAG_MODE='CID HCD NA'
DEST_DP=/scratch/feiw/frag-gnn-data/frags/

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
if [ -f "${SCRIPT_DIR}/frag_gen_common.sh" ]; then
    source "${SCRIPT_DIR}/frag_gen_common.sh"
else
    source "${FRAGNNET_DP}/scripts/manual_slurm/fir/frag_gen_common.sh"
fi
