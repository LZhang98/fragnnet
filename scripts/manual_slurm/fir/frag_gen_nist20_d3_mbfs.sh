#!/bin/bash
#SBATCH --account=def-rgreiner
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_nist20_d3_mbfs
#SBATCH --mem=180G
#SBATCH --cpus-per-task=48
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_nist20_d3_mbfs_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_nist20_d3_mbfs_%j.err
#SBATCH --open-mode=append

DATASET_NAME=nist20
MAX_DEPTH=3
MAX_H_TRANSFER=4
MAX_TIME=-1
NB_FLAG=False
ISO_FLAG=False
MIN_FRAG_ATOMS=0
SMARTS_PREPASS=False
USE_LMDB=False
MULTI_CUT_BFS=True
MAX_CUT_SIZE=2
MAX_RING_CUT_STEPS=1
RESTRICT_RING_OPEN=False
SAVE_NAME=${DATASET_NAME}_ma_mi_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_iso${ISO_FLAG}_smp${SMARTS_PREPASS}_mbfs
DSETS='nist20_hr'
PREC='[M+H]+ [M-H]- [M+Na]+ [M+K]+ [M+NH4]+ [M+Cl]-'
INST='FT QTOF IT'
FRAG_MODE='CID HCD'
DEST_DP=/scratch/feiw/frag-gnn-data/frags/

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
if [ -f "${SCRIPT_DIR}/frag_gen_common.sh" ]; then
    source "${SCRIPT_DIR}/frag_gen_common.sh"
else
    source "${FRAGNNET_DP}/scripts/manual_slurm/fir/frag_gen_common.sh"
fi
