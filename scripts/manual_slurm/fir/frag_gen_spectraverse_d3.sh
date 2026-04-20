#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_spectraverse_d3
#SBATCH --mem=180G
#SBATCH --cpus-per-task=48
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_spectraverse_d3_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_spectraverse_d3_%j.err
#SBATCH --open-mode=append

INSTALL_STORAGE_DEPS=False
DATASET_NAME=spectraverse
MAX_DEPTH=3
MAX_H_TRANSFER=4
MAX_TIME=-1
NB_FLAG=False
ISO_FLAG=False
SAVE_NAME=${DATASET_NAME}_ma_mi_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_iso${ISO_FLAG}
DSETS='spectraverse'
PREC='[M+H]+ [M-H]- [M+Na]+ [M+K]+ [M+NH4]+ [M+Cl]- [M+CH3COOH-H]-'
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
