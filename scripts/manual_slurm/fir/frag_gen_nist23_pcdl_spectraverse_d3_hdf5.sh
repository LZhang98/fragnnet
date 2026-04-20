#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_nist23_pcdl_spectraverse_d3_hdf5
#SBATCH --mem=256G
#SBATCH --cpus-per-task=128
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_nist23_pcdl_spectraverse_d3_hdf5_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_nist23_pcdl_spectraverse_d3_hdf5_%j.err
#SBATCH --open-mode=append

DATASET_NAME=nist23_pcdl_spectraverse
MAX_DEPTH=3
MAX_H_TRANSFER=4
MAX_TIME=-1
NB_FLAG=False
ISO_FLAG=False
SMARTS_PREPASS=False
USE_LMDB=False
USE_HDF5=True
SAVE_NAME=${DATASET_NAME}_ma_mi_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_iso${ISO_FLAG}_smp${SMARTS_PREPASS}_hdf5
DSETS='nist23_hr1 nist23_hr2 mona23 pcdl spectraverse'
PREC='[M+H]+ [M-H]- [M+Na]+ [M+K]+ [M+NH4]+ [M+Cl]- [M+]'
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
