#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_mimedb_d3
#SBATCH --mem=384G
#SBATCH --cpus-per-task=192
#SBATCH --time=0-12:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_mimedb_d3_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_mimedb_d3_%j.err
#SBATCH --open-mode=append
#SBATCH --array=1-3

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

FRAGNNET_DP=/home/feiw/project/feiw/frag-gnn
WORKING_DIR=${SLURM_TMPDIR}/${SLURM_JOB_ID}
# create work dir
mkdir -p ${WORKING_DIR}
# create local env
cd  ${WORKING_DIR}
virtualenv FRAG-GNN
source FRAG-GNN/bin/activate

# intall from whl
cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${WORKING_DIR}/pyteomics-4.7.5-py3-none-any.whl
pip install --no-deps pyteomics-4.7.5-py3-none-any.whl

# copy script code
rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/preproc_scripts ${WORKING_DIR}
rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/scripts ${WORKING_DIR}
# copy src code - this is needed to avoid issues with multiple install read the same src code
rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" ${FRAGNNET_DP}/src ${WORKING_DIR}
cp ${FRAGNNET_DP}/setup.cfg ${WORKING_DIR}/
cp ${FRAGNNET_DP}/setup.py ${WORKING_DIR}/
cp ${FRAGNNET_DP}/pyproject.toml ${WORKING_DIR}/

pip install --no-deps -I ${WORKING_DIR}

# Copy files
# create local dir
MAX_DEPTH=3
MAX_H_TRANSFER=4
NB_FLAG=False
ISO_FALSE=False
SAVE_NAME=mimedb_d${MAX_DEPTH}_h${MAX_H_TRANSFER}${ISO_FALSE}_${SLURM_ARRAY_TASK_ID}
DEST_DP=/scratch/feiw/frag-gnn-data/frags/
PROC_DP=${FRAGNNET_DP}/data/other/mimedb/proc

echo MAX_DEPTH: ${MAX_DEPTH}
echo SAVE_NAME: ${SAVE_NAME}
echo NB_FLAG: ${NB_FLAG}
echo ISO_FALSE: ${ISO_FALSE}
echo MAX_H_TRANSFER: ${MAX_H_TRANSFER}
echo PROC_DP: ${PROC_DP}
echo DEST_DP: ${DEST_DP}

echo create local proc_dp: ${WORKING_DIR}/data/proc
mkdir -p ${WORKING_DIR}/data/proc

echo create local frag_dp: ${WORKING_DIR}/data/frag/
mkdir -p ${WORKING_DIR}/data/frag/


# copy proc file
cp -R ${PROC_DP}/* ${WORKING_DIR}/data/proc/
echo list file in local proc dir: $(ls ${WORKING_DIR}/data/proc/)

# set chunk size
CHUNK_SIZE=8000

cd ${WORKING_DIR}
python preproc_scripts/inference/02_run_ms2c_frag_gen.py \
    --max_h_transfer ${MAX_H_TRANSFER} \
    --max_depth=${MAX_DEPTH} \
    --proc_dp=./data/proc  \
    --frag_dp=./data/frag/${SAVE_NAME} \
    --start_row=$(( (SLURM_ARRAY_TASK_ID - 1) * ${CHUNK_SIZE} )) \
    --end_row=$(( SLURM_ARRAY_TASK_ID * ${CHUNK_SIZE} ))

# go to frag directory
cd ${WORKING_DIR}/data/frag/
tar -cf ${SAVE_NAME}.tar ${SAVE_NAME}

# copy tar to scratch
echo create dir at ${DEST_DP}
mkdir -p ${DEST_DP}
echo copy tar: ${SAVE_NAME}.tar to ${DEST_DP}
cp ${SAVE_NAME}.tar ${DEST_DP}

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running