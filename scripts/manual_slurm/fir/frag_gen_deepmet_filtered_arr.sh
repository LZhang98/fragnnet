#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_deepmet_filtered_arr
#SBATCH --mem=384G
#SBATCH --cpus-per-task=192
#SBATCH --time=0-24:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_deepmet_filtered_arr_%A_%a.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_deepmet_filtered_arr_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --array=1-8
#SBATCH --signal=B:USR1@120 #This will send signal 120s before wall time to this bash

# record start
echo `date`: Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} is allocated resources
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
pip install --no-index --upgrade pip
pip install --no-index  \
 torch==2.4.1 \
 torch_geometric==2.6.1  \
 torch_scatter  \
 pandas  \
 wandb  \
 tqdm  \
 scipy  \
 joblib==1.3.2  \
 matplotlib  \
 pytorch_lightning==2.4.0  \
 dgl==2.4.0  \
 omegaconf  \
 cython==3.0.12

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
SAVE_NAME=mas_250905_filtered_d${MAX_DEPTH}_h${MAX_H_TRANSFER}${ISO_FALSE}_${SLURM_ARRAY_TASK_ID}
DEST_DP=/scratch/feiw/frag-gnn-data/frags/
PROC_DP=${FRAGNNET_DP}/data/other/mas_250905_filtered/proc/

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
CHUNK_SIZE=40000

# Validate SLURM_ARRAY_TASK_ID and set default if needed
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "Warning: SLURM_ARRAY_TASK_ID not set, using default value 1"
    SLURM_ARRAY_TASK_ID=1
fi

#Setup Exit, this will collect results to another location
function OnExit {
# copy tar to scratch
    echo create dir at ${DEST_DP}
    mkdir -p ${DEST_DP}
    echo copy tar: ${SAVE_NAME}.tar to ${DEST_DP}
    cp ${SAVE_NAME}.tar ${DEST_DP}
}
trap OnExit EXIT

#Gracefully exit predictions
function EndTask {
   echo "USR1 or TERM received, send TERM to $PID"
   kill $PID 2>/dev/null
   wait $PID
}
trap EndTask USR1
trap EndTask TERM
trap EndTask SIGINT

echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Chunk Size: ${CHUNK_SIZE}"

# Calculate start and end rows
START_ROW=$(( (SLURM_ARRAY_TASK_ID - 1) * CHUNK_SIZE ))
END_ROW=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))

echo "Processing rows ${START_ROW} to ${END_ROW}"

cd ${WORKING_DIR}
python preproc_scripts/inference/02_run_inference_frag_gen.py \
    --max_h_transfer ${MAX_H_TRANSFER} \
    --max_depth=${MAX_DEPTH} \
    --proc_dp=./data/proc  \
    --frag_dp=./data/frag/${SAVE_NAME} \
    --start_row=${START_ROW} \
    --end_row=${END_ROW} \
    --max_time=-1 &
PID=$!
wait $PID

# go to frag directory
#cd ${WORKING_DIR}/data/frag/
#tar -cf ${SAVE_NAME}.tar ${SAVE_NAME}

# record ending
echo `date`: Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} finished running