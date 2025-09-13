#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_mimedb_d3
#SBATCH --mem=180G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-4:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gen_mimedb_d3_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gen_mimedb_d3_%j.err
#SBATCH --open-mode=append


# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

FRAGNNET_DP=/home/feiw/feiw/frag-gnn
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
pip install pyteomics-4.7.5-py3-none-any.whl

# copy script code
rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/preproc_scripts ${WORKING_DIR}
rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/scripts ${WORKING_DIR}
# copy src code - this is needed to avoid issues with multiple install read the same src code
rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" ${FRAGNNET_DP}/src ${WORKING_DIR}
cp ${FRAGNNET_DP}/setup.cfg ${WORKING_DIR}/
cp ${FRAGNNET_DP}/setup.py ${WORKING_DIR}/
pip install -I ${WORKING_DIR}

# Copy files
# create local dir
MAX_DEPTH=3
MAX_H_TRANSFER=4
NB_FLAG=False
ISO_FALSE=False
SAVE_NAME=mimedb_d${MAX_DEPTH}_h${MAX_H_TRANSFER}${ISO_FALSE}
DEST_DP=/scratch/feiw/frag-gnn-data/frags/
PROC_DP=./data/others/mimedb/proc

echo DATASET_NAME: ${DATASET_NAME}
echo MAX_DEPTH: ${MAX_DEPTH}
echo SAVE_NAME: ${SAVE_NAME}
echo NB_FLAG: ${NB_FLAG}
echo ISO_FALSE: ${ISO_FALSE}
echo MAX_H_TRANSFER: ${MAX_H_TRANSFER}
echo PROC_DP: ${PROC_DP}
echo DEST_DP: ${DEST_DP}

echo create local proc_dp: ${WORKING_DIR}/data/proc
mkdir -p ${WORKING_DIR}/data/proc
cp -R ${PROC_DP}/* ${WORKING_DIR}/data/proc/
echo create local frag_dp: ${WORKING_DIR}/data/frag/
mkdir -p ${WORKING_DIR}/data/frag/
#echo create local split_dp: ${WORKING_DIR}/data/split/
#mkdir -p ${WORKING_DIR}/data/split/
# create local config dir
#echo create local config_dp: ${WORKING_DIR}/config/
#mkdir -p ${WORKING_DIR}/config/

# copy proc file
rsync -av ${FRAGNNET_DP}/data/proc/${DATASET_NAME} ${WORKING_DIR}/data/proc 
echo list file in local proc dir: $(ls ${WORKING_DIR}/data/proc/${DATASET_NAME})

cd ${WORKING_DIR}
python preproc_scripts/inference/02_run_ms2c_frag_gen.py \
    --max_h_transfer ${MAX_H_TRANSFER} \
 	--max_depth=${MAX_DEPTH} \
    --frag_dp=./data/proc  \
    --proc_dp=./data/frag/${SAVE_NAME} \
         
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