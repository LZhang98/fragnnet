#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_nist23_d3
#SBATCH --mem=180G
#SBATCH --cpus-per-task=48
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_nist23_d3_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/frag_gen_nist23_d3_%j.err
#SBATCH --open-mode=append

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
 cython==3.0.12 \
 pandarallel==1.6.5 \
 lmdb==1.7.5 \
 h5py

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
cp ${FRAGNNET_DP}/pyproject.toml ${WORKING_DIR}/
# install without deps to avoid conflict
pip install --no-deps -I ${WORKING_DIR}

# Copy files
# create local dir
DATASET_NAME=nist23
MAX_DEPTH=3
MAX_H_TRANSFER=4
NB_FLAG=False
ISO_FALSE=False
SAVE_NAME=${DATASET_NAME}_ma_mi_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_iso${ISO_FALSE}
DSETS='nist23_hr1 nist23_hr2'
PREC='[M+H]+ [M-H]- [M+Na]+ [M+K]+ [M+NH4]+ [M+Cl]-'
INST='FT QTOF IT'
FRAG_MODE='CID HCD'
DEST_DP=/scratch/feiw/frag-gnn-data/frags/

echo DATASET_NAME: ${DATASET_NAME}
echo MAX_DEPTH: ${MAX_DEPTH}
echo SAVE_NAME: ${SAVE_NAME}
echo NB_FLAG: ${NB_FLAG}
echo DSETS: ${DSETS}
echo PREC: ${PREC}
echo INST: ${INST}
echo FRAG_MODE: ${FRAG_MODE}
echo ISO_FALSE: ${ISO_FALSE}
echo DEST_DP: ${DEST_DP}

echo create local proc_dp: ${WORKING_DIR}/data/proc
mkdir -p ${WORKING_DIR}/data/proc
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
python preproc_scripts/03_prepare_dag_feats.py \
    --frag_dp=./data/frag/${SAVE_NAME} \
    --proc_dp=./data/proc/${DATASET_NAME}  \
    --max_depth=${MAX_DEPTH} \
    --wandb_mode=disabled \
    --wandb_run_name=fool \
    --prec_types ${PREC}\
    --inst_types ${INST} \
    --nb_isomorphic ${NB_FLAG} \
    --isotopes ${ISO_FALSE} \
    --frag_mode ${FRAG_MODE} \
    --max_h_transfer ${MAX_H_TRANSFER} \
    --dsets ${DSETS}

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