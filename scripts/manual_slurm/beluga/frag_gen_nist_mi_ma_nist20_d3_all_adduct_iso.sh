#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_nist_mi_ma_nist20_d3_all_adduct_iso
#SBATCH --mem=180G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-4:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gen_nist_mi_ma_nist20_d3_all_adduct_iso_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gen_nist_mi_ma_nist20_d3_all_adduct_iso_%j.err
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
pip install --no-index torch==2.4.1
pip install --no-index torch_geometric==2.6.1 
pip install --no-index torch_scatter
pip install --no-index pandas
pip install --no-index wandb
pip install --no-index tqdm
pip install --no-index scipy
pip install --no-index joblib==1.3.2
pip install --no-index matplotlib
pip install --no-index pytorch_lightning==2.4.0
pip install --no-index dgl==2.4.0
pip install --no-index omegaconf
pip install --no-index cython==3.0.12

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

# setup jobs
DATASET_NAME=nist20mona23v3
MAX_DEPTH=3
MAX_H_TRANSFER=4
NB_FLAG=False
ISO_FALSE=True
SAVE_NAME=nist20mona23v3_ma_mi_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_iso${ISO_FALSE}
DSETS='nist20_hr'
PREC='any'
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
    --wandb_mode=off \
    --wandb_run_name=fool \
    --prec_types ${PREC}\
    --inst_types ${INST} \
    --nb_isomorphic ${NB_FLAG} \
    --isotopes ${ISO_FALSE} \
    --frag_mode ${FRAG_MODE} \
    --max_h_transfer ${MAX_H_TRANSFER}
        
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