#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_nist_mi_ma_nist20_d4
#SBATCH --mem=180G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-8:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gen_nist_mi_ma_nist20_d4_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gen_nist_mi_ma_nist20_d4_%j.err
#SBATCH --open-mode=append

FRAGNNET_DP=/home/feiw/feiw/frag-gnn

# load modules
module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

# go to slurm and create virtual env
cd  ${SLURM_TMPDIR}

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
cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${SLURM_TMPDIR}/pyteomics-4.7.5-py3-none-any.whl 
pip install pyteomics-4.7.5-py3-none-any.whl 

# go to source location
cd ${FRAGNNET_DP}
# instal fragnnet
pip install -I .

# copy script source code code
rsync -av --exclude=".*" --exclude="__pycache__" preproc_scripts ${SLURM_TMPDIR}
rsync -av --exclude=".*" --exclude="__pycache__" scripts ${SLURM_TMPDIR}

# Copy files
# create local dir
DATASET_NAME=nist20
MAX_DEPTH=4
MAX_H_TRANSFER=4
NB_FLAG=False
ISO_FALSE=False
SAVE_NAME=nist20_ma_mi_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_iso${ISO_FALSE}
DSETS='nist20_hr'
PREC='[M+H]+ [M-H]- [M+Na]+ [M+Cl]-'
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

echo create local proc_dp: ${SLURM_TMPDIR}/data/proc
mkdir -p ${SLURM_TMPDIR}/data/proc
echo create local frag_dp: ${SLURM_TMPDIR}/data/frag/
mkdir -p ${SLURM_TMPDIR}/data/frag/
#echo create local split_dp: ${SLURM_TMPDIR}/data/split/
#mkdir -p ${SLURM_TMPDIR}/data/split/
# create local config dir
#echo create local config_dp: ${SLURM_TMPDIR}/config/
#mkdir -p ${SLURM_TMPDIR}/config/

# copy proc file
rsync -av ${FRAGNNET_DP}/data/proc/${DATASET_NAME} ${SLURM_TMPDIR}/data/proc 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/data/proc/${DATASET_NAME})

cd ${SLURM_TMPDIR}
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
cd ${SLURM_TMPDIR}/data/frag/
tar -cf ${SAVE_NAME}.tar ${SAVE_NAME}

# copy tar to scratch
echo create dir at ${DEST_DP}
mkdir -p ${DEST_DP}
echo copy tar: ${SAVE_NAME}.tar to ${DEST_DP}
cp ${SAVE_NAME}.tar ${DEST_DP}

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running