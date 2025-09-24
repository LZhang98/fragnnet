#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=preproc_infer_deepmet
#SBATCH --mem=180G
#SBATCH --cpus-per-task=96
#SBATCH --time=0-2:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_deepmet_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_deepmet_%j.err
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
DATASET_NAME=mas_250905
SAVE_NAME=${DATASET_NAME}
DEST_DP=/scratch/feiw/frag-gnn-data/inference/proc
INPUT_FP=${FRAGNNET_DP}/data/other/mas_250905/db_deepmet_hmdb_50_preprocessed.csv.gz
INPUT_BASENAME=$(basename ${INPUT_FP})
echo SAVE_NAME: ${SAVE_NAME}
echo DEST_DP: ${DEST_DP}
echo INPUT_BASENAME: ${INPUT_BASENAME}

# copy proc file
# rsync -av ${INPUT_FP} ${WORKING_DIR}/data/proc 
# echo list file in local proc dir: $(ls ${WORKING_DIR}/data/proc/${DATASET_NAME})
# mkdir -p ${WORKING_DIR}/${DATASET_NAME}/proc
cp ${INPUT_FP} ${WORKING_DIR}/${INPUT_BASENAME}
echo list file in local proc dir: $(ls ${WORKING_DIR}/${INPUT_BASENAME})

cd ${WORKING_DIR}
python preproc_scripts/inference/01_prepare_inference_input.py \
	--smiles_file ${INPUT_BASENAME}\
    --smiles_colname smiles \
    --ft_ces 10eV \
    --inst_types FT \
    --prec_types [M-H]- \
    --output_dir ${SAVE_NAME}/proc

# go to frag directory
tar -cf ${SAVE_NAME}.tar ${SAVE_NAME}

# copy tar to scratch
echo create dir at ${DEST_DP}
mkdir -p ${DEST_DP}
echo copy tar: ${SAVE_NAME}.tar to ${DEST_DP}
cp ${SAVE_NAME}.tar ${DEST_DP}

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running