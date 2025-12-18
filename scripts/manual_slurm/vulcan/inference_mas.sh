#!/bin/bash
#SBATCH --account=aip-rgreiner
#SBATCH --nodes=1
#SBATCH --job-name=fraggnn_d3_nist20_jss_hun
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=1
#SBATCH --output=/project/aip-rgreiner/feiw/gitRepo/frag-gnn/jobs/inference_deepmet_%j.out
#SBATCH --error=/project/aip-rgreiner/feiw/gitRepo/frag-gnn/jobs/inference_deepmet_%j.err
#SBATCH --open-mode=append


echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

FRAGNNET_DP=/project/aip-rgreiner/feiw/gitRepo/frag-gnn
WORKING_DIR=${SLURM_TMPDIR}/${SLURM_JOB_ID}
# create work dir
mkdir -p ${WORKING_DIR}
# create local env
cd  ${WORKING_DIR}
virtualenv FRAG-GNN
source FRAG-GNN/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch==2.4.1 \
    torch_geometric==2.6.1 \
    torch_scatter \
    pandas \
    wandb \
    tqdm \
    scipy==1.15.2 \
    joblib==1.3.2 \
    matplotlib \
    pytorch_lightning==2.4.0 \
    dgl==2.4.0 \
    omegaconf \
    cython==3.0.12 \
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
# install with out dependencies, assume all dependencies are installed manually
pip install -I --no-deps ${WORKING_DIR}

# copy model ckpt and data to working dir
mkdir -p ${WORKING_DIR}/saved_ckpts/deepmet
cp ${FRAGNNET_DP}/saved_ckpts/deepmet/nist20_inchikey_fraggnn_d3_ma_mi_amolc_cv_all_hifszbst.ckpt ${WORKING_DIR}/saved_ckpts/deepmet/
# copy config
mkdir -p ${WORKING_DIR}/config
cp ${FRAGNNET_DP}/config/template.yml ${WORKING_DIR}/config
cp ${FRAGNNET_DP}/config/exp_fraggnn_ma_mi/d3_ma_mi_nist20_5cvs/fraggnn_d3_ma_mi_nist20_cv_all.yml ${WORKING_DIR}/config

# copy frag data
cp /scratch/feiw/frag-gnn-data/frags/mas_250905_filtered_d3_h4False_1.tar ${WORKING_DIR}/
mkdir -p ${WORKING_DIR}/frag_dp
tar -xf ${WORKING_DIR}/mas_250905_filtered_d3_h4False_1.tar -C ${WORKING_DIR}/frag_dp
# mkdir for saving outputs
mkdir -p /scratch/feiw/frag-gnn-data/inference_outputs/mas_250905_filtered

# run inference
python scripts/run_inference.py \
	--model_ckpt saved_ckpts/deepmet/nist20_inchikey_fraggnn_d3_ma_mi_amolc_cv_all_hifszbst.ckpt \
	--mol_fp /scratch/feiw/frag-gnn-data/inference/proc/mas_250905_filtered/proc/mol_df.pkl.gz \
	--spec_fp /scratch/feiw/frag-gnn-data/inference/proc/mas_250905_filtered/proc/spec_df.pkl.gz \
	--frag_dp ${WORKING_DIR}/frag_dp/mas_250905_filtered_d3_h4False_1 \
    --batch_size 1 \
	--save_dp /scratch/feiw/frag-gnn-data/inference_outputs/mas_250905_filtered \
    --save_format pkl.gz \
    --custom_fp config/fraggnn_d3_ma_mi_nist20_cv_all.yml
