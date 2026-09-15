#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=preproc_infer_cspirit
#SBATCH --mem=180G
#SBATCH --cpus-per-task=96
#SBATCH --time=0-24:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_cspirit_%j.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/preproc_infer_cspirit_%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

# load modules
module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

FRAGNNET_DP=/home/feiw/project/feiw/frag-gnn
WORKING_DIR=${SLURM_TMPDIR}/${SLURM_JOB_ID}

DATASET_NAME=cspirit
MAX_DEPTH=3
MAX_H_TRANSFER=4
MAX_TIME=150
FRAG_SAVE_NAME=${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}

DEST_PROC_DP=/scratch/feiw/frag-gnn-data/inference/proc
DEST_FRAG_DP=/scratch/feiw/frag-gnn-data/inference/frags

INPUT_FP=${FRAGNNET_DP}/data/other/cspirit/cspirit.csv
INPUT_BASENAME=$(basename ${INPUT_FP})

echo DATASET_NAME: ${DATASET_NAME}
echo FRAG_SAVE_NAME: ${FRAG_SAVE_NAME}
echo DEST_PROC_DP: ${DEST_PROC_DP}
echo DEST_FRAG_DP: ${DEST_FRAG_DP}
echo INPUT_BASENAME: ${INPUT_BASENAME}

# create work dir and virtualenv
mkdir -p ${WORKING_DIR}
cd ${WORKING_DIR}
virtualenv FRAG-GNN
source FRAG-GNN/bin/activate
pip install --no-index --upgrade pip
pip install --no-index \
    torch==2.4.1 \
    torch_geometric==2.6.1 \
    torch_scatter \
    pandas \
    wandb \
    tqdm \
    scipy \
    joblib==1.3.2 \
    matplotlib \
    pytorch_lightning==2.4.0 \
    dgl==2.4.0 \
    omegaconf \
    cython==3.0.12 \
    pyopenms==3.3.0 \
    
# install from whl
cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${WORKING_DIR}/pyteomics-4.7.5-py3-none-any.whl
pip install pyteomics-4.7.5-py3-none-any.whl

# copy scripts and source
rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/preproc_scripts ${WORKING_DIR}
rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/scripts ${WORKING_DIR}
rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" ${FRAGNNET_DP}/src ${WORKING_DIR}
cp ${FRAGNNET_DP}/setup.cfg ${WORKING_DIR}/
cp ${FRAGNNET_DP}/setup.py ${WORKING_DIR}/
cp ${FRAGNNET_DP}/pyproject.toml ${WORKING_DIR}/
pip install --no-deps -I ${WORKING_DIR}

# copy input file
cp ${INPUT_FP} ${WORKING_DIR}/${INPUT_BASENAME}
echo input file: $(ls ${WORKING_DIR}/${INPUT_BASENAME})

cd ${WORKING_DIR}

# === Step 01: mol_df ===
echo `date`: Step 01 - preparing mol_df
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
    --smiles_file ${INPUT_BASENAME} \
    --smiles_colname smiles \
    --id_colname compound_id \
    --output_dir ${DATASET_NAME}/proc

# === Step 02: spec_df ===
echo `date`: Step 02 - preparing spec_df
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
    --mol_df_path ${DATASET_NAME}/proc/mol_df.pkl.gz \
    --prec_types "[M+H]+" "[M-H]-" \
    --ft_ces 20.0NCE 40.0NCE 60.0NCE \
    --qtof_ces 10.0eV 20.0eV 40.0eV \
    --inst_types FT QTOF \
    --output_dir ${DATASET_NAME}/proc

# save proc to scratch
echo `date`: saving proc to ${DEST_PROC_DP}
mkdir -p ${DEST_PROC_DP}
tar -cf ${DATASET_NAME}_proc.tar ${DATASET_NAME}/proc
cp ${DATASET_NAME}_proc.tar ${DEST_PROC_DP}/

# === Step 03: fragment generation ===
echo `date`: Step 03 - fragment generation
mkdir -p frag/${FRAG_SAVE_NAME}
python preproc_scripts/inference/03_inf_run_inference_frag_gen.py \
    --proc_dp ${DATASET_NAME}/proc \
    --frag_dp frag/${FRAG_SAVE_NAME} \
    --max_depth ${MAX_DEPTH} \
    --max_h_transfer ${MAX_H_TRANSFER} \
    --max_time ${MAX_TIME}

# save frags to scratch
echo `date`: saving frags to ${DEST_FRAG_DP}
mkdir -p ${DEST_FRAG_DP}
cd frag/
tar -cf ${FRAG_SAVE_NAME}.tar ${FRAG_SAVE_NAME}
cp ${FRAG_SAVE_NAME}.tar ${DEST_FRAG_DP}/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
