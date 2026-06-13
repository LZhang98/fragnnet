#!/bin/bash

echo `date`: Job ${SLURM_JOB_ID} is allocated resources

module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
WORKING_DIR=${WORKING_DIR:-${SLURM_TMPDIR}/${SLURM_JOB_ID}}
DEST_DP=${DEST_DP:-/scratch/feiw/frag-gnn-data/frags/}
MAX_H_TRANSFER=${MAX_H_TRANSFER:-4}
MAX_TIME=${MAX_TIME:-150}
NB_FLAG=${NB_FLAG:-False}
ISO_FLAG=${ISO_FLAG:-False}
FRAG_MODE=${FRAG_MODE:-'CID HCD'}
INST=${INST:-'FT QTOF IT'}

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
 pandarallel==1.6.5 \
 lmdb==1.7.5 \
 h5py

cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${WORKING_DIR}/pyteomics-4.7.5-py3-none-any.whl
pip install pyteomics-4.7.5-py3-none-any.whl

rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/preproc_scripts ${WORKING_DIR}
rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/scripts ${WORKING_DIR}
rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" ${FRAGNNET_DP}/src ${WORKING_DIR}
cp ${FRAGNNET_DP}/setup.cfg ${WORKING_DIR}/
cp ${FRAGNNET_DP}/setup.py ${WORKING_DIR}/
cp ${FRAGNNET_DP}/pyproject.toml ${WORKING_DIR}/
pip install --no-deps -I ${WORKING_DIR}

echo DATASET_NAME: ${DATASET_NAME}
echo MAX_DEPTH: ${MAX_DEPTH}
echo SAVE_NAME: ${SAVE_NAME}
echo NB_FLAG: ${NB_FLAG}
echo DSETS: ${DSETS}
echo PREC: ${PREC}
echo INST: ${INST}
echo FRAG_MODE: ${FRAG_MODE}
echo ISO_FLAG: ${ISO_FLAG}
echo DEST_DP: ${DEST_DP}
if [ -n "${SMARTS_PREPASS+x}" ]; then
    echo SMARTS_PREPASS: ${SMARTS_PREPASS}
fi
if [ -n "${USE_HDF5+x}" ]; then
    echo USE_HDF5: ${USE_HDF5}
fi
if [ -n "${USE_LMDB+x}" ]; then
    echo USE_LMDB: ${USE_LMDB}
fi
if [ -n "${MIN_FRAG_ATOMS+x}" ]; then
    echo MIN_FRAG_ATOMS: ${MIN_FRAG_ATOMS}
fi

echo create local proc_dp: ${WORKING_DIR}/data/proc
mkdir -p ${WORKING_DIR}/data/proc
echo create local frag_dp: ${WORKING_DIR}/data/frag/
mkdir -p ${WORKING_DIR}/data/frag/

rsync -av ${FRAGNNET_DP}/data/proc/${DATASET_NAME} ${WORKING_DIR}/data/proc
echo list file in local proc dir: $(ls ${WORKING_DIR}/data/proc/${DATASET_NAME})

read -r -a PREC_ARGS <<< "${PREC}"
read -r -a INST_ARGS <<< "${INST}"
read -r -a FRAG_MODE_ARGS <<< "${FRAG_MODE}"
read -r -a DSET_ARGS <<< "${DSETS}"

PREPARE_ARGS=(
    --frag_dp=./data/frag/${SAVE_NAME}
    --proc_dp=./data/proc/${DATASET_NAME}
    --max_depth=${MAX_DEPTH}
    --wandb_mode=disabled
    --wandb_run_name=fool
    --prec_types "${PREC_ARGS[@]}"
    --inst_types "${INST_ARGS[@]}"
    --nb_isomorphic ${NB_FLAG}
    --isotopes ${ISO_FLAG}
    --frag_mode "${FRAG_MODE_ARGS[@]}"
    --max_h_transfer ${MAX_H_TRANSFER}
    --max_time ${MAX_TIME}
    --dsets "${DSET_ARGS[@]}"
)

if [ -n "${SMARTS_PREPASS+x}" ]; then
    PREPARE_ARGS+=(--smarts_prepass ${SMARTS_PREPASS})
fi
if [ -n "${USE_LMDB+x}" ]; then
    PREPARE_ARGS+=(--use_lmdb ${USE_LMDB})
fi
if [ -n "${USE_HDF5+x}" ]; then
    PREPARE_ARGS+=(--use_hdf5 ${USE_HDF5})
fi
if [ -n "${MULTI_CUT_BFS+x}" ]; then
    PREPARE_ARGS+=(--multi_cut_bfs ${MULTI_CUT_BFS})
fi
if [ -n "${MAX_CUT_SIZE+x}" ]; then
    PREPARE_ARGS+=(--max_cut_size ${MAX_CUT_SIZE})
fi
if [ -n "${MIN_FRAG_ATOMS+x}" ]; then
    PREPARE_ARGS+=(--min_frag_atoms ${MIN_FRAG_ATOMS})
fi
if [ -n "${MAX_RING_CUT_STEPS+x}" ]; then
    PREPARE_ARGS+=(--max_ring_cut_steps ${MAX_RING_CUT_STEPS})
fi

cd ${WORKING_DIR}
python preproc_scripts/03_prepare_dag_feats.py "${PREPARE_ARGS[@]}"

cd ${WORKING_DIR}/data/frag/
tar -cf ${SAVE_NAME}.tar ${SAVE_NAME}

echo create dir at ${DEST_DP}
mkdir -p ${DEST_DP}
echo copy tar: ${SAVE_NAME}.tar to ${DEST_DP}
cp ${SAVE_NAME}.tar ${DEST_DP}

echo `date`: Job ${SLURM_JOB_ID} finished running
