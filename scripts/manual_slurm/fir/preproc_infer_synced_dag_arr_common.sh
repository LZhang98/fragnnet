#!/bin/bash
# Shared body for the preproc_infer_synced_dag_arr_*.sh array variants.
#
# Runs STEP 03 ONLY (DAG feature generation) of the SYNCED inference preprocessing
# pipeline. Steps 01/02 are assumed to have already been run: this reads the existing
# mol_df / spec_df out of ${SRC_PROC_DP} (data/inference/synced) instead of rebuilding
# them from the raw SMILES csv.
#
# Each array task takes a contiguous slice of mol_df rows
#   [SLURM_ARRAY_TASK_ID*CHUNK_SIZE, (SLURM_ARRAY_TASK_ID+1)*CHUNK_SIZE)
# writes a task-local proc dir, runs 03_prepare_dag_feats.py on it, and ships that
# shard's output to ${DEST_FRAG_DP}/${SAVE_NAME}/.
#
# Output layout depends on USE_HDF5:
#   False -> one .tar per shard holding dags/{mol_id}.pkl.bz2. The shards are disjoint,
#            so untarring them on top of each other reconstitutes the full frag dir.
#   True  -> one shard_<task>/dags.h5 + meta_info.json per shard. The dataset loader
#            wants a SINGLE dags.h5, so the shards must be merged afterwards with
#            preproc_scripts/inference/merge_dag_h5_shards.py.
#
# The caller must set: DATASET_NAME, SAVE_NAME, USE_HDF5, and an #SBATCH --array range
# consistent with CHUNK_SIZE. Everything else has a default below.

echo "$(date): Job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}) is allocated resources"

module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
WORKING_DIR=${WORKING_DIR:-${SLURM_TMPDIR}/${SLURM_JOB_ID}}

DATASET_NAME=${DATASET_NAME:-synced}
MAX_DEPTH=${MAX_DEPTH:-3}
MAX_H_TRANSFER=${MAX_H_TRANSFER:-4}
MAX_TIME=${MAX_TIME:-150}
USE_HDF5=${USE_HDF5:-False}
SKIP_STATS=${SKIP_STATS:-True}
SAVE_NAME=${SAVE_NAME:-${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}}

CHUNK_SIZE=${CHUNK_SIZE:-20000}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_IDX=$((TASK_ID * CHUNK_SIZE))

# --- filters ---------------------------------------------------------------
# Must match what 02_inf_prepare_spec_df.py wrote into spec_df (dset=inference,
# frag_mode=CID, spec_type=MS2, ion_mode derived from the prec_type charge).
# Do NOT copy the training-side filter lists here: none of those values exist in
# an inference spec_df and the filter would select zero rows.
PREC=${PREC:-'[M+H]+ [M-H]-'}
INST=${INST:-'FT QTOF'}
FRAG_MODE=${FRAG_MODE:-'CID'}
ION_MODES=${ION_MODES:-'P N'}
DSETS=${DSETS:-'inference'}
SPEC_TYPE=${SPEC_TYPE:-MS2}

# --- io --------------------------------------------------------------------
# Step 01/02 output dir. Falls back to the synced_smiles/ spelling when the
# synced/ dir is absent.
SRC_PROC_DP=${SRC_PROC_DP:-${FRAGNNET_DP}/data/inference/synced}
if [ ! -d "${SRC_PROC_DP}" ] && [ -d "${FRAGNNET_DP}/data/inference/synced_smiles" ]; then
    SRC_PROC_DP=${FRAGNNET_DP}/data/inference/synced_smiles
fi
DEST_FRAG_DP=${DEST_FRAG_DP:-/scratch/feiw/frag-gnn-data/inference/frags}
# Optional prebuilt venv; when unset/absent a fresh one is built in ${WORKING_DIR}.
VENV_DP=${VENV_DP:-}

echo DATASET_NAME: ${DATASET_NAME}
echo SAVE_NAME: ${SAVE_NAME}
echo MAX_DEPTH: ${MAX_DEPTH}
echo MAX_H_TRANSFER: ${MAX_H_TRANSFER}
echo MAX_TIME: ${MAX_TIME}
echo USE_HDF5: ${USE_HDF5}
echo SKIP_STATS: ${SKIP_STATS}
echo PREC: ${PREC}
echo INST: ${INST}
echo FRAG_MODE: ${FRAG_MODE}
echo ION_MODES: ${ION_MODES}
echo DSETS: ${DSETS}
echo SRC_PROC_DP: ${SRC_PROC_DP}
echo DEST_FRAG_DP: ${DEST_FRAG_DP}
echo TASK_ID: ${TASK_ID}
echo CHUNK_SIZE: ${CHUNK_SIZE}
echo START_IDX: ${START_IDX}

if [ ! -d "${SRC_PROC_DP}" ]; then
    echo "ERROR: SRC_PROC_DP does not exist: ${SRC_PROC_DP}" >&2
    exit 1
fi

# --- env -------------------------------------------------------------------
mkdir -p ${WORKING_DIR}
cd ${WORKING_DIR}

if [ -n "${VENV_DP}" ] && [ -f "${VENV_DP}/bin/activate" ]; then
    echo `date`: using prebuilt venv ${VENV_DP}
    source ${VENV_DP}/bin/activate
    cd ${FRAGNNET_DP}
    WORKING_DIR=${FRAGNNET_DP}
else
    echo `date`: building venv in ${WORKING_DIR}
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
        omegaconf \
        cython==3.0.12 \
        pyopenms==3.3.0 \
        pandarallel==1.6.5 \
        h5py

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
    cd ${WORKING_DIR}
fi

# === shard the proc dfs ====================================================
# 03_prepare_dag_feats.py has no start/stride argument, so slice mol_df here and
# keep only the spec rows belonging to the selected molecules. It reads plain
# mol_df.pkl / spec_df.pkl, while 01/02 write .pkl.gz — pandas reads either, and
# the shard is written uncompressed.
SHARD_PROC_DP=${SLURM_TMPDIR:-/tmp}/proc_shard_${TASK_ID}
mkdir -p ${SHARD_PROC_DP}

echo `date`: sharding mol_df/spec_df from ${SRC_PROC_DP} into ${SHARD_PROC_DP}
python - <<PYEOF
import os
import sys

import pandas as pd

src_dp = "${SRC_PROC_DP}"
dst_dp = "${SHARD_PROC_DP}"
start = ${START_IDX}
chunk = ${CHUNK_SIZE}


def read_df(name):
    for suffix in (".pkl", ".pkl.gz"):
        fp = os.path.join(src_dp, name + suffix)
        if os.path.exists(fp):
            print(f"read {fp}")
            return pd.read_pickle(fp)
    raise FileNotFoundError(f"{name}.pkl[.gz] not found in {src_dp}")


mol_df = read_df("mol_df")
spec_df = read_df("spec_df")
print(f"full: {len(mol_df)} molecules, {len(spec_df)} spectra")

if start >= len(mol_df):
    print(f"start {start} beyond mol_df ({len(mol_df)}), nothing to do")
    sys.exit(101)

mol_shard = mol_df.iloc[start : start + chunk]
spec_shard = spec_df[spec_df["mol_id"].isin(set(mol_shard["mol_id"]))]
print(f"shard [{start}, {start + len(mol_shard)}): {len(mol_shard)} molecules, {len(spec_shard)} spectra")

mol_shard.to_pickle(os.path.join(dst_dp, "mol_df.pkl"))
spec_shard.to_pickle(os.path.join(dst_dp, "spec_df.pkl"))
PYEOF

RC=$?
if [ ${RC} -eq 101 ]; then
    echo `date`: task ${TASK_ID} has no molecules assigned, exiting cleanly
    exit 0
elif [ ${RC} -ne 0 ]; then
    echo "ERROR: sharding failed with code ${RC}" >&2
    exit ${RC}
fi
ls -l ${SHARD_PROC_DP}

# === Step 03: DAG feature generation =======================================
echo `date`: Step 03 - DAG feature generation for shard ${TASK_ID}
read -r -a PREC_ARGS <<< "${PREC}"
read -r -a INST_ARGS <<< "${INST}"
read -r -a FRAG_MODE_ARGS <<< "${FRAG_MODE}"
read -r -a ION_MODE_ARGS <<< "${ION_MODES}"
read -r -a DSET_ARGS <<< "${DSETS}"

FRAG_OUT_DP=${SLURM_TMPDIR:-/tmp}/frag_${TASK_ID}/${SAVE_NAME}
mkdir -p ${FRAG_OUT_DP}

python ${WORKING_DIR}/preproc_scripts/03_prepare_dag_feats.py \
    --max_depth ${MAX_DEPTH} \
    --max_h_transfer ${MAX_H_TRANSFER} \
    --max_time ${MAX_TIME} \
    --frag_dp ${FRAG_OUT_DP} \
    --proc_dp ${SHARD_PROC_DP} \
    --prec_types "${PREC_ARGS[@]}" \
    --inst_types "${INST_ARGS[@]}" \
    --frag_modes "${FRAG_MODE_ARGS[@]}" \
    --ion_modes "${ION_MODE_ARGS[@]}" \
    --dsets "${DSET_ARGS[@]}" \
    --spec_type "${SPEC_TYPE}" \
    --wandb_mode disabled \
    --wandb_run_name fool \
    --use_hdf5 ${USE_HDF5} \
    --skip_stats ${SKIP_STATS}

# === save this shard's frags to scratch ====================================
echo `date`: saving shard frags to ${DEST_FRAG_DP}/${SAVE_NAME}
mkdir -p ${DEST_FRAG_DP}/${SAVE_NAME}
cd $(dirname ${FRAG_OUT_DP})
ls -l ${SAVE_NAME} | head

if [ "${USE_HDF5}" = "True" ] || [ "${USE_HDF5}" = "true" ]; then
    # HDF5 layout: a single dags.h5 plus meta_info.json — one per shard, copy directly.
    mkdir -p ${DEST_FRAG_DP}/${SAVE_NAME}/shard_${TASK_ID}
    cp ${SAVE_NAME}/dags.h5 ${DEST_FRAG_DP}/${SAVE_NAME}/shard_${TASK_ID}/
    cp ${SAVE_NAME}/meta_info.json ${DEST_FRAG_DP}/${SAVE_NAME}/shard_${TASK_ID}/
    echo copied dags.h5 to ${DEST_FRAG_DP}/${SAVE_NAME}/shard_${TASK_ID}/
else
    # Pickle layout: one compressed artifact per molecule under dags/ — tar the shard.
    tar -cf ${SAVE_NAME}_${TASK_ID}.tar ${SAVE_NAME}
    cp ${SAVE_NAME}_${TASK_ID}.tar ${DEST_FRAG_DP}/${SAVE_NAME}/
    echo copied tar: ${SAVE_NAME}_${TASK_ID}.tar to ${DEST_FRAG_DP}/${SAVE_NAME}
fi

echo `date`: Job ${SLURM_JOB_ID} task ${TASK_ID} finished running
