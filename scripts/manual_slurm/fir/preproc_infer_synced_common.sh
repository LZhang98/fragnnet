#!/bin/bash
# Shared body for the preproc_infer_synced_*.sh variants.
#
# Runs the full inference preprocessing pipeline for the SYNCED molecule set:
#   01_inf_prepare_mol_df.py  -> mol_df.pkl.gz
#   02_inf_prepare_spec_df.py -> spec_df.pkl.gz  (dummy spectra, empty peaks)
#   03_prepare_dag_feats.py   -> DAGs (dags.h5 or dags/*.pkl.bz2)
#
# The caller must set: DATASET_NAME, SAVE_NAME, USE_HDF5.
# Everything else has a default below and can be overridden by the caller.

echo `date`: Job ${SLURM_JOB_ID} is allocated resources

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

# --- filters ---------------------------------------------------------------
# These MUST match what 02_inf_prepare_spec_df.py actually writes into spec_df:
#   dset      -> always "inference"   (hardcoded)
#   ion_mode  -> derived from the prec_type charge: "P" for +, "N" for -
#   frag_mode -> always "CID"         (hardcoded)
#   spec_type -> always "MS2"         (hardcoded)
#   prec_type / inst_type -> only what PREC / INST below generate
# Do NOT copy the training-side filter lists (nist23_hr1/mona23/..., P N, [M+Na]+ ...)
# here: none of those values exist in an inference spec_df and the filter would
# select zero rows. ION_MODES must list both P and N whenever PREC mixes +/- adducts.
PREC=${PREC:-'[M+H]+ [M-H]-'}
INST=${INST:-'FT QTOF'}
FRAG_MODE=${FRAG_MODE:-'CID'}
ION_MODES=${ION_MODES:-'P N'}
DSETS=${DSETS:-'inference'}
SPEC_TYPE=${SPEC_TYPE:-MS2}

FT_CES=${FT_CES:-'10.0NCE 20.0NCE 30.0NCE 40.0NCE'}
QTOF_CES=${QTOF_CES:-'10.0eV 20.0eV 30.0eV'}

# --- io --------------------------------------------------------------------
INPUT_FP=${INPUT_FP:-${FRAGNNET_DP}/data/raw/synced/synced_smiles_inchi_filled.csv}
SMILES_COLNAME=${SMILES_COLNAME:-moldb_smiles}
ID_COLNAME=${ID_COLNAME:-id}
# Step 03 (DAG generation) runs here by default, doing 01+02+03 in one job. Set False to
# stop after the proc dir: that is the mode to use when the DAGs are built by the
# preproc_infer_synced_dag_arr_*.sh array instead, which needs 01/02 output to already
# exist. The array must then be pointed at this run's proc with
# SRC_PROC_DP=<untarred ${DEST_PROC_DP}/${DATASET_NAME}_proc.tar>/${DATASET_NAME}/proc,
# and run_inference.py must read the SAME proc - both slice mol_df by row index, so a
# regenerated mol_df with different row order silently mismatches shard i against slice i.
RUN_STEP_03=${RUN_STEP_03:-True}
DEST_PROC_DP=${DEST_PROC_DP:-/scratch/feiw/frag-gnn-data/inference/proc}
DEST_FRAG_DP=${DEST_FRAG_DP:-/scratch/feiw/frag-gnn-data/inference/frags}
INPUT_BASENAME=$(basename ${INPUT_FP})

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
echo INPUT_FP: ${INPUT_FP}
echo DEST_PROC_DP: ${DEST_PROC_DP}
echo DEST_FRAG_DP: ${DEST_FRAG_DP}

# --- env -------------------------------------------------------------------
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

# copy input file
cp ${INPUT_FP} ${WORKING_DIR}/${INPUT_BASENAME}
echo input file: $(ls -l ${WORKING_DIR}/${INPUT_BASENAME})

cd ${WORKING_DIR}

PROC_DP=${DATASET_NAME}/proc

# === Step 01: mol_df =======================================================
echo `date`: Step 01 - preparing mol_df
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
    --smiles_file ${INPUT_BASENAME} \
    --smiles_colname ${SMILES_COLNAME} \
    --id_colname ${ID_COLNAME} \
    --output_dir ${PROC_DP}

# === Step 02: spec_df ======================================================
echo `date`: Step 02 - preparing spec_df
read -r -a PREC_ARGS <<< "${PREC}"
read -r -a INST_ARGS <<< "${INST}"
read -r -a FT_CE_ARGS <<< "${FT_CES}"
read -r -a QTOF_CE_ARGS <<< "${QTOF_CES}"

python preproc_scripts/inference/02_inf_prepare_spec_df.py \
    --mol_df_path ${PROC_DP}/mol_df.pkl.gz \
    --prec_types "${PREC_ARGS[@]}" \
    --ft_ces "${FT_CE_ARGS[@]}" \
    --qtof_ces "${QTOF_CE_ARGS[@]}" \
    --inst_types "${INST_ARGS[@]}" \
    --output_dir ${PROC_DP}

# 03_prepare_dag_feats.py reads plain mol_df.pkl / spec_df.pkl, but the inference
# scripts write .pkl.gz — decompress in place (gzip'd pickle -> plain pickle).
echo `date`: decompressing proc pickles for 03_prepare_dag_feats.py
gunzip -c ${PROC_DP}/mol_df.pkl.gz > ${PROC_DP}/mol_df.pkl
gunzip -c ${PROC_DP}/spec_df.pkl.gz > ${PROC_DP}/spec_df.pkl
ls -l ${PROC_DP}

# save proc to scratch
echo `date`: saving proc to ${DEST_PROC_DP}
mkdir -p ${DEST_PROC_DP}
tar -cf ${DATASET_NAME}_proc.tar ${PROC_DP}
cp ${DATASET_NAME}_proc.tar ${DEST_PROC_DP}/

if [ "${RUN_STEP_03}" != "True" ] && [ "${RUN_STEP_03}" != "true" ]; then
    echo `date`: RUN_STEP_03=${RUN_STEP_03} - stopping after the proc dir.
    echo "proc tar: ${DEST_PROC_DP}/${DATASET_NAME}_proc.tar"
    echo "Build the DAGs with preproc_infer_synced_dag_arr_hdf5.sh, pointing"
    echo "SRC_PROC_DP at this proc dir."
else
    # === Step 03: DAG feature generation =======================================
    echo `date`: Step 03 - DAG feature generation
    read -r -a FRAG_MODE_ARGS <<< "${FRAG_MODE}"
    read -r -a ION_MODE_ARGS <<< "${ION_MODES}"
    read -r -a DSET_ARGS <<< "${DSETS}"

    mkdir -p data/frag
    python preproc_scripts/03_prepare_dag_feats.py \
        --max_depth ${MAX_DEPTH} \
        --max_h_transfer ${MAX_H_TRANSFER} \
        --max_time ${MAX_TIME} \
        --frag_dp ./data/frag/${SAVE_NAME} \
        --proc_dp ./${PROC_DP} \
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

    # === save frags to scratch =================================================
    echo `date`: saving frags to ${DEST_FRAG_DP}
    mkdir -p ${DEST_FRAG_DP}
    cd ${WORKING_DIR}/data/frag
    ls -l ${SAVE_NAME}

    if [ "${USE_HDF5}" = "True" ] || [ "${USE_HDF5}" = "true" ]; then
        # HDF5 layout: a single dags.h5 plus meta_info.json — copy directly, no tar.
        mkdir -p ${DEST_FRAG_DP}/${SAVE_NAME}
        cp ${SAVE_NAME}/dags.h5 ${DEST_FRAG_DP}/${SAVE_NAME}/
        cp ${SAVE_NAME}/meta_info.json ${DEST_FRAG_DP}/${SAVE_NAME}/
        echo copied dags.h5 to ${DEST_FRAG_DP}/${SAVE_NAME}/
    else
        # Pickle layout: one compressed artifact per molecule under dags/ — tar it.
        tar -cf ${SAVE_NAME}.tar ${SAVE_NAME}
        cp ${SAVE_NAME}.tar ${DEST_FRAG_DP}/
        echo copied tar: ${SAVE_NAME}.tar to ${DEST_FRAG_DP}
    fi

fi

echo `date`: Job ${SLURM_JOB_ID} finished running
