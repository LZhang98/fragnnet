#!/bin/bash
# Shared body for the infer_synced_arr_*.sh array variants.
#
# Runs scripts/run_inference.py over one shard of the SYNCED inference set. The shard
# is a contiguous slice of mol_df rows
#   [SLURM_ARRAY_TASK_ID*CHUNK_SIZE, (SLURM_ARRAY_TASK_ID+1)*CHUNK_SIZE)
# which is exactly the slice preproc_infer_synced_dag_arr*.sh used, so task i consumes
# frag shard i and NO merge step is needed.
#
# FRAG_BACKEND picks how that shard's DAGs are staged:
#   tar  -> ${DEST_FRAG_DP}/${SAVE_NAME}/${SAVE_NAME}_<task>.tar, extracted to
#           ${SAVE_NAME}/dags/ (the folder backend; frag_dp points at dags/)
#   hdf5 -> ${DEST_FRAG_DP}/${SAVE_NAME}/shard_<task>/dags.h5 (the packed backend;
#           frag_dp points at the .h5 itself)
#
# Each task writes its own predictions dir (predictions_chunk_* filenames are only
# unique within a dir, so tasks must NOT share one). Resume is per-task: re-running
# the array skips spectra already present in that task's output dir.
#
# The caller must set: SAVE_NAME, FRAG_BACKEND, MODEL_CKPT, and an #SBATCH --array
# range consistent with CHUNK_SIZE.

echo "$(date): Job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}) is allocated resources"

module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
WORKING_DIR=${WORKING_DIR:-${SLURM_TMPDIR}/${SLURM_JOB_ID}}

DATASET_NAME=${DATASET_NAME:-synced}
MAX_DEPTH=${MAX_DEPTH:-3}
MAX_H_TRANSFER=${MAX_H_TRANSFER:-4}
SAVE_NAME=${SAVE_NAME:-${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}}
FRAG_BACKEND=${FRAG_BACKEND:-tar}

CHUNK_SIZE=${CHUNK_SIZE:-20000}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# One array task per frag shard: task i consumes shard i and predicts exactly the mol_df
# slice that built it. Partial progress is handled by resume (each task skips the spec_ids
# already in its own output dir), not by splitting shards further.
START_IDX=$((TASK_ID * CHUNK_SIZE))
NUM_MOLS=${CHUNK_SIZE}

# --- model -----------------------------------------------------------------
# Set exactly one of:
#   MODEL_CKPT    path to a .ckpt (absolute, or relative to FRAGNNET_DP)
#   WANDB_RUN_ID  W&B run id, whose best checkpoint is downloaded into MODEL_SAVE_DP
# WANDB_RUN_ID works from the compute nodes, but every array task downloads
# independently: with a cold cache they all shutil.copy (not atomic) onto the same
# ${MODEL_SAVE_DP}/<run_name>_<run_id>.ckpt, so a task can load a half-written file.
# Warm the cache once before submitting, then the tasks just hit it:
#   python -c "from fragnnet.utils.misc_utils import get_best_ckpt_from_wandb; \
#              print(get_best_ckpt_from_wandb('saved_ckpts', '<run_id>'))"
# Passing the printed path as MODEL_CKPT is equivalent and one less moving part.
MODEL_CKPT=${MODEL_CKPT:-}
WANDB_RUN_ID=${WANDB_RUN_ID:-}
MODEL_SAVE_DP=${MODEL_SAVE_DP:-${FRAGNNET_DP}/saved_ckpts}
# The checkpoint's training config. SET THIS for any checkpoint not trained against the
# current code: get_config_from_ckpt merges config/template.yml ONLY when custom_fp is
# given, so otherwise config_d is the ckpt's saved hparams alone and every config key
# added since it was trained is missing -> model init dies on self.hparams.<new_key>
# (e.g. AttributeError: ... has no attribute 'depth_ce_bias').
CUSTOM_FP=${CUSTOM_FP:-}

if [ -z "${MODEL_CKPT}" ] && [ -z "${WANDB_RUN_ID}" ]; then
    echo "ERROR: set MODEL_CKPT (preferred) or WANDB_RUN_ID" >&2
    exit 1
fi

# Resolve relative paths against FRAGNNET_DP: the job may run out of a copy of the repo
# in ${SLURM_TMPDIR}, where a repo-relative ckpt/config path would not exist.
case "${MODEL_CKPT}" in "" | /*) ;; *) MODEL_CKPT=${FRAGNNET_DP}/${MODEL_CKPT} ;; esac
case "${CUSTOM_FP}" in "" | /*) ;; *) CUSTOM_FP=${FRAGNNET_DP}/${CUSTOM_FP} ;; esac

# --- inference knobs -------------------------------------------------------
BATCH_SIZE=${BATCH_SIZE:-16}
# 10.0 = keep peaks above 1% of base peak (pred_ints is rescaled per spectrum to a
# 0-1000 base peak, filtered with a strict >). Higher than run_inference.py's own 1.0
# default: measured on NIST20 val (2,000 spectra, 9t63zils), 1.0 -> 10.0 costs
# cos_hun -0.0001 while cutting peaks ~3x (p50 39 -> 13, p90 202 -> 60), and 13/60
# matches what real spectra carry (p50 11, p90 45). At 4.05M spectra that is the
# difference between a ~3x larger export and the same score.
# Trade-off: the export is then even further from the DAG support, so peak-coverage
# analysis off these files is not possible - see
# docs/findings/dominant_peak_outcome_decomposition.md ("min_peak_int correction").
MIN_INTENSITY=${MIN_INTENSITY:-10.0}
SAVE_FORMAT=${SAVE_FORMAT:-pkl}
# Peaks + fragment annotation. MIN_INTENSITY prunes peaks, formulas, the joint grid and
# (as of the node-pruning fix) the node arrays, so every array below covers only what is
# behind a surviving peak.
#
#   pred_mzs / pred_ints            the spectrum
#   frag_formula_peak_idxs          peak -> formula        (always written)
#   pred_formula_strs               NEUTRAL fragment formula (RETURN_FORMULA_STR)
#   pred_formula_adducts            precursor delta for that row, signed: +H, -H, +Na
#                                   (written whenever EITHER annotation flag is set)
#   pred_formula_probs              P(f)
#   pred_joint_formula_idxs/_node_idxs + pred_formula_node_probs
#                                   formula -> candidate node, with P(n|f)
#   pred_node_ccs                   node -> atom bitmask   (RETURN_CC_HASH)
#                                   2 x int64, MASK_SIZE=128; resolve to SMILES against
#                                   the parent molecule downstream
#
# So peak k's formula is pred_formula_strs[frag_formula_peak_idxs[k]], and its fragment
# is the argmax of pred_formula_node_probs over the joint rows with that formula - all
# indices are rebased per spectrum, so the lookups are local.
#
# The formula string is the NEUTRAL fragment; pred_formula_adducts carries the delta that
# makes it the observed ion, so a peak reads as "C6H4O2 -H" and the annotated fragment is
# that cc PLUS that adduct. With frag_params.include_cmf the formula axis is
# NULL, CRF[1:F], CMF[1:F] and the two halves carry different deltas (+Na vs +H for
# [M+Na]+), so the adduct column is what tells the two channels apart.
#
# The joint index columns come in automatically with RETURN_CC_HASH (they are what prunes
# the node arrays); P(n|f) is what makes them usable, hence its place in OUTPUT_SUBSET.
# Add 'pred_node_probs' for P(n), or drop 'pred_formula_probs' if P(f) is not needed.
#
# Size at MIN_INTENSITY=10.0, measured on NIST20 val (9t63zils, 1,000 spectra): per
# spectrum p50 12 peaks / 12 formulas / 49 joint rows / 38 nodes, mean 20.8 / 20.8 /
# 102.9 / 64.4. Over SYNCED's 4.05M spectra that is ~4.2 GB of cc bitmasks and ~8.3 GB
# of joint rows. Unpruned it would be ~152 GB and ~2.4 TB - so do NOT lower
# MIN_INTENSITY without recomputing the budget.
OUTPUT_SUBSET=${OUTPUT_SUBSET:-'pred_mzs pred_ints pred_formula_probs pred_formula_node_probs'}
RETURN_FORMULA_STR=${RETURN_FORMULA_STR:-True}
RETURN_CC_HASH=${RETURN_CC_HASH:-True}
SAVE_CHUNK_SIZE=${SAVE_CHUNK_SIZE:-10000}

# --- io --------------------------------------------------------------------
SRC_PROC_DP=${SRC_PROC_DP:-${FRAGNNET_DP}/data/inference/synced}
if [ ! -d "${SRC_PROC_DP}" ] && [ -d "${FRAGNNET_DP}/data/inference/synced_smiles" ]; then
    SRC_PROC_DP=${FRAGNNET_DP}/data/inference/synced_smiles
fi
DEST_FRAG_DP=${DEST_FRAG_DP:-/scratch/feiw/frag-gnn-data/inference/frags}
# The output dir has to identify the MODEL, not just the DAGs. Resume skips any spec_id
# already present in the task's dir, so two checkpoints sharing one OUT_DP would not
# collide loudly - the second run would skip whatever the first predicted and leave a
# silently mixed-model prediction set. SAVE_NAME covers only the dataset + DAG params and
# is shared with DEST_FRAG_DP (which is model-independent), hence this separate tag.
# Mirrors the MODEL_CKPT-wins precedence used to build CKPT_ARGS below.
MODEL_TAG=${MODEL_TAG:-}
if [ -z "${MODEL_TAG}" ]; then
    if [ -n "${MODEL_CKPT}" ]; then
        MODEL_TAG=$(basename "${MODEL_CKPT}" .ckpt)
    else
        MODEL_TAG=${WANDB_RUN_ID}
    fi
fi
OUT_DP=${OUT_DP:-/scratch/feiw/frag-gnn-data/inference_outputs/${SAVE_NAME}_${MODEL_TAG}}

echo SAVE_NAME: ${SAVE_NAME}
echo FRAG_BACKEND: ${FRAG_BACKEND}
echo MODEL_CKPT: ${MODEL_CKPT}
echo WANDB_RUN_ID: ${WANDB_RUN_ID}
echo MODEL_SAVE_DP: ${MODEL_SAVE_DP}
echo CUSTOM_FP: ${CUSTOM_FP}
echo SRC_PROC_DP: ${SRC_PROC_DP}
echo DEST_FRAG_DP: ${DEST_FRAG_DP}
echo MODEL_TAG: ${MODEL_TAG}
echo OUT_DP: ${OUT_DP}
echo TASK_ID: ${TASK_ID}
echo CHUNK_SIZE: ${CHUNK_SIZE}
echo START_IDX: ${START_IDX}
echo NUM_MOLS: ${NUM_MOLS}
echo BATCH_SIZE: ${BATCH_SIZE}
echo SAVE_FORMAT: ${SAVE_FORMAT}
echo OUTPUT_SUBSET: ${OUTPUT_SUBSET}
echo RETURN_FORMULA_STR: ${RETURN_FORMULA_STR}
echo RETURN_CC_HASH: ${RETURN_CC_HASH}
nvidia-smi || true

if [ ! -d "${SRC_PROC_DP}" ]; then
    echo "ERROR: SRC_PROC_DP does not exist: ${SRC_PROC_DP}" >&2
    exit 1
fi

# --- array width vs frag shards --------------------------------------------
# #SBATCH --array is a static directive, so nothing keeps it in step with the number of
# shards preprocessing actually wrote. Count the shards on disk and report the comparison
# here, before the venv build, so the numbers are at the top of the log rather than behind
# a pip install.
if [ "${FRAG_BACKEND}" = "hdf5" ]; then
    NUM_FRAG_SHARDS=${NUM_FRAG_SHARDS:-$(ls -d ${DEST_FRAG_DP}/${SAVE_NAME}/shard_* 2>/dev/null | wc -l | tr -d ' ')}
else
    NUM_FRAG_SHARDS=${NUM_FRAG_SHARDS:-$(ls ${DEST_FRAG_DP}/${SAVE_NAME}/${SAVE_NAME}_*.tar 2>/dev/null | wc -l | tr -d ' ')}
fi
echo NUM_FRAG_SHARDS: ${NUM_FRAG_SHARDS}
if [ "${NUM_FRAG_SHARDS}" -eq 0 ]; then
    echo "ERROR: no ${FRAG_BACKEND} frag shards under ${DEST_FRAG_DP}/${SAVE_NAME}" >&2
    echo "       Run preproc_infer_synced_dag_arr_hdf5.sh first, or set DEST_FRAG_DP/SAVE_NAME." >&2
    exit 1
fi
# A narrower array is a normal way to run this (one shard at a time, a rerun of a few
# failed tasks), so the mismatch is reported and the task continues. What it costs you:
# shards with no task are simply not predicted, and every task still exits 0, so the run
# can cover only part of mol_df without any failure to notice. Reconcile shard count
# against task count yourself before treating a sweep as complete.
if [ -n "${SLURM_ARRAY_TASK_MAX}" ] &&
   { [ $((SLURM_ARRAY_TASK_MAX + 1)) -ne "${NUM_FRAG_SHARDS}" ] ||
     [ "${SLURM_ARRAY_TASK_COUNT:-$((SLURM_ARRAY_TASK_MAX + 1))}" -ne "${NUM_FRAG_SHARDS}" ]; }; then
    echo "NOTE: array spans ${SLURM_ARRAY_TASK_MIN:-0}-${SLURM_ARRAY_TASK_MAX} with ${SLURM_ARRAY_TASK_COUNT:-unknown} tasks," \
         "but ${DEST_FRAG_DP}/${SAVE_NAME} holds ${NUM_FRAG_SHARDS} frag shards."
    echo "      A full sweep is:  sbatch --array=0-$((NUM_FRAG_SHARDS - 1)) <script>"
fi

# --- env -------------------------------------------------------------------
mkdir -p ${WORKING_DIR}
cd ${WORKING_DIR}

# Always build the venv in the job. pyarrow is deliberately absent: on fir it resolves to
# the Alliance dummy wheel, which fails the build and takes torch down with it, and
# nothing here needs it (--save_format pkl; parquet cannot hold the torch tensors this
# export writes). setup.cfg is gone from the repo - pyproject.toml is the build config.
echo `date`: building venv in ${WORKING_DIR}
virtualenv FRAG-GNN
source FRAG-GNN/bin/activate
pip install --no-index --upgrade pip
pip install --no-index \
    torch==2.4.1 \
    torch_geometric==2.6.1 \
    torch_scatter==2.1.2 \
    pandas \
    wandb==0.21.2 \
    tqdm \
    scipy==1.15.2 \
    joblib==1.3.2 \
    matplotlib \
    lightning==2.4.0 \
    dgl==2.4.0 \
    omegaconf \
    cython==3.0.12 \
    pyopenms==3.3.0 \
    pandarallel==1.6.5 \
    h5py

cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${WORKING_DIR}/pyteomics-4.7.5-py3-none-any.whl
pip install pyteomics-4.7.5-py3-none-any.whl

rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/preproc_scripts ${WORKING_DIR}
rsync -av --exclude=".*" --exclude="__pycache__" ${FRAGNNET_DP}/scripts ${WORKING_DIR}
rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" ${FRAGNNET_DP}/src ${WORKING_DIR}
cp ${FRAGNNET_DP}/setup.py ${WORKING_DIR}/
cp ${FRAGNNET_DP}/pyproject.toml ${WORKING_DIR}/
mkdir -p ${WORKING_DIR}/config
cp ${FRAGNNET_DP}/config/template.yml ${WORKING_DIR}/config/
pip install --no-deps -I ${WORKING_DIR}
cd ${WORKING_DIR}

# === stage this shard's DAGs ===============================================
STAGE_DP=${SLURM_TMPDIR:-/tmp}/frag_${TASK_ID}
mkdir -p ${STAGE_DP}

if [ "${FRAG_BACKEND}" = "hdf5" ]; then
    SHARD_H5=${DEST_FRAG_DP}/${SAVE_NAME}/shard_${TASK_ID}/dags.h5
    SHARD_META=${DEST_FRAG_DP}/${SAVE_NAME}/shard_${TASK_ID}/meta_info.json
    if [ ! -f "${SHARD_H5}" ]; then
        echo "ERROR: missing frag shard ${SHARD_H5}" >&2
        exit 1
    fi
    echo `date`: staging ${SHARD_H5}
    cp ${SHARD_H5} ${STAGE_DP}/dags.h5
    # meta_info.json must sit next to the .h5 - the dataset reads it for node/edge counts.
    [ -f "${SHARD_META}" ] && cp ${SHARD_META} ${STAGE_DP}/
    FRAG_ARG=${STAGE_DP}/dags.h5
else
    SHARD_TAR=${DEST_FRAG_DP}/${SAVE_NAME}/${SAVE_NAME}_${TASK_ID}.tar
    if [ ! -f "${SHARD_TAR}" ]; then
        echo "ERROR: missing frag shard ${SHARD_TAR}" >&2
        exit 1
    fi
    echo `date`: staging ${SHARD_TAR}
    cp ${SHARD_TAR} ${STAGE_DP}/
    tar -xf ${STAGE_DP}/$(basename ${SHARD_TAR}) -C ${STAGE_DP}
    rm -f ${STAGE_DP}/$(basename ${SHARD_TAR})
    # The tar holds <SAVE_NAME>/{dags/,meta_info.json}; frag_dp is the dags/ dir.
    FRAG_ARG=${STAGE_DP}/${SAVE_NAME}/dags
    if [ ! -d "${FRAG_ARG}" ]; then
        echo "ERROR: extracted tar has no dags/ dir at ${FRAG_ARG}" >&2
        ls -R ${STAGE_DP} | head -20
        exit 1
    fi
fi
echo FRAG_ARG: ${FRAG_ARG}

# === run inference =========================================================
TASK_OUT_DP=${OUT_DP}/shard_${TASK_ID}
mkdir -p ${TASK_OUT_DP}

CUSTOM_ARGS=()
if [ -n "${CUSTOM_FP}" ]; then
    CUSTOM_ARGS=(--custom_fp "${CUSTOM_FP}")
fi

read -r -a OUTPUT_SUBSET_ARGS <<< "${OUTPUT_SUBSET}"

# run_inference.py falls back to the W&B download whenever --model_ckpt is absent OR
# points at a nonexistent file, so an unset MODEL_CKPT is simply not passed.
CKPT_ARGS=()
if [ -n "${MODEL_CKPT}" ]; then
    if [ ! -f "${MODEL_CKPT}" ]; then
        echo "ERROR: MODEL_CKPT does not exist: ${MODEL_CKPT}" >&2
        exit 1
    fi
    CKPT_ARGS=(--model_ckpt "${MODEL_CKPT}")
else
    CKPT_ARGS=(--wandb_run_id "${WANDB_RUN_ID}" --model_save_dp "${MODEL_SAVE_DP}")
fi

echo `date`: running inference for shard ${TASK_ID}
python ${WORKING_DIR}/scripts/run_inference.py \
    "${CKPT_ARGS[@]}" \
    --mol_fp ${SRC_PROC_DP}/mol_df.pkl.gz \
    --spec_fp ${SRC_PROC_DP}/spec_df.pkl.gz \
    --frag_dp ${FRAG_ARG} \
    --start_mol_idx ${START_IDX} \
    --num_mols ${NUM_MOLS} \
    --batch_size ${BATCH_SIZE} \
    --min_intensity ${MIN_INTENSITY} \
    --save_dp ${TASK_OUT_DP} \
    --save_format ${SAVE_FORMAT} \
    --output_subset "${OUTPUT_SUBSET_ARGS[@]}" \
    --return_formula_str ${RETURN_FORMULA_STR} \
    --return_cc_hash ${RETURN_CC_HASH} \
    --save_chunk_size ${SAVE_CHUNK_SIZE} \
    "${CUSTOM_ARGS[@]}"

echo `date`: Job ${SLURM_JOB_ID} task ${TASK_ID} finished running
ls -l ${TASK_OUT_DP} | head
