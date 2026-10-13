#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=infer_synced_arr_hdf5
# Resources are the fir:h100-3g.40gb preset from GPU_PROFILES in scripts/setup_job_cc.py
# (python scripts/setup_job_cc.py --list_gpu_profiles). Keep the three lines below as one
# unit when switching presets: the cpu/mem values are that slice's proportional share, so
# a mismatched pair bills at the higher ratio. Inference runs with num_workers=0, so the
# cpu count only covers the shard staging copy, not dataloading.
#SBATCH --mem=140G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100_3g.40gb:1
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/frag-gnn/jobs/infer_synced_arr_hdf5_%A_%a.out
#SBATCH --error=/home/feiw/project/feiw/frag-gnn/jobs/infer_synced_arr_hdf5_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=0-5

# SYNCED inference, packed HDF5 frag backend, as an array job.
#
# Consumes the per-shard HDF5 written by preproc_infer_synced_dag_arr_hdf5.sh:
#   ${DEST_FRAG_DP}/synced_d3_h4_hdf5/shard_<task>/{dags.h5,meta_info.json}
# Task i predicts mol_df rows [i*CHUNK_SIZE, (i+1)*CHUNK_SIZE) - the same slice that
# built shard i - so merge_dag_h5_shards.py is NOT needed here; merge only when you
# want to run inference as a single job. Keep --array and CHUNK_SIZE in sync with the
# preprocessing array (289,249 molecules / 48209 -> 6 tasks); a narrower array is allowed
# (run one shard, rerun a few failed tasks) and the common body only notes the mismatch -
# the shards it leaves out are silently not predicted. A task that runs out of wall time
# is resumed by resubmitting the array - it skips whatever it already wrote.
#
# Output: ${OUT_DP}/shard_<task>/predictions_chunk_*.pkl

DATASET_NAME=synced
MAX_DEPTH=3
MAX_H_TRANSFER=4
SAVE_NAME=${DATASET_NAME}_d${MAX_DEPTH}_h${MAX_H_TRANSFER}_hdf5
FRAG_BACKEND=hdf5
CHUNK_SIZE=48209

# --- MODEL: set one of these two ------------------------------------------
# A checkpoint and its training config are ONE pairing, never two independent knobs:
# get_config_from_ckpt writes CUSTOM_FP's contents over the checkpoint's saved hparams
# with no check that the two belong together, so a config from the wrong run loads
# silently and mistrains the architecture it builds. The defaults below are therefore
# set as a unit - override a model selector and you MUST pass CUSTOM_FP with it.
#
# W&B run id, resolved to that run's best checkpoint under MODEL_SAVE_DP. The download
# works from the compute nodes, but WARM THE CACHE ONCE before submitting so the array
# tasks do not race each other writing the same cached .ckpt:
#   python -c "from fragnnet.utils.misc_utils import get_best_ckpt_from_wandb; \
#              print(get_best_ckpt_from_wandb('saved_ckpts', 'jnokddr9'))"
WANDB_RUN_ID=${WANDB_RUN_ID:-}
# ...or a local checkpoint, which wins when set. Relative paths resolve against
# FRAGNNET_DP. Passing the path printed by the command above skips the W&B lookup
# entirely, which is one less moving part across 6 tasks.
MODEL_CKPT=${MODEL_CKPT:-}
# Training config for the selected checkpoint (relative to FRAGNNET_DP). REQUIRED, not
# optional: get_config_from_ckpt merges config/template.yml ONLY when custom_fp is given,
# so leaving this empty makes config_d the ckpt's saved hparams alone and every config key
# added since it was trained is missing -> model init dies on self.hparams.<new_key>.
CUSTOM_FP=${CUSTOM_FP:-}

if [ -z "${MODEL_CKPT}" ] && [ -z "${WANDB_RUN_ID}" ]; then
    # Nothing overridden: use the default run together with the config it was trained on.
    WANDB_RUN_ID=jnokddr9
    CUSTOM_FP=${CUSTOM_FP:-config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml}
elif [ -z "${CUSTOM_FP}" ]; then
    echo "ERROR: MODEL_CKPT/WANDB_RUN_ID was overridden but CUSTOM_FP was not." >&2
    echo "       Pass the training config belonging to that checkpoint:" >&2
    echo "         CUSTOM_FP=config/<...>.yml sbatch <script>" >&2
    echo "       There is no default that is safe here - the config is written over the" >&2
    echo "       checkpoint's hparams unchecked, so the wrong one loads silently." >&2
    exit 1
fi

# Batching: cap the fragment nodes per batch instead of fixing the spectra per batch.
# See the "inference knobs" block in infer_synced_arr_common.sh for the mechanism; the
# number comes from SYNCED's own size distribution, measured on 400 random molecules
# (d3/h4, the DAGs this job consumes):
#
#   frag_node per molecule   mean 1,549   p50 332   p90 1,160   p99 24,609   max 108,721
#
# Every one of a molecule's 14 spec_df rows carries that whole DAG, so the old
# BATCH_SIZE=64 produced batches of mean 99,591 nodes but p99 1,165,156 and max 1,552,120
# - a 16x spread, with the GPU sized for a tail batch that is 50x the median one.
# 100,000 is that same mean, so average batch cost and batch count are unchanged (90
# batches vs 87 on the simulated stream), while the peak drops 14x to 108,721.
#
# Do not lower it hoping for a smaller peak: 2 of those 400 molecules have a DAG bigger
# than the cap on their own, so below ~110,000 the peak is pinned by the largest single
# molecule and a smaller cap only costs batches (143 at 50,000, 233 at 25,000). Raising
# it is the useful direction - this config TRAINS at dynamic_batch_sampler_max: 400000
# with gradients, so inference has room. Read the `>> resources:` GPU peak line from a
# first shard before changing it, and keep frag_node: this config sets pyg_edges: False,
# so frag_edge measures 0 on every sample and run_inference.py raises.
#
# DYNAMIC_BATCH_MAX=0 sbatch <script>   restores fixed batches of BATCH_SIZE spectra.
DYNAMIC_BATCH_MAX=${DYNAMIC_BATCH_MAX:-100000}
BATCH_SIZE=${BATCH_SIZE:-64}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRAGNNET_DP=${FRAGNNET_DP:-/home/feiw/project/feiw/frag-gnn}
if [ -f "${SCRIPT_DIR}/infer_synced_arr_common.sh" ]; then
    source "${SCRIPT_DIR}/infer_synced_arr_common.sh"
else
    source "${FRAGNNET_DP}/scripts/manual_slurm/fir/infer_synced_arr_common.sh"
fi
