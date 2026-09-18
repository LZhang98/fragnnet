#!/bin/bash
# Generate (but do not submit) the Vulcan L40S jobs for the loss_batch_size A/B.
#
# Question: what is the ~1 s/step of non-candidate work in MASShatter?  Normalising both
# arms to 256 spectra per optimizer step, the non-candidate cost is ~0.94 s/step on the
# merged/cuda arm and ~0.98 s/step on the unmerged/cpu arm -- i.e. invariant to the
# candidate backend and to the 95% duplicate component pathways (those collapse to the
# ~232 unique fragments the intensity head actually sees).  The remaining suspect is the
# sparse cross-entropy chunk loop.
#
# `loss_batch_size` is a CEILING on chunk width in `_build_loss_chunks`
# (src/fragnnet/model/loss.py), so lb=8 at train_batch_size=128 gives >=16 sequential
# padded-Cartesian chunks per step.  The inner guard breaks a chunk as soon as
# `padded_pairs > flat_pairs`, so raising the ceiling only merges chunks where padding
# stays favourable -- it cannot blow memory up the way a naive increase would.
#
# The three configs differ ONLY in loss_batch_size, wandb_name and one tag, so this
# isolates the chunk count at fixed batch size, effective batch and LR.
#
# Baseline is already measured: run `qhs040sy` (lb=8) reached a steady-state 157 s/epoch
# on an L40S under a 1h wall.  These jobs use the same 1h wall so the comparison is
# matched -- ~23 epochs each, enough for both the timing read and an early quality curve.
#
# Read the result with the per-epoch _runtime delta, NOT total_runtime/epoch:
#   h = run.history(keys=['epoch','_runtime']).dropna().drop_duplicates('epoch', keep='last')
#   h['_runtime'].diff()[2:].median()
# Startup dominates the first epochs and makes the average badly misleading at low epoch
# counts.  The loss curves should overlay: chunking is a partition, so it changes only
# float association order, not the loss.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masshatter_l40s"
JOB_DIR="${REPO_ROOT}/jobs"

JOB_NAMES=(
    msh_mrg_fast_cuda_lb32
    msh_mrg_fast_cuda_lb64
)

CONFIGS=(
    config/masshatter/ma_mi_merged/nist20_ma_mi_merged_fast_cuda_lb32.yml
    config/masshatter/ma_mi_merged/nist20_ma_mi_merged_fast_cuda_lb64.yml
)

cd "${REPO_ROOT}"

if [[ "${#JOB_NAMES[@]}" -ne "${#CONFIGS[@]}" ]]; then
    echo "JOB_NAMES and CONFIGS must contain the same number of entries." >&2
    exit 1
fi

# 1. Validate every config before creating any jobs.
for config_fp in "${CONFIGS[@]}"; do
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
done

# 2. Generate the Vulcan L40S job scripts.
for index in "${!CONFIGS[@]}"; do
    python scripts/setup_job_cc.py \
        --job_name "${JOB_NAMES[index]}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile vulcan:l40s \
        --time 1:00:00 \
        --cc_account aip-rgreiner \
        --script_kwargs w=online c="${CONFIGS[index]}" j="${JOB_NAMES[index]}" \
        --no_submit=True
done

echo
echo "Review, then submit from a Vulcan login node:"
for job_name in "${JOB_NAMES[@]}"; do
    echo "  bash -n ${SCRIPT_DIR}/${job_name}.sh"
done
for job_name in "${JOB_NAMES[@]}"; do
    echo "  sbatch ${SCRIPT_DIR}/${job_name}.sh"
done
