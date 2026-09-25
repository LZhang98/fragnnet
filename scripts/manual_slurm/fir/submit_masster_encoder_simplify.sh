#!/bin/bash
# Generate (but do not submit) the Fir job for the 2026-09-23 encoder-simplification arm.
#
#   g1_gine_gsa5_no_bond_head  one 5-layer GINE_GSA atom encoder in place of 2 GINE_GSA layers
#                              plus the 3-block BondFocusedBreakHead; bonds scored from their
#                              endpoints; boundary CutBondEncoder off. 9.22M -> 7.69M parameters.
#
# NO CONTROL JOB IS GENERATED. `c4a_base_s1`/`s2` already ran and are the control (tail mean
# 0.7270, the two replicates spanning 0.0007). g1 is a diff of `c4a_base_s1` at the same seed and
# the same monitoring, so hours per epoch compare directly.
#
# Profile fir:h100-3g.40gb, identical to the c4a arms: batch 512, loss_batch_size 64 and
# `compute_train_auxiliary_metrics: false`, which kept the control at 30.4 GiB peak. Wall time
# 3:00:00 matches the c4a arms' 128-epoch budget.
#
# READING THE ARM -- val_cos_sim_0.01_epoch/mean, 118-127 tail mean, gate +/-0.005:
#   g1 - control      whether the simpler encoder holds cosine; pair it with hours per epoch.
#   g1 - c4a_no_bbe   (`pkavjg8z`, 0.7246) the depth move with the boundary key held equal.
#
# The arm passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_encoder_simplify"
JOB_DIR="${REPO_ROOT}/jobs"
CAMPAIGN_DIR="config/masster/ma_mi_merged/encoder_simplify_2026_09_23"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"

JOB_NAMES=(msh_es_g1_gsa5_no_bond_head)
CONFIGS=("${CAMPAIGN_DIR}/g1_gine_gsa5_no_bond_head.yml")

cd "${REPO_ROOT}"

# 1. Validate everything before creating any job script.
if [[ "${#JOB_NAMES[@]}" -ne "${#CONFIGS[@]}" ]]; then
    echo "JOB_NAMES and CONFIGS must contain the same number of entries." >&2
    exit 1
fi
for config_fp in "${CONFIGS[@]}"; do
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
done
# The generated job installs both archives from the checkout; wheels/ is Git-ignored, so a fresh
# cluster clone does not have them and the failure would otherwise appear only after the job starts.
for wheel in wheels/pyteomics-4.7.5-py3-none-any.whl wheels/torch_linear_assignment-0.0.6.tar.gz; do
    if [[ ! -f "${wheel}" ]]; then
        echo "Missing ${wheel} -- the generated job installs it and wheels/ is Git-ignored." >&2
        exit 1
    fi
done

# 2. Generate the Fir 3g.40gb job scripts.
for index in "${!CONFIGS[@]}"; do
    python scripts/setup_job_cc.py \
        --job_name "${JOB_NAMES[index]}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile fir:h100-3g.40gb \
        --time 3:00:00 \
        --cc_account "${ACCOUNT}" \
        --cc_email "${EMAIL}" \
        --script_kwargs w=online c="${CONFIGS[index]}" j="${JOB_NAMES[index]}" \
        --no_submit=True
done

echo
echo "Generated scripts hold a W&B credential -- keep them out of shared records."
echo "Review, then submit from a Fir login node:"
for job_name in "${JOB_NAMES[@]}"; do
    echo "  bash -n ${SCRIPT_DIR}/${job_name}.sh"
done
for job_name in "${JOB_NAMES[@]}"; do
    echo "  sbatch ${SCRIPT_DIR}/${job_name}.sh"
done
echo
echo "Control is the EXISTING c4a_base_s1/s2 pair; do not re-run it."
