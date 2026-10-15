#!/bin/bash
# Generate (but do not submit) the Fir job that re-evaluates the MSGYM c64 checkpoint
# (u80w840n, U256/E4096, best val_primary_loss at epoch 21) on the val split at several
# unique-fragment targets U, writing one row per spectrum per U. The per-spectrum rows give each
# molecule's U* (the smallest U past which recall stops improving) for the adaptive-U question.
# Plan: docs/plans/masster/msgym_short_budget_screen_2026_10_14.md
#
# Weights stay fixed and only U changes; E stays at the trained 4096, so U384 can hit E.
# The checkpoint is fetched from the W&B run files (upload_checkpoints: true).
#
# PROFILE fir:h100-3g.40gb. Eval batch 32 with loss_batch_size 1 (the sweep script's setting).
# WALL TIME 3:00:00 for five budgets on the val split.
#
# Outputs are written straight to the checkout's output/eval/, not node-local scratch.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/msgym_u_budget_sweep"
JOB_DIR="${REPO_ROOT}/jobs"
OUTPUT_DIR="${REPO_ROOT}/output/eval/msgym_u_budget_sweep_c64"
CONFIG="config/masster/massspecgym/msgym_sb_c64.yml"
WANDB_RUN_ID="u80w840n"
JOB_NAME="msgym_u_budget_sweep_c64"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="3:00:00"

cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 1
fi
for wheel in wheels/pyteomics-4.7.5-py3-none-any.whl wheels/torch_linear_assignment-0.0.6.tar.gz; do
    if [[ ! -f "${wheel}" ]]; then
        echo "Missing ${wheel} -- the generated job installs it and wheels/ is Git-ignored." >&2
        exit 1
    fi
done
mkdir -p "${SCRIPT_DIR}" "${OUTPUT_DIR}"

python scripts/setup_job_cc.py \
    --job_name "${JOB_NAME}" \
    --python_script_fp scripts/eval/masster_mask_budget_sweep.py \
    --frag_gnn_dp "${REPO_ROOT}" \
    --script_dp "${SCRIPT_DIR}" \
    --job_dp "${JOB_DIR}" \
    --gpu_profile "${PROFILE}" \
    --time "${TIME}" \
    --cc_account "${ACCOUNT}" \
    --cc_email "${EMAIL}" \
    --script_kwargs \
        "custom_fp=${CONFIG}" \
        "wandb_run_id=${WANDB_RUN_ID}" \
        "mask_budgets=64 128 192 256 384" \
        "eval_split=val" \
        "eval_batch_size=32" \
        "wandb_mode=disabled" \
        "output_fp=${OUTPUT_DIR}/budget_means.csv" \
        "per_sample_output_fp=${OUTPUT_DIR}/per_spectrum.csv" \
    --no_submit=True

echo
echo "Generated script holds a W&B credential -- keep it out of shared records."
echo "Review, then submit from a Fir login node:"
echo "  bash -n ${SCRIPT_DIR}/${JOB_NAME}.sh"
echo "  sbatch ${SCRIPT_DIR}/${JOB_NAME}.sh"
