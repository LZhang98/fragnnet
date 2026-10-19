#!/bin/bash
# Generate (but do not submit) the Fir jobs for the MSGYM short-schedule / candidate-budget screen:
# the tiwhun9q leader recipe (720jwmai, 10.14M parameters, pow0.75, loss-selected checkpoint) at
# 64 epochs, with one arm per question.
# Plan: docs/plans/masster/msgym_short_budget_screen_2026_10_14.md
#
# STAGE selects which jobs are generated:
#   all      (default) all six. Submit them together so they share code and queue conditions.
#   control  c64 and c32 only.
#   arms     u128, e1024, u128_e1024, floor01 only (read against c64).
#
# All arms: seed 397623, batch 256, loss chunks 16, checkpoint on val_primary_loss min, test split
# disabled, candidate coverage metrics off so wall-clock is comparable across arms.
#
# PROFILE fir:h100-3g.40gb, the profile the sweep and the leader re-run used.
#
# WALL TIME 5:00:00. The leader re-run f44tzxoc took 4.98 h for 128 epochs, so 64 epochs should take
# ~2.5 h; the margin covers slower MIG hosts. floor01 and e1024 have not had a real-data GPU smoke
# (no local MSGYM spec_df): check their first epoch.
#
# Every job passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/msgym_short_budget"
JOB_DIR="${REPO_ROOT}/jobs"
MSGYM_DIR="config/masster/massspecgym"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="5:00:00"

CTRL_ARMS=(c64 c32)
TEST_ARMS=(u128 e1024 u128_e1024 floor01)

case "${STAGE}" in
    all)     ARMS=("${CTRL_ARMS[@]}" "${TEST_ARMS[@]}") ;;
    control) ARMS=("${CTRL_ARMS[@]}") ;;
    arms)    ARMS=("${TEST_ARMS[@]}") ;;
    *) echo "Usage: $0 [all|control|arms]" >&2; exit 1 ;;
esac

JOB_NAMES=()
CONFIGS=()
for arm in "${ARMS[@]}"; do
    JOB_NAMES+=("msgym_sb_${arm}")
    CONFIGS+=("${MSGYM_DIR}/msgym_sb_${arm}.yml")
done

cd "${REPO_ROOT}"

# 1. Validate everything before creating any job script.
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
        --gpu_profile "${PROFILE}" \
        --time "${TIME}" \
        --cc_account "${ACCOUNT}" \
        --cc_email "${EMAIL}" \
        --script_kwargs w=online "c=${CONFIGS[index]}" "j=${JOB_NAMES[index]}" \
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
