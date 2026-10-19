#!/bin/bash
# Generate (but do not submit) the Fir jobs for the MSGYM rewire-rules screen: msgym_sb_c32.yml (tiwhun9q
# leader recipe at 32 epochs, U256/E4096, pow0.75, loss-selected checkpoint) with mechanism rules applied
# after fragment selection.
# Plan: docs/plans/masster/masster_rewire_rules_training_plan.md
#
# STAGE selects which jobs are generated:
#   all      (default) control, a0, a0d1, a0null. Submit together so they share code and queue conditions.
#   control  the same-batch c32 control only (earlier c32 runs: gz4fa02d, 3402ar4d).
#   arms     a0, a0d1, a0null only.
#
# Arms: a0 = rules, products only; a0d1 = rules plus one continuation cut (top 4 bonds); a0null = the
# same edit shapes at random sites (controls for the extra candidates). Cap: 0.5 x kept fragments,
# ceiling 256. Seed 397623, batch 256, test split disabled, coverage metrics off.
#
# PROFILE fir:h100-3g.40gb, as the short-budget screen.
#
# WALL TIME 3:00:00. c32 ran at 2.29 min/epoch (73 min); G0b measured the stage at 1.11x forward+backward
# on i1, and a0d1's cost is unmeasured, so the margin is wide. Local GPU smoke (256 spectra, batch 32,
# 2026-10-15) passed for all four arms.
#
# Every job passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/msgym_rewire"
JOB_DIR="${REPO_ROOT}/jobs"
MSGYM_DIR="config/masster/massspecgym"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="3:00:00"

CTRL_ARMS=(control)
TEST_ARMS=(a0 a0d1 a0null)

case "${STAGE}" in
    all)     ARMS=("${CTRL_ARMS[@]}" "${TEST_ARMS[@]}") ;;
    control) ARMS=("${CTRL_ARMS[@]}") ;;
    arms)    ARMS=("${TEST_ARMS[@]}") ;;
    *) echo "Usage: $0 [all|control|arms]" >&2; exit 1 ;;
esac

JOB_NAMES=()
CONFIGS=()
for arm in "${ARMS[@]}"; do
    JOB_NAMES+=("msgym_rw_${arm}")
    CONFIGS+=("${MSGYM_DIR}/msgym_rw_${arm}.yml")
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
