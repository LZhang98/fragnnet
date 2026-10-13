#!/bin/bash
# Generate (but do not submit) the Fir jobs that re-train the MSGYM val-loss sweep tiwhun9q
# leader and its matched control, so both have a loss-selected checkpoint.
# Plan: docs/plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leader-re-run-prepared-2026-10-12
#
# STAGE selects which jobs are generated:
#   both     (default) the leader and the control. Submit them together.
#   leader   720jwmai's recipe alone (batch 256, 10.14M parameters). The sweep trial was
#            interrupted at epoch ~76 and uploaded no checkpoint.
#   control  the sweep base run alone (batch 512, 9.23M parameters).
#
# Both are pow0.75, seed 397623, loss chunks 16, checkpoint on val_primary_loss min, test split
# disabled: they differ only in the swept hyperparameters.
#
# PROFILE fir:h100-3g.40gb, the profile every sweep trial ran on. A full forward/backward peak at
# the leader's depth has not been measured; check the first epoch for OOM.
#
# WALL TIME 8:00:00. All 33 interrupted sweep trials stopped at ~2.95 h (the leader at epoch ~76),
# and finished trials took 3.2-4.9 h for 128 epochs.
#
# Every job passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-both}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/msgym_sweep_leader"
JOB_DIR="${REPO_ROOT}/jobs"
MSGYM_DIR="config/masster/massspecgym"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="8:00:00"

LEADER_JOBS=(msgym_sweep_leader_720jwmai)
LEADER_CONFIGS=("${MSGYM_DIR}/msgym_sweep_leader_720jwmai.yml")

CTRL_JOBS=(msgym_sweep_base_control)
CTRL_CONFIGS=("${MSGYM_DIR}/msgym_sweep_base_control.yml")

case "${STAGE}" in
    both)    JOB_NAMES=("${LEADER_JOBS[@]}" "${CTRL_JOBS[@]}")
             CONFIGS=("${LEADER_CONFIGS[@]}" "${CTRL_CONFIGS[@]}") ;;
    leader)  JOB_NAMES=("${LEADER_JOBS[@]}"); CONFIGS=("${LEADER_CONFIGS[@]}") ;;
    control) JOB_NAMES=("${CTRL_JOBS[@]}"); CONFIGS=("${CTRL_CONFIGS[@]}") ;;
    *) echo "Usage: $0 [both|leader|control]" >&2; exit 1 ;;
esac

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
