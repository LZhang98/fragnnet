#!/bin/bash
# Generate (but do not submit) the Fir jobs for the proposal early-stop arms, 2026-09-23.
# Plan: 95341fce:docs/plans/masster/masster_proposal_early_stop_plan.md, removed 2026-10-18
#       95341fce:docs/findings/masster/masster_overfit_readout_2026_09_22.md, removed 2026-10-18, section 3
#
# Question: the validation PROPOSAL term turns at epochs 26-47 while the spectrum term keeps
# improving to 52-120. Does stopping (freeze) or decaying (proposal-only wd) the proposal-only
# heads -- 44.8% of the parameters -- help the spectrum?
#
# STAGE selects which jobs are generated:
#   rec   (default) p1 + p3, the two arms the plan recommends.
#   p1    freeze proposal-only modules from epoch 32.
#   p2    freeze from epoch 64. Held by the plan; generate only if p1 is interesting.
#   p3    proposal-only weight decay 0.1.
#   all   p1 + p2 + p3.
#
# Resolved-config diffs against c4a_base_s1, verified 2026-09-23 through load_config (besides
# wandb_name/tags/group):
#   p1  masster_proposal_frozen_from_epoch  null -> 32
#   p2  masster_proposal_frozen_from_epoch  null -> 64
#   p3  masster_proposal_weight_decay       null -> 0.1
# Same seed 397623 as c4a_base_s1, so the control is the existing 0.7270 tail mean.
#
# Profile fir:h100-3g.40gb, the slice c4a_base_s1/s2 ran on; the control is only like-for-like
# on the same slice. compute_train_auxiliary_metrics is false in all three, so peak memory is the
# control's ~30.4 GiB, not the ~37 GiB that forced the wd10 jobs onto the 80gb card.
#
# Wall time 3:00:00: min_epochs=max_epochs=128 and c4a_base finished 127 epochs in ~1.8 h.
#
# READING IT -- tail mean of epochs 118-127 of val_cos_sim_0.01_epoch/mean, control 0.7270,
# gate +/-0.005. Read val_wrecall next to cosine: coverage is already flat train-to-val, so
# these may be nulls. Two caveats from the plan: the shared encoders keep training, so the
# proposal FUNCTION still drifts after the freeze; and freezing puts the heads in eval mode,
# dropping their head_dropout 0.1 -- a second change that p3 does not carry.
#
# Every job passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-rec}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_proposal_early_stop"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/ma_mi_merged/proposal_early_stop_2026_09_22"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"

P1_JOBS=(msh_pes_p1_freeze_e32); P1_CONFIGS=("${CONFIG_DIR}/p1_proposal_freeze_e32.yml"); P1_TIMES=(3:00:00)
P2_JOBS=(msh_pes_p2_freeze_e64); P2_CONFIGS=("${CONFIG_DIR}/p2_proposal_freeze_e64.yml"); P2_TIMES=(3:00:00)
P3_JOBS=(msh_pes_p3_wd10);       P3_CONFIGS=("${CONFIG_DIR}/p3_proposal_wd10.yml");       P3_TIMES=(3:00:00)

case "${STAGE}" in
    rec) JOB_NAMES=("${P1_JOBS[@]}" "${P3_JOBS[@]}")
         CONFIGS=("${P1_CONFIGS[@]}" "${P3_CONFIGS[@]}")
         TIMES=("${P1_TIMES[@]}" "${P3_TIMES[@]}") ;;
    p1)  JOB_NAMES=("${P1_JOBS[@]}"); CONFIGS=("${P1_CONFIGS[@]}"); TIMES=("${P1_TIMES[@]}") ;;
    p2)  JOB_NAMES=("${P2_JOBS[@]}"); CONFIGS=("${P2_CONFIGS[@]}"); TIMES=("${P2_TIMES[@]}") ;;
    p3)  JOB_NAMES=("${P3_JOBS[@]}"); CONFIGS=("${P3_CONFIGS[@]}"); TIMES=("${P3_TIMES[@]}") ;;
    all) JOB_NAMES=("${P1_JOBS[@]}" "${P2_JOBS[@]}" "${P3_JOBS[@]}")
         CONFIGS=("${P1_CONFIGS[@]}" "${P2_CONFIGS[@]}" "${P3_CONFIGS[@]}")
         TIMES=("${P1_TIMES[@]}" "${P2_TIMES[@]}" "${P3_TIMES[@]}") ;;
    *) echo "Usage: $0 [rec|p1|p2|p3|all]" >&2; exit 1 ;;
esac

cd "${REPO_ROOT}"

# 1. Validate everything before creating any job script.
if [[ "${#JOB_NAMES[@]}" -ne "${#CONFIGS[@]}" || "${#JOB_NAMES[@]}" -ne "${#TIMES[@]}" ]]; then
    echo "JOB_NAMES, CONFIGS and TIMES must contain the same number of entries." >&2
    exit 1
fi
for config_fp in "${CONFIGS[@]}"; do
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
    if ! grep -q '^max_epochs: 128$' "${config_fp}"; then
        echo "${config_fp} is not at max_epochs: 128 -- the 3 h box will kill it mid-schedule." >&2
        exit 1
    fi
    # The 3g.40gb slice has ~40 MiB of headroom with this metric on (see
    # submit_masster_regularization.sh); it would OOM here.
    if grep -q '^compute_train_auxiliary_metrics: true$' "${config_fp}"; then
        echo "${config_fp} turns on compute_train_auxiliary_metrics -- it will OOM on" >&2
        echo "${PROFILE}. Turn it off or move the job to fir:h100-80gb." >&2
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

# 2. Generate the Fir job scripts.
for index in "${!CONFIGS[@]}"; do
    python scripts/setup_job_cc.py \
        --job_name "${JOB_NAMES[index]}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile "${PROFILE}" \
        --time "${TIMES[index]}" \
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
