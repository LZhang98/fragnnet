#!/bin/bash
# Generate (but do not submit) the Fir jobs for the 2026-09-22 intensity-escape campaign.
# Plan: 95341fce:docs/plans/masster/masster_supervised_escape_plan.md, removed 2026-10-18.
#
# STAGE selects which arms are generated:
#   ladder  e0 + e1, both seeds -- the full 4-job one-knob ladder. This is the default and the
#           only stage the plan authorizes.
#   control e0 only, both seeds -- generate these first if the 2g slice needs re-checking for
#           the independent output before spending the feature arms.
#
# The ladder exists because the escape cannot be tested against h1 directly: h1 keeps
# `masster_intensity_output: probability` to dodge the CE-vs-direct-loss bundle confound,
# and the escape REQUIRES `independent`. So e0 isolates the output mode against h1, and e1
# isolates the escape against e0. Read e1 against e0, never against h1.
#
# Profile is fir:h100-2g.20gb, matching the loss-objective lineage these are derived from:
# microbatch 256 x accumulate 2 (effective 512), loss_batch_size 32, num_workers 3 for that
# profile's 3 CPUs. The escape adds 257 parameters (a 256-wide linear plus bias) to a 9.22M
# model and one appended coordinate per spectrum in the loss, so it does not change the memory
# shape and needs no re-sizing.
#
# Wall time 4:00:00 matches the h1 arm this is derived from, whose 128-epoch budget is set by
# the 2026-09-21 saturation finding. Read the real per-epoch cost off e0 and tighten the rest.
#
# n=2 seeds per arm is deliberate: this family's replicate floor is 0.003-0.005, so an n=1 arm
# cannot resolve what it produces.
#
# Every arm passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-ladder}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_intensity_escape"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/ma_mi_merged/intensity_escape_2026_09_22"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"

CONTROL_JOBS=(msh_esc_e0_indep_s1 msh_esc_e0_indep_s2)
CONTROL_CONFIGS=(e0_independent_s1.yml e0_independent_s2.yml)
CONTROL_TIMES=(4:00:00 4:00:00)

FEATURE_JOBS=(msh_esc_e1_escape_s1 msh_esc_e1_escape_s2)
FEATURE_CONFIGS=(e1_escape_s1.yml e1_escape_s2.yml)
FEATURE_TIMES=(4:00:00 4:00:00)

case "${STAGE}" in
    control) JOB_NAMES=("${CONTROL_JOBS[@]}"); CONFIGS=("${CONTROL_CONFIGS[@]}")
             TIMES=("${CONTROL_TIMES[@]}") ;;
    ladder)  JOB_NAMES=("${CONTROL_JOBS[@]}" "${FEATURE_JOBS[@]}")
             CONFIGS=("${CONTROL_CONFIGS[@]}" "${FEATURE_CONFIGS[@]}")
             TIMES=("${CONTROL_TIMES[@]}" "${FEATURE_TIMES[@]}") ;;
    *) echo "Usage: $0 [ladder|control]" >&2; exit 1 ;;
esac

cd "${REPO_ROOT}"

# 1. Validate everything before creating any job script.
if [[ "${#JOB_NAMES[@]}" -ne "${#CONFIGS[@]}" || "${#JOB_NAMES[@]}" -ne "${#TIMES[@]}" ]]; then
    echo "JOB_NAMES, CONFIGS and TIMES must contain the same number of entries." >&2
    exit 1
fi
for config_fn in "${CONFIGS[@]}"; do
    if [[ ! -f "${CONFIG_DIR}/${config_fn}" ]]; then
        echo "Missing config: ${CONFIG_DIR}/${config_fn}" >&2
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

# 2. Generate the Fir 2g.20gb job scripts.
for index in "${!CONFIGS[@]}"; do
    python scripts/setup_job_cc.py \
        --job_name "${JOB_NAMES[index]}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile fir:h100-2g.20gb \
        --time "${TIMES[index]}" \
        --cc_account "${ACCOUNT}" \
        --cc_email "${EMAIL}" \
        --script_kwargs w=online c="${CONFIG_DIR}/${CONFIGS[index]}" j="${JOB_NAMES[index]}" \
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
echo "Read e1 against e0 on val_cos_sim_0.01_epoch/mean ONLY."
echo "spec_peak_distance is NOT comparable across the escape: it gains a coordinate the model"
echo "can trivially get right, so it is guaranteed to fall and means nothing."
echo "Check escape_share moved off its 0.1 initialization and tracks 1 - wrecall (~0.109)."
echo "If wrecall/recall/precision differ between e0 and e1, something is wrong, not better."
