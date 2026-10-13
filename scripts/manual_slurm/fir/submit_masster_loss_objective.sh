#!/bin/bash
# Generate (but do not submit) the Fir jobs for the 2026-09-21 loss-objective campaign.
# Plan: docs/plans/masster/masster_nonprobabilistic_intensity_plan.md, "Prepared arms".
#
# STAGE selects which arms are generated:
#   step1  h1/h2/h3 + c1 -- the only arms the plan authorizes now.
#   gated  h4/h5/h6/h7   -- the leak sweep and the CE auxiliary. The plan gates these on h1
#                           showing the predicted recovery improvement AND leaving a measured
#                           false-intensity problem. Do not generate these before h1 reports.
#   all    both, for inspection only.
#
# Profile is fir:h100-2g.20gb, NOT the 3g.40gb house default. This lineage is sized for the 2g
# slice: microbatch 256 x accumulate 2 (effective 512), loss_batch_size 32, and num_workers 3 to
# match that profile's 3 CPUs. ctrl_2g measured a 12.7 GiB peak in the ~19.5 GiB usable slice.
# Running it on 3g would not change a single number, only the price and the like-for-like with
# the controls the arms are read against.
#
# Wall times are derived from the measured 51 s/epoch for this exact shape on 2g
# (docs/findings/masster/masster_campaign_readout_2026_09_21.md, run ppk9k8l9 = 68 epochs/hour):
#   128 epochs x 51 s ~= 109 min for cross_entropy.
# The direct-loss arms carry an older 1.46x penalty that was attributed to the unbatched
# per-spectrum loop in sparse_peak_list_distance -- that loop is now chunked, so 1.46x is an
# UPPER bound and these walls are deliberately loose. h3 is looser still: mass-Hungarian at
# loss_batch_size 8 means more chunks plus the assignment solve, and it has never been timed.
# Read the real per-epoch cost off the first arm and tighten the rest.
#
# Every arm passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-step1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_loss_objective"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/ma_mi_merged/loss_objective_2026_09_21"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"

STEP1_JOBS=(msh_lo_h1_hel_closest msh_lo_h2_cos_closest msh_lo_h3_hel_hungarian msh_lo_c1_cond_ce)
STEP1_CONFIGS=(h1_hellinger_closest.yml h2_cosine_closest.yml h3_hellinger_hungarian.yml
               c1_conditional_cross_entropy.yml)
STEP1_TIMES=(4:00:00 4:00:00 6:00:00 3:00:00)

GATED_JOBS=(msh_lo_h4_leak_0p3 msh_lo_h5_leak_1 msh_lo_h6_leak_3 msh_lo_h7_ce_aux_0p1)
GATED_CONFIGS=(h4_hellinger_leak_0p3.yml h5_hellinger_leak_1.yml h6_hellinger_leak_3.yml
               h7_hellinger_ce_aux_0p1.yml)
GATED_TIMES=(4:00:00 4:00:00 4:00:00 4:00:00)

case "${STAGE}" in
    step1) JOB_NAMES=("${STEP1_JOBS[@]}"); CONFIGS=("${STEP1_CONFIGS[@]}"); TIMES=("${STEP1_TIMES[@]}") ;;
    gated) JOB_NAMES=("${GATED_JOBS[@]}"); CONFIGS=("${GATED_CONFIGS[@]}"); TIMES=("${GATED_TIMES[@]}") ;;
    all)   JOB_NAMES=("${STEP1_JOBS[@]}" "${GATED_JOBS[@]}")
           CONFIGS=("${STEP1_CONFIGS[@]}" "${GATED_CONFIGS[@]}")
           TIMES=("${STEP1_TIMES[@]}" "${GATED_TIMES[@]}") ;;
    *) echo "Usage: $0 [step1|gated|all]" >&2; exit 1 ;;
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
