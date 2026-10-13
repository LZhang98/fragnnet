#!/bin/bash
# Generate (but do not submit) the Fir jobs for the 2026-09-23 fragment-selection arms.
# Plan: docs/plans/masster/masster_fragment_marginal_selection_plan.md.
#
#   f1_mask_order_u128        naive U128: the unique target alone drops to 128 (mask order).
#   f2_marginal_k128_pool256  marginal selection: keep 128 of the control's U256 pool by
#                             closed-form release probability; proposal loss on the whole pool.
#   f3_marginal_k128_pool512  f2 from a U512 pool at the same K and E4096.
#
# STAGE (first argument) selects which arms are generated:
#   arms    f1, f2, f3 (default) -- submitted 2026-09-23 as uorwhapr, u7youmib, 9nxd8y8y.
#   ladder  f4_mask_order_u64, f5_mask_order_u128_ctlmon -- the budget ladder, run with the
#           CONTROL'S monitoring (no coverage/cut-bucket metrics) so hours per epoch compare
#           directly with c4a_base. f5 is f1 without the extra metrics: the clean U128 speed.
#
# NO CONTROL JOB IS GENERATED. `c4a_base_s1`/`s2` already ran and are the control (tail mean
# 0.7270, the two replicates spanning 0.0007). Every arm is a diff of `c4a_base_s1` at the same
# seed; the arms also switch on the (monitoring-only) candidate coverage and cut-bucket metrics,
# which the control never logged, so compare coverage among the arms and cosine against the
# control.
#
# Profile fir:h100-3g.40gb, identical to the c4a arms: batch 512, loss_batch_size 64 and
# `compute_train_auxiliary_metrics: false`, which kept the control near 26.8 GiB. K128 halves the
# intensity-head tokens; f3 doubles the enumerated pool and its pathway tensors, so it is the arm
# to watch for memory.
#
# Wall time 3:00:00 matches the c4a arms' 128-epoch budget.
#
# READING THE ARMS -- val_cos_sim_0.01_epoch/mean, 118-127 tail mean, gate +/-0.005:
#   f1 - control   what a mask-order U128 budget costs on current code.
#   f2 - f1        what ranking by the fragment's own probability buys at the same K.
#   f2 - control   whether K128 marginal recovers U256; pair it with hours per epoch.
#   f3 - f2        whether a deeper pool helps at the same K.
# Coverage: val_candidate_wrecall (what the intensity head sees) against
# val_pool_candidate_wrecall (the ceiling for any selection from the pool), per cut bucket.
#
# Every arm passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-arms}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_fragment_selection"
JOB_DIR="${REPO_ROOT}/jobs"
CAMPAIGN_DIR="config/masster/ma_mi_merged/fragment_selection_2026_09_23"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"

case "${STAGE}" in
    arms)
        JOB_NAMES=(msh_fs_f1_mask_u128 msh_fs_f2_marg_k128_p256 msh_fs_f3_marg_k128_p512)
        CONFIGS=(
            "${CAMPAIGN_DIR}/f1_mask_order_u128.yml"
            "${CAMPAIGN_DIR}/f2_marginal_k128_pool256.yml"
            "${CAMPAIGN_DIR}/f3_marginal_k128_pool512.yml"
        )
        ;;
    ladder)
        JOB_NAMES=(msh_fs_f4_mask_u64 msh_fs_f5_mask_u128_ctlmon)
        CONFIGS=(
            "${CAMPAIGN_DIR}/f4_mask_order_u64.yml"
            "${CAMPAIGN_DIR}/f5_mask_order_u128_ctlmon.yml"
        )
        ;;
    *) echo "Usage: $0 [arms|ladder]" >&2; exit 1 ;;
esac

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
echo "Read f2 against f1 for the ranking, and against the control together with hours per epoch."
