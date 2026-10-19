#!/bin/bash
# Generate (but do not submit) the Fir jobs for the one open regularization question:
# does weight_decay 0.1 narrow the train/validation gap at full screen length without
# suppressing train cosine?
# Plans: docs/plans/masster/masster_regularization_sweep_plan.md
#        95341fce:docs/findings/masster/masster_overfit_readout_2026_09_22.md, removed 2026-10-18, section 7.2 item 1 / section 8 item 3
#
# STAGE selects which jobs are generated:
#   wd10     (default) BOTH jobs -- the arm and its train-metric control. Submit them together.
#   arm      c4d_wd10 alone. Only useful if c4a_base_s1_trainaux has already run.
#   control  c4a_base_s1_trainaux alone. This is also recommendation 1 of the overfit read-out
#            ("turn on the two metrics that make this answerable on the next control"), so it
#            is worth having even if the wd10 arm is dropped.
#   transfer the 2026-09-23 ranking-transfer screen: the same control plus t2 below.
#   t2       t2_ema998 alone (EMA of the weights, decay 0.998).
#
# RANKING-TRANSFER ARMS (2026-09-23, plan 95341fce:docs/plans/masster/masster_ranking_transfer_plan.md, removed 2026-10-18).
# The val ranking error is a generalization failure (train ranking gap 0.011 vs val 0.121), so
# this arm tries to make the ranking transfer rather than add a ranking loss. It is one key off
# c4a_base_s1_trainaux, the SAME control job as the wd10 stage: submit that control once, whichever
# stages you run. Read t2's train/val gap with care -- its train metrics measure the raw weights
# while validation measures the average. The boundary-block dropout arm t1 was dropped on
# 2026-09-24: molecule identity reaches every head input, so blocking one can reroute the lookup.
#
# WHY TWO JOBS. The question is about the train/validation gap, so it needs a TRAIN cosine.
# `compute_train_auxiliary_metrics` is False in c4a_base_s1/s2, so the existing control supplies
# a matched val cosine and no matched train cosine at all. `ctrl_a`/`ctrl_b` (2026-09-19) do have
# the key on, but they ran before the three candidate/h-transfer defaults flipped on 2026-09-20,
# so they are a different architecture and cannot serve as this control. Resolved-config diffs,
# verified 2026-09-22:
#   c4a_base_s1_trainaux - c4a_base_s1  = compute_train_auxiliary_metrics       (1 key)
#   c4d_wd10             - c4a_base_s1  = that key + weight_decay               (2 keys)
#   c4d_wd10 - c4a_base_s1_trainaux     = weight_decay 0.01 -> 0.1              (1 key)  <-- the read
# The control also reuses seed 397623, so its val cosine is a third same-seed replicate of
# c4a_base_s1 and a free check that turning the metric on does not move the optimization.
#
# PROFILE fir:h100-80gb, NOT the 3g.40gb slice the rest of this lineage uses, and this is the
# whole reason these jobs cost double. Measured 2026-09-22 from the W&B system stream:
#   compute_train_auxiliary_metrics ON   peak 36.2-37.2 GiB  (`ri3zm215`, `kalc4mzo`)
#   compute_train_auxiliary_metrics OFF  peak 30.4 GiB       (`c4a_base_s1`)
# A 3g.40gb slice is 40 GB = 37.25 GiB, so the ON case leaves ~40 MiB before the CUDA context is
# even counted. Do not "save" a box by moving these to 3g without turning the key back off --
# the run will OOM and you lose the box anyway. Slice size does not affect the metric, only
# throughput, and every read here is at matched epoch, never matched wall clock.
#
# WALL TIME 3:00:00. The key costs +6% per epoch (0.91 vs 0.86 min/epoch measured on the same
# runs), so 128 epochs is ~1.95 h. The 2026-09-19 wd10 screen died at epoch 65 because it had a
# 1 h box, not because of this key.
#
# READING IT -- all at matched epoch 128, on the tail mean of epochs 118-127:
#   val_cos_sim_0.01_epoch/mean   c4d_wd10 vs c4a_base_s1_trainaux. The family's tail-mean
#                                 reproducibility floor is 0.0004 and the cross-seed control
#                                 spread is 0.0007, but the standing gate is +/-0.005.
#   train cosine vs val cosine    the actual question. The 2026-09-19 screen's claim is that
#                                 wd 0.1 narrows this gap WITHOUT suppressing train cosine;
#                                 a narrowed gap that also drops train cosine is underfitting,
#                                 which is what `drop03`, `b3_wd_groups` and `e2` all did.
#   val_wrecall                   expected flat (spread 0.0006 over eleven runs). If it moves,
#                                 something other than allocation changed.
# NOTE: narrowing the gap is not the objective and has never predicted cosine in this family.
# Three arms have now narrowed it and lost. This arm is run to CLOSE the question, not because
# the prior favours it.
#
# Every job passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-wd10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_regularization"
JOB_DIR="${REPO_ROOT}/jobs"
MS_DIR="config/masster/ma_mi_merged"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-80gb"

ARM_JOBS=(msh_rg_c4d_wd10)
ARM_CONFIGS=("${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4d_wd10.yml")
ARM_TIMES=(3:00:00)

CTRL_JOBS=(msh_rg_base_s1_trainaux)
CTRL_CONFIGS=("${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_base_s1_trainaux.yml")
CTRL_TIMES=(3:00:00)

TRANSFER_DIR="${MS_DIR}/ranking_transfer_2026_09_23"
T2_JOBS=(msh_rt_t2_ema998)
T2_CONFIGS=("${TRANSFER_DIR}/t2_ema998.yml")
T2_TIMES=(3:00:00)

case "${STAGE}" in
    wd10)    JOB_NAMES=("${CTRL_JOBS[@]}" "${ARM_JOBS[@]}")
             CONFIGS=("${CTRL_CONFIGS[@]}" "${ARM_CONFIGS[@]}")
             TIMES=("${CTRL_TIMES[@]}" "${ARM_TIMES[@]}") ;;
    arm)     JOB_NAMES=("${ARM_JOBS[@]}"); CONFIGS=("${ARM_CONFIGS[@]}"); TIMES=("${ARM_TIMES[@]}") ;;
    control) JOB_NAMES=("${CTRL_JOBS[@]}"); CONFIGS=("${CTRL_CONFIGS[@]}"); TIMES=("${CTRL_TIMES[@]}") ;;
    transfer) JOB_NAMES=("${CTRL_JOBS[@]}" "${T2_JOBS[@]}")
             CONFIGS=("${CTRL_CONFIGS[@]}" "${T2_CONFIGS[@]}")
             TIMES=("${CTRL_TIMES[@]}" "${T2_TIMES[@]}") ;;
    t2)      JOB_NAMES=("${T2_JOBS[@]}"); CONFIGS=("${T2_CONFIGS[@]}"); TIMES=("${T2_TIMES[@]}") ;;
    *) echo "Usage: $0 [wd10|arm|control|transfer|t2]" >&2; exit 1 ;;
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
done
# The 630-epoch horizon these configs were written with is what killed the first six attention
# arms mid-schedule. Both were moved to 128/128 on 2026-09-22; fail loudly if one drifts back.
for config_fp in "${CONFIGS[@]}"; do
    if ! grep -q '^max_epochs: 128$' "${config_fp}"; then
        echo "${config_fp} is not at max_epochs: 128 -- the 3 h box will kill it mid-schedule." >&2
        exit 1
    fi
done
# The 80gb profile below is chosen FOR this key. If it is off, these jobs are paying double for
# nothing and belong on fir:h100-3g.40gb instead.
for config_fp in "${CONFIGS[@]}"; do
    if ! grep -q '^compute_train_auxiliary_metrics: true$' "${config_fp}"; then
        echo "${config_fp} does not set compute_train_auxiliary_metrics: true -- either turn it" >&2
        echo "on, or move this job to fir:h100-3g.40gb and stop paying for the full card." >&2
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

# 2. Generate the Fir 80gb job scripts.
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
echo "These are fir:h100-80gb jobs (price 12.2 each, 2x the 3g.40gb slice) because"
echo "compute_train_auxiliary_metrics peaks at ~37 GiB. Review, then submit from a Fir login node:"
for job_name in "${JOB_NAMES[@]}"; do
    echo "  bash -n ${SCRIPT_DIR}/${job_name}.sh"
done
for job_name in "${JOB_NAMES[@]}"; do
    echo "  sbatch ${SCRIPT_DIR}/${job_name}.sh"
done
echo
echo "Submit the control with the arm. Without it there is no matched train cosine at 128"
echo "epochs and the arm answers only the val-cosine question, which the 2026-09-19 screen"
echo "already left inside +/-0.007."
