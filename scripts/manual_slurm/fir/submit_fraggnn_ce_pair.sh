#!/bin/bash
# Generate (but do not submit) the Fir jobs for the FraGNNet FT contender-pair Stage A pilot,
# 2026-09-23.
# Plan: docs/plans/fragnnet/ft_contender_pair_minimal_training_plan.md (Run matrix, Stage A)
#       docs/findings/fragment_intensity_ranking_findings.md
#
# Question: does the online adjacent-CE winner-switch BCE (`ce_pair_loss_weight`) improve held-out
# spectra beyond a matched NodeMLP? The primary contrast is pair arm vs SAMPLER control; the
# original-sampler control only reports the net deployable effect.
#
# STAGE selects which jobs are generated:
#   all      (default) all four arms. Submit them together.
#   controls c0 + c1 only.
#   pair     p1 + p2 only. Useless without c1.
#
# Arms, all from fraggnn_d3_ma_mi_nist20_amolc_all.yml (the `jnokddr9` config), seed 42:
#   c0  original sampler, pair weight 0                      100 epochs  (historical budget)
#   c1  ce_context sampler, pair weight 0                     48 epochs
#   p1  ce_context sampler, pair weight 1.5 (lambda_weak)     48 epochs
#   p2  ce_context sampler, pair weight 6.0 (lambda_medium)   48 epochs
# lambda = target initial pair/primary gradient ratio (5% / 20%) / g, with g = 0.0344 measured
# at initialization on ~406-spectrum ce_context steps (see the configs for the full provenance).
# Every arm, controls included, carries target_rel_ints_thresh 0.01 and
# log_adduct_inst_ce_metrics true. The first is required by the pair loss AND filters the train
# and validation targets, so `jnokddr9` (thresh 0.0) is NOT a valid control for these runs.
#
# BUDGET. The original sampler (simple_group_sampler, ~5 spectra per group per epoch) shows
# 138,875 of 287,721 train spectra per epoch at 542 steps x 256 spectra. ce_context emits all
# 287,721 at 709 steps x ~406 spectra (steps fire once >=256 spectra accumulate). 48 ce_context
# epochs = 13.81M spectra vs c0's 13.89M: matched on spectra seen and wall clock, not on steps
# (34.0k vs 54.2k). c1/p1/p2 are matched to each other on everything but the pair weight.
#
# EDGE BUDGET. ce_context requires frag_edge batching. 1,500,000 edges gives a per-batch node max
# of 281,988 against the original sampler's 322,399 (measured on the train split), so peak memory
# stays at or below c0's. `jnokddr9` peaked at 34.2 GB allocated on an L40S, which fits the
# 37.25 GiB of a 3g.40gb slice.
#
# WALL TIME is a margin, not a measurement: `jnokddr9` took 7.8 h on a Vulcan L40S and no
# FraGNNet run of this config has been timed on a Fir 3g.40gb slice. c1 sees the same spectra as
# c0, so 16:00:00 for both. The pair loss adds +17.5% per train step (0.357 vs 0.303 s, same
# weights and batches, local RTX 2070) -- CPU-side target discovery at ~0.8 ms per spectrum,
# ~4 min per ce_context epoch -- so p1/p2 get 20:00:00.
#
# READING IT -- best and last val_cos_sim_0.01 (and cos_hun_inst_tol) at the final epoch:
#   p1/p2 vs c1   the pilot's answer. The plan's promotion bar (+0.005 over three seeds, paired
#                 intervals excluding zero) is for Stage B, not this one-seed pilot.
#   c1 vs c0      the sampler effect. Nonzero is expected; it is why c1 exists.
#   per-CE bins   the effect should sit at high FT NCE, where base-peak accuracy falls 92% -> 42%.
#   ce_pair_eligible_endpoints   must be nonzero in p1/p2 train logs, or the arm tested nothing.
#
# Every job passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/fraggnn_ce_pair"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/exp_fraggnn_ma_mi/ce_pair_2026_09_23"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"

CTRL_JOBS=(fg_cep_c0_orig_ctrl fg_cep_c1_sampler_ctrl)
CTRL_CONFIGS=("${CONFIG_DIR}/c0_orig_ctrl.yml" "${CONFIG_DIR}/c1_sampler_ctrl.yml")
CTRL_TIMES=(16:00:00 16:00:00)

PAIR_JOBS=(fg_cep_p1_pair_weak fg_cep_p2_pair_medium)
PAIR_CONFIGS=("${CONFIG_DIR}/p1_pair_weak.yml" "${CONFIG_DIR}/p2_pair_medium.yml")
PAIR_TIMES=(20:00:00 20:00:00)

case "${STAGE}" in
    all)      JOB_NAMES=("${CTRL_JOBS[@]}" "${PAIR_JOBS[@]}")
              CONFIGS=("${CTRL_CONFIGS[@]}" "${PAIR_CONFIGS[@]}")
              TIMES=("${CTRL_TIMES[@]}" "${PAIR_TIMES[@]}") ;;
    controls) JOB_NAMES=("${CTRL_JOBS[@]}"); CONFIGS=("${CTRL_CONFIGS[@]}"); TIMES=("${CTRL_TIMES[@]}") ;;
    pair)     JOB_NAMES=("${PAIR_JOBS[@]}"); CONFIGS=("${PAIR_CONFIGS[@]}"); TIMES=("${PAIR_TIMES[@]}") ;;
    *) echo "Usage: $0 [all|controls|pair]" >&2; exit 1 ;;
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
    # The pair loss requires this floor and it also filters the evaluation target, so an arm
    # without it is scored against a different validation spectrum.
    if ! grep -q '^target_rel_ints_thresh: 0.01$' "${config_fp}"; then
        echo "${config_fp} does not set target_rel_ints_thresh: 0.01 -- it would not be" >&2
        echo "scored against the same validation target as the other arms." >&2
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
echo
echo "Submit c1 with the pair arms: pair vs sampler control is the pilot's only clean contrast."
