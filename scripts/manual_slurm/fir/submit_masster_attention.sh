#!/bin/bash
# Generate (but do not submit) the Fir jobs for the five UNRUN arms of the 2026-09-21
# attention / budget campaign.
# Plan: 95341fce:docs/plans/masster/masster_architecture_review_todo.md, removed 2026-10-18, section 2.
#
# STAGE selects which arms are generated:
#   lead        b1 + a2 (default). The plan's co-leads. `b1` is parameter-identical to
#               `c4a_base_s1`, so it is the cleanest one-key arm in the set; `a2` is the
#               topology-only control that `a3`, `a6` and `a7` cannot be attributed without.
#   conditional a4 + a5. GATED on section 1 of the plan: submit only after the epoch-128 read
#               of the six arms that already ran, and only if at least one of `a1`/`a3`
#               clears the 0.005 floor. `a5` is read against `a3`, so `a3` must be read first.
#               The gate opened on 2026-09-22 (`a1` +0.0056), but prefer the `a5` stage below:
#               that read also made `a4` a poor bet.
#   a5          `a5_ce_pair_bias` alone -- the last unrun arm with a mechanism behind it.
#               READ IT AGAINST `a3_full_massbias` (`4fg0x5vw`), NOT against the control: it
#               carries `attention_type: full` and the mass pair bias too, and `a3` is the arm
#               that holds both of those without the CE term. `a5` - `a3` isolates
#               `intensity_ce_pair_bias`; `a5` - control does not.
#   rebalance   b4. Lowest priority and NOT a one-key arm -- it moves bond_num_layers 3 -> 1
#               while widening the intensity head (256 -> 384) and the latent set (64 -> 128),
#               so a result cannot be attributed. Split it before believing anything it says.
#   all         all five, for inspection only.
#
# NO CONTROL JOB IS GENERATED, and the six arms that already ran are deliberately absent.
# `c4a_base_s1`/`s2` are the control (mean 0.7286, the two replicates spanning 0.0007).
# `a1`, `a3`, `a6`, `a7`, `b2` and `b3` still carry `min_epochs/max_epochs: 630` on purpose so
# their config matches the run in the index -- if one of them is replicated, set the horizon at
# submission time rather than editing the config.
#
# Profile fir:h100-3g.40gb with 6 CPUs and 140G, identical to the c4a arms these are read
# against: same microbatch 512, loss_batch_size 64, num_workers 6. This lineage is NOT the
# 256 x accumulate-2 family that is sized for the 2g slice.
#
# Wall time 3:00:00 for the 128-epoch screen (~1 min/epoch on this box, so ~2.2 h). `b4` gets
# 4:00:00: the wider transformer and the doubled inducing set have never been timed, and its
# peak GPU is above the ~26.8 GiB the narrower arms measure. Read b4's real per-epoch cost off
# the first job rather than trusting this number.
#
# READING THE ARMS -- val_cos_sim_0.01_epoch/mean at matched epoch 128, never matched wall clock:
#   b1 - base   what a sum instead of a degree-normalized mean is worth in the bond head.
#   a2 - base   full attention at REDUCED capacity (8,144,981 vs 9,215,573 params); it is also
#               the control that lets a3/a6/a7 be read, since those carry the same -41 to -44%
#               intensity-head cut on top of their own key.
#   a5 - a3     what CE conditioning of the pair bias is worth given the mass pair bias.
#   The family's same-seed replicate floor is 0.003-0.005, so anything under ~0.007 at n=1 is
#   unresolved; an arm inside +/-0.005 is a `hold` at this exposure, not a rejection.
#   For a4/a5 also dump `intensity_head.mass_pair_bias.ce_weights` from the checkpoint: a flat
#   w_r(CE) is informative about CE-invariance whether or not cosine moves.
#
# Every arm passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-lead}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_attention"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/ma_mi_merged/attention_2026_09_21"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"

LEAD_JOBS=(msh_at_b1_bond_sum msh_at_a2_full_attn)
LEAD_CONFIGS=(b1_bond_sum.yml a2_full_attention.yml)
LEAD_TIMES=(3:00:00 3:00:00)

COND_JOBS=(msh_at_a4_stacked msh_at_a5_ce_pair)
COND_CONFIGS=(a4_full_massbias_prenorm_qknorm.yml a5_ce_pair_bias.yml)
COND_TIMES=(3:00:00 3:00:00)

# `a5` alone. The epoch-128 read closed the question `a4` was queued for -- stacking has now
# cost twice in this family (`a7` below both `a1` and `a6`; `e5` below `e4`) -- while `a5`
# became MORE valuable, because `a2` showed the mass pair bias is an exact null at matched
# topology (-0.0002) and CE-FiLM is worth +0.0033, making CE the live half of review item 1.
A5_JOBS=(msh_at_a5_ce_pair)
A5_CONFIGS=(a5_ce_pair_bias.yml)
A5_TIMES=(3:00:00)

REBAL_JOBS=(msh_at_b4_rebalance)
REBAL_CONFIGS=(b4_budget_rebalance.yml)
REBAL_TIMES=(4:00:00)

case "${STAGE}" in
    lead)        JOB_NAMES=("${LEAD_JOBS[@]}"); CONFIGS=("${LEAD_CONFIGS[@]}"); TIMES=("${LEAD_TIMES[@]}") ;;
    conditional) JOB_NAMES=("${COND_JOBS[@]}"); CONFIGS=("${COND_CONFIGS[@]}"); TIMES=("${COND_TIMES[@]}") ;;
    a5)          JOB_NAMES=("${A5_JOBS[@]}"); CONFIGS=("${A5_CONFIGS[@]}"); TIMES=("${A5_TIMES[@]}") ;;
    rebalance)   JOB_NAMES=("${REBAL_JOBS[@]}"); CONFIGS=("${REBAL_CONFIGS[@]}"); TIMES=("${REBAL_TIMES[@]}") ;;
    all)         JOB_NAMES=("${LEAD_JOBS[@]}" "${COND_JOBS[@]}" "${REBAL_JOBS[@]}")
                 CONFIGS=("${LEAD_CONFIGS[@]}" "${COND_CONFIGS[@]}" "${REBAL_CONFIGS[@]}")
                 TIMES=("${LEAD_TIMES[@]}" "${COND_TIMES[@]}" "${REBAL_TIMES[@]}") ;;
    *) echo "Usage: $0 [lead|conditional|a5|rebalance|all]" >&2; exit 1 ;;
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
# These five were moved to 128/128 on 2026-09-22 because the 630-epoch horizon they were
# written with is what killed the first six arms mid-schedule. Fail loudly if one drifts back.
for config_fn in "${CONFIGS[@]}"; do
    if ! grep -q '^max_epochs: 128$' "${CONFIG_DIR}/${config_fn}"; then
        echo "${CONFIG_DIR}/${config_fn} is not at max_epochs: 128 -- the 3 h box will kill it mid-schedule." >&2
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
echo "Control is the EXISTING c4a_base_s1/s2 pair; do not re-run it, and do not regenerate the"
echo "six arms that already ran -- they are still at 630 epochs so their config matches the index."
echo "Read at matched epoch 128. Under ~0.007 at n=1 is inside this family's replicate floor."
