#!/bin/bash
# Generate (but do not submit) the Fir jobs for the 2026-09-22 candidate-token campaign.
# Plan: docs/plans/masshatter/masshatter_cut_set_features_plan.md.
# Measurement behind it: docs/findings/masshatter/masshatter_candidate_cut_descriptors_2026_09_22.md.
#
# STAGE selects which arms are generated:
#   all       all seven arms (default).
#   features  e1, e4, e5 -- the cut-description ladder. These three answer one question and are
#             the stage to run first if only part of the campaign can be afforded.
#   dropout   e2, e3 -- the component-pool dropout rate ladder. Read the note below first: the
#             gap this arm targets is NOT logged by this lineage.
#   molfeats  c4a_ring_topology + e6 -- the 2026-09-22 graph features, and the same features with
#             the cut-bond block that lets the new BOND features reach the intensity token.
#             Submit these two together or neither: e6 alone confounds the feature set with the
#             routing. c4a_ring_topology has never been run (W&B, 2026-09-22).
#
# NO CONTROL JOB IS GENERATED. `c4a_base_s1`/`s2` already ran and are the control (mean 0.7286,
# the two replicates spanning 0.0007). Every arm here is a one-key diff from `c4a_base_s1` at the
# same seed, except e5, which is deliberately e1+e4.
#
# Profile fir:h100-3g.40gb with 6 CPUs and 140G, identical to the c4a arms: same batch 512,
# loss_batch_size 64, num_workers 6, and `compute_train_auxiliary_metrics: false` to keep peak
# GPU near 26.8 GiB. The feature blocks add at most 32 columns to a 1,285-dim token (18 for the
# cut-bond sum and mean at the base bond width, 32 at e6's wider one, 2 for adjacency) and the
# dropout adds no parameters, so none of the seven changes the memory shape or needs re-sizing.
#
# Wall time 3:00:00 matches the c4a arms' 128-epoch budget, itself set by the 2026-09-21
# saturation finding (val cosine plateaus by ~epoch 100; the last 529 of 630 epochs buy +0.0017).
#
# READING THE ARMS -- val_cos_sim_0.01_epoch/mean at matched epochs, never matched wall clock:
#   e1 - base   what the severed bonds' own features are worth.
#   e4 - base   what cut adjacency is worth alone.
#   e5 - e1     what adjacency is worth GIVEN the bond sum. This is the comparison the 0.489
#               residual-ambiguity measurement is about; e5 - base is not.
#   If e5 lands at e1 or at e4 rather than above both, the cut description saturates.
#   The family's same-seed replicate floor is 0.003-0.005, so treat anything under ~0.007 at
#   n=1 as unresolved and buy a replicate before believing it.
#
# GOTCHA for e2/e3. These are regularizers, so the quantity that should decide them is the
# train/validation gap -- and this lineage does not log train cosine (`compute_train_auxiliary
# _metrics: false`, false in 232 of 232 historical runs). What IS logged is train/val LOSS, so
# read the loss gap, or add
#   --script_kwargs ... override=compute_train_auxiliary_metrics=True
# and re-check that the job still fits the 3g slice before trusting the timing.
#
# Every arm passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masshatter_candidate_token"
JOB_DIR="${REPO_ROOT}/jobs"
CAMPAIGN_DIR="config/massshatter/ma_mi_merged/candidate_token_2026_09_22"
MERGED_DIR="config/massshatter/ma_mi_merged"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"

FEATURE_JOBS=(msh_ct_e1_cut_bond msh_ct_e4_adjacency msh_ct_e5_bond_adj)
FEATURE_CONFIGS=(
    "${CAMPAIGN_DIR}/e1_cut_bond_feats.yml"
    "${CAMPAIGN_DIR}/e4_cut_adjacency.yml"
    "${CAMPAIGN_DIR}/e5_cut_bond_plus_adjacency.yml"
)

DROPOUT_JOBS=(msh_ct_e2_ccdrop01 msh_ct_e3_ccdrop02)
DROPOUT_CONFIGS=(
    "${CAMPAIGN_DIR}/e2_cc_atom_drop01.yml"
    "${CAMPAIGN_DIR}/e3_cc_atom_drop02.yml"
)

MOLFEAT_JOBS=(msh_ct_ring_topology msh_ct_e6_ring_cut_bond)
MOLFEAT_CONFIGS=(
    "${MERGED_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_ring_topology.yml"
    "${CAMPAIGN_DIR}/e6_ring_topology_cut_bond.yml"
)

case "${STAGE}" in
    features) JOB_NAMES=("${FEATURE_JOBS[@]}"); CONFIGS=("${FEATURE_CONFIGS[@]}") ;;
    dropout)  JOB_NAMES=("${DROPOUT_JOBS[@]}"); CONFIGS=("${DROPOUT_CONFIGS[@]}") ;;
    molfeats) JOB_NAMES=("${MOLFEAT_JOBS[@]}"); CONFIGS=("${MOLFEAT_CONFIGS[@]}") ;;
    all)      JOB_NAMES=("${FEATURE_JOBS[@]}" "${DROPOUT_JOBS[@]}" "${MOLFEAT_JOBS[@]}")
              CONFIGS=("${FEATURE_CONFIGS[@]}" "${DROPOUT_CONFIGS[@]}" "${MOLFEAT_CONFIGS[@]}") ;;
    *) echo "Usage: $0 [all|features|dropout|molfeats]" >&2; exit 1 ;;
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
echo "Read e5 against e1, not against the base: adjacency's measured value is residual given the"
echo "cut-bond sum. Anything under ~0.007 at n=1 is inside this family's replicate floor."
echo "e2/e3 target the train/validation gap, which this lineage logs only as a LOSS gap."
echo "e6 is read against c4a_ring_topology, which isolates routing the new bond features into"
echo "the intensity token from having them in the encoder at all."
