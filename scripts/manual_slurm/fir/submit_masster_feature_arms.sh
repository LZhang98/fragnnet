#!/bin/bash
# Generate (but do not submit) the Fir jobs for the feature arms that are prepared and have
# never run. W&B audit on 2026-09-22 found 5 of 14 prepared feature arms submitted; these are
# the other 9, plus an optional replication stage.
#
# STAGE selects which arms are generated:
#   attribution  the five c4a_* bundle-attribution arms. project-state lists these as "Next".
#                Each is a one-key diff from c4a_base_s1 at the same seed 397623 (bundle_off is
#                deliberately three keys: it turns the whole 2026-09-20 bundle off at once).
#   glacier      the three GLACIER-parity MASSter arms. The handoff question "do they have
#                more features?" is unmeasured on the feature side; these measure it.
#   fraggnn      the FraGNNet GLACIER graph-feature-parity arm. DIFFERENT FAMILY: model_type
#                frag_gnn, 100 epochs, seed 42, 60 keys different from the MASSter base. It
#                is NOT read against c4a_base -- it needs its own FraGNNet control, and it
#                carries its own 6 h wall rather than the 3 h used for the MASSter arms.
#   replicate    second seeds for the two arms sitting at ~3 sigma that the epoch-128 read-out
#                held against a +/-0.005 gate: e4_cut_adjacency (+0.0031) and a2_full_attention
#                (+0.0030). Measured tail-mean replicate spread across four pairs is
#                0.0004-0.0014, so the gate, not the data, is what holds them. These reuse the
#                existing configs with --override seed=811237 (verified to change only `seed`),
#                matching c4a_base_s2's seed so the pair is read the same way as the control.
#   all          every stage except fraggnn, which has a different control and wall time.
#
# Profile is fir:h100-3g.40gb, the house default AND what c4a_base_s1/s2 ran on. The control is
# only like-for-like on the same slice, so do not move these to 2g.
#
# Wall time 3:00:00 for the MASSter arms: the configs are min_epochs=max_epochs=128 and
# c4a_base finished 127 epochs inside this wall at the measured ~70 epochs/hour (~1.8 h).
#
# Wall time 12:00:00 for fraggnn: the 6:00:00 box killed m8qgi0ft (2026-09-23) at epoch 77 of
# max_epochs 100 after 5.96 h (~13 epochs/hour, no memory growth), so 100 epochs needs ~7.7 h.
# 12 h is ~1.5x headroom; a rerun with the same j= resumes from its last.ckpt.
#
# Every arm passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.

set -euo pipefail

STAGE="${1:-attribution}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_feature_arms"
JOB_DIR="${REPO_ROOT}/jobs"
MS_DIR="config/masster/ma_mi_merged"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"

ATTR_JOBS=(msh_fa_bundle_off msh_fa_no_acharge msh_fa_no_bbe msh_fa_no_charge_feat msh_fa_no_mass_feats)
ATTR_CONFIGS=(
    "${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_bundle_off.yml"
    "${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_no_acharge.yml"
    "${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_no_bbe.yml"
    "${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_no_charge_feat.yml"
    "${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_no_mass_feats.yml")
ATTR_TIMES=(3:00:00 3:00:00 3:00:00 3:00:00 3:00:00)
ATTR_OVERRIDES=("" "" "" "" "")

GLACIER_JOBS=(msh_fa_ring_topology msh_fa_complement_comp msh_fa_e6_ring_cutbond)
GLACIER_CONFIGS=(
    "${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_ring_topology.yml"
    "${MS_DIR}/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_complement_composition.yml"
    "${MS_DIR}/candidate_token_2026_09_22/e6_ring_topology_cut_bond.yml")
GLACIER_TIMES=(3:00:00 3:00:00 3:00:00)
GLACIER_OVERRIDES=("" "" "")

FRAGGNN_JOBS=(fg_fa_graph_feature_parity)
FRAGGNN_CONFIGS=("config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_graph_feature_parity.yml")
FRAGGNN_TIMES=(12:00:00)
FRAGGNN_OVERRIDES=("")

REPLICATE_JOBS=(msh_fa_e4_cut_adjacency_s2 msh_fa_a2_full_attention_s2)
REPLICATE_CONFIGS=(
    "${MS_DIR}/candidate_token_2026_09_22/e4_cut_adjacency.yml"
    "${MS_DIR}/attention_2026_09_21/a2_full_attention.yml")
REPLICATE_TIMES=(3:00:00 3:00:00)
REPLICATE_OVERRIDES=("seed=811237" "seed=811237")

case "${STAGE}" in
    attribution) JOB_NAMES=("${ATTR_JOBS[@]}"); CONFIGS=("${ATTR_CONFIGS[@]}")
                 TIMES=("${ATTR_TIMES[@]}"); OVERRIDES=("${ATTR_OVERRIDES[@]}") ;;
    glacier)     JOB_NAMES=("${GLACIER_JOBS[@]}"); CONFIGS=("${GLACIER_CONFIGS[@]}")
                 TIMES=("${GLACIER_TIMES[@]}"); OVERRIDES=("${GLACIER_OVERRIDES[@]}") ;;
    fraggnn)     JOB_NAMES=("${FRAGGNN_JOBS[@]}"); CONFIGS=("${FRAGGNN_CONFIGS[@]}")
                 TIMES=("${FRAGGNN_TIMES[@]}"); OVERRIDES=("${FRAGGNN_OVERRIDES[@]}") ;;
    replicate)   JOB_NAMES=("${REPLICATE_JOBS[@]}"); CONFIGS=("${REPLICATE_CONFIGS[@]}")
                 TIMES=("${REPLICATE_TIMES[@]}"); OVERRIDES=("${REPLICATE_OVERRIDES[@]}") ;;
    all)         JOB_NAMES=("${ATTR_JOBS[@]}" "${GLACIER_JOBS[@]}" "${REPLICATE_JOBS[@]}")
                 CONFIGS=("${ATTR_CONFIGS[@]}" "${GLACIER_CONFIGS[@]}" "${REPLICATE_CONFIGS[@]}")
                 TIMES=("${ATTR_TIMES[@]}" "${GLACIER_TIMES[@]}" "${REPLICATE_TIMES[@]}")
                 OVERRIDES=("${ATTR_OVERRIDES[@]}" "${GLACIER_OVERRIDES[@]}" "${REPLICATE_OVERRIDES[@]}") ;;
    *) echo "Usage: $0 [attribution|glacier|fraggnn|replicate|all]" >&2; exit 1 ;;
esac

cd "${REPO_ROOT}"

# 1. Validate everything before creating any job script.
if [[ "${#JOB_NAMES[@]}" -ne "${#CONFIGS[@]}" || "${#JOB_NAMES[@]}" -ne "${#TIMES[@]}" \
      || "${#JOB_NAMES[@]}" -ne "${#OVERRIDES[@]}" ]]; then
    echo "JOB_NAMES, CONFIGS, TIMES and OVERRIDES must contain the same number of entries." >&2
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

# 2. Generate the Fir job scripts.
for index in "${!CONFIGS[@]}"; do
    SCRIPT_KWARGS=(w=online "c=${CONFIGS[index]}" "j=${JOB_NAMES[index]}")
    if [[ -n "${OVERRIDES[index]}" ]]; then
        SCRIPT_KWARGS+=("override=${OVERRIDES[index]}")
    fi
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
        --script_kwargs "${SCRIPT_KWARGS[@]}" \
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
