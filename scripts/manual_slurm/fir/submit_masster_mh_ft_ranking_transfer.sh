#!/bin/bash
# Generate (but do not submit) the Fir jobs for the unmerged [M+H]+/FT shortcut-reduction arms, 2026-09-24.
#
# Question: the ranking error is a generalization failure (checkpoint 6h67zps9: train ranking gap
# 0.011, val 0.121), and the shortcut audit (overfit read-out §7) leaves one measured shortcut: a
# per-molecule allocation lookup. Molecule identity reaches every head input, so these arms limit
# the allocator as a whole or remove optimizer noise, instead of blocking one input:
#   h1   intensity-allocator dropout 0.1 -> 0.2   (masster_intensity_transformer_dropout)
#   h2   intensity-allocator width 256 -> 128     (masster_intensity_transformer_dim; head 2.46M -> 0.97M)
#   u2   EMA of the weights, decay 0.999          (ema_decay; ~3.2 epochs at 308 steps/epoch)
# The boundary-block dropout arm (u1) was dropped on 2026-09-24: blocking one input can reroute the
# lookup. The key stays in the code, off by default.
# Plan: docs/plans/masster/masster_ranking_transfer_plan.md
#
# STAGE:
#   all   (default) h1, h2 and u2.
#   h1    h1_head_dropout02 alone.
#   h2    h2_head_dim128 alone.
#   u2    u2_ema999 alone.
#
# CONTROL IS AN EXISTING RUN. Each arm is f0_mh_ft_control.yml plus one key, so the control is
# 3bej6fq7 (val cos_sim_0.01 tail mean 0.7298 over epochs 54-63, best 0.7318). No control job is
# generated here. For an exact same-code control, regenerate f0 with
# scripts/manual_slurm/fir/submit_masster_mh_ft_clean.sh f0.
#
# READING IT: the tail mean of val_cos_sim_0.01_epoch/mean over epochs 54-63 against 0.7298, then
# the offline ranking decomposition of each best checkpoint's val export and a train sample (a
# shortcut reduction raises the TRAIN ranking gap and lowers the VAL one), then val cosine by
# nearest-train Tanimoto bin (a real reduction lifts the least-similar molecules most).
# upload_checkpoints is true in both configs, so the checkpoints land on W&B.
#
# Profile fir:h100-3g.40gb, as for the f0/f1 pair these configs derive from: train-side metrics are
# off (peak ~26.8 GiB). Wall time 5:00:00: these arms ran at 2.73-2.86 min/epoch and f0 at 2.99,
# 64 epochs in ~2.9-3.2 h, so 5 h is ~1.5x headroom, and j=<job_name> makes a requeued job resume
# from last.ckpt.

set -euo pipefail

STAGE="${1:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_mh_ft_ranking_transfer"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/mh_ft_ranking_transfer_2026_09_23"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"

H1_JOBS=(msh_mhft_h1_head_dropout02);   H1_CONFIGS=(h1_head_dropout02.yml);   H1_TIMES=(5:00:00)
H2_JOBS=(msh_mhft_h2_head_dim128);      H2_CONFIGS=(h2_head_dim128.yml);      H2_TIMES=(5:00:00)
U2_JOBS=(msh_mhft_u2_ema999);           U2_CONFIGS=(u2_ema999.yml);           U2_TIMES=(5:00:00)

case "${STAGE}" in
    h1)   JOB_NAMES=("${H1_JOBS[@]}"); CONFIGS=("${H1_CONFIGS[@]}"); TIMES=("${H1_TIMES[@]}") ;;
    h2)   JOB_NAMES=("${H2_JOBS[@]}"); CONFIGS=("${H2_CONFIGS[@]}"); TIMES=("${H2_TIMES[@]}") ;;
    u2)   JOB_NAMES=("${U2_JOBS[@]}"); CONFIGS=("${U2_CONFIGS[@]}"); TIMES=("${U2_TIMES[@]}") ;;
    all)  JOB_NAMES=("${H1_JOBS[@]}" "${H2_JOBS[@]}" "${U2_JOBS[@]}")
          CONFIGS=("${H1_CONFIGS[@]}" "${H2_CONFIGS[@]}" "${U2_CONFIGS[@]}")
          TIMES=("${H1_TIMES[@]}" "${H2_TIMES[@]}" "${U2_TIMES[@]}") ;;
    *) echo "Usage: $0 [all|h1|h2|u2]" >&2; exit 1 ;;
esac

cd "${REPO_ROOT}"

# 1. Validate before creating any job script.
if [[ "${#JOB_NAMES[@]}" -ne "${#CONFIGS[@]}" || "${#JOB_NAMES[@]}" -ne "${#TIMES[@]}" ]]; then
    echo "JOB_NAMES, CONFIGS and TIMES must contain the same number of entries." >&2
    exit 1
fi
for config_fn in "${CONFIGS[@]}"; do
    config_fp="${CONFIG_DIR}/${config_fn}"
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
    # The control ran 64 epochs; a different horizon breaks the matched-epoch read.
    if ! grep -q '^max_epochs: 64$' "${config_fp}"; then
        echo "${config_fp} is not at max_epochs: 64 -- it would not match the 3bej6fq7 control." >&2
        exit 1
    fi
    # Train metrics on would add a second key against the control and push peak GPU past a 3g slice.
    if ! grep -q '^compute_train_auxiliary_metrics: false$' "${config_fp}"; then
        echo "${config_fp} must keep compute_train_auxiliary_metrics: false (control match, 3g fit)." >&2
        exit 1
    fi
    # The offline ranking decomposition needs the checkpoints off the cluster.
    if ! grep -q '^upload_checkpoints: true$' "${config_fp}"; then
        echo "${config_fp} must set upload_checkpoints: true -- the read needs the checkpoints." >&2
        exit 1
    fi
done
if [[ ! -d data/split/nist20_inchikey ]]; then
    echo "Missing data/split/nist20_inchikey -- the [M+H]+/FT split these configs point at." >&2
    exit 1
fi
# The generated job installs both archives from the checkout; wheels/ is Git-ignored, so a fresh
# cluster clone does not have them and the failure would otherwise appear only after job start.
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
echo "Control is the existing run 3bej6fq7 (tail mean 0.7298, epochs 54-63); no control job needed."
