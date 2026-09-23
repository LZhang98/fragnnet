#!/bin/bash
# Generate (but do not submit) the Fir jobs for the [M+H]+/FT target-cleaning pair, 2026-09-22.
#
# Question: does cleaning the training target -- 20 ppm peak merge, then precursor-bounded
# subformula retention at the FT instrument tolerance, then an above-precursor cut -- make a
# better model, or just an easier denominator?
#
# STAGE:
#   pair   both arms. This is the only stage that answers anything; f1 alone cannot.
#   f0     control only, filters off.
#   f1     arm only, all three filters on.
#
# SCOPE AND SPLIT. [M+H]+ / FT only, on data/split/nist20_inchikey, which ALREADY EXISTS and is
# already that scope: 157,678 / 52,020 / 52,621 spectra at 60.1/19.8/20.1 by 2D InChIKey. No new
# split and no new DAG were needed -- spec_fp, mol_fp and frag_dp are identical to the ma_mi
# lineage and only split_dp differs. This scope is also where the subformula filter is known
# safe: its Na/Cl charge-migration hazard cannot fire without Na/Cl, and measured retention here
# is 97.07% of intensity.
#
# READING THE RESULT. f1's cosine is NOT comparable to f0's. clean_spectra runs at dataset load,
# so each arm is scored against its own target and f1 has had the peaks it was failing on
# deleted. Expect val_wrecall to jump; that is arithmetic. The real test is a cross-eval of the
# f1 CHECKPOINT against f0's unfiltered target -- run that with
# scripts/eval/run_save_inference_eval.py once f1 has a checkpoint, and read the pair only then.
#
# UNMERGED: one row per spectrum. Nothing produced here is comparable to a merged cosine.
#
# Profile fir:h100-3g.40gb, matching the c4a_base lineage these configs derive from.
# 64 EPOCHS, NOT 128. The 128-epoch screen was calibrated on merged runs; the two reference
# unmerged runs at this batch/loss_batch (6ks7v8ce, h0gvuf4f) sit 0.0043 and 0.0011 below their
# own best by epoch 64, and one declines after it.
# 12 h wall. Measured rate here is ~12.8-16.2 ep/h (8.9 and 7.0 ep/h on ma_mi unmerged against
# 70.0 merged, scaled by this split being 1.825x smaller), putting 64 epochs at ~4.0-5.0 h. So
# 12 h is ~2.5x headroom and absorbs a slow node or a rate below the observed range.
# j=<job_name> also means a requeue resumes from last.ckpt rather than restarting.
#
# Both arms set upload_checkpoints: true, so the checkpoints land in W&B artifacts and outlive
# the job's scratch directory. The cross-eval below depends on that.

set -euo pipefail

STAGE="${1:-pair}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masshatter_mh_ft_clean"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/massshatter/mh_ft_clean_2026_09_22"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"

F0_JOBS=(msh_mhft_f0_control);      F0_CONFIGS=(f0_mh_ft_control.yml);      F0_TIMES=(12:00:00)
F1_JOBS=(msh_mhft_f1_all_filters);  F1_CONFIGS=(f1_mh_ft_all_filters.yml);  F1_TIMES=(12:00:00)

case "${STAGE}" in
    f0)   JOB_NAMES=("${F0_JOBS[@]}"); CONFIGS=("${F0_CONFIGS[@]}"); TIMES=("${F0_TIMES[@]}") ;;
    f1)   JOB_NAMES=("${F1_JOBS[@]}"); CONFIGS=("${F1_CONFIGS[@]}"); TIMES=("${F1_TIMES[@]}") ;;
    pair) JOB_NAMES=("${F0_JOBS[@]}" "${F1_JOBS[@]}")
          CONFIGS=("${F0_CONFIGS[@]}" "${F1_CONFIGS[@]}")
          TIMES=("${F0_TIMES[@]}" "${F1_TIMES[@]}") ;;
    *) echo "Usage: $0 [pair|f0|f1]" >&2; exit 1 ;;
esac

cd "${REPO_ROOT}"

# 1. Validate before creating any job script.
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
