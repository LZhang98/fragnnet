#!/bin/bash
# Generate (but do not submit) the Fir jobs for the rare chemistry arms, 2026-10-14.
# Plan: docs/plans/masster/rare_chemistry_augmentation_plan.md
#
# Sampling arms (sections 2 and 4), seed 397623, fixed batch 512, len(train) draws per epoch:
#   a0        NIST20 A0: i1 rerun on current code, uniform per-spectrum shuffle (shared control)
#   a1        NIST20 A1: molecule -> acquisition key -> spectrum sampler       (control: a0)
#   msgym_a0  MSGYM A0: k1/EMA/Fourier control rerun on current code
#   msgym_a1  MSGYM A1                                                         (control: msgym_a0)
# A3 is not offered: the class census failed its support gate and awaits a user redesign
# (docs/findings/masster/rare_chemistry_class_census_2026_10_10.md).
#
# Rare-element route 1 (plan section "Rare-element handling"), NIST20, same seed and recipe:
#   e0  train without Br molecules, i1 features                   (hidden-Br control)
#   e1  train without Br molecules, + a_periodic                  (control: e0)
#   e2  full train, + a_periodic                                  (control: a0)
# READ: e1 - e0 on val Br molecules, paired by molecule; the share of the a0 - e0 gap e1 closes;
# non-Br val within the 0.0042 seed spread. e2 vs a0: val_cos_sim_0.01_epoch/mean tail mean over
# epochs 54-63, +0.005 to count on one seed. e0/e1 take ~8% fewer steps per epoch than a0.
#
# STAGE (required):
#   nist     a0 a1
#   route1   e0 e1 e2 (submit with a0, which both e0 and e2 are read against)
#   msgym    msgym_a0 msgym_a1
#   all      every arm above
#   a0 | a1 | e0 | e1 | e2 | msgym_a0 | msgym_a1  one job.
#
# Profile fir:h100-3g.40gb for every arm. NIST20 wall time 5:00:00: the i1 lineage finished 64
# epochs in 3.2-3.3 h. MSGYM wall time 8:00:00: its 128-epoch runs took 3.2-4.9 h. j=<job_name>
# makes a requeued job resume from last.ckpt.

set -euo pipefail

STAGE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_rare_chemistry"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/rare_chemistry_2026_10_10"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"

declare -A JOB_BY_ARM=(
    [a0]=msh_rc_a0_control_ob_obf
    [a1]=msh_rc_a1_uniform_mol_ob_obf
    [e0]=msh_rc_e0_no_br_ob_obf
    [e1]=msh_rc_e1_no_br_periodic_ob_obf
    [e2]=msh_rc_e2_periodic_ob_obf
    [msgym_a0]=msh_rc_msgym_a0_control
    [msgym_a1]=msh_rc_msgym_a1_uniform_mol
)
declare -A CONFIG_BY_ARM=(
    [a0]=nist20_rc_a0_control_ob_obf.yml
    [a1]=nist20_rc_a1_uniform_mol_ob_obf.yml
    [e0]=nist20_rc_e0_no_br_ob_obf.yml
    [e1]=nist20_rc_e1_no_br_periodic_ob_obf.yml
    [e2]=nist20_rc_e2_periodic_ob_obf.yml
    [msgym_a0]=msgym_rc_a0_control.yml
    [msgym_a1]=msgym_rc_a1_uniform_mol.yml
)

case "${STAGE}" in
    nist) ARMS=(a0 a1) ;;
    route1) ARMS=(e0 e1 e2) ;;
    msgym) ARMS=(msgym_a0 msgym_a1) ;;
    all) ARMS=(a0 a1 e0 e1 e2 msgym_a0 msgym_a1) ;;
    a0|a1|e0|e1|e2|msgym_a0|msgym_a1) ARMS=("${STAGE}") ;;
    *)
        echo "Usage: $0 [nist|route1|msgym|all|a0|a1|e0|e1|e2|msgym_a0|msgym_a1]" >&2
        exit 1
        ;;
esac

cd "${REPO_ROOT}"

# 1. Validate before creating any job script.
for arm in "${ARMS[@]}"; do
    config_fp="${CONFIG_DIR}/${CONFIG_BY_ARM[${arm}]}"
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
    # Arms share their control's horizon for the matched tail-mean read.
    if [[ "${arm}" == msgym_* ]]; then
        expected_epochs=128
    else
        expected_epochs=64
    fi
    if ! grep -q "^max_epochs: ${expected_epochs}$" "${config_fp}"; then
        echo "${config_fp} is not at max_epochs: ${expected_epochs} -- it would not match its control." >&2
        exit 1
    fi
    if ! grep -q '^compute_train_auxiliary_metrics: false$' "${config_fp}"; then
        echo "${config_fp} must keep compute_train_auxiliary_metrics: false (control match, 3g fit)." >&2
        exit 1
    fi
    # The cohort reads run offline on the checkpoints.
    if ! grep -q '^upload_checkpoints: true$' "${config_fp}"; then
        echo "${config_fp} must set upload_checkpoints: true -- the cohort read needs the checkpoints." >&2
        exit 1
    fi
done
# The checkout must carry the code these arms need; a stale clone would fail at config load or,
# worse, train without the feature.
if ! grep -q 'molecule_acquisition_sampler' src/fragnnet/dataset/loading.py; then
    echo "This checkout lacks the molecule-acquisition sampler -- check out rare-chemistry-sampling." >&2
    exit 1
fi
if ! grep -q 'class AtomPeriodic' src/fragnnet/utils/feat_utils.py \
    || ! grep -q 'filter_exclude_elements' src/fragnnet/dataset/base_dataset.py; then
    echo "This checkout lacks a_periodic or filter_exclude_elements -- pull rare-chemistry-sampling." >&2
    exit 1
fi
for arm in "${ARMS[@]}"; do
    if [[ "${arm}" == msgym_* ]]; then
        data_dps=(data/proc/ms_gym_benchmark_1_5 data/split/ms_gym_benchmark_1_5)
    else
        data_dps=(data/proc/nist20 data/split/nist20_inchikey)
    fi
    for data_dp in "${data_dps[@]}"; do
        if [[ ! -d "${data_dp}" ]]; then
            echo "Missing ${data_dp} -- needed by ${CONFIG_BY_ARM[${arm}]}." >&2
            exit 1
        fi
    done
done
# The generated job installs both archives from the checkout; wheels/ is Git-ignored.
for wheel in wheels/pyteomics-4.7.5-py3-none-any.whl wheels/torch_linear_assignment-0.0.6.tar.gz; do
    if [[ ! -f "${wheel}" ]]; then
        echo "Missing ${wheel} -- the generated job installs it and wheels/ is Git-ignored." >&2
        exit 1
    fi
done

# 2. Generate the Fir job scripts.
for arm in "${ARMS[@]}"; do
    if [[ "${arm}" == msgym_* ]]; then
        time_limit="8:00:00"
    else
        time_limit="5:00:00"
    fi
    python scripts/setup_job_cc.py \
        --job_name "${JOB_BY_ARM[${arm}]}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile "${PROFILE}" \
        --time "${time_limit}" \
        --cc_account "${ACCOUNT}" \
        --cc_email "${EMAIL}" \
        --script_kwargs w=online c="${CONFIG_DIR}/${CONFIG_BY_ARM[${arm}]}" j="${JOB_BY_ARM[${arm}]}" \
        --no_submit=True
done

echo
echo "Code revision: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)$(git diff --quiet || echo ' (dirty)')"
echo "Generated scripts hold a W&B credential -- keep them out of shared records."
echo "Review, then submit from a Fir login node:"
for arm in "${ARMS[@]}"; do
    echo "  bash -n ${SCRIPT_DIR}/${JOB_BY_ARM[${arm}]}.sh"
done
for arm in "${ARMS[@]}"; do
    echo "  sbatch ${SCRIPT_DIR}/${JOB_BY_ARM[${arm}]}.sh"
done
