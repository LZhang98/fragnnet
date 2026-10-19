#!/bin/bash
# Generate (but do not submit) the Fir jobs for rare-element route 1, 2026-10-14.
# Plan: docs/plans/masster/rare_chemistry_augmentation_plan.md (section "Rare-element handling").
# The A1/A3 sampling arms and their sampler were removed at the user's request on 2026-10-14.
#
# NIST20, seed 397623, fixed batch 512, i1 recipe:
#   a0  i1 rerun on current code                                  (shared control)
#   e0  train without Br molecules, i1 features                   (hidden-Br control)
#   e1  train without Br molecules, + a_periodic                  (control: e0)
#   e2  full train, + a_periodic                                  (control: a0)
#   e3  train without Br molecules, + a_periodic, no a_onehot     (controls: e1, e0; added 2026-10-15)
# READ: e1 - e0 on val Br molecules, paired by molecule; the share of the a0 - e0 gap e1 closes;
# non-Br val within the 0.0042 seed spread. e2 vs a0: val_cos_sim_0.01_epoch/mean tail mean over
# epochs 54-63, +0.005 to count on one seed. e0/e1 take ~8% fewer steps per epoch than a0.
#
# STAGE (required):
#   all      a0 e0 e1 e2 e3
#   route1   e0 e1 e2 e3 (a0 must also run: e0 and e2 are read against it)
#   a0 | e0 | e1 | e2 | e3  one job.
#
# Profile fir:h100-3g.40gb, wall time 5:00:00: the i1 lineage finished 64 epochs in 3.2-3.3 h.
# j=<job_name> makes a requeued job resume from last.ckpt.

set -euo pipefail

STAGE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_rare_chemistry"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/rare_chemistry_2026_10_10"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="5:00:00"

declare -A JOB_BY_ARM=(
    [a0]=msh_rc_a0_control_ob_obf
    [e0]=msh_rc_e0_no_br_ob_obf
    [e1]=msh_rc_e1_no_br_periodic_ob_obf
    [e2]=msh_rc_e2_periodic_ob_obf
    [e3]=msh_rc_e3_no_br_periodic_no_onehot_ob_obf
)
declare -A CONFIG_BY_ARM=(
    [a0]=nist20_rc_a0_control_ob_obf.yml
    [e0]=nist20_rc_e0_no_br_ob_obf.yml
    [e1]=nist20_rc_e1_no_br_periodic_ob_obf.yml
    [e2]=nist20_rc_e2_periodic_ob_obf.yml
    [e3]=nist20_rc_e3_no_br_periodic_no_onehot_ob_obf.yml
)

case "${STAGE}" in
    all) ARMS=(a0 e0 e1 e2 e3) ;;
    route1) ARMS=(e0 e1 e2 e3) ;;
    a0|e0|e1|e2|e3) ARMS=("${STAGE}") ;;
    *)
        echo "Usage: $0 [all|route1|a0|e0|e1|e2|e3]" >&2
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
    if ! grep -q '^max_epochs: 64$' "${config_fp}"; then
        echo "${config_fp} is not at max_epochs: 64 -- it would not match its control." >&2
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
if ! grep -q 'class AtomPeriodic' src/fragnnet/utils/feat_utils.py \
    || ! grep -q 'filter_exclude_elements' src/fragnnet/dataset/base_dataset.py; then
    echo "This checkout lacks a_periodic or filter_exclude_elements -- pull rare-chemistry-sampling." >&2
    exit 1
fi
for data_dp in data/proc/nist20 data/split/nist20_inchikey; do
    if [[ ! -d "${data_dp}" ]]; then
        echo "Missing ${data_dp} -- the NIST20 [M+H]+ data these configs point at." >&2
        exit 1
    fi
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
    python scripts/setup_job_cc.py \
        --job_name "${JOB_BY_ARM[${arm}]}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile "${PROFILE}" \
        --time "${TIME}" \
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
