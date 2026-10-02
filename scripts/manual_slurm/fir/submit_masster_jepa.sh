#!/bin/bash
# Generate (but do not submit) the Fir jobs for the MASSter JEPA auxiliary, Gate 1 (2026-10-01).
# Plan: docs/plans/masster/masster_jepa_embedding_plan.md
#
#   j1a  i1 + JEPA term, encoder source, weight 0.1    (control: i1, i50ig7rf)
#   j1b  i1 + JEPA term, encoder source, weight 1.0    (control: i1, i50ig7rf)
#   j2   i1 + JEPA term, allocator source, weight 1.0  (control: i1, i50ig7rf)
#
# Each arm is i1_inst_ob_obf plus W&B names and four masster_jepa_* keys; the predictor is built
# last on a forked RNG, so every i1 module starts from i1's initialization. The arms read the
# target table data/proc/nist20/jepa_dreams_ssl_1pct_floor.npz (434 MB, md5 below), which must be
# copied into this checkout first; setup_job_cc.py stages it to the node. They also need a checkout
# with the masster_jepa_* code (masster_pl.py, masster_model.py, bond_fragment_heads.py,
# utils/jepa_utils.py and setup_job_cc.py, 2026-10-01) and the relabelled OB / OB_F spec_df.pkl.
#
# READ: tail mean of val_cos_sim_0.01_epoch/mean over epochs 54-63 against i1 (tail 0.7607);
# +0.005 to count on one seed (seed spread 0.0042); the fa1@63 floor (independent selector) on
# the uploaded checkpoints; train vs val jepa_cos; wall clock per epoch against i1.
#
# Profile and wall time as i1's lineage: fir:h100-3g.40gb, 5:00:00 (i1-type runs finish 64 epochs
# in 3.2-3.3 h; the term added ~7% per step at batch 8 locally).
#
# STAGE: gate1 (all three) | j1a | j1b | j2

set -euo pipefail

STAGE="${1:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_jepa"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR="config/masster/mh_jepa_2026_10_01"
TABLE="data/proc/nist20/jepa_dreams_ssl_1pct_floor.npz"
TABLE_MD5="576c115361781cf3f8b37df1d96fbad8"
SPEC_DF_MD5="bdc009bbeaf6a38d126326d127630cad"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="5:00:00"

declare -A JOB_BY_ARM=(
    [j1a]=msh_jepa_j1a_encoder_w0p1
    [j1b]=msh_jepa_j1b_encoder_w1
    [j2]=msh_jepa_j2_allocator_w1
)
declare -A CONFIG_BY_ARM=(
    [j1a]=j1a_jepa_encoder_w0p1.yml
    [j1b]=j1b_jepa_encoder_w1.yml
    [j2]=j2_jepa_allocator_w1.yml
)

case "${STAGE}" in
    gate1) ARMS=(j1a j1b j2) ;;
    j1a|j1b|j2) ARMS=("${STAGE}") ;;
    *)
        echo "Usage: $0 [gate1|j1a|j1b|j2]" >&2
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
    # The matched tail-mean read needs i1's horizon, train-metric setting and logged coverage, and
    # the floor read needs the checkpoints.
    for line in 'max_epochs: 64' 'compute_train_auxiliary_metrics: false' \
        'upload_checkpoints: true' 'masster_candidate_coverage_metrics: true' \
        "masster_jepa_target_fp: ${TABLE}"; do
        if ! grep -q "^${line}$" "${config_fp}"; then
            echo "${config_fp} must set '${line}' (control match / JEPA table)." >&2
            exit 1
        fi
    done
done
if ! grep -q '^masster_jepa_loss_weight:' config/template.yml || [[ ! -f src/fragnnet/utils/jepa_utils.py ]]; then
    echo "This checkout lacks the masster_jepa_* code (template key or src/fragnnet/utils/jepa_utils.py)." >&2
    exit 1
fi
if [[ ! -f "${TABLE}" ]]; then
    echo "Missing ${TABLE} -- copy it from the workstation (md5 ${TABLE_MD5})." >&2
    exit 1
fi
if [[ "$(md5sum "${TABLE}" | cut -d' ' -f1)" != "${TABLE_MD5}" ]]; then
    echo "${TABLE} md5 differs from ${TABLE_MD5} -- recopy it." >&2
    exit 1
fi
# The table's per-batch peak check would fail on another spec_df anyway; failing here is cheaper.
if [[ "$(md5sum data/proc/nist20/spec_df.pkl | cut -d' ' -f1)" != "${SPEC_DF_MD5}" ]]; then
    echo "data/proc/nist20/spec_df.pkl is not the relabelled OB / OB_F copy (md5 ${SPEC_DF_MD5})." >&2
    exit 1
fi
if [[ ! -d data/split/nist20_inchikey ]]; then
    echo "Missing data/split/nist20_inchikey." >&2
    exit 1
fi
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
echo "Generated scripts hold a W&B credential -- keep them out of shared records."
echo "Check each stages ${TABLE}, then submit from a Fir login node:"
for arm in "${ARMS[@]}"; do
    echo "  bash -n ${SCRIPT_DIR}/${JOB_BY_ARM[${arm}]}.sh && grep -c jepa_dreams ${SCRIPT_DIR}/${JOB_BY_ARM[${arm}]}.sh"
done
for arm in "${ARMS[@]}"; do
    echo "  sbatch ${SCRIPT_DIR}/${JOB_BY_ARM[${arm}]}.sh"
done
