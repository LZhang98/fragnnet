#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts"
JOB_DIR="${REPO_ROOT}/jobs"

JOB_NAME=fraggnn_d3_ma_mi_subformula_filter
CONFIG=config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_subformula_filter.yml

cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 1
fi

# 1. Generate the Fir job script (does not submit).
python scripts/setup_job_cc.py \
    --job_name "${JOB_NAME}" \
    --python_script_fp scripts/run_pl_model_fit.py \
    --frag_gnn_dp "${REPO_ROOT}" \
    --script_dp "${SCRIPT_DIR}" \
    --job_dp "${JOB_DIR}" \
    --gpu_profile fir:h100-3g.40gb \
    --time 24:00:00 \
    --cc_account aip-rgreiner \
    --cc_email fw4@ualberta.ca \
    --script_kwargs w=online c="${CONFIG}" \
    --no_submit=True

# 2. Inspect the generated script, then submit manually:
echo "Review ${SCRIPT_DIR}/${JOB_NAME}.sh, then run:"
echo "  sbatch ${SCRIPT_DIR}/${JOB_NAME}.sh"
