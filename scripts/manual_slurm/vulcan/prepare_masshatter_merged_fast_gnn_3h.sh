#!/bin/bash
# Generate three independent three-hour L40S jobs. Submit the printed scripts separately.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
: "${CC_EMAIL:?Set CC_EMAIL to your scheduler notification email}"
FRAGNNET_ACCOUNT="${CC_ACCOUNT:-aip-rgreiner}"
FRAGNNET_PREFIX="${JOB_PREFIX:-msh_ma_mi_mrg_cfb}"
CONFIG_BASE=config/massshatter/merged/unique_fragment/nist20_ma_mi_merged_fast_gnn_proposal

# 1. Generate each job with the matching fragment and beam budget.
for budget in 128 256 512; do
    config_fp="${CONFIG_BASE}_fragments${budget}.yml"
    if [[ "${budget}" == 256 ]]; then
        config_fp="${CONFIG_BASE}.yml"
    fi
    job_name="${FRAGNNET_PREFIX}_${budget}_3h"
    python scripts/setup_job_cc.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --job_name "${job_name}" \
        --gpu_profile vulcan:l40s \
        --cc_account "${FRAGNNET_ACCOUNT}" \
        --cc_email "${CC_EMAIL}" \
        --time 3:00:00 \
        --script_kwargs w=online c="${config_fp}" \
        --no_submit=True
    bash -n "slurm_scripts/${job_name}.sh"
done

# 2. Print submission commands after all scripts pass shell syntax validation.
for budget in 128 256 512; do
    echo "sbatch slurm_scripts/${FRAGNNET_PREFIX}_${budget}_3h.sh"
done
