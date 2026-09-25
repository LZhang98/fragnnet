#!/bin/bash
# Generate (but do not submit) the Vulcan L40S job for the unmerged MASSter CMF anchor.
#
# This is the corrected unmerged control of docs/plans/masster/masster_improvement_plan.md step 1.
# Two things differ from the earlier three-hour unmerged jobs that never converged:
#   - a twelve-hour wall time;
#   - a job id (j=), which records job_id/<JOB_NAME>.id and makes a re-submission resume
#     from last.ckpt in the same W&B run. The .id file is removed on clean completion,
#     so re-running after a finished job starts a new run rather than resuming.
# Checkpoints land in wandb/run-*/files/ckpt inside the cluster checkout, not in node-local
# scratch, so they survive a wall-time kill.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_l40s"
JOB_DIR="${REPO_ROOT}/jobs"

JOB_NAME=msh_unmrg_cmf_dualce
CONFIG=config/masster/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol.yml

cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 1
fi

python scripts/setup_job_cc.py \
    --job_name "${JOB_NAME}" \
    --python_script_fp scripts/run_pl_model_fit.py \
    --frag_gnn_dp "${REPO_ROOT}" \
    --script_dp "${SCRIPT_DIR}" \
    --job_dp "${JOB_DIR}" \
    --gpu_profile vulcan:l40s \
    --time 12:00:00 \
    --cc_account aip-rgreiner \
    --script_kwargs w=online c="${CONFIG}" j="${JOB_NAME}" \
    --no_submit=True

echo
echo "Review, then submit:"
echo "  bash -n ${SCRIPT_DIR}/${JOB_NAME}.sh"
echo "  sbatch ${SCRIPT_DIR}/${JOB_NAME}.sh"
