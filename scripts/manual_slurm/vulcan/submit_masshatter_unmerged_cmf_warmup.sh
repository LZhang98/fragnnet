#!/bin/bash
# Generate (but do not submit) the Vulcan L40S jobs for the unmerged CMF proposal-warmup ladder.
#
# Single-factor ladder over mass_shatter_proposal_only_until_epoch. The no-warmup control is the
# anchor launched by submit_masshatter_unmerged_cmf.sh, which leaves that key null; do not
# regenerate it here, so an already-submitted anchor keeps its job id and resume state.
# Boundaries are inclusive: e0_N trains proposal-only for epochs 0..N, that is N+1 epochs.
# Scoped to the two short boundaries; the merged ladder's 9 and the merged screen's 15 are
# not run here.
#
# Same twelve-hour wall time and j=<JOB_NAME> resume as the anchor: a re-submission continues
# from last.ckpt in the same W&B run, and the job_id/<JOB_NAME>.id file is removed on clean
# completion so a later re-run starts fresh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masshatter_l40s"
JOB_DIR="${REPO_ROOT}/jobs"
CONFIG_DIR=config/massshatter/ma_mi_unmerged/cmf
STEM=nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol

BOUNDARIES=(2 4)

cd "${REPO_ROOT}"

# 1. Validate every config before generating any job.
for boundary in "${BOUNDARIES[@]}"; do
    config_fp="${CONFIG_DIR}/${STEM}_warmup_e0_${boundary}.yml"
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
done

# 2. Generate one Vulcan L40S job script per warmup boundary.
for boundary in "${BOUNDARIES[@]}"; do
    job_name="msh_unmrg_cmf_wu${boundary}"
    python scripts/setup_job_cc.py \
        --job_name "${job_name}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile vulcan:l40s \
        --time 12:00:00 \
        --cc_account aip-rgreiner \
        --script_kwargs w=online c="${CONFIG_DIR}/${STEM}_warmup_e0_${boundary}.yml" j="${job_name}" \
        --no_submit=True
done

# 3. Print the submit commands; review before running them.
echo
echo "Review, then submit:"
for boundary in "${BOUNDARIES[@]}"; do
    echo "  sbatch ${SCRIPT_DIR}/msh_unmrg_cmf_wu${boundary}.sh"
done
