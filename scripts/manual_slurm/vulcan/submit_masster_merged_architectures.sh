#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/masster_l40s"
JOB_DIR="${REPO_ROOT}/jobs"

JOB_NAMES=(
    msh_mrg_gineefa384
    msh_mrg_ginegraph384
    msh_mrg_graph384
    msh_mrg_graph512x5
    msh_mrg_int_base
    msh_mrg_int384x3
    msh_mrg_int384x3_atom4
    msh_mrg_int384x3_atom4_mix2
)

CONFIGS=(
    config/masster/merged/unique_fragment/nist20_merged_unique256_enum4096_gine_efa384_bondcentric_a2_b2_transformer_ce_embed_proposal_selection_only.yml
    config/masster/merged/unique_fragment/nist20_merged_unique256_enum4096_gine_graphormer384_bondcentric_a2_b2_transformer_ce_embed_proposal_selection_only.yml
    config/masster/merged/unique_fragment/nist20_merged_unique256_enum4096_graphormer384_bondcentric_a2_b2_transformer_ce_embed_proposal_selection_only.yml
    config/masster/merged/unique_fragment/nist20_merged_unique256_enum4096_graphormer512x5_transformer_ce_embed_proposal_selection_only.yml
    config/masster/merged/unique_fragment/nist20_merged_unique256_enum1024_gine_efa384_bondcentric_a2_b3_transformer_ce_embed_proposal_selection_only_separate_proposal_encoder_no_forced_single_cuts.yml
    config/masster/merged/unique_fragment/nist20_merged_unique256_enum1024_gine_efa384_bondcentric_a2_b3_transformer_ce_embed_proposal_selection_only_separate_proposal_encoder_no_forced_single_cuts_intensity384x3_ff1024_induce128.yml
    config/masster/merged/unique_fragment/nist20_merged_unique256_enum1024_gine_efa384_bondcentric_a2_b3_transformer_ce_embed_proposal_selection_only_separate_proposal_encoder_no_forced_single_cuts_intensity384x3_ff1024_induce128_atom_decoder4.yml
    config/masster/merged/unique_fragment/nist20_merged_unique256_enum1024_gine_efa384_bondcentric_a2_b3_transformer_ce_embed_proposal_selection_only_separate_proposal_encoder_no_forced_single_cuts_intensity384x3_ff1024_induce128_atom_decoder4_proposal_mix2.yml
)

cd "${REPO_ROOT}"


if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available; run this script from a Vulcan login node." >&2
    exit 1
fi
if [[ "${#JOB_NAMES[@]}" -ne "${#CONFIGS[@]}" ]]; then
    echo "JOB_NAMES and CONFIGS must contain the same number of entries." >&2
    exit 1
fi

# 1. Validate every config before creating or submitting any jobs.
for config_fp in "${CONFIGS[@]}"; do
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
done

# 2. Generate all Vulcan L40S job scripts.
for index in "${!CONFIGS[@]}"; do
    python scripts/setup_job_cc.py \
        --job_name "${JOB_NAMES[index]}" \
        --python_script_fp scripts/run_pl_model_fit.py \
        --frag_gnn_dp "${REPO_ROOT}" \
        --script_dp "${SCRIPT_DIR}" \
        --job_dp "${JOB_DIR}" \
        --gpu_profile vulcan:l40s \
        --time 3:00:00 \
        --cc_account aip-rgreiner \
        --script_kwargs w=online c="${CONFIGS[index]}" \
        --no_submit=True
done

# 3. Submit only after every job script was generated successfully.
for job_name in "${JOB_NAMES[@]}"; do
    sbatch "${SCRIPT_DIR}/${job_name}.sh"
done
