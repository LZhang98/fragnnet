#!/bin/bash
# Generate (but do not submit) the Fir jobs for the NIST23 EI TMS rewire screen: e6_control, e6_tms,
# e6_tms_null, all the e5 recipe (config/masster/ei/e5_hr_ei_holdout_nominal.yml) at seed 397624.
# Plan: docs/plans/masster/masster_ei_smarts_plan.md
#
# STAGE selects which jobs are handled:
#   all      (default) control, tms, tms_null. Submit together so they share code and queue conditions.
#   control  the seed-397624 control only.
#   arms     tms, tms_null only.
# MODE (second argument):
#   generate (default) validate, write the job scripts, print the sbatch lines. Never submits.
#   print    validate and print what generate would run. Writes nothing.
#
# Run on a Fir login node from the Fir checkout, with `python` from the cluster environment.
#
# DATA. spec_df.pkl, mol_df.pkl and the split are Git-ignored, so rsync them from the workstation
# into the same paths of the Fir checkout first (about 740 MB; check the /project quota). The
# md5 values below are the workstation files of 2026-10-18; e5's own W&B metadata records no
# revision, so these are the files e6 trains on, not a proven match to e5's.
#
# CODE. The checkout must carry the [M]+ rewire rule (tms_siloxy_silylium); the guard below refuses
# a checkout without it, since the tms arms would otherwise be a silent no-op.
#
# PROFILE fir:h100-3g.40gb. WALL TIME 24:00:00 is a PLACEHOLDER: e5's epoch time was not found in the
# repository (W&B run eugy8s9n); max_epochs is 64 with no early stop, so set TIME from that run.
# The 40 GB fit at train_batch_size 512 / loss_batch_size 64 is unmeasured on EI; smoke locally first.
#
# Every job passes j=<job_name> so a requeued job resumes from last.ckpt rather than restarting.
# Do not reuse a job name for a different config.

set -euo pipefail

STAGE="${1:-all}"
MODE="${2:-generate}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/slurm_scripts/ei_tms"
JOB_DIR="${REPO_ROOT}/jobs"
EI_DIR="config/masster/ei"
ACCOUNT="rrg-dwishart"
EMAIL="fw4@ualberta.ca"
PROFILE="fir:h100-3g.40gb"
TIME="24:00:00"
SPLIT_DP="data/split/nist23_ei_inchikey_hr_ei_excluded_2026_10_17"
RULE_MARKER="tms_siloxy_silylium"

CTRL_ARMS=(control)
TEST_ARMS=(tms tms_null)

case "${STAGE}" in
    all)     ARMS=("${CTRL_ARMS[@]}" "${TEST_ARMS[@]}") ;;
    control) ARMS=("${CTRL_ARMS[@]}") ;;
    arms)    ARMS=("${TEST_ARMS[@]}") ;;
    *) echo "Usage: $0 [all|control|arms] [generate|print]" >&2; exit 1 ;;
esac
case "${MODE}" in
    generate|print) ;;
    *) echo "Usage: $0 [all|control|arms] [generate|print]" >&2; exit 1 ;;
esac

JOB_NAMES=()
CONFIGS=()
for arm in "${ARMS[@]}"; do
    JOB_NAMES+=("ei_e6_${arm}")
    CONFIGS+=("${EI_DIR}/e6_${arm}.yml")
done

# Pinned md5 of the workstation files (2026-10-18).
EXPECTED_MD5=(
    "8c54f0d11dbd2990a9427b563428d678  data/proc/nist23_ei/spec_df.pkl"
    "a88d73a46dc032bfe50779cbaf2f4160  data/proc/nist23_ei/mol_df.pkl"
    "663359d446a46252525a7a0e8d14770e  ${SPLIT_DP}/train_ids.csv"
    "4ba9ae5e6a1f7264ceb7ed9b7a52e20b  ${SPLIT_DP}/val_ids.csv"
    "f286940c1d586601dfad872e59203c26  ${SPLIT_DP}/test_ids.csv"
)

cd "${REPO_ROOT}"

# 1. Validate everything before creating any job script.
for config_fp in "${CONFIGS[@]}"; do
    if [[ ! -f "${config_fp}" ]]; then
        echo "Missing config: ${config_fp}" >&2
        exit 1
    fi
done
for wheel in wheels/pyteomics-4.7.5-py3-none-any.whl wheels/torch_linear_assignment-0.0.6.tar.gz; do
    if [[ ! -f "${wheel}" ]]; then
        echo "Missing ${wheel} -- the generated job installs it and wheels/ is Git-ignored." >&2
        exit 1
    fi
done
if ! grep -q "${RULE_MARKER}" src/fragnnet/models/masster/candidates/rewire_rules.py; then
    echo "This checkout lacks the [M]+ rule ${RULE_MARKER}; pull the revision that adds it." >&2
    exit 1
fi
for entry in "${EXPECTED_MD5[@]}"; do
    data_fp="${entry#*  }"
    if [[ ! -f "${data_fp}" ]]; then
        echo "Missing ${data_fp} -- rsync it from the workstation into this checkout." >&2
        exit 1
    fi
done
printf '%s\n' "${EXPECTED_MD5[@]}" | md5sum -c --quiet - || {
    echo "EI data md5 mismatch: these are not the files the e6 arms were prepared with." >&2
    exit 1
}

# 2. Generate the Fir 3g.40gb job scripts.
for index in "${!CONFIGS[@]}"; do
    command=(
        python scripts/setup_job_cc.py
        --job_name "${JOB_NAMES[index]}"
        --python_script_fp scripts/run_pl_model_fit.py
        --frag_gnn_dp "${REPO_ROOT}"
        --script_dp "${SCRIPT_DIR}"
        --job_dp "${JOB_DIR}"
        --gpu_profile "${PROFILE}"
        --time "${TIME}"
        --cc_account "${ACCOUNT}"
        --cc_email "${EMAIL}"
        --script_kwargs w=online "c=${CONFIGS[index]}" "j=${JOB_NAMES[index]}"
        --no_submit=True
    )
    if [[ "${MODE}" == "print" ]]; then
        printf '%q ' "${command[@]}"
        echo
    else
        "${command[@]}"
    fi
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
