#!/bin/bash
# Formula-test retrieval for the MSGYM short-schedule / candidate-budget screen: one 1-hour job
# per checkpoint on a 2g.20gb slice that infers, scores and summarizes all 20 query shards.
# Run from the Fir checkout's login node.
# Plan: docs/plans/masster/msgym_short_budget_screen_2026_10_14.md
#
# Usage: bash scripts/manual_slurm/fir/submit_msgym_sb_retrieval.sh [generate|submit] [arm ...]
#   generate (default) writes the job scripts and checks them with bash -n; submits nothing.
#   submit   generates, then submits. If the shared query shards do not exist yet
#            (no queries.csv), one prepare job (export, build-proc, 20 shards) is submitted
#            first and every checkpoint job waits for it: prepare is not locked.
#   arm ...  optional subset of c32 c64 u128 e1024 u128_e1024 floor01 (default: all).
#
# Checkpoints are the best-val-loss epochs of the first replicate (2026-10-14 20:43), expected
# at saved_ckpts/msgym/sb_<arm>_<run>_ep<NNN>.ckpt (download them from W&B first).
# loglik_cal keeps the ll04i00m calibration, as in the f44tzxoc / pow075 reads.
# A timed-out job resumes from its last scored shard on resubmit. Do not run two jobs for the
# same checkpoint at once: they would write the same shard directories.

set -euo pipefail

MODE="${1:-generate}"
case "${MODE}" in generate|submit) ;; *) echo "Usage: $0 [generate|submit] [arm ...]" >&2; exit 1 ;; esac
shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
LAUNCHER=scripts/manual_slurm/prepare_msgym_retrieval.sh
MSGYM_DIR=config/masster/massspecgym
PROFILE="${FRAGNNET_PROFILE:-fir:h100-2g.20gb}"
# Inference batch by slice (FRAGNNET_EVAL_BATCH_SIZE overrides). Peak allocated memory is ~1.2 GiB
# at 64 and ~4.0 GiB at 256 (pow075, test shard 0), so ~8 GiB at 512 and ~16 GiB at 1024. A shard
# that still runs out of memory is retried at half the batch by the driver.
case "${PROFILE}" in
    fir:h100-1g.10gb) DEFAULT_BATCH=512 ;;
    *) DEFAULT_BATCH=1024 ;;
esac
export FRAGNNET_EVAL_BATCH_SIZE="${FRAGNNET_EVAL_BATCH_SIZE:-${DEFAULT_BATCH}}"
export FRAGNNET_TIME=1:00:00
# Shards, per-checkpoint scores and the prediction HDF5s (kept: --keep_predictions defaults to
# true, ~0.4-1 GB per shard per checkpoint) all go to scratch. Never fall back to output/ under
# /project, where rrg-dwishart has no quota (EDQUOT). Run directory: <this>/formula_test.
export FRAGNNET_RETRIEVAL_DP="${FRAGNNET_RETRIEVAL_DP:-/scratch/feiw/frag-gnn-data/msgym}"
mkdir -p "${FRAGNNET_RETRIEVAL_DP}"
RUN_DP="${FRAGNNET_RETRIEVAL_DP}/formula_test"

# arm:run:epoch
ALL_CKPTS=(c32:gz4fa02d:018 c64:u80w840n:021 u128:28dptwet:023 e1024:5esngzcl:026
           u128_e1024:2m4rb2c2:020 floor01:c43o9rbn:044)
CKPTS=()
if [ "$#" -eq 0 ]; then
    CKPTS=("${ALL_CKPTS[@]}")
else
    for want in "$@"; do
        found=""
        for entry in "${ALL_CKPTS[@]}"; do [ "${entry%%:*}" = "${want}" ] && found="${entry}"; done
        [ -n "${found}" ] || { echo "unknown arm: ${want}" >&2; exit 1; }
        CKPTS+=("${found}")
    done
fi

gen() {  # gen <stages> <arm> <run> <epoch>; prints the job script path
    local stages="$1" arm="$2" run="$3" ep="$4"
    FRAGNNET_PROFILE="${GEN_PROFILE:-${PROFILE}}" FRAGNNET_STAGES="${stages}" \
        bash "${LAUNCHER}" fir "${MSGYM_DIR}/msgym_sb_${arm}.yml" \
        "saved_ckpts/msgym/sb_${arm}_${run}_ep${ep}.ckpt" formula test >&2
    echo "slurm_scripts/msgym_ret_formula_test_sb_${arm}_${run}_ep${ep}_${stages// /-}.sh"
}

# 1. Validate inputs before generating anything.
for entry in "${CKPTS[@]}"; do
    IFS=: read -r arm run ep <<< "${entry}"
    for fp in "${MSGYM_DIR}/msgym_sb_${arm}.yml" "saved_ckpts/msgym/sb_${arm}_${run}_ep${ep}.ckpt"; do
        [ -f "${fp}" ] || { echo "missing: ${fp}" >&2; exit 1; }
    done
done

# 2. Generate: prepare only if the shards are missing, then one job per checkpoint.
PREPARE_SH=""
if [ ! -f "${RUN_DP}/queries.csv" ]; then
    IFS=: read -r arm0 run0 ep0 <<< "${CKPTS[0]}"
    # Prepare always gets 2g.20gb: build-proc needs most of a 31 GB host, more than 1g.10gb's 35G
    # allows with headroom, and export is CPU-bound.
    PREPARE_SH="$(GEN_PROFILE=fir:h100-2g.20gb gen prepare "${arm0}" "${run0}" "${ep0}")"
    echo "No ${RUN_DP}/queries.csv: generated a prepare job."
fi
JOB_SH=()
for entry in "${CKPTS[@]}"; do
    IFS=: read -r arm run ep <<< "${entry}"
    JOB_SH+=("$(gen "infer score summarize" "${arm}" "${run}" "${ep}")")
done
echo "Generated ${#JOB_SH[@]} checkpoint jobs (${PROFILE}, ${FRAGNNET_TIME}, eval batch ${FRAGNNET_EVAL_BATCH_SIZE})."
echo "Generated scripts hold a W&B credential -- keep them out of shared records."

if [ "${MODE}" = generate ]; then
    echo "Review, then: bash $0 submit $*"
    exit 0
fi

# 3. Submit.
dep=()
if [ -n "${PREPARE_SH}" ]; then
    prep_id="$(sbatch --parsable "${PREPARE_SH}")"
    echo "prepare ${prep_id}"
    dep=(--dependency="afterok:${prep_id}")
fi
for sh in "${JOB_SH[@]}"; do
    echo "$(sbatch --parsable ${dep[@]+"${dep[@]}"} "${sh}")  ${sh}"
done
echo "Check: squeue -u \$USER"
echo "GPU usage per shard: ${RUN_DP}/gpu_usage_sb_*.csv (peak memory, throughput, wall times)"
echo "Job-level: sacct -j <id> --format=JobID,JobName%50,Elapsed,State,MaxRSS,AllocTRES%60,TRESUsageInMax%80"
echo "Summaries: ${RUN_DP}/summary_sb_*.csv"
echo "Predictions: ${RUN_DP}/shard_*/sb_*/predictions_test.hdf5"
