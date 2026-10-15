#!/bin/bash
# Formula-test retrieval for the MSGYM short-schedule / candidate-budget screen, split into
# 1-hour jobs on small MIG slices. Run from the Fir checkout's login node.
# Plan: docs/plans/masster/msgym_short_budget_screen_2026_10_14.md
#
# Usage: bash scripts/manual_slurm/fir/submit_msgym_sb_retrieval.sh [generate|submit]
#   generate (default) writes every job script and checks it with bash -n; submits nothing.
#   submit   generates, then submits the chain with dependencies:
#              1 prepare job (export, build-proc, 20 query shards; 2g.20gb for its 70 GB RAM)
#              -> per checkpoint, 5 infer/score jobs of 4 shards each (2g.20gb)
#              -> per checkpoint, 1 summarize job (1g.10gb) once its 5 shard jobs succeed.
#            Prepare is not locked, so every shard job waits for it.
#
# Checkpoints are the best-val-loss epochs of the first replicate (2026-10-14 20:43), expected
# at saved_ckpts/msgym/sb_<arm>_<run>_ep<NNN>.ckpt (download them from W&B first).
# loglik_cal keeps the ll04i00m calibration, as in the f44tzxoc / pow075 reads.
#
# Timing: a full test fold took 1 h 38 min on the local 8 GB GPU (about 5 min per shard), so
# 4 shards fit in 1 h with setup. A timed-out job resumes from its last scored shard on resubmit.

set -euo pipefail

MODE="${1:-generate}"
case "${MODE}" in generate|submit) ;; *) echo "Usage: $0 [generate|submit]" >&2; exit 1 ;; esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
LAUNCHER=scripts/manual_slurm/prepare_msgym_retrieval.sh
MSGYM_DIR=config/masster/massspecgym
export FRAGNNET_TIME=1:00:00
# Shards, per-checkpoint scores and the prediction HDF5s (kept: --keep_predictions defaults to
# true, ~0.4-1 GB per shard per checkpoint) all go to scratch. Never fall back to output/ under
# /project, where rrg-dwishart has no quota (EDQUOT). Run directory: <this>/formula_test.
export FRAGNNET_RETRIEVAL_DP="${FRAGNNET_RETRIEVAL_DP:-/scratch/feiw/frag-gnn-data/msgym}"
mkdir -p "${FRAGNNET_RETRIEVAL_DP}"

# arm:run:epoch
CKPTS=(c32:gz4fa02d:018 c64:u80w840n:021 u128:28dptwet:023 e1024:5esngzcl:026
       u128_e1024:2m4rb2c2:020 floor01:c43o9rbn:044)
SHARD_GROUPS=("0 1 2 3" "4 5 6 7" "8 9 10 11" "12 13 14 15" "16 17 18 19")

gen() {  # gen <profile> <stages> <shards> <arm> <run> <epoch>; prints the job script path
    local profile="$1" stages="$2" shards="$3" arm="$4" run="$5" ep="$6"
    local ckpt="saved_ckpts/msgym/sb_${arm}_${run}_ep${ep}.ckpt"
    FRAGNNET_PROFILE="${profile}" FRAGNNET_STAGES="${stages}" FRAGNNET_SHARDS="${shards}" \
        bash "${LAUNCHER}" fir "${MSGYM_DIR}/msgym_sb_${arm}.yml" "${ckpt}" formula test >&2
    local job="msgym_ret_formula_test_sb_${arm}_${run}_ep${ep}_${stages// /-}"
    [ -n "${shards}" ] && job="${job}_s${shards// /-}"
    echo "slurm_scripts/${job}.sh"
}

# 1. Validate inputs before generating anything.
for entry in "${CKPTS[@]}"; do
    IFS=: read -r arm run ep <<< "${entry}"
    for fp in "${MSGYM_DIR}/msgym_sb_${arm}.yml" "saved_ckpts/msgym/sb_${arm}_${run}_ep${ep}.ckpt"; do
        [ -f "${fp}" ] || { echo "missing: ${fp}" >&2; exit 1; }
    done
done

# 2. Generate: prepare once, then shard and summarize jobs per checkpoint.
IFS=: read -r arm0 run0 ep0 <<< "${CKPTS[0]}"
PREPARE_SH="$(gen fir:h100-2g.20gb prepare "" "${arm0}" "${run0}" "${ep0}")"
declare -A SHARD_SH SUMMARY_SH
for entry in "${CKPTS[@]}"; do
    IFS=: read -r arm run ep <<< "${entry}"
    scripts=()
    for group in "${SHARD_GROUPS[@]}"; do
        scripts+=("$(gen fir:h100-2g.20gb "infer score" "${group}" "${arm}" "${run}" "${ep}")")
    done
    SHARD_SH["${arm}"]="${scripts[*]}"
    SUMMARY_SH["${arm}"]="$(gen fir:h100-1g.10gb summarize "" "${arm}" "${run}" "${ep}")"
done
echo "Generated 1 prepare, $(( ${#CKPTS[@]} * ${#SHARD_GROUPS[@]} )) shard and ${#CKPTS[@]} summarize jobs."
echo "Generated scripts hold a W&B credential -- keep them out of shared records."

if [ "${MODE}" = generate ]; then
    echo "Review, then: bash $0 submit"
    exit 0
fi

# 3. Submit the chain.
prep_id="$(sbatch --parsable "${PREPARE_SH}")"
echo "prepare ${prep_id}"
for entry in "${CKPTS[@]}"; do
    IFS=: read -r arm run ep <<< "${entry}"
    ids=()
    for sh in ${SHARD_SH[${arm}]}; do
        ids+=("$(sbatch --parsable --dependency=afterok:${prep_id} "${sh}")")
    done
    deps="$(IFS=:; echo "${ids[*]}")"
    sum_id="$(sbatch --parsable --dependency=afterok:${deps} "${SUMMARY_SH[${arm}]}")"
    echo "${arm}: shards ${ids[*]} -> summarize ${sum_id}"
done
echo "Check: squeue -u \$USER"
echo "Summaries: ${FRAGNNET_RETRIEVAL_DP}/formula_test/summary_sb_*.csv"
echo "Predictions: ${FRAGNNET_RETRIEVAL_DP}/formula_test/shard_*/sb_*/predictions_test.hdf5"
