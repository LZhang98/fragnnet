#!/bin/bash
# Formula-test retrieval for the MSGYM rewire screen: one job per checkpoint on a 2g.20gb slice
# that infers, scores and summarizes all 20 query shards. Run from the Fir checkout's login node.
# Plan: docs/plans/masster/masster_rewire_rules_training_plan.md#msgym-screen-2026-10-15
#
# Usage: bash scripts/manual_slurm/fir/submit_msgym_rw_retrieval.sh [generate|submit] [arm ...]
#   generate (default) writes the job scripts and checks them with bash -n; submits nothing.
#   submit   generates, then submits one job per checkpoint.
#   arm ...  optional subset of control a0 a0d1 a0null (default: all).
#
# Rewire arms re-parse each candidate's SMILES and index the graph with it; the shared shards
# store the raw SMILES next to the standardized mol, which disagree on atom order for ~1.2% of
# candidates and crash rewire expansion. Every arm, control included, therefore reads an aligned
# copy, <FRAGNNET_RETRIEVAL_DP>/aligned/formula_test, built once from the shared formula_test
# shards (prepared by submit_msgym_sb_retrieval.sh) with scripts/ms2c/align_retrieval_shards.py.
#
# Checkpoints are the best-val_primary_loss epochs of the 2026-10-16 03:48 UTC runs, expected at
# saved_ckpts/msgym/rw_<arm>_<run>_ep<NNN>.ckpt (W&B run files ckpt/model-epoch=<NNN>.ckpt).
# Rewire arms match rules on every candidate SMILES at inference (MassterPL LRU cache): local a0
# shard 0 took 7:39 inference vs 2:49 for f44tzxoc. loglik_cal keeps the ll04i00m calibration, as
# in the f44tzxoc / pow075 / sb reads.
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
PROFILE=fir:h100-2g.20gb
export FRAGNNET_TIME="${FRAGNNET_TIME:-1:00:00}"
# Shards, per-checkpoint scores and the prediction HDF5s all go to scratch. Never fall back to
# output/ under /project, where rrg-dwishart has no quota (EDQUOT).
export FRAGNNET_RETRIEVAL_DP="${FRAGNNET_RETRIEVAL_DP:-/scratch/feiw/frag-gnn-data/msgym}"
mkdir -p "${FRAGNNET_RETRIEVAL_DP}"
SRC_DP="${FRAGNNET_RETRIEVAL_DP}/formula_test"
ALIGNED_ROOT="${FRAGNNET_RETRIEVAL_DP}/aligned"
RUN_DP="${ALIGNED_ROOT}/formula_test"

# arm:run:epoch
ALL_CKPTS=(control:rpa6uq8k:019 a0:oj85dppl:019 a0d1:16gpi651:014 a0null:47t8i9s0:014)
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
    FRAGNNET_RETRIEVAL_DP="${ALIGNED_ROOT}" FRAGNNET_PROFILE="${PROFILE}" FRAGNNET_STAGES="${stages}" \
        bash "${LAUNCHER}" fir "${MSGYM_DIR}/msgym_rw_${arm}.yml" \
        "saved_ckpts/msgym/rw_${arm}_${run}_ep${ep}.ckpt" formula test >&2
    echo "slurm_scripts/msgym_ret_formula_test_rw_${arm}_${run}_ep${ep}_${stages// /-}.sh"
}

# 1. Validate inputs before generating anything. A missing checkpoint is linked from the training
#    job's W&B run directory in this checkout (jobs symlink wandb/ here; delete_checkpoints is off),
#    else downloaded from the W&B run files (login nodes have internet).
mkdir -p saved_ckpts/msgym
for entry in "${CKPTS[@]}"; do
    IFS=: read -r arm run ep <<< "${entry}"
    [ -f "${MSGYM_DIR}/msgym_rw_${arm}.yml" ] || { echo "missing: ${MSGYM_DIR}/msgym_rw_${arm}.yml" >&2; exit 1; }
    ckpt="saved_ckpts/msgym/rw_${arm}_${run}_ep${ep}.ckpt"
    if [ ! -f "${ckpt}" ]; then
        found="$(ls -1 wandb/run-*-"${run}"/files/ckpt/model-epoch="${ep}".ckpt 2>/dev/null | tail -n 1 || true)"
        if [ -n "${found}" ]; then
            echo "linking ${ckpt} -> ${found}" >&2
            ln -s "${REPO_ROOT}/${found}" "${ckpt}"
        fi
    fi
    if [ ! -f "${ckpt}" ]; then
        echo "fetching ${ckpt} from W&B run ${run}" >&2
        python - "${run}" "${ep}" "${ckpt}" <<'EOF'
import os, shutil, sys, tempfile
import wandb
run_id, ep, out = sys.argv[1:]
with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
    f = wandb.Api(timeout=120).run(f"frag-gnn/frag-gnn/{run_id}").file(f"ckpt/model-epoch={ep}.ckpt")
    shutil.move(f.download(root=tmp, replace=True).name, out)
EOF
        [ -s "${ckpt}" ] || { echo "missing: ${ckpt} (W&B download failed)" >&2; exit 1; }
    fi
done

# 2. Generate one job per checkpoint, on the aligned shards.
if [ ! -f "${RUN_DP}/queries.csv" ]; then
    echo "missing aligned shards ${RUN_DP}. Build them once (CPU, a few minutes) with:" >&2
    echo "  python scripts/ms2c/align_retrieval_shards.py ${SRC_DP} ${RUN_DP}" >&2
    [ -f "${SRC_DP}/queries.csv" ] || echo "  after preparing ${SRC_DP}: bash scripts/manual_slurm/fir/submit_msgym_sb_retrieval.sh submit c32" >&2
    exit 1
fi
JOB_SH=()
for entry in "${CKPTS[@]}"; do
    IFS=: read -r arm run ep <<< "${entry}"
    JOB_SH+=("$(gen "infer score summarize" "${arm}" "${run}" "${ep}")")
done
echo "Generated ${#JOB_SH[@]} checkpoint jobs (${PROFILE}, ${FRAGNNET_TIME})."
echo "Generated scripts hold a W&B credential -- keep them out of shared records."

if [ "${MODE}" = generate ]; then
    echo "Review, then: bash $0 submit $*"
    exit 0
fi

# 3. Submit.
for sh in "${JOB_SH[@]}"; do
    echo "$(sbatch --parsable "${sh}")  ${sh}"
done
echo "Check: squeue -u \$USER"
echo "Summaries: ${RUN_DP}/summary_rw_*.csv"
echo "Predictions: ${RUN_DP}/shard_*/rw_*/predictions_test.hdf5"
