#!/usr/bin/env bash
# Generate (not submit) a Fir or Vulcan job that runs MSGYM 1.5 retrieval end to end for one
# MASSter checkpoint: scripts/ms2c/run_msgym_retrieval.py (prepare, infer, score, summarize).
# Run from the cluster checkout's login node; submit the printed script after inspection.
#
# Usage: bash scripts/manual_slurm/prepare_msgym_retrieval.sh <fir|vulcan> <config> <ckpt> \
#            <formula|mass> <val|test>
#   config  the checkpoint's training config (its spec_fp/split_dp define the queries)
#   ckpt    a repo-relative checkpoint; the job stages it onto the node
# Environment overrides: FRAGNNET_ACCOUNT, FRAGNNET_EMAIL, FRAGNNET_TIME (default 8:00:00),
#   FRAGNNET_RETRIEVAL_DP (default ${SCRATCH}/msgym_retrieval, else output/msgym_retrieval),
#   FRAGNNET_PROFILE (a setup_job_cc.py GPU profile, e.g. fir:h100-2g.20gb),
#   FRAGNNET_STAGES (subset of "prepare infer score summarize"; default all),
#   FRAGNNET_SHARDS (space-separated shard indices for infer/score; default all).
# Setting STAGES or SHARDS adds a suffix to the job name, so one checkpoint can be split into
# short shard-range jobs. Prepare is not locked: let one job finish it before the others start.
# The run directory is shared by every checkpoint of one track/fold; re-submitting resumes it.
set -euo pipefail

USAGE="Usage: bash $0 <fir|vulcan> <config> <ckpt> <formula|mass> <val|test>"
CLUSTER="${1:?$USAGE}"
CONFIG="${2:?$USAGE}"
CKPT="${3:?$USAGE}"
TRACK="${4:?$USAGE}"
FOLD="${5:?$USAGE}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

case "$CLUSTER" in
  fir) PROFILE="${FRAGNNET_PROFILE:-fir:h100-3g.40gb}"; ACCOUNT="${FRAGNNET_ACCOUNT:-rrg-dwishart}" ;;
  vulcan) PROFILE="${FRAGNNET_PROFILE:-vulcan:l40s}"; ACCOUNT="${FRAGNNET_ACCOUNT:-aip-rgreiner}" ;;
  *) echo "unknown cluster: $CLUSTER" >&2; exit 1 ;;
esac
case "$TRACK" in formula|mass) ;; *) echo "unknown track: $TRACK" >&2; exit 1 ;; esac
case "$FOLD" in val|test) ;; *) echo "unknown fold: $FOLD" >&2; exit 1 ;; esac

# Inputs read in place from the checkout (absolute: the job runs from node-local scratch).
TSV="$REPO_ROOT/data/raw/msgym_1_5/MassSpecGym1.5.tsv"
CANDS="$REPO_ROOT/data/raw/msgym_1_5/MassSpecGym1.5_retrieval_candidates_${TRACK}.json"
for fp in "$CONFIG" "$CKPT" "$TSV" "$CANDS"; do
  [ -f "$fp" ] || { echo "missing: $fp" >&2; exit 1; }
done
RUN_DP="${FRAGNNET_RETRIEVAL_DP:-${SCRATCH:-$REPO_ROOT/output}/msgym_retrieval}/${TRACK}_${FOLD}"
CKPT_STEM="$(basename "$CKPT" .ckpt)"
JOB="msgym_ret_${TRACK}_${FOLD}_${CKPT_STEM//,/_}"
EXTRA_KWARGS=()
if [ -n "${FRAGNNET_STAGES:-}" ]; then
  EXTRA_KWARGS+=("stages=${FRAGNNET_STAGES}")
  JOB="${JOB}_${FRAGNNET_STAGES// /-}"
fi
if [ -n "${FRAGNNET_SHARDS:-}" ]; then
  EXTRA_KWARGS+=("shards=${FRAGNNET_SHARDS}")
  JOB="${JOB}_s${FRAGNNET_SHARDS// /-}"
fi

python scripts/setup_job_cc.py \
  --frag_gnn_dp "$REPO_ROOT" \
  --python_script_fp scripts/ms2c/run_msgym_retrieval.py \
  --job_name "$JOB" \
  --gpu_profile "$PROFILE" \
  --cc_account "$ACCOUNT" \
  --cc_email "${FRAGNNET_EMAIL:-fw4@ualberta.ca}" \
  --time "${FRAGNNET_TIME:-8:00:00}" \
  --script_kwargs custom_fp="$CONFIG" model_ckpt="$CKPT" tsv_fp="$TSV" candidates_fp="$CANDS" \
    fold="$FOLD" run_dp="$RUN_DP" ${EXTRA_KWARGS[@]+"${EXTRA_KWARGS[@]}"} \
  --no_submit=True

bash -n "slurm_scripts/${JOB}.sh"
echo "Run directory: $RUN_DP"
echo "Submit: sbatch slurm_scripts/${JOB}.sh"
