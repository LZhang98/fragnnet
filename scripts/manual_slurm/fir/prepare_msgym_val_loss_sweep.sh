#!/usr/bin/env bash
# Run from the Fir checkout. Generate only; submit the printed script after inspection.
set -euo pipefail

SWEEP_ID="${1:?Usage: bash scripts/manual_slurm/fir/prepare_msgym_val_loss_sweep.sh SWEEP_ID}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

python scripts/setup_job_cc.py \
  --frag_gnn_dp "$REPO_ROOT" \
  --python_script_fp scripts/run_wandb_sweep.py \
  --job_name msgym_val_loss_48_40gb \
  --gpu_profile fir:h100-3g.40gb \
  --cc_account "${FRAGNNET_ACCOUNT:-rrg-dwishart}" \
  --cc_email "${FRAGNNET_EMAIL:-fw4@ualberta.ca}" \
  --time "${FRAGNNET_SWEEP_TIME:-6:00:00}" \
  --num_sweep_jobs 48 \
  --num_sweep_agents "${FRAGNNET_SWEEP_CONCURRENCY:-1}" \
  --script_kwargs w=online c=config/masster/massspecgym/msgym_val_loss_sweep_40gb.yml \
    s="$SWEEP_ID" schedule_train_size=99306 \
  --no_submit=True

bash -n slurm_scripts/msgym_val_loss_48_40gb.sh
echo 'Generated 48 one-trial jobs, one 40 GB GPU per job.'
echo 'Submit: sbatch slurm_scripts/msgym_val_loss_48_40gb.sh'
