# MASSter MSGYM validation-loss sweep, 48 trials

> **Status:** PREPARED · **Updated:** 2026-10-03
> W&B sweep created; Fir submission awaits the SSH login and checkout path. No training launched.

## Objective and acceptance

Run 48 random-search trials on Fir H100 `3g.40gb`, selecting the lowest validation spectral
cross-entropy. W&B [sweep tiwhun9q](https://wandb.ai/frag-gnn/frag-gnn/sweeps/tiwhun9q)
has a global `run_cap: 48`. Rank `val_primary_loss_best/mean`; save checkpoints on
`val_primary_loss_epoch/mean`, mode `min`. Keep the test split disabled during selection.
Record completed/failed trials separately; the cap counts assigned trials, not 48 successes.

## Config and search

- [Sweep definition](../../../sweep/masster/msgym_val_loss_48_40gb.yml).
- [Training base](../../../config/masster/massspecgym/msgym_val_loss_sweep_40gb.yml), copied
  from the current MSGYM k1/EMA/Fourier control. Only labels, checkpoint selection, and explicit
  test disabling differ before sampling.
- Batch 256/512; encoder depth 2/3; bond-head depth 1/2/3; intensity-transformer depth 1/2/3.
- Encoder dropout 0/0.1/0.2/0.3; head, boundary-encoder, intensity-input and transformer dropout
  each 0.1/0.2/0.3; LR log-uniform 1e-4–2e-3; weight decay 0.001/0.01/0.1.
- Fixed seed 397623, width 384, U256, ranked-mask budget 4096, linear-intensity Gaussian CE,
  loss chunks 64, CUDA candidate backend, EMA 0.999, Fourier CE and 128 epochs.
- Sample/candidate budgets are held fixed for comparable loss; tune them in a separate screen.
  Winning settings still need seed replication before claiming improvement.

## Memory and schedule

The [previous memory measurements](masster_regularization_sweep_plan.md#slice-sizing-is-a-measured-open-question-not-a-settled-choice)
put batch512/loss64 at 25.1–26.9 GiB in another MASSter setup. They do not establish this
sweep's worst-case memory. CPU construction checks cover depth extremes (6.11M and 11.18M
parameters); a real Fir forward/backward memory check remains required. No GPU is accessible
in this workspace. Inspect the largest architecture on Fir before increasing concurrency;
record OOM failures rather than silently changing individual trials' batch or candidate budgets.

The agent receives `--schedule_train_size 99306`. After W&B merges sampled values, the runner
sets total cosine steps to `128 * ceil(ceil(99306 / batch) / accumulation)` and preserves the
control's warmup fraction. Batch512 uses 24,832 total/1,000 warmup steps; batch256 uses
49,664/2,000. Both derived fields are written back to W&B and checkpoint hparams.
The runner rejects dropped final batches, non-simple samplers, and a different actual train count.
Existing runs without this option retain their schedule behavior.

## Launch on Fir

Follow the [cluster guide](../../guides/submit_cluster_runs.md) for environment, source staging,
credentials and data checks. Use the actual Fir checkout; the locally generated ignored script
contains workstation paths and must be regenerated there. The launcher defaults to one concurrent
trial, six-hour wall time, account `rrg-dwishart`, and the existing Fir email; override with
`FRAGNNET_SWEEP_CONCURRENCY`, `FRAGNNET_SWEEP_TIME`, `FRAGNNET_ACCOUNT`, `FRAGNNET_EMAIL` as needed.
One array task runs one W&B trial on one 40 GB GPU. Six hours is an allocation ceiling, not a
measured runtime. Do not pass a resumable `-j` to these fresh sweep trials.

```bash
bash scripts/manual_slurm/fir/prepare_msgym_val_loss_sweep.sh tiwhun9q
# Inspect resource directives, staging, environment and final agent command.
sbatch slurm_scripts/msgym_val_loss_48_40gb.sh
```

Before submission record executed Git revision, dirty source/config identities, environment,
and input fingerprints under the [session workflow](../../workflow/session.md). The source
revision when prepared was `f707710354e6238ba91f79d38bb00a66df9172f2`, plus this working-tree
change. Input identities are recorded in the
[control header](../../../config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_k1_ema999_fourier.yml);
they have not been reverified on Fir. W&B/offline artifacts use the repository `wandb/` link;
scheduler logs go to `jobs/`; best checkpoints upload to each W&B run. Preserve exact source
through queued-job staging. The existing retrieval finding edit was present before this task.

## Validation and next check

119 focused tests passed, then all 7 sweep tests passed after adding the primary-loss metric
regression. Ruff passed. Both depth extremes construct on CPU and register primary loss as a
minimizing best metric. Local generation verified array `0-47%1`, one H100 `3g.40gb`, 140G host
memory, six CPUs, online W&B, correct base, sweep ID and schedule count. Both launcher and
generated script pass `bash -n`. No GPU training, validation-set evaluation, or full suite ran.

Next: obtain Fir login/checkout, stage the prepared source, measure worst-case GPU memory,
submit the array, and record its job ID, actual source/input identity and first trial result.
