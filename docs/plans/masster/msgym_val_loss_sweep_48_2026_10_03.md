# MASSter MSGYM validation-loss sweep, 48 trials

> **Status:** ACTIVE · **Updated:** 2026-10-13
> Sweep ran: 15 finished, 33 crashed. Results below supersede the historical preparation state;
> checkpoint recovery and seed replication remain open.

## Objective and acceptance

Run 48 random-search trials on Fir H100 `3g.40gb`, selecting the lowest validation spectral
cross-entropy. W&B [sweep tiwhun9q](https://wandb.ai/frag-gnn/frag-gnn/sweeps/tiwhun9q)
has a global `run_cap: 48`. Rank `val_primary_loss_best/mean`; save checkpoints on
`val_primary_loss_epoch/mean`, mode `min`. Keep the test split disabled during selection.
Record completed/failed trials separately; the cap counts assigned trials, not 48 successes.

## Config and search

- [Sweep definition](../../../sweep/masster/msgym_val_loss_48_40gb.yml).
- [Training base](../../../config/masster/massspecgym/msgym_val_loss_sweep_40gb.yml), copied
  from the current MSGYM k1/EMA/Fourier control. Labels, checkpoint selection, and explicit
  test disabling differed at creation; on 2026-10-08 the base also took the pow075 pair
  (`ints_transform: pow0.75` + `masster_merge_same_mz_rows: true`) after the
  [screen read](../../findings/masster/masster_msgym_formula_retrieval.md#msgym-arm-screen-val-and-test-retrieval-for-wd10-and-pow075)
  put `pow075` ahead of `ll04i00m` on test retrieval (`mspred_ent` hit@1 +0.031, CI excludes
  zero) with val cosine not ranking checkpoints for retrieval. `pm1` was deliberately not taken
  (`pow075_pm1` measured control-level val, no retrieval read); `wd10` is covered by the swept
  `weight_decay` values. `val_primary_loss` is now CE against pow0.75 targets — comparable
  within the sweep, not against pre-change runs or the screen's val table.
- Batch 256/512; encoder depth 2/3; bond-head depth 1/2/3; intensity-transformer depth 1/2/3.
- Encoder dropout 0/0.1/0.2/0.3; head, boundary-encoder, intensity-input and transformer dropout
  each 0.1/0.2/0.3; LR log-uniform 1e-4–2e-3; weight decay 0.001/0.01/0.1.
- Fixed seed 397623, width 384, U256, ranked-mask budget 4096, pow0.75-intensity Gaussian CE
  (linear until 2026-10-08), loss chunks 16 (64 until 2026-10-08, see Memory below), CUDA
  candidate backend, EMA 0.999, Fourier CE and 128 epochs. Sweep `tiwhun9q`'s W&B description
  still says "linear-intensity Gaussian CE"; its sampled parameters are unaffected.
- Sample/candidate budgets are held fixed for comparable loss; tune them in a separate screen.
  Winning settings still need seed replication before claiming improvement.

## Memory and schedule

The [previous memory measurements](masster_regularization_sweep_plan.md#slice-sizing-is-a-measured-open-question-not-a-settled-choice)
put batch512/loss64 at 25.1–26.9 GiB in another MASSter setup, and the 2026-10-03 MSGYM fourier
screen (batch512/loss64, full H100) peaked at 18.5–36.3 GB NVML across shape-identical variants —
the peak is set by worst-batch composition, so single measurements understate it. On 2026-10-08
the base config dropped `loss_batch_size` 64 → 16 (accumulation-order only, per the linked
slice-sizing section; ~0.22 GiB/lb unit above a ~12.3 GiB base at batch 512, so projected
~15.8 GiB at batch512/lb16). The sweep samples no `loss_batch_size`, so the edited base applies
to all trials at submission; sweep tiwhun9q predates the edit but has launched no trials.
These numbers still do not establish this sweep's worst-case memory: CPU construction checks
cover depth extremes (6.11M and 11.18M parameters); a real Fir forward/backward memory check
remains required. No GPU is accessible in this workspace. Inspect the largest architecture on
Fir before increasing concurrency; record OOM failures rather than silently changing individual
trials' batch or candidate budgets.

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


## Live results read, 2026-10-11

Source: W&B [tiwhun9q](https://wandb.ai/frag-gnn/frag-gnn/sweeps/tiwhun9q), all 48
run configs/summaries and unsampled epoch histories via `wandb.Api().sweep(...).runs` and
`run.scan_history(keys=["epoch", "val_primary_loss_epoch/mean",
"val_cos_sim_0.01_epoch/mean", "val_wrecall_epoch/mean"], page_size=1000)`.
Checkpoint availability checked with `run.files()` for the first three loss-ranked trials.
This is validation selection only; no new training or retrieval evaluation was performed.

### Integrity and interruptions

- All 48 actually used pow0.75, loss chunks 16, seed 397623 and test evaluation disabled.
  Batch256 uses total/warmup steps 49664/2000; batch512 uses 24832/1000.
  The online sweep description saying linear intensity is stale.
- 15 finished all 128 epochs; 33 crashed at about 2.936–2.970 hours, versus 3.236–4.946 hours
  for completed trials. This strongly suggests a time-limit interruption, but scheduler logs
  were not read, so the cause is unconfirmed. These are not demonstrated OOM failures.
- Every trial has validation history through epoch 75. Restricting all trials to epochs 0–75
  leaves every best-loss value and the loss ranking unchanged. Interrupted runs remain censored
  for later improvement; their current best checkpoints are still useful candidates.
- All best observed losses occur at epochs 11–66 (median 28); best cosine has median epoch 49.
  Only one run's best cosine is after epoch 75. Among completed runs, final loss is worse than
  best by 0.110 on average. Early stopping deserves a follow-up; changing cosine schedule length
  is a different intervention and is not validated by truncating these curves.

### Leading runs

Lower loss is better. Cosine is linear 0.01 Da. Loss-selected and cosine-selected epochs differ.

| Run | State | Best loss | Epoch | Cosine at loss epoch | Peak cosine (epoch) |
| --- | --- | ---: | ---: | ---: | ---: |
| `720jwmai` | crashed | -1.662002 | 32 | 0.597606 | 0.599811 (46) |
| `6egv3347` | crashed | -1.651483 | 28 | 0.586189 | 0.589625 (59) |
| `86f3wm2f` | finished | -1.648701 | 25 | 0.591627 | 0.592739 (58) |
| `7ms4lgh2` | crashed | -1.645117 | 28 | 0.591067 | 0.593777 (47) |
| `1t7bverf` | finished | -1.642208 | 33 | 0.591194 | 0.592072 (35) |

`720jwmai` and `6egv3347` have no W&B checkpoint files. `86f3wm2f` has
`ckpt/model-epoch=025.ckpt` and `ckpt/last.ckpt` in W&B; neither was downloaded here.
The runner uploads checkpoints after successful training, explaining why interrupted runs
can have metrics without uploaded weights. Cluster-local checkpoint survival is unknown.

The previous pow075 run `q3oxbpgx` has best loss -1.590353 at epoch 25 and best cosine 0.578978
at epoch 55. Thus the sweep leader improves observed best loss by 0.07165 and peak cosine by
0.02083; the best finished trial improves peak cosine by 0.01376. Both train against pow0.75,
although the previous run used loss chunks 64 and selected its saved checkpoint by cosine.
These are selected single-run validation differences, not confirmed test/retrieval gains.

### What the random search suggests

Unadjusted group means over all trials (same best-loss results through epoch 75):

| Encoder dropout | Trials | Mean best loss | Mean peak cosine through epoch 75 |
| --- | ---: | ---: | ---: |
| 0.0 | 8 | -1.4526 | 0.5516 |
| 0.1 | 10 | -1.5504 | 0.5740 |
| 0.2 | 16 | -1.5868 | 0.5796 |
| 0.3 | 14 | -1.6089 | 0.5852 |

- Encoder dropout 0.2–0.3 and higher LR are the strongest follow-up candidates. LR bins
  <=3e-4 / (3e-4,7e-4] / >7e-4 have 14/16/18 trials and mean best losses
  -1.4988/-1.5754/-1.6027. LR versus loss Spearman correlation is -0.676; LR versus encoder
  dropout is only 0.021. An exploratory additive standardized regression over the 11 sampled
  parameters (log10 LR and weight decay) retains dropout and LR as the largest coefficients.
- Batch256 has mean peak cosine 0.5808 versus 0.5705 for batch512. Weight decay 0.1 is also
  promising; these remain confounded random-search associations, not matched ablations.
- Do not infer a depth winner. The apparent marginal benefit of bond depth reverses in the
  additive adjustment, and leading runs use different depths. Other dropout settings are
  less clearly resolved. All trials share one seed, with 11 jointly varied parameters.

Leader `720jwmai`: batch256, LR 0.0015548897423024086, weight decay 0.1; encoder/bond/intensity
transformer depths 3/2/3; encoder/head/boundary/intensity-input/transformer dropout
0.2/0.3/0.2/0.2/0.1. Best completed `86f3wm2f`: batch256, LR 0.0005775193973727379,
weight decay 0.1; depths 3/3/2; dropouts 0.3/0.2/0.3/0.2/0.1.

### Next checks

Recover `720jwmai` epoch 32 from the cluster if present; otherwise reproduce its recipe.
Use available `86f3wm2f` epoch 25 as the immediate evaluation candidate. Run validation retrieval
and re-select calibration per checkpoint before one held-out test read. Replicate the leader and
pow075 control on additional matched seeds. Inspect scheduler logs before changing allocation
limits; preserve best checkpoints during training in any separately authorized reliability fix.
No submission, checkpoint download or code change was made in this analysis.

### Leader re-run prepared, 2026-10-12

[`msgym_sweep_leader_720jwmai.yml`](../../../config/masster/massspecgym/msgym_sweep_leader_720jwmai.yml)
re-trains `720jwmai`'s recipe (seed 397623, 128 epochs, batch 256, 49664/2000 steps). Every
top-level key matches the W&B run config. Nested `spec_params` differ only by new off-by-default
RT/CCS keys. It builds on `refactor/fragnnet-masster-reorg` (10.14M parameters, CPU). Not submitted.
Batch-256 trials reached about epoch 76 in about 2.95 h, so use a wall time of at least 8 h.

Matched control:
[`msgym_sweep_base_control.yml`](../../../config/masster/massspecgym/msgym_sweep_base_control.yml)
is the sweep base run unchanged under its own W&B name, so it uses the same seed, pow0.75 CE,
loss selection and disabled test split as the leader (9.23M parameters, CPU build). Not submitted.
The leader's +0.91M parameters come from depth only: encoder +1 layer (+0.89M), intensity
transformer +1 (+1.05M), bond head -1 (-1.04M).
Launcher: [`submit_msgym_sweep_leader.sh`](../../../scripts/manual_slurm/fir/submit_msgym_sweep_leader.sh).

### Leader re-run: formula test retrieval, 2026-10-13

W&B `f44tzxoc` (`msgym_masster_sweep_leader_720jwmai`, Fir) finished 128 epochs in 4.98 h.
Loss selection kept epoch 19 (val primary loss −1.6651, val `cos_sim_0.01` 0.5937). Val cosine
peaked at 0.5966 at epoch 81 and ended at 0.5911. Checkpoint:
`saved_ckpts/msgym/msgym_masster_sweep_leader_720jwmai_f44tzxoc.ckpt`.

The test was run on the formula track, test fold, 9,954 queries, with
[the retrieval runner](../../guides/run_msgym_retrieval.md). It was scored against `pow075`
(`q3oxbpgx`@55), re-run the same day on the same shards. That re-run reproduces the
[arm screen finding](../../findings/masster/masster_msgym_formula_retrieval.md#msgym-arm-screen-val-and-test-retrieval-for-wd10-and-pow075)
to three decimals. Expected hit@1 (hit@5 / hit@20), with the paired difference and 95% molecule
bootstrap:

| score | `f44tzxoc` @19 | `pow075` @55 | f44 − pow075 |
| --- | ---: | ---: | --- |
| `cos_sim_0.01` | 0.365 (0.719 / 0.905) | 0.362 (0.722 / 0.898) | +0.003 [−0.010, +0.015] |
| `cos_hun` | 0.370 | 0.368 | +0.001 [−0.011, +0.014] |
| `cos_sim_sqrt_0.01` | 0.470 (0.809 / 0.945) | 0.467 (0.808 / 0.940) | +0.003 [−0.010, +0.015] |
| `mspred_ent` | 0.523 (0.844 / 0.956) | 0.516 (0.843 / 0.956) | +0.007 [−0.006, +0.021] |
| `loglik` | 0.503 (0.837 / 0.960) | 0.504 (0.836 / 0.957) | −0.001 [−0.014, +0.013] |
| `loglik_cal` | 0.545 (0.866 / 0.969) | 0.542 (0.862 / 0.970) | +0.003 [−0.010, +0.017] |

- **The leader ties `pow075` on every score.** Its +0.015 val cosine (0.594 vs 0.579) does not
  carry into retrieval, which is the same pattern as `wd10`.
- **Per instrument,** `loglik_cal` is ORB 0.600 vs 0.603 and QTOF 0.257 vs 0.218 (1,580 queries,
  not tested paired).
- **`loglik_cal` was not re-selected on val** for either checkpoint. It uses the `ll04i00m`
  parameters.
- **Single seed, single checkpoint.** The matched control
  (`msgym_sweep_base_control.yml`) has not been retrieval-tested.

Artifacts are in `output/msgym_retrieval/formula_test/`: `run_f44tzxoc_2026_10_13.log`,
`run_pow075_rerun_2026_10_13.log`, `summary_<ckpt stem>.csv`, `compare_f44tzxoc_pow075.csv`,
and `shard_NNN/<ckpt stem>/` (predictions and scores; the `pow075` shard outputs were deleted
2026-10-17, `f44tzxoc` remains).
The 2026-10-07 per-candidate outputs of `wd10` and `ll04i00m` in that directory were deleted
by 2026-10-13 01:03; only `pow075` was regenerated.
