# Unmerged MASSter generalization audit

> **Status:** FINDING · **Updated:** 2026-09-16 (America/Chicago)
> Live W&B histories support a growing loss generalization gap in mature runs, not a universal architecture-invariant ceiling or a proven dropout remedy.

## Scope and method

Read-only W&B query completed 2026-09-17 04:49:52 UTC, project `frag-gnn/frag-gnn`,
filter `config.model_type = masster`: 197 runs, including 27 with
`spec_params.merge = False` and 170 merged. All 27 unmerged histories were fetched with
`scan_history`, without requiring train and validation keys on the same history row.
Non-null records were combined by epoch. The table below includes every unmerged run;
epoch numbers are zero-based. W&B marked 26 crashed and one running. Crash status alone
does not establish scheduler wall-time as the cause; scheduler logs were not fetched.

Source checkout: `a4c87b64`, clean at audit start. Current-code findings do not prove
historical behavior for every run revision. Scratch evidence (not portable):
`/tmp/unmerged_audit_index.json`, `/tmp/unmerged_audit_histories.json`,
`/tmp/unmerged_audit_stats.json`; query script `/tmp/audit_unmerged_current.py`.
Run links and the essential measurements are preserved here.

## What the curves establish

Of 16 runs with at least 15 logged epochs, 14 end with training loss below its value
at the best validation-loss epoch and validation loss above its minimum. Their median
validation-loss regression, including the two with zero regression, is 0.01449.
This minimum-to-final comparison is descriptive and selects an extreme; it is not a
significance test or a reliable estimate of how much dropout would recover.

The three longest runs provide stronger evidence of diminishing generalization:

| Run | Epoch interval | Train spectral CE | Validation spectral CE | Validation cosine |
|---|---|---|---|---|
| [qnjttwod](https://wandb.ai/frag-gnn/frag-gnn/runs/qnjttwod) | 10 → 65 | −2.8835 → −3.1566 | −2.4755 → −2.4824 | 0.6746 → 0.6872 |
| [bu8mzm40](https://wandb.ai/frag-gnn/frag-gnn/runs/bu8mzm40) | 10 → 55 | −2.9166 → −3.1902 | −2.4473 → −2.4503 | 0.6738 → 0.6865 |
| [9gms37kj](https://wandb.ai/frag-gnn/frag-gnn/runs/9gms37kj) | 10 → 53 | −2.9176 → −3.1891 | −2.4519 → −2.4615 | 0.6733 → 0.6885 |

Train CE improves approximately 0.27 nats while validation CE improves only
0.003–0.010. Final spectral-CE gaps are 0.6743, 0.7399, and 0.7276 respectively,
not a universal 0.42–0.58 budget. Five-epoch averaging gives the same broad picture:
epochs 10–14 versus the final five show train CE improvements of 0.229–0.236,
versus validation improvements of only 0.0024–0.0226.

However, their cosine continues to improve after epoch 10. In the final ten epochs,
comparing consecutive five-epoch windows, only `bu8mzm40` worsens in validation CE
(+0.0063); `qnjttwod` and `9gms37kj` improve slightly. Therefore the defensible
interpretation is substantial training-specific improvement and mild/noisy late loss
overfitting, not uniform validation collapse or proof that more training always hurts.
Training metrics average changing weights in training mode, whereas validation uses
end-of-epoch weights in evaluation mode; these are not matched checkpoint evaluations.

> The within-run train/validation spectral-CE figure referenced here was never checked into
> `docs/figures/`. The per-epoch numbers it plotted are in the tables below.

## Where the gap appears, and plausible causes

**It is not isolated to the final intensity head.** In `bu8mzm40`, epoch 10 → 55,
base-distribution train CE improves −2.4824 → −2.7867 while validation improves
−2.1494 → −2.2264. The final head's gain over this base remains 0.4342 → 0.4035
on training but declines 0.2980 → 0.2239 on validation. `9gms37kj` shows the same
pattern. The code defines `base_ce_gain = base_spec_ce - spec_ce` on the same support.
Both the base distribution and the final correction generalize less well than they fit
training; this is not a causal isolation of individual modules, which share features.

**Optimization and regularization have barely been tested.** All 27 use LR 0.001,
no scheduler, weight decay 0.01, head dropout 0.1, and intensity-transformer dropout 0.1.
Molecular dropout is 0.1 in 24 and 0.05 in three Graphormer runs. There is no matched
regularization or LR-decay sweep. Repeated spectra with `train_sample_weight: none`
(all 27) give molecules with more spectra greater aggregate training weight. This
makes correlated observations and weighting plausible contributors, but does not
establish that they cause the plateau. Effective batch sizes also differ; epoch
counts are not universally comparable optimizer budgets. For example, `qnjttwod`
advances about 1,124 optimizer steps per epoch versus 562 for `bu8mzm40`.

**Missing conditioning and target variability remain hypotheses.** The older 26 runs
record `spec_params.ace: False, nce: True`; the current run enables both. The heads
use the `spec_params` switches, rather than the separate top-level `use_ace` value.
There is no controlled comparison separating dual CE, corrected chemistry, objective,
and architecture in the newest run. No new data census or replicate-noise measurement
was performed here. Averaging 22 spectra does not establish 4.7× less label noise:
that square-root argument requires suitable independence and equal-variance assumptions,
and spectra at distinct collision energies need not measure the same target.

Candidate coverage does not deteriorate in the three long runs: candidate weighted
recall increases 0.8760 → 0.8885, 0.9077 → 0.9133, and 0.9068 → 0.9127 over the
intervals above. Weighted precision improves too. These observations oppose a
coverage-collapse explanation; they do not prove coverage is irrelevant or equate
candidate weighted recall with an oracle cosine.

## Corrections to the supplied summary and code claims

- The best-cosine range over **all 27** is 0.3803–0.6933, not 0.688–0.693.
  Even among runs with at least 15 epochs it is 0.6273–0.6933. The narrow band
  describes a successful subset, not demonstrated architecture invariance.
- [s3gy1ro4](https://wandb.ai/frag-gnn/frag-gnn/runs/s3gy1ro4) was still running
  at epoch 8, cosine 0.693215, with its best validation loss at that epoch.
  It is not yet evidence of a late overfitting plateau.
- OOS was already disabled in **eight** of the 27 runs, including the newest run;
  it is not a change unique to that run relative to all previous 26. Nineteen use
  `oos_loss=True / oos_tolerance_multiple=1`; eight use `False / -1`.
  Cross-run total losses also differ through base-CE and proposal terms. Compare
  within-run trajectories and use matched controls to attribute improvements.
- All 27 disable train auxiliary metrics; no train cosine was found in their summaries
  or full histories. Train loss already supplies evidence about loss generalization,
  but the size of the train–validation cosine gap is unknown.
- `GroupSampler` does **not** freeze merely because its `set_epoch` is not called:
  `SpectrumPL._seed_dataloader` reseeds its generator and invokes
  `_pre_compute_batches` through the normal non-curriculum path. A direct current-code
  probe of that hook changed membership across epochs (only 3 of 20 selected indices
  shared). Its metadata initialization does materialize dataset items. None of the
  27 enables `group_sampler`, so this is not their observed failure mechanism.
- MASSter inherits `SpectrumPL.configure_optimizers`, which does not forward
  `lr_schedule_type`, `lr_total_steps`, or `lr_min_ratio` to `build_lr_scheduler`.
  Enabling the scheduler therefore selects its default **exponential** schedule,
  even if the config asks for cosine. FraGNNet's override does forward these keys.
  Fix/verify that wiring before proposing a MASSter cosine-schedule experiment.
- `mol_dropout`, `masster_head_dropout`, and
  `masster_intensity_transformer_dropout` are the wired MASSter knobs;
  `mlp_dropout` and `frag_dropout` are not its head regularizers.
- `ef7fl5c6` is merged (best 0.751542); `8v0i0fq7` is merged (best 0.800989).
  The older unmerged-anchor description in project state and the improvement plan
  is stale. Neither is an unmerged baseline.

## Next discriminating measurement

Evaluate a recovered, mature checkpoint on fixed training and validation subsets in
**evaluation mode**, using identical cosine/JSS, spectral CE, mass tolerances, and
CE/adduct/instrument strata. Include both row-weighted and molecule-weighted summaries.
This directly measures the same-checkpoint gap without dropout or moving-weight confounds.
For future training, `compute_train_auxiliary_metrics: True` is useful instrumentation,
but its online train metric alone is not this matched evaluation.

Then compare a fixed-objective control against LR decay and a separate moderate increase
in the actually wired dropout settings, with equal optimizer budgets and multiple seeds.
A train≈validation result alone does not prove selection limitation or that dropout hurts;
underfitting, optimization limits, and missing conditioning can also produce it.
A train≫validation result motivates regularization experiments but does not identify their
best form or guarantee a gain. No training, checkpoint evaluation, configuration change,
or model-code change was performed during this audit.

## Complete unmerged run census

The loss columns describe each run's own objective, not comparable absolute scales.
Zero regression includes runs ending at their minimum; it is not evidence that the run
has converged. Short runs must not be treated as ceiling measurements.

| Run | Epochs logged | Best cosine (epoch) | Loss-min epoch | Final loss − minimum | Train loss change since val minimum |
|---|---:|---:|---:|---:|---:|
| [993apayd](https://wandb.ai/frag-gnn/frag-gnn/runs/993apayd) | 11 | 0.6437 (9) | 9 | +0.00208 | -0.01521 |
| [frnowiw6](https://wandb.ai/frag-gnn/frag-gnn/runs/frnowiw6) | 26 | 0.6273 (22) | 22 | +0.01542 | -0.03775 |
| [eyzhkms9](https://wandb.ai/frag-gnn/frag-gnn/runs/eyzhkms9) | 19 | 0.6375 (16) | 16 | +0.00790 | -0.00550 |
| [0w4w7hc1](https://wandb.ai/frag-gnn/frag-gnn/runs/0w4w7hc1) | 16 | 0.6442 (14) | 14 | +0.05863 | -0.01611 |
| [mekhqj66](https://wandb.ai/frag-gnn/frag-gnn/runs/mekhqj66) | 15 | 0.6435 (14) | 14 | +0.00000 | +0.00000 |
| [wzwawe12](https://wandb.ai/frag-gnn/frag-gnn/runs/wzwawe12) | 23 | 0.6383 (19) | 19 | +0.04200 | -0.03129 |
| [qnjttwod](https://wandb.ai/frag-gnn/frag-gnn/runs/qnjttwod) | 66 | 0.6891 (64) | 34 | +0.00972 | -0.07197 |
| [d1iv8151](https://wandb.ai/frag-gnn/frag-gnn/runs/d1iv8151) | 26 | 0.6933 (20) | 20 | +0.01279 | -0.04300 |
| [gditriob](https://wandb.ai/frag-gnn/frag-gnn/runs/gditriob) | 25 | 0.6914 (20) | 14 | +0.03575 | -0.11030 |
| [9gms37kj](https://wandb.ai/frag-gnn/frag-gnn/runs/9gms37kj) | 54 | 0.6889 (44) | 36 | +0.01926 | -0.05880 |
| [bu8mzm40](https://wandb.ai/frag-gnn/frag-gnn/runs/bu8mzm40) | 56 | 0.6883 (34) | 35 | +0.02345 | -0.06761 |
| [3z9y6yu9](https://wandb.ai/frag-gnn/frag-gnn/runs/3z9y6yu9) | 31 | 0.6904 (30) | 16 | +0.01862 | -0.10509 |
| [er846qd7](https://wandb.ai/frag-gnn/frag-gnn/runs/er846qd7) | 16 | 0.6864 (14) | 10 | +0.01681 | -0.08320 |
| [3oqibfwg](https://wandb.ai/frag-gnn/frag-gnn/runs/3oqibfwg) | 16 | 0.6907 (15) | 15 | +0.00000 | +0.00000 |
| [msqh275h](https://wandb.ai/frag-gnn/frag-gnn/runs/msqh275h) | 9 | 0.6796 (8) | 8 | +0.00000 | +0.00000 |
| [youaraa3](https://wandb.ai/frag-gnn/frag-gnn/runs/youaraa3) | 16 | 0.6885 (14) | 14 | +0.01357 | -0.01456 |
| [gwze40e8](https://wandb.ai/frag-gnn/frag-gnn/runs/gwze40e8) | 17 | 0.6470 (16) | 15 | +0.00074 | -0.01659 |
| [b2lizgoe](https://wandb.ai/frag-gnn/frag-gnn/runs/b2lizgoe) | 1 | 0.5920 (0) | 0 | +0.00000 | +0.00000 |
| [ohx1fino](https://wandb.ai/frag-gnn/frag-gnn/runs/ohx1fino) | 1 | 0.5498 (0) | 0 | +0.00000 | +0.00000 |
| [dkoqkmdt](https://wandb.ai/frag-gnn/frag-gnn/runs/dkoqkmdt) | 1 | 0.5758 (0) | 0 | +0.00000 | +0.00000 |
| [vt9avb13](https://wandb.ai/frag-gnn/frag-gnn/runs/vt9avb13) | 4 | 0.6175 (3) | 3 | +0.00000 | +0.00000 |
| [k4ui3h2m](https://wandb.ai/frag-gnn/frag-gnn/runs/k4ui3h2m) | 16 | 0.6378 (15) | 10 | +0.01258 | -0.11919 |
| [jdgpkca4](https://wandb.ai/frag-gnn/frag-gnn/runs/jdgpkca4) | 6 | 0.4298 (4) | 4 | +0.11495 | -0.06331 |
| [k1zsnr5l](https://wandb.ai/frag-gnn/frag-gnn/runs/k1zsnr5l) | 3 | 0.3859 (1) | 2 | +0.00000 | +0.00000 |
| [37az0wir](https://wandb.ai/frag-gnn/frag-gnn/runs/37az0wir) | 4 | 0.4406 (3) | 3 | +0.00000 | +0.00000 |
| [ax4ikteo](https://wandb.ai/frag-gnn/frag-gnn/runs/ax4ikteo) | 3 | 0.3803 (2) | 2 | +0.00000 | +0.00000 |
| [s3gy1ro4](https://wandb.ai/frag-gnn/frag-gnn/runs/s3gy1ro4) | 9 | 0.6932 (8) | 8 | +0.00000 | +0.00000 |
