# Spectraverse MASSter: short-schedule and clean-train arms

> **Status:** PREPARED · **Updated:** 2026-10-03 · Six configs (3 arms × cv0/cv1) and the
> clean split built and resolved. Nothing has been submitted.

## Why

On SPV, val `cos_sim_0.01` peaks at epoch 15–18 and declines to epoch 39. About 12% of training
spectra have a subformula ceiling ≤ 0.5, and the model cannot fit them even in training (0.05)
([oracle finding](../../findings/masster/spectraverse_oracle_ceiling_2026_10_03.md)). Both are
SPV-specific: NIST20 and MSGYM val plateau instead, and their targets are clean. On NIST20 the
generic levers are already null (weight decay `w1` +0.0010; DreaMS/JEPA
[closed null](masster_jepa_embedding_plan.md#gate-1-result-2026-10-03)). These arms are SPV
cleanup and are not expected to close the 0.13 train-to-val gap.

## Arms

| Dir (`config/spectraverse/…`) | Change against `masster/cv{0,1}.yml` |
| --- | --- |
| `masster_c0rerun` | Control. W&B identity only. |
| `masster_short18` | `min/max_epochs` 18, `lr_total_steps` 13,626 / 13,590 (18 × 757 / 755). |
| `masster_subceil05` | `split_dp` `data/split/spectraverse_benchmark_subceil05/cv{f}`, `lr_total_steps` 26,720 / 26,440 (40 × 668 / 661). |

A fresh control is needed because the 2026-09-27 runs `ggifzj9m` / `u46v0k0g` trained before
`fill_missing_ace` (44% of rows have no CE), the ORB rename and the `cos_hun_inst_tol`
checkpoint metric. A distinct `wandb_name` keeps the job from resuming an old checkpoint.

**Clean split.** `preproc_scripts/spectraverse/clean_train_split.py --folds 0 1`. The per-spectrum
peak-level subformula ceiling is in `data/proc/spectraverse/subformula_ceiling.csv` (all
488,797 spectra; 3 Sn spectra NaN). It agrees with the binned `sub_inst` oracle on 2,047 of
2,048 sampled spectra (r = 0.998).

| Fold | Train before | After | Dropped | Train molecules before → after |
| --- | ---: | ---: | ---: | --- |
| cv0 | 387,399 | 341,563 | 45,836 (11.8%) | 33,947 → 33,676 |
| cv1 | 386,134 | 337,927 | 48,207 (12.5%) | 33,770 → 33,505 |

Across all spectra, the share dropped by adduct is `[M+K]+` 41.8%, `[M+Na]+` 33.2%, `[M+H]+`
10.9%, `[M+Cl]-` 9.7%, `[M+NH4]+` 6.3%, `[M-H]-` 3.5%, the rest ≤ 3.3%. Val and test ids are
byte copies of the source split. Train `train_ids.csv` md5: cv0 `43749a16ab45920ff853ce9cd2817c0d`,
cv1 `cb46ac1c08eab2a129066806090a2aee`. `data/` is ignored, so the split dir must be copied to
Fir.

## Validation

All six configs resolve through `load_config`. Against the parent fold config, the resolved
differences are exactly the keys in the table plus W&B name, group and tag, and each
`lr_total_steps` equals epochs × ceil(train / 512). No training smoke was run: the workstation GPU
was in use by another session, and the configs change no code path that the 2026-09-25 smokes
did not cover.

## Launch (Fir, needs MFA)

1. Copy `data/split/spectraverse_benchmark_subceil05/` to the Fir checkout and check the md5s.
2. For each arm dir, run
   `setup_job_cc.py --cv_config_dp config/spectraverse/<arm> --cv_start 0 --cv_end 1 --cv_use_job_id`
   on `fir:h100-3g.40gb` (12 h; 6 h is enough for `masster_short18`). See the
   [submit guide](../../guides/submit_cluster_runs.md#4-cross-validation-and-existing-batch-launchers).
3. Record job IDs and W&B runs here.

## Read

Primary: best-epoch `val_cos_sim_0.01_epoch/mean` per fold against `masster_c0rerun`, plus a
last-5-epoch tail mean. Promotion needs both folds positive. The NIST20 two-run (seed) spread is
0.0042; SPV has no replicate yet, so treat a gain smaller than that as unresolved. Also log the wall
clock, since `masster_short18` costs 45% of the control. If `masster_subceil05` wins, check that it holds on the
held-out Na/K spectra (it must not just abandon them) by rerunning
`scripts/debug/spectraverse/spectraverse_oracle_ceiling.py` per adduct.
