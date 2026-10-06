# Spectraverse MASSter oracle cosine ceiling

> **Status:** FINDING · **Updated:** 2026-10-06
> 1,024 sampled validation spectra per fold (cv0, cv1), last checkpoints, CPU; intensity allocation is the largest gap. No config or training change.

## Result

The SPV cosine is held down mostly by intensity allocation, not by the candidate ceiling. A
second, separate drag is a 14% block of spectra that no subformula explains.

Each oracle keeps the true m/z and intensity of the target peaks that a rule calls reachable,
zeroes the rest, and scores that with the training metric's 0.01 Da binning (`cos_sim_0.01`).

| Score (pooled, n = 2,048) | cv0 | cv1 | Pooled [95% CI] |
| --- | ---: | ---: | --- |
| `model`: MASSter `cos_sim_0.01` | 0.457 | 0.459 | 0.458 [0.443, 0.473] |
| `opt_emit`: repo `opt_cos_sim` over the emitted bins | 0.737 | 0.732 | 0.734 [0.718, 0.750] |
| `cand_inst`: candidate mass within instrument tolerance | 0.806 | 0.799 | 0.803 [0.788, 0.817] |
| `cand_inst_iso`: plus +1/+2 × 1.003355 above a candidate | 0.827 | 0.818 | 0.822 |
| `cand_005`: candidate within ±0.05 Da | 0.898 | 0.895 | 0.896 |
| `sub_inst`: any precursor-bounded subformula ion, instrument tolerance | 0.881 | 0.830 | 0.855 [0.843, 0.869] |
| `sub_005`: subformula within ±0.05 Da | 0.940 | 0.927 | 0.934 |

The pooled headroom splits into these steps:
- 0.145 is lost to target intensity that no subformula reaches (1 − `sub_inst`).
- 0.052 is subformulas that MASSter's candidate set lacks.
- 0.068 is between "a candidate within tolerance" and "an exact emitted 0.01 Da bin".
- **0.276** is between the emitted-bin oracle and the model. MASSter reaches 62% of its own
  emitted-bin ceiling.

`sub_inst` is bimodal: median 0.999, 25th percentile 0.957, 10th percentile 0.213.

| `sub_inst` bin | n | `model` | `opt_emit` | `cand_005` | `sub_inst` |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤ 0.5 | 287 | 0.037 | 0.127 | 0.467 | 0.141 |
| 0.5–0.9 | 142 | 0.204 | 0.497 | 0.779 | 0.721 |
| 0.9–0.99 | 227 | 0.443 | 0.766 | 0.954 | 0.961 |
| > 0.99 | 1,392 | 0.574 | 0.879 | 0.987 | 0.999 |

On the 86% of spectra with `sub_inst` > 0.5, the model scores 0.527 against an emitted-bin
ceiling of 0.833. On the cleanest 68%, it scores 0.574 against 0.879. The ≤ 0.5 block is 54% of
`[M+K]+`, 42% of `[M+Na]+`, 44% of Other, 16% of ORB and 10% of QTOF spectra. On those spectra
the ±0.05 Da candidate ceiling (0.467) sits far above the instrument-tolerance subformula
ceiling (0.141). That is either a mass offset or chance proximity; this read cannot tell which.

| Instrument | n | `model` | `opt_emit` | `sub_inst` |
| --- | ---: | ---: | ---: | ---: |
| ORB | 1,154 | 0.502 | 0.771 | 0.839 |
| QTOF | 750 | 0.419 | 0.699 | 0.888 |
| NA | 85 | 0.475 | 0.893 | 0.976 |
| Other | 59 | 0.084 | 0.242 | 0.584 |

| Adduct | n | `model` | `opt_emit` | `sub_inst` |
| --- | ---: | ---: | ---: | ---: |
| `[M+H]+` | 1,041 | 0.488 | 0.782 | 0.883 |
| `[M-H]-` | 471 | 0.538 | 0.809 | 0.924 |
| `[M+Na]+` | 238 | 0.208 | 0.404 | 0.590 |
| `[M+NH4]+` | 137 | 0.461 | 0.771 | 0.895 |
| `[M+K]+` | 46 | 0.117 | 0.340 | 0.489 |

## Reading

- Better candidates alone cannot close the gap to FraGNNet-level or NIST-level scores. The
  emitted-bin ceiling is already 0.73, and the model uses 62% of it, the same intensity-allocation
  pattern as NIST20 and the [EI audit](masster_ei_performance_audit_2026_10_01.md).
- The 14% unexplained block caps every formula-constrained model, FraGNNet included. On it the
  model scores 0.04. Dropping or repairing those spectra moves the reported mean, not the model.
- Isotope rows (`cand_inst_iso`) add at most +0.019 to the ceiling.
- The ±0.05 Da window lifts the candidate ceiling by +0.09, but part of that is chance
  proximity (see the shifted-target control in the
  [recall finding](spectraverse_recall_tolerance_2026_10_01.md#wider-window-diagnostic-2026-10-01)).

## Method and limits

Script: `scripts/debug/spectraverse/spectraverse_oracle_ceiling.py`. Checkpoints: the row-swapped copies
`output/thiazolium_pred_2026_10_01/{ggifzj9m,u46v0k0g}_ob.ckpt` of the W&B `last.ckpt`. Their
`state_dict` holds the EMA weights, and their instrument rows match the current
NA/ORB/Other/QTOF order; the model is built from the current fold config. Sampling uses seed
20261001 on each fold's validation IDs (1,024 per fold), the same seed as the recall finding.
The recomputed model cosine matches `inference_step`'s `cos_sim_0.01` to 2e-7. Last-epoch W&B
full-validation values are 0.430 / 0.456. `has_ce` was recorded but is True everywhere,
because `fill_missing_ace` fills the missing CE.

The `sub_*` ceilings use `_subformula_masses`, which includes charge-migration ions, so they
are generous for Na/K. At ±0.05 Da, subformula matches on large molecules are partly chance.
These are last-checkpoint numbers on a sample, not best-checkpoint full-validation numbers.

Artifacts (ignored): `output/analysis/spectraverse_oracle_2026_10_03/oracle_cv{0,1}_1024.csv`,
logs, and the sampled subsets.

## Train versus validation (cv0, added 2026-10-03)

The same script with `--split train` (1,024 sampled cv0 training spectra, same checkpoint)
separates fitting from generalization:

| cv0 | `model` | `opt_emit` | `model` / `opt_emit` | `model`, `sub_inst` > 0.99 | `model`, `sub_inst` ≤ 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.590 | 0.706 | 0.84 | 0.719 | 0.050 |
| val | 0.457 | 0.737 | 0.62 | 0.555 | 0.029 |

Most of the intensity gap is generalization (0.13 train-to-val), not fitting. The model also
cannot fit the unexplained spectra even in training. The W&B val curve agrees: it peaks at
epoch 15 (cv0 0.439) and epoch 15–18 (cv1 0.465), then falls to 0.430 / 0.456 at epoch 39. The
config already has cosine LR, EMA 0.999 and pre-norm, and dropout is 0.1 throughout.
Regularization has never been varied on SPV. Artifact: `oracle_train_cv0_1024.csv`.

## MassSpecGym comparison (added 2026-10-03)

Same script, `--config config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_glacier_feats.yml`,
on `ll04i00m` at its best epoch 60 (`output/annotation_retrieval_pilot_2026_10_02/ckpt/ll04i00m_ep060_orb.ckpt`),
1,024 sampled spectra per split. Artifacts: `output/analysis/msgym_overfit_2026_10_03/`.

| | `model` | `opt_emit` | `model` / `opt_emit` | `sub_inst` |
| --- | ---: | ---: | ---: | ---: |
| MSGYM train | 0.884 | 0.952 | 0.93 | 0.997 |
| MSGYM val | 0.584 | 0.933 | 0.63 | 0.997 |
| SPV cv0 train | 0.590 | 0.706 | 0.84 | 0.871 |
| SPV cv0 val | 0.457 | 0.737 | 0.62 | 0.881 |

- MSGYM has a 0.30 train-to-val gap with a near-perfect ceiling: it is almost all
  generalization. ORB train 0.920 vs val 0.605; QTOF 0.816 vs 0.484.
- Its val curve does not decline much. The three finished MASSter runs (`1imawkx1`, `ll04i00m`,
  `57bavwzy`) peak at epoch 60–62 (0.570–0.581) and lose 0.003–0.006 by epoch 127. The model
  memorizes the training molecules and val plateaus; it is not getting worse.
- On SPV, the train-to-val gap is smaller (0.13) because the noisy targets also cap train fit.
  Its val curve does decline after epoch 15.
- Validation efficiency is the same on both datasets (62–63% of the emitted-bin ceiling). The
  shared limit is generalization to unseen molecules, not SPV data quality.
