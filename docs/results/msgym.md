# MassSpecGym results

> **Status:** LEDGER · **Updated:** 2026-10-18
> Every measured result on MassSpecGym 1.5, one row per run, checkpoint, split and evaluation. Numbers are copied from the linked source, which stays canonical.

Add a row when a finding or plan records a new measurement; fix a row when its source is corrected. Hub: [msgym](../workstreams/msgym.md).

## Current best

- **Spectrum prediction, test `cos_sim_0.01` (9,954):** `ll04i00m`@60 0.5713 against 0.5455 for GLACIER's released checkpoint (+0.0258, 95% CI +0.016 to +0.037; 99.9% CI +0.0085 to +0.0436); in sqrt space +0.0107 (95% CI +0.003, +0.019; 99.9% CI crosses zero). No forward test read exists for any later checkpoint ([row](#spectrum-prediction-test-9954-queries)).
- **Validation `val_cos_sim_0.01`:** highest among the sweep `tiwhun9q` loss-leaders is `720jwmai` (crashed, no checkpoint) peak 0.599811 at epoch 46; the re-run `f44tzxoc` peaked 0.5966 at epoch 81 and its kept checkpoint (@19, loss-selected) has 0.5937 ([row](#validation-48-trial-sweep-tiwhun9q)). Best of the 2026-10-03 arm screen: `wd10` `8afoui7e`@50 0.5867 ([row](#validation-arm-screen-and-parity-features)).
- **Formula-track retrieval, test (9,954 queries), best checkpoint by score (all unpaired, interval ±0.034 per arm):** `loglik_cal` `c32` `gz4fa02d`@18 0.559 (0.870 / 0.973); `mspred_ent` `c64` `u80w840n`@21 0.532; `cos_sim_0.01` `e1024` `5esngzcl`@26 0.375. `c32`, `c64` and `f44tzxoc`@19 tie on every score ([rows](#formula-track-retrieval-test-9954-queries)).
- **Paired test result:** `pow075` `q3oxbpgx`@55 over `ll04i00m`@60, `mspred_ent` +0.031 [+0.017, +0.046] and `loglik` +0.018 [+0.003, +0.033] hit@1; `f44tzxoc` vs `pow075` every difference crosses zero ([rows](#formula-track-retrieval-test-9954-queries)).
- **MASSter vs GLACIER retrieval, test:** calibrated likelihood `loglik_cal` 0.528 vs GLACIER 0.490 under the same rule (+0.039 [+0.022, +0.055]); GLACIER is level on ms-pred entropy (+1.15 [−0.54, +2.86] GLACIER − MASSter hit@1 in %) ([rows](#formula-retrieval-ms-pred-protocol-and-similarity-variants-test)).
- **Mass-track retrieval, test (`pow075` only, no comparator):** top-1 `loglik_cal` 74.52 [70.92, 77.93], `loglik` 71.35, `mspred_ent` 69.18, `cos_sim_0.01` 52.08 ([rows](#mass-track-retrieval-test-9954-queries)).

## Comparability

- **Metrics.** `cos_sim_0.01` is linear cosine on 0.01 Da bins (sum pooling, `mz_max` 1500), MassSpecGym's `cos_sim`; `cos_sim_sqrt_0.01` takes the sqrt of binned intensities; `cos_hun_inst_tol` is Hungarian matching at instrument tolerance (ORB 10 ppm with a 200 floor, QTOF 20 ppm with a 750 floor). GLACIER outputs sqrt intensity and is squared before every linear comparison ([GLACIER finding](../findings/masster/masster_glacier_released_ckpt.md#masster-vs-glaciers-released-checkpoint-on-the-massspecgym-simulation-test-set)).
- **Split.** Test is the 9,954-spectrum simulation-challenge set (`[M+H]+`, every row has a CE; ORB 8,374, QTOF 1,580). Val is the `ms_gym_benchmark_1_5` val split: 9,731 formula queries, 2,331 molecules. Val numbers are not comparable with published MassSpecGym test numbers.
- **Query sets.** Formula track: val 9,731 queries (1,609,708 candidates), test 9,954 queries (1,657,238 candidates). The full retrieval challenge also has `[M+Na]+` and no-CE queries (val 19,429, test 17,556); not run, so nothing here is leaderboard-comparable. Mass track: 9,954 test queries, 2,466,243 candidates. Pilot: 400 val molecules (375 ORB / 25 QTOF), one spectrum each, 72,946 formula candidates (100,001 mass candidates); the 176 odd-hash molecules carry the blend rows.
- **Scoring functions.** `cosine` = in-model `cos_sim_0.01`. `mspred_ent` = ms-pred's binned 0.01 Da sqrt entropy similarity. `loglik` = Σ y·log max(p̂, 1e-5), p̂ the predicted intensity within instrument tolerance. `loglik_cal` = rule `ll_g0.75_t2_e1e-4_a0.5` (Σ_k (√y_k / Σ√y) · log max(p̂_k, 1e-4), p̂_k from p^0.75 renormalized, within 2× tolerance), val-selected for `ll04i00m` and **not re-selected** for any later checkpoint; a transformed-intensity arm's exported p differs, so the rule does not carry over exactly.
- **Ties.** Hit@k is an expected value under random ties unless a row says "stable ties" (ms-pred's list-order tie-break, which favours the first-listed candidate, the truth). Truth is candidate 0 in MassSpecGym's lists: never tie-break by position.
- **Intervals.** Retrieval deltas: 95% molecule bootstrap, 2,000 resamples, paired, unless stated. Forward-similarity table: 95% molecule bootstrap, and a 99.9% interval over 2,319 molecules where stated; instrument audit: 5,000 resamples, seed 0. ms-pred's 99.9% BCa intervals (20,000 resamples) treat queries as i.i.d. and are narrower. Pilot: 2,000-resample query bootstrap (one query per molecule). Short-budget retrieval arms are unpaired, ±0.034 per arm.
- **Training batch.** `loss_batch_size` was 64 for every run up to and including the 2026-10-03 screen (arm screen, `ll04i00m`, `pow075` `q3oxbpgx`) and is 16 from 2026-10-08 (sweep `tiwhun9q`, `f44tzxoc`, short-budget and rewire arms). It sets only accumulation order. W&B hparams record it, not the config files.
- **Checkpoint selection differs by block.** Arm screen: top-1 epoch by `val_cos_sim_0.01_epoch/mean` (max), EMA weights. Sweep, leader re-run, short-budget and rewire arms: `val_primary_loss` (min), pow0.75 targets. The sweep's `val_primary_loss` is CE against pow0.75 targets, comparable within the sweep, not against pre-change runs or the screen.
- **Noise.** One seed per arm. Val control spread 0.0025 (three controls plus `s2`); identical same-seed reruns differ by up to 0.0081 val cosine (`e1024`), so 0.0025 understates it. No test-retrieval seed replicate exists.
- **Stale hparams.** `ll04i00m`'s hparams name the retired `FT` label; it was copied with `FT` renamed to `ORB` (weights unchanged) for retrieval and the pilot.

## Spectrum prediction (test, 9,954 queries)

Test split, simulation-challenge `[M+H]+` set. Four metrics from the model's own `metric_fn`; differences are ours − GLACIER.

| Date | Model / arm | Run@epoch | `cos_sim_0.01` | `cos_sim_sqrt_0.01` | `cos_hun_inst_tol` | `cos_hun_inst_tol_sqrt` | vs comparator (Δ, CI) | Notes | Source |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2026-09-24 | MASSter GLACIER-parity features arm, all peaks | `ll04i00m`@60 | 0.5713 | 0.5878 | 0.5955 | 0.6151 | vs GLACIER, 95% CI: +0.0258 (+0.016, +0.037); +0.0107 (+0.003, +0.019); +0.0306 (+0.020, +0.041); +0.0163 (+0.008, +0.024) | n=1; not the best recipe (`1imawkx1` 0.5754 val has no checkpoint). Higher on 53.6% of spectra (linear) | [finding](../findings/masster/masster_glacier_released_ckpt.md#four-metric-table-models-own-metric_fn) |
| 2026-09-24 | same, 0.1% cutoff | `ll04i00m`@60 | 0.5712 | 0.5894 | 0.5954 | 0.6168 | vs GLACIER, 95% CI: +0.0258 (+0.016, +0.036); +0.0123 (+0.005, +0.020); +0.0306 (+0.020, +0.041); +0.0181 (+0.010, +0.026) | median 65 aggregated peaks kept | [finding](../findings/masster/masster_glacier_released_ckpt.md#four-metric-table-models-own-metric_fn) |
| 2026-09-24 | GLACIER released MSG checkpoint (w/ CF; `--sparse-k 256`; output squared) | released ckpt | 0.5455 | 0.5771 | 0.5648 | 0.5988 | comparator | early-stopped after 24 epochs; one checkpoint | [finding](../findings/masster/masster_glacier_released_ckpt.md#four-metric-table-models-own-metric_fn) |
| 2026-09-24 | same, 99.9% CI over 2,319 molecules (`ll04i00m` − GLACIER) | `ll04i00m`@60 | 0.5713 | 0.5878 | — | — | linear +0.0258 (+0.0085 to +0.0436); sqrt +0.0107 (−0.0023 to +0.0240) | "ours − GLACIER" in `cos_sim_0.01` / `cos_sim_sqrt_0.01` | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | ms-pred native (per-peak sqrt, 150,000 bins, raw targets), top 256 peaks each | `ll04i00m`@60 | — | — | — | — | +0.0059 (ours 0.5717, GLACIER 0.5658); CI not computed | with all ~1,900 candidates ours scores 0.5658 | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | ms-pred native, top 1,024 peaks each | `ll04i00m`@60 | — | — | — | — | +0.0015 (ours 0.5664, GLACIER 0.5649); CI not computed | per-peak sqrt depends on peak count | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | GLACIER scored un-squared (first version, superseded) | released ckpt | 0.5430 | 0.5125 | — | — | ours − GLACIER +0.028 | squaring raises linear 0.5430 → 0.5455 and sqrt 0.5125 → 0.5771 | [finding](../findings/masster/masster_glacier_released_ckpt.md#masster-vs-glaciers-released-checkpoint-on-the-massspecgym-simulation-test-set) |
| 2026-10-02 | MASSter re-check under current code | `ll04i00m`@60 | 0.571272 | — | — | — | 0.571277 recorded | 87 of 9,954 spectra moved by more than 1e-4 (max 0.0145) | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#setup) |
| 2026-10-03 | MASSter, Orbitrap (8,374 spectra, 2,231 molecules) | `ll04i00m`@60 | 0.58925 | 0.61137 | 0.60385 | 0.62715 | vs GLACIER, 95% CI (5,000 resamples): +0.03061 [+0.02099, +0.04013]; +0.01305 [+0.00627, +0.01955]; +0.03326 [+0.02368, +0.04284]; +0.01615 [+0.00927, +0.02273] | | [finding](../findings/masster/masster_glacier_released_ckpt.md#qtof-is-not-an-established-glacier-win) |
| 2026-10-03 | GLACIER, Orbitrap | released ckpt | 0.55864 | 0.59832 | 0.57059 | 0.61100 | comparator | | [finding](../findings/masster/masster_glacier_released_ckpt.md#qtof-is-not-an-established-glacier-win) |
| 2026-10-03 | MASSter, QTOF (1,580 spectra, 195 molecules) | `ll04i00m`@60 | 0.47598 | 0.46292 | 0.55101 | 0.55135 | vs GLACIER, 95% CI: +0.00041 [−0.02779, +0.03131]; −0.00159 [−0.02389, +0.02426]; +0.01662 [−0.01288, +0.04765]; +0.01740 [−0.00593, +0.04313] | every QTOF interval includes zero: not an established GLACIER win | [finding](../findings/masster/masster_glacier_released_ckpt.md#qtof-is-not-an-established-glacier-win) |
| 2026-10-03 | GLACIER, QTOF | released ckpt | 0.47557 | 0.46452 | 0.53439 | 0.53394 | comparator | instrument-tolerance matching lifts linear scores by 0.07503 (MASSter) and 0.05882 (GLACIER) | [finding](../findings/masster/masster_glacier_released_ckpt.md#qtof-is-not-an-established-glacier-win) |
| — | GLACIER paper Table 4 (not measured here) | — | 0.58 (95% CI 0.57-0.59) w/o CF; 0.58 w/ CF | — | — | — | JSS 0.50 / 0.48 | scored in their own evaluation space; the checkpoint scores 0.566-0.577 there | [finding](../findings/masster/masster_glacier_released_ckpt.md#setup) |

## Forward similarity by instrument and CE (test)

`ll04i00m`@60 vs GLACIER released checkpoint, 2026-09-24; linear = `cos_sim_0.01`; CIs not given per slice.

| Date | Slice | n | `cos_sim_0.01` ours | `cos_sim_0.01` GLACIER | Δ linear | Δ sqrt | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-09-24 | FT / Orbitrap | 8,374 | 0.5893 | 0.5586 | +0.0306 | +0.0131 | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | QTOF | 1,580 | 0.4760 | 0.4756 | +0.0004 | −0.0016 | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | CE ≤ 15 | 1,012 | 0.5925 | 0.5765 | +0.0160 | −0.0031 | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | CE 15-25 | 2,402 | 0.6552 | 0.6496 | +0.0057 | −0.0041 | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | CE 25-35 | 1,608 | 0.5709 | 0.5590 | +0.0119 | −0.0022 | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | CE 35-50 | 1,409 | 0.5411 | 0.5175 | +0.0236 | +0.0076 | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-09-24 | CE > 50 | 3,522 | 0.5200 | 0.4704 | +0.0496 | +0.0320 | [finding](../findings/masster/masster_glacier_released_ckpt.md#metric-variants) |
| 2026-10-03 | QTOF reweighted to the whole test set's five stored-CE-bin proportions | 1,580 | — | — | +0.00778 | +0.00952 | [finding](../findings/masster/masster_glacier_released_ckpt.md#dataset-composition-and-limits) |
| 2026-10-03 | Orbitrap reweighted the same way | 8,374 | — | — | +0.03082 | +0.01299 | [finding](../findings/masster/masster_glacier_released_ckpt.md#dataset-composition-and-limits) |
| 2026-10-03 | Orbitrap, 107 molecules on both instruments only | 1,488 | — | — | +0.01757 | — | [finding](../findings/masster/masster_glacier_released_ckpt.md#dataset-composition-and-limits) |
| 2026-10-03 | QTOF, 107 molecules on both instruments only | 1,010 | — | — | −0.00099 | — | [finding](../findings/masster/masster_glacier_released_ckpt.md#dataset-composition-and-limits) |

## Prediction diagnostics (test, `ll04i00m`@60 vs GLACIER released checkpoint)

Linear intensity, GLACIER squared, 0.01 Da bins unless stated; descriptive, one checkpoint each.

| Date | Diagnostic | `ll04i00m` | GLACIER | Notes | Source |
| --- | --- | ---: | ---: | --- | --- |
| 2026-09-24 | peaks / bins per spectrum, median | 902 | 160 | targets 19 | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-09-24 | of those ≥ 1% of base, median | 19 | 7 | | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-09-24 | of those ≥ 5% of base, median | 7 | 4 | targets 6 | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-09-24 | peaks ≥ 5% of base with no target peak in the bin, median (mean) | 3 (7.2) | 1 (2.6) | | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-09-24 | predicted intensity with no target peak in the bin | 0.347 | 0.238 | one bin away 0.039 / 0.036; at precursor 0.010 / 0.011; elsewhere ≥ 5% 0.177 / 0.144; elsewhere < 5% 0.121 / 0.048 | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-09-24 | target intensity with no predicted peak in the bin | 0.146 | 0.251 | | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-09-24 | ours: drop predicted peaks below 1% / 5% / 10% of base, Δ `cos_sim_0.01` | −0.0011 / −0.0058 / −0.0121 | — | sharpening intensity^1.1 / ^1.25 / ^1.5: −0.0033 / −0.0095 / −0.0204; flattening ^0.9: +0.0016, ^0.75: −0.0023 | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-09-24 | export cutoff 0.1% (per candidate, pre-aggregation), Δ linear / sqrt cos | −0.0001 / +0.0023 | — | median 80 candidates kept; aggregated export: median 65 of 924 peaks, −0.0001 linear, +0.0016 sqrt (`cos_hun_inst_tol_sqrt` +0.0017) | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-09-24 | export cutoff 0.5% / 1%, Δ linear / sqrt cos | −0.0005 / +0.0015 and −0.0011 / −0.0003 | — | median kept 31 / 20 (none: ~2,060, 0.5713 / 0.5878) | [finding](../findings/masster/masster_glacier_released_ckpt.md#peak-over-prediction-added-2026-09-24) |
| 2026-10-03 | Orbitrap: target intensity covered / base peak supported / predicted base = target base / predicted intensity off target / median peaks ≥1% of predicted base | 0.89681 / 0.95080 / 0.42441 / 0.29845 / 17 | 0.79340 / 0.86745 / 0.41557 / 0.19117 / 7 | CPU diagnostics, instrument windows, not CUDA 0.01 Da bins | [finding](../findings/masster/masster_glacier_released_ckpt.md#what-the-peak-lists-show) |
| 2026-10-03 | QTOF: same five | 0.88718 / 0.95000 / 0.38038 / 0.40888 / 27 | 0.73872 / 0.86962 / 0.40570 / 0.29392 / 8 | median target peak counts 20 (ORB), 19 (QTOF) | [finding](../findings/masster/masster_glacier_released_ckpt.md#what-the-peak-lists-show) |
| 2026-10-03 | rows above stored CE 50: QTOF / Orbitrap | 21.2% / 38.1% | — | MASSter linear advantage in that band +0.05153 (QTOF), +0.04940 (Orbitrap) | [finding](../findings/masster/masster_glacier_released_ckpt.md#dataset-composition-and-limits) |

## Validation: arm screen and parity features

All runs `data/split/ms_gym_benchmark_1_5`, 128 epochs (`e64`: 64), seed 397623 (`s2`: 20261003); `val_cos_sim_0.01_epoch/mean`, EMA `state_dict`; arm = suffix of `msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_`; `loss_batch_size` 64. "val best" is a max over 128 epochs (selection-biased upward). Dates: screen ran 2026-10-02/03.

| Date | Model / arm | Run@epoch | val best | val last | vs comparator (Δ, CI) | Notes | Source |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| 2026-09-24 | `1imawkx1` base recipe (pre-norm, `i4` on MSGYM) | `1imawkx1`@62 | 0.5754 | — | vs `ll04i00m`: tail (epochs 118-127) 0.5695 vs 0.5669, −0.0026 for `ll04i00m`; `ll04i00m` below it at 101 of 128 epochs | no uploaded checkpoint; no MSGYM seed replicate; val best of `ll04i00m` 0.5705 (@60) | [plan](../plans/masster/masster_msgym_training_arms_plan.md#glacier-parity-features-arm-ran-no-gain-wb-read-2026-09-24) |
| 2026-09-24 | `ll04i00m` train vs val (generalization gap) | `ll04i00m` | — | — | — | train 0.884 against val 0.584 (63% of emitted-bin ceiling); both recipes peak near epoch 60 and drift down | [plan](../plans/masster/masster_msgym_training_arms_plan.md#objective) |
| 2026-10-03 | glacier_feats (reference) | `ll04i00m`@60 | 0.5705 | 0.5669 | — | checkpoint uploaded | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_wd10 | `8afoui7e`@50 | 0.5867 | 0.5805 | +0.0049 over control mean 0.5819 | best val of the screen | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_pm1 | `3ki6nyy6`@40 | 0.5859 | 0.5804 | +0.0041 over control mean | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_mol3 | `7wh3jaqw`@39 | 0.5849 | 0.5776 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_bond1 | `1vvyc70p`@49 | 0.5838 | 0.5766 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_enc256 | `ocx2i6h5`@37 | 0.5835 | 0.5779 | — | 5.76M params | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier (control rerun) | `kb5by4di`@46 | 0.5833 | 0.5766 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_enc512 | `apm3ok8v`@51 | 0.5831 | 0.5784 | — | 13.89M params | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_pow075_pm1 | `8szjefc2`@51 | 0.5829 | 0.5774 | — | measured control-level val; no retrieval read | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_e64 | `01pkzb92`@38 | 0.5821 | 0.5800 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_s2 (seed replicate) | `2y8rl6fs`@47 | 0.5819 | 0.5766 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier (control rerun) | `dz7u8i6w`@39 | 0.5815 | 0.5764 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier (control) | `57bavwzy`@62 | 0.5809 | 0.5777 | — | control spread (three controls plus `s2`) 0.5809-0.5833, 0.0025 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_pow075 | `q3oxbpgx`@55 | 0.5790 | 0.5728 | — | val cosine below `wd10` by 0.008 yet best on retrieval | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_itf1 | `cgivkl1j`@38 | 0.5785 | 0.5737 | — | 8.18M params | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | glacier_feats_fourier | `p3i0tudq`@84 | 0.5770 | 0.5733 | — | rerun of crashed `zb7lhmn4` (no checkpoint) | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_lr5e4 | `3s4o3l02`@47 | 0.5736 | 0.5705 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_sqrt_pm1 | `ymaktpzu`@29 | 0.5721 | 0.5648 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_sqrt | `0hmzfttj`@43 | 0.5712 | 0.5642 | — | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |
| 2026-10-03 | k1_ema999_fourier_b256 | `1p4s2w61` | 0.5875 | — | — | crashed at epoch 110 (3.94 h, 25.7 GB peak; likely wall-time limit, unconfirmed); no checkpoint | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-screen-wb-all-msgym-runs-since-ll04i00m) |

## Validation: 48-trial sweep tiwhun9q

Random search, 48 trials (15 finished all 128 epochs, 33 crashed at about 2.936-2.970 h vs 3.236-4.946 h for completed; cause unconfirmed). Fixed seed 397623, pow0.75 CE, `loss_batch_size` 16, test disabled. Lower `val_primary_loss` is better; not comparable with the arm-screen table.

| Date | Model / arm | Run@epoch | best `val_primary_loss` | val `cos_sim_0.01` at loss epoch | peak val `cos_sim_0.01` (epoch) | vs comparator | Source |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-10-11 | leader, crashed, no W&B checkpoint (batch 256, LR 0.0015548897423024086, wd 0.1, depths 3/2/3) | `720jwmai`@32 | -1.662002 | 0.597606 | 0.599811 (46) | vs `q3oxbpgx`: loss improves 0.07165, peak cosine 0.02083 | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leading-runs) |
| 2026-10-11 | crashed, no W&B checkpoint | `6egv3347`@28 | -1.651483 | 0.586189 | 0.589625 (59) | | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leading-runs) |
| 2026-10-11 | best finished (batch 256, LR 0.0005775193973727379, wd 0.1, depths 3/3/2); has `ckpt/model-epoch=025.ckpt` | `86f3wm2f`@25 | -1.648701 | 0.591627 | 0.592739 (58) | peak cosine +0.01376 over `q3oxbpgx` | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leading-runs) |
| 2026-10-11 | crashed | `7ms4lgh2`@28 | -1.645117 | 0.591067 | 0.593777 (47) | | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leading-runs) |
| 2026-10-11 | finished | `1t7bverf`@33 | -1.642208 | 0.591194 | 0.592072 (35) | | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leading-runs) |
| 2026-10-11 | previous `pow075` (screen arm, `loss_batch_size` 64, cosine-selected) | `q3oxbpgx`@25 (loss) / @55 (cos) | -1.590353 | — | 0.578978 (55) | reference for the rows above; selected single-run differences, not confirmed test gains | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leading-runs) |
| 2026-10-11 | encoder dropout 0.0 / 0.1 / 0.2 / 0.3, group means over all trials | 8 / 10 / 16 / 14 trials | -1.4526 / -1.5504 / -1.5868 / -1.6089 | — | 0.5516 / 0.5740 / 0.5796 / 0.5852 (through epoch 75) | unadjusted, confounded random-search associations | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#what-the-random-search-suggests) |
| 2026-10-11 | LR bins ≤3e-4 / (3e-4,7e-4] / >7e-4, group mean best loss | 14 / 16 / 18 trials | -1.4988 / -1.5754 / -1.6027 | — | — | LR vs loss Spearman −0.676; LR vs encoder dropout 0.021 | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#what-the-random-search-suggests) |
| 2026-10-11 | batch 256 vs batch 512, mean peak cosine | — | — | — | 0.5808 vs 0.5705 | confounded | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#what-the-random-search-suggests) |
| 2026-10-11 | all 48 trials: best-loss epochs / best-cosine epoch / final-minus-best loss | — | epochs 11-66 (median 28) | — | best cosine median epoch 49 | completed runs: final loss worse than best by 0.110 on average | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#integrity-and-interruptions) |
| 2026-10-13 | leader re-run (`msgym_sweep_leader_720jwmai.yml`, 128 epochs, 4.98 h, loss-selected kept epoch 19) | `f44tzxoc`@19 | -1.6651 | 0.5937 | 0.5966 (81); ended 0.5911 | vs `q3oxbpgx` 0.578978 peak: +0.015 val cosine (0.594 vs 0.579) | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leader-re-run-formula-test-retrieval-2026-10-13) |

## Validation: short-schedule and candidate-budget screen

Leader recipe (10.14M params, pow0.75, batch 256, LR 1.55e-3, wd 0.1, seed 397623), `loss_batch_size` 16. Each arm run twice (A: 2026-10-14 20:43, B: 2026-10-15 01:55) at identical config and seed; values at the best-`val_primary_loss` epoch; Δ against the `c64` A/B mean. "U reached" is `unique_fragment_target_reached`.

| Date | Arm | Run@epoch (A, B) | min/epoch | val `cos_sim_0.01` A / B | Δ cos (mean) | val wrecall A / B | Δ wrecall | U reached | Source |
| --- | --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| 2026-10-15 | `c64` (64 epochs, U256/E4096) | `u80w840n`@21, `ixvr0qgk` | 2.30 | 0.5928 / 0.5926 | — | 0.9013 / 0.9003 | — | 0.68 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#results-2026-10-15) |
| 2026-10-15 | `c32` (32 epochs) | `gz4fa02d`@18, `3402ar4d` (best epoch 18 in both) | 2.29 | 0.5933 / 0.5958 | +0.0019 | 0.9001 / 0.9017 | +0.0001 | 0.70 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#results-2026-10-15) |
| 2026-10-15 | `u128` (U128) | `28dptwet`@23, `w9mjor6f` | 1.51 | 0.5904 / 0.5866 | −0.0042 | 0.8611 / 0.8635 | −0.0385 | 0.99 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#results-2026-10-15) |
| 2026-10-15 | `e1024` (E1024) | `5esngzcl`@26, `kt1xn7v5` | 1.42 | 0.5919 / 0.6000 | +0.0032 | 0.8786 / 0.8786 | −0.0222 | 0.06 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#results-2026-10-15) |
| 2026-10-15 | `u128_e1024` | `2m4rb2c2`@20, `fddrhwcd` | 1.20 | 0.5864 / 0.5864 | −0.0063 | 0.8590 / 0.8603 | −0.0412 | 0.73 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#results-2026-10-15) |
| 2026-10-15 | `floor01` (fragment probability floor 0.01) | `c43o9rbn`@44, `sqj14xmm` | 2.47 | 0.5861 / 0.5899 | −0.0047 | 0.7881 / 0.8010 | −0.1062 | 0.63 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#results-2026-10-15) |

Same-seed noise: identical reruns differ by up to 0.0081 val cosine (`e1024`) and 0.0038 (`u128`, `floor01`); wrecall reproduces within 0.0016 except `floor01` (0.013) ([plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#results-2026-10-15)).

## Validation: rewire rules screen

Config copies of `msgym_sb_c32.yml` (leader recipe at 32 epochs), `fir:h100-3g.40gb`, ran 2026-10-16 03:48 UTC, `loss_batch_size` 16; same-seed noise up to 0.008 cosine. Wall-clock in hours (multiple of control); `_runtime` includes setup on shared MIG slices, so ratios are rough.

| Date | Arm | Run | Wall-clock h (x control) | best val `cos_sim_0.01` (epoch) | best-`val_primary_loss` epoch | final wprec / wrecall | vs comparator | Source |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 2026-10-16 | control | `rpa6uq8k` | 1.23 (1.00) | 0.5944 (24) | 19 | 0.7574 / 0.9020 | reference | [plan](../plans/masster/masster_rewire_rules_training_plan.md#validation-read-ran-2026-10-16-0348-utc-32-epochs-firh100-3g40gb) |
| 2026-10-16 | a0 | `oj85dppl` | 1.58 (1.28) | 0.5925 (21) | 19 | 0.7592 / 0.9099 | cosine 0.002 below control; wrecall +0.008 | [plan](../plans/masster/masster_rewire_rules_training_plan.md#validation-read-ran-2026-10-16-0348-utc-32-epochs-firh100-3g40gb) |
| 2026-10-16 | a0d1 | `16gpi651` | 1.96 (1.59) | 0.5921 (22) | 14 | 0.7625 / 0.9105 | cosine 0.002 below control; wrecall +0.008; 1.59x cost crosses kill (d) | [plan](../plans/masster/masster_rewire_rules_training_plan.md#validation-read-ran-2026-10-16-0348-utc-32-epochs-firh100-3g40gb) |
| 2026-10-16 | a0null | `47t8i9s0` | 1.59 (1.29) | 0.5923 (16) | 14 | 0.7581 / 0.9046 | cosine 0.002 below control; wrecall +0.003; a0 = a0null (0.5925 vs 0.5923) | [plan](../plans/masster/masster_rewire_rules_training_plan.md#validation-read-ran-2026-10-16-0348-utc-32-epochs-firh100-3g40gb) |

## Formula-track retrieval (val, 9,731 queries)

MASSter only, `ll04i00m`@60, hit@1 / hit@5 / hit@20; Δ vs `cos_sim_0.01`, 95% molecule bootstrap. Date 2026-10-03.

| Date | Model / arm | Run@epoch | Score | hit@1 | hit@5 | hit@20 | vs comparator (Δ, CI) | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-10-03 | MASSter | `ll04i00m`@60 | `cos_sim_0.01` | 0.383 | 0.718 | 0.894 | — | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-9731-queries-masster-only) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | `cos_hun_inst_tol` | 0.391 | 0.721 | 0.895 | +0.008 [+0.003, +0.012] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-9731-queries-masster-only) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | numpy sqrt cosine (0.01 Da) | 0.479 | 0.804 | 0.936 | +0.095 [+0.083, +0.107] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-9731-queries-masster-only) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | entropy similarity | 0.466 | 0.790 | 0.929 | +0.083 [+0.072, +0.094] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-9731-queries-masster-only) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | query log-likelihood | 0.511 | 0.823 | 0.953 | +0.127 [+0.113, +0.142] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#val-9731-queries-masster-only) |

Without the 400 pilot molecules (8,095 queries), the likelihood gain is +0.130 [+0.114, +0.147].

## Formula-track retrieval (test, 9,954 queries)

Hit@1 with (hit@5 / hit@20) where the source gives them; expected hit@k under random ties. `loglik_cal` uses the `ll04i00m` rule (not re-selected) for every row except the 2026-10-03 calibration read. One seed and one checkpoint per arm. Test candidates 1,657,238.

| Date | Model / arm | Run@epoch | `cos_sim_0.01` | `cos_sim_sqrt_0.01` | `mspred_ent` | `loglik` | `loglik_cal` | vs comparator (Δ, CI) | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-03 | MASSter, frozen `ll` (pre-calibration) | `ll04i00m`@60 | 0.358 (0.711 / 0.894) | 0.455 (0.796 / 0.939) | — | 0.485 (0.815 / 0.948) | — | `loglik` − `cos_sim_0.01` +0.127 [+0.113, +0.142]; `loglik` − GLACIER `cos_sim_sqrt_0.01` +0.057 [+0.039, +0.075]; `cos_hun_inst_tol` − `cos_sim_0.01` +0.006 [+0.002, +0.010]; `cos_hun_inst_tol_sqrt` − `cos_sim_0.01` +0.104 [+0.094, +0.116] | `cos_hun_inst_tol` 0.364 (0.716 / 0.894), `cos_hun_inst_tol_sqrt` 0.463 (0.804 / 0.943), "entropy similarity" 0.444 (0.788 / 0.935); ORB (8,374) / QTOF (1,580) hit@1: `cos_sim_0.01` 0.398 / 0.150, `loglik` 0.538 / 0.208; QTOF `cos_hun_inst_tol` 0.144 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#test-9954-queries) |
| 2026-10-03 | MASSter, calibrated `ll_g0.75_t2_e1e-4_a0.5` (val-selected, frozen before the test read; one test read) | `ll04i00m`@60 | — | — | — | 0.4853 (0.8150 / 0.9477) frozen | **0.5284** (0.8530 / 0.9648) | vs frozen `ll`: +0.0431 [+0.0347, +0.0521] hit@1; +0.0380 [+0.0304, +0.0453] hit@5; +0.0171 [+0.0115, +0.0230] hit@20 | ORB Δ +0.0485 [+0.0389, +0.0585], QTOF Δ +0.0146 [−0.0064, +0.0361]; val held-out half predicted +0.0441 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#calibrated-likelihood-val-selected-one-test-read-added-2026-10-03) |
| 2026-10-03 | GLACIER released checkpoint (batch 64, sparse-k 100, squared) | released ckpt | 0.335 (0.688 / 0.878) | 0.428 (0.770 / 0.919) | 0.497 (0.827 / 0.942) | 0.444 (0.793 / 0.936) old `ll` | 0.490 (0.829 / 0.951) | vs MASSter: `cos_sim_0.01` −0.023 [−0.037, −0.009]; `cos_sim_sqrt_0.01` −0.027 [−0.043, −0.012]; calibrated (2× tol) MASSter − GLACIER +0.039 [+0.022, +0.055] | ORB / QTOF hit@1: `cos_sim_0.01` 0.371 / 0.147, sqrt 0.474 / 0.184; calibrated ORB 0.545 vs MASSter 0.586, QTOF 0.195 vs 0.222 (+0.027 [−0.019, +0.074], unresolved); `mspred_ent` and calibrated read from different tables | [finding](../findings/masster/masster_msgym_formula_retrieval.md#test-9954-queries) |
| 2026-10-03 | spectrum-blind consensus controls | — | 0.046 | — | — | 0.042 | — | chance level | hit@5 0.17, hit@20 0.31 / 0.32 (cosine / ll) | [finding](../findings/masster/masster_msgym_formula_retrieval.md#test-9954-queries) |
| 2026-10-07 | MASSter (re-run on the same runner; reproduces 2026-10-03 to three decimals) | `ll04i00m`@60 | 0.358 (0.711 / 0.894) | 0.455 (0.796 / 0.939) | 0.485 (0.818 / 0.949) | 0.485 (0.815 / 0.948) | 0.528 (0.853 / 0.965) | reference | `cos_hun` 0.365 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#formula-track-test-retrieval) |
| 2026-10-07 | `wd10` (weight decay 0.1; best val) | `8afoui7e`@50 | 0.366 (0.725 / 0.897) | 0.467 (0.806 / 0.936) | 0.495 (0.826 / 0.946) | 0.496 (0.829 / 0.951) | 0.532 (0.857 / 0.963) | vs `ll04i00m`: `cos_sim_0.01` +0.008 [−0.005, +0.021]; sqrt +0.012 [−0.000, +0.024]; `mspred_ent` +0.010 [−0.003, +0.023]; `loglik` +0.011 [−0.003, +0.025]; `loglik_cal` +0.003 [−0.011, +0.018] | `cos_hun` 0.372; `loglik_cal` ORB 0.591, QTOF 0.217; every CI crosses zero | [finding](../findings/masster/masster_msgym_formula_retrieval.md#formula-track-test-retrieval) |
| 2026-10-07 | `pow075` (target y^0.75) | `q3oxbpgx`@55 | 0.362 (0.722 / 0.898) | 0.467 (0.808 / 0.940) | 0.516 (0.842 / 0.955) | 0.504 (0.836 / 0.957) | 0.542 (0.862 / 0.970) | vs `ll04i00m`: `cos_sim_0.01` +0.004 [−0.009, +0.017]; sqrt +0.012 [−0.002, +0.027]; `mspred_ent` **+0.031 [+0.017, +0.046]**; `loglik` **+0.018 [+0.003, +0.033]**; `loglik_cal` +0.014 [−0.002, +0.029]. vs `wd10`: `mspred_ent` +0.021 [+0.007, +0.036], others cross zero (cosine −0.004 [−0.015, +0.007], sqrt +0.000 [−0.013, +0.014], `loglik` +0.008 [−0.006, +0.021], `loglik_cal` +0.011 [−0.003, +0.025]) | `cos_hun` 0.368; `loglik_cal` ORB 0.603, QTOF 0.218; one seed; `loglik_cal` under-tuned for new checkpoints | [finding](../findings/masster/masster_msgym_formula_retrieval.md#formula-track-test-retrieval) |
| 2026-10-13 | `pow075` re-run (same shards; reproduces the finding to three decimals) | `q3oxbpgx`@55 | 0.362 (0.722 / 0.898) | 0.467 (0.808 / 0.940) | 0.516 (0.843 / 0.956) | 0.504 (0.836 / 0.957) | 0.542 (0.862 / 0.970) | reference for `f44tzxoc` | `cos_hun` 0.368 | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leader-re-run-formula-test-retrieval-2026-10-13) |
| 2026-10-13 | sweep leader re-run (batch 256, 128 epochs) | `f44tzxoc`@19 | 0.365 (0.719 / 0.905) | 0.470 (0.809 / 0.945) | 0.523 (0.844 / 0.956) | 0.503 (0.837 / 0.960) | 0.545 (0.866 / 0.969) | vs `pow075`: `cos_sim_0.01` +0.003 [−0.010, +0.015]; `cos_hun` +0.001 [−0.011, +0.014]; sqrt +0.003 [−0.010, +0.015]; `mspred_ent` +0.007 [−0.006, +0.021]; `loglik` −0.001 [−0.014, +0.013]; `loglik_cal` +0.003 [−0.010, +0.017] | ties `pow075` on every score; `cos_hun` 0.370; `loglik_cal` ORB 0.600 vs 0.603, QTOF 0.257 vs 0.218 (1,580 queries, not paired); matched control `msgym_sweep_base_control.yml` not retrieval-tested | [plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leader-re-run-formula-test-retrieval-2026-10-13) |
| 2026-10-16 | rewire a0 smoke, original shard 0 only (65,406 candidates), not a comparison | a0@19 (run id not given in that bullet) | — | — | — | — | 0.426 (hit@1) | `f44tzxoc` 0.428 on the same shard, different lineage | inference 3:39 at eval batch 256; retrieval for `a0`, `a0d1`, `a0null` not yet scored | [plan](../plans/masster/masster_rewire_rules_training_plan.md#formula-test-retrieval-prepared-2026-10-16) |
| 2026-10-17 | `c64` (64 epochs, replicate A) | `u80w840n`@21 | 0.371 | 0.482 | 0.532 | 0.513 | 0.552 (0.870 / 0.970) | reference; unpaired intervals ±0.034 on every arm | `loglik_cal` ORB 0.605, QTOF 0.270 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17) |
| 2026-10-17 | `c32` (32 epochs) | `gz4fa02d`@18 | 0.365 | 0.481 | 0.528 | 0.514 | 0.559 (0.870 / 0.973) | `loglik_cal` +0.007 vs `c64` | ORB 0.613, QTOF 0.273; ties `c64` and `f44tzxoc` on every score | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17) |
| 2026-10-17 | rewire control (config twin of `c32`, best epoch 19) | `rpa6uq8k`@19 | 0.369 | 0.475 | 0.519 | 0.509 | 0.555 (0.876 / 0.970) | `loglik_cal` +0.003 vs `c64`; vs `c32`: 0.004 `loglik_cal`, 0.009 `mspred_ent`, 0.006 sqrt cosine | ORB 0.609, QTOF 0.266; QTOF `mspred_ent` 0.266 vs 0.228 for `c32`, so QTOF per-arm differences unresolved | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17) |
| 2026-10-17 | `e1024` | `5esngzcl`@26 | 0.375 | 0.478 | 0.529 | 0.508 | 0.545 (0.867 / 0.969) | `loglik_cal` −0.007 vs `c64` (inside replicate spread; −0.003 `mspred_ent`, +0.004 cosine) | ORB 0.600, QTOF 0.248; not yet paired (`--compare sb_e1024:sb_c32` pending on Fir) | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17) |
| 2026-10-17 | `u128` | `28dptwet`@23 | 0.372 | 0.473 | 0.529 | 0.495 | 0.535 (0.856 / 0.965) | `loglik_cal` −0.016 vs `c64` (`loglik` −0.017) | ORB 0.594, QTOF 0.225 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17) |
| 2026-10-17 | `u128_e1024` | `2m4rb2c2`@20 | 0.361 | 0.468 | 0.526 | 0.494 | 0.537 (0.862 / 0.964) | `loglik_cal` −0.015 vs `c64` | ORB 0.592, QTOF 0.246 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17) |
| 2026-10-17 | `floor01` (floor 0.01; rejected) | `c43o9rbn`@44 | 0.371 | 0.469 | 0.517 | 0.477 | 0.511 (0.837 / 0.955) | `loglik_cal` −0.041 vs `c64` (`loglik` −0.036, `mspred_ent` −0.015) | ORB 0.564, QTOF 0.233 | [plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17) |

## Formula retrieval: ms-pred protocol and similarity variants (test)

Test, 9,954 queries, `ll04i00m`@60 (MASSter) vs GLACIER released checkpoint (w/ CF); hit@1 in %. Date 2026-10-03.

### ms-pred-style rescoring (our candidate list, hit@1 %)

| Date | Model | Run@epoch | Score | random ties | stable ties (ms-pred) | ours: 95% molecule bootstrap | ms-pred: 99.9% BCa, i.i.d. | ORB / QTOF | Source |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-10-03 | GLACIER | released ckpt | our `cos_sim_0.01` | 33.53 | 33.69 | [31.2, 36.0] | [32.2, 35.3] | 37.1 / 14.7 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | GLACIER | released ckpt | our `cos_sim_sqrt_0.01` | 42.80 | 42.96 | [40.0, 45.6] | [41.4, 44.6] | 47.4 / 18.4 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | GLACIER | released ckpt | ms-pred cosine | 40.89 | 41.13 | [38.2, 43.7] | [39.6, 42.7] | 45.6 / 16.0 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | GLACIER | released ckpt | **ms-pred entropy** | 49.65 | 49.67 | [46.5, 52.8] | [48.1, 51.4] | 54.8 / 22.1 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | our `cos_sim_0.01` | 35.84 | 35.92 | [33.3, 38.5] | [34.3, 37.5] | 39.8 / 15.0 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | our `cos_sim_sqrt_0.01` | 45.50 | 45.58 | [42.4, 48.7] | [44.0, 47.2] | 50.5 / 19.2 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | ms-pred cosine | 45.55 | 45.66 | [42.6, 48.8] | [44.0, 47.3] | 50.5 / 19.6 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | **ms-pred entropy** | 48.50 | 48.51 | [45.5, 51.7] | [46.9, 50.3] | 53.5 / 22.3 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | MASSter | `ll04i00m`@60 | query log-likelihood | 48.53 | 48.57 | [45.3, 51.8] | [47.0, 50.3] | 53.8 / 20.8 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |

Paired GLACIER − MASSter hit@1 (95% molecule bootstrap, %): ms-pred cosine −4.66 [−6.42, −3.11]; ms-pred entropy +1.15 [−0.54, +2.86] ([finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03)).

### Reproduction of the paper's retrieval (ms-pred candidate lists, stable ties)

| Date | Model | Run@epoch | Candidate list | hit@1 / @5 / @20 (%) | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- |
| — | GLACIER paper, w/o CF (not measured here) | — | ms-pred | 48.49 / 76.56 / 90.60 | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| — | GLACIER paper, w/ CF (not measured here) | — | ms-pred | 49.88 / 78.13 / 90.75 | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | GLACIER, ms-pred entropy | released ckpt | our candidate list | 49.67 / 82.65 / 94.22 | candidate sets differ only in stereo duplicates (ms-pred 187 rows per query vs our 166) | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | GLACIER, ms-pred entropy | released ckpt | **ms-pred full list** | **49.67** / **78.20** / **90.80** | reproduces the paper's w/ CF row; released checkpoint is w/ CF (EarlyStopping patience 3, batch 32) | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | MASSter, ms-pred entropy | `ll04i00m`@60 | ms-pred full list | 48.51 / 76.92 / 91.96 | trails GLACIER by 1.2 @1 (+1.15 [−0.54, +2.86]), 1.3 @5, leads 1.2 @20 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | MASSter, query log-likelihood | `ll04i00m`@60 | ms-pred full list | 48.57 / 76.71 / 91.32 | | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| — | GLACIER paper mass top-1 (not measured here) | — | ms-pred mass list | 69.95 (w/ CF), 69.69 (w/o CF) | unpaired; our mass list not checked against ms-pred's | [finding](../findings/masster/masster_msgym_formula_retrieval.md#mass-track-test-retrieval-pow075-2026-10-08) |

### Instrument-tolerance entropy on ms-pred's full candidate table (stable ties)

hit@1 / @5 / @20 in %; last columns are GLACIER − MASSter.

| Date | Similarity | GLACIER (CF) | MASSter `ll04i00m`@60 | Δ @1 [95% mol. bootstrap] | Δ @5 | Δ @20 | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-03 | binned 0.01 Da sqrt entropy (ms-pred) | 49.67 / 78.20 / 90.80 | 48.51 / 76.92 / 91.96 | +1.16 [−0.54, +2.87] | +1.28 [−0.50, +3.16] | −1.17 [−2.86, +0.49] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | **inst-tol sqrt entropy** | **51.40** / **79.04** / 91.91 | 49.07 / 77.41 / **92.06** | **+2.33 [+0.61, +4.15]** | +1.64 [−0.14, +3.58] | −0.15 [−1.55, +1.24] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | inst-tol Li-weighted entropy | 48.88 / 77.36 / 91.47 | 48.82 / 76.66 / 91.72 | +0.06 [−1.56, +1.62] | +0.69 [−0.94, +2.38] | −0.25 [−1.51, +1.06] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | inst-tol JSS (linear) | 41.99 / 72.25 / 88.45 | 45.11 / 74.30 / 90.57 | −3.11 [−4.73, −1.54] | −2.05 [−3.64, −0.37] | −2.12 [−3.60, −0.67] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | MASSter query log-likelihood (GLACIER inst-tol sqrt − this) | — | 48.57 / 76.71 / 91.32 | +2.82 [+1.11, +4.65] | +2.33 [+0.42, +4.38] | +0.59 [−0.92, +2.01] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |

Random-tie hit@1 with the 1% prediction filter, GLACIER vs MASSter: sqrt 46.74 vs 49.44; Li 45.80 vs 49.30; JSS 40.51 vs 44.32 (stable ties inflate GLACIER by about 1.6 points under the filter) ([finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03)).

### Entropy variants: binning, instrument tolerance, 1% filter (hit@1 / @5 in %, random ties)

| Date | Variant | GLACIER hit@1 / @5 | MASSter `ll04i00m`@60 hit@1 / @5 | GLACIER − MASSter hit@1 [95% mol. bootstrap] | Source |
| --- | --- | --- | --- | --- | --- |
| 2026-10-03 | binned 0.01 Da (ms-pred) | 49.65 / 82.62 | 48.50 / 81.83 | +1.15 [−0.54, +2.86] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#entropy-variants-binning-instrument-tolerance-1-filter-added-2026-10-03) |
| 2026-10-03 | binned, 1% filter | 45.04 / 78.77 | 48.64 / 82.00 | −3.60 [−5.10, −2.20] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#entropy-variants-binning-instrument-tolerance-1-filter-added-2026-10-03) |
| 2026-10-03 | ppm (instrument tolerance) | **51.32** / 83.49 | 49.03 / 82.11 | **+2.29 [+0.56, +4.11]** | [finding](../findings/masster/masster_msgym_formula_retrieval.md#entropy-variants-binning-instrument-tolerance-1-filter-added-2026-10-03) |
| 2026-10-03 | ppm, 1% filter | 46.79 / 79.57 | **49.44** / 82.77 | −2.65 [−4.02, −1.33] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#entropy-variants-binning-instrument-tolerance-1-filter-added-2026-10-03) |

MASSter's query log-likelihood is 48.53 / 81.50 for reference. Instrument-tolerance matching helps both: GLACIER +1.67 [+1.01, +2.37], MASSter +0.53 [+0.03, +1.02]. The 1% filter costs GLACIER 4.53 points [3.50, 5.64] and does not change MASSter (+0.41 [−0.52, +1.42]).

### Intensity weighting at instrument tolerance (hit@1 / @5 in %, ORB / QTOF hit@1 after `|`)

| Date | Weighting | GLACIER | MASSter `ll04i00m`@60 | GLACIER − MASSter hit@1 [95% mol. bootstrap] | Source |
| --- | --- | --- | --- | --- | --- |
| 2026-10-03 | sqrt (ms-pred) | **51.32** / 83.49 \| 56.6 / 23.6 | 49.03 / 82.11 \| 53.7 / 24.2 | +2.29 [+0.56, +4.11] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#intensity-weighting-at-instrument-tolerance-sqrt-vs-linear-jss-vs-li-weighted) |
| 2026-10-03 | sqrt, 1% filter | 46.79 / 79.57 | **49.44** / 82.77 | −2.65 [−4.02, −1.33] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#intensity-weighting-at-instrument-tolerance-sqrt-vs-linear-jss-vs-li-weighted) |
| 2026-10-03 | linear (JSS) | 41.92 / 76.81 \| 46.4 / 18.2 | 45.07 / 79.29 \| 49.9 / 19.5 | −3.15 [−4.77, −1.57] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#intensity-weighting-at-instrument-tolerance-sqrt-vs-linear-jss-vs-li-weighted) |
| 2026-10-03 | linear, 1% filter | 40.55 / 75.14 | 44.33 / 78.64 | −3.77 [−5.24, −2.31] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#intensity-weighting-at-instrument-tolerance-sqrt-vs-linear-jss-vs-li-weighted) |
| 2026-10-03 | Li-weighted | 48.81 / 82.12 \| 54.3 / 19.8 | 48.79 / 81.69 \| 53.9 / 21.8 | +0.02 [−1.58, +1.58] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#intensity-weighting-at-instrument-tolerance-sqrt-vs-linear-jss-vs-li-weighted) |
| 2026-10-03 | Li-weighted, 1% filter | 45.85 / 79.69 | 49.30 / 83.03 | −3.45 [−4.93, −1.99] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#intensity-weighting-at-instrument-tolerance-sqrt-vs-linear-jss-vs-li-weighted) |

Paired hit@1 changes (GLACIER / MASSter): JSS − sqrt −9.40 [−10.71, −8.16] / −3.96 [−5.05, −2.83]; Li − sqrt −2.51 [−3.52, −1.60] / −0.24 [−1.45, +1.09]; Li − JSS +6.89 [+5.78, +7.99] / +3.72 [+2.68, +4.78]; MASSter log-likelihood − this score: JSS +3.47 [+2.47, +4.49], Li −0.25 [−1.31, +0.80], sqrt −0.49 [−1.85, +0.88].

### Bin-edge and matching decomposition (hit@1 %, random ties, our candidate list, sqrt entropy)

| Date | Scoring | GLACIER | MASSter `ll04i00m`@60 | GLACIER − MASSter [95% mol. bootstrap] | Source |
| --- | --- | ---: | ---: | --- | --- |
| 2026-10-03 | ms-pred binned | 49.65 | 48.50 | +1.15 [−0.54, +2.86] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | + precursor cut on both spectra | 49.64 | 48.50 | — | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | + sqrt after summing linear (prediction) | 49.89 | 48.41 | — | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | + same for target (b3) | 49.91 | 48.42 | +1.49 [−0.26, +3.26] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | b3, bin edges shifted half a bin | 50.52 | 48.56 | +1.96 [+0.24, +3.65] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | one-to-one ±1 mDa | 49.82 | 47.84 | +1.98 [+0.40, +3.56] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | one-to-one ±2 mDa | 51.18 | 48.55 | +2.63 [+0.94, +4.31] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | one-to-one ±5 mDa | 51.41 | 49.15 | +2.26 [+0.57, +3.99] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | one-to-one ±10 mDa | 51.18 | 49.09 | +2.09 [+0.40, +3.87] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |
| 2026-10-03 | one-to-one instrument tolerance | 51.32 | 49.03 | +2.29 [+0.56, +4.11] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03) |

Derived in the source: ±10 mDa one-to-one minus b3 is GLACIER +1.27 [+0.69, +1.86], MASSter +0.66 [+0.15, +1.18]; half-bin edge shift adds +0.61 [+0.07, +1.17] (GLACIER) and +0.14 [−0.44, +0.71] (MASSter) ([finding](../findings/masster/masster_msgym_formula_retrieval.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03)).

## Formula retrieval: likelihood variants and output filters (test)

Test, 9,954 queries, expected top-k; GLACIER released checkpoint (batch 64, sparse-k 100, squared) unless stated; `ll04i00m`@60 for MASSter; all diagnostics are test reads, the test-tuned rows are bounds not reportable scores. Date 2026-10-03.

| Date | Model / setting | Run@epoch | Score | top1 | top5 | top20 | vs comparator (Δ, CI) | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-10-03 | MASSter, calibrated ll (2× tol, frozen) | `ll04i00m`@60 | `loglik_cal` | 0.528 | 0.853 | 0.965 | top2 0.686, top3 0.767, top10 0.928 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#glacier-under-the-same-likelihood-rules-added-2026-10-03) |
| 2026-10-03 | GLACIER, calibrated ll (2× tol, frozen) | released ckpt | `loglik_cal` | 0.490 | 0.829 | 0.951 | MASSter − GLACIER: top1 +0.039 [+0.022, +0.055]; top2 +0.035; top3 +0.032; top5 +0.024 [+0.009, +0.040]; top10 +0.022; top20 +0.014 [+0.003, +0.027] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#glacier-under-the-same-likelihood-rules-added-2026-10-03) |
| 2026-10-03 | MASSter, calibrated ll (1× tol, read at user request, not a selected rule) | `ll04i00m`@60 | calibrated, 1× tol | 0.531 | 0.854 | 0.965 | top2 0.691, top3 0.771, top10 0.928 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#glacier-under-the-same-likelihood-rules-added-2026-10-03) |
| 2026-10-03 | GLACIER, calibrated ll (1× tol) | released ckpt | calibrated, 1× tol | 0.491 | 0.832 | 0.952 | top2 0.655, top3 0.740, top10 0.908 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#glacier-under-the-same-likelihood-rules-added-2026-10-03) |
| 2026-10-03 | MASSter, old ll | `ll04i00m`@60 | `loglik` | 0.485 | 0.815 | 0.948 | top2 0.642, top3 0.728, top10 0.897 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#glacier-under-the-same-likelihood-rules-added-2026-10-03) |
| 2026-10-03 | GLACIER, old ll | released ckpt | `loglik` | 0.444 | 0.793 | 0.936 | top2 0.611, top3 0.700, top10 0.885 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#glacier-under-the-same-likelihood-rules-added-2026-10-03) |
| 2026-10-03 | each model at its own test-best (test-tuned bound): MASSter g 0.75 / GLACIER g 0.25 | `ll04i00m`@60 / released ckpt | likelihood sweep | 0.530 / 0.506 | — | — | +0.025 [+0.007, +0.043] top1; top10 +0.026 [+0.013, +0.041]; top20 +0.023 [+0.012, +0.037] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#each-model-in-its-own-likelihood-space-test-tuned-diagnostic-added-2026-10-03) |
| 2026-10-03 | MASSter frozen vs GLACIER test-best | `ll04i00m`@60 / released ckpt | likelihood sweep | 0.528 / 0.506 | — | — | +0.023 [+0.005, +0.040] top1 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#each-model-in-its-own-likelihood-space-test-tuned-diagnostic-added-2026-10-03) |
| 2026-10-03 | best top1 over tolerance and floor, by prediction power g (0 / 0.1 / 0.25 / 0.5 / 0.75 / 1.0) | `ll04i00m`@60 | likelihood sweep | MASSter 0.068 / 0.356 / 0.488 / 0.521 / **0.531** / 0.519 | — | — | GLACIER 0.436 / 0.482 / **0.506** / 0.504 / 0.495 / 0.479 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#each-model-in-its-own-likelihood-space-test-tuned-diagnostic-added-2026-10-03) |
| 2026-10-03 | MASSter, 1% filter on predictions (frozen rule) | `ll04i00m`@60 | `loglik_cal` | 0.488 | — | — | vs unfiltered −0.041 [−0.050, −0.030]; vs GLACIER 1% +0.045 [+0.030, +0.060]; each at own test-best 0.501 / 0.458, +0.043 [+0.028, +0.057]; vs GLACIER unfiltered own best 0.506: −0.018 [−0.034, −0.001] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#1-filter-on-predictions-test-diagnostic-added-2026-10-03) |
| 2026-10-03 | MASSter-only filter 1% (median 15 peaks) | `ll04i00m`@60 | `loglik_cal` | 0.488 | 0.824 | 0.957 | val top1 0.502 (−0.051); test Δ vs none −0.041; vs GLACIER own-best 0.506: −0.018 [−0.034, −0.001] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-only-output-filter-added-2026-10-03) |
| 2026-10-03 | MASSter-only filter 0.5% (22) | `ll04i00m`@60 | `loglik_cal` | 0.497 | 0.833 | 0.960 | val 0.515 (−0.038); test −0.032; vs own-best −0.009 [−0.026, +0.008] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-only-output-filter-added-2026-10-03) |
| 2026-10-03 | MASSter-only filter 0.1% (53) | `ll04i00m`@60 | `loglik_cal` | 0.519 | 0.843 | 0.963 | val 0.539 (−0.014); test −0.009; vs own-best +0.013 [−0.004, +0.031] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-only-output-filter-added-2026-10-03) |
| 2026-10-03 | MASSter-only filter 0.05% (76) | `ll04i00m`@60 | `loglik_cal` | 0.524 | 0.844 | 0.963 | val 0.543 (−0.010); test −0.004; vs own-best +0.018 [+0.001, +0.036] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-only-output-filter-added-2026-10-03) |
| 2026-10-03 | MASSter-only filter **0.01%** (172) | `ll04i00m`@60 | `loglik_cal` | 0.527 | 0.850 | 0.965 | val 0.550 (−0.003 [−0.006, +0.000]); test −0.002 [−0.005, +0.001]; vs own-best +0.021 [+0.004, +0.038] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-only-output-filter-added-2026-10-03) |
| 2026-10-03 | MASSter-only filter 0.001% (407) | `ll04i00m`@60 | `loglik_cal` | 0.529 | 0.852 | 0.965 | val 0.553 (−0.000); test +0.001; vs own-best +0.023 [+0.005, +0.041] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-only-output-filter-added-2026-10-03) |
| 2026-10-03 | MASSter, no filter (939 peaks) | `ll04i00m`@60 | `loglik_cal` | 0.528 | 0.853 | 0.965 | val 0.553; vs own-best +0.023 [+0.005, +0.040] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-only-output-filter-added-2026-10-03) |
| 2026-10-03 | GLACIER, ms-pred entropy (our candidate list) | released ckpt | `mspred_ent` | 0.497 | 0.827 | 0.942 | top2 0.659, top3 0.744, top10 0.899; reproduces 49.67 / 82.65 / 94.22 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-calibrated-vs-glacier-on-ms-preds-own-ranking-added-2026-10-03) |
| 2026-10-03 | MASSter calibrated (no filter) − GLACIER ms-pred entropy | `ll04i00m`@60 | `loglik_cal` | 0.528 | 0.853 | 0.965 | +0.032 [+0.015, +0.050] top1; top5 +0.026 [+0.009, +0.046]; top20 +0.023 [+0.010, +0.036] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-calibrated-vs-glacier-on-ms-preds-own-ranking-added-2026-10-03) |
| 2026-10-03 | MASSter filter 0.01% / 0.05% / 0.1% / 1% − GLACIER ms-pred entropy | `ll04i00m`@60 | `loglik_cal` | — | — | — | top1 +0.030 [+0.013, +0.048] / +0.027 [+0.010, +0.045] / +0.022 [+0.005, +0.041] / −0.009 [−0.025, +0.008] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-calibrated-vs-glacier-on-ms-preds-own-ranking-added-2026-10-03) |
| 2026-10-03 | MASSter capped at top 100 peaks (median/max 100 / 100) | `ll04i00m`@60 | `loglik_cal` | 0.527 | 0.848 | 0.966 | val top1 Δ vs none −0.000 [−0.005, +0.005]; vs GLACIER ms-pred 0.497: +0.030 [+0.013, +0.048], top5 +0.022 [+0.002, +0.041] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-capped-at-100-peaks-glaciers-budget-added-2026-10-03) |
| 2026-10-03 | MASSter ≥ 0.025% of base (median/max 114 / 1,365) | `ll04i00m`@60 | `loglik_cal` | 0.527 | 0.847 | 0.964 | val Δ −0.008 [−0.012, −0.004]; vs GLACIER +0.031 [+0.014, +0.049] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-capped-at-100-peaks-glaciers-budget-added-2026-10-03) |
| 2026-10-03 | MASSter top 100 and ≥ 0.025% | `ll04i00m`@60 | `loglik_cal` | 0.525 | 0.844 | 0.965 | val Δ −0.005 [−0.011, +0.000]; vs GLACIER +0.029 [+0.011, +0.046] | [finding](../findings/masster/masster_msgym_formula_retrieval.md#masster-capped-at-100-peaks-glaciers-budget-added-2026-10-03) |

## Mass-track retrieval (test, 9,954 queries)

`pow075` `q3oxbpgx`@55 only, no comparison model run; 2,466,243 candidates (one unloadable non-truth candidate excluded; 19 queries with two truth rows); expected top-k in %, random ties; `loss_batch_size` 64. Date 2026-10-08.

| Date | Model / arm | Run@epoch | Score | top-1 [95% CI] | top-3 | top-5 | top-20 | ORB top-1 (8,374) | QTOF top-1 (1,580) | Source |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-10-08 | `pow075` | `q3oxbpgx`@55 | `loglik` | 71.35 [67.97, 74.66] | 86.78 | 91.23 | 96.59 | 75.89 | 47.28 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#mass-track-test-retrieval-pow075-2026-10-08) |
| 2026-10-08 | `pow075` | `q3oxbpgx`@55 | `loglik_cal` | 74.52 [70.92, 77.93] | 89.42 | 93.54 | 97.98 | 79.51 | 48.10 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#mass-track-test-retrieval-pow075-2026-10-08) |
| 2026-10-08 | `pow075` | `q3oxbpgx`@55 | `cos_sim_0.01` | 52.08 [49.15, 54.95] | 72.16 | 78.29 | 88.38 | 56.22 | 30.16 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#mass-track-test-retrieval-pow075-2026-10-08) |
| 2026-10-08 | `pow075` | `q3oxbpgx`@55 | `mspred_ent` | 69.18 [65.63, 72.48] | 83.80 | 87.89 | 94.91 | 74.06 | 43.29 | [finding](../findings/masster/masster_msgym_formula_retrieval.md#mass-track-test-retrieval-pow075-2026-10-08) |

Paired top-1: `loglik` − cosine +19.26 [+17.48, +21.06]; `loglik` − entropy +2.17 [+0.79, +3.43]. Likelihood margin over cosine is larger than on the formula track (+14.1) and largest on QTOF (cosine top-20 75.4 vs 92.2 for `loglik`) ([finding](../findings/masster/masster_msgym_formula_retrieval.md#mass-track-test-retrieval-pow075-2026-10-08)).

## Retrieval pilot (400 queries)

400 molecules from the val split (seed 20261002; 375 ORB / 25 QTOF; median nearest-train Tanimoto 0.36), `ll04i00m`@60 (FT→ORB rename), val only, formula track unless stated; random hit@1 = 0.025. Not the plan's Gate 1; the `ll`-based blends, `cos_sqrt`, `ent` and the precursor split were added after the first read. Date 2026-10-02.

### Single scores, formula track (400 queries)

| Date | Model / arm | Run@epoch | Score | hit@1 [95% CI] | hit@5 | hit@20 | vs comparator (Δ vs `cos`, CI) | Source |
| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| 2026-10-02 | MASSter | `ll04i00m`@60 | (a) `cos` | 0.478 [0.428, 0.528] | 0.760 | 0.888 | — | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#single-scores-400-queries-random-hit1--0025) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos_hun` | 0.480 | 0.758 | 0.880 | +0.003 [−0.015, +0.018] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#single-scores-400-queries-random-hit1--0025) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | (b) `supp` | 0.239 | 0.540 | 0.808 | −0.239 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#single-scores-400-queries-random-hit1--0025) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | (b) `supp_c` | 0.187 | 0.446 | 0.713 | −0.291 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#single-scores-400-queries-random-hit1--0025) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | **(c) `ll`, ε 1e-5** | **0.625 [0.578, 0.668]** | **0.908** | **0.983** | **+0.148 [+0.100, +0.195]** | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#single-scores-400-queries-random-hit1--0025) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | (c) `ll`, ε 1e-4 / 1e-6 | 0.618 / 0.630 | 0.888 / 0.908 | 0.978 / 0.983 | +0.140 / +0.153 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#single-scores-400-queries-random-hit1--0025) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | (F1) `cost` / `cost_c` | 0.342 / 0.340 | 0.705 / 0.688 | 0.932 / 0.915 | −0.136 / −0.138 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#single-scores-400-queries-random-hit1--0025) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | (F3) `pars` / `pars_c` | 0.330 / 0.314 | 0.703 / 0.692 | 0.935 / 0.940 | −0.147 / −0.164 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#single-scores-400-queries-random-hit1--0025) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos_sqrt` (post hoc) | 0.588 | 0.848 | 0.935 | vs `ll` −0.038 [−0.078, +0.000] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#what-the-likelihood-gain-is-post-hoc-400-queries) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `ent` (unweighted entropy similarity, post hoc) | 0.573 | 0.845 | 0.930 | vs `ll` −0.053 [−0.090, −0.015] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#what-the-likelihood-gain-is-post-hoc-400-queries) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `ll` without the precursor term (post hoc) | 0.628 | 0.895 | 0.981 | vs `ll` +0.003 [−0.020, +0.023]; precursor carries a median 0.4% of query intensity | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#what-the-likelihood-gain-is-post-hoc-400-queries) |
| 2026-10-02 | MASSter, `ll` ε sweep | `ll04i00m`@60 | `ll` ε 1e-5 / 1e-4 / 1e-3 / 1e-2 | 0.625 / 0.618 / 0.595 / 0.525 | — | — | capping the per-peak penalty hurts | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#query-peaks-no-subformula-explains-added-2026-10-02-post-hoc) |

### Blends, 176 odd-hash molecules (α chosen on the 224 even)

| Date | Model / arm | Run@epoch | Blend | α (even) | hit@1 | Δhit@1 [95% CI] | Δhit@5 | Δhit@20 | Source |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos` + `supp_c` | 0.9 | 0.432 | +0.000 [−0.023, +0.023] | +0.000 | +0.011 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#blends-reported-on-the-176-odd-molecules) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos` + `cost_c` | 0.7 | 0.466 | +0.034 [−0.006, +0.080] | +0.074 [+0.028, +0.125] | +0.063 [+0.028, +0.102] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#blends-reported-on-the-176-odd-molecules) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos` + `cost` | 0.7 | 0.483 | +0.051 [+0.006, +0.097] | +0.063 [+0.017, +0.108] | +0.063 [+0.023, +0.102] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#blends-reported-on-the-176-odd-molecules) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos` + `pars_c` | 0.8 | 0.466 | +0.034 [−0.006, +0.074] | +0.051 [+0.011, +0.097] | +0.046 [+0.011, +0.085] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#blends-reported-on-the-176-odd-molecules) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos` + `ll` | 0.2 | 0.585 | +0.153 [+0.091, +0.216] | +0.125 [+0.074, +0.176] | +0.080 [+0.046, +0.125] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#blends-reported-on-the-176-odd-molecules) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `ll` + `cost_c` (post hoc) | 0.9 | 0.608 | −0.011 [−0.034, +0.011] | +0.006 | +0.000 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#blends-reported-on-the-176-odd-molecules) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `ll` + `pars_c` (post hoc) | 0.9 | 0.608 | −0.011 [−0.034, +0.011] | +0.000 | +0.000 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#blends-reported-on-the-176-odd-molecules) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `ll` + `supp_c` (post hoc) | 0.9 | 0.625 | +0.006 [−0.011, +0.028] | −0.006 | +0.000 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#blends-reported-on-the-176-odd-molecules) |

Odd-half baselines: cosine 0.432 / 0.767 / 0.898; `ll` 0.619 / 0.909 / 0.983.

### Controls, subgroups, unexplained peaks

| Date | Evaluation | Result | Source |
| --- | --- | --- | --- |
| 2026-10-02 | spectrum-blind consensus pseudo-query (top 30 bins of the mean predicted spectrum of the query's candidates), hit@1 | `cos` 0.033, `ll` 0.028, `cost_c` 0.018, `pars_c` 0.031 (range 0.018-0.033) | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#controls) |
| 2026-10-02 | same-formula partner query (49 queries), truth hit@1 | `cos` 0.102, `ll` 0.061, `cost_c` 0.041 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#controls) |
| 2026-10-02 | support size and bond count, hit@1 | 0.019-0.028 either direction | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#controls) |
| 2026-10-02 | subgroup hit@1 `cos` / `ll` / `cost_c` / `pars_c`: ORB (375) | 0.493 / 0.635 / 0.348 / 0.324 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#subgroups-hit1) |
| 2026-10-02 | QTOF (25) | 0.240 / 0.480 / 0.213 / 0.153 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#subgroups-hit1) |
| 2026-10-02 | nearest-train Tanimoto < 0.3 (64) | 0.297 / 0.438 / 0.264 / 0.212 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#subgroups-hit1) |
| 2026-10-02 | Tanimoto 0.3-0.4 (203) | 0.537 / 0.675 / 0.372 / 0.354 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#subgroups-hit1) |
| 2026-10-02 | Tanimoto ≥ 0.4 (133) | 0.474 / 0.639 / 0.327 / 0.302 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#subgroups-hit1) |
| 2026-10-02 | query-intensity classes, mean share (median; 95th pct): precursor / feasible / isotopologue / unexplained | 0.133 (0.004; 0.760) / 0.848 (0.979; 1.000) / 0.001 (0.000; 0.006) / 0.017 (0.005; 0.056) | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#query-peaks-no-subformula-explains-added-2026-10-02-post-hoc) |
| 2026-10-02 | donor formula explains / true formula explains non-precursor intensity | 0.765 / 0.969 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#query-peaks-no-subformula-explains-added-2026-10-02-post-hoc) |
| 2026-10-02 | standardized candidates vs raw, hit@1 / @5 (`cos`; `ll`; `cos_sqrt`; `ent`) | raw 0.477 / 0.760; 0.625 / 0.907; 0.588 / 0.848; 0.573 / 0.845. standardized 0.482 / 0.752; 0.623 / 0.905; 0.583 / 0.848; 0.568 / 0.840. Δhit@1 +0.005 [−0.007, +0.020]; −0.003 [−0.013, +0.007]; −0.005 [−0.018, +0.007]; −0.005 [−0.015, +0.007] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#pipeline-findings) |

### Mass track, same 400 queries (100,001 candidates, median 254 per query)

| Date | Model / arm | Run@epoch | Score | hit@1 | hit@5 | hit@20 | top-1 has true formula | Δhit@1 vs `cos` | Source |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos` | 0.623 | 0.800 | 0.885 | 0.782 | — | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos_hun` | 0.640 | 0.812 | 0.885 | 0.800 | +0.018 [+0.000, +0.035] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cos_sqrt` | 0.755 | 0.877 | 0.930 | 0.875 | +0.133 [+0.092, +0.172] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `ent` | 0.743 | 0.870 | 0.917 | 0.860 | +0.120 [+0.083, +0.160] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | **`ll`, ε 1e-5** | **0.828** | **0.950** | **0.983** | **0.950** | **+0.205 [+0.160, +0.250]** | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `ll`, ε 1e-3 / 1e-2 | 0.785 / 0.680 | 0.912 / 0.848 | 0.958 / 0.917 | 0.900 / 0.807 | +0.163 / +0.058 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | `cfeas` (formula explainability alone) | 0.008 | 0.049 | 0.178 | 0.565 | −0.614 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02) |
| 2026-10-02 | MASSter | `ll04i00m`@60 | consensus control `cos` / `ll` | 0.005 / 0.003 | — | — | — | — | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#mass-track-same-400-queries-added-2026-10-02) |

`ll` picks the right formula 95% of the time; 49 of its 69 misses lose to a same-formula isomer. 893 candidates (0.9%, in 270 queries) could not be featurized and are ranked last ("opt"). Removing dirt peaks from `ll` changes no hit@1, @5 or @20.

### GLACIER released checkpoint on the same 400 queries (`--sparse-k 256`, output squared, top-256 peaks)

| Date | Score | MASSter `ll04i00m`@60 hit@1 / @5 / @20 | GLACIER hit@1 / @5 / @20 | GLACIER − MASSter hit@1 [95% CI] | Source |
| --- | --- | --- | --- | --- | --- |
| 2026-10-02 | `cos` | 0.477 / 0.760 / 0.887 | 0.465 / 0.748 / 0.875 | −0.013 [−0.068, +0.037] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#glaciers-released-msg-checkpoint-on-the-same-400-queries-added-2026-10-02) |
| 2026-10-02 | `cos_sqrt` | 0.588 / 0.848 / 0.935 | 0.562 / 0.858 / 0.948 | −0.025 [−0.080, +0.028] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#glaciers-released-msg-checkpoint-on-the-same-400-queries-added-2026-10-02) |
| 2026-10-02 | `ent` | 0.573 / 0.845 / 0.930 | 0.565 / 0.845 / 0.940 | −0.007 [−0.060, +0.045] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#glaciers-released-msg-checkpoint-on-the-same-400-queries-added-2026-10-02) |
| 2026-10-02 | `ll` (ε 1e-5) | 0.625 / 0.907 / 0.983 | 0.615 / 0.887 / 0.958 | −0.010 [−0.060, +0.037] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#glaciers-released-msg-checkpoint-on-the-same-400-queries-added-2026-10-02) |
| 2026-10-02 | `ll` without precursor | 0.627 / 0.895 / 0.980 | 0.600 / 0.877 / 0.958 | −0.027 [−0.077, +0.023] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#glaciers-released-msg-checkpoint-on-the-same-400-queries-added-2026-10-02) |

Truth's mean `cos` 0.615 (GLACIER) vs 0.634 (MASSter); QTOF (n=25) GLACIER `cos` hit@1 0.240, same as MASSter. GLACIER `ll` is on its top 256 peaks vs MASSter's mean 952.

### Interim full-test read (19 shards, 9,500 queries, 2,279 molecules; superseded by the 20-shard read)

GLACIER at batch 64 / sparse-k 100. Date 2026-10-03.

| Date | Model / score | Overall hit@1 | ORB hit@1 | QTOF hit@1 | Notes | Source |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 2026-10-03 | MASSter linear cosine (`ll04i00m`@60) | 0.362 | 0.402 | 0.143 | 8,025 ORB / 1,475 QTOF queries | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |
| 2026-10-03 | GLACIER linear cosine | 0.339 | 0.376 | 0.137 | GLACIER − MASSter −0.023 [−0.037, −0.009] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |
| 2026-10-03 | MASSter sqrt cosine | 0.459 | 0.509 | 0.186 | | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |
| 2026-10-03 | GLACIER sqrt cosine | 0.429 | 0.476 | 0.171 | GLACIER − MASSter −0.030 [−0.045, −0.013] | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |
| 2026-10-03 | MASSter likelihood | 0.489 | 0.542 | 0.199 | | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |
| 2026-10-03 | mean true-candidate / strongest-decoy linear cosine: ORB MASSter | — | 0.59306 / 0.58385 | — | truth − decoy +0.00950 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |
| 2026-10-03 | ORB GLACIER | — | 0.55877 / 0.57060 | — | −0.01115 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |
| 2026-10-03 | QTOF MASSter | — | — | 0.47408 / 0.58364 | −0.10956 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |
| 2026-10-03 | QTOF GLACIER | — | — | 0.45744 / 0.58214 | −0.12470 | [pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md#retrieval-versus-forward-similarity-2026-10-03-interim-full-test-read) |

## QTOF diagnostic (val, MASSter `ll04i00m`@60)

Date 2026-10-03; [source](../findings/masster/masster_msgym_formula_retrieval.md#qtof-diagnostic-val-masster).

| Quantity | ORB | QTOF |
| --- | ---: | ---: |
| intensity explained by precursor subformulas at instrument tolerance (chance control 0%) | 99.7% | 100% |
| median mass error of explained peaks | 1.2 ppm | 3.8 ppm |
| replicate agreement, same molecule/instrument/CE (inst-tol cosine) | 0.958 | 0.913 |
| train-molecule fit, inst-tol cosine / `cos_sim_0.01` | 0.940 / 0.916 | 0.921 / 0.819 |
| new-molecule (val) truth fit, inst-tol cosine | 0.601 | 0.546 |
| median candidates per query | 230 | 88 |
| mean truth cosine / mean best-decoy cosine | 0.587 / 0.578 | 0.477 / 0.580 |

## Swap tests (test, 9,954 queries)

GLACIER contrastive fine-tuning wrong-spectrum test; released GLACIER checkpoint (w/ CF) and `ll04i00m`@60, our candidate list, saved predictions (no new inference). Donors: `sameF` (1,114 queries, same formula, different 2D key) and `near` (nearest precursor m/z, all queries). `ent` = ms-pred binned 0.01 Da sqrt entropy; `ll` = frozen calibrated likelihood. Score = expected hit@k of the query's own truth; chance = mean of min(k, n)/n. Date 2026-10-03.

| Date | Spectrum | n | chance hit@1 / @5 / @20 | GLACIER `ent` | GLACIER `ll` | MASSter `ent` | MASSter `ll` | Source |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| 2026-10-03 | true | 9,954 | 0.025 / 0.105 / 0.270 | 0.497 / 0.826 / 0.942 | 0.490 / 0.829 / 0.951 | 0.485 / 0.818 / 0.949 | 0.528 / 0.853 / 0.965 | [finding](../findings/masster/masster_glacier_released_ckpt.md#results-1) |
| 2026-10-03 | `near` (wrong) | 9,954 | 0.025 / 0.105 / 0.271 | 0.030 / 0.117 / 0.304 | 0.030 / 0.113 / 0.302 | 0.032 / 0.130 / 0.319 | 0.031 / 0.122 / 0.315 | [finding](../findings/masster/masster_glacier_released_ckpt.md#results-1) |
| 2026-10-03 | true, `sameF` subset | 1,114 | 0.014 / 0.072 / 0.192 | 0.433 / 0.692 / 0.873 | 0.440 / 0.688 / 0.867 | 0.425 / 0.727 / 0.938 | 0.481 / 0.738 / 0.929 | [finding](../findings/masster/masster_glacier_released_ckpt.md#results-1) |
| 2026-10-03 | `sameF` (wrong) | 1,114 | 0.016 / 0.074 / 0.195 | 0.076 / 0.205 / 0.443 | 0.079 / 0.205 / 0.446 | 0.105 / 0.311 / 0.562 | 0.107 / 0.281 / 0.537 | [finding](../findings/masster/masster_glacier_released_ckpt.md#results-1) |

| Date | Contrast, hit@1 | `near` | `sameF` | Source |
| --- | --- | --- | --- | --- |
| 2026-10-03 | GLACIER `ent` − chance | +0.005 [−0.001, +0.011] | +0.060 [+0.029, +0.101] | [finding](../findings/masster/masster_glacier_released_ckpt.md#results-1) |
| 2026-10-03 | MASSter `ll` − chance | +0.005 [−0.002, +0.013] | +0.091 [+0.054, +0.132] | [finding](../findings/masster/masster_glacier_released_ckpt.md#results-1) |
| 2026-10-03 | GLACIER − MASSter, `ent` | −0.002 [−0.009, +0.005] | −0.029 [−0.060, +0.008] | [finding](../findings/masster/masster_glacier_released_ckpt.md#results-1) |
| 2026-10-03 | GLACIER − MASSter, `ll` | −0.000 [−0.007, +0.006] | −0.028 [−0.056, +0.006] | [finding](../findings/masster/masster_glacier_released_ckpt.md#results-1) |

Under `sameF` at @5 and @20 MASSter leads GLACIER by 0.076-0.119 (`ent` @5: −0.106 [−0.209, −0.009]). Spectrum-free rankers, hit@1 (chance 0.025): GLACIER sharpness 0.019, MASSter sharpness 0.019, GLACIER support size 0.025, MASSter support size 0.016. When the donor's own truth is in the query's candidate list (about 600 queries per arm), the swapped spectrum puts it first 18-24% of the time for both models ([finding](../findings/masster/masster_glacier_released_ckpt.md#results-1)).

## Retired models

| Date | Model / arm | Run@epoch | Val (metric not named in the hub) | vs comparator | Notes | Source |
| --- | --- | --- | --- | --- | --- | --- |
| — | ChemBERTa-3 E1 | `tm9qwq78` | peak after epoch 0 at 0.199; end 0.169 | structure-blind prior scores 0.194 | retired 2026-10-12; plan removed 2026-10-14 (`git show 8a3cc7a5:docs/plans/chemberta3_msgym_plan.md`); number only in the hub | [hub](../workstreams/msgym.md#settled--read-before-re-proposing) |
| — | ChemBERTa-3 T2 | `ku742mhq` | peak after epoch 0 at 0.194; end 0.152 | structure-blind prior scores 0.194 | random SMILES hurt; `chemlm` E4 never ran; code at `13b1599d` | [hub](../workstreams/msgym.md#settled--read-before-re-proposing) |
