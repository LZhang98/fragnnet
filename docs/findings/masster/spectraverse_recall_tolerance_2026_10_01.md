# Spectraverse MASSter recall at instrument tolerance

> **Status:** FINDING · **Updated:** 2026-10-01 · Two finished validation folds rescored on 4,096 spectra each; recall logging corrected for future evaluations.

## Result

**Caveat (2026-10-01):** both checkpoints were trained with the retired `FT` label, and the script builds them from the current OB config. The sorted instrument index therefore fed OB spectra the trained NA embedding. OB rows below were scored with the wrong instrument row; QTOF rows are unaffected. See [the `[M]+` finding](spectraverse_mplus_handling_2026_10_01.md#old-spv-checkpoints-carry-the-retired-ft-label).

The W&B `val_recall` and `val_wrecall` metrics use the global 10 ppm tolerance with a 200 Da floor. The SPV loss uses 10 ppm / 200 Da for OB, 20 ppm / 750 Da for QTOF, and 20 ppm / 500 Da for Other and NA. Thus the logged recall is stricter than the loss for non-OB spectra. The independent scoring implementation was checked against `calculate_match_mzs` on 100 random mass sets at all three tolerances.

| Last checkpoint and validation fold | Sampled spectra | Peak recall, flat 10 ppm | Peak recall, instrument tolerance | Intensity-weighted recall, flat 10 ppm | Intensity-weighted recall, instrument tolerance |
| --- | ---: | ---: | ---: | ---: | ---: |
| [cv0 `ggifzj9m`](https://wandb.ai/frag-gnn/frag-gnn/runs/ggifzj9m) | 4,096 | 0.4619 | 0.5419 | 0.6415 | 0.7068 |
| [cv1 `u46v0k0g`](https://wandb.ai/frag-gnn/frag-gnn/runs/u46v0k0g) | 4,096 | 0.4098 | 0.5399 | 0.6184 | 0.7302 |

The QTOF change is largest: cv0 recall 0.2817 to 0.5139 (1,339 spectra), and cv1 0.3985 to 0.6935 (1,663 spectra). OB uses the same tolerance in both reads, so its recall stays 0.5597 on cv0 and 0.4202 on cv1. Other remains low even at its wider tolerance: 0.1129 on cv0 (73 spectra), 0.1752 on cv1 (156 spectra). NA is present too: 116 spectra at 0.7417 recall on cv0 and 210 at 0.7726 on cv1. The literal `NA` label must be preserved when reading the CSV (`keep_default_na=False`); pandas otherwise treats it as missing and omits that group from `groupby` summaries.

The `candidate_mzs_by_cut` mass set and emitted `pred_mzs` give **identical target coverage on every sampled spectrum** at both tolerances, including intensity-weighted coverage. Here candidate coverage is the ceiling from the run's selected fragment/ion mass set, not an exhaustive chemical-formula ceiling. The gap after rescoring is missing mass support under that set; these recall values do not measure intensity ranking among covered masses.

## Method and limits

Run `scripts/debug/analyze_spectraverse_recall.py` with each fold's W&B `ckpt/last.ckpt`, `--sample-size 4096`, and the corresponding `--fold`. The script samples validation spectrum IDs with seed 20261001, uses the checkpoint's original local SPV config and split, preprocesses targets with `preproc_spec`, and scores each emitted prediction and candidate m/z against those same targets. Per-spectrum tables are in `output/analysis/spectraverse_recall_2026_10_01/` (ignored artifact store). The checkpoint SHA-256 values are `f87d11dab3b70c15803890633aa70cdff03eb407a06c41c235d3db0867aad0a9` for cv0 and `9a9c0cf6a46c71d55d89039805790acc9a0d62bb16b48e034c717693349fcfd9` for cv1. Source revision was `69209277`; unrelated working-tree edits existed, while the two SPV configs had no diff.

These are sampled last-checkpoint results, not full validation or best-checkpoint results. W&B's full-validation last-epoch recall was 0.4499 / 0.4125 for cv0 / cv1; sample values differ by sampling. The diagnostic does not adjudicate whether the instrument-specific loss windows are chemically appropriate. A further mass-error calibration read would be needed before tightening them.

## Metric correction (2026-10-01)

`SpectrumPL` now uses `inst_type_loss_tol` for `recall` and `wrecall` when that map is configured. The historical W&B values above remain flat-10-ppm results; they were not retroactively rewritten. The OB/QTOF regression checks that both recall metrics differ at an 8 mDa offset at m/z 500. Precision retains its existing global-tolerance definition. The focused tolerance suite passed (36 tests).

## Wider-window diagnostic (2026-10-01)

A checkpoint sweep on 512 validation spectra per fold reused the same sampled predictions and targets. The shifted-mass control adds 0.1 Da to each target m/z before matching against the same selected candidate masses. It estimates chance candidate proximity, not formal false-discovery rate; only one shift and two small samples were tested. Means below pool both folds by spectrum.

| Instrument (spectra) | Window | Candidate recall | Candidate wRecall | Shifted-target recall | Shifted-target wRecall |
| --- | --- | ---: | ---: | ---: | ---: |
| OB (567) | current 10 ppm / 200 Da | 0.521 | 0.733 | 0.011 | 0.009 |
| OB (567) | 20 ppm / 500 Da | 0.610 | 0.773 | 0.053 | 0.043 |
| OB (567) | 50 ppm / 750 Da | 0.745 | 0.865 | 0.142 | 0.119 |
| QTOF (375) | current 20 ppm / 750 Da | 0.603 | 0.735 | 0.061 | 0.053 |
| QTOF (375) | 30 ppm / 750 Da | 0.645 | 0.766 | 0.080 | 0.068 |
| QTOF (375) | 50 ppm / 750 Da | 0.692 | 0.803 | 0.126 | 0.110 |

The OB 20 ppm / 500 Da window adds 0.040 observed weighted recall but also 0.034 shifted-target weighted recall; a broader window does not clearly recover chemically correct support. QTOF 30 ppm / 750 Da adds 0.031 observed versus 0.015 shifted-target weighted recall. This control is too weak to calibrate a training-loss change. On the full processed SPV data, precursor theoretical-mass errors have 95th-percentile absolute values of 6.08 ppm (OB) and 7.55 ppm (QTOF); precursor calibration alone does not justify a broad increase. The 512-spectrum per-fold sweep artifacts are `output/analysis/spectraverse_recall_2026_10_01/spv_tol_decoy_cv{0,1}_512.csv`. No SPV tolerance config was changed. Next: test multiple decoy offsets and trusted formula-assigned fragment peaks by source and adduct before choosing a new loss window.

## Fixed ±0.1 Da diagnostic (2026-10-01)

On the same 512-validation-spectrum samples per fold, a fixed absolute ±0.1 Da window gives selected-candidate and emitted-prediction coverage that is identical spectrum by spectrum. The checkpoint and target preprocessing are unchanged.

| Fold | Instrument-tolerance candidate recall / wRecall | ±0.1 Da candidate recall / wRecall |
| --- | ---: | ---: |
| cv0 | 0.551 / 0.716 | 0.822 / 0.889 |
| cv1 | 0.546 / 0.733 | 0.780 / 0.894 |

The wider window also raises OB recall from 0.586 to 0.884 (cv0) and 0.433 to 0.742 (cv1), and QTOF from 0.494 to 0.712 and 0.691 to 0.825. At m/z 500, ±0.1 Da is ±200 ppm, versus the current OB 10 ppm and QTOF 20 ppm windows. Precursor theoretical-mass errors on all processed SPV data have 95th-percentile absolute values of 6.08 ppm (OB) and 7.55 ppm (QTOF), so this broad recall gain is not evidence that the instruments are miscalibrated by 0.1 Da. A +0.5 Da shifted-target control matches only 0.026/0.032 peak recall, but it is a weak null because it changes mass defect. The ±0.1 Da numbers are a broad mass-proximity ceiling, not calibrated chemical support. Artifacts: `output/analysis/spectraverse_recall_2026_10_01/spv_abs0p1_cv{0,1}_512.csv`. No training tolerance was changed.

## Fixed ±0.05 Da read (2026-10-01)

Using the saved nearest-candidate errors from the same 512 spectra per fold as the ±0.1 Da read, candidate recall / wRecall is 0.7589 / 0.8443 on cv0 and 0.7145 / 0.8463 on cv1. The ±0.1 Da values on exactly these spectra are 0.8218 / 0.8890 and 0.7803 / 0.8944. OB contributes 0.8337 / 0.9117 (cv0) and 0.6721 / 0.8323 (cv1) at ±0.05 Da; QTOF contributes 0.6289 / 0.7244 and 0.7756 / 0.8816. At m/z 500, ±0.05 Da is ±100 ppm, so this is still a broad mass-proximity diagnostic rather than a calibrated training tolerance. Source peak-level artifacts: `output/analysis/spectraverse_recall_2026_10_01/spv_nearest_cv{0,1}_512_peaks.csv`.

## Follow-up training decision (2026-10-01)

The user selected fixed ±0.05 Da for a new SPV MASSter training arm. The ten prepared configs and checks are recorded in the [training plan](../../plans/masster/spectraverse_abs005_training_plan.md). This decision does not make the wider-window matches chemically validated; the historical instrument-tolerance configs and completed runs remain the control. No full training run has started.
