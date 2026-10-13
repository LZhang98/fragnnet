# MASSter vs GLACIER's released checkpoint on the MassSpecGym simulation test set

> **Status:** FINDING · **Updated:** 2026-10-06
> Linear-cosine gap +0.026 for MASSter over GLACIER's released checkpoint; instrument audit 2026-10-03; GLACIER outputs sqrt intensity.
>
> GLACIER predicts sqrt(intensity), so its output is squared before a linear comparison.
> The test set is the 9,954-spectrum MassSpecGym simulation-challenge test set. On the linear
> cosine MassSpecGym requires (`cos_sim`), computed with our exact `cos_sim_0.01`, MASSter
> `ll04i00m` reaches **0.5713** against **0.5455** for GLACIER's released checkpoint. The gap is
> **+0.026** (99.9% CI +0.009 to +0.044, molecule bootstrap), and it is concentrated at high CE
> and on FT. In sqrt space, where GLACIER is trained and ms-pred evaluates, the gap shrinks to
> +0.011 (CI −0.002 to +0.024), or +0.002 to +0.006 with per-peak sqrt at a matched peak count.
> The first version of this finding scored GLACIER's output un-squared (0.5430, +0.028).

## Setup

- **GLACIER.** This is the MassSpecGym checkpoint linked from the
  [ms-pred README](https://github.com/coleygroup/ms-pred/blob/main/README.md) (Dropbox,
  downloaded 2026-09-24). Its internal path is `results/joint_train_msg/split_rnd1/version_5`,
  and it early-stopped on val loss after 24 epochs. It ran through ms-pred's own
  `src/ms_pred/glacier/predict_smis_joint.py` (`main` at `318bb7b`) with `--sparse-k 256`. (Correction 2026-10-03: this is not upstream's setting. `max_k` in
  ms-pred's retrieval entries is the candidate-list size; upstream's `03_run_retrieval.py` uses
  `--sparse-k 100 --batch-size 64`. GLACIER's output also depends on batch size; see the
  [full retrieval finding](masster_msgym_formula_retrieval_full_2026_10_03.md#glacier-batch-dependence).) All 9,954 spectra got a prediction. At
  `--sparse-k 1024` it scores slightly lower, so 256 does not truncate it.
- **Is it the paper's model?** It is the **w/ CF** model (resolved 2026-10-03; the reasoning below
  that called it w/o CF is wrong). Its EarlyStopping patience is 3, which `train_contr_joint.py`
  hardcodes; `train_joint.py` uses 5. The stage-1 hyperparameters it stores are expected, because
  the CF script restores stage 1 with `load_from_checkpoint` and sets the CF values only as
  attributes. Ranked by ms-pred's protocol it reproduces the paper's w/ CF retrieval row; see the
  [full retrieval finding](masster_msgym_formula_retrieval_full_2026_10_03.md#matching-glaciers-reported-retrieval-numbers-added-2026-10-03).
  Original assessment, superseded: The GLACIER paper,
  Table 4, gives MassSpecGym cosine 0.58 (95% CI 0.57-0.59) without contrastive fine-tuning and
  0.58 with it, and JSS 0.50 / 0.48. The paper does not say whether the released weights are
  the reported models.
  - For: the link is in the official README. The zip is dated 2026-06-11/12, which fits the
    June 2026 preprint. All 33 model hyperparameters that `configs/glacier/joint_train_msg.yaml`
    sets match the checkpoint. It differs from `joint_contr_finetune_msg.yaml` on learning rate
    (4e-4 vs 2.7e-4) and `contr_threshold`, so it is the joint-training (w/o CF) stage.
  - Against: it was saved under `results/joint_train_msg/split_rnd1/version_5` with Lightning
    1.6.5. The public config's run directory would be `results/glacier_msg/split_rnd1/`, so this
    is an internal run, the sixth version there.
  - Score: in their evaluation space it scores 0.566-0.577 against the reported 0.58
    (0.57-0.59). That is consistent, but it does not identify the file.
- **What that checkpoint trained on.** Only indirectly known: it ran 3,098 optimizer steps per
  epoch at batch 32. That matches the simulation-challenge train fold (99,341 spectra, 3,105
  batches). All train-fold spectra would be 6,067 batches, and all known-CE ones 3,176. The
  GLACIER paper quotes 231,104 spectra for the dataset and does not say which it trained on.
  For the missing-CE handling in its code, see the MSGYM entry in
  [project state](../../project-state.md).
- **Ours.** `ll04i00m` at epoch 60 (`model-epoch=060.ckpt` on W&B), the GLACIER-parity-features
  arm. Its val best is 0.5705; the best MSGYM run, `1imawkx1` (0.5754), uploaded no checkpoint.
  So this number is a conservative stand-in for our best recipe. It was exported with
  `scripts/eval/run_save_inference_eval.py --eval_split test`.
- **Test set.** Our `ms_gym_benchmark_1_5` test split is exactly MassSpecGym's 9,954
  `simulation_challenge` test rows. Identifier, CE, precursor m/z and instrument all match the raw
  TSV. Every row has a numeric CE, so missing-CE handling does not enter at inference.

## GLACIER's output is sqrt intensity

In ms-pred, `data_scripts/forms/01_assign_subformulae.py` calls `MassSpec.process_spec_file`,
which calls `common.process_spec_file`. That function divides by the maximum and takes `np.sqrt`
before `ms2_inten` is written to `no_subform.hdf5`. `data_scripts/dag/add_dag_intens.py` maps
those values onto GLACIER's training targets, and GLACIER trains with a cosine loss on them.
Nothing in `src/ms_pred/glacier/` squares the output.

ms-pred's `analysis/spec_pred_eval.py` reads the same `ms2_inten` as its target, so its cosine
compares sqrt with sqrt. Empirically, squaring GLACIER's output raises its score in both spaces:
linear 0.5430 → 0.5455, sqrt 0.5125 → 0.5771.

MassSpecGym's own simulation code (`massspecgym/models/simulation/base.py`, `f259fe3`) requires
`cos_sim`. That is the cosine of untransformed intensities; `cos_sim_sqrt` is optional. Our
`cos_sim_0.01` is the same linear quantity.

## Metric and its validation

The scorer applies fragnnet's `batched_bin_func` and `cos_sim_helper` with the eval settings to
the test targets, dumped as the eval dataset builds them. The settings are 0.01 Da, sum pooling
and `mz_max` 1500; the sqrt variant takes the square root of binned intensities, as our
`cos_sim_sqrt_0.01` does. It reproduces the export's per-spectrum `cos_sim_0.01` for `ll04i00m`
on all 9,954 spectra (max |diff| 1.4e-6) only when it copies the eval's numerics: CUDA, float32
targets, float64 predicted m/z, batches of 64.

Many MassSpecGym target m/z values are rounded to 0.01 Da, so they sit exactly on a bin edge. CPU
binning or other dtype mixes move 155-482 spectra, and the mean by up to 0.0004.

## Results

### Four-metric table (model's own `metric_fn`)

Both models are scored by MASSter's `model.metric_fn` on the same targets
(`model.preproc_spec`), with the eval's tolerance table. That scorer reproduces `ll04i00m`'s
in-model scores to 6e-7 (`output/msgym_ll04i00m_inst_tol_eval_2026_09_24/analysis/score_filters.py`).
The tracked script is `scripts/eval/compare_msgym_predictions.py`, tested in
`tests/test_compare_msgym_predictions.py`. Its `score` stage scores our export (optionally
re-filtered at `--min_peak_int`) or GLACIER's PredSpecDB, squared. Its `table` stage writes
`output/glacier_msg_eval_2026_09_24/compare/ours_vs_glacier_msgym_test.csv`, which also carries
`cos_hun` linear and sqrt. Test set n = 9,954; paired differences with 95% CIs from a molecule
bootstrap.

Our predictions come from an m/z-aggregated export. MASSter emits one row per candidate
fragment, and candidates sharing a formula sit at one m/z; 45% of rows were exact duplicates. The
export now sums candidates per peak (4-decimal groups, exact m/z kept: `aggregate_peaks_exact_mz`,
added 2026-09-24). That drops the median from about 2,060 rows to 924 peaks per spectrum and
changes no score. The unaggregated export and the aggregated one score identically on all six
metrics, and the aggregated export reproduces the in-model scores to 4.8e-7. An intermediate
version returned 4-decimal-rounded m/z, which moved 378 spectra across 0.01 Da bin edges.

The 0.1% cutoff is applied to aggregated peaks: median 65 kept. It was first measured per
candidate (median 80 kept, sqrt +0.0023). Per peak, the sqrt gain is +0.0016-0.0018.
Two export runs of the same checkpoint differ on 129 spectra (1.3%, up to 0.015 per spectrum)
from GPU nondeterminism in the model forward pass; means agree to 1e-5.

| metric | ours (all peaks) | ours (0.1% cutoff) | GLACIER | ours − GLACIER (all peaks) | 0.1% cutoff |
| --- | ---: | ---: | ---: | --- | --- |
| cos sim, 0.01 Da | **0.5713** | 0.5712 | 0.5455 | +0.0258 (+0.016, +0.037) | +0.0258 (+0.016, +0.036) |
| cos, inst tol (Hungarian) | **0.5955** | 0.5954 | 0.5648 | +0.0306 (+0.020, +0.041) | +0.0306 (+0.020, +0.041) |
| cos sim, 0.01 Da, sqrt | 0.5878 | **0.5894** | 0.5771 | +0.0107 (+0.003, +0.019) | +0.0123 (+0.005, +0.020) |
| cos, inst tol, sqrt | 0.6151 | **0.6168** | 0.5988 | +0.0163 (+0.008, +0.024) | +0.0181 (+0.010, +0.026) |

All eight 95% CIs exclude zero. The earlier sqrt-space interval (−0.002 to +0.024, below) is a
99.9% CI.

### Metric variants

GLACIER's output is squared to give linear intensity in every row.

| space | ours `ll04i00m` | GLACIER | ours − GLACIER | 99.9% CI (molecule bootstrap, 2,319) |
| --- | ---: | ---: | ---: | --- |
| **linear, our `cos_sim_0.01`** (MassSpecGym's `cos_sim`) | **0.5713** | **0.5455** | **+0.0258** | +0.0085 to +0.0436 |
| sqrt after binning, our `cos_sim_sqrt_0.01` | 0.5878 | 0.5771 | +0.0107 | −0.0023 to +0.0240 |
| ms-pred native (per-peak sqrt, 150,000 bins, raw targets), top 256 peaks each | 0.5717 | 0.5658 | +0.0059 | not computed |
| same, top 1,024 peaks each | 0.5664 | 0.5649 | +0.0015 | not computed |

Ours is higher on 53.6% of spectra (linear). The per-peak sqrt rows depend on peak count because
per-peak sqrt inflates low-intensity peaks. Our export carries about 1,900 candidates per
spectrum; with all of them, ours scores 0.5658.

By instrument and raw CE (eV):

| slice | n | linear ours | linear GLACIER | linear diff | sqrt diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| FT / Orbitrap | 8,374 | 0.5893 | 0.5586 | +0.0306 | +0.0131 |
| QTOF | 1,580 | 0.4760 | 0.4756 | +0.0004 | −0.0016 |
| CE ≤ 15 | 1,012 | 0.5925 | 0.5765 | +0.0160 | −0.0031 |
| 15-25 | 2,402 | 0.6552 | 0.6496 | +0.0057 | −0.0041 |
| 25-35 | 1,608 | 0.5709 | 0.5590 | +0.0119 | −0.0022 |
| 35-50 | 1,409 | 0.5411 | 0.5175 | +0.0236 | +0.0076 |
| > 50 | 3,522 | 0.5200 | 0.4704 | **+0.0496** | **+0.0320** |

In sqrt space GLACIER is level with or slightly ahead of us at CE ≤ 35 and on QTOF. Our lead in
either space is above 35 eV on FT, the high-CE allocation regime of the
[NIST GLACIER-gap finding](../fragnnet/glacier_gap_ce_decomposition_2026_09_21.md).

## Peak over-prediction (added 2026-09-24)

Test set, linear intensity (GLACIER squared), 0.01 Da bins. A predicted bin counts as on target
when the target has a peak in the same bin.

| per spectrum | targets | ours | GLACIER |
| --- | ---: | ---: | ---: |
| peaks / bins, median | 19 | 902 | 160 |
| of those ≥ 1% of base, median | — | 19 | 7 |
| of those ≥ 5% of base, median | 6 | 7 | 4 |
| peaks ≥ 5% of base with no target peak in the bin, median (mean) | — | 3 (7.2) | 1 (2.6) |
| predicted intensity with no target peak in the bin | — | 0.347 | 0.238 |
| … one bin away from a target peak | — | 0.039 | 0.036 |
| … at the precursor when the target has none | — | 0.010 | 0.011 |
| … elsewhere, in peaks ≥ 5% of base | — | 0.177 | 0.144 |
| … elsewhere, in peaks < 5% of base | — | 0.121 | 0.048 |
| target intensity with no predicted peak in the bin | — | **0.146** | **0.251** |

By count, above 1% of base, we predict as many peaks as the target has. GLACIER predicts fewer.
We put about 11 points more of our intensity off target than GLACIER does, and most of the excess
(7 points) is in small peaks. The return is 10 points less missed target intensity.

The spread pays on cosine, so it is hedging rather than a defect. Dropping our predicted peaks
below 1%, 5% or 10% of base changes test `cos_sim_0.01` by −0.0011, −0.0058 and −0.0121.
Sharpening hurts (intensity^1.1 / ^1.25 / ^1.5: −0.0033 / −0.0095 / −0.0204). Slight flattening
helps (^0.9: +0.0016; ^0.75: −0.0023). That agrees with the NIST finding that predictions are
slightly too sharp. Script: `analysis/overprediction.py`, `analysis/offtarget_anatomy.py`.

Export cutoffs, measured on the same test set: a threshold relative to each spectrum's base peak
(`--min_peak_int` on the 0-1000 export scale is that percentage × 10).

| cutoff (per candidate, pre-aggregation export) | candidates kept, median | linear cos (Δ) | sqrt cos (Δ) |
| --- | ---: | ---: | ---: |
| none | ~2,060 | 0.5713 | 0.5878 |
| 0.1% (`--min_peak_int 1`, the default) | 80 | −0.0001 | **+0.0023** |
| 0.5% | 31 | −0.0005 | +0.0015 |
| 1% | 20 | −0.0011 | −0.0003 |

With the aggregated export (the current code), 0.1% keeps a median of 65 of 924 peaks and gives
−0.0001 linear and +0.0016 sqrt (`cos_hun_inst_tol_sqrt` +0.0017).

0.1% is free on the benchmark metric and helps in sqrt space. 1% costs 0.0011 here, ten times
the NIST20 cost. Until 2026-09-24 the export filter was a silent no-op for MASSter, fixed in
`batch_filter_by_peak_intensity` (see [project state](../../project-state.md)).

## Limits

- **Reported versus measured.** GLACIER's sqrt-space score here (0.566-0.577) is close to the
  roughly 0.58 recorded for it on MassSpecGym. That fits ms-pred reporting sqrt-space cosine. Its
  eval also reads subformula-assigned targets with a top-k and intensity-threshold filter
  (`max_inten_spec`), which were not reproduced.
- Each model is trained for its own intensity space: ours linear, GLACIER sqrt. Each space
  favours its own model somewhat. The benchmark's required metric is the linear one.
- n=1 per model; `ll04i00m` is not our best recipe.
- MassSpecGym's Jensen-Shannon similarity was not computed.

## Artifacts

All under `output/glacier_msg_eval_2026_09_24/` (git-ignored):

- **Inputs:** `labels_test.tsv` (the GLACIER input table) and `spec_id_map.tsv`.
- **GLACIER outputs:** `test_preds/` and `test_preds_k1024/`.
- **Our export:** `ours_ll04i00m/`.
- **Scripts in `analysis/`:** `dump_targets.py`, `score.py` (with a `sqrt` option),
  `score_glacier.py`, `score_ours.py`, `score_sqrt_check.py`, `score_mspred_style.py` and
  `score_mspred_native.py`.
- **Per-spectrum scores:** `analysis/*_test_cos.pkl` and `analysis/sqrt_check_scores.pkl`.

## Instrument prediction audit (2026-10-03)

Rechecked the saved exact-m/z aggregated MASSter export (`ll04i00m` epoch 60, no cutoff)
against the original GLACIER sparse-k 256 predictions. These are the simulation predictions
above, not the later retrieval sparse-k 100 run. Both score tables have unique, identical
spec_id sets and exactly matching molecule, instrument and CE metadata. GLACIER peaks are
squared before linear-intensity analysis. No inference, training or preprocessing change ran.

### QTOF is not an established GLACIER win

Fresh paired 95% molecule-bootstrap intervals (5,000 resamples, seed 0):

| Instrument | Metric | MASSter | GLACIER | Difference, 95% CI |
| --- | --- | ---: | ---: | --- |
| Orbitrap | linear 0.01 Da bins | 0.58925 | 0.55864 | +0.03061 [+0.02099, +0.04013] |
| Orbitrap | sqrt 0.01 Da bins | 0.61137 | 0.59832 | +0.01305 [+0.00627, +0.01955] |
| Orbitrap | linear instrument-tolerance Hungarian | 0.60385 | 0.57059 | +0.03326 [+0.02368, +0.04284] |
| Orbitrap | sqrt instrument-tolerance Hungarian | 0.62715 | 0.61100 | +0.01615 [+0.00927, +0.02273] |
| QTOF | linear 0.01 Da bins | 0.47598 | 0.47557 | +0.00041 [−0.02779, +0.03131] |
| QTOF | sqrt 0.01 Da bins | 0.46292 | 0.46452 | −0.00159 [−0.02389, +0.02426] |
| QTOF | linear instrument-tolerance Hungarian | 0.55101 | 0.53439 | +0.01662 [−0.01288, +0.04765] |
| QTOF | sqrt instrument-tolerance Hungarian | 0.55135 | 0.53394 | +0.01740 [−0.00593, +0.04313] |

Orbitrap has 8,374 spectra from 2,231 molecules; QTOF has 1,580 from only 195 molecules.
Every QTOF interval includes zero. QTOF matching uses 20 ppm with a 750 Da floor (at least
0.015 Da); Orbitrap uses 10 ppm with a 200 Da floor. Thus QTOF's 0.01 Da binned score is
particularly sensitive to mass offsets and bin boundaries. Switching to instrument-tolerance
matching lifts linear scores by 0.07503 for MASSter and 0.05882 for GLACIER and reverses the
small sqrt ordering. This establishes metric sensitivity, not which individual mass offsets
come from acquisition, rounding or formula assignment. Binned cosine remains the benchmark.

### What the peak lists show

Independent diagnostics on saved target peaks and both actual prediction HDF5 files:

| Instrument | Model | Target intensity covered | Target base peak supported | Predicted base peak matches target base peak | Predicted intensity off target | Median peaks ≥1% of predicted base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Orbitrap | MASSter | 0.89681 | 0.95080 | 0.42441 | 0.29845 | 17 |
| Orbitrap | GLACIER | 0.79340 | 0.86745 | 0.41557 | 0.19117 | 7 |
| QTOF | MASSter | 0.88718 | 0.95000 | 0.38038 | 0.40888 | 27 |
| QTOF | GLACIER | 0.73872 | 0.86962 | 0.40570 | 0.29392 | 8 |

Support means **any positive predicted peak** inside the target's instrument window; it is not
a one-to-one assignment, a useful-intensity guarantee or a cosine oracle. Predicted base peaks
are determined after exact-m/z duplicate aggregation. Off-target intensity is the L1 fraction
with no target within that window. These CPU diagnostic values do not replace the validated
CUDA score tables; they do not use 0.01 Da bins. Median target peak counts are 20 (Orbitrap) and
19 (QTOF).

The evidence points to an allocation tradeoff: MASSter covers more target intensity on both
instruments, but spreads more predicted intensity onto absent peaks. On QTOF it also ranks the
observed base peak first less often, despite supporting it more often. That supports a QTOF
intensity-allocation weakness, rather than deficient candidate coverage. It does not prove an
architectural cause or that GLACIER is better overall on QTOF.

### Dataset composition and limits

QTOF has 21.2% of rows above stored CE 50, against Orbitrap's 38.1%. MASSter's linear advantage
in that band is +0.05153 on QTOF and +0.04940 on Orbitrap. Below or at 50, the four QTOF bin
means favour GLACIER (−0.00937, −0.01663, −0.02623, −0.00885), while Orbitrap's favour MASSter.
Reweighting both instruments to the whole test set's five stored-CE-bin proportions changes
QTOF's linear difference to +0.00778 and sqrt to +0.00952; Orbitrap stays +0.03082 and +0.01299.
This descriptive control is not matched chemistry or calibrated energy. Stored CE has unresolved
unit/provenance issues; see the [metadata audit](masster_metadata_usage_audit_2026_10_03.md#7-msgym-config-check-2026-10-03).

Only 107 molecules occur on both instruments. Restricting to those molecules gives linear
+0.01757 on Orbitrap (1,488 spectra) and −0.00099 on QTOF (1,010 spectra). This restriction
still does not match energy, source or numbers of spectra per molecule. Chemistry composition
alone therefore does not obviously remove the pattern, but instrument causality is unproven.

Artifacts: `output/msgym_instrument_audit_2026_10_03/audit.py`, `metrics.csv`,
`paired_scores.csv`, `prediction_diagnostics.pkl`, and `audit.log`. Reproduce from the repository
root using the configured FRAGNNET-GPU interpreter. One checkpoint per model; no training-seed
uncertainty estimated. The next causal check is matched molecule/energy/source strata and
QTOF intensity-allocation errors, not an instrument-specific architecture claim.
