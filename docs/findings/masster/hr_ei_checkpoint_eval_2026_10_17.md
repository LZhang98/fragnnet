# Frozen MASSter EI checkpoint on measured HR-EI spectra

> **Status:** FINDING · **Updated:** 2026-10-17
> Exact monoisotopic outputs already agree closely with HREI-MSDB; agreement is weaker and source-dependent on RECETOX. No fine-tuning or training-overlap exclusions.

## Question and setup

Can the current nominal-trained EI model predict accurate-mass spectra reasonably
well without retraining? The user explicitly requested inclusion regardless of
nominal-training overlap. We evaluated the mapping-consistent pool from the
[raw audit](hr_ei_raw_audit_2026_10_17.md): 329 HREI-MSDB, 383 RECETOX Exposome,
and 250 RECETOX Metabolome spectra. Castro remains excluded because its JCAMP
records have no resolved structures. Isotope-labelled and inconsistent mappings
remain excluded. Counts are spectrum records, not independent unique molecules;
cross-library overlap and stereoisomer/repeat records are retained.

The model is e1, W&B `jeljdikj`, epoch 37:
`saved_ckpts/e1_nominal_natural_isotopes_jeljdikj/ckpt/model-epoch=037.ckpt`, SHA-256
`b281a13a892fa5e3e4ed6a268c06284b1166a79dc60893e3546713d205b53831`.
All trained weights loaded strictly. Saved hyperparameters override current template
defaults; missing newer settings are supplied by the template. Retired `OB`/`OB_F`
keys in two metadata dictionaries are renamed in memory before migration.
Inference uses CPU, four torch threads, one joblib worker, seed 20261017,
batch size four, and the saved 256-fragment target/4,096-mask cap. The checkpoint
file and raw data are unchanged. Unrelated UMAP edits in the working tree were left alone.

Two inference settings change: mass mode becomes `exact` and isotope envelope becomes
`none`. The existing natural-envelope implementation requires nominal mass mode.
Consequently this is a test of exact **monoisotopic channel predictions**, not a
complete high-resolution isotope-envelope simulation. Reference isotope peaks are
retained and can lower similarity. Linear intensity transform (`none`) is retained.
No targets are used for model inference; raw measured intensities are scored externally.

## Results

Mean linear-intensity cosine with maximum-product one-to-one peak assignment:

| Reference | Spectra | 5 ppm | 10 ppm | 20 ppm | Rounded-mass control |
| --- | ---: | ---: | ---: | ---: | ---: |
| HREI-MSDB | 329 | 0.8390 | 0.8391 | 0.8392 | 0.8491 |
| RECETOX Exposome | 383 | 0.6114 | 0.6549 | 0.6592 | 0.6902 |
| RECETOX Metabolome | 250 | 0.6009 | 0.6009 | 0.6025 | 0.6517 |

The 5/10/20 ppm settings have absolute floors of 0.0005/0.001/0.002 Da respectively.
Matched contributions at the same mass are first aggregated using a four-decimal
mass key; the intensity-weighted mass is retained for tolerance matching. Cosine
norms include unmatched predicted and reference peaks. All positive predictions are
retained, without a relative-intensity cutoff. HREI-MSDB predictions are restricted
to its declared acquisition window, 40–500 m/z. Other predictions use 0–1500 m/z;
their acquisition cutoffs were not inferred from individual target peak lists.

The rounded-mass control rounds these same monoisotopic predictions and measured
masses to integers and sums contributions. It is not a separate forward pass in
the original nominal/natural-isotope configuration. Numerical comparisons admit
1e-10 Da roundoff so aggregated integer masses match reliably.

At 5 ppm, median cosine is 0.8662, 0.6997, and 0.6263 respectively. About 69.3%,
36.6%, and 28.4% of spectra reach cosine >=0.8. Mean measured intensity with any
predicted supporting mass is 87.65%, 61.12%, and 57.94%; this support metric does
not assess whether predicted intensities are correct.

HREI-MSDB shows little degradation from the rounded control at 5 ppm. Exposome
improves by 0.0435 from the first to second tolerance setting, indicating sensitivity
to mass matching; both the ppm threshold and absolute floor change. This does not
isolate calibration, chemistry, or instrument effects. The remaining source gaps
can involve intensity allocation, missing isotope envelopes, different acquisition
windows, derivative mapping, and domain differences. This experiment does not
separate those explanations or establish unseen-molecule generalization.

## Outputs and reproduction

Canonical scored results are in `output/hr_ei_checkpoint_eval_2026_10_17/scored/`:

- `scores.csv`: all 962 per-spectrum scores, names, and local training-overlap flags.
- `metric_summary.csv`: source counts, means, medians, and intensity recall.
- `hr_vs_nominal.png`: paired scores and tolerance sensitivity.
- `median_examples.png` and `plotted_examples.csv`: one median-scoring example per source.
- `cache/exact_predictions.npz`: exact-mass prediction arrays.
- `selected_records.csv`, `resolved_config.json`, `provenance.json`: identities and settings.

The frozen-weight forward pass took approximately 149 seconds on CPU. Original
inference inputs and its uncorrected nominal-control scores remain under
`output/hr_ei_checkpoint_eval_2026_10_17/monoisotopic/`; use `scored/` for conclusions.
The original control used zero tolerance on intensity-weighted integer masses and
missed some HREI-MSDB peaks because of floating-point roundoff. Rescoring corrected
that issue; HR scores were unchanged. A regression test covers this failure.

`output/hr_ei_checkpoint_eval_2026_10_17/run_record.json` records the forward command,
Git revision, dirty-diff identity, and script/template hashes. Primary MSP hashes
were checked against the audit provenance. Regenerate into fresh output directories:

```bash
FRAGNNET_JOBLIB_N_JOBS=1 MPLCONFIGDIR=/tmp/fragnnet_mpl \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m scripts.debug.ei.evaluate_hr_ei_checkpoint \
  --output output/hr_ei_checkpoint_eval_2026_10_17/rerun
MPLCONFIGDIR=/tmp/fragnnet_mpl FRAGNNET_JOBLIB_N_JOBS=1 \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m scripts.debug.ei.plot_hr_ei_checkpoint_eval \
  output/hr_ei_checkpoint_eval_2026_10_17/rerun
```

The scorer also accepts `--rescore-from` to evaluate saved predictions into a fresh
directory. Source scripts are
[evaluate_hr_ei_checkpoint.py](../../../scripts/debug/ei/evaluate_hr_ei_checkpoint.py)
and [plot_hr_ei_checkpoint_eval.py](../../../scripts/debug/ei/plot_hr_ei_checkpoint_eval.py).

## Validation and next check

Five focused scoring tests pass: aggregation, nominal-vs-exact discrimination,
one-to-one assignment with unmatched normalization, ppm/floor thresholds, and
integer-mass roundoff. Ruff checks pass. All 962 selected spectra survive dataset
preparation and receive scores; plots were visually inspected.

## Follow-up: natural exact-mass isotope output

The [exact isotope/exclusion implementation](../../plans/masster/hr_ei_isotopes_holdout_plan.md)
now supports natural exact-mass fine structure. The same frozen e1 weights were
evaluated on all 962 reference records with `--isotope-envelope natural`; the
other inference/scoring settings remain as above. This is still a contaminated
implementation baseline, not a held-out result for the new excluded split.

| Reference | 5 ppm, monoisotopic | 5 ppm, natural exact | 10 ppm, natural exact | 20 ppm, natural exact |
| --- | ---: | ---: | ---: | ---: |
| HREI-MSDB | 0.8390 | 0.8617 | 0.8618 | 0.8619 |
| RECETOX Exposome | 0.6114 | 0.6804 | 0.7549 | 0.7615 |
| RECETOX Metabolome | 0.6009 | 0.5971 | 0.5973 | 0.5986 |

Natural isotope output improves mean 5 ppm cosine by 0.0227 on HREI-MSDB and
0.0690 on Exposome, but decreases it by 0.0037 on Metabolome. It therefore does
not establish a uniform improvement across sources. Isotope fine structure is
emitted before any instrument-specific peak broadening or unresolved-peak centroiding.

Outputs, score summaries, prediction arrays, and plots are under
`output/hr_ei_checkpoint_eval_2026_10_17/natural_exact/`, using the same filenames
as `scored/`. Its `provenance.json` records the natural-envelope setting and
approximately 392-second CPU evaluation runtime. `implementation_record.json` adds the command,
Git/dirty-diff identity, and hashes of the implementation files and template.
All 962 spectra completed successfully with strict checkpoint loading. The
original monoisotopic results remain unchanged.

Next: train e5 from scratch on the HR-EI-excluded split, then repeat this comparison
with that fresh model. Nominal isotope offsets must not simply be added as integer
shifts to exact masses.
