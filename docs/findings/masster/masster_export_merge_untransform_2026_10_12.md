# MASSter annotated export untransformed each pathway before summing

> **Status:** FINDING · **Updated:** 2026-10-12
> Bug confirmed and fixed in the working tree (not committed); real-checkpoint size unmeasured.

With an intensity transform (`ints_transform: sqrt` or `log10`) and merged rows (nominal mass, or
`masster_merge_same_mz_rows: true`), MASSter's tabular and annotation exports reported
Σᵢ f⁻¹(sᵢ) per peak instead of f⁻¹(Σᵢ sᵢ). The training loss and every metric use the merged
form, which is exactly what the guard in `MassterPL` (transform + exact mass requires the merge)
exists to enforce. Exports made through the annotation path with such a model are wrong.

## Cause

`MassterModel.forward` merged same-m/z rows only when `return_annotation` was false. The tabular
exporter requests annotation (`inference_batch_options` → `return_explainability_tensors`), so
it received one row per pathway. `SpectrumPL.preproc_spec(pred=True)` then untransformed each
row, L1-normalized, and `MassterPL.postprocess_export_peaks` summed the rows per peak.

Not affected: `predict_step` without annotation, the prediction-only pickle export
(`run_inference_pkl`), any `ints_transform: none` model (f⁻¹ is linear), and exact-mass models
without the merge (one row per pathway is then the training semantics).

## Measurement

Seeded random-weight MASSter, 3 molecules, 54 pathways on 48 peaks. Annotated export against
the merged `inference_step`, both L1-normalized:

| Transform | Output mode | Max abs diff | Max rel diff |
| --- | --- | --- | --- |
| sqrt | probability | 0.054 | 0.54 |
| sqrt | independent | 0.044 | 0.58 |
| log10 | probability | 0.39 | 8.8 |
| log10 | independent | 0.20 | 9.1 |
| none | both | 1.5e-8 | 1.7e-7 |

These are untrained-model numbers; the error on a trained checkpoint depends on how often
several pathways share a peak and was not measured.

## Fix (working tree, 2026-10-12)

- The merge now also runs under `return_annotation`. `pred_mzs`/`pred_logprobs`/`pred_ints` are
  the merged spectrum in every mode. With annotation and merged rows, the per-pathway rows are
  returned as `pred_pathway_mzs`, `pred_pathway_batch_idxs` and `pred_pathway_scores` (linear
  pre-merge score), aligned with the other `pred_pathway_*` tensors.
- `postprocess_export_peaks` keeps the merged intensities and gives each pathway its share of its
  peak, sᵢ / Σ s × merged intensity, so formula, node and joint annotation still sum to the
  exported spectrum. Under the merge the export sums regardless of `sum_ints`, as the model's
  scores do. Unmerged models keep the old path.
- Contract change: code that read `pred_mzs` etc. aligned with `pred_pathway_*` under annotation
  must read `pred_pathway_*`. `analyze_masster_ei_checkpoint.py` and six memorization debug
  scripts were updated; `tests/test_masster_nominal_isotopes.py` and
  `tests/test_masster_merge_same_mz_rows.py` were updated to the new keys.
- Regression test: `test_annotation_export_peaks_equal_the_merged_spectrum` in
  `tests/test_masster_pl.py` ({sqrt, log10} × {probability, independent}); it fails on the old
  code in all four cases. The CLI export test compares against the merged `predict_step`.

## Follow-ups

- Any analysis built on annotated MASSter exports from sqrt/log10 models should be regenerated.
  No existing exports were regenerated.
- The annotated export reports each peak's exact m/z (intensity-weighted over pathways); the
  plain merged output reports the 4-decimal key. The difference is below 5e-5 Da and predates
  the fix.
- The edited debug scripts were syntax-checked, not run (they need checkpoints and data).
