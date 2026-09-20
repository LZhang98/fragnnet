# Inference step consolidation

> **Status:** LANDED · **Updated:** 2026-09-15 · Implemented and tested in the working tree, uncommitted. No model evaluation run; see Validation.

## Objective

Collapse the three inference entry points on `SpectrumPL` into one, so that every
consumer receives peak intensities in a single, fixed space. This deletes
`resolve_pred_ints` and its `step_untransformed` flag rather than documenting them
better.

Scope is the inference/export path only. Training and loss paths keep `pred_logprobs`
and are not touched.

## The current state

Three entry points feed the exporters:

| Entry point | Forward | Untransform + L1 | Ground-truth preproc + metrics |
| --- | --- | --- | --- |
| [`predict_step`](../../src/fragnnet/pl_model/spectrum_pl.py) (line 1246) | yes | no | no |
| [`inference_step`](../../src/fragnnet/pl_model/spectrum_pl.py) (line 1659) | yes | yes | no |
| [`inference_with_ground_truth_step`](../../src/fragnnet/pl_model/spectrum_pl.py) (line 1552) | yes | yes | yes |

The prediction half of `inference_with_ground_truth_step` is the same computation as
`inference_step`: both call `preproc_spec(train=False, pred=True, log_in=True)`, which
applies `ints_untransform_func` and then L1-normalizes (`spectrum_pl.py` lines
1205-1209). They differ only in how the result is stored — `inference_step` returns it
linear as `pred_ints`, the ground-truth step returns it logged as `pred_logprobs`.

The only genuine axis of variation is whether ground truth is also preprocessed and
scored.

### Consequences

**`pred_logprobs` names two different spaces.** `forward` emits transformed-space model
log-probs under that key; `inference_with_ground_truth_step` writes final-scale
log-probs into the same key. Consumers cannot tell them apart from the dict.

**The disambiguator is supplied by the consumer.** `resolve_pred_ints`
([`script_utils.py`](../../src/fragnnet/utils/script_utils.py) line 573) takes a
`step_untransformed` boolean. The two call sites derive it differently — hardcoded
`True` at line 1065, `=evaluate` at line 1668 — because they branch to different steps:

| Consumer | `evaluate=True` | `evaluate=False` |
| --- | --- | --- |
| `run_inference_pkl` (lines 1037/1040) | `inference_with_ground_truth_step` | `inference_step` |
| `process_batch_to_tabular_row` (lines 1644/1646) | `inference_with_ground_truth_step` | `predict_step` |

So `evaluate` predicts the intensity space in one function and not in the other. Every
new consumer has to re-derive the flag by reading both branches.

**A wrong flag is undetectable at the output.** `ints_untransform_func`
([`spec_utils.py`](../../src/fragnnet/utils/spec_utils.py) lines 183-189) is
`rescale_to_max` composed with `inverse_transform`. Applying it twice re-establishes
`max == 1000`, so the result passes any output-side check; only the relative
intensities are wrong — squared, under `ints_transform: sqrt`.

## Measurement: the two `evaluate=False` routes already agree

The `predict_step` branch is not a third intensity mode. For a model simplex
`p = [0.5, 0.3, 0.2 | 0.7, 0.3]` over two spectra under `ints_transform: sqrt`:

| Route | Result |
| --- | --- |
| `predict_step` then `ints_untransform_func` | `[999.9995, 359.9999, 159.9999, 999.9996, 183.6734]` |
| `inference_step` then `batched_mf1000_normalize` | `[1000.0000, 360.0001, 160.0000, 1000.0000, 183.6735]` |

Maximum absolute difference `4.9e-04`. `ints_untransform_func` already lands the base
peak on exactly 1000, and the intervening L1 normalization is undone by the subsequent
`mf1000` rescale. The two exporters have been computing the same intensities by two
different routes.

### Why the duplication exists

`git log` shows independent origin, not a deliberate split: `inference_step` came out of
refactoring commit `d560656d` ("break super long code to classes"), and
`inference_with_ground_truth_step` was added later in `10ad2a9c` ("update prediction
pipeline"). Neither was reconciled against the other.

## Proposal

One inference entry point, with ground-truth scoring as a keyword:

```python
def inference_step(self, batch, *, metrics: str | None = None):
    """Forward, untransform, L1-normalize. If `metrics` is a split name,
    also preprocess ground truth and score against it."""
```

`metrics=None` reproduces today's `inference_step`; `metrics="test"` reproduces today's
`inference_with_ground_truth_step`. Everything ground-truth-dependent sits behind that
one condition: the `true_*` preproc, the `metric_fn` call, and the auxiliary-metric
explainability demand currently ORed on at `spectrum_pl.py` lines 1582-1584.

`predict_step` remains as Lightning's prediction hook (`return self.forward(...)`) but
leaves the export path.

**Postcondition, unconditionally:** `pred_ints`, linear, L1-normalized per spectrum,
alongside `pred_mzs` and `pred_batch_idxs`. Ground-truth keys and metric values appear
only when `metrics` is set, as does the `pred_logprobs` compatibility key described
below. `pred_ints` is the sole intensity contract regardless of mode.

### What this removes

- `resolve_pred_ints` and `step_untransformed` are deleted. Exporters call
  `batched_mf1000_normalize` once, unconditionally; there is no branch to get wrong.
- `pred_logprobs` reverts to one meaning — `forward`'s transformed-space output.
- `evaluate` stops selecting a step and becomes `metrics="test" if evaluate else None`.

### OOS keys

`inference_step` returns `pred_oos_logprobs`; the ground-truth step returns
`oos_prob = exp(...)`. These are a model output and a metric, not two spellings of one
value: `oos_prob` is a declared metric name
([`fragnnet_pl.py`](../../src/fragnnet/pl_model/fragnnet_pl.py) line 337). The merged
step keeps both — `pred_oos_logprobs` always, `oos_prob` under `metrics` — which is what
the two steps already do. Nothing to decide.

### Decided: keep `pred_logprobs` in metrics mode

The single-postcondition proposal replaces the ground-truth step's `pred_logprobs`
(final-scale, logged) with `pred_ints` (final-scale, linear). That is not a purely
internal rename:
[`calculate_ensemble_agreement_stats.py`](../../scripts/stats_scripts/calculate_ensemble_agreement_stats.py)
reads `pred_logprobs`, `true_logprobs`, `oos_prob`, and `true_oos_prob` (lines 24-30)
out of `vals.pkl` files written by `run_inference_pkl`, and ensembles them **in log
space** (`log_mean` at lines 64 and 70, feeding
`sparse_cosine_distance_hungarian` at line 362). When `output_set is None`,
`run_inference_pkl` passes the step's dict through unfiltered (`script_utils.py` line
1073), so the model's raw key names are the saved pickle format.

**Decision (2026-09-15): existing `vals.pkl` artifacts must stay readable.** The merged
step emits `pred_ints` as the canonical intensity *and* additionally emits
`pred_logprobs = log(pred_ints)` when `metrics` is set, purely for pickle
compatibility. `calculate_ensemble_agreement_stats.py` is not modified and old pickles
keep loading unchanged.

The consequence to accept knowingly: the `pred_logprobs` overload survives in the saved
pickle format, where the same key still means transformed space in an
`evaluate=False` export and final space in an `evaluate=True` one. It no longer exists
in any live code path, because no consumer reads `pred_logprobs` to obtain intensities
any more — `pred_ints` is the only intensity contract. Removing the compatibility key is
a follow-up gated on regenerating or shimming the ensembling artifacts.

## Blast radius

- **Producers:** two merge into one, in
  [`spectrum_pl.py`](../../src/fragnnet/pl_model/spectrum_pl.py).
- **Consumers:** two call sites in
  [`script_utils.py`](../../src/fragnnet/utils/script_utils.py) simplify; both lose an
  argument. `run_inference_tabular`, `run_inference_hdf5`, and
  `run_inference_tabular_iter` all reach the model through
  `process_batch_to_tabular_row` and need no change beyond that.
- **No CLI change.** `pred_logprobs` is not in `OUTPUT_SUBSET_OPTIONS`
  (`script_utils.py` lines 544-556). The user-facing `pred_ints` and `pred_oos_prob`
  tokens are unaffected. Note that the token vocabulary is *not* the whole export
  surface: with `output_set is None` the pickle writer saves the step's raw dict, so the
  key names are also a persisted file format — see the open decision above.
- **No metric change.** `pred_logprobs` is not in `extra_metric_args`, which holds only
  index arrays (`spectrum_pl.py` lines 503-516).
- **Other callers:** [`debug_identity_shortcut_eval.py`](../../scripts/debug/debug_identity_shortcut_eval.py)
  line 141 and [`node_vs_h_transfer_diagnostic.py`](../../scripts/eval/node_vs_h_transfer_diagnostic.py)
  line 390 call the ground-truth step directly, but read only metric scores and
  marginals — never intensities. They need the keyword rename and nothing else.

## Completed work

`SpectrumPL.inference_with_ground_truth_step` and `SpectrumPL.inference_step` are now one
method, `inference_step(batch, *, metrics=None)`
([`spectrum_pl.py`](../../src/fragnnet/pl_model/spectrum_pl.py) line 1552).
`predict_step` is unchanged and no longer on the export path.

`resolve_pred_ints` and `step_untransformed` are deleted (58 lines). Both exporters now
call `batched_mf1000_normalize` unconditionally:
[`script_utils.py`](../../src/fragnnet/utils/script_utils.py) in `run_inference_pkl` and
`process_batch_to_tabular_row`, each of which passes `metrics="test" if evaluate else
None` and no longer branches on the step.

[`debug_identity_shortcut_eval.py`](../../scripts/debug/debug_identity_shortcut_eval.py)
and [`node_vs_h_transfer_diagnostic.py`](../../scripts/eval/node_vs_h_transfer_diagnostic.py)
took the keyword rename only.

The metrics branch is a faithful move of the old body: `eval_pred_logprobs` is still
`safelog` of the same tensor (`eval_pred_probs` renamed to `eval_pred_ints`), so the
compatibility key is bit-identical to what the old step wrote, not merely close. Two keys
are *added* to the metrics-mode dict, `pred_ints` and `pred_oos_logprobs`, which widens
`vals.pkl` by two columns and removes nothing.

## Validation

Run with `FRAGNNET_PYTHON=/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`.

- **Full suite:** `pytest tests -q` - 2284 passed, 1 skipped, 2 xfailed.
- **Ruff:** no new findings on any changed file; `spectrum_pl.py` + `script_utils.py` went
  from 25 to 23 findings (one duplicated `warnings.warn` and one `C405` disappeared with
  the merge). Changed test files are clean.
- **Contract tests** ([`test_inference_intensity_contract.py`](../../tests/test_inference_intensity_contract.py),
  rewritten): both `evaluate` modes export identical intensities; `evaluate` maps to
  `metrics`; `min_peak_int` sees the 0-1000 scale; the exporter rescale is idempotent; and
  the measured route equivalence above is now pinned against the real `preproc_spec`.
- **Prediction-only isolation:** a batch subclass that raises on `spec_mzs`/`spec_ints`/
  `spec_batch_idxs`/`spec_prec_mz` proves `metrics=None` never reads ground truth.
- **Untransform is not re-applied:** every exporter stub now installs an
  `ints_untransform_func` that raises, so a regression would fail rather than silently
  square.
- **Explainability demand** ([`test_explainability_demand.py`](../../tests/test_explainability_demand.py)):
  14 passed, including a new case that `metrics=None` adds no demand.

### Not run

No validation-set model evaluation. The change is a refactor with no intended numerical
effect, and the exporter equivalence is pinned by unit tests, but a spectra-level A/B on a
real checkpoint has not been done.

### Threshold units in tests

`test_inference_annotation_without_probs.py` thresholds moved from `0.25` to `500.0`. The
old stub installed an identity `ints_untransform_func`, so its `evaluate=False` path
filtered against raw simplex values; the exporter now always emits base-peak-1000, which
is the scale `--min_intensity` is documented against. The peaks kept and dropped are
unchanged.

## Follow-on cleanup (2026-09-15)

A review of the working tree found four places where the diff above had patched a symptom.
All four are now fixed, with no change to any output artifact:

**`COLUMN_SPECS` is the export schema.** The column order, the HDF5 ragged-family grouping,
the row builder's positional list and `OUTPUT_SUBSET_OPTIONS` were four independent
spellings of one table, coupled only by hand-matched ordering. They are now all read off
`COLUMN_SPECS` in [`script_utils.py`](../../src/fragnnet/utils/script_utils.py).
`get_tabular_columns` and `process_batch_to_tabular_row` both call
`selected_column_specs`, so a row cannot drift out of step with its header.
`get_tabular_columns`'s dead commented-out subset filter is gone; the documented
NaN-plus-`--drop_none_cols` contract is unchanged and now pinned by a test.

**`PredArrays` replaces the 20-positional / 16-tuple filter signature.**
`batch_filter_by_peak_intensity` takes and returns one dataclass grouped by axis
(peak/formula/joint/node). All three bugs fixed in this diff were index-ordering bugs,
which is the class the positional signature invited. The batch-index arrays are still
passed raw, exactly as the tuple call did.

**`expl_metric_args` is a plain attribute** set at the end of `_setup_metric_fns`,
replacing a `self.__dict__` cache, a function-local import and a `getattr` fallback. There
is no circular import between `fragnnet_model` and `pl_model` (verified), `extra_metric_args`
is always set before it, and the value is a frozenset intersection against a ≤4-element list.
`TestExplMetricArgs` now builds real `FraGNNetPL` models instead of stubs.

**Dead twin removed.** `filter_by_peak_intensity` (187 lines, zero callers) still carried
the vacuous node mask, the missing P(n) pruning and the rebase-skipping early return that
the batch version lost in this diff.

Smaller: the resume block in [`run_inference.py`](../../scripts/run_inference.py) was
indented inside its per-chunk loop, re-filtering `spec_df`/`mol_df` once per file and
reporting a "skipped" count from only the last one; `keep_mask_j` is now guarded explicitly
rather than relying on `mask[None]` adding an axis instead of raising;
`run_inference_tabular` asserts its row width like `run_inference_hdf5` already did; and a
docstring example still showed `to_parquet` after parquet was dropped.

### Validation

- **Full suite:** `pytest tests -q` - 2310 passed, 1 skipped, 2 xfailed (2284 before; net
  +26 = 27 new schema tests, 4 stub-based tests replaced by 3 real-model ones, 188 lines of
  dead code removed).
- **Schema equivalence:** the table reproduces the previous column order, annotation
  ordering, token set and ragged-family membership exactly - checked directly, and pinned
  by [`test_export_schema.py`](../../tests/test_export_schema.py).
- **Ruff:** 26 findings on the three main files, down from 27 at HEAD; no new findings, and
  the new test files are clean.
- **Not run:** no validation-set model evaluation. As with the refactor above, these
  changes have no intended numerical effect and the exporter contract is pinned by unit
  tests, but no spectra-level A/B on a real checkpoint has been done.

## End-to-end smoke test (2026-09-16)

The refactor above was pinned by unit tests but had never been run end to end against a
real checkpoint. A debug model trained on `debug_tinyv4`
([`config/debug/debug_tiny_inference_smoke.yml`](../../config/debug/debug_tiny_inference_smoke.yml),
128 molecules, 30 epochs, `mol_hidden_size: 64`, best `val_cos_sim_0.01` 0.271 at epoch 0)
was used to predict 4 molecules x 6 conditions through the documented three-step path. It
found two defects that no unit test covered.

**Prediction-only runs were dropping every spectrum.** `spec_params.drop_empty_spectra` is
True in [`config/models_template`](../../config/models_template), and
`BaseDataset._setup_dfs` runs the spectrum cleanup group whenever the config carries it. An
inference `spec_df` has `peaks == []` by construction, so cleanup removed all 24 spectra:
the dataset came back empty and `run_inference.py` printed `Total predictions: 0` and
exited 0. `apply_inference_config` now takes `drop_empty_spectra`, which
[`run_inference.py`](../../scripts/run_inference.py) passes as False. It is opt-in, not
forced, because the same helper configures the evaluation scripts, which score against real
spectra and must keep their cleanup behavior; and it is only applied when the checkpoint
config already carries the cleanup group, since writing the key into a legacy config that
omits it would trip that same missing-cleanup-params check. `run_inference.py` now also
raises when the dataset is empty although spectra reached it - the legitimate empty cases
(empty shard, fully resumed) already exit earlier with their own message.

**Saved predictions held CUDA tensors.** `pred_mzs`, `pred_ints` and `pred_node_ccs` were
explicitly moved to the CPU in `process_batch_to_tabular_row`; the index and probability
arrays were not, so a GPU run pickled a mix and the artifact could not be unpickled on a
host without that CUDA device. Every exported tensor is now moved at the single point where
the row is assembled.

### Validation

- **Full suite:** `pytest tests -q` - 2314 passed, 1 skipped, 2 xfailed.
- **New tests:**
  [`test_inference_prediction_only_config.py`](../../tests/test_inference_prediction_only_config.py)
  (4 tests: the drop that motivated the flag, the opt-out, evaluation left alone, legacy
  config untouched) and `TestRowLeavesTheDevice` in
  [`test_export_schema.py`](../../tests/test_export_schema.py), which fails without the
  `.cpu()` move (verified by reverting it).
- **End to end:** 24/24 spectra predicted. Fixed batching (`--batch_size 8`) and dynamic
  batching (`--dynamic_batch_max 20000`) agree exactly on every column; all exported
  tensors are CPU; `--min_intensity 10.0` holds (every intensity > 10, base peak 1000); the
  annotation indices resolve in range with 4.17 candidate nodes per formula; no peak sits
  above the precursor m/z; chunked writes and the resume path both behave.
- **Not run:** still no validation-set evaluation on a production checkpoint. The debug
  model is a smoke-test artifact, not a scientific result.

## Dead flag and the pkl exporter (2026-09-16)

**`--save_interval` removed.** It was declared and documented in
[`run_inference.py`](../../scripts/run_inference.py) and read nowhere - its
`add_argument` call was its only reference in the repository. Only `--save_chunk_size`
ever triggered a flush.

**`run_inference_pkl` is kept.** It is the only exporter that returns predictions,
ground truth and per-spectrum scores together: `true_mzs` / `true_logprobs` /
`true_batch_idxs` / `true_prec_mzs` and `cos_hun_inst_tol` / `cos_sim_0.01` alongside the
`pred_*` arrays, as one dict of flat arrays concatenated over the whole run with globally
cumulative indices. The tabular and HDF5 exporters are prediction-side and per-spectrum,
so folding it into them would lose that. Two defects in its caller were fixed instead.

**The pkl branch was writing under the DataFrame's filename.** `--output_format pkl`
wrote `predictions_<split>.pkl`, but every consumer of that name - `computed_difference.py`,
`qtof_20ppm_reeval.py`, both `plot_annotated_spectra_*.py` and ~6 notebooks - expects the
per-spectrum DataFrame the older `run_inference_tabular` path produced, and indexes it as
`df[df["spec_id"].isin(...)]` / `.itertuples()`. The dict is a different artifact, so it now
writes `predictions_<split>_raw.pkl`. It is deliberately **not** re-keyed to `spec_id`:
`spec_unique_id` is `spec_entry[self.id_key]`, which is `group_id` under
`spec_params.merge` ([`base_dataset.py`](../../src/fragnnet/dataset/base_dataset.py) line
449), so calling it `spec_id` would repeat the merged-mode confusion that already makes
resume wrong.

**The reported count was peak-level.** `num_predict` took
`len(next(iter(results.values())))`, and the first value is a peak array - it printed
`Total predictions: 350761` for a 378-spectrum split. It now takes the max over the
`*_batch_idxs` arrays, which range over spectra whatever subset was requested.

### Validation

- **Full suite:** `pytest tests -q` - 2314 passed, 1 skipped, 2 xfailed. Ruff clean on both
  changed scripts.
- **Run against the debug model:** `--output_format pkl` on the `debug_tinyv4` test split
  reports 378 for the full split and 24 under `--batch_cutoff 3`, both matching the
  `num_datapoints` the exporter already printed, and writes `predictions_test_raw.pkl` in
  each case. `run_inference.py --help` no longer advertises `--save_interval`.
- **Not fixed here:** `run_inference_pkl` still calls `len(dataloader)`, so it cannot take a
  streaming loader; and resume in `run_inference.py` still filters on `spec_id` against a
  column holding `id_key`, so it stays wrong in merged-spectra mode.

## Export-mode sweep (2026-09-16)

Five modes checked against the debug checkpoint: HDF5 output, evaluation mode,
`--output_everything`, peak-only output, and the `min_intensity` post-process. Both
exporter scripts were covered.

**HDF5 output mode is sound.** `run_save_inference_eval.py --output_format hdf5` on the
378-spectrum test split writes four ragged families (peak 23,241 / formula 23,241 /
joint 136,254 / node 55,688) plus per-spectrum scalars. Checked directly: every family's
start/end offsets are contiguous, start at 0 and end exactly at its array length; no
per-spectrum index is out of range for its own slice across all 378 spectra; annotation
resolves (base peak of spectrum 22462 to `C8H12N2OS` over 4 candidate nodes).

**Evaluation mode does not square intensities.** The one measurement that settles the old
`evaluate=True` double-untransform: with `ints_transform: sqrt`, prediction mode
(`run_inference.py`) and evaluation mode (`run_save_inference_eval.py`) were run over the
same 378 spectra and their `pred_ints` agree to **2.2e-5** relative, with zero spectra
where the evaluation output resembles the squared prediction. `resolve_pred_ints` is gone
and the paths really do agree.

**`cos_hun_inst_tol` was silently exporting -1 for every spectrum - fixed.**
`inst_type_loss_tol` is null in [`config/models_template`](../../config/models_template) (its
entries are commented out), so `_setup_tolerance` registers `inst_tol_lookup` as None and
`_common_step` skips the metric - but the name is registered regardless, so the column was
written as -1.0 for all 378 spectra and the run reported success. The score is in
`run_save_inference_eval.py`'s **default** `--auxiliary_scores`, so any config that does
not set the tolerance table produces it. `_setup_tolerance` now raises when a score needing
the lookup is requested without it; setting `inst_type_loss_tol` yields real values
(mean 0.2032, matching `cos_sim_0.01` at 0.2013), and dropping the score from
`--auxiliary_scores` also passes.

**The eval script no longer asks for it by default.** Promoting `inst_type_loss_tol` to a
template default was tried and reverted (it changes the training loss for every config that
never set the table, and breaks comparability with historical `cos_hun` on QTOF sets), so
`run_save_inference_eval.py` now defaults to plain `cos_hun` and `cos_hun_inst_tol` is opt-in.
The counting behind the reverted attempt was also wrong and is corrected here: **164 of 530**
configs set the table, not 164 of 180 - the majority that do not include
`nist20_merged_inchikey/`, `nist20_merged_scaffold/` and `massspecgym_merged/`, so the old
default would have raised on most checkpoints. Those configs also leave
`spec_params.inst_types` at the template's `["FT"]`, where `cos_hun_inst_tol` equals `cos_hun`
anyway. Measured on the debug config, which does declare QTOF: `cos_hun` 0.2023 vs
`cos_hun_inst_tol` 0.2032, the difference being the QTOF spectra's wider window.

**`--output_everything` could not produce `pred_oos_prob` - fixed.** A leftover of this
refactor: when the three steps collapsed into one `inference_step`, the
`results["oos_prob"] = exp(pred_oos_logprobs)` line stayed *below* the `if metrics is None:
return results` early return, on the ground-truth side. P(out of spectrum) is a plain
forward output and needs no reference spectrum, so prediction-only runs were emitting the
column as NaN for no reason. The exp() now sits above the early return.
`run_inference.py --output_everything` emits all 18 columns with nothing dropped, and
`pred_oos_prob` is a real probability in [0, 1].

**Peak-only output and `min_intensity` behave as documented.**
`--output_subset pred_mzs pred_ints` with both annotation flags off emits the 7 peak-axis
columns and drops all 9 formula/joint/node columns. The threshold is strict `>` on the
0-1000 rescaled intensity, the base peak is exactly 1000 at every setting, and the filter
cascades to all four axes - per-spectrum medians:

| `--min_intensity` | peaks | formulas | joint | nodes | lowest intensity kept |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 240 | 241 | 572 | 107 | 0.0000 |
| 1.0 | 86 | 86 | 221 | 70 | 1.0189 |
| 10.0 | 36 | 36 | 113 | 41 | 10.6681 |
| 50.0 | 13 | 13 | 32 | 24 | 50.2493 |

The node column shrinking with the threshold is the behavior the earlier fix introduced; it
used to be threshold-independent. On the evaluation side `--min_peak_int 10.0` reports its
own filter stats (350,761 peaks to 23,241, 93.4% removed).

### Validation

- **Full suite:** `pytest tests -q` - 2321 passed, 1 skipped, 2 xfailed (+7 new).
- **New tests:** `TestInstTolScoresRequireConfig` in
  [`test_per_inst_type_tolerance.py`](../../tests/test_per_inst_type_tolerance.py) - both
  guarded scores raise, the error names every offending score, configured tolerances build
  the lookup, and not requesting them preserves the no-lookup fast path. Two more in
  `TestRealStepPostcondition`
  ([`test_inference_intensity_contract.py`](../../tests/test_inference_intensity_contract.py))
  pin `oos_prob` on the `metrics=None` path, and that the key stays absent when the model
  predicts no OOS; the first fails when the exp() is moved back below the early return
  (verified).
- **`pred_oos_prob` is constant** (0.1515) across both the 24 prediction spectra and the 378
  eval spectra. The arrays are per-spectrum shaped throughout, so this is the undertrained
  debug model's OOS head, not a broadcast bug. It is not evidence about the production
  model.
- **Ruff:** no new findings on either changed file; the 20 in `spectrum_pl.py` and 1 in the
  test file are pre-existing and elsewhere.

## `run_inference.py` end-to-end matrix (2026-09-16)

The earlier smoke test drove `run_inference.py` with 4 molecules / 24 spectra, which is too
small to exercise chunking, resume or sharding. Re-run on a 50-molecule / 600-spectrum
prediction-only input built with the real preprocessing scripts
(`01_inf_prepare_mol_df.py` -> `02_inf_prepare_spec_df.py`, 2 precursor types x 3 FT NCEs x
3 QTOF eVs, `peaks == []` everywhere), against
`ckpts/debug_tiny_inference_smoke/last.ckpt`.

| Axis | Result |
| --- | --- |
| Default run, `--save_chunk_size 250` | 600/600 spectra, 3 chunks (256/256/88), 600 unique `spec_id`, 0 duplicates |
| Tensor device | 0 non-CPU tensors in any column of any chunk |
| Resume | dropped chunk 2, reran: "Found 512 finished spectra", 88 reprocessed, identical `spec_id` set and identical m/z |
| `--output_everything True` | all 18 columns, `pred_oos_prob` present and **0 NaN** (the fix reaches this script too) |
| `--min_intensity 0 / 10 / 50` | 137,424 -> 8,514 -> 2,484 peaks; lowest surviving intensity 0.0 / 10.02 / 50.37 (strict `>`); base peak exactly 1000.0 in every spectrum |
| `--save_format pkl.gz` | writes `predictions_chunk_0.pkl.gz`, loads, equals the `pkl` run |
| `--dynamic_batch_max 20000` vs `--batch_size 8` | same 120 spectra, every index/string column bit-identical, `pred_ints` max abs diff 3.1e-4 on the 0-1000 scale (~3e-7 relative) |
| `--start_mol_idx 40 --num_mols 10` | exactly the 120 spectra of `mol_df.iloc[40:50]` |
| `--start_mol_idx 500` (past the end) | "nothing to do", exit 0 |
| No `--frag_dp` (on-the-fly DAGs) | 5 molecules in 5.9 s, 0 failures, temp dir cleaned up |
| Export axes | peak / formula / joint / node columns length-aligned per spectrum; 0 out-of-range indices after rebasing; all probabilities in [0, 1]; base-peak annotation resolves to a valid node for all 120 spectra |
| `--save_interval` | `error: unrecognized arguments` (flag is gone) |

Resume reproduces the same spectra but not bit-identical intensities (84/88 rows differ, max
7.3e-4 on the 0-1000 scale): a resumed run packs different batches, and float32 reductions
are not batch-composition invariant. Same cause as the dynamic-vs-fixed delta above.

### Fixed here

The empty-dataset guard blamed the wrong thing when the fragment directory was the problem.
With a `--frag_dp` holding no DAGs, the molecule filter empties `spec_df` first, so the
guard fired as *"Dataset is empty although 0 spectra reached it; the dataset dropped all of
them"* - self-contradictory, and it pointed the reader at `spec_params`. The fragment filter
now raises for its own failure, naming the directory and its entry count, and the dataset
guard is left for the case it was written for.

`pred_oos_prob` is still the constant 0.1515 here, on 600 spectra from 50 molecules rather
than the earlier 24. Per-spectrum shapes are correct throughout, so this remains the
undertrained debug head, not a broadcast bug.

## HDF5 export size (2026-09-16)

Two changes, after checking a real 96k-spectrum eval
(`output/inference_eval/jnokddr9/predictions_test.hdf5`, 212 MB).

**`--drop_none_cols` now reaches the HDF5 writer.** The flag existed on
`run_save_inference_eval.py` but only `run_inference.py` ever read it, so an HDF5 eval
carried a full float64 column of NaN for every output the run did not request - 9 of the
20 datasets in that file. The decision is made on the **first flush and enforced after**:
this writer appends to one file, so a column skipped early and written late would be short
by every earlier spectrum. A dropped column that later holds values raises instead.

**Datasets are chunked and compressed** (`gzip` level 4, byte-shuffle for fixed-width
numbers, `chunks = (65536,) + trailing`). This replaces a dtype downcast that was proposed
and rejected:

| index column, written through the real streaming path | size |
| --- | --- |
| current (int64, no options) | 80.1 MB |
| int32 downcast | 40.0 MB |
| shuffle + gzip4, h5py auto-chunk (1024 rows) | 9.3 MB |
| shuffle + gzip4, `chunks=(65536,)` | **5.1 MB** |

Compression wins 16x where the downcast wins 2x, and stacking int32 on top adds ~5%. So
**no dtype changes**: int64 indices stay int64, and the probability columns were already
float32. Compression is very uneven and it is worth knowing which way: int64 index columns
compress 15-26x (small integers padded into 8 bytes, byte-shuffled into runs of zeros),
float32 probabilities only 1.1-1.3x (high-entropy mantissas). Even the float64 metric
columns land at 0.30 MB compressed, below the 0.4 MB a float32 downcast would have given,
so that idea is superseded too.

Measured end to end: the debug eval's default output 1,549,728 -> 171,571 bytes (9.0x);
the jnokddr9 file recompressed column by column 212.5 -> 65.7 MB (3.2x) with every column
byte-identical and dtype-unchanged.

The large chunk and the compression are one decision, not two. HDF5 allocates a whole chunk
as soon as any of it is touched, so a 65536-row chunk holding a 378-row column costs 528 KB
raw and 7 KB compressed. `compression=None` therefore falls back to `chunks=True`.

`gzip` over `lzf` deliberately: lzf is an h5py-only filter, gzip is standard HDF5. Every
reader in this repo is h5py, but gzip is both smaller at level 4 and portable, and the
write cost is ~0.46 s per 80 MB column.

### Not done

int32 indices. If it is ever revisited, the guard has to be: check max/min **before**
casting (numpy wraps silently), decide once at the first flush and raise later (h5py cannot
widen a dataset's dtype), gate on `rebase_indices=True` (unrebased indices scale with run
size, rebased ones are bounded by one spectrum's DAG - max 36,734 measured), and never
touch `start_points`/`end_points` or `spec_id`/`mol_id`. At 1M spectra the joint/node
offsets reach ~700M, only 3.1x under the int32 ceiling, and `spec_id` already reaches
1,010,470.

### Validation

- **Full suite:** 2337 passed, 1 skipped, 2 xfailed (+16: 7 for `drop_none_cols`, 9 for
  compression).
- New tests pin that every dataset is chunked and compressed, that shuffle is on for
  numbers and off for variable-length strings, that floats/int64 dtypes/vlen strings/index
  pairs all survive the round trip, and that `compression=None` opts out of both the filter
  and the large chunk.
- **Ruff:** clean on the new test file; the 3 findings in `script_utils.py` are
  pre-existing and elsewhere.

## Follow-up

Remove the `pred_logprobs` compatibility key once the ensembling artifacts are regenerated
or `calculate_ensemble_agreement_stats.py` is switched to log `pred_ints` itself.

`ColumnSpec.token` now records which token requests each column, so making `--output_subset`
actually narrow the emitted columns (retiring `--drop_none_cols`) is a small change. It is
deliberately not done here: it would change the saved artifact schema, which
[the guide](../guides/run_inference.md) documents as fixed.
