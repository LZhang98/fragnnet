# MASShatter code audit

> **Status:** REFERENCE · **Updated:** 2026-09-07 · Pickle API fix tested in working tree; bfloat16 issue deferred by user.

## SMARTS merged-run readiness (2026-09-09)

Checked clean revision `7698d285`. SMARTS matching/cache preparation, candidate assembly,
mechanism normalization and gradients, standalone prediction, and forced-merge behavior pass
the focused CPU suite: **494 passed, 5 skipped** across `test_mass_shatter_smarts.py`,
`test_mass_shatter_model.py`, `test_mass_shatter_pl.py`, and
`test_mass_shatter_forced_merge.py`, using the configured FRAGNNET-GPU interpreter.
An additional synthetic diphenyl-sulfoxide forward/backward probe combined SMARTS,
unique fragments, GNN context, and merged NCE readings 20/40. Total probability was 1.0,
loss was 4.61925, and mechanism-embedding gradient L1 was 0.003975; all parameter
gradients present were finite. This used the small model-test factory, not the full run config.

**Current merged warmup configs cannot enable SMARTS directly.**
`MassShatterPL._setup_model` explicitly rejects SMARTS with any non-null
`mass_shatter_proposal_only_until_epoch`; the current
[merged context pair](../plans/conditioning_refactor_plan.md#merged-masshatter-context-at-gnn-input)
sets that boundary to 15. Existing regression coverage verifies the rejection.
The supported configuration is `mass_shatter_smarts_templates: True` with
`mass_shatter_proposal_only_until_epoch: null`. For a controlled SMARTS comparison,
both SMARTS-on and SMARTS-off arms need the same schedule. Alternatively, supporting
SMARTS warmup requires an explicit mechanism-head training policy and implementation/tests;
removing the guard alone is not validated.

No model/config changes or training runs were made during the audit. CUDA was unavailable; real-data GPU
throughput, full-config training, and held-out quality remain unverified. Next: select the
matched schedule, then run a real-data GPU smoke check before the merged experiment.

Config follow-up: prepared the requested
[SMARTS merged GNN-context config](../../config/masshatter/merged/smarts/nist20_merged_full_metadata_context_gnn_heads_smarts_no_warmup.yml)
with SMARTS enabled and proposal-only warmup disabled. Runner-resolved comparison against
the GNN-context arm confirms only those two behavior settings and W&B name/group/tags differ.
The full resolved `MassShatterPL` constructs successfully with its mechanism head and GNN
context enabled. No training launched. Next: real-data GPU smoke testing; a matched SMARTS-off
control must also disable warmup to isolate the template effect.

## Third audit pass: remaining API issues

This pass inspected `77fdbea2` plus the uncommitted inference and feasibility fixes
described below. Those changes were preserved; no source or tests changed during the
audit itself. The subsequent user-authorized fix addresses finding 6 only. Finding 7
is deferred because bfloat16 is not used in the current workflow.

### 6. P2: `run_inference_pkl` rejects its prediction-only intensity output

The separate [run_inference_pkl API](../../src/fragnnet/utils/script_utils.py) invokes
`model.inference_step` when `evaluate=False`. That step returns already processed
`pred_ints`. However, the `pred_ints` token branch at lines 955–960 insists on
`pred_logprobs` and raises instead of consuming the available intensities.

Reproduced with real `MassShatterPL` models for both probability and independent
intensity output, an ethanol graph, precursor/instrument index 0, NCE 25,
`evaluate=False`, and `output_set={'pred_mzs', 'pred_ints'}`. `inference_step` succeeds
and supplies `pred_ints`; the exporter raises
`KeyError: output token 'pred_ints' needs model output 'pred_logprobs', which is absent/None`.
The existing six pickle-token tests use an evaluation stub supplying `pred_logprobs`,
so they do not cover this contract.

This is distinct from the fixed CLI `--save_format pkl` path, which uses the tabular
iterator. **Fixed in the working tree:** the exporter now accepts processed `pred_ints`
without applying the inverse transform again, while preserving raw-logprob handling
for evaluation outputs. Two real-model regressions cover both MASShatter output modes
over two batches with a square-root intensity transform. They assert exact preservation
of processed intensities and masses, plus correct batch-index offsets. Both failed
before the fix. Validation: **90 passed** across `test_mass_shatter_pl.py` and
`test_run_inference_pkl_tokens.py`; CPU only. No checkpoint or config changes are needed.

### 7. P2: CPU bfloat16 autocast rejects valid proposal mixture weights

[BernoulliMixtureWeightHead](../../src/fragnnet/model/mass_shatter_model.py) runs
`log_softmax` without specifying a computation dtype. Under CPU bfloat16 autocast,
its returned log weights remain bfloat16. The
[candidate builder](../../src/fragnnet/frag/mass_shatter_candidates.py) then converts
them to float64 and requires their log normalizer to be within `1e-6` of zero.
Bfloat16 rounding routinely exceeds this tolerance even for the head's own output.

Reproduced with the model-test factory on ethanol: `proposal_num_mixtures=2`,
`torch.autocast('cpu', dtype=torch.bfloat16)`, precursor index 0, and NCE 25 raises
`ValueError: mixture_log_weights must be finite and log-normalized`. Float32 with
two mixtures and CPU bfloat16 with one mixture both produce finite predictions.
A head-only probe returned bfloat16 log weights `[-0.71484375, -0.66796875]`, whose
float64 log normalizer is approximately `0.0020`.

Scope limitation: the standard runner explicitly falls back to float32 on CPU,
so its normal CPU training path avoids this failure. This finding applies to direct
CPU mixed-precision model calls; CUDA behavior was not verified. **Deferred by user**
on 2026-09-07 because bfloat16 is not used in the current workflow. If revisited, compute
mixture log normalization in sufficient precision before selection/scoring, and test
mixed-precision mixtures in addition to the existing single-mixture autocast case.

Validation this pass: **156 passed, 5 skipped** for `test_mass_shatter_model.py`,
`test_mass_shatter_candidates.py`, and `test_run_inference_pkl_tokens.py`. The failures
above were reproduced with separate CPU probes. No GPU or full-suite run was made.
Follow-up: finding 6 is fixed as noted above; finding 7 remains deferred. No full-suite
or GPU validation was added for the pickle API fix. Ruff introduces no new findings;
the shared utility file retains its four pre-existing findings. Git whitespace checks pass.

## Continued-audit fix outcome

Findings 4 and 5 below are fixed in the working tree:

- The CLI uses model-specific dataset dispatch and limits DAG generation/filtering
  to DAG datasets. MASShatter retains inputs even with a missing or empty DAG directory;
  FraGNNet retains its existing DAG filtering, including the CLI's legacy name alias.
- Tabular export accepts exactly one of `pred_logprobs` and `pred_intensities` and
  applies the same inverse intensity transform to the resulting normal-space weights.
- The remaining feasibility helper now uses neutral elemental mass before the ion
  delta, matching the earlier model fix. Its artificial zero-mass boundary discrepancy
  is covered by a regression case.

Executing the CLI in regression tests also exposed an output directory created after
its first use and column-major iterator output passed to a row-oriented DataFrame
constructor. These were corrected so prediction files can actually be written. Parquet
output converts tensor cells to scalars/lists; pickle output preserves tensor cells.

Validation: **781 passed, 8 skipped** across MASShatter, bond heads, formula utilities,
checkpoint/HDF5/pickle utilities, and inference integration tests. After adding Parquet
boundary coverage, the six CLI regressions passed again. They execute the actual CLI,
model prediction, inverse square-root intensity transform, iterator, and chunk writing;
checkpoint loading and dataset preprocessing are mocked. They cover both output modes,
missing/empty DAG directories, periodic/final writes, new output directories, two-spectrum
row alignment, expected masses/intensities, and preserved FraGNNet DAG filtering.

Neither `pyarrow` nor `fastparquet` is installed. In that case the test replaces the
Parquet codec with a writer that validates scalar/list cells and persists pickle data;
real pickle serialization is checked, but a real Parquet round trip is **not verified**.
No dependencies were installed. No GPU, full repository suite, real checkpoint/data CLI
run, training, or validation-set evaluation was performed.

Ruff passes on the CLI and tests. The shared utility files retain five pre-existing
Ruff findings, verified against `HEAD`; no new lint findings were introduced. Git
whitespace and documentation links pass. No checkpoint/config schema or cache migration
is required. Input inclusion now follows the correct dataset path, so prior incomplete
CLI outputs should be regenerated. The feasibility correction may change inclusion at
the artificial near-zero-mass boundary described below. Next check: a real-checkpoint
CLI smoke run in the target environment, including its Parquet engine.

## Continued audit at `77fdbea2`

The checkout was clean at the start of this pass. The initial fixes below are now
present in Git history, so earlier working-tree status descriptions are historical.
No source or test changes were made during the audit itself. The subsequent authorized
fixes are summarized above; the findings below preserve the pre-fix evidence.

### 4. P1: Inference CLI still requires DAG files for MASShatter

[run_inference.py](../../scripts/run_inference.py) skips DAG generation for MASShatter
but unconditionally lists `config_d['frag_dp']`, filters molecule and spectrum tables
to IDs found there, and constructs `SpecMolFragDataset`. MASShatter's canonical
dataset selection in [setup_utils.py](../../src/fragnnet/utils/setup_utils.py) is
`SpecMolDataset`; external DAG files are not an input to its candidate generation.

Executing the CLI's exact filtering statements (AST nodes at lines 357–361) with
`model_type='mass_shatter'`, an empty temporary fragment directory, and one molecule
and spectrum removes both rows. A nonexistent directory instead fails at `os.listdir`.
Even a populated, unrelated DAG directory restricts output to its molecule IDs.
This is separate from the previously fixed SMARTS sidecar problem: fixing the model
wrapper does not remove these earlier CLI restrictions.

Next fix: limit DAG generation/filtering to DAG models and use canonical dataset
dispatch. Verify the CLI produces predictions for molecules with no DAG files.

### 5. P1: Tabular export crashes for independent intensity output

[process_batch_to_tabular_row](../../src/fragnnet/utils/script_utils.py) unconditionally
reads `batch_result['pred_logprobs']` at line 1512. The standard CLI calls it through
`run_inference_tabular_iter` with `evaluate=False`, which takes raw `predict_step`
output. Independent MASShatter returns `pred_intensities`, not `pred_logprobs`.

Reproduction used a real `MassShatterPL` from the test config factory, independent
intensity output, unique-fragment/proposal-selection mode, proposal anchoring, proposal
loss weight 0.1, and cosine-distance loss. For an ethanol graph, precursor/instrument
index 0, and NCE 25, prediction succeeds and contains `pred_intensities`. Passing the
same batch to `process_batch_to_tabular_row` raises `KeyError: 'pred_logprobs'`.
This remains a blocker even after the CLI DAG dependency is corrected.

Next fix: handle both sparse intensity contracts, applying the correct inverse
intensity transform. Cover the actual tabular export path with independent output.

### Remaining mass-filter consistency issue

`mass_shatter_has_valid_ion` in
[formula_utils.py](../../src/fragnnet/utils/formula_utils.py) still subtracts summed
formal charge times electron mass before applying the ion delta. It is only a
positive-mass feasibility check, not the source of predicted masses. The discrepancy
is reproducible for the artificial `[H+]`, `[M-H]-`, zero-H-transfer/no-CMF case:
the filter returns False while the corrected model allows m/z 0.00054858. This is
not evidence of a realistic dataset failure; ordinary positive masses do not change
feasibility under the tiny correction. Align this helper in a chemistry follow-up,
but do not treat it as a demonstrated training-data regression.

### Ruled out and validation

Independent output's auxiliary `cross_ev_fragment_logprobs` does not represent its
actual summed channel intensities. A controlled head-output probe showed zero
auxiliary delta versus a 0.57596 normalized intensity delta. However,
`MassShatterPL._setup_loss_fn` explicitly rejects independent output combined with
the cross-EV loss. This is therefore not a supported-training correctness finding.

The configured interpreter reran all MASShatter tests and
`tests/test_bond_fragment_heads.py`: **718 passed, 7 skipped**. The inference failures
above were verified with targeted probes outside the existing tests. No full CLI run,
GPU validation, training, full repository suite, or dataset-level evaluation was run.
Next action: fix findings 4 and 5 with end-to-end inference regression coverage.

## Fix outcome and compatibility

All three findings below describe the original `0b5ad3f4` behavior. The authorized
follow-up fixes are implemented without changing configuration schemas, checkpoint
parameter shapes, or preprocessing caches:

- `MassShatterPL.forward` now constructs the same cached SMARTS templates as the
  transfer hook when standalone inference lacks them. Direct `MassShatterModel`
  callers must supply matched templates when SMARTS is enabled; missing templates
  raise instead of silently omitting mechanisms. Explicit empty matches remain valid.
- Component masses now sum neutral elemental masses, leaving the electron correction
  to the existing CRF/CMF deltas. Relative to the audited revision, each affected
  component's m/z changes by its summed precursor formal charge times 0.00054858 Da.
- SMARTS mechanism scoring now uses proposal embeddings and proposal context. Shared
  proposals use their CE-independent context. Tests cover shared/separate encoders,
  shared/per-spectrum proposals, and gradients from both objectives.

Existing checkpoints remain structurally loadable, but corrected masses and SMARTS
mechanism scores can change predictions and resumed training. Do not compare old
cached predictions with new evaluations as if the implementations were identical.
Re-evaluate affected checkpoints under one revision; pin the audited revision and
original inference path when exact historical reproduction is required. No automatic
checkpoint conversion or data regeneration is needed. Dataset-level scientific impact
has not been measured.

Validation: all MASShatter tests plus `tests/test_bond_fragment_heads.py` passed:
**718 passed, 7 skipped**, using the configured interpreter. Eleven new parameterized
cases cover the fixes and missing-template errors; the original failures were verified
before implementation. Ruff and Git whitespace checks passed. No full repository suite,
GPU validation, training, or validation-set evaluation was run. The Cython extension
was unchanged. Next action: review the diff and evaluate affected checkpoints before
using revised metrics in experiment comparisons.

## Scope and validation

Originally audited revision `0b5ad3f4`, starting with a clean working tree. Reviewed candidate
construction, model chemistry and channel selection, proposal gradient routing, Lightning
integration, and standalone inference. No model or preprocessing changes were made during
the initial audit; see the follow-up fixes above.

The configured `FRAGNNET-GPU` interpreter ran all 11 `tests/test_mass_shatter_*.py`
files: **659 passed, 5 skipped**. CUDA was unavailable; GPU behavior, the full repository
suite, training, and validation-set evaluation were not checked. The installed Cython
extension was used without rebuilding. The focused probes below reused `_build_batch`
and `_build_model` from `tests/test_mass_shatter_model.py` via `runpy.run_path`.

## 1. Standalone inference silently omits SMARTS mechanisms

[Model forward](../../src/fragnnet/model/mass_shatter_model.py) initializes empty templates
and populates them only when `static_graph.smarts_templates` exists. That sidecar is built
by [MassShatterPL.transfer_batch_to_device](../../src/fragnnet/pl_model/mass_shatter_pl.py).
However, [standalone inference](../../src/fragnnet/utils/script_utils.py), including
`process_batch_to_tabular_row`, uses `to_device` directly and bypasses that hook.
Consequently, enabling `mass_shatter_smarts_templates` changes training support but does
not enable those mechanisms in the ordinary standalone inference path.

Reproduction: evaluate `c1ccccc1S(=O)c1ccccc1` with SMARTS enabled, unique-fragment mode,
`max_masks=4`, `max_enumerated_masks=8`, precursor index 0, and NCE 25. With a populated
sidecar the model returns 9 unique components and 27 channels; without it, 7 components
and 21 channels. Six masses disappear, including 47.96643704 and 154.07770174.

Next check: route inference through the model's transfer hook or construct templates
independently of transfer, then assert identical candidate support between Lightning and
standalone inference for a SMARTS-enabled checkpoint.

## 2. Charged components receive an extra electron-mass correction

`_fragment_masses_and_valid_channels` subtracts atom formal charge times electron mass
from neutral isotope masses before applying CRF/CMF deltas. Those deltas already include
the output ion's electron correction. Components with nonzero summed formal charge
therefore disagree with the repository's formula-based mass calculation. This can also
affect fragments split from a net-neutral charge-separated precursor.

Reproduction: the intact `C[N+](C)(C)C` graph, `[M]+`, no CMF, and zero H transfer produces
**74.095877224**. The canonical
`formula_to_peak_mzs('C4H12N', '[M]+', isotopes=False)` produces **74.096425804**.
The difference is exactly one electron mass, 0.00054858 Da (about 7.4 ppm here).

Next check: establish neutral elemental component masses before applying ion deltas;
add charged-component formula parity tests. Treat a fix as a scientific behavior change
with the compatibility work required by the engineering rules.

## 3. SMARTS proposal gradients bypass encoder separation

Mechanism scoring pools `chemical_bond_embeddings` from the intensity-side
`atom_embeddings` and consumes live `context`. It does not use proposal-side embeddings
or the detached proposal context. This bypasses the gradient isolation otherwise
implemented by `proposal_selection_only` and `separate_proposal_encoder`.

Reproduction: use the SMARTS molecule and budgets above, with
`proposal_selection_only=True`, `separate_proposal_encoder=True`, and
`use_base_probability_feature=False`. Backpropagate only
`-output['proposal_fragment_logprobs'].mean()`. Absolute parameter-gradient sums are
0.29058 for `precursor_encoder`, 0.01575 for `context_encoder`, and 34.15009 for
`proposal_encoder`. The first two should remain zero under this isolation contract.
The existing isolation test covers a molecule with no SMARTS mechanisms.

Next check: derive mechanism inputs from the appropriate proposal stream and add a
SMARTS-enabled variant of the existing objective-isolation regression test.
