# MASSter code audit

> **Status:** REFERENCE · **Updated:** 2026-09-13
> Candidate-attention A/B audit: seven findings fixed in the working tree; CUDA bfloat16 autocast now passes.

## Candidate attention topology audit (2026-09-13)

Audit of the uncommitted `full` candidate-attention arm on
`research/masshatter-ion-h-selection`, covering the MASSter GNN wiring, the candidate
intensity transformer, and the prepared A/B config. Seven findings; all fixed in the
working tree. No training or scientific evaluation ran.

### Verified correct before the fixes

Bucketed contextualization masking is exact for both topologies. Against a per-molecule
unpadded reference with ragged counts `[1, 3, 4, 9, 4]` — chosen so two power-of-two
buckets hold non-contiguous molecule sets — the maximum absolute deviation is 2.384e-07
(`full`) and 4.172e-07 (`induced`); permuting row order changes nothing beyond 4.768e-07.
No cross-molecule leakage, and no fully masked query row is reachable because every
molecule contributes at least one real candidate.

Measured contextualization cost at the head geometry (dim 384, 3 layers, ff 1024, B=16),
on an RTX 2070 SUPER rather than the L40S the config cites:

| N | topology | forward | peak (fwd+bwd) |
| --- | --- | --- | --- |
| 256 | induced | 15.10 ms | 1053.1 MiB |
| 256 | full | 10.32 ms | 664.5 MiB |
| 1024 | induced | 37.24 ms | 2968.6 MiB |
| 1024 | full | 50.12 ms | 2471.0 MiB |

`full` wins at the configured 256-fragment budget and loses by N=1024, which is what
motivates finding 4.

### 1. Hyperparameter fallback default and missing checkpoint migration

`MassterPL` read the new topology through
`self.hparams.get("masster_intensity_transformer_attention", "induced")` in both
`_setup_model` and `_setup_loss_fn`. The [Python rules](../../rules/python.md) forbid a
fallback default for a model hyperparameter, and every other `masster_*` key in that
file uses direct attribute access. Fixed by direct attribute access plus an explicit
migration in `get_pl_hparams`, matching the `masster_context_in_gnn` precedent, so
pre-topology checkpoints resolve to `induced` while a fresh config that omits the key
fails during setup. Regression coverage in `test_masster_pl.py` asserts strict
state-dict loading after migration and the `AttributeError` for an undeclared key.

Note: the loss-setup value check is unreachable, because the head validates
`attention_type` during model construction, which precedes it. The adjacent
`masster_intensity_output` check has the same property, so it was left in place for
consistency rather than removed.

### 2. Config parameter counts described a different head geometry

The A/B config quoted 4,643,732 versus 8,835,092 intensity-head parameters. Those
correspond to `atom_decoder_num_layers=0` and `separate_ion_h_support_head=False`; the
config sets 4 and `True`. At the configured geometry the counts are 12,542,119 (`full`)
against 16,733,479 (`induced`). The 4,191,360 difference is identical either way and lies
entirely in the contextualization layers, so the capacity argument was sound and only the
absolute figures were wrong. Corrected in the config comment.

### 3. Unstated A/B confound

A `full` layer carries one attention and one feed-forward block where an `induced` layer
carries two of each, so the arm is both cheaper and roughly half the contextualization
capacity. A win is clean evidence. A null or a loss cannot separate "full attention is no
better" from "half the per-layer capacity hurt" and needs a depth- or width-matched full
arm before concluding against the topology. Recorded in the config comment, together with
the check that both arms should track the control exactly through the epoch-15
proposal-only warmup, where the intensity head is frozen in eval mode and the loss is
replaced by the proposal loss.

### 4. `full` attention was not bound to unique-fragment mode

Only unique-fragment mode bounds a molecule's candidate count by `max_masks`. The legacy
pathway-token branch emits one token per component occurrence — thousands per molecule at
a 1024-mask budget — where the quadratic topology is not affordable, as the N=1024
measurement above shows. `MassterModel.__init__` now rejects the combination, with a
regression test covering both the rejection and construction under unique-fragment mode.

### 5. CUDA bfloat16 autocast crash in retained-fragment diagnostics

`test_masster_supports_cuda_bfloat16_autocast` failed with
`index_add_(): self (BFloat16) and source (Float) must have the same scalar type`.
Autocast keeps `exp()` in float32 while `bond_logits` stays bfloat16, so the accumulator
built by `bond_logits.new_zeros(batch_size)` could not take the float32 contribution.
Pre-existing in committed code and independent of the attention arm; it does not affect
the prepared run, which uses `precision: 32`. Fixed by building the accumulator in the
contribution dtype and casting back. Distinct from the CPU bfloat16 proposal-mixture
issue deferred by the user above.

### 6. Two degenerate intensity-head inputs in unique-fragment mode

Confirmed by gradient probe on a spectrum-output backward pass, not by inference:
`component_residual_head` receives no gradient, because the caller discards the
component-level residual and reads the mask-pooled one; and the head is called with
identity mask indices, so the per-mask component count is one everywhere and its `log1p`
input feature is a constant.

Both remain live in the legacy pathway-token mode. Removing them would change
`input_projection`'s input width and the state-dict keys, invalidating every existing
MASSter checkpoint, which the [engineering rules](../../rules/engineering.md) put behind
explicit authorization and compatibility work. Documented at the head and the call site
instead; the removal is still open if the user wants it.

### 7. Formatting and a per-layer host sync

`ruff format` deviations in `masster_model.py` were applied. `_efa_linear_attn`
derived `num_graphs` via `.item()` inside the checkpointed function, costing one host sync
per GINE-EFA layer, doubled by checkpoint recomputation. It is now derived once per
encoder forward and passed in; the refactor is bit-identical (maximum absolute difference
0.000e+00 against the previous formulation).

### Validation

274 tests passed across `test_bond_fragment_heads.py`, `test_masster_model.py`,
`test_masster_pl.py`, `test_masster_gnn_context.py`, and
`test_merged_ce_context.py`, including the previously failing CUDA bfloat16 case. Ruff
check and format pass on every changed file; the remaining format deviations in
`nn_blocks.py` and `test_masster_pl.py` are pre-existing lines outside these changes.
Both A/B arms construct from their configs, and their resolved configurations differ only
in `wandb_name`, `wandb_tags`, and `masster_intensity_transformer_attention`.

A full-suite run at 18:08-18:10 gave 92 failed, 2,873 passed, 1 skipped, 2 xfailed. Every
failure came from the then-stale multi-cut Cython extension, and all 92 carried one of two
signatures: `compute_ccs_multi_cut() got an unexpected keyword argument
'max_ring_cut_steps'` (85) or `... 'ring_open_mask'` (7), accounting for the total exactly.
They fell in `test_multi_cut_bfs.py` (86), `test_frag_utils_compute_dags.py` (3),
`test_nl_prepass.py` (2), and `test_num_cut_bonds.py` (1); none import the changed modules
except `test_num_cut_bonds.py`, which builds MPNN/NodeMLP rather than the GINE-EFA path this
change touches. These changes are Python-only and cannot affect the Cython BFS.

Both extensions were rebuilt in place at 2026-09-13 18:12, after those runs.
`masster_core` was then force-rebuilt at 22:38 from the current source; the result is
byte-identical to the 18:12 binary, which confirms that build already matched
`masster_core.pyx`, so the earlier two-week staleness gap had already been closed
rather than being a live defect. All four extensions now carry binaries newer than their
sources.

After the rebuild the full suite passes: **2,975 passed, 1 skipped, 2 xfailed, 0 failed**.
This is the current baseline for this checkout.

No training, cluster submission, or validation-set evaluation ran. Timings come from an
RTX 2070 SUPER, so the config's L40S figures were not reproduced, only their ordering.

## SMARTS proposal-only-warmup guard removed (2026-09-11)

`MassterPL._setup_model` no longer rejects `masster_smarts_templates=True`
combined with a non-null `masster_proposal_only_until_epoch`. The prior guard
(introduced in `612efaae8`) had no documented mathematical incompatibility (see the
readiness audit below); the mechanism head was already routed through
`_proposal_modules()` and, per the "Fix outcome and compatibility" section above,
SMARTS mechanism scoring already uses proposal-side embeddings and context, so it
follows the same gradient-isolation contract as the rest of the proposal branch
regardless of the staged-training epoch.

Two new regression tests in `test_masster_pl.py` cover the combination:
`test_proposal_warmup_supports_smarts_templates` confirms construction succeeds and
`mechanism_head` is included in `_proposal_modules()` and stays trainable across the
epoch boundary, and `test_smarts_proposal_warmup_isolates_gradients_across_boundary`
(parametrized over shared/separate proposal encoders) runs a real forward/backward
through `MassterPL._apply_staged_training` on a diphenyl-sulfoxide molecule with
`masster_unique_fragment_mode`: during proposal-only epochs `mechanism_head`
receives nonzero gradient and `intensity_head` receives none (it is frozen), and after
the boundary a joint backward gives both heads (and, for the separate-encoder case,
both `proposal_encoder` and `precursor_encoder`) nonzero gradient. All existing
MASSter suites still pass: 558 passed (one pre-existing, unrelated CPU bfloat16
autocast failure, tracked separately below, reproduced identically before this change).

This clears the blocker recorded in the SMARTS merged-run readiness audit below and in
`docs/project-state.md`. The prepared
[SMARTS warmup config](../../../config/masster/merged/smarts/nist20_merged_full_metadata_context_gnn_heads_smarts_warmup_e0_15.yml)
can now construct. No training was launched and no real-data GPU smoke check was run;
that remains the next step before comparing results against the no-warmup SMARTS arm.

## SMARTS merged-run readiness (2026-09-09)

Checked clean revision `7698d285`. SMARTS matching/cache preparation, candidate assembly,
mechanism normalization and gradients, standalone prediction, and forced-merge behavior pass
the focused CPU suite: **494 passed, 5 skipped** across `test_masster_smarts.py`,
`test_masster_model.py`, `test_masster_pl.py`, and
`test_masster_forced_merge.py`, using the configured FRAGNNET-GPU interpreter.
An additional synthetic diphenyl-sulfoxide forward/backward probe combined SMARTS,
unique fragments, GNN context, and merged NCE readings 20/40. Total probability was 1.0,
loss was 4.61925, and mechanism-embedding gradient L1 was 0.003975; all parameter
gradients present were finite. This used the small model-test factory, not the full run config.

**Current merged warmup configs cannot enable SMARTS directly.**
`MassterPL._setup_model` explicitly rejects SMARTS with any non-null
`masster_proposal_only_until_epoch`; the current
[merged context pair](../../plans/conditioning_refactor_plan.md#merged-masster-context-at-gnn-input)
sets that boundary to 15. Existing regression coverage verifies the rejection.
The supported configuration is `masster_smarts_templates: True` with
`masster_proposal_only_until_epoch: null`. For a controlled SMARTS comparison,
both SMARTS-on and SMARTS-off arms need the same schedule. Alternatively, supporting
SMARTS warmup requires validating the existing mechanism-head gradient path and staged
transition before relaxing the guard; removing the guard alone is not validated. Follow-up
source inspection found that `_proposal_modules()` already includes the mechanism head and
unique-fragment proposal probabilities include mechanism probabilities. Thus the code already
suggests training that head through proposal coverage during warmup. The guard was introduced
in `612efaae8` ("warmup and new baseline config"); no specific mathematical incompatibility
or failure rationale was found. The restriction is established; its necessity is not.

No model/config changes or training runs were made during the audit. CUDA was unavailable; real-data GPU
throughput, full-config training, and held-out quality remain unverified. Next: select the
matched schedule, then run a real-data GPU smoke check before the merged experiment.

Config follow-up: prepared the requested
[SMARTS merged GNN-context config](../../../config/masster/merged/smarts/nist20_merged_full_metadata_context_gnn_heads_smarts_no_warmup.yml)
with SMARTS enabled and proposal-only warmup disabled. Runner-resolved comparison against
the GNN-context arm confirms only those two behavior settings and W&B name/group/tags differ.
The full resolved `MassterPL` constructs successfully with its mechanism head and GNN
context enabled. No training launched. Next: real-data GPU smoke testing; a matched SMARTS-off
control must also disable warmup to isolate the template effect.

## Third audit pass: remaining API issues

This pass inspected `77fdbea2` plus the uncommitted inference and feasibility fixes
described below. Those changes were preserved; no source or tests changed during the
audit itself. The subsequent user-authorized fix addresses finding 6 only. Finding 7
is deferred because bfloat16 is not used in the current workflow.

### 6. P2: `run_inference_pkl` rejects its prediction-only intensity output

The separate [run_inference_pkl API](../../../src/fragnnet/utils/script_utils.py) invokes
`model.inference_step` when `evaluate=False`. That step returns already processed
`pred_ints`. However, the `pred_ints` token branch at lines 955–960 insists on
`pred_logprobs` and raises instead of consuming the available intensities.

Reproduced with real `MassterPL` models for both probability and independent
intensity output, an ethanol graph, precursor/instrument index 0, NCE 25,
`evaluate=False`, and `output_set={'pred_mzs', 'pred_ints'}`. `inference_step` succeeds
and supplies `pred_ints`; the exporter raises
`KeyError: output token 'pred_ints' needs model output 'pred_logprobs', which is absent/None`.
The existing six pickle-token tests use an evaluation stub supplying `pred_logprobs`,
so they do not cover this contract.

This is distinct from the fixed CLI `--save_format pkl` path, which uses the tabular
iterator. **Fixed in the working tree:** the exporter now accepts processed `pred_ints`
without applying the inverse transform again, while preserving raw-logprob handling
for evaluation outputs. Two real-model regressions cover both MASSter output modes
over two batches with a square-root intensity transform. They assert exact preservation
of processed intensities and masses, plus correct batch-index offsets. Both failed
before the fix. Validation: **90 passed** across `test_masster_pl.py` and
`test_run_inference_pkl_tokens.py`; CPU only. No checkpoint or config changes are needed.

### 7. P2: CPU bfloat16 autocast rejects valid proposal mixture weights

[BernoulliMixtureWeightHead](../../../src/fragnnet/model/masster_model.py) runs
`log_softmax` without specifying a computation dtype. Under CPU bfloat16 autocast,
its returned log weights remain bfloat16. The
[candidate builder](../../../src/fragnnet/frag/masster_candidates.py) then converts
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

Validation this pass: **156 passed, 5 skipped** for `test_masster_model.py`,
`test_masster_candidates.py`, and `test_run_inference_pkl_tokens.py`. The failures
above were reproduced with separate CPU probes. No GPU or full-suite run was made.
Follow-up: finding 6 is fixed as noted above; finding 7 remains deferred. No full-suite
or GPU validation was added for the pickle API fix. Ruff introduces no new findings;
the shared utility file retains its four pre-existing findings. Git whitespace checks pass.

## Continued-audit fix outcome

Findings 4 and 5 below are fixed in the working tree:

- The CLI uses model-specific dataset dispatch and limits DAG generation/filtering
  to DAG datasets. MASSter retains inputs even with a missing or empty DAG directory;
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

Validation: **781 passed, 8 skipped** across MASSter, bond heads, formula utilities,
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

### 4. P1: Inference CLI still requires DAG files for MASSter

[run_inference.py](../../../scripts/run_inference.py) skips DAG generation for MASSter
but unconditionally lists `config_d['frag_dp']`, filters molecule and spectrum tables
to IDs found there, and constructs `SpecMolFragDataset`. MASSter's canonical
dataset selection in [setup_utils.py](../../../src/fragnnet/utils/setup_utils.py) is
`SpecMolDataset`; external DAG files are not an input to its candidate generation.

Executing the CLI's exact filtering statements (AST nodes at lines 357–361) with
`model_type='masster'`, an empty temporary fragment directory, and one molecule
and spectrum removes both rows. A nonexistent directory instead fails at `os.listdir`.
Even a populated, unrelated DAG directory restricts output to its molecule IDs.
This is separate from the previously fixed SMARTS sidecar problem: fixing the model
wrapper does not remove these earlier CLI restrictions.

Next fix: limit DAG generation/filtering to DAG models and use canonical dataset
dispatch. Verify the CLI produces predictions for molecules with no DAG files.

### 5. P1: Tabular export crashes for independent intensity output

[process_batch_to_tabular_row](../../../src/fragnnet/utils/script_utils.py) unconditionally
reads `batch_result['pred_logprobs']` at line 1512. The standard CLI calls it through
`run_inference_tabular_iter` with `evaluate=False`, which takes raw `predict_step`
output. Independent MASSter returns `pred_intensities`, not `pred_logprobs`.

Reproduction used a real `MassterPL` from the test config factory, independent
intensity output, unique-fragment/proposal-selection mode, proposal anchoring, proposal
loss weight 0.1, and cosine-distance loss. For an ethanol graph, precursor/instrument
index 0, and NCE 25, prediction succeeds and contains `pred_intensities`. Passing the
same batch to `process_batch_to_tabular_row` raises `KeyError: 'pred_logprobs'`.
This remains a blocker even after the CLI DAG dependency is corrected.

Next fix: handle both sparse intensity contracts, applying the correct inverse
intensity transform. Cover the actual tabular export path with independent output.

### Remaining mass-filter consistency issue

`masster_has_valid_ion` in
[formula_utils.py](../../../src/fragnnet/utils/formula_utils.py) still subtracts summed
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
`MassterPL._setup_loss_fn` explicitly rejects independent output combined with
the cross-EV loss. This is therefore not a supported-training correctness finding.

The configured interpreter reran all MASSter tests and
`tests/test_bond_fragment_heads.py`: **718 passed, 7 skipped**. The inference failures
above were verified with targeted probes outside the existing tests. No full CLI run,
GPU validation, training, full repository suite, or dataset-level evaluation was run.
Next action: fix findings 4 and 5 with end-to-end inference regression coverage.

## Fix outcome and compatibility

All three findings below describe the original `0b5ad3f4` behavior. The authorized
follow-up fixes are implemented without changing configuration schemas, checkpoint
parameter shapes, or preprocessing caches:

- `MassterPL.forward` now constructs the same cached SMARTS templates as the
  transfer hook when standalone inference lacks them. Direct `MassterModel`
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

Validation: all MASSter tests plus `tests/test_bond_fragment_heads.py` passed:
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

The configured `FRAGNNET-GPU` interpreter ran all 11 `tests/test_masster_*.py`
files: **659 passed, 5 skipped**. CUDA was unavailable; GPU behavior, the full repository
suite, training, and validation-set evaluation were not checked. The installed Cython
extension was used without rebuilding. The focused probes below reused `_build_batch`
and `_build_model` from `tests/test_masster_model.py` via `runpy.run_path`.

## 1. Standalone inference silently omits SMARTS mechanisms

[Model forward](../../../src/fragnnet/model/masster_model.py) initializes empty templates
and populates them only when `static_graph.smarts_templates` exists. That sidecar is built
by [MassterPL.transfer_batch_to_device](../../../src/fragnnet/pl_model/masster_pl.py).
However, [standalone inference](../../../src/fragnnet/utils/script_utils.py), including
`process_batch_to_tabular_row`, uses `to_device` directly and bypasses that hook.
Consequently, enabling `masster_smarts_templates` changes training support but does
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
