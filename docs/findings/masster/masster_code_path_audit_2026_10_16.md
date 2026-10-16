# MASSter current code-path audit

> **Status:** FINDING · **Updated:** 2026-10-16
> Config-inventory tests repaired; all 177 MASSter PL tests pass. Package lint findings remain open.

## Scope

Reviewed HEAD `81c50c43` and the live working tree: MASSter hooks and dataset filtering,
model proposal sharing, rewire expansion and proposal isolation, channel selection,
spectrum emission, Lightning loss/metric integration, checkpoint migration and annotation
export. This is a targeted audit, not exhaustive verification of the native kernels or all
experiment configurations.

The working tree already contained the rewire prediction-row sorting fix in
`src/fragnnet/models/masster/masster_model.py`. Its regression test in
`tests/test_masster_rewire_model.py` appeared during the audit and was also exercised.
Neither change was authored by this audit. The sort keeps exact-mass, unmerged predictions
grouped by spectrum, as the vectorized CE loss requires; annotation rows follow the same
permutation. The mixed-batch versus singleton loss regression passes.

## Findings

1. **P2 — retired config inventories still break the MASSter PL test suite.**
   `tests/test_masster_pl.py:174` expects eight independent-distance configs, but only
   one remains. Line 197 expects three proposal-warmup configs, but none remain.
   Both tests fail before exercising model behavior. Replace the retired-file inventories
   with supported configuration fixtures and keep assertions of the actual runtime behavior.
   The previous project-state statement that the CPU suite passes does not describe this
   checkout's selected suite.
2. **P3 — the MASSter package fails its lint check.** Ruff reports unsorted imports in
   `masster_model.py:32` and `masster_pl.py:15`, an unused `n_atoms` assignment in
   `rewire_expansion.py:425`, and an unnecessary identity dictionary comprehension at
   `rewire_expansion.py:521`. These are maintenance defects; no runtime failure was
   demonstrated from them.

No further runtime correctness finding was confirmed in the inspected paths. Existing hub
descriptions of many changes as uncommitted are stale: the initial Git status contained only
the model change above, followed later by its test change.

## Validation

Interpreter: `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`, from the repository root.

- `-m pytest tests/test_masster_model.py tests/test_masster_pl.py tests/test_masster_rewire_model.py tests/test_masster_rewire_pl_export.py tests/test_masster_merge_same_mz_rows.py tests/test_masster_nominal_isotopes.py -q`
  — **418 passed, 2 failed, 10 skipped**. Failures are finding 1.
- `-m pytest tests/test_masster_candidates.py tests/test_masster_heads.py tests/test_masster_dataset.py tests/test_masster_backend_compatibility.py tests/test_masster_inference_backend.py tests/test_inference_masster_annotation.py tests/test_inference_masster_peak_aggregation.py tests/test_masster_rewire_expansion.py tests/test_masster_rewire_parity.py tests/test_masster_smarts_pathway.py tests/test_masster_intensity_reliability.py tests/test_masster_rt_ccs.py -q`
  — **312 passed, 5 skipped**.
- `-m ruff check src/fragnnet/models/masster --output-format concise`
  — **4 violations**, finding 2.

CUDA was unavailable to these processes. Native GPU correctness, performance, a real-checkpoint
inference run and scientific validation-set evaluation were not verified. The full repository
suite was not run. No implementation, test or configuration edits were made by this audit.

Next: repair the obsolete config tests and lint baseline, then validate the native GPU path
and real-checkpoint inference when the device is available.


## Unit-test repair (2026-10-16)

Finding 1 is repaired in `tests/test_masster_pl.py`. The retained merged JSS config is
resolved through `runner.load_config`, including the template's atom-decoder default.
The retired warm-up config inventory is replaced with parameterized runtime checks at
epochs 0, 14, 15 and 16: the shared encoder stays trainable and the intensity head becomes
trainable after the inclusive epoch-15 boundary. Existing direct-loss and warm-up gradient
and loss tests remain in place.

Validation with the configured interpreter: `-m pytest tests/test_masster_pl.py -q`
passed **177 tests**; `-m ruff check tests/test_masster_pl.py --output-format concise`
and `git diff --check` passed. The full repository suite was not rerun. No production code
or experiment configs changed. Finding 2 and GPU/real-checkpoint validation remain open.


## Batch and padding coverage (2026-10-16)

Added `tests/test_masster_batch_padding.py` with 14 cases covering:

- Three attention/feed-forward layers under induced, full and no-attention modes, with
  pre- and post-normalization. Unpadded references must match zero padding and large finite
  positive/negative padding values; real-token input gradients must match and padding
  gradients must be exactly zero.
- Shuffled candidate rows with per-spectrum counts 1, 3, 4, 5, 8 and 9. This spans shared
  buckets and power-of-two boundaries. Batched outputs and input gradients must match
  singleton references, with exactly zero input gradient in other spectra.
- End-to-end singleton versus mixed/permuted spectrum predictions for supported legacy and
  unique-fragment attention combinations. Inputs include a bondless sodium atom, ethanol,
  a larger aromatic molecule, both precursor types and distinct collision energies.

The deterministic comparisons disable dropout and use evaluation mode; the head tests
still exercise backward propagation. They do not claim training-mode batch invariance for
BatchNorm or stochastic regularizers, or validation of every optional attention bias.

Validation: `-m pytest tests/test_masster_batch_padding.py tests/test_masster_heads.py tests/test_masster_attention_and_budget.py tests/test_masster_rewire_model.py -q`
passed **168 tests, 3 skipped**, using the configured interpreter. Ruff on the new file
and `git diff --check` passed. No runtime defect was exposed. GPU checks remain unverified;
no production code changed.

## Speed audit of the current path (2026-10-16)

Inspected HEAD `b8546dd2` plus the live working tree. The four pre-existing modified files
are `candidates/rewire_null.py`, `candidates/rewire_rules.py`, `masster_pl.py` and
`tests/test_masster_rewire_pl_export.py`, under their current MASSter/test directories.
These supersede the initial checkout description above. No production changes were made
in this speed audit. Findings below identify concrete work, not measured current speedups.

Resolved `i1`, `i4` and `a0_rewire_after_selection_ob_obf` through `runner.load_config`:
all use CUDA candidates, FP32 with TF32 enabled, microbatch 512, loss chunks capped at 64,
induced candidate attention, ranked mask budget 4096, diagnostics/base-CE logging off,
and proposal sharing off. The largest historical switches are therefore already active.
See [earlier measurements](masster_training_speed_2026_09_17.md); their timings are not a
profile of this revision.

### Opportunities, in implementation order

1. **Rewire: remove avoidable device/host round-trips.**
   `rewire_expansion.py:424` downloads atom pointers and atomic numbers and compares
   downloaded bond topology with the host sidecar on every active forward. At line 592,
   per-spectrum ordinary-fragment counts go GPU → NumPy → GPU solely to compute
   `min(max_extra, floor(fraction * count + 1e-9))`. Keep that cap calculation on-device
   and reuse immutable host chemistry/validation from graph preparation. Preserve the
   float64 rounding and epsilon policy of `extra_limits`, including boundary cases.
   Additional per-chunk scalar checks and component-label convergence checks
   (`label_components`, line 132) synchronize CUDA. Use the existing asynchronous
   validation helper for suitable checks; convergence needs a separate kernel/design,
   not removal of its correctness condition. This affects rewire arms only.

2. **Rewire: avoid repeatedly sorting the entire growing product pool.**
   The local `absorb` at `rewire_expansion.py:485` first reduces each chunk and then
   concatenates it with the full accumulated pool and reduces again. With many chunks,
   early products are sorted repeatedly; continuation cuts add more absorbs. Benchmark
   a bounded hierarchical merge or a final reduction if memory permits. Keep exact atom-set
   deduplication, rule-indicator union, parent ranking and deterministic tie behavior.
   Small workloads may not benefit. The final extra cap is applied after validity and
   deduplication, so simply capping intermediate products is not equivalent.

3. **Base path: benchmark candidate packing and compilation of fixed-shape layers.**
   `_contextualize_by_molecule` (`masster_candidate_transformer.py:1159`) sorts rows,
   downloads counts, constructs host buckets and performs `isin`/`nonzero` compaction for
   every bucket, then launches each layer separately per bucket. A GPU-built layout or
   reuse of producer counts could reduce packing overhead. Compile the fixed-shape
   attention/feed-forward blocks separately before attempting whole-model compilation:
   native candidate calls, host lists and ragged outputs remain graph breaks. Attention
   already uses SDPA/`need_weights=False`, and validation already uses asynchronous CUDA
   assertions; neither should be proposed as a missing optimization. An earlier packing
   rewrite failed its end-to-end gate (linked speed finding), so repeat only with evidence
   from the current workload. Mixed precision needs compatibility work because the native
   CUDA candidate path currently requires FP32.

4. **Base path: reduce padding in the spectral CE loss.**
   `loss/cross_entropy.py:559` creates per-chunk `[spectra, max_true, max_pred]` matching
   tensors. `loss/padding.py:72` already adaptively splits oversized chunks, but it processes
   spectra in their original order. Internally grouping similar true/predicted peak counts
   could reduce padded work while preserving the training sampler, with the inverse
   permutation restoring per-spectrum outputs and tolerances. First measure padding waste
   and actual chunk count. Increasing `loss_batch_size` alone need not improve throughput.

5. **Conditional: proposal sharing needs a favorable batch, and is not a free switch.**
   `_shared_proposal_layout` (`masster_model.py:3414`) can share precursor encoding and
   construction across repeated groups, but itself builds/validates a GPU layout. Measure
   repeated-group multiplicity first. Current configs do not enable sharing; changing the
   sampler changes batch composition, and the PL shared-proposal loss merges targets by
   group (`masster_pl.py:670`). Treat enabling this as a matched training experiment,
   rather than a behavior-preserving speed repair.

The working-tree transfer optimization in `masster_pl.py:2650` already excludes molecule
binary bytes from Lightning's recursive device transfer. Preserve it; its comment's timing
was not independently benchmarked here. CPU candidate threading, cut-bond deduplication,
unique-fragment transformer tokens and persistent loader workers also already exist.

### Validation and next measurement

Configured interpreter: PyTorch `2.4.1+cu121`; `torch.cuda.is_available()` returned false.
`tests/test_masster_rewire_pl_export.py` and `tests/test_masster_batch_padding.py` passed
**49 tests** on CPU. These are regression checks, not performance evidence. No training,
checkpoint evaluation, GPU benchmark or full-suite run was performed.

Next: use the [full-training benchmark](../../../scripts/debug/bench/benchmark_masster_full_training.py)
on a supported GPU with fixed batches and source identity, compare base versus rewire stage
costs, and profile synchronized forward/backward plus optimizer throughput. Prioritize
the inexpensive rewire cap/host-metadata changes, then the measured dominant base stage.
Apply the [benchmark guide's](../../guides/benchmark_masster_cpu.md) end-to-end gain gates;
do not rank work by historical candidate timers alone.
