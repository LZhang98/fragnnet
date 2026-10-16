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
