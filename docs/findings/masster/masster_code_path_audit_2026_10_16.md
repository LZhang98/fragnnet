# MASSter current code-path audit

> **Status:** FINDING · **Updated:** 2026-10-16
> Selected CPU checks pass apart from two stale config-inventory tests; package lint has four violations. No additional runtime defect confirmed.

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
