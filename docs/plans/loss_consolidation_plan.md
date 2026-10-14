# Shared loss consolidation plan

> **Status:** LANDED · **Updated:** 2026-10-12
> Numerical fixes and consolidation committed in `1a63257b`; BCE retirement in `0bfe032b`. Follow-up file merge and renames are in the working tree ([below](#file-merge-and-renames-2026-10-12)).

## Objective and baseline

Make the loss package smaller and easier to review while keeping each objective's math
explicit. Keep the existing files and public entry points. Consolidation means fewer
copies of an operation, not moving the package back into one large file.

The [audit](../findings/loss_simplification_audit_2026_10_12.md) is the canonical source
for reproductions, limitations and existing test results. Hungarian cosine feasibility
and untruncated Gaussian CE were fixed in `6b359293`. The follow-up findings remain open.
The [architecture map](../architecture.md) and
[reorganization plan](repo_reorg_plan.md#phase-8--naming-and-model-packages-done-2026-10-12)
define package ownership. The earlier
direct-loss batching work (removed 2026-10-13; `git show e65339ed:docs/plans/masster/masster_direct_loss_batching_plan.md`) establishes
why preparation can be shared while MASSter peak-list and FraGNNet Hungarian objectives
must remain distinct.

At plan creation, loss source and tests match HEAD and the index is empty. Audit and
workstream documentation, plus unrelated files, have uncommitted changes. Preserve them.

## Compatibility constraints

- Preserve public import paths, signatures, tuple ordering, tensor shapes, dtype/device
  behavior, config keys/defaults, metric names and state-dict keys.
- Keep `loss_vectorized` and both selectable implementations. Sequential code is not
  dead code. The explicitly authorized BCE retirement below is the sole exception.
- Separate numerical fixes from equivalence refactors. Establish the corrected baseline
  before measuring equivalence; do not update golden expectations merely to pass tests.
- Preserve matching semantics: many-to-one closest, mass-Hungarian, cosine-Hungarian,
  JS-gain Hungarian, and all-in-window marginalization are different operations.
- Preserve exact tolerance boundaries, observed-mass versus predicted-mass divisors,
  coalescing order, normalization, OOS/null handling, escape mass and sample weighting.
- Preserve the four CE outputs. In particular, the sequential path computes conditional
  OOS entropy while the vectorized path returns zeros there. Resolving that difference
  is separate work; dispatch cleanup must preserve it.
- Keep existing chunk policies and limits, sparse candidate search, CPU target discovery,
  detached attribution priors and assignment indices. Do not introduce a dense whole-grid
  allocation or move target selection into the prediction autograd graph.
- Shared layout operations belong in `padding.py`; helpers used within one objective
  family stay private in that family's module. Reuse existing utilities where suitable.
  Keep the initializer empty of re-exports and core free of model imports.

## Target ownership

| Module | Consolidation target | Keep explicit |
| --- | --- | --- |
| `padding.py` | Counts, chunk boundaries and reusable flat-to-padded layouts | Caller-selected sorting and chunk policy |
| `cross_entropy.py` | Bound dispatch arguments and per-spectrum tolerance broadcasting | Gaussian density, peak marginalization and OOS reductions |
| `pairwise.py` | Binning/compaction and shared row-pointer preparation | L1/L2 normalization and symmetric/asymmetric block traversal |
| `spectrum_distance.py` | Reuse layouts for masses and intensities | Each distance and matching objective |
| `peak_list.py` | Reuse layouts and normalized distributions | Escape coordinate, KL smoothing and entropy mismatch |
| `channel_support.py` | Shared pair Gaussian log-likelihood | Posterior versus hard assignment and balanced BCE |
| `entropy.py` | Zero-safe entropy terms and repeated final scaling | Marginal versus conditional reductions |
| `penalty.py` | No planned change | Existing small scalar functions |

No new class hierarchy, registry, mutable context object or generic loss engine is needed.

## Phase 0 Establish numerical contracts

Turn the three follow-up numerical reproductions into failing unit tests, then fix each
in a separately reviewable change:

1. **Hungarian JSD:** normalize positive mass without an EPS floor changing its total.
   Test invariance under uniform log-intensity shifts, ambiguous matching, gradients and
   scalar/vectorized agreement. Audit the assignment helper's normalization too; fixing
   only the final distance normalizer does not guarantee assignment parity.
2. **Entropy:** make exact zero probabilities contribute zero before multiplication,
   including conditional and OOS terms. Verify finite values and backward behavior for
   finite sentinels and `-inf`; do not mask an already-created NaN after the fact.
3. **Absolute Gaussian widths:** use supplied per-spectrum widths consistently for
   matching and density. Test absolute/relative modes, scalar/per-spectrum tolerances,
   truncated/untruncated windows, and prediction gradients.

Before deduplicating affected branches, document their empty-input behavior and sorted
batch-index preconditions. Preserve existing supported behavior elsewhere. The lower-priority
empty-binned-prediction findings stay separate: propose an explicit
empty policy and test sampler guarantees before broadening those contracts. They do not
block unrelated bin compaction or grouping extraction.

**Gate:** new cases fail before the fixes and pass afterward; existing focused checks pass.
Record intentional numerical changes in the audit. No signature or config migration is needed.

## Phase 1 Retire precursor and CE-switch BCE

User authorized removal on 2026-10-12, superseding the proposed shared BCE helper.
Removed `peak_pair_bce.py`, both model integrations, logging and batch payloads,
runner validation, template keys and experiment arms/generator. Disabled legacy
settings are stripped during config/checkpoint loading; nonzero weights fail clearly.
CE attention bias and the CE-context sampler remain independent features.

Removed BCE-only tests; retained training-step coverage for the remaining auxiliaries.
Golden training records intentionally change because their enabled CE-switch term is gone.
Validation: 340 regression checks passed across loss training records, validation,
checkpoint/config migrations, registry, import boundaries and the new retirement tests.
The existing loss/runner/MASSter suite passed 306 tests; shared-objective and FraGNNet
audit tests passed 20. Ruff and `git diff --check` passed. Config YAML comparisons
verified that retained experiment files changed only by removal of retired keys.
CUDA and validation-set model evaluation were not run; no training jobs were submitted.

## Phase 2 Consolidate pairwise preparation

Replace `unique` plus max-sized bin lookup arrays with `unique(return_inverse=True)`.
For cross-JSS, compact concatenated true/predicted bin IDs once and split the inverse
indices. Share repeated sparse-bin preparation and the row pointers used by cosine and CE.
Keep normalization choices and matrix block loops visible in their public functions.
Correct the JSS memory docstring to describe its actual dense `[B, occupied_bins]` allocation.

**Gate:** dense-reference values and gradients agree across chunk sizes, duplicate peaks,
shuffled inputs, disjoint support, and non-default bin geometries. Verify CE asymmetry and
cosine/JSS symmetry. Avoid a callback framework for the distinct matrix calculations.

## Phase 3 Consolidate padding and tolerance preparation

First reuse existing layouts: build one per spectrum side per chunk, then apply it to
both masses and intensities in vectorized Hungarian JSD and leakage. Next factor the
repeated counts/cumulative-boundary preparation used by CE and peak-list distance into
`padding.py` only where the call sites have the same contract. Prefer a small function
returning named local values over a new container hierarchy.

Broadcast per-spectrum tolerances over padded rows instead of expanding them to flat
peaks, slicing and padding them again. Keep true-peak matching scales and predicted-peak
Gaussian widths separate. Preserve padding fill values and mask padded pairs explicitly.
Retain adaptive CE/peak-list chunk bounds and fixed-count chunking elsewhere.

**Gate:** values and gradients agree on ragged batches, mixed instruments, no-overlap cases,
empty rows where supported, and chunk-boundary changes. Existing synchronization-removal
and chunk-size tests still pass. Inspect for newly introduced device-to-host scalar reads.

## Phase 4 Simplify dispatch and local reductions

- In `get_sparse_cross_entropy_fn`, choose implementation and fixed keyword arguments
  once; bind with `functools.partial` or one forwarding wrapper. Preserve optional keyword
  arguments, error behavior and all four returns. Correct annotations and stale docstrings.
- In `channel_support.py`, compute pair Gaussian log-likelihood through one private helper
  shared by posterior and Hungarian attribution, retaining each reduction and detach rule.
- In `peak_list.py`, normalize aligned intensities once for the non-cosine objective and
  entropy mismatch. Cosine retains its L2 calculation; entropy still uses L1 distributions.
- In `entropy.py`, share final clamp/rescale checks after Phase 0 establishes the numerical
  contract. Keep support-size conventions and batch-size inference explicit.

**Gate:** focused output/gradient comparisons and enabled-loss integration tests pass.
Do not remove unused public parameters such as Hungarian JSD's `log_min` in this phase;
document its current behavior and leave deprecation to a separate compatibility change.

## Validation and acceptance

Use the configured interpreter and commands in the [session workflow](../workflow/session.md).
Build independent expected values for mathematical regressions. For pure refactors,
compare against the corrected pre-refactor implementation on deterministic inputs, then
retain compact invariant tests rather than a permanent duplicate implementation.

| Area | Existing test starting points |
| --- | --- |
| Numerical fixes and retired BCE compatibility | `test_loss_audit_regressions.py`, `test_loss.py`, `test_retired_pair_bce.py` |
| CE and padding | `test_sparse_ce_peak_marginal.py`, `test_loss_sync_removal.py`, `test_loss_cumsum_bincount.py`, `test_per_inst_type_tolerance.py` |
| Pairwise | `test_loss.py`, `test_cross_jss_contrastive.py` |
| Peak-list objectives | `test_masster_peak_list_distance.py` |
| Channel support | Relevant cases in `test_masster_pl.py`, `test_fragnnet_formula_support.py` |
| Integration and boundaries | `test_reorg_losses.py`, `test_reorg_boundary.py`, `test_reorg_lazy_exports.py` |

Run Ruff on changed files and `git diff --check` after each phase. Run the complete suite
before a PR when practical; classify pre-existing failures separately. Test CPU and CUDA
when available, including supported mixed-precision paths. CUDA absence is an unverified
GPU result, not evidence of equivalence or memory savings.

For preparation changes, compare forward-plus-backward time and peak allocated memory
on the same device using representative ragged synthetic inputs, fixed seeds and warm-up.
Include a long-spectrum tail and both assignment backends when available. Investigate
repeatable regressions before accepting a refactor; preserve memory bounds even if a
shorter expression is available. No training or scheduler submission is part of this plan.

Completion means the listed duplicate operations have one implementation, public contracts
remain compatible, numerical fixes have regression coverage, and focused/integration checks
pass. Record before/after production line counts and helper counts; do not set an arbitrary
line-reduction quota or claim a speedup without measurement. Unmeasured GPU behavior and
any deferred contract decisions must be explicit in the handoff.

## Completion and validation (2026-10-12)

Phases 0, 2, 3 and 4 are implemented. Phase 1 was committed separately in `0bfe032b`.
The loss package keeps its public signatures, chunk policies, normalization choices,
matching rules and model/config interfaces. Remaining unrelated numerical contracts
(empty binned predictions and sequential/vectorized OOS entropy differences) are deferred.

- **Numerical fixes:** normalize Hungarian JSD by positive mass, use a benign denominator
  for zero mass (including the JS-gain assignment helper), replace `log(0)` before entropy
  multiplication, and honor absolute per-spectrum Gaussian widths. Sequential OOS entropy
  also handles exact-zero target peaks without `-inf * 0`.
- **Pairwise:** one sparse bin-preparation path; compact with inverse indices; share sorted
  rows and row pointers between cosine and CE. Cross-JSS compacts both supports together.
- **Padding:** paired counts and boundaries transfer to CPU once. JSD and leakage reuse
  layouts; CE broadcasts row tolerances instead of gathering/slicing/padding peak arrays.
  Preserve the historical within-row argsort order: skipping it changed float32 reductions
  and optimizer snapshots despite equivalent mathematics.
- **Local reductions:** one CE forwarding closure, one channel Gaussian likelihood helper,
  one L1 normalization for peak-list objectives and entropy, one entropy rescaling helper.
  JSS memory documentation and unused Hungarian `log_min` documentation are corrected.

Loss-package source shrank from **3,085 to 3,003 lines**; private helper count rose
from 10 to 16 as repeated preparation moved into small functions. The shared assignment
helper also changed in `utils/spec_similarity.py`.

### Tests

Two new files contain **230 parameterized cases** (229 CPU passes and one CUDA-only skip):
`tests/test_loss_numerical_contracts.py` and `tests/test_loss_consolidation.py`.
They cover independent dense values and gradients, exhaustive feasible JS assignments,
independent intensity shifts, exact zeros and `-inf`, conditional/OOS entropy, Gaussian
widths and truncation, duplicate peaks, shuffled inputs, non-default bin geometry,
ragged/empty rows, chunk boundaries, CE asymmetry, cosine/JSS symmetry, escape mass,
and posterior/Hungarian channel targets with detached priors.

Against the saved pre-fix implementation, the 76 numerical cases produced **27 failures
and 49 passes**; all 76 pass after the fixes. This includes a constrained assignment where
an EPS normalization floor selected the wrong feasible permutation.

Focused loss/model validation: **834 passed, 21 skipped** (before the final six assignment
cases were added); those six also passed. All **17 training-step golden tests pass unchanged**.
No golden files were regenerated. The additional run covering spectrum utilities, shared objectives and both new test
files passed **284 tests with one CUDA skip** (overlaps the first run). All **5,753 repository tests collect successfully**; the full suite
was not executed. Ruff on the changed source/tests and `git diff --check` pass.
CUDA/NVML is unavailable, so GPU values, mixed-precision GPU execution and peak allocated
GPU memory remain unverified. No validation-set training/evaluation or cluster job ran.

### Bounded CPU timing probe

One thread, float32, seed 73, eight spectra with peak counts `[8,32,5,96,11,24,7,48]`,
chunk size 3, three warmups and 20 forward-plus-backward repetitions. Compare the corrected
pre-refactor implementations with the new code; all seven cases had exactly equal outputs
and prediction gradients. These one-off measurements are checks for gross regressions,
not a general speed claim. Autograd memory was not profiled.

| Path | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Pairwise cosine | 1.674 | 1.684 |
| Pairwise JSS | 2.248 | 2.149 |
| Pairwise CE | 2.194 | 2.162 |
| Gaussian CE | 6.225 | 6.121 |
| Leakage | 2.404 | 2.158 |
| Hungarian JSD | 6.541 | 6.127 |
| Peak-list Hellinger plus entropy | 8.525 | 7.706 |

Committed in `1a63257b`. **Next:** GPU validation when available.

## File merge and renames (2026-10-12)

The user asked to merge files and improve names, superseding "keep the existing files" above.
Nine modules became six; no function body changed. The changes are uncommitted.

| Before | After | Why |
| --- | --- | --- |
| `entropy.py` | `cross_entropy.py` | CE imported the private `_plogp` from it. |
| `peak_list.py` | `spectrum_distance.py` | Both compare a pair of sparse spectra using the same matching, alignment and padding helpers. |
| `penalty.py` (`get_edge_loss_fn`) | `models/fragnnet/fragnnet_pl.py` | FraGNNet-only; its single caller. |

Renamed functions whose name said "factory" or "getter" but which return a value:

| Old | New |
| --- | --- |
| `sparse_entropy_fn`, `sparse_conditional_entropy_fn` | `sparse_entropy`, `sparse_conditional_entropy` |
| `get_pairwise_cossim`, `get_pairwise_jss_sim`, `get_pairwise_cross_entropy` | `pairwise_cosine_similarity`, `pairwise_jss_similarity`, `pairwise_cross_entropy` |
| `get_cross_jss_contrastive` | `cross_jss_contrastive_loss` |
| `mog_ce_fn`, `pm_ce_fn` | `gaussian_ce`, `peak_marginal_ce` (the `dist` values that select them) |

The factories `get_sparse_cross_entropy_fn` and `get_edge_loss_fn` keep their names. Config keys,
loss and metric names and state-dict keys are unchanged; checkpoints store no loss-module paths.
Importers in `src/`, `scripts/` and `tests/` and the live reference docs were rewritten; frozen
findings keep their old paths (see the documentation rules).

Validation (CPU): the 55 test files that touch the loss package, the Lightning modules or the
reorg boundaries gave **2,115 passed, 8 failed** before and after the change. The 8 failures are
in `tests/test_export_schema.py`, from the uncommitted MASSter export work, and were identical
in both runs. Ruff passes on the changed files.
