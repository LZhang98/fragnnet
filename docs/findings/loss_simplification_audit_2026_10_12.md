# Shared loss simplification audit

> **Status:** FINDING · **Updated:** 2026-10-12
> First two fixes committed in `6b359293`. Follow-up audit reproduced additional edge-case defects below; simplifications remain proposed.

## Scope and conclusion

Reviewed all ten Python files in [loss](../../src/fragnnet/loss/), 3,446 lines including
comments and docstrings, plus their model callers and focused tests. The existing module
boundaries are useful. The best reduction comes from sharing repeated preparation inside
those modules, not another directory split or a generic loss framework.

The two correctness issues below were fixed before consolidation. Keep scientific fixes separate from
behavior-preserving cleanup. The sequential implementations are still selectable through
`loss_vectorized`; they are not dead code. Existing unrelated working-tree changes were
preserved. The project-state working-tree inventory was incomplete at audit entry (it
omitted plotting/utility changes and the MASSter audit).

## Correctness findings

### Hungarian cosine includes infeasible assignments

**Fixed in `6b359293`.** In the audited version of
[spectrum_distance.py](../../src/fragnnet/loss/spectrum_distance.py),
`sparse_cosine_distance_hungarian`, lines 311–322, invalid edges receive a negative score
before assignment, but the returned indices are used to gather the original probabilities
without checking feasibility again. Removing rows and columns with no matches does not
guarantee that the remaining graph has a complete feasible assignment.

CPU float32 reproduction: observed masses `[100, 100.001, 101]`, predicted masses
`[100, 101, 101.001]`, equal intensities, absolute tolerance `0.01`. The first two observed
peaks compete for one prediction; the last observed peak matches either of the other two.
Only two pairs are feasible. The audited function returned distance **0.0**, whereas the
maximum feasible cosine is **2/3**, so distance must be **1/3**. The extra assigned pair
also contributes an erroneous gradient. FraGNNet exposes this function through its
Hungarian cosine loss modes.

The fix filters assigned indices through the original match mask before summing
probabilities. Infeasible scores are zero rather than a huge negative sentinel, so they
contribute no cosine and do not swamp differences between feasible scores during assignment.
Regression tests cover this deficient graph with equal and unequal intensities, gradients
for a unique optimum, logarithmic distance, and no-match cases. The peak-list mass-Hungarian
matcher remains separate: it optimizes mass error, while this objective optimizes intensity products.

### Gaussian CE changes meaning when vectorization is enabled

**Fixed in `6b359293`.** In the audited version of
[cross_entropy.py](../../src/fragnnet/loss/cross_entropy.py), the sequential path treats
`tolerance_multiple == -1` as untruncated (lines 273–274), while the vectorized path feeds
the negative multiple into the matching predicate (lines 521–536). Every match is false.
With Gaussian renormalization disabled, one observed and predicted peak at mass 100,
unit intensity, sigma `0.01`, and predicted OOS log-probability `-2`, the outputs are:

| Path | In-support CE | OOS CE | Target OOS log mass |
| --- | --- | --- | --- |
| Sequential | -3.6862316 | 0 | finite log-zero sentinel |
| Vectorized | 0 | 2 | 0 |

Negative density CE is valid here. The discrepancy is the lost in-support gradient and
changed coverage. The template sets the multiple to `-1` and vectorization to `False`;
enabling vectorization alone can trigger it. This was already documented in
[test_reorg_losses.py](../../tests/test_reorg_losses.py), which overrides the multiple
to 1 rather than testing this case. It is not a newly introduced regression.

The vectorized fix uses every valid within-spectrum pair when the Gaussian multiple is
`-1`, retaining padding masks and the existing rejection of Gaussian renormalization in
this mode. Tests compare values and gradients with the sequential implementation across
absolute, relative, and per-spectrum tolerances and two chunk sizes.
The public four-result contract is retained: sequential CE computes actual conditional
OOS entropy, while vectorized CE currently returns zeros in that position. Consolidation
must not silently erase or change that difference.

## Follow-up audit of the remaining losses

Reviewed at `6b359293` after the first two fixes were committed. The index was empty
at entry; no source or test files changed in this follow-up. Numerical probes used CPU
float32. These findings are separate from the already-fixed negative tolerance multiple.

### Hungarian JSD loses scale invariance for small predicted mass

**Medium priority; selectable FraGNNet loss path.**
[spectrum_distance.py](../../src/fragnnet/loss/spectrum_distance.py), lines 396–402,
normalizes scalar Hungarian JSD with `sum.clamp(min=EPS)`, where `EPS=1e-7`.
For positive mass below that threshold the resulting vector does not sum to one, but
the final `1 - gain/log(2)` formula assumes normalized distributions.

Reproduction: both spectra have masses `[100, 200]` and relative intensities `[0.4, 0.6]`.
With predicted logs `log([0.4, 0.6])`, scalar JSD is approximately zero. Subtracting 30
from both predicted logs changes scalar JSD to **0.9999901**, while vectorized JSD stays
**0.0**. This should be a normalization-invariant comparison. FraGNNet's spectrum
probabilities include a predicted in-support mass factor, so low total mass is relevant,
though occurrence in trained models has not been measured.

Normalize nonempty spectra in log space and handle zero mass explicitly. Add scale
invariance and gradient-parity tests before consolidating the two Hungarian JSD paths.
The vectorized assignment helper also has its own normalization floor; simply replacing
the scalar function with a wrapper needs tests over tiny masses and ambiguous matches.

### Exact zero probabilities produce NaN entropy

**Medium priority for the public helpers; production incidence unverified.**
[entropy.py](../../src/fragnnet/loss/entropy.py), lines 24 and 69, evaluates
`exp(logp) * logp`. A legitimate zero probability represented by `logp=-inf` produces
`0 * -inf`. Marginal entropy for logs `[0, -inf]` returns **NaN** instead of zero and
has a NaN gradient at the zero entry. Conditional entropy with one marginal state and
the same two conditional logs also returns NaN. The analogous OOS term has the same risk.
The final range checks do not detect NaNs because comparisons with NaN are false.

Repository callers often use a finite `LOG_ZERO` sentinel, which does not reproduce
this failure. Define zero-probability handling explicitly, sanitize the log term before
multiplication, and test the backward pass as well as the value. Reuse that zero-safe
operation between marginal and conditional entropy instead of duplicating it.

### Absolute Gaussian CE ignores per-spectrum widths when vectorized

**Medium priority for an exposed API combination.**
[cross_entropy.py](../../src/fragnnet/loss/cross_entropy.py), line 575, uses the global
scalar Gaussian width in absolute mode even when `tol_per_sample` is supplied. Matching
uses the per-spectrum tolerance, and the sequential path uses it for both operations.

One matching observed/predicted peak at 100, unit intensity, `relative=False`, global
tolerance `0.01`, per-spectrum tolerance `[0.1]`, floor `[200]`, multiple 1, and no
renormalization gives CE **-1.3836465** sequentially versus **-3.6862316** vectorized.
The difference is `log(10)`, from using sigma 0.01 instead of 0.1. Existing instrument
tolerance tests exercise relative mode, so this is not evidence that current instrument
recipes are affected. Use the supplied width in absolute mode and test value/gradient
parity for both finite and untruncated windows.

### Lower priority helper contracts

- **Empty binned predictions return NaN.**
  [spectrum_distance.py](../../src/fragnnet/loss/spectrum_distance.py), line 227, connects
  a no-overlap result to gradients with `0 * mean(pred_logprobs)`. An empty prediction
  list has an undefined mean. A unit observed peak with empty prediction arrays returns
  NaN rather than a maximum distance or an explicit rejection. Define empty-spectrum
  behavior before using an empty-safe sum. Current binned model heads ordinarily emit
  predictions; no model-level failure was reproduced.
- **CE-pair eligibility relies on its sampler.**
  [peak_pair_bce.py](../../src/fragnnet/loss/peak_pair_bce.py), line 115, accepts equal
  or nonfinite endpoint CEs. Starting from `_ce_switch_inputs` in `tests/test_loss.py`,
  replacing CEs with either `[20, 20]` or `[20, NaN]` still produces losses
  `[0.2959593, 0.3162326]` and eligible counts `[1, 1]`. However, runner validation requires
  the CE-context sampler, which separates replicates/fallbacks and keeps at most one
  chunk of a context in each batch. This is a missing helper precondition or defensive
  guard, not a demonstrated defect in the prepared Stage A training recipe. Keep it
  lower priority than the numerical findings.

### Follow-up validation and cleanup order

Existing focused tests: **373 passed, 8 skipped** across `test_loss.py`,
`test_cross_jss_contrastive.py`, `test_masster_peak_list_distance.py`,
`test_fragnnet_formula_support.py`, `test_loss_audit_regressions.py`, and 12 selected
channel-target/support tests in `test_masster_pl.py`. CUDA was unavailable. The new
edge-case probes are not covered by those passing tests. No additional concrete defect
was established in peak-list distance, channel attribution, padding, or scalar penalties;
this is not a claim of exhaustive correctness.

Keep the cleanup order below: duplicate BCE grouping, compact bin preparation, then
padded layout reuse. Fix the three numerical findings with regression tests separately.
Avoid unifying normalization and empty-spectrum handling across objectives until their
contracts are explicit; those apparent duplicates currently encode different behavior.

## Simplifications in recommended order

### Share observed peak grouping between the BCE objectives

[peak_pair_bce.py](../../src/fragnnet/loss/peak_pair_bce.py), lines 56–103 and 259–305,
duplicates batch-to-index lists, tolerance matching, and observed-alias grouping with
ambiguity rejection. Extract small private functions in this module for index grouping
and observed groups. Keep CE-transition discovery and precursor-contender selection
separate: they have different eligibility rules and weighting.

This is the clearest bounded first cleanup. Preserve strict tolerance boundaries,
ambiguous-chain rejection, intensity-weighted group masses, CPU target discovery, and
prediction gradients. The existing BCE cases in `tests/test_loss.py` provide a starting
point for output and gradient equivalence checks.

### Compact pairwise bins once with inverse indices

[pairwise.py](../../src/fragnnet/loss/pairwise.py) repeats binning, normalization,
batch-offset removal, compact-bin lookup allocation, and chunk-size handling in its
four public functions. Replace the repeated `unique` plus max-sized lookup table with
`unique(..., return_inverse=True)`. For the cross-matrix objective, compact concatenated
true/predicted indices once and split the inverse indices afterward.

Share sparse bin preparation and the row-pointer preparation used by cosine and CE.
Keep L2 versus L1 normalization explicit and retain symmetric versus asymmetric block
loops. A universal callback-driven matrix engine would obscure these differences.

The JSS docstring currently claims it avoids the full `[B, K]` dense matrix, but lines
214–218 allocate exactly that matrix; chunking bounds the mixture blocks instead.
Correct the documentation and measure training memory before claiming savings:
autograd can retain tensors from multiple blocks until backward.

### Reuse padded layouts and broadcast spectrum tolerances

[cross_entropy.py](../../src/fragnnet/loss/cross_entropy.py), lines 378–518, and
[peak_list.py](../../src/fragnnet/loss/peak_list.py), lines 361–406, repeat counts,
cumulative boundaries, chunk slicing, and layout construction. Small helpers in the
existing [padding.py](../../src/fragnnet/loss/padding.py) can own this preparation while
leaving each objective visible in its own module.

CE expands spectrum tolerances into flat peak arrays, slices them, then pads them back
into spectrum rows. Broadcast the per-spectrum slice over padded rows, as peak-list
distance already does. Preserve observed-mass matching versus predicted-mass Gaussian
widths, relative/absolute behavior, dtypes, and padding masks.

The vectorized Hungarian JSD path builds the same layouts separately for m/z and
intensities (four `pad_flat_by_batch` calls per chunk). Leakage similarly builds the
prediction layout twice. Reuse one layout per side. Preserve CE/peak-list adaptive
chunk bounds; the other paths currently use fixed spectrum-count chunks. Reusing a
layout does not require changing their chunking policies.

### Simplify CE dispatch after specifying path parity

[get_sparse_cross_entropy_fn](../../src/fragnnet/loss/cross_entropy.py), lines 619–688,
has two nearly identical forwarding closures. Select the implementation and its fixed
keyword arguments once, then bind them with `functools.partial` or one small wrapper.
The current inner return annotation says `Tensor`, although four tensors are returned.
Repair the sequential placeholder docstrings and vectorized return documentation too.

Do not remove the sequential path during this cleanup. Beyond the reproduced negative
multiple discrepancy and OOS-entropy difference, the optional tolerance arguments have
different fallback handling and need explicit parity tests.

### Finish with small local reductions

- [channel_support.py](../../src/fragnnet/loss/channel_support.py): Gaussian posterior
  and Hungarian target builders repeat the same pair log-likelihood calculation. Share
  that expression; keep sparse candidate search, exact tolerance algebra, detached
  priors, and the two attribution rules intact.
- [peak_list.py](../../src/fragnnet/loss/peak_list.py): non-cosine objectives normalize
  aligned intensities, then repeat normalization for entropy mismatch. Normalize once
  for both consumers while leaving cosine's L2 calculation explicit.
- [entropy.py](../../src/fragnnet/loss/entropy.py): marginal and conditional entropy
  repeat the final clamp/rescale checks. A small helper is reasonable, but batch-size
  inference and empty-support conventions need a separate contract decision.
- The two Hungarian JSD functions accept `log_min` but do not use it. Document/deprecate
  that argument with compatibility in mind; the binned JSD function does use it.
- Leave the 28-line [penalty.py](../../src/fragnnet/loss/penalty.py) alone. It is simple,
  and rewriting it as a dispatch abstraction has little benefit.

## Validation and next step

Validation uses `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`.
Ruff on `src/fragnnet/loss` passed. The two numerical reproductions above ran on CPU
with float32; an initial float64 probe was rejected by the existing `LOG_ZERO` helper.
Focused tests: **440 passed, 20 skipped** in 179.62 seconds. CUDA was unavailable;
the run emitted NVML/autocast warnings. The command was:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m pytest \
  tests/test_loss.py tests/test_loss_sync_removal.py \
  tests/test_loss_cumsum_bincount.py tests/test_sparse_ce_peak_marginal.py \
  tests/test_per_inst_type_tolerance.py tests/test_cross_jss_contrastive.py \
  tests/test_masster_peak_list_distance.py tests/test_reorg_losses.py -q
```

These passing tests do not cover the two failing numerical expectations reproduced
above. New audit links resolve and `git diff --check` passes.

## Fix validation

[test_loss_audit_regressions.py](../../tests/test_loss_audit_regressions.py) adds 12 cases.
Before the fixes, 10 failed and the two no-match cases passed. All 12 pass after the fixes.
The existing reorganization fixture retains its positive tolerance multiple; its stale
explanation of the former negative-window bug has been corrected.

Post-fix validation: **191 passed, 12 skipped** across `test_loss_audit_regressions.py`,
`test_loss.py`, `test_loss_sync_removal.py`, `test_loss_cumsum_bincount.py`,
`test_sparse_ce_peak_marginal.py`, `test_per_inst_type_tolerance.py`, and
`test_reorg_losses.py`. Ruff on changed Python files and `git diff --check` pass.
CUDA was unavailable; checks ran on CPU with NVML/autocast warnings.

Function signatures, output shapes, configuration keys and model state-dict keys are
unchanged. Loss values and gradients intentionally change for forced infeasible Hungarian
assignments and vectorized untruncated Gaussian CE. Historical results using those paths
are not numerically interchangeable with corrected losses.

Next: deduplicate BCE grouping and pairwise preparation in separate changes.
For cleanup, compare outputs and prediction gradients, including ragged batches,
duplicate peaks, empty cases supported by each API, and tolerance-boundary cases.
Preserve signatures, configuration keys, logged values, and model state-dict keys.
No training, validation-set evaluation, full-suite run, or GPU memory benchmark was performed.

## Pair BCE retirement (2026-10-12)

The user explicitly requested removal of precursor BCE and CE-switch BCE after the usage
audit. Both implementations, training plumbing and experiment configs are removed.
Disabled legacy settings migrate away; enabled legacy objectives raise an explicit error.
The [consolidation plan](../plans/loss_consolidation_plan.md#phase-1-retire-precursor-and-ce-switch-bce)
records the scope and validation. Earlier BCE findings above are historical.
