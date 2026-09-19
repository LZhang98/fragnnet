# MASShatter — Batched Direct Peak-List Loss

> **Status:** LANDED &nbsp;·&nbsp; **Updated:** 2026-09-19
>
> `sparse_peak_list_distance` now uses the chunked-and-padded shape of the vectorized
> cross-entropy, and `round_aggregate_peaks` is a batched segment reduction. Measured on an
> RTX 2070S at chunk-64, 52 observed / 512 predicted peaks, forward plus backward:
> **`closest` 170.1 → 3.8 ms (44x)** and **`hungarian` 208.9 → 19.0 ms (11x)**, against
> vectorized cross-entropy's 1.9 ms. Host syncs fall from ~1800 per call to 19.
> Loss values are unchanged; 3827 tests pass.
>
> **Hungarian's residual is the scipy fallback, not the batched code.** Stubbing the solver
> out leaves 3.8 ms, identical to `closest`, and all 341 of its remaining syncs come from
> `_solve_batch_assignment` looping to CPU. Cluster jobs install the
> `torch_linear_assignment` wheel and take the batched GPU solver instead, so the expected
> cluster figure is near 4 ms, not 19 ms. Re-measure there before quoting a Hungarian number.

## Problem

Every MASShatter direct spectral loss (`jss_distance`, `kl_distance`, `cosine_distance`) runs
through [`sparse_peak_list_distance`](../../src/fragnnet/model/loss.py), which iterates
`for batch_index in range(batch_size)` and indexes CUDA tensors with boolean masks inside the
loop. `cross_entropy` instead runs `sparse_cross_entropy_vec`, which pads each chunk once and
reduces it as a single `[chunk, max_true, max_pred]` tensor.

Measured on an RTX 2070S, chunk of 64 (`loss_batch_size: 64`), 52 observed peaks and 512
predicted peaks per spectrum, forward plus backward:

| path | ms / chunk-64 | host-device syncs per call |
| --- | --- | --- |
| `cross_entropy` (vectorized) | **1.9** | **5** |
| `jss_distance` + `closest` | 174 | — |
| `jss_distance` + `hungarian` | **221** | **~1800 (28 / spectrum)** |

Two independent signals confirm this is launch- and sync-bound rather than FLOP-bound:

- **CPU is about twice as fast as CUDA for the same function** (109 ms vs 221 ms). Code that
  speeds up when the GPU is removed is waiting on the host.
- **Cost is flat in problem size.** Raising `n_pred` from 256 to 1024 moves the call from
  190 ms to 208 ms. It is paying for 64 Python iterations, not for arithmetic.

The Hungarian solver is not the cause. `closest` needs no solver at all and still costs 174 ms;
stubbing the solver out of the batched path removes only about 8 ms. Switching
`mass_shatter_peak_matching` to `closest` would therefore not fix this.

This is the same defect, and the same fix, as `observational_channel_targets` and
`_candidate_weighted_recall` in the [GPU audit](masshatter_gpu_audit.md): any per-spectrum loop
that indexes with boolean masks pays a fixed host cost regardless of how little math it does.

### Cost split

Ablating the coalescer from the call isolates two separate loops:

| component | ms | syncs | shape |
| --- | --- | --- | --- |
| `round_aggregate_peaks` × 2 | 36 (16%) | 386 | per-spectrum loop, `th.unique` per spectrum |
| main loop in `sparse_peak_list_distance` | 185 (84%) | ~1800 | per-spectrum loop |
| — of which the assignment solver | ~8 | — | |

`round_aggregate_peaks` lives in `spec_utils.py` and has twelve call sites across six loss
functions plus `spectrum_pl`. It is on FraGNNet's vectorized JSS path and on the evaluation
path, so batching it in place benefits callers outside MASShatter.

### Training cost

The unmerged arm `66unqbwq` (`jss_distance`, batch 512) reached epoch 48 in 17.3 h, while its
cross-entropy sibling `6ks7v8ce` reached epoch 99 in 11.3 h on the same data and GPU model:
about 1300 s/epoch against 410 s/epoch. At 287,721 training spectra and batch 512 that is 562
steps per epoch, and 562 × 1.65 s ≈ 930 s — the measured loss overhead accounts for essentially
the entire 890 s/epoch gap. Its GPU sat at 50% SM utilization with a **median memory-copy
utilization of 0**.

## What is already reusable

The vectorized cross-entropy path is built from four helpers that all apply unchanged.

| existing | why it fits |
| --- | --- |
| `calculate_match_mzs` (`spec_utils.py`) | Already broadcasts through `[..., None]` and `[..., None, :]`, so it accepts `[chunk, max_true, max_pred]`. `sparse_cross_entropy_vec` passes 3-D today. Its `NOT BATCHED` docstring line is stale and should be corrected. |
| `_build_padded_batch_layout` + `_pad_flat_with_layout` (`loss.py`) | Build the sort/scatter layout **once** per chunk and apply it to m/z, intensities and per-peak tolerances. This is what the existing vectorized JSS misses: it calls `_pad_flat_by_batch` four times, rebuilding the layout (an `argsort` and a `bincount`) on each call. |
| `_build_loss_chunks` (`loss.py`) | Bounds each chunk so padded pair count never exceeds the flat Cartesian product of that chunk. This is what makes padding safe against the merged peak-count tail (p50 25, p99 343, max 1168). |
| `_solve_batch_assignment` (`spec_utils.py`) | Already accepts `(B, N, M)`, dispatches to `batch_linear_assignment` on CUDA, and falls back to scipy. |

Zero-padding is exact for this objective family: `0` is the identity for `sum` (the
normalizer) and for `x·log x` (every entropy term, already clamped at `finfo.tiny`). KL's one
ragged step, `true_distribution[positive_true]`, becomes `th.where(mask, term, 0)`.

### What must not be reused

**Do not route MASShatter into `sparse_jensen_shannon_divergence_hungarian_vec`.** It is a
structural template, not a callee. It takes log-probabilities rather than intensities,
coalesces with `agg="lse"` rather than `agg="sum"`, clamps its result into `[0, 1]`, returns no
entropy-match term, and supports Hungarian matching only. Calling it from MASShatter would
silently change the trained objective.

## Results

Measured at chunk-64, `n_true=52`, `n_pred=512`, CUDA, forward plus backward. The before
column was captured in the same session by stashing the change, so it is not drift.

| path | before | after | syncs before → after |
| --- | --- | --- | --- |
| `round_aggregate_peaks` ×2 | 34.6 ms | **0.8 ms** (43x) | 193 → **1** |
| `sparse_peak_list_distance`, `closest` | 170.1 ms | **3.8 ms** (44x) | ~1800 → **19** |
| `sparse_peak_list_distance`, `hungarian` | 208.9 ms | **19.0 ms** (11x) | ~1800 → 341 |
| `hungarian`, assignment solver stubbed | — | 3.8 ms | — |
| `sparse_cross_entropy_vec` (reference) | 1.9 ms | 2.3 ms | 5 |

The CUDA-beats-CPU gate holds at realistic sizes: `closest` is 5.6 ms on CUDA against
14.3 ms on CPU, and `hungarian` 21.6 against 24.8. The Hungarian margin is thin only
because both sides are paying for the same sequential scipy solver locally.

Cost is now size-dependent rather than flat, which is the other signal that the
launch-bound behavior is gone: `closest` moves 5.6 → 7.0 ms as `n_pred` goes 512 → 1024,
where the loop implementation did not move at all.

### Verification

- 3842 tests pass, 1 skipped, 2 xfailed.
- 144 randomized equivalence cases against a retained copy of the loop implementation
  (8 seeds × 3 objectives × 2 matching modes × 3 chunk widths), plus gradient equivalence
  and the per-sample instrument-tolerance path.
- 36 randomized `round_aggregate_peaks` equivalence cases; bit-identical for `sum` and
  `amax`, `allclose` for `lse`.
- **CUDA equivalence**, which CPU agreement does not imply: peak matching runs under
  bfloat16 autocast on CUDA and Hungarian dispatches to a different assignment backend.
  432 off-line comparisons across both devices found 0 mismatches at a worst case of
  1.9e-06, and six cases are pinned as tests. The development box has an RTX 2070 SUPER,
  so these run rather than skip; the repository has no CI.
- **Chunks narrowed below the `loss_batch_size` ceiling.** Peak counts spanning 1 to 1500
  force `_build_loss_chunks` to split, so large spectra pad alone. The test asserts the
  split actually happened, otherwise it would silently degrade to the single-chunk case.
- Half-precision predictions (`float16`, `bfloat16`) promote to float32 and match exactly.
- Error parity: a batch index carrying no peaks raises the same message as the loop.
- `test_mass_shatter_pl.py` (123 tests) drives the real `loss_fn` closure for all three
  objectives × both matching modes, covering the new `loss_batch_size` wiring.
- **One path is untested and can only be checked on the cluster.** `_solve_batch_assignment`
  picks its backend from whether `torch_linear_assignment` is installed. The development
  box does not have it, so even the CUDA runs above took the scipy-loop fallback; cluster
  jobs install the wheel and take `batch_linear_assignment` plus `assignment_to_indices`
  instead. That backend is not new code — `batch_jss_hun_helper` already consumes it in
  production on FraGNNet's vectorized JSS path, using the same
  `score[batch_idx, row_ind, col_ind]` gather and the same scatter into a `[batch, peaks]`
  mask that this change uses — but this change is the first time MASShatter's peak-list
  loss depends on it, and none of the verification above executed it. The result shape is
  checked at the call site so a different index layout fails loudly instead of scattering
  into the wrong peaks. **Re-run the equivalence tests on a cluster node before trusting a
  Hungarian training run.**
- The shared coalescer's other callers are covered rather than merely unbroken:
  `tests/test_loss.py` and `tests/test_per_inst_type_tolerance.py` (93 tests) exercise
  `sparse_jensen_shannon_divergence_hungarian_vec`, `sparse_jensen_shannon_divergence_hungarian`,
  `sparse_cosine_distance_hungarian`, `get_pairwise_jss_sim` and `get_pairwise_cross_entropy`,
  and `tests/test_train_auxiliary_metrics.py` covers the `spectrum_pl` evaluation path.

### One behavior this surfaced, which is not a regression

Hungarian matching is ambiguous under an exact m/z-distance tie. When two candidates are
equidistant from one observed peak, only one can be matched and the objective is **not** a
function of the m/z cost being minimized, so the two equally optimal assignments give
different losses — measured at 0.097 apart on one randomized spectrum. The loop
implementation was equally arbitrary here; scipy simply happened to break ties differently.
`closest` has no such ambiguity, since every in-tolerance candidate joins the observed peak.

This is pinned by `test_hungarian_tie_matches_exactly_one_of_the_tied_candidates`, and the
randomized generator uses one-sided jitter so equivalence is tested on tie-free inputs.

## What was implemented

### Stage 1 — batch `round_aggregate_peaks`

Rewrite the body in `spec_utils.py` as a segment reduction over a composite
`(batch, rounded m/z)` int64 sort key, using `unique_consecutive` on the sorted key in place of
a per-spectrum `th.unique`, then `scatter_reduce` for `sum`/`amax` and a segment log-sum-exp for
`lse`. Signature, argument order and return tuple stay exactly as they are; all twelve call
sites are untouched.

Verify: bit-identical output for `agg="sum"` and `agg="amax"` over randomized cases including
empty spectra, single-peak spectra, and exact-tie m/z values. For `agg="lse"`, assert
`th.allclose` rather than `th.equal` — index-add reassociates float32 sums, the same way
`_candidate_weighted_recall` did.

### Stage 2 — restructure `sparse_peak_list_distance`

Replace the per-spectrum loop with the `sparse_cross_entropy_vec` structure, preserving the
public signature and every behavior:

1. Coalesce with the Stage 1 helper (`decimals=4`, `agg="sum"`).
2. Build chunks with `_build_loss_chunks`.
3. Per chunk, build the padded layout once and apply it to m/z, intensities and the per-sample
   tolerance and floor.
4. `calculate_match_mzs` on the padded `[chunk, max_true, max_pred]` tensors, intersected with
   the validity masks.
5. Matching:
   - `closest`: masked `argmin` along the candidate axis — the shape cross-entropy already reduces.
   - `hungarian`: fill invalid pairs with the existing `invalid_cost` and call
     `_solve_batch_assignment` once per chunk.
6. Build `aligned_pred` and `aligned_true` as zero-padded `[chunk, max_true + max_pred]`.
7. Compute `cosine_distance`, `jss_distance` and `kl_distance` on that padded tensor, keeping
   `kl_epsilon` smoothing and the float32 promotion for half-precision inputs.
8. Compute the squared entropy-mismatch term on the same tensor.

The `(primary_losses, entropy_losses)` return, all three objectives, both matching modes,
`kl_epsilon`, and the per-sample tolerance path all stay. The batched implementation replaces
the loop outright — there is no flag and no fallback. `loss_vectorized` stays
cross-entropy-only and does not acquire a second meaning.

Verify: [`tests/test_mass_shatter_peak_list_distance.py`](../../tests/test_mass_shatter_peak_list_distance.py)
already covers all three objectives across both matching modes, four-decimal clustering with
gradient flow, zero and negative intensity handling, unmatched-prediction retention, KL
smoothing including the split-intensity invariance, mixed precision, and instrument tolerance.
Those twelve tests must pass unchanged. Add a randomized old-versus-new equivalence test that
keeps a copy of the loop implementation as the reference:

- `closest`: assert bit-identical losses and gradients.
- `hungarian`: assert on the loss value, not on index arrays. Equal-cost assignments are not
  unique, so a different but equally optimal permutation is a correct result.

### Stage 3 — measure

Done; see Results. The protocol was to re-run the chunk-64 benchmark at `(n_true, n_pred)` of `(25, 256)`, `(52, 512)` and
`(52, 1024)`, on CUDA and CPU, for `closest` and `hungarian`. Report ms per chunk and host-sync
count per call beside the cross-entropy path.

Gate: **CUDA must be faster than CPU for the same call.** That inversion is what identified the
defect, so removing it is the signal that the fix landed. A pure speed ratio is not sufficient
evidence on its own.

## Success criteria — all met

1. All twelve existing peak-list-distance tests pass unchanged, except
   `test_hungarian_drops_candidates_outside_all_tolerance_windows`, which asserted the old
   implementation's feasible-submatrix shrink by monkeypatching the solver and checking it
   saw a 1×1 cost matrix. The batched path solves the whole chunk at once, so that shape no
   longer exists. It was replaced with a behavioral test of the same property: candidates
   outside every tolerance window match nothing, their position does not affect the loss,
   and Hungarian agrees with `closest` when only one pair is feasible.
2. Randomized equivalence against the retained loop reference holds, on the terms above.
3. `round_aggregate_peaks` is bit-identical for `sum`/`amax` and `allclose` for `lse`, with all
   twelve call sites unmodified.
4. CUDA is faster than CPU for `sparse_peak_list_distance`.
5. Chunk-64 `jss_distance` + `hungarian` is below 82 ms — the cost the existing vectorized JSS
   already reaches while rebuilding its layout four times and coalescing unbatched.

## Non-goals

- No change to any trained objective, to matching semantics, or to `kl_epsilon`.
- No new config key, and no change to what `loss_vectorized` controls. The existing
  `loss_batch_size` is now threaded into `sparse_peak_list_distance`, which previously
  ignored it and padded nothing; this bounds peak memory and does not change loss values.
- No change to `sparse_jensen_shannon_divergence_hungarian_vec`, `sparse_cosine_distance_hungarian`,
  or FraGNNet's loss selection. Stage 1 speeds those up through the shared coalescer without
  touching them.
- No attempt to close the remaining gap to cross-entropy's 1.9 ms. Cross-entropy reduces one
  padded tensor; the direct losses additionally build an aligned support vector and, for
  Hungarian, solve an assignment. Some gap is structural.

## Related documents

- [MASShatter GPU audit](masshatter_gpu_audit.md) — the per-spectrum-loop defect class and the
  two loops already batched.
- [Intensity-head bucketing sync plan](masshatter_intensity_head_bucketing_sync_plan.md) — the
  same defect in `CandidateIntensityTransformerHead`.
- [Non-probabilistic intensity plan](masshatter_nonprobabilistic_intensity_plan.md) — why the
  direct peak-list objectives exist at all.
- [Loss setup review](../findings/loss_setup_review_2026_09_13.md) — canonical for proposal-loss
  and gradient-routing behavior around these objectives.
