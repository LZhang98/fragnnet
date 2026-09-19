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

## Cluster validation runs — four JSS/KL arms on `fir:h100-2g.20gb`

> **Status:** CONFIGS WRITTEN AND VERIFIED LOCALLY · **Nothing generated or submitted on Fir.**

Four one-hour training arms that exercise the batched path in a real run and, for the first time,
execute `_solve_batch_assignment`'s `torch_linear_assignment` branch. The generated job installs
that wheel (`setup_job_cc.py:396`), so a cluster job reaches the backend a local CUDA box cannot.

| Job name | Config suffix | `loss_type` | `mass_shatter_peak_matching` |
| --- | --- | --- | --- |
| `ms_loss_jss_closest` | `_jss_closest` | `jss_distance` | `closest` |
| `ms_loss_jss_hun` | `_jss_hungarian` | `jss_distance` | `hungarian` |
| `ms_loss_kl_closest` | `_kl_closest` | `kl_distance` | `closest` |
| `ms_loss_kl_hun` | `_kl_hungarian` | `kl_distance` | `hungarian` |

All four share the stem `config/masshatter/ma_mi_merged/nist20_ma_mi_merged_fast_batch256x2_cuda_lb32`
and are derived from `..._ctrl_2g.yml` by textual substitution, so the model, data, shape and
instrumentation keys are byte-identical to that control. Verified after generation: each resolves
through `load_config` against the current `config/template.yml`, all five configs carry the **same
416-key set**, and each arm's resolved diff against the control is exactly `loss_type`,
`mass_shatter_peak_matching` (where it leaves the `closest` default) and the three `wandb_*`
identity keys — nothing else.

`mass_shatter_intensity_output` stays `probability`. The code under test is reached identically
under `independent`, and keeping `probability` makes the existing CE control a same-shape
throughput baseline instead of the six-key CE-vs-JSS bundle described in
[the loss-setup review](../findings/loss_setup_review_2026_09_13.md).

`mass_shatter_peak_matching` is written explicitly in all four arms even though `closest` is the
template default, so the four files share one key set and a future default change cannot silently
re-point two of them.

### The control is `ms_reg_ctrl_2g`, and it is not optional

The comparison that makes these runs worth an hour each is against
`..._batch256x2_cuda_lb32_ctrl_2g.yml` on the same slice — the
[regularization sweep's](masshatter_regularization_sweep_plan.md#stage-a--paired-slice-pilot-2-jobs-1-h-wall-clock-958-units)
Stage A 2g pilot. Run it in the same batch if it has not already run on 2g; its `ep/h` is the
denominator for every throughput claim below. A 3g control cannot substitute.

### Generation

Run on the **Fir login node**, from the Fir checkout, with the cluster's `python` — not
`FRAGNNET_PYTHON`. Generates only. The generated scripts embed a resolved W&B credential, so keep
them private.

```bash
FRAGNNET_ACCOUNT=rrg-dwishart
FRAGNNET_EMAIL=fw4@ualberta.ca
CFG=config/masshatter/ma_mi_merged/nist20_ma_mi_merged_fast_batch256x2_cuda_lb32

for arm in jss_closest:jss_closest jss_hungarian:jss_hun kl_closest:kl_closest kl_hungarian:kl_hun; do
  suffix="${arm%%:*}"; name="ms_loss_${arm##*:}"
  python scripts/setup_job_cc.py \
    --frag_gnn_dp . \
    --python_script_fp scripts/run_pl_model_fit.py \
    --job_name "$name" \
    --gpu_profile fir:h100-2g.20gb \
    --cc_account "$FRAGNNET_ACCOUNT" \
    --cc_email "$FRAGNNET_EMAIL" \
    --time 1:00:00 \
    --script_kwargs w=online c="${CFG}_${suffix}.yml" j="$name" \
    --no_submit=True
  bash -n "slurm_scripts/${name}.sh"
done
```

No launch overrides. `loss_batch_size: 32` and `num_workers: 3` are already in the configs, unlike
the regularization sweep where they are passed at launch. `w=online` is required for `j=` to work
at all — the resume bookkeeping lives inside the W&B branch.

Expected directives: `--mem=70G --cpus-per-task=3 --time=1:00:00 --gres=gpu:h100_2g.20gb:1`.
Check them, and the staging paths and final training command, before submitting.

### Submission

```bash
for name in ms_loss_jss_closest ms_loss_jss_hun ms_loss_kl_closest ms_loss_kl_hun; do
  sbatch "slurm_scripts/${name}.sh"
done
```

Four arms at 3.48 price units each is 13.9 units for the set, plus 3.48 for the 2g control.

### What each gate reads

1. **The jobs start and stay up.** A wrong index layout out of `batch_linear_assignment` fails
   loudly at the call-site shape check rather than scattering into the wrong peaks, so a hungarian
   arm that survives its first steps has executed the backend correctly at least structurally.
2. **`closest` is the reference for `hungarian`, per objective.** This is the main correctness
   read, and the reason the sweep is 2x2 rather than two arms. The two matchings are different
   objectives and are *not* expected to give equal losses — but they are expected to track. A
   hungarian arm whose training loss sits far off its `closest` sibling, or that diverges while the
   sibling trains, indicts the solver backend, since that is the only code the pair does not share.
3. **Throughput, against `ms_reg_ctrl_2g` at matched epoch.** The claim under test is that the
   1.46x `ep/h` penalty (CE 20.5-20.8 vs JSS 14.0-14.7, KL 13.8-14.8, measured pre-refactor) is
   gone. Read `ep/h` on the 2g slice; do not compare to the L40S numbers those figures came from.
4. **`hungarian` against `closest` on time.** Locally hungarian is 19.0 ms against 3.8 ms, entirely
   the scipy fallback. With the wheel installed the expected cluster figure is **near 4 ms, not
   19 ms**, so the two matchings should be close in `ep/h`. A large residual gap means the wheel did
   not install or the GPU solver was not taken — check the job's pip output before reading anything
   else into it.
5. **`mass_shatter_candidate_backend_resolved == cuda`** in each run's saved config, read from the
   run and not from the file.
6. **Peak GPU memory.** `256 x lb32` measured 12.7 GiB under CE. The direct losses add
   `[chunk, max_true + max_pred]` aligned tensors, which are linear rather than quadratic, and
   `_build_loss_chunks` bounds the pairwise tensors to the same flat Cartesian product CE already
   runs under — so the expectation is CE plus a modest margin, inside the ~19.5 GiB slice. The
   hungarian arms additionally carry `batch_linear_assignment`'s own workspace, which has never
   been measured here. **On OOM, drop `loss_batch_size` to 16, then 8, and drop it on all four arms
   together** — mixing `loss_batch_size` across arms breaks the throughput comparison.

**Cosine is not a gate.** At one hour these arms are at 6-9% of the configured schedule, the
same-seed replicate floor is 0.003-0.005, and CE/JSS/KL are already established as agreeing to 1e-4
epoch-matched. These runs validate the refactor; they do not re-open the objective question.

### The stricter pre-gate, if you want it before spending four GPU-hours

`pytest tests/test_mass_shatter_peak_list_distance.py` on a node that has the wheel installed is
the direct test of the backend, and it is minutes rather than hours. It needs the same environment
the generated job builds (`module load cuda/12.2`, then the venv recipe in the generated script,
including `pip install --no-build-isolation torch_linear_assignment-0.0.6.tar.gz`). Whether the Fir
checkout already has a reusable interactive environment is not established here — if it does, an
`salloc` on `fir:h100-1g.10gb` is the cheap way to run it; if it does not, gate 2 above is the
practical substitute and costs nothing extra.

## Related documents

- [MASShatter GPU audit](masshatter_gpu_audit.md) — the per-spectrum-loop defect class and the
  two loops already batched.
- [Intensity-head bucketing sync plan](masshatter_intensity_head_bucketing_sync_plan.md) — the
  same defect in `CandidateIntensityTransformerHead`.
- [Non-probabilistic intensity plan](masshatter_nonprobabilistic_intensity_plan.md) — why the
  direct peak-list objectives exist at all.
- [Loss setup review](../findings/loss_setup_review_2026_09_13.md) — canonical for proposal-loss
  and gradient-routing behavior around these objectives.
