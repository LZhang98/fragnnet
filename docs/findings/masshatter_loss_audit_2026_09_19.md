# MASShatter loss-path audit after the batching refactor

> **Status:** FINDING · **Updated:** 2026-09-19 · Reviewed `86cbf98d` (clean tree). Measured on a
> local RTX 2070S without `torch_linear_assignment`, so Hungarian magnitudes are the SciPy
> fallback and need a cluster re-measure. No code, configuration, or training state changed.

## Scope

Audited the MASShatter loss composition as of `86cbf98d` ("mass shatter loss refactor"):

- [`sparse_peak_list_distance`](../../src/fragnnet/model/loss.py) and its padding and chunking
  helpers `_build_padded_batch_layout`, `_pad_flat_with_layout`, `_build_loss_chunks`.
- [`round_aggregate_peaks`](../../src/fragnnet/utils/spec_utils.py), rewritten in the same commit.
- The `_setup_loss_fn` validation block, the `loss_fn` closure, and every auxiliary term it can
  add in [`mass_shatter_pl.py`](../../src/fragnnet/pl_model/mass_shatter_pl.py).

The objective-level semantics (JSS vs KL gradient behaviour, matching and support ambiguity,
unmatched-intensity invariance) were already reviewed in
[loss_setup_review_2026_09_13.md](loss_setup_review_2026_09_13.md) and are not repeated.
Throughput context is in
[masshatter_direct_loss_batching_plan.md](../plans/masshatter_direct_loss_batching_plan.md).

Tests run: `tests/test_mass_shatter_peak_list_distance.py` and
`tests/test_spec_utils_preprocessing.py` (298 passed), `tests/test_mass_shatter_pl.py`
(123 passed).

## What the refactor gets right

These were checked rather than assumed, and none of them needs work.

- **Zero padding is exact for all three objectives.** `0` is the identity for the L1 and L2
  normalizers and for `_x_log_x`, KL's ragged step is a `th.where` on `true_distribution > 0`,
  and padded rows and columns are removed from `match_mask` by `true_valid`/`pred_valid`
  before any distance is formed.
- **The Hungarian invalid-pair cost still orders cardinality first, and is tighter than the
  loop version.** `invalid_cost` is now the maximum over in-tolerance pairs only
  (`loss.py:274`), which for a 10 ppm window is a few mDa instead of the old feasible-submatrix
  maximum; the `+1.0` term still dominates `K x max_valid_distance`, so a valid pair is always
  preferred.
- **The assignment input is detached** (`loss.py:282`), so defect 2 of the 2026-09-13 review
  (backward through the SciPy fallback) does not recur on this path.
- **The backend's index layout is checked, not trusted** (`loss.py:286-293`), which matters
  because only the SciPy branch is exercisable without the wheel installed.
- **`new_empty` result buffers are fully written.** `_build_loss_chunks` covers `[0, batch_size)`
  with no gaps and cannot emit an empty chunk, so no uninitialized value can escape.
- **`round_aggregate_peaks` preserves its (batch, m/z) ordering contract**, which is what makes
  the contiguous chunk slicing in `sparse_peak_list_distance` valid.

## Findings

### 1. Padding to the per-chunk tail, not the mean, dominates the direct-loss cost

The batching benchmark in the plan used a uniform 52 observed peaks per spectrum. The measured
merged distribution is log-normal with p50 25, p99 343, max 1168
(`reference_merged_peak_count_distribution`). Because every spectrum in a chunk is padded to the
chunk's widest peak list, a single tail spectrum inflates the whole chunk, and for Hungarian it
inflates the assignment problem each spectrum has to solve.

Batch 64, 512 predicted peaks, JSS, forward plus backward, same mean of 72 observed peaks in
both rows:

| observed peak counts | matching | `loss_batch_size` 8 | 64 |
| --- | --- | ---: | ---: |
| uniform 72 | `closest` | 16.5 ms | **3.7 ms** |
| ragged, max 1168 | `closest` | 16.7 ms | 9.9 ms |
| uniform 72 | `hungarian` | 38.4 ms | **23.1 ms** |
| ragged, max 1168 | `hungarian` | 116.3 ms | **441.6 ms** |

Two consequences:

- Hungarian **gets worse as `loss_batch_size` grows** on real shapes (116 → 442 ms), the
  opposite of the uniform-input benchmark, because a wider chunk means a wider padded cost
  matrix for every spectrum in it.
- `closest` still improves up to about 32 and then flattens (16.7 / 11.4 / **7.6** / 9.7 ms at
  8 / 16 / 32 / 64, 512 predicted peaks).

The cheapest fix is to build chunks over a peak-count-sorted permutation instead of the
batch-major order, so each chunk's `max_true` is close to its members' own sizes; the layout
helper already supports a non-sorted input through `sort_idx`, but the result scatter at
`loss.py:207-366` assumes contiguous slices and would become an `index_copy`. A smaller fix is
to give `_build_loss_chunks` an absolute padded-pair budget (see finding 2).

**Caveat:** measured with the SciPy fallback. The batched CUDA solver has far better constants,
but its work still scales with the padded matrix, so the direction holds and only the magnitude
is open. Re-measure on a cluster node with `torch_linear_assignment` before quoting numbers.

### 2. The chunk-splitting guard cannot fire for MASShatter's shape

`_build_loss_chunks` (`loss.py:709`) splits when `k * max_true * max_pred > sum_true * sum_pred`.
With roughly equal predicted-peak counts per spectrum, `sum_pred = k * max_pred`, so the test
reduces to `max_true > sum_true`, which is impossible. In general it fires only when one
spectrum dominates **both** peak lists at once.

Measured: 256 spectra drawn from the merged distribution with one 1168-peak spectrum, 512
predicted peaks, `loss_batch_size` 64 produced exactly 4 chunks, that is, no split at all, and a
widest padded block of 38.3M pairs. The docstring's "chunks are narrowed further whenever
padding would exceed the flat Cartesian pair count" is true but inert here; `loss_batch_size` is
the only real control.

Peak allocation for that batch, forward plus backward:

| n_pred | `closest` | `hungarian` | `cross_entropy` control, same shapes |
| ---: | ---: | ---: | ---: |
| 512 | 463 MiB | 660 MiB | 585 MiB |
| 2048 | 1483 MiB | 2623 MiB | — |

This is not a regression against the shipped CE path at matched shapes, and it is shared with
`sparse_cross_entropy_vec`, which uses the same helper. It does mean memory now scales with
`loss_batch_size x max_true x max_pred`, where before the refactor this path was per-spectrum and
independent of `loss_batch_size`.

### 3. `loss_batch_size` is newly meaningful for the direct losses

The pre-refactor `sparse_peak_list_distance` had no `loss_batch_size` parameter (`git show
HEAD^`), so on the direct-loss arms the key only ever affected the unused CE path. Every
checked-in direct-loss config, for example
[`nist20_merged_independent_jss_hattr_gaussian.yml`](../../config/masshatter/merged/channel_selection/nist20_merged_independent_jss_hattr_gaussian.yml),
sets `loss_batch_size: 8`. Given finding 1 that value is close to optimal for `hungarian` and
leaves roughly 2x on the table for `closest`. It is worth re-tuning per matching mode rather
than inheriting the CE arm's value, and the plan's claim that the key "does not change the loss"
remains true numerically.

### 4. The batch size is inferred from the observed peaks, so a trailing empty spectrum shrinks the result

`loss.py:166` derives `batch_size` from `true_batch_idxs.max()`. An interior spectrum with no
observed peaks raises a clear error, but a **trailing** one is invisible: declaring three spectra
where index 2 has no observed peaks returns vectors of length 2. Reproduced directly. Downstream
this surfaces as a shape mismatch against the other per-spectrum terms rather than as a wrong
number, and `SpectrumPL` filtering keeps at least the base peak, so incidence in training is
expected to be zero. The CE control does not have the ambiguity: it takes the batch size from
`pred_oos_logprobs.shape[0]` (`loss.py:777`). The caller already holds `kwargs["batch_size"]`
(`mass_shatter_pl.py:1496`) and could pass it.

### 5. The channel diagnostics cost more than the loss they are logged beside

`channel_support_metrics_enabled` is `mass_shatter_channel_support_metrics` **or**
`mass_shatter_channel_diagnostics`, and every config under
`config/masshatter/merged/channel_selection/` sets `mass_shatter_channel_diagnostics: True`. That
turns on two per-spectrum Python loops inside `loss_fn`, on every training step:

| function | line | CUDA | CPU |
| --- | --- | ---: | ---: |
| `_channel_support_metrics` | `mass_shatter_pl.py:536` | 30.5 ms | 25.8 ms |
| `_ion_h_required_k_metrics` | `mass_shatter_pl.py:722` | 51.1 ms | 36.2 ms |

Batch 64, 256 fragments per spectrum, 2 ion modes x 9 H channels. Together 81.6 ms per batch,
against 3.8-20 ms for the primary loss itself, and **faster on CPU** - the same signature as
`observational_channel_targets`, which was 230 ms per batch and the same on CPU before being
batched to 2.7 ms (`project_per_spectrum_loops_sync_bound`). Both loops index dense tensors with
a boolean `fragment_batch_idxs == batch_index` mask, and `_ion_h_required_k_metrics` adds a
per-spectrum `th.quantile` and an `if not batch_supported.any()` sync.

These are monitoring only. The cheapest action is to stop implying them from
`mass_shatter_channel_diagnostics` on every step, or to compute them on validation batches only;
batching them is the same segment-reduction rewrite already applied twice in this file.

The other three per-spectrum loops on this path, `_shared_group_proposal_coverage_loss`
(`:613`), `_cross_ev_fragment_delta_loss` (`:722`, `:788`) and `_candidate_coverage_by_cut_count`
(`:129`), are gated off by default and off in the merged base config, so they are not costing
current runs.

### 6. Minor

- `decimals=4` is hard-coded at the call site (`mass_shatter_pl.py:1552`) while every other
  tolerance on this path comes from `inst_type_loss_tol`. It only coalesces numerically
  identical peaks at the current value, so nothing is wrong today, but it is the same
  hard-coded-tolerance-duplicate pattern that bit the subformula filter.
- `true_intensities = kwargs.get("true_intensities", true_logprobs.exp())`
  (`mass_shatter_pl.py:1500`) hides a required field behind a default, against the dict-access
  rule in [python.md](../rules/python.md). `SpectrumPL` always supplies the key and its value is
  exactly `true_logprobs.exp()` (`spectrum_pl.py:1309`), so the fallback is dead today.

## Suggested order

1. Stop paying finding 5 on every step; it is the largest measured cost and it buys no gradient.
2. Re-measure finding 1 on a node with `torch_linear_assignment`, then decide between
   size-sorted chunking and an absolute pair budget.
3. Re-tune `loss_batch_size` per matching mode (finding 3) once 2 is settled.
4. Findings 4 and 6 are small correctness and hygiene fixes with no expected metric effect.

## Limitations

- Peak lists are synthetic, drawn from the measured merged count distribution; no real batch was
  profiled end to end.
- `torch_linear_assignment` is absent locally, so every Hungarian timing is the SciPy fallback.
- Timings come from one consumer GPU; only ratios within a row of a table should be carried over.
