# MASSter loss-path audit after the batching refactor

> **Status:** FINDING · **Updated:** 2026-09-19
> Reviewed `86cbf98d` (clean tree). Measured on a local RTX 2070S without `torch_linear_assignment`, so Hungarian magnitudes are the SciPy fallback and need a cluster re-measure. No code, configuration, or training state changed.
>
> **Proposal coverage:** the proposal term's objective-level properties are consolidated in
> [MASSter proposal loss — consolidated audit](masster_proposal_loss_audit_2026_09_22.md).

## Scope

Audited the MASSter loss composition as of `86cbf98d` ("mass shatter loss refactor"):

- [`sparse_peak_list_distance`](../../../src/fragnnet/model/loss.py) and its padding and chunking
  helpers `_build_padded_batch_layout`, `_pad_flat_with_layout`, `_build_loss_chunks`.
- [`round_aggregate_peaks`](../../../src/fragnnet/utils/spec_utils.py), rewritten in the same commit.
- The `_setup_loss_fn` validation block, the `loss_fn` closure, and every auxiliary term it can
  add in [`masster_pl.py`](../../../src/fragnnet/pl_model/masster_pl.py).

The objective-level semantics (JSS vs KL gradient behaviour, matching and support ambiguity,
unmatched-intensity invariance) were already reviewed in
[loss_setup_review_2026_09_13.md](../loss_setup_review_2026_09_13.md) and are not repeated.
Throughput context is in
[masster_direct_loss_batching_plan.md](../../plans/masster/masster_direct_loss_batching_plan.md).

Tests run: `tests/test_masster_peak_list_distance.py` and
`tests/test_spec_utils_preprocessing.py` (298 passed), `tests/test_masster_pl.py`
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
([peak-count distribution](#peak-count-distribution-from-local-memory-measured-2026-09-17)). Because every spectrum in a chunk is padded to the
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

### 2. The chunk-splitting guard cannot fire for MASSter's shape

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
[`nist20_merged_independent_jss_hattr_gaussian.yml`](../../../config/masster/merged/channel_selection/nist20_merged_independent_jss_hattr_gaussian.yml),
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
(`masster_pl.py:1496`) and could pass it.

### 5. The channel diagnostics cost more than the loss they are logged beside

`channel_support_metrics_enabled` is `masster_channel_support_metrics` **or**
`masster_channel_diagnostics`, and every config under
`config/masster/merged/channel_selection/` sets `masster_channel_diagnostics: True`. That
turns on two per-spectrum Python loops inside `loss_fn`, on every training step:

| function | line | CUDA | CPU |
| --- | --- | ---: | ---: |
| `_channel_support_metrics` | `masster_pl.py:536` | 30.5 ms | 25.8 ms |
| `_ion_h_required_k_metrics` | `masster_pl.py:722` | 51.1 ms | 36.2 ms |

Batch 64, 256 fragments per spectrum, 2 ion modes x 9 H channels. Together 81.6 ms per batch,
against 3.8-20 ms for the primary loss itself, and **faster on CPU** - the same signature as
`observational_channel_targets`, which was 230 ms per batch and the same on CPU before being
batched to 2.7 ms (GPU audit (removed 2026-10-13; `git show e65339ed:docs/plans/masster/masster_gpu_audit.md`)). Both loops index dense tensors with
a boolean `fragment_batch_idxs == batch_index` mask, and `_ion_h_required_k_metrics` adds a
per-spectrum `th.quantile` and an `if not batch_supported.any()` sync.

These are monitoring only. The cheapest action is to stop implying them from
`masster_channel_diagnostics` on every step, or to compute them on validation batches only;
batching them is the same segment-reduction rewrite already applied twice in this file.

**Applied 2026-09-19:** `masster_channel_diagnostics` is now `False` in all 47 configs under
`config/masster/` that set it, matching the template default and the precedent already set by
[`..._warmup_e0_4_dense_subformula_fast.yml`](../../../config/masster/merged/warmup_duration/nist20_merged_independent_jss_warmup_e0_4_dense_subformula_fast.yml).
`masster_channel_support_metrics` is `True` in no config, so both loops are now off
everywhere. The objective is unchanged: `channel_targets_enabled` is also the OR of the support
and ion-mode loss weights, and those arms keep `masster_ion_h_support_loss_weight: 0.1`.
The four `config/debug/debug_tiny_masster_*.yml` smoke configs keep it on by choice.
**Consequence:** `scripts/eval/masster_ion_h_top_k_sweep.py` refuses a config with the flag
off, so the Phase 2 channel audit now needs
`--override masster_channel_diagnostics=True` when it is pointed at a training YAML.

The other three per-spectrum loops on this path, `_shared_group_proposal_coverage_loss`
(`:613`), `_cross_ev_fragment_delta_loss` (`:722`, `:788`) and `_candidate_coverage_by_cut_count`
(`:129`), are gated off by default and off in the merged base config, so they are not costing
current runs.

### 6. Minor

- `decimals=4` is hard-coded at the call site (`masster_pl.py:1552`) while every other
  tolerance on this path comes from `inst_type_loss_tol`. It only coalesces numerically
  identical peaks at the current value, so nothing is wrong today, but it is the same
  hard-coded-tolerance-duplicate pattern that bit the subformula filter.
- `true_intensities = kwargs.get("true_intensities", true_logprobs.exp())`
  (`masster_pl.py:1500`) hides a required field behind a default, against the dict-access
  rule in [python.md](../../rules/python.md). `SpectrumPL` always supplies the key and its value is
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

## Loss-objective comparison and JSS-vs-CE verdict (from local memory, verified 2026-10-06)

Source: agent-memory notes of 2026-09-19 reconstructed from all 226-230 `model_type: masster` W&B
runs. W&B figures were not re-queried today; the keys named below exist in `config/template.yml`.

- **Cosine is a wash.** Loss-matched merged pair `7abal2ru` (CE, `probability`) against `38ugaxen`
  (JSS, `independent`), 199 epochs: 0.7985 against 0.7996 (+0.0011); KL `kj17tksj` 0.7984. The 0.8010
  headline `8v0i0fq7` is not loss-matched (warmup `e0_4`, swept for JSS only). Matched paired-seed
  screen at epoch 42 (`*_warmup_e0_15_dense`): CE 0.7711 / 0.7636, KL 0.7710 / 0.7635 (two seeds,
  agreement to 1e-4), JSS 0.7662. Long subformula runs epoch-matched at 60/100/140: JSS
  0.7912/0.7936/0.8010, CE 0.7846/0.7921/0.7951, KL 0.7868/0.7902/0.7970, spread 0.006 against a
  within-arm seed spread of 0.004-0.008. The primary objective is not a lever.
- **Same-seed replicate floor.** Identical config, seed and epoch count: `lzhb3ozh`/`mgs5qts9`
  0.7941 vs 0.7990 (0.0049), `k5gslw0f`/`96i3q8up` 0.7915 vs 0.7949 (0.0034), `zp06g7ml`/`6ru0lg6f`
  0.7987 vs 0.7981 (0.0007). Effects below about 0.005 cannot be resolved at n=1 (wprecision floor
  0.003-0.005, recall 0.0010, wrecall 0.0002). Do not re-run a JSS-vs-CE cosine comparison at n=1.
- **Wall clock favours CE where one would stop.** Merged best val by hour, 3 h: CE 0.7872 vs JSS
  0.7715; 5 h: 0.7966 vs 0.7905; JSS crosses at about 8 h. Unmerged is parity at 1.5x cost:
  `6ks7v8ce` CE 0.7058 at 99 ep / 11.3 h against `66unqbwq` JSS 0.7057 at 48 ep / 17.3 h; widening the
  head (`h0gvuf4f`, 384x3) reached 0.7101, four times what the loss did. Throughput was 20.5-20.8 ep/h
  for CE and 14.0-14.8 for JSS/KL before the loss batching refactor (1.46x), with candidate time
  107-122 ms/batch in every arm.
- **The one real JSS effect is weighted precision.** Merged at epoch 99: CE 0.9211, JSS 0.9500,
  KL 0.9288; at 199: 0.9366 / 0.9497 / 0.9322; epoch 20: 0.7344 / 0.8774 / 0.7352. Recall and wrecall
  are identical and unweighted precision is flat, so JSS puts less *intensity* on wrong peaks rather
  than emitting fewer. `38ugaxen` and `kj17tksj` differ in exactly one key (`loss_type`), so this is
  the objective. Mechanism: CE and KL are forward-KL shaped and never visit `p = 0` (KL sums only over
  `positive_true`), so spurious predicted mass is penalized only through the normalizer; JSS is
  symmetric.
- **Confound map.** CE vs JSS/KL is a five-key bundle (`loss_type`, `masster_intensity_output`
  probability vs independent, `masster_ion_h_support_loss_weight` 0 vs 0.1, the support-gate and
  support-head switches); JSS vs KL is clean. `masster_peak_matching` has never been varied within a
  loss (all JSS/KL runs `hungarian`, the single `cosine_distance` run `closest` and diverged to 0.148)
  and is inert under CE. The entropy-match auxiliary (`masster_intensity_entropy_match_weight`) was 0
  in all 117 runs that carry it, and `masster_kl_epsilon` stayed 1e-4 (caps the KL penalty at 9.2 nats).
  The sqrt intensity transform is harmful twice (JSS 0.7668 to 0.7566, KL 0.7710 to 0.7505 at epoch 42).
- **Batching measurements behind the sync-bound finding** (RTX 2070S, fwd+bwd): per-spectrum loops
  with boolean-mask indexing cost about 900 us per spectrum regardless of arithmetic, cost is flat in
  problem size, and CPU is as fast or faster than CUDA. `observational_channel_targets` 230 ms to 2.7 ms
  (82x, bit-identical, 13 implicit syncs per spectrum before, 9 per call after).
  `_candidate_weighted_recall` 45.0 to 1.3 ms (35x) at batch 128, agreeing to 2.98e-8 (index-add
  reassociation; do not assert `th.equal`). Diagnose with `th.cuda.set_sync_debug_mode("warn")`.
  Still-looping siblings at the time: `_candidate_coverage_by_cut_count` (one match pass per bucket per
  spectrum) and other channel-metric helpers.

## Peak-count distribution (from local memory, measured 2026-09-17)

On the `nist20_ma_mi` train split: 287,721 unmerged rows in 45,639 groups; merged statistics
from a 4,000-group sample.

| Peaks per row | mean | p50 | p90 / p95 | p99 | max |
|---|---|---|---|---|---|
| Unmerged | 31.9 | 17 | p95 110 | 200 | 804 |
| Merged | 52.0 | 25 | p90 134 | 343 (p99.9 582) | 1168 |

Merged is log-normal (log-space sd 1.29, max/mean 22.5x). Size padded buffers from the tail
(max about 1170, p99 about 343), never the mean.
