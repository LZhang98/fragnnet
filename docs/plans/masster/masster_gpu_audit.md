# MASSter GPU utilization audit

> **Status:** REFERENCE · **Updated:** 2026-09-17 · A GPU became available on this machine after
> the static audits below were written; see "Measured profile" for real PyTorch-profiler numbers
> that confirm and sharpen (not replace) those priorities. Earlier text in this file still says
> "GPU timing unavailable" — that was true when it was written, not now.

Follow-up (2026-09-17): the [s3gy1ro4 training-speed audit](../../findings/masster/masster_training_speed_2026_09_17.md)
measures the proposed monitoring reduction on the current unmerged CMF config: 1.40× local
batch-128 forward/backward throughput, with a matched fast config prepared. The L40S epoch-speed
gate remains open. An attention-layout experiment did not establish useful whole-batch gains and
was removed; the bucketing plan below remains unimplemented.

## Measured profile (2026-09-14, RTX 2070 SUPER, `research/masshatter-ion-h-selection` @ `43e880b0`)

Ran the exact H-support config
(`config/masster/merged/channel_selection/nist20_merged_u256_e1024_mix2_no_barcode_h_support_w01.yml`)
through `scripts/run_pl_model_fit.py` with the built-in `profiler: pytorch` (`MyPyTorchProfiler`,
`with_stack=True`, flame-graph export). The config's native `train_batch_size: 128` OOMs on this
card's 8 GB (tried, failed at step 0: "Tried to allocate 2.20 GiB... 6.06 GiB already in use").
Re-ran at `train_batch_size=16` / `accumulate_grad_batches=16` (same effective batch, fits in
~6 GB) and captured 8 active training steps (`profiler_wait_steps=3, warmup=2, active=8`) after a
short warmup. This is one short run on a much smaller/older GPU than the production L40S target —
treat the relative op ranking as signal, not the absolute microsecond totals or a throughput claim.

Self-time leaves aggregated across the 8 profiled steps (`self_cpu_time_total` / `self_cuda_time_total`,
10.0M us / 10.6M us total respectively; the `_record_function_enter_new` CPU bucket, 67%, is
profiler instrumentation overhead itself and should be ignored — it is not a production cost):

| Rank | CUDA self-time | % | Site |
|---|---|---|---|
| 1 | 2.72M us | **25.71%** | `spec_utils.py:483` `int(th.max(batch_idxs).item()) + 1` inside `batched_rel_filter_func` |
| 2 | 1.61M us | 15.17% | `loss.backward()` |
| 3 | 1.31M us | 12.39% | `masster_model.py` `forward` (line attribution collapses to the `def forward` at 1657) |
| 4 | 1.17M us | 11.02% | `nn.functional.linear` |
| 5 | 0.90M us | 8.46% | `scaled_dot_product_attention` (intensity transformer) |
| 6 | 0.73M us | 6.86% | `bond_fragment_heads.py:575` `forward` |
| 7 | 0.67M us | 6.34% | `th.bincount` inside `bond_fragment_heads.py:489` `_contextualize_by_molecule` |

Two findings stand out as new, measured, and actionable beyond the static text below:

1. **`spec_utils.py:483` is the single largest attributed cost in the whole trace, at a quarter of
   all CUDA self-time**, for what is just `dim_size = int(th.max(batch_idxs).item()) + 1` over the
   ground-truth peak batch-index tensor inside `batched_rel_filter_func` (called once per training
   step from `spectrum_pl.py:1150` `_preproc_spec`, on a tensor of at most a few thousand elements).
   A reduction that small cannot genuinely take that long on a GPU: the `.item()` forces a host sync,
   and the profiler is attributing the GPU pipeline drain (waiting for already-enqueued kernels to
   finish before the scalar can be read back) to this op rather than measuring real compute.

   **Follow-up measurement corrected the actionable part of this — read before citing the 25.71%.**
   A targeted benchmark (host time spent inside the call while ~40 matmuls are still queued,
   RTX 2070 SUPER, batch 128 × 60 peaks) isolates where the block actually comes from:

   | variant | host-blocked |
   |---|---|
   | old: `.item()` sync + scatter + boolean mask indexing | 10.71 ms/call |
   | new: no `.item()`, no scatter, boolean mask indexing | 10.35 ms/call |
   | mask computed but **no boolean indexing** | 0.18 ms/call |

   The `.item()` was only the *first* of two sync points. The function ends in
   `mzs[mask], ints[mask], batch_idxs[mask]` — boolean-mask indexing has a data-dependent output
   size, so the host must sync to allocate the result regardless. Removing the `.item()` shifts the
   drain to the mask indexing instead of eliminating it (~0.35 ms of ~10.7 ms, ~3%). **The 25.71%
   is a profiler attribution artifact for GPU-pipeline drain, not 25% of recoverable throughput.**
   Whichever op syncs first absorbs that number, so expect it to reappear on a different line.
   Genuinely removing it needs the filter to stop materializing variable-sized outputs (stay masked
   or padded) — the same "padded prediction/loss contract" change the base-branch audit calls a
   larger job in its finding 4. Note also that host-blocked time is not by itself lost GPU time: the
   GPU is busy during the drain. It costs throughput only insofar as the host is then too late to
   enqueue the next step's kernels.

   **Fixed anyway (2026-09-14), for a smaller but real reason:** with the default
   `target_rel_ints_thresh: 0.0`, line 490's `ints > rel_thresh * max_ints[batch_idxs]` multiplies
   the scatter result by zero — it reduces to `ints > 0`, so the sync *and* the whole
   `scatter_reduce` were dead work. The early return above them does not fire, because it requires
   `mz_max <= 0.0` too and `mz_max` defaults to 1500.0. `batched_rel_filter_func` now takes an
   optional caller-supplied `dim_size` (threaded from `batch["batch_size"]`, already a host int in
   `_common_step`) and skips the maxima entirely when `rel_thresh <= 0.0`.

   **This was never MASSter-specific.** The code is in `SpectrumPL`, and `FraGNNetPL`,
   `MassterPL`, `PrecursorPL`, `BinnedPL`, `GNNPL` and `NeimsPL` all inherit it; none override
   `_common_step`/`preproc_spec`/`_preproc_spec`/`_setup_spec_fns`. Resolving configs through
   `load_config` confirms FraGNNet and MASSter land on identical values
   (`target_rel_ints_thresh=0.0`, `mz_max=1500.0`), so every FraGNNet run took the same dead-work
   path. `misc_utils.py:565`'s `scatter_reduce` has the same `int(th.max(index))` shape but already
   accepts `dim_size`; it syncs only where a caller leaves it `None`, which is worth auditing per
   call site rather than treating as one bug.
2. **`bond_fragment_heads.py:489 _contextualize_by_molecule`'s `th.bincount` alone is 6.34% of all
   CUDA self-time** (662k us over 8 steps) — this is on top of, and separate from, the `isin`/
   `nonzero`/`searchsorted` bucketing costs the static audit already flagged for this function
   (priority 3 below). Unlike finding 1 this is a real kernel doing real work, so it is the better
   throughput target of the two.

   **Checked whether the static audit's suggested fix applies here — it does not.** The static text
   below suggests reusing CPU-known per-molecule component counts instead of a GPU `bincount`. At
   this call site (`masster_model.py:2185`, `component_batch_indices=fragment_batch_indices`),
   the counts are only knowable *after* `fragment_keep` — a boolean mask computed from the model's
   own current-step `fragment_log_probabilities` at `masster_model.py:2058` — has already run.
   There is no CPU-side value to substitute; this is genuinely data-dependent, unlike finding 1's
   `dim_size`. What *is* fixable without that: the surrounding `isin`/`nonzero`/`searchsorted` calls
   each introduce their own additional sync beyond the one unavoidable `counts.tolist()`, and that
   part is pure avoidable overhead. Scoped out in
   [masster_intensity_head_bucketing_sync_plan.md](masster_intensity_head_bucketing_sync_plan.md)
   — not yet implemented.

`masster_model.py`'s `forward`-attributed `.to()` calls (991k us, ~10% of CPU self-time) could
not be pinned to an exact line from the flame graph (function-level stack attribution collapses to
the `def forward` line for this file); worth a finer `record_function`-annotated re-profile if this
becomes the next target.

**Not yet done:** the config's actual `train_batch_size=128` / L40S-scale run (this profile used
1/8th batch size on a consumer 8 GB card), a longer active window (8 steps is short), and separating
data-loading/candidate-CPU wall time from the numbers above (this profile only shows *model* CPU/CUDA
self-time inside `training_step`, not the `masster_candidates.py` CPU-topology phase called by the
same forward — that phase's cost is folded into the `masster_model.py forward` bucket above and
not further broken out here).

## H-selection branch follow-up

Checked `research/masshatter-ion-h-selection` at
`43e880b0712ce15b4da0f5013ced010a031d6e81`, extracted read-only to
`/tmp/masshatter-h-gpu-audit`. The checkout remains `feiw_v2`. Source paths and
line numbers in this section refer to that snapshot. No production code changed.

### Corrections to the earlier audit

- **Early topology termination is implemented.** `frag/masster_candidates.py:713`
  passes the unique target and forced prefix into the compiled builder.
  `frag/masster_core.pyx:799–811` stops after the required overshoot proof and
  forced prefix, preserving the old truncation semantics. Ranked selection still
  constructs its budget first; topology no longer necessarily processes it all.
- **Selection is parallelized.** `masster_candidates.py:264` uses a thread pool
  over a GIL-releasing mixture kernel. The template sets four selection threads.
  Grouped views share selection and cached topology. Per-molecule assembly remains
  a host loop; increasing DataLoader workers still does not parallelize it.
- **Packing already fills one host buffer directly from chunks** (line 125).
  Its upload remains an unpinned blocking `.to(device)` at line 146.
- The branch's topology performance plan and benchmark already cover unique targets,
  static sidecars, grouped views, SMARTS and scanned/retained counts. The plan still
  explicitly lacks its canonical full CUDA training-step throughput gate. Earlier
  remarks about the limited base-branch benchmark do not apply here.

### Revised optimization priorities

1. **Remove mandatory training diagnostics from the support-loss path.**
   `pl_model/masster_pl.py:1227–1231` enables channel targets for either
   diagnostics or a support loss. Lines 1468–1482 then unconditionally compute both
   support metrics and required-top-k metrics. The latter loops over spectra,
   boolean-indexes GPU arrays, branches on `batch_supported.any()` (line 474), and
   computes quantiles. This lives inside the loss closure, so
   `compute_train_auxiliary_metrics: False` does not skip it; turning off
   `masster_channel_diagnostics` alone also does not skip it when support loss
   is enabled. Separate required targets/loss from monitoring and run expensive
   monitoring on validation or a sampled schedule.
2. **Vectorize Gaussian support attribution and remove scalar validation waits.**
   `model/candidate_support.py:50` checks finite values using `.item()`;
   lines 54–146 repeatedly compact channels and targets per spectrum. Line 103
   extracts a scalar validity result, and line 116 branches on a GPU reduction.
   `balanced_binary_support_loss` repeats host scalar checks at lines 165–166.
   Use asynchronous validation and padded/chunked matching with masked tensor
   reductions and explicit empty-row handling. Preserve detached posterior targets
   and intensity-prior semantics. The Hungarian attribution arm needs separate
   profiling; the inspected H-support configs use Gaussian posterior attribution.
3. **Restore a synchronization-light attention layout.**
   `model/bond_fragment_heads.py:517` now calls `counts.tolist()` on GPU counts;
   line 531 still calls `nonzero` per bucket. Unlike the base branch, the head no
   longer accepts CPU component counts. Its atom decoder additionally computes
   `counts.max().item()` at line 467 and pads all fragment atom memories to that
   maximum. The inspected H-support config enables four atom-decoder layers.
   Pass CPU-derived layouts when membership is static, and use a known safe atom
   bound or bucketed atom layout. Account for GPU fragment filtering: the unique
   path filters/remaps memberships at `masster_model.py:2080–2116`, so raw
   builder counts cannot blindly replace post-filter counts. Reuse compacted index
   maps, or retain padded membership with masks on that path.
4. **Disable diagnostic-only base CE when it is not needed.**
   The inspected `merged/channel_selection/nist20_merged_u256_e1024_mix2_no_barcode_h_support_w01.yml`
   has `masster_log_base_ce: True` but `masster_base_ce_weight: 0.0`.
   Lines 1246–1248 and 1559–1570 in `masster_pl.py` still evaluate the second
   CE. Setting logging false skips this monitoring computation without removing
   the separately weighted proposal or H-support objectives. If base CE monitoring
   is retained, reuse matching/layout preparation with primary CE.
5. **Reduce repeated compaction and avoid full diagnostic output when unused.**
   `masster_model.py:1526–1533` still compacts unique memberships repeatedly.
   Lines 2483–2503 independently compact both full and selected channels into
   masses, probabilities, batch IDs, cut counts and fragment IDs. Reuse one index
   vector per mask. Full candidate arrays have real proposal/coverage consumers,
   so gate them based on actual consumers rather than deleting them wholesale.
   In direct JSS/KL loss modes, `masster_pl.py:1420` additionally invokes the
   entire cut-bucket coverage routine inside the loss just to consume overall
   weighted recall; split that scalar monitoring need from per-bucket diagnostics.
6. **The candidate round trip remains a structural barrier.**
   `masster_candidates.py:475` copies logits to CPU; line 513 additionally
   transfers mixture weights, and line 507 transfers group IDs when enabled.
   Selection/assembly and H2D still separate proposal GPU work from intensity GPU
   work. Pack dynamic inputs together, retain static metadata on CPU, and measure
   the remaining host fraction before considering pipelining or GPU topology.

The fixed top-k H selector itself (`masster_model.py:274–296`) uses a stable
GPU sort and scatter without host scalar extraction. Preserve its tie behavior.
The optional absolute support-floor selector is different: lines 216–228 use
compaction, a Python condition on a GPU reduction, and per-spectrum fallback
searches; use segmented tensor fallback selection if that option is active.

**Next measurement:** profile the exact H-support training config after warmup,
separating support targets, support diagnostics, proposal loss, base CE, atom
decoder, candidate attention and CPU candidate phases. Compare spectra/second,
optimizer steps/second and peak VRAM with unchanged scientific losses. These
priorities are source-backed, not measured speedup estimates. No GPU benchmarks
or model tests ran for this documentation-only follow-up.

## Earlier base-branch audit

Audited 2026-09-14: `research/masshatter` at `8e39b46160ab383d5235b3a6b789c4bbb19882be`.
The active checkout is `feiw_v2`; source was inspected in
`/tmp/masshatter-gpu-audit` without changing branches. References below are to
the audited branch, not the current checkout. This is a static audit: the local
NVIDIA driver is unavailable, so rankings are optimization priorities, not measured
runtime shares or speedup claims. Production code was not changed.

## Findings, in priority order

### 1. Candidate generation blocks the middle of every forward pass

`src/fragnnet/model/masster_model.py:979` calls the candidate builder after
the precursor GNN and bond head. In
`src/fragnnet/frag/masster_candidates.py:185`, `bond_logits.detach().cpu()`
waits for those GPU results. The host then selects masks and builds topology
serially per molecule (line 265), before uploading the candidate indices
(lines 701–734). The downstream GPU computation depends on that result.

This creates the dependency chain **GPU encoder → CPU selection/topology → GPU
intensity head**. More DataLoader workers cannot parallelize this forward-pass
work. `non_blocking=True` alone cannot remove the dependency: CPU selection must
wait until the logits are actually available.

First optimize the CPU interval: prepare immutable graph endpoints and batch
offsets during collation, and move batch-level numeric work into the compiled
builder. Independent molecule processing could use native parallelism after
making the kernels release the GIL; merely declaring helpers `nogil` does not
release it at callers. Longer term, GPU selection/topology or explicit pipelining
across microbatches could remove/hide the gap. Both require substantial work;
pipelining must preserve gradient accumulation, optimizer timing and randomness.

### 2. Unique-fragment enumeration does full-budget work before truncating

`masster_candidates.py:287` requests `max_enumerated_masks`. Line 569 calls
`build_component_arrays_raw` on that mask set; lines 580–640 then trim its
already-built output to the shortest prefix reaching the unique-fragment target.
Thus U256/enum2048 can pay selection, topology and assembly costs for the full
budget even when the retained prefix is much shorter.

Implement incremental ranked enumeration and incremental component deduplication,
stopping after the exact target-reaching mask, subject to retaining all forced
empty/single-cut masks. Preserve ranked tie order, forced-mask order, all components
of the final mask, representative identities and the existing budget-exhaustion
behavior. Measure enumerated versus retained masks to quantify avoidable work.
This is the strongest structural optimization candidate for unique mode.

### 3. Attention bucketing synchronizes on indices already knowable on CPU

`src/fragnnet/model/bond_fragment_heads.py:223–288` sorts component batch indices,
counts them, and uses `isin` plus `nonzero` for every size bucket. CUDA `nonzero`
must expose a data-dependent result size to the host. The production call already
passes CPU `component_counts`, and the builder emits components grouped by molecule.

Build bucket gather indices, local row indices and positions from those CPU counts
and prefix offsets, then include them in the packed candidate transfer. Keep the
generic sorting path for callers without a guaranteed grouped layout. This removes
per-bucket dynamic compaction and redundant sorting without changing attention.
The existing power-of-two buckets and inducing-point attention are useful and
should be retained.

### 4. Repeated GPU compaction in unique memberships and output channels

`masster_model.py:864–872` separately boolean-indexes atom and boundary
memberships for unique representatives, although those identities originated on
CPU. Emit canonical unique memberships directly from the builder.

At lines 1269–1277, the same validity mask separately compacts m/z, log probability,
batch index and cut count. Compute valid flat indices once and reuse integer
gathers, including the base-spectrum output. This retains one dynamic-size boundary
but avoids repeating it. Eliminating compaction completely would require a padded
prediction/loss contract and is a larger change.

### 5. Proposal-anchored training repeats sparse CE setup and matching

`src/fragnnet/pl_model/masster_pl.py:379–409` calls sparse CE twice when base
CE is enabled. Both calls share true peaks, predicted masses, batch indices and
tolerances, differing in predicted log probabilities. Each invocation of
`src/fragnnet/model/loss.py:427–438` computes counts and transfers cumulative
offsets with `.tolist()`, then rebuilds chunk layouts and mass matching.

Separate shared matching/layout preparation from differentiable probability
evaluation, and reuse it for both heads. Preserve base loss gradients and
`detach_base_ion_h`; disabling base CE would change the proposal-anchored objective.
The current vectorized loss already avoids per-spectrum scalar synchronization
inside this loop. Do not replace it with the sequential loss.

### 6. Candidate uploads and diagnostics have smaller avoidable costs

`masster_candidates.py:87` uploads concatenated NumPy-backed indices with
blocking `.to(device)`. This allocation is made inside forward and is not covered
by DataLoader pinning. Four additional diagnostic tensors are created on-device
at lines 769–788, and `masster_model.py:1344` creates timing tensors individually.

Benchmark reusable pinned staging storage with asynchronous copies and correct
buffer lifetime management. Pack small diagnostic uploads together; compute CPU-known
counts and productive-mask statistics in the builder. Do not expect transfer
changes alone to hide serial topology work. Avoid pinning fresh memory every step
without measuring its allocation/copy overhead.

## Configuration and conditional costs

The inspected merged U256/enum2048 proposal-anchored configuration uses FP32,
TF32 enabled, training batch 128, accumulation 2, vectorized loss chunks of 8,
and no CE-pair loss. These are representative settings, not a claim about the
user's running job.

- Benchmark BF16 for neural layers while retaining FP32 masses, matching and
  probability reductions. Casting chemistry to BF16 can destroy mass tolerance
  accuracy. Mixed precision may shorten GPU compute without improving utilization
  percentage when CPU work dominates; judge spectra/second and quality.
- Accumulation does not enlarge individual GPU operations. Sweep actual batch size
  and loss chunk size under a fixed effective batch and memory limit. Larger loss
  chunks can increase padded peak-pair memory substantially.
- The template already has `pin_memory: True` and
  `compute_train_auxiliary_metrics: False`. Do not attribute normal training stalls
  to validation-only coverage metrics. Coverage itself loops over spectra and five
  cut buckets (`masster_pl.py:68–132`) and merits separate evaluation profiling.
- If CE-pair loss is enabled, `loss.py:2304–2313` copies multiple peak/metadata arrays
  to CPU and performs host matching. It is an additional conditional bottleneck,
  absent from the inspected configuration.
- CUDA validation in `bond_fragment_heads.py:13` uses `_assert_async` already.
  CPU-side `.item()` in exploration is not a CUDA synchronization by itself.
- Whole-model compilation is rejected by `masster_pl.py:163`. Selective
  compilation of tensor-only encoder/head regions is an experiment after removing
  dynamic compaction; it cannot compile away the CPU candidate dependency.

## Measurement and acceptance

Profile a real training batch distribution after warmup, including forward,
loss, backward and optimizer, with CPU/CUDA activities. Label encoder/bond head,
candidate D2H, selection, topology, assembly/H2D, unique gathering, intensity head,
CE setup/evaluation, backward and optimizer. Inspect GPU idle gaps, memcpy events,
`nonzero`, scalar extraction, and kernel launch density. Measure training and
validation separately; repeat throughput measurement without profiler overhead.

Existing `perf_counter` candidate timers are not GPU kernel timers: input transfer
can include waiting for preceding encoder work, output transfer includes host
allocation/copy overhead, and asynchronously enqueued scoring is not fully timed.
The existing synthetic chain benchmark also omits the static graph sidecar and
unique mode used in production. Do not treat its phase split as production evidence.

For each optimization compare fixed-weight candidate identities, ordering,
memberships, valid channels, spectra, loss and gradients. Include rings,
disconnected and zero-bond graphs, ranked ties, forced cuts, target overshoot and
budget exhaustion. Report spectra/second, peak VRAM, CPU candidate latency and GPU
idle time at the same candidate budgets; reducing budgets changes model support.

Recommended implementation order: CPU-derived attention/unique layouts and shared
CE matching; incremental unique enumeration; transfer batching; then selective
compilation/mixed precision and, if still needed, candidate pipelining or GPU kernels.
