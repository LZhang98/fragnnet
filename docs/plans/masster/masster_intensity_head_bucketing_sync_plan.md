# MASSter — Intensity-Head Bucketing Sync Removal Plan

> **Status:** PROPOSED · **Updated:** 2026-09-14
> Scoped from the GPU audit; not implemented, payoff unmeasured (the measured 6.34% `bincount` kernel is not what this plan removes).

## Why this isn't a quick fix

`CandidateIntensityTransformerHead.forward` (`bond_fragment_heads.py:575`) is called from
`masster_model.py:2185` with `component_batch_indices=fragment_batch_indices`, which has
already been filtered by `fragment_keep` — a boolean mask computed at `masster_model.py:2058`
from `fragment_log_probabilities`, i.e. from **the model's own current-step output**. The
per-molecule fragment counts `_contextualize_by_molecule` buckets by are therefore only knowable
after that GPU computation completes; there is no CPU-side substitute the way `batch["batch_size"]`
was for the rel-filter fix. Any redesign has to keep the one sync that derives these counts from
GPU state — the target is everything *downstream* of that sync, not the sync itself.

## What the current code does, and where each sync is

```python
order = th.argsort(component_batch_indices, stable=True)              # GPU, no sync
sorted_batch_indices = component_batch_indices[order]                  # GPU, no sync
counts = th.bincount(sorted_batch_indices, minlength=batch_size)       # GPU, no sync
count_values = counts.tolist()                                         # SYNC 1 (necessary)
for bucket_limit, molecule_indices in bucketed_molecules.items():      # pure CPU, cheap
    molecule_index_tensor = th.tensor(molecule_indices, device=...)    # H2D upload, no sync
    token_mask = th.isin(sorted_batch_indices, molecule_index_tensor)  # GPU, no sync
    sorted_token_indices = th.nonzero(token_mask, as_tuple=False)      # SYNC 2..N (avoidable)
    ...
```

`th.nonzero`'s output size is data-dependent, so materializing it forces a device sync — and this
happens once **per distinct bucket size present in the batch** (up to `log2(max_count)` times, so
plausibly 5-10 syncs per forward call, not one). `_refine_from_fragment_atoms` (`bond_fragment_heads.py:421`,
active whenever `atom_decoder_num_layers > 0` — true in the profiled config) has the identical
shape one level up: `max_atoms = int(counts.max().item())` is SYNC 0, before any of the above even
starts. So a single `CandidateIntensityTransformerHead.forward` call currently pays on the order of
7-11 sequential host syncs. Only two CPU-side *values* are actually forced by data the CPU cannot
otherwise know — `max_atoms` (SYNC 0) and the per-molecule counts (SYNC 1) — and those two can
share a single small transfer (see step 1). Note also that a data-dependent sync has already fired
~70 lines upstream at `masster_model.py:2114-2117` (`fragment_batch_indices[fragment_keep]`,
then `int(...shape[0])`), so the pipeline is already drained on entry to the head; that bounds how
much queued GPU work any prefetch inside the head can hide behind.

## The insight: SYNC 1 already contains the whole bucketing *plan*

Once the per-molecule counts (`batch_size` ints) are on the CPU, the code has enough information to
decide the entire bucketing plan at **molecule granularity** — which molecules go in which bucket,
and each molecule's start offset into `order` — with a running cumulative sum and no further GPU
queries. The current code re-derives parts of this on the GPU anyway (`isin`, `nonzero`,
`searchsorted`), and `nonzero` reintroduces a sync that the CPU-side plan makes unnecessary.

The plan stops at molecule granularity. **Token-level layout must stay on device**: because `order`
is a stable sort, a molecule's tokens are a contiguous slice, so per-token row/position tensors are
a `repeat_interleave` away on the GPU and should never be materialized as Python lists (step 3).

## Proposed redesign

1. **Transfer only what each consumer needs, in one payload.** `_refine_from_fragment_atoms` needs
   exactly one CPU value — `max_atoms` — not the per-fragment atom counts; the rest of its layout
   (`offsets`, `positions`) is already computed on device and stays there. So the payload is
   `counts.max()` (one element) plus the per-molecule fragment counts (`batch_size` elements), not
   `num_fragments + batch_size` elements. Concatenate them into one `batch_size + 1` tensor.

2. **Respect the dependency order: the wait goes at the *earlier* consumer.** `max_atoms` is needed
   at `bond_fragment_heads.py:468` to allocate `memory` *before* the four atom-decoder layers run;
   `count_values` is not needed until the bucketing loop *after* them. Placing the wait before the
   bucketing loop is therefore **too late** — the atom decoder would already have had to block. One
   payload, copied at entry, waited on once before `max_atoms` is read, and read again for free
   afterwards. The hideable window shrinks accordingly to `input_projection` + `argsort` rather than
   the decoder layers, so do not expect step 2 alone to hide much; its value is that the transfer is
   `batch_size + 1` elements (step 1) and costs one wait instead of two.

   ```python
   # at entry: atom_fragment_indices and component_batch_indices are both already available,
   # so both bincounts can be hoisted here, ahead of any of the head's real work.
   atom_counts = th.bincount(atom_fragment_indices, minlength=num_components)
   mol_counts  = th.bincount(component_batch_indices, minlength=batch_size)
   packed = th.cat((atom_counts.max().reshape(1), mol_counts))      # batch_size + 1 elements
   staging[: batch_size + 1].copy_(packed, non_blocking=True)       # reusable pinned CPU buffer
   done = th.cuda.Event(); done.record()

   ...                                                 # input_projection, argsort, offsets
   done.synchronize()                                  # explicit wait (small transfer, cheap)
   max_atoms = int(staging[0])                         # first consumer: memory allocation
   ...                                                 # four atom-decoder layers
   count_values = staging[1 : batch_size + 1].tolist() # second consumer: already resident
   ```

   The alternative — two separate copies, `max_atoms` waited on early and `mol_counts` waited on
   after the decoder layers — buys the decoder layers back as hiding for the second value at the
   cost of a second round trip and a second event. Worth trying only if profiling shows the single
   early wait is actually costing something; measure before adding the complexity.

   **`.tolist()` is not a wait.** `tensor.to("cpu", non_blocking=True)` does not name a destination
   buffer and PyTorch silently falls back to a blocking copy for pageable destinations; and when the
   destination *is* pinned, reading the CPU tensor — `.tolist()`, indexing, `int(...)` — does **not**
   synchronize against the in-flight copy, so it can read stale buffer contents. The correct form is
   a persistent pinned buffer plus `copy_(src, non_blocking=True)` and an explicit
   `th.cuda.Event.record()` / `.synchronize()` (or `stream.synchronize()`) before the first read.
   See PyTorch's pinned-memory / non-blocking-transfer guidance. A reusable buffer also means
   allocating the pinned staging once per head (sized to `max_batch_size + 1`), not per forward.

3. **Replace the per-bucket `isin` / `nonzero` / `searchsorted` with contiguous-slice gathers.**
   Because `order = argsort(component_batch_indices, stable=True)`, every molecule's tokens are a
   **contiguous slice** of `order`. Bucket membership is therefore a union of ranges whose bounds are
   a running cumsum over `count_values` — pure CPU arithmetic over `batch_size` ints. Do **not**
   build per-token `(row, position)` destination lists in Python: that is O(num_tokens) of
   interpreter work per forward on the side of the profile that is already saturated (the audit puts
   67% in the CPU bucket), and it turns one upload into many. Instead keep every token-sized tensor
   on device:

   ```python
   bucket_token_indices = th.cat([order[s:s + c] for s, c in bucket_slices])  # device slicing
   bucket_counts = counts[molecule_index_tensor]                              # as today
   local_batch_indices = th.repeat_interleave(arange(n_mols), bucket_counts)
   positions = th.arange(total, device=...) - offsets[local_batch_indices]    # unchanged
   ```

   `molecule_index_tensor` stays the only H2D upload, exactly as today, and `local_batch_indices` /
   `positions` keep the same device expressions they have now — `isin`, `nonzero` and `searchsorted`
   all disappear with no new Python per-token loop and no token-sized host→device traffic.

   **Do not claim "only one sync remains" without measuring it.** Constructing CUDA tensors from
   Python lists (`th.tensor(list, device=...)`) is not a device sync but *is* host-blocking through a
   pageable staging copy, and small-tensor construction, `repeat_interleave` and `th.cat` each have
   their own launch and allocator behavior. The sync count after this change is a claim to verify
   with the profiler (same method as `masster_gpu_audit.md`), not to assert from the source.

## Correctness risk and how to bound it

This changes *how* the bucket-to-token index maps are built, not the bucketing policy (same
power-of-2 buckets, same membership, same within-bucket row order and within-row token order). It is
a refactor with a fixed target, so validate against the current implementation as an oracle before
touching `masster_model.py`:

- **Exactness where exactness is available.** `padded_tokens` packing, `padding_mask`, and the
  bucket→token index maps must match **bitwise** — these are gathers and scatters of unchanged
  values with unique indices, so there is no float reassociation to excuse a difference. Assert on
  `th.equal`, not `allclose`.
- **Tolerances only for what flows through attention.** The final `contextualized` output and all
  parameter gradients go through the attention layers, where different kernel selection or batching
  can legitimately perturb the last bits. Compare those with `allclose` at an explicit fp32
  tolerance (start `rtol=1e-5, atol=1e-6`) and record the observed max deviation rather than
  loosening silently.
- **Gradients, not just the forward.** Backward through `index_select`/scatter is where an index-map
  refactor most plausibly breaks (wrong accumulation into a repeated index). Run
  `loss.backward()` on both implementations and compare grads for `input_projection`, the attention
  layer weights, and `inducing_tokens`.
- **Both attention topologies.** `_contextualize_by_molecule` dispatches to
  `_InducedCandidateAttentionLayer` when `num_inducing_points > 0` (128 in the profiled config) and
  `_FullCandidateAttentionLayer` when it is 0. The padding mask is consumed differently in each;
  cover both.
- **Controlled RNG.** The head has dropout. Either `model.eval()` or reseed immediately before each
  implementation's call, otherwise the comparison measures the RNG stream, not the refactor.
- **Edges and both call sites.** Single-molecule batches, one token per molecule, a batch whose
  counts all land in one bucket, and a count distribution from the `unique_fragment_mode: False`
  path (`masster_model.py:2291`, `num_masks > 1`, much larger component counts) as well as the
  merged path.
- Run the whole harness on CPU and CUDA.

`_refine_from_fragment_atoms` gets steps 1-2 only; it has no `isin`/`nonzero` problem, but its
`max_atoms = int(counts.max().item())` is on the same critical path and is the earlier of the two
block points.

## Suggested order of work

1. Equivalence harness for `_contextualize_by_molecule` (old implementation as the oracle), with
   the forward/gradient/topology/RNG/edge coverage listed above.
2. Step 3 (contiguous-slice gathers) in isolation — the correctness-bearing change — verified
   against the harness on CPU and CUDA.
3. Steps 1-2 (single small payload, pinned buffer, event-gated two-point wait) — pure reordering,
   lower risk, verified by the same harness plus a real profiled run (same method as
   `masster_gpu_audit.md`'s measured section) to confirm the sync actually moves off the
   critical path rather than just changing which line the profiler blames, the way the
   `batched_rel_filter_func` fix did. Count the surviving syncs in the profile rather than
   asserting the number from the source.
4. Re-profile the full config and update `masster_gpu_audit.md`'s measured section with the
   result — do not claim a throughput win before that re-measurement, for the same reason the
   rel-filter fix's initial 25% estimate turned out to be a profiler-attribution artifact rather
   than recoverable time.
