# MASSter — Candidate-Generation Pathway Refactor

> **Status:** LANDED · **Updated:** 2026-08-29
> All five layers implemented and verified behavior-identical on 72 configurations. Exploration deleted. Selection is hoisted out of the assembly loop, `mixtures=1` now threads, and two latent bugs surfaced (see Outcome). Committed in `ebfb64ed`; the 2026-08-29 follow-up cleanup below is uncommitted. 2304 tests pass.

## Why now

Candidate generation was measured at **~40% of training wall clock** on run `mbar8hpu`
(`train_candidate_total_batch_ms` 1058 ms/batch, of which mask selection 915 ms, against
~275 s epochs at ~100 batches/epoch). Optimizing it pulled that to ~140 ms/batch, but each
optimization was bolted onto the existing branch structure rather than simplifying it — the
threading work added a *fourth* arm to an already 4-way selector. The structure is now the
limiting factor on further work, not the algorithms.

Two facts make a refactor tractable that would not have been before:

1. **Exploration is dead** (user, 2026-08-28). It is the keystone of two of the seven dimensions.
2. **The mixture selection kernel is now nogil and CSR-native**, and the single-component kernel
   is already CSR-native, so the production selection path has no Python-object representation
   left to accommodate.

## Current pathways

`build_masster_candidates` in
[masster_candidates.py](../../../src/fragnnet/frag/masster_candidates.py) spans **1249
lines** and branches on seven independent dimensions:

| # | Dimension | Effect |
|---|-----------|--------|
| 1 | `explore_masks == 0` vs `> 0` | Splits selection into two sibling blocks: L517–693 and L694–871 |
| 2 | Mask representation | CSR tuple vs `list[tuple]` — a **union type** |
| 3 | `num_mixtures == 1` vs `> 1` | Two different kernels |
| 4 | `max_unique_components` set or not | Two forced-mask merge policies, ~60 lines each |
| 5 | `candidate_group` | Two caches: `ranked_mask_cache`, `topology_cache` |
| 6 | Mask-selection threading | Added a third/fourth selector arm |
| 7 | `static_graph` / device | Three-way CPU-copy branch |

### Smell 1 — the union type

`broken_mask_input` is a CSR tuple on the non-exploration path and a list of Python tuples on the
exploration path. `build_component_arrays_raw` accepts either, so the driver carries
`isinstance(broken_mask_input, tuple)` checks at L910 and L958 to tell them apart.
**Exploration is the only reason this union exists.**

### Smell 2 — the four-way selector

At L538–570:

```python
if   cached_ranked_masks is not None:              # group cache hit
elif num_mixtures == 1:                            # single-Bernoulli kernel
elif molecule_index in precomputed_mixture_pools:  # threaded pool
else:                                              # serial pool
```

Arms 3 and 4 compute the same thing by different routes. This is also why threading covers only
`num_mixtures > 1`: the `mixtures=1` arm calls `top_k_cut_masks_csr_raw`, which holds the GIL for
its whole body, so it stays single-core (measured 39.6 ms/batch128: 29.1 ms in the kernel,
10.4 ms driver glue).

`precomputed_mixture_pools` is a batch-local
`dict[molecule_index, (mask_ptr, broken_bonds)]`. Before the main molecule loop, worker threads run
mixture selection for each molecule that needs computation; when `candidate_group` sharing is
enabled, only the first molecule in each group is submitted. The loop then consumes the stored CSR
arrays instead of invoking the same kernel serially. Despite the variable name, each value is
already the final deduplicated, mixture-scored, floor-filtered, and limit-truncated mask selection,
not the larger intermediate component pool. The dictionary exists only to carry threaded results
across the boundary into the serial assembly loop; L2 removes that boundary and therefore removes
the dictionary.

## Proposed design

The unifying observation: **per molecule, mask selection is a pure function**

```
(logits[num_bonds, M], mixture_log_weights, limit, floors) -> CSR(mask_ptr, broken_bonds)
```

and everything else is downstream. Once exploration is gone this holds with no exceptions, so CSR
becomes the single currency and dimensions 1, 2, 3 and 6 collapse.

```
L1  select_masks(logits[nb,M], weights, limit, floors) -> CSR       nogil, per molecule
L2  select_masks_batch(logits_csr, groups, ..., threads) -> [CSR]   group dedup + thread pool
L3  merge_forced_masks(ranked, forced, unique_mode) -> CSR          one function, one flag
L4  build_component_arrays_raw(...)                                 unchanged, always CSR
L5  _packed_long_tensors(...)                                       unchanged
```

**L1** folds `M == 1` and `M > 1` behind one entry point, skipping pooling/dedup/rerank when there
is nothing to pool. The public entry point is shared, but its
compiled body dispatches to two numeric implementations: `M == 1` preserves the existing
`(penalty, flip_positions)` ordering and `M > 1` preserves mixture probability plus lexicographic
tie-breaking. The single-component implementation must move its enumeration body under `nogil`;
calling today's `top_k_cut_masks_csr_raw` from a worker would still hold the GIL. This drops the
1.55× pooling overhead `M == 1` pays if routed through today's mixture kernel (40.3 ms vs 26.0 ms
per batch128 in the isolated selection benchmark) without changing its selected-mask prefix.

**L2** owns group dedup, the thread pool, and the "identical logits" precondition — checked once
up front for every member of a group instead of per molecule on every cache hit. Grouped entries
must have identical local-logit shapes and values and identical mixture weights; topology remains
validated by the topology cache downstream. L2 replaces `ranked_mask_cache`,
`precomputed_mixture_pools`, and the serial fallback with one path. Selection leaves the
per-molecule loop entirely. `threads=1` uses the same batch function without creating a pool.

**L3** turns the two inline merge policies into one function. Forced masks are constructed from
the molecule's explicit bond count, not from a representation-specific logits container.

**L4** loses both `isinstance` checks.

The loop then does only: merge forced masks → topology → accumulate. Estimated **1249 → ~400
lines** plus three small named functions.

## Deleting exploration

Exploration is the keystone: dimensions 1 and 2 both go with it. Removable:

- The L694–871 block (~178 lines) and both sub-modes (`bernoulli_replace`, `bounded_additive`)
- MASSter's imports and uses of `RankedCutMask`, `iter_ranked_cut_masks`, and
  `top_k_cut_masks`; these symbols remain in `bond_mask_generator.py` for its independent public
  fragment-generation API
- `bounded_additive_masks_raw`, which has no caller after MASSter exploration is removed
- The `generator` parameter
- Active model plumbing for `masster_explore_masks`, `masster_exploration_mode`,
  `masster_exploration_temperature`, and
  `masster_exploration_cut_count_weights`; the keys remain temporarily in the template as
  deprecated compatibility entries
- `exploration_mask_count` and `exploration_mean_cut_count` from
  `MassterCandidateBatch`, `MassterModel` outputs, `MassterPL` loss metrics, and the
  special weighted reduction in `SpectrumPL`

The generic mask-generator tests are not exploration tests and remain unchanged. MASSter tests
that exercise `bernoulli_replace` or `bounded_additive` are deleted; numeric CSR parity tests keep
using the generic iterator as an independent oracle.

### Config and checkpoint compatibility

The four exploration keys cannot be removed from `config/*_template.yml` immediately: checked-in
experiment YAMLs are required to contain only keys present in the template, and historical custom
configs would otherwise fail before model construction. Use one deprecation window:

1. Keep the four keys in `config/*_template.yml`, grouped under a retired-feature comment. Their
   values remain the historical defaults.
2. `runner.validate_sampler_config` rejects a new training config with
   `masster_explore_masks > 0` using an explicit "exploration has been retired" error. The
   other three values are accepted but ignored while the keys remain in the schema.
3. Remove the four keys from active checked-in MASSter configs. Legacy exploration configs stay
   under `config/masster/legacy/` as historical records and continue to parse far enough to
   produce the explicit retirement error.
4. A later schema cleanup may remove the deprecated template keys together with the remaining
   legacy config references; that cleanup is not part of this refactor.

Historical checkpoints do **not** need a tensor migration for this change. `MassterPL` accepts
checkpoint hyperparameters through `**kwargs`, so retired extra keys can remain in old checkpoint
metadata and are ignored by `_setup_model`. Add a checkpoint-load regression test using historical
exploration keys rather than creating a migration script.

### Known behavior correction — forced single cuts for mixtures

Current `M > 1` code sets `detached_logits = []` and then builds forced single cuts from
`range(len(detached_logits))`. Consequently, `include_single_bond_cuts=True` does not force any
single cuts for a Bernoulli mixture. L3 intentionally fixes this by using the molecule's bond
count.

This is a behavior correction, not refactor parity. Land and test it as an isolated first stage so
the subsequent differential tests compare against the corrected baseline. Record the change in
the parent MASSter plan because it can alter candidate support for an `M > 1` checkpoint that
enables forced single cuts.

## Decision — `mask_selection_ms` semantics

The threading change already in the tree **silently altered what this metric means**. It was the
sum of per-molecule elapsed times; it is now pool wall-clock plus per-molecule glue. So
`train_candidate_mask_selection_batch_ms` in wandb is no longer comparable to pre-threading runs —
it drops by roughly the thread count even though total CPU work is unchanged.

Keep `mask_selection_ms` as phase wall-clock time, matching the other fields in
`MassterCandidateTimings`. L2 measures one interval around the complete batch selection call;
the per-molecule loop does not add selection time. Do not add a CPU-time metric in this refactor.
Mark the threading change as the comparability boundary in run notes: pre-threading and
post-threading `train_candidate_mask_selection_batch_ms` values are not directly comparable.

## Implementation stages and verification

### Stage 0 — isolate the known behavior correction

- Add a failing test showing that `M > 1` plus `include_single_bond_cuts=True` retains every
  single-bond mask even when it falls outside the learned top-k prefix.
- Build forced masks from the explicit local bond count.
- Update the parent MASSter plan with the behavior and reproducibility note.

**Gate:** the new regression passes and existing candidate tests pass before structural changes.

### Stage 1 — retire exploration with compatibility

- Remove the exploration branches, generator plumbing, raw sampler, output diagnostics, model
  attributes, and logging metrics listed above.
- Keep the generic `bond_mask_generator` API.
- Apply the config deprecation behavior and checkpoint-load regression described above.

**Gate:** all non-exploration tests pass; a historical-style checkpoint with the four extra hparams
loads; active configs load; a legacy config with `explore_masks > 0` fails with the retirement
message.

### Stage 2 — introduce L1 and L2

- Extract a common CSR selection entry point with specialized `M == 1` and `M > 1` compiled paths.
- Move single-component enumeration under `nogil` without changing its arithmetic or tie order.
- Batch group validation, deduplication, and threaded execution in L2.

**Gate:** exact differential tests compare the pre-refactor corrected baseline with L1/L2 for:

| Dimension | Required cases |
|-----------|----------------|
| Mixtures | `M=1`, `M=2`, tied logits, nonuniform mixture weights |
| Execution | `threads=1`, `2`, and `6` |
| Sharing | no groups, valid repeated groups, mismatched logits, weights, and topology |
| Limits | fixed mask budget and unique-fragment budget |
| Floors | disabled, absolute, relative, and excludes-all |
| Molecules | zero bonds, one bond, rings, disconnected inputs, batch mixtures of sizes |

For valid inputs, compare every CSR array and every final candidate index tensor exactly; compare
floating outputs with zero tolerance where the old tests require bit parity. Invalid inputs must
raise the same exception class and equivalent message.

### Stage 3 — extract L3 and simplify the driver

- Move both forced-mask policies into `merge_forced_masks`.
- Make component assembly CSR-only and remove the two representation checks.
- Keep topology caching, SMARTS assembly, and packed output transfer behavior unchanged.

**Gate:** extend the differential matrix across `include_single_bond_cuts` on/off, SMARTS templates
on/off, grouped topology reuse, and unique-fragment truncation. Run the focused MASSter test
suite and the candidate benchmark. The refactor is complete only if candidate selection and
topology outputs are unchanged from the corrected Stage 0 baseline and `M == 1` selection
demonstrates parallel speedup without regressing `M > 1` selection materially.

## Measurements this rests on

All on 128 molecules × 30 atoms, `max_masks=256`, `max_enumerated_masks=1024`, CPU, 6 physical
cores. Reproduce with `scripts/debug/bench/benchmark_masster_candidates.py`.

| Path | When | mask_selection | State |
|------|------|----------------|-------|
| A — mixture pool | `explore=0`, `M>1` | 105 → **41 ms** (6 threads) | Threaded |
| B — single Bernoulli | `explore=0`, `M=1` | 38.9 ms | GIL-held C, single-core |
| C — exploration | `explore>0` | 76.9 ms | Python objects — **being deleted** |

Mixture cost is superlinear in `M` because each component is enumerated to full depth, pooled,
scored under all `M` components, then truncated back to `limit`:

| M | ms/batch128 | vs M=1 | pooled | kept |
|---|-------------|--------|--------|------|
| 1 | 40.3 | 1.00× | 1024 | 1024 |
| 2 | 88.6 | 2.20× | 2048 | 1024 |
| 3 | 147.9 | 3.67× | 3072 | 1024 |
| 4 | 222.7 | 5.53× | 4096 | 1024 |

Dedup recovers almost nothing — once components differ at all, only ~2% of the pooled masks are
duplicates — and the surviving split is lopsided and unpredictable (at moderate component spread,
1011 of the final 1024 came from one component and 58 from the other). A bound-based prune could
skip exact scoring for masks that cannot reach the top `limit`, but that **changes which masks are
selected**, so it is out of scope here and needs its own gate.

## Outcome

Landed 2026-08-28. Verified against a 72-configuration reference (mixtures 1/2/3 × unique/non-unique
× single-cuts on/off × grouped/ungrouped × three floor settings) comparing every tensor field of
`MassterCandidateBatch`: **0 mismatches**. Plus the 1080-case mixture-kernel reference, a
400-trial `M=1` equivalence gate, and an exhaustive tie-break check. Full suite 2291 passed.
Committed in `ebfb64ed`. See **Follow-up cleanup** below for what the first pass left behind.

### Structure

| Symbol | Lines |
|--------|-------|
| `build_masster_candidates` | 1249 → **871** |
| `_select_masks_batch` (L2, new) | 98 |
| `_merge_forced_masks` (L3, new) | 113 |

The ~400-line estimate was optimistic: it did not account for the SMARTS mechanism assembly
(~180 lines) or the topology cache/reuse block, neither of which the five layers covered. Both are
cohesive and self-contained, and are the obvious next extractions if the driver needs to shrink
further.

### Two bugs surfaced

1. **`include_single_bond_cuts` was a no-op for `num_mixtures > 1`.** The pre-refactor code derived
   the bond count from a list it deliberately left empty for mixtures, so the flag silently forced
   nothing. **No shipped config combines the two**, so nothing in the record is affected. Behavior
   was preserved rather than fixed — see the `NOTE:` at the `forced_bond_sets` construction. Fixing
   it is a live decision, since it changes which masks are selected.
2. **`_compare_bitsets` implemented the wrong order** (introduced with the nogil kernel). For sets
   differing first at bond `k`, whether the owner of `k` sorts first depends on whether the other
   side has any bond above `k` — a prefix sorts first (`(1,2) < (1,2,3)`) but a smaller leading
   bond also sorts first (`(1,2) < (1,3)`). The comparator assumed one uniform rule. **Fixed and
   now exhaustively verified** over every subset for 1–8 bonds under full probability ties.
   It never fired before L1 because random logits give distinct probabilities, so the lexicographic
   tie-break is unreachable; routing `mixtures=1` through the kernel exposed it via the existing
   `logits=[0,0,0,0]` test.

### Performance

`mask_selection`, ms per batch of 128 × 30 atoms:

| | threads=1 | threads=4 | threads=6 |
|---|---|---|---|
| `mixtures=1` | 43.6 | 32.6 | **31.7** |
| `mixtures=2` | 113.3 | 40.1 | **35.5** |

`mixtures=1` was **38.9 ms and single-core-only** before; it now threads to 31.7 ms. It is ~12%
slower if run at `threads=1`, because the unified kernel still allocates pool buffers and runs a
scoring pass for the probability floors that the old dedicated kernel folded into enumeration. The
`M=1` fast path already skips the pooling, dedup, and rerank.

### Config keys

Removed from the template and from 77 configs that set them to no-op values. **10 configs that
actually used exploration were deliberately left untouched** so they fail loudly at
`runner.py:61` rather than silently becoming non-exploration runs under names like `explore512`.
Nine are under `config/masster/legacy/`; the tenth is
`config/masster/merged/baseline/nist20_merged_mask64_gine384_mixed_topk_local8g.yml`, which is
**not** in `legacy/` and is the config behind the most recent local wandb run — it needs an explicit
decision.

### Follow-up cleanup — 2026-08-29

A review of the landed code found three loose ends from the Stage 1 removal list. The first two are
now done; the third is a live decision.

**Exploration diagnostics were still plumbed end to end.** `exploration_mask_count` and
`exploration_mean_cut_count` survived as constants: the driver appended `0` and `0.0` per molecule,
`build_masster_candidates` built two zero tensors and moved them to the model device every
batch, and both were logged to wandb each epoch. `SpectrumPL` also still carried the
exploration-only reduction — a weight multiplication in `_accumulate_running_metrics` and two
`sum_w == 0` guards in `_reduce_running_metrics` — which by then could only ever see zero weight.
All of it is removed: the two `MassterCandidateBatch` fields, the accumulators and tensors in
[masster_candidates.py](../../../src/fragnnet/frag/masster_candidates.py), the output-dict
entries in [masster_model.py](../../../src/fragnnet/model/masster_model.py), the `loss_fn`
parameters, `loss_values` entries and `loss_names` in
[masster_pl.py](../../../src/fragnnet/pl_model/masster_pl.py), the special-case reduction in
[spectrum_pl.py](../../../src/fragnnet/pl_model/spectrum_pl.py), and the assertions in four test files.
The metric reduction path is now uniform for every metric.

**`bounded_additive_masks_raw` is gone** from
[masster_core.pyx](../../../src/fragnnet/frag/masster_core.pyx), together with the
`_splitmix64` and `_uniform_open` PRNG helpers that existed only to feed its Gumbel top-k sampling
— 172 lines, 1787 → 1615. Every remaining `cimport` is still used. The extension was rebuilt with
`python setup.py build_ext --inplace`. The only surviving mentions of `bounded_additive` are the
legacy config YAMLs, which already fail at load, and these docs.

**Stage 1's config deprecation window was never built.** The four keys were removed from
`config/*_template.yml` outright, so the 10 configs listed under **Config keys** fail at
`runner.py:61`, inside `load_config`, with a bare `AssertionError` naming the missing keys. A check
in `validate_sampler_config` would not fix this: `load_config` fails first, and sampler validation
does not run until after `init_dataset`. The correct compatibility fix is a retired-key check in
`load_config`, immediately before the unknown-key assertion, that raises a self-explanatory error
without restoring dead keys to the active template.

One active config still needs an owner decision before another run:
`config/masster/merged/baseline/nist20_merged_mask64_gine384_mixed_topk_local8g.yml` requests 64
`bounded_additive` exploration masks. Either archive it as a retired experiment or define a new
deterministic replacement experiment; silently deleting the exploration settings would change the
experiment represented by its name and history. Resolve that choice before starting the topology
and assembly follow-up below.

Re-verified against the recompiled extension: exhaustive tie-break order for every subset of 1–8
bonds under full probability ties, `M=1` exactness against brute-force enumeration over 300
randomized cases (logits, limits, absolute and relative floors; 900 counting the pre-rebuild run),
and the full suite at 2304 passed plus one failure,
`test_data_utils.py::test_rdkit_ml_standardize_neutralizes_phosphorus_oxide`, which is pre-existing
and unrelated (rdkit leaves `[P+][O-]` unneutralized).

Two things this review could not check, worth knowing:

- **None of the verification harnesses cited above are in the tree.** The 72-configuration
  differential, the 1080-case mixture reference and the 400-trial `M=1` gate were scratch scripts,
  so nothing protects the parity claim from a future regression. The tie-break and `M=1` exactness
  checks are cheap and belong in `tests/`.
- **`M > 1` selection is not globally exact when the pool is truncated.** Each component is
  enumerated to its own top-k, then pooled and reranked, so a mask with high mixture probability
  that is outside every component's top-k cannot be selected — measured at 1 case in 400 random
  configurations with a tight `max_enumerated_masks`. Inherent to pool-then-rerank and unchanged by
  this refactor; it disappears as the enumeration budget grows relative to `max_masks`.

## Proposed performance follow-up

Topology and assembly optimization is not part of this landed refactor. Its measurement gates,
abort criteria, and conditional implementation stages live in
[masster_topology_assembly_performance_plan.md](masster_topology_assembly_performance_plan.md).

## Related

- [masster_plan.md](masster_plan.md) — the parent plan this code implements.
- [masster_rl_exploration_plan.md](masster_rl_exploration_plan.md) — already superseded
  2026-08-19; deleting the exploration code path closes out its remaining surface.
