# MASShatter — Topology and Assembly Performance Plan

> **Status:** ACTIVE · **Added:** 2026-08-29 · **Updated:** 2026-09-17
> Some CPU stages are implemented; canonical full-training performance gates remain open.

The [CPU-first, optional-GPU backend plan](masshatter_candidate_backend_plan.md) now governs
the order of new work. This document preserves landed-stage history, detailed CPU designs and
their evidence gates. Its original exclusion of selection/config changes applies to the scope
below; the linked follow-up explicitly adds selector improvements and a later optional backend.

## Implementation progress

- Stage 0's synthetic harness now supports unique targets, ranked-mask budgets, warm-up, the static
  graph sidecar, SMARTS augmentation, grouped views, retained/scanned counts, transferred bytes,
  and finer assembly timings. The canonical full-training-step gate still requires a CUDA training
  run with the production DataLoader.
- Stage 1 is implemented with exact overshoot-aware termination and grouped-cache reuse. Permanent
  full-build-then-slice differentials cover raw arrays, complete candidate batches, and gradients.
- Stage 3's single-fill packed host buffer is implemented and permanently compared with field-wise
  concatenation. Its retention gate still requires the canonical CUDA training run.
- SMARTS assembly now builds atom-set lookup entries only for unique representative components and
  caches molecule-local bond/atom lookup data across grouped views. On the synthetic grouped chain
  workload, SMARTS assembly fell from 273.15 ms to 27.47 ms at target 128 and from 1,141.44 ms to
  120.50 ms when all 4,116 masks were retained; candidate totals improved by 2.83x and 3.56x,
  respectively. These are CPU diagnostics, not substitutes for the canonical CUDA retention gate.
- Forced-mask merging now uses cardinality for the base empty/single-cut prefix, examines only
  ranked masks whose cut counts match genuinely additional SMARTS masks, and reuses merged CSR for
  grouped views with the same forced signature. On the target-128 grouped workload this reduced
  forced merging from 290.97 ms to 1.39 ms and candidate total from 386.93 ms to 96.83 ms. With all
  masks retained, forced merging fell from 296.47 ms to 1.31 ms and candidate total from 699.16 ms
  to 409.40 ms.
- Stage 2 is now implemented in the CPU-first follow-up: numeric GIL release, one persistent
  executor, local zero-based outputs and deterministic group remapping. Local full-training
  throughput improved about 9.3%; the canonical L40S gate remains open. See the
  [current evidence](../findings/masshatter_training_speed_2026_09_17.md#cpu-implementation-and-full-training-measurements-2026-09-17).
  Stage 4 is deferred by the current upload-only projection; Stage 5 remains unimplemented.

## Why now

The research constraint is that the MA/MI-unmerged arm is not completing enough optimization to
converge inside the three-hour wall-clock budget. Faster candidate generation is useful only if it
materially increases optimizer steps in that fixed window. The last recorded candidate total was
about 140 ms/batch, while post-refactor mask selection measured 31.7--35.5 ms/batch128, but the
remainder has not been attributed. In particular, SMARTS mechanism assembly runs after the local
kernel's topology and assembly timers have closed.

This plan therefore does not assume that topology or assembly is the next bottleneck. Stage 0
measures their contribution and aborts the work if their removal cannot materially change the
three-hour result.

This follow-up is behavior-preserving. Any implemented stage must retain exact mask order,
component order, unique indices, SMARTS mechanisms, candidate-group semantics, public model
outputs, and proposal-logit gradients. It does not change mask selection or the config schema.

## Phase boundary

For this work:

- **Topology** means connected-component construction, component-signature deduplication, and the
  decision that the retained mask prefix has reached the unique-fragment target.
- **Local numeric assembly** means materializing one molecule's component, membership, boundary,
  and broken-bond arrays.
- **SMARTS assembly** means the Python mechanism/component augmentation after ordinary topology.
- **Batch assembly** means applying global offsets, packing host buffers, and transferring
  candidate indices to the model device.

Moving unique-component hashing into topology changes the meaning of `topology_ms` and
`assembly_ms`; that is the first timing comparability boundary. Parallel local construction creates
a second boundary because summing concurrent per-molecule durations is not batch wall time. Both
boundaries must be recorded in run notes.

## Common evidence threshold

Stage 0 measures the candidate-generation fraction of full training-step wall time. Every later
projection and retention decision is converted through that fraction and reported as expected
**end-to-end optimizer-step throughput**, not only candidate-local speed.

For candidate fraction `c` and a stage that removes fraction `r` of candidate wall time, the
projected step-time fraction removed is `s = c * r`, and the projected throughput gain is:

```text
throughput_gain = 1 / (1 - s) - 1
```

Unless a stage states a stricter gate, it is implemented only when its projected end-to-end gain is
at least 5% and retained only when the measured end-to-end gain is at least 5% on the canonical
unmerged workload. Candidate-local percentages are supporting diagnostics, not acceptance gates.
The complete retained follow-up must demonstrate at least 10% measured optimizer-step throughput
gain; otherwise it has not met the three-hour motivation.

## Durable verification rule

Every differential or oracle used to land a stage belongs under `tests/`, not only in
`scripts/debug/` or a scratch script. Before Stage 1 begins, add permanent versions of the cheap
single-mixture exactness and probability-tie ordering checks that the landed selection refactor
cited but did not retain. Each later stage adds its own reference-path differential before the old
path is removed. Performance harnesses remain under `scripts/debug/`; correctness gates do not.

## Stage 0 — measurement and abort gate

Extend `scripts/debug/benchmark_mass_shatter_candidates.py` before changing the kernels:

1. Add unique-fragment target and ranked-mask-budget arguments.
2. Exercise the static graph sidecar, SMARTS on/off, and configurable candidate-group multiplicity.
3. Add warm-up passes and report medians after warm-up.
4. Report, per molecule:
   - `selected_mask_count`: masks present after forced-mask merging and before the current
     full-topology call;
   - `retained_mask_count`: the current `num_local_masks` after the post-topology truncation block;
   - `retained_mask_count / selected_mask_count`;
   - component occurrences, representative count, unique components, atom memberships, boundary
     memberships, and transferred bytes.
5. Split timings into input preparation, selection, forced-mask merge, connectivity, unique
   deduplication, local numeric assembly, SMARTS mechanism assembly, batch packing, device
   transfer, and probability scoring.
6. Measure candidate wall time as a fraction of a representative full training step for the
   canonical MA/MI-unmerged run, including the actual DataLoader worker configuration.

At the pre-Stage-1 baseline, topology processed 100% of `selected_mask_count`;
`retained_mask_count` was only a proxy for potential early-stopping work. Stage 1 now reports
actual `scanned_mask_count`; use that for current topology work. The ranked-mask config ceiling
is not a valid denominator because probability floors
and finite molecular state spaces can make selection return fewer masks without creating any work
for Stage 1 to remove.

CUDA measurements use external synchronization or CUDA events; training does not gain new
per-batch synchronizations merely for metrics. The benchmark must reproduce current candidate
tensors, while the reference-output differential is committed under `tests/`.

**Abort the whole follow-up after Stage 0** if eliminating topology, local assembly, SMARTS
assembly, and batch packing entirely would improve end-to-end optimizer-step throughput by less
than 10%. Phase shares are reported to route later stages, but no raw candidate-phase-share gate may
veto Stage 1; its retained-prefix reduction is categorically different from the constant-factor
stages.

If the gate fails, profile the actual dominant full-step phase instead. Do not implement Stages
1--5 merely because their code admits optimization.

## Stage 1 — conditional early topology termination

This is the only proposed stage with an asymptotic win: when the unique target is reached early,
work scales with the retained mask prefix rather than all masks returned by selection. It has no
payoff if `retained_mask_count` normally equals `selected_mask_count`.

**Go/no-go:** implement Stage 1 only if the canonical unique-fragment workload has median
`retained_mask_count / selected_mask_count <= 0.75` and Stage 0 projects at least a 5% end-to-end
optimizer-step throughput gain from avoiding the discarded suffix.

Extend `build_component_arrays_raw` with an optional unique target and the number of forced masks
at the front of the mask CSR. For each mask, the compiled kernel will:

1. Build connected components in the existing order.
2. Construct each component signature and look it up in the unique-component hash table
   immediately, preserving first-occurrence unique indices.
3. Finish every component of the current mask.
4. Record the first mask prefix that reaches the target, but continue scanning until a later
   component proves the final unique count exceeds the target.
5. Stop after that overshoot has been observed and every forced mask has been processed.

The kernel returns `scanned_mask_count`; `retained_mask_count` is the length of its output mask
arrays, and the candidate batch reports both. They can differ: the old path retains every selected
mask when the reachable unique count equals the target exactly, and it
truncates only when the final count exceeds the target. Therefore an online kernel cannot safely
stop at equality. It scans through the first overshoot, then materializes only the previously
recorded target/forced prefix. The driver deletes the current full-topology-then-slice block.
Finishing the target-reaching mask is required for parity: the current implementation retains all
components and all newly encountered unique atom sets from that mask, even if the target is crossed
partway through it. If the target is unreachable or reached exactly without an overshoot, the
kernel processes and retains all `selected_mask_count` masks exactly as today.

`_TopologyCache` stores `scanned_mask_count` and the retained prefix of `mask_ptr` and
`broken_bonds`. A grouped member reuses the representative's stopping point rather than deriving
`num_local_masks` independently from its untruncated selection. Topology validation compares the
same retained CSR prefix on cache hits.

The initial implementation may allocate buffers for `selected_mask_count` and fill only the
retained prefix. Reducing allocation capacity is separate and is not required to land early
termination.

Verification committed under `tests/`:

- Compare against the current full-build-then-slice result for targets below, equal to, and above
  the reachable unique count.
- Cover selection returning fewer masks than the config ceiling while retaining all selected masks.
- Cover a target reached before the forced-mask prefix ends.
- Cover grouped reuse where the representative terminates early.
- Cover zero-bond molecules, rings, disconnected graphs, redundant cuts, multigraphs, and graphs
  above 256 atoms.
- Require exact equality of every non-timing candidate field and exact proposal-logit gradients.

## Stage 2 — conditional parallel local construction

This is a constant-factor optimization and is Amdahl-limited. Shared-CE grouping already reduces
the work to one topology task per representative, while SMARTS augmentation, driver glue, and
batch packing remain serial. It also competes with DataLoader worker processes for CPU cores.

Before splitting the kernel, remove its obsolete `list[tuple]` input branch and update the direct
unit tests to supply CSR. Production has only one caller and already supplies CSR; retaining the
test-only representation would complicate the `nogil` boundary and leave the parent refactor's
union-type smell in place.

Then split the Cython entry point into Python validation/allocation and a numeric local kernel that
runs under `nogil`. `perf_counter()` does not run inside the `nogil` worker. The worker returns
untimed local counters and zero-based molecule-local arrays.

The coordinator measures one wall-clock interval around the complete batch of concurrent local
construction tasks. It must not sum per-worker topology or assembly intervals. If connectivity and
local assembly remain fused in each worker, replace the two old metrics with one
`topology_assembly_batch_ms`; split wall-clock metrics require separate batch barriers. Mark this
parallelization as the second timing comparability boundary.

After forced-mask merging, the driver will:

1. Submit one local-construction task per candidate-group representative.
2. Reuse the representative's result for the remaining group views.
3. Apply global offsets only during batch assembly.

Use a lazily created persistent executor cached by configured thread count, so the training process
does not construct and destroy a `ThreadPoolExecutor` every batch. Create it only on first use in
the training process. Do not add OpenMP.

This stage intentionally widens `mass_shatter_mask_selection_threads` to govern all MASShatter CPU
candidate kernels rather than introducing a config migration. When Stage 2 lands, update the
`config/models_template` comment to document that wider scope. The key remains a compatibility name;
no silent second concurrency pool is introduced.

**Go/no-go:** use Stage 0's representative count and serial fraction to compute the four-thread
Amdahl bound. Skip Stage 2 unless the bound projects at least a 5% end-to-end optimizer-step
throughput gain. After implementation, keep it only if local construction improves by at least
1.25x, measured end-to-end throughput improves by at least 5%, and DataLoader throughput does not
fall.

Verification committed under `tests/`: thread counts 1, 2, and 4 produce bit-identical arrays;
grouped and ungrouped construction agree; worker exceptions preserve current validation messages.
Benchmark both many-small-molecule and few-large-molecule batches under the production DataLoader
configuration.

## Stage 3 — conditional packed batch assembly

This is a constant-factor optimization. Implement it only if Stage 0 projects at least a 5%
end-to-end optimizer-step throughput gain from eliminating the measured redundant packing work.

Replace the list-of-chunks, per-field `np.concatenate`, and final cross-field packing sequence with
a size-and-fill pass:

1. Collect the sizes of every local result.
2. Compute prefix offsets for masks, mechanisms, components, unique components, atom memberships,
   boundaries, and broken bonds.
3. Allocate one contiguous `int64` NumPy buffer and define field views into it.
4. Copy each local result directly into its final view while applying global offsets.
5. Create one CPU tensor, perform one device transfer, and split it into the existing tensor
   fields.

This removes one complete host-memory copy and the per-field concatenations without changing the
candidate-batch contract or model forward. Pinned asynchronous transfer is deferred: variable-size
pinning can add a copy and is retained only if an end-to-end CUDA benchmark shows a benefit.

Verification committed under `tests/`: compare exact values and ordering against the old chunk
implementation for mixed molecule sizes, empty boundary arrays, grouped views, unique and
non-unique modes, and SMARTS mechanisms. Record host bytes copied and device-transfer bytes in
addition to elapsed time. Keep the stage only if measured end-to-end throughput improves by at
least 5% on the canonical workload.

## Stage 4 — optional unique-aware membership contract

This stage is last among the topology/assembly changes because it touches the model forward and
gradient path, while its proposed replacement is a host-side version of an already vectorized
device-side filter. Evaluate its evidence gate against the **post-Stage-3 tree** if packed assembly
lands; Stage 3 changes the marginal cost of every byte Stage 4 would remove.

Proceed only if the post-Stage-3 measurement shows that removing full occurrence memberships
projects at least a 5% end-to-end optimizer-step throughput gain. Membership share of transferred
candidate bytes is reported as an explanatory metric, not a substitute acceptance threshold.

The mode split is real and must be explicit: 39 of 65 active MASShatter configs enable unique mode,
while 26 use the template's non-unique default. Do not add top-level fields whose meaning or
validity silently changes with the mode. If the gate passes, represent memberships with explicit
internal typed variants, for example `UniqueComponentMemberships` and
`PathwayComponentMemberships`, while keeping common pathway metadata in
`MassShatterCandidateBatch`.

Unique mode still needs every latent component's mask, cut count, mechanism, and unique-component
mapping, but only representative atom and boundary memberships. Its compact membership object must
carry every value currently returned by `_unique_component_memberships`:

```text
unique_batch_indices
unique_atom_indices
atom_unique_indices
unique_boundary_bond_indices
boundary_unique_indices
boundary_counts
```

SMARTS assembly remains after ordinary topology. When a SMARTS component introduces a new atom
set, append its atom and boundary membership once; when it matches an existing atom set, reuse the
existing unique index. SMARTS components remain outside the stopping decision, matching current
behavior.

Verification committed under `tests/`: the compact variant exactly equals the current
`_unique_component_memberships` transformation, including SMARTS cases. The pathway variant
preserves non-unique behavior. Public prediction outputs, losses, gradients, and checkpoint
compatibility remain unchanged. Drop this stage if measured end-to-end throughput does not improve
by at least 5%.

## Stage 5 — optional grouped-view compaction

Do not combine this with Stages 1--4. If shared-CE batches still spend material time in assembly
and pooling afterward, retain topology, memberships, and topology-derived fragment/boundary
embeddings once per representative precursor. Carry spectrum-to-representative maps and fan out
the compact embeddings immediately before CE-conditioned intensity scoring. Per-spectrum
prediction indices, diagnostics, and losses remain expanded.

This is a model-forward change rather than a candidate-kernel optimization and therefore gets its
own permanent differential, parity gate, gradient gate, and common 5% end-to-end throughput gate.

## Acceptance criteria

1. Stage 0 either aborts the follow-up or provides measured phase fractions and projected
   three-hour optimizer-throughput impact for every stage taken forward.
2. Every correctness differential used to land a stage remains under `tests/`.
3. Every non-timing internal candidate field, every public model output, and proposal-logit
   gradients match the reference path exactly unless an explicitly typed internal membership
   contract replaces an old field.
4. Focused candidate, component-array, SMARTS, model, and Lightning tests pass, followed by the
   full suite.
5. Early termination, if admitted, scales with `retained_mask_count / selected_mask_count` and
   handles grouped early-stop reuse exactly.
6. Parallel timing is batch wall time, never a sum of overlapping worker durations.
7. Every retained stage improves measured end-to-end optimizer-step throughput by at least 5%, and
   the complete follow-up improves it by at least 10%; otherwise drop the stage or stop the work.

The original gated evaluation order was: **measurement/abort → early termination if the retained/selected
distribution supports it → parallel local construction if the Amdahl bound supports it → packed
batch assembly if host copies support it → unique membership compaction if post-packing transfer
cost supports it → optional grouped-view compaction**. Only Stage 1 changes asymptotic work; the
remaining stages must justify their added complexity with measured constant-factor gains. The
linked backend plan orders the remaining work without repeating landed stages.

## Related

- [masshatter_candidate_pathway_refactor.md](masshatter_candidate_pathway_refactor.md) — landed
  selection/candidate-path refactor that exposed this measurement question.
- [masshatter_plan.md](masshatter_plan.md) — parent MASShatter research plan.
- [masshatter_nonprobabilistic_intensity_plan.md](masshatter_nonprobabilistic_intensity_plan.md) —
  unmerged objective and proposal-architecture gates constrained by the same three-hour budget.
