# MASShatter candidate generation: CPU improvements, then an optional GPU backend

> **Status:** ACTIVE · **Updated:** 2026-09-17 · CPU implementation is validated locally;
> full CUDA passes local correctness/speed gates and final regressions. L40S checks remain.

## Objective and scope

Increase training throughput within a three-hour L40S allocation while preserving MASShatter's
candidate universe, ordering, probability marginalization, losses, and checkpoint parameters.
CPU candidate construction remains the default and correctness reference. GPU construction is an
explicit option; target L40S acceptance remains separate from the measured local speedup.

This is the execution plan for the new work. The
[topology/assembly plan](masshatter_topology_assembly_performance_plan.md) retains the detailed
CPU kernel designs, landed-stage history, and performance gates. Its early termination and packed
host assembly already exist; do not reimplement them. The new selector work and optional backend
extend that plan's original scope, which excluded selection and configuration changes.

Smaller enumeration budgets, random/approximate sampling, mixed precision, changed objectives,
and altered chemistry are separate experiments. Keep U256, E4096 and FP32 for the primary comparison.

## Established evidence and current state

The [training-speed finding](../findings/masshatter_training_speed_2026_09_17.md) is the canonical
source for W&B run `s3gy1ro4`, phase timings, the reduced-monitoring baseline, and batch-128/256/512
configs. Its host timers identify topology and assembly as larger targets than selection; they
are not isolated CUDA timings or predictions of recoverable wall time. The local larger-batch
test did not establish a throughput gain. The implementation checkpoint below records subsequent
CPU and GPU measurements.

Relevant boundaries at planning start (the implementation checkpoint records changes):

- [Candidate builder](../../src/fragnnet/frag/mass_shatter_candidates.py): detached logits move
  to CPU; threaded exact selection precedes serial local topology; sparse indices return to GPU.
- [Cython kernels](../../src/fragnnet/frag/mass_shatter_core.pyx): ranked heap enumeration already
  releases the GIL. Single-mixture scoring still repeats work when probability floors are disabled;
  numeric CSR output loops retain the GIL. Local topology/assembly is not yet a parallel worker.
- [Model](../../src/fragnnet/model/mass_shatter_model.py): unique mode discards duplicate
  occurrence memberships after upload. Differentiable mask probabilities and latent-pathway
  marginalization must remain connected to the live logits.
- [Training benchmark](../../scripts/debug/benchmark_mass_shatter_training.py): fixed-batch
  forward/backward diagnostic only; production loading, optimizer updates and validation are
  currently excluded. It is not yet the acceptance benchmark for this plan.

## Invariants and backend contract

For identical graph inputs and detached proposal logits, preserve exact selected mask identities
and tie ordering, forced-mask order, component order, first-occurrence unique representatives,
group reuse validation, and all latent pathway mappings. Component ordering follows the existing
maximum-atom-ID convention. Preserve the strict overshoot rule: reaching U256 alone does not prove
that truncation is valid; honor the forced prefix and retain the same prefix as the CPU reference.

Duplicate atom sets may share memberships, but duplicate latent pathways still contribute to
probability marginalization. Prediction shapes, chemistry, losses, trainable parameters, and
checkpoint tensor names/shapes stay unchanged. CUDA reduction differences require explicit
numerical tolerances and gradient checks; they do not excuse different discrete candidates.

The GPU integration introduces this setting in
[template.yml](../../config/template.yml), with matching config comments and validation:

```yaml
# Candidate construction only, independent of the CPU backend's model device.
mass_shatter_candidate_backend: cpu  # cpu | cuda
```

- `cpu`: improved Cython/CPU construction, with the existing device-side probability computation.
  The encoder and intensity head can still run on GPU. Existing configs resolve to this value.
- `cuda`: GPU mask selection, topology and numeric assembly for explicitly supported modes.
  Initial target: one mixture, disabled probability floors, unique mode, SMARTS off, including
  forced single cuts and valid grouped views. Publish the supported shapes/dtypes/modes with the
  implementation. Other modes continue to work through an explicitly selected `cpu` backend.
- Requesting `cuda` without a usable CUDA device/kernel or with an unsupported mode raises a
  clear error. No automatic selection, silent CPU fallback, or silent mask truncation on overflow.
  Validate configuration at setup and tensor device at invocation: models are normally built on
  CPU before device transfer. Optional CUDA imports/builds must not break CPU installation or execution.
- The hybrid prototype below is a benchmark variant named `cuda_topology`, not a third public
  backend or a claim that ranking has moved to GPU. The `cuda` value ships only after full backend
  integration and its correctness/performance checks pass.
- Add explicit legacy migration of a missing key to `cpu` at config/checkpoint compatibility
  boundaries. Cover repository loaders and direct Lightning checkpoint loading; do not add
  fallback hyperparameter reads in the model. Explicit values and invalid values remain visible.
- Record the resolved backend after W&B initialization and on resume, because saved run config
  can overwrite local settings. Keep backend choice out of the trainable state dictionary.

## Stage 0: establish the acceptance benchmark

1. Use the existing fast config as the primary CPU baseline, with both monitoring switches off.
   Fix source/diff identity, environment, data split, ordered spectrum IDs, weights, seed,
   determinism, U/E budgets and optimizer settings. Compare backend variants within each batch
   size; do not attribute batch-size changes to a backend.
2. Extend the diagnostic harness or add a narrowly scoped full-training benchmark that includes
   production DataLoader workers, batch transfer, loss, backward, gradient accumulation and
   optimizer updates. Use warmup and repeated equal-spectrum windows, alternating variant order.
   Record cold compilation/setup separately. Use synchronization at benchmark boundaries and
   profiler/CUDA events for diagnosis, without adding per-molecule synchronization to training.
3. Benchmark 128×2 and 256×1, both effective batch 256; evaluate 512×1 separately because its
   effective batch is 512. Include representative small/large molecules, grouped views and
   memory stress batches. Preserve the existing CPU allocation while measuring each change.
4. Record spectra/s, optimizer steps/s, batch latency distribution, peak GPU allocated/reserved
   memory, host memory, GPU utilization, CPU thread/worker counts, candidate counts, and transferred
   bytes. Separate selection, topology/assembly, packing, transfers and GPU model work. Concurrent
   worker durations must not be summed and called batch wall time.
5. Confirm retained stages on L40S; local GPU measurements are diagnostic. A subsequent matched
   three-hour training check includes validation/checkpoint overhead and reports total spectra,
   optimizer steps, epoch times and validation quality at comparable training progress.

Use the [existing gates](masshatter_topology_assembly_performance_plan.md#common-evidence-threshold):
project at least 5% full-step throughput gain before a substantive stage; retain it only with at
least 5% measured gain. Treat the small selector changes in A1 as one stage. The retained CPU
follow-up targets at least 10% aggregate improvement over Stage 0. Reprofile after every retained
stage; a failed gate closes that stage with a recorded result instead of adding unmeasured work.
GPU stages are judged against the completed CPU baseline, not the slower original configuration.

## Phase A: CPU improvements first

### A1. Remove unnecessary selector work

- Specialize one-mixture, zero-floor selection to skip probability rescoring and unused score
  buffers. Keep exact heap order and CSR output. Preserve the general mixture/floor path.
- Guard the optimization's numerical domain: finite raw float64 inputs can still produce
  nonfinite intermediate sums. Test extreme logits and retain reference behavior outside the
  proven safe case; production FP32 logits do not justify changing the raw API contract.
- Allocate NumPy output arrays with the GIL, then release it around typed numeric CSR count/fill
  loops. Existing selection threads can overlap more of the call.
- Rebuild the Cython extension in the configured environment before differential tests and timing.

### A2. Parallelize topology and local assembly

Apply [Stage 2's design](masshatter_topology_assembly_performance_plan.md#stage-2--conditional-parallel-local-construction):
separate validation/allocation from numeric `nogil` work, dispatch one task per distinct candidate
group representative, and collect results in input order before applying global offsets.

Use one lazily created, process-local persistent executor for selection and local construction,
without nested pools or OpenMP. Retain `mass_shatter_mask_selection_threads` as the compatibility
key and update its comment to cover CPU candidate kernels. Preserve inline execution at one
thread, worker error propagation, and safe teardown/process creation. Sweep 1/2/4 threads with the
production DataLoader; more workers can compete for the same allocated cores. Apply the existing
additional 1.25× local-construction gate and check that loading throughput does not fall.

### A3. Compact unique memberships before upload, if still material

Use [Stage 4's contract](masshatter_topology_assembly_performance_plan.md#stage-4--optional-unique-aware-membership-contract):
materialize representative atom/boundary memberships once in unique mode while retaining every
pathway's mask, mechanism and unique mapping. Use explicit internal membership types and update
consumers together; non-unique and SMARTS behavior must remain valid. Compare with the current
device-side `_unique_component_memberships` transformation before removing duplicate transfers.

### A4. Stream ranked masks into topology only if the unused suffix justifies it

After A1–A3, measure enumerated versus scanned/retained prefixes. If meaningful selection work
remains unused, make the exact single-mixture heap resumable and feed bounded chunks into the
existing topology/unique state. Preserve forced merging, ties, strict overshoot stopping and group
reuse. Other modes retain the existing CPU algorithm unless independently validated.

Audit diagnostics before implementation: `proposal_single_cut_fraction` currently describes the
full learned pool, and selected/scanned counts have distinct meanings. Do not silently relabel
prefix statistics as full-pool statistics. Preserve them cheaply or defer this stage until an
explicit diagnostic-contract change is designed and documented. Smaller E budgets are not an
equivalent substitute for exact streaming.

**Phase-A exit:** each attempted stage has parity and benchmark evidence or a documented failed
gate; freeze the best CPU baseline and its reference fixtures. Then begin Phase B. Do not continue
speculative CPU work indefinitely to satisfy a target that measurements reject.

## Phase B: develop the optional GPU solution

### B1. Prototype GPU topology and assembly with fixed CPU masks

Keep CPU exact selection and forced merging. Transfer compact masks/graph indices, then run
batched connected components, atom-set deduplication and membership assembly on GPU. Parallelize
across molecule/mask pairs with tiled scratch space, sparse edges and multiword atom bitsets.
Canonicalize component and unique ordering explicitly; unordered atomics/hash insertion cannot
define representative identities. Deduplication must compare full signatures, not trust hashes.

Process bounded mask tiles and apply the same retained-prefix decision. Extra speculative work
within a tile must not change retained outputs; report physical masks processed separately from
the reference logical stopping count. Avoid dense `[batch, masks, atoms, atoms]` storage and
per-mask Python kernel launches. Compare every discrete output against the frozen CPU backend.
This prototype retains CPU ranking and its synchronization; measure its actual full-step gain.

### B2. Move exact selection to GPU

Only after topology/assembly is validated, prototype device-resident exact ranking for the
supported single-mixture case. The priority heap has dependent operations within each molecule;
a direct serial kernel is a baseline, not an assumed speedup. Preserve float64 ranking semantics,
stable ties, forced masks and the exact prefix. Parallel algorithms are acceptable only if those
outputs match the oracle, including near ties and extreme values.

Keep logits, masks, connectivity, unique mappings and memberships on device through the intensity
head. Reuse the shared differentiable probability calculation; do not detach training probabilities.
Benchmark full GPU construction against both the hybrid prototype and the improved CPU backend.
If exact GPU selection loses, record that result; do not silently replace it with random/beam
sampling or ship the hybrid under the full `cuda` name.

### B3. Integrate, bound memory and expose the option

Add the backend setting and migration described above through config, Lightning, model and
candidate builder. Inspect available CUDA/compiler tooling before choosing the kernel build path;
keep GPU compilation optional and verify the cluster staging/build flow. Bound device scratch
and handle capacity errors explicitly; no graph-size ceiling may silently alter candidates.
The implementation uses per-call tensors backed by PyTorch's caching allocator. A custom
persistent workspace is deferred because the measured full path already passes the local gate.
Review the Lightning static-graph transfer sidecar. Implementation adjustment: retain immutable
host graph metadata for CUDA too, because bounded native allocation and validation use atom/bond
counts and endpoints; dropping the sidecar would force a device-to-host graph copy each batch.
Dynamic logits and masks remain on GPU. Keep the existing `compile: False` restriction initially.

Audit the variable-length candidate interface: Python count tuples, dynamic compaction and shape
reads can reintroduce synchronization. Prefer device counts with explicit validity/lengths inside
the backend; where consumers require exact-sized tensors, measure and document the compaction
boundary. Full GPU computation does not by itself establish zero host synchronization or CUDA
graph compatibility. Keep public predictions compatible and update internal consumers as a unit.

Prepare matched opt-in configs for batches 128/256/512 only after the setting works. Preserve the
existing CPU configs. Resolve/report the backend in training, validation, inference and resume;
ship only after CPU-only execution, unsupported-mode errors and strict checkpoint loading pass.

## Correctness, performance and rollout checks

- Permanent tests cover all retained optimizations in the existing selector, component-array,
  candidate, SMARTS, model and Lightning suites; add CUDA differentials for the new backend.
  Keep reference oracles independent of the optimized implementation.
- Cover zero bonds, rings, disconnected graphs, redundant/parallel edges, multiword bitsets,
  forced-prefix collisions, equal/near-equal logits, full state-space exhaustion, target equality
  without overshoot, grouped reuse and invalid groups. CPU tests also cover mixtures/floors,
  non-unique mode and SMARTS even where the initial CUDA option rejects them.
- Require exact discrete candidate arrays/mappings and CPU thread-count parity. Compare fixed-input
  model outputs, losses and proposal/intensity gradients; define CUDA tolerances before measuring
  speed. Test legacy config/checkpoint migration and strict state-dict compatibility.
- Rebuild changed native kernels, run focused tests and Ruff for changed Python, then the full
  suite when practical. Use the configured interpreter and
  [session workflow](../workflow/session.md); record skipped GPU/data-dependent checks.
- For CUDA retention, require at least 5% full-step improvement per retained stage and at least
  10% aggregate improvement over the improved CPU backend on the canonical workload. Report the
  entire 128/256/512 matrix, including regressions/OOMs; higher utilization alone is not success.
- Use the [cluster guide](../guides/submit_cluster_runs.md) with `--time 3:00:00` for later runs.
  Run submission is separate from this planning task. Record run IDs, resolved configs, data and
  code identities, hardware, timing windows, memory and validation outcomes in the linked finding.

## Implementation checkpoint

Implementation was authorized after the planning handoff. The original CPU Python/native pair is
frozen at `/tmp/msh_cpu_reference_20260917`; the isolated A1 pair is at
`/tmp/msh_a1_reference_20260917`. These are local diagnostic artifacts, not portable releases.

- Stage 0: a new full-training benchmark includes production workers, loading, transfer,
  accumulation, clipping, optimizer/scheduler updates and repeated fixed-identity windows.
  Reference modules are loaded in isolated processes; no working-tree binary swapping is needed.
  It excludes logging, validation and checkpoint overhead. Short smoke windows passed;
  longer matched local measurements completed; target-GPU confirmation remains open.
- A1 score elision: **CLOSED NULL** locally. Exact parity passed, including extreme float64
  inputs, but full-training throughput improved only 3.06% (226.65 versus 219.93 spectra/s),
  below the 5% gate. The score/buffer specialization was removed. Numeric CSR GIL release
  remains part of A2's complete threaded-kernel experiment and passed the CPU full suite.
- A2: numeric topology/assembly releases the GIL; one process-local persistent pool handles both
  selection and topology. Workers return local indices; input-order remapping preserves groups and
  SMARTS. Oversized unique-array views are copied compactly and group results released after use.
  Correctness checks passed; local full-training gain is 9.26% versus original CPU, with the
  separate score specialization removed. The aggregate 10% target and L40S gate remain open.
- A3/A4: deferred by current projections; see the canonical
  [CPU measurements](../findings/masshatter_training_speed_2026_09_17.md#cpu-implementation-and-full-training-measurements-2026-09-17).
- CPU full suite: 3,322 passed, 22 skipped, 2 xfailed. Final CPU reference frozen at
  `/tmp/msh_cpu_final_reference_20260917`; three-hour target commands are in the
  [benchmark guide](../guides/benchmark_masshatter_cpu.md).
- B1: native CUDA topology and model integration pass exact candidate and gradient checks.
  The first matched full-training test is 5.72% faster than improved CPU; see the canonical
  [GPU measurements](../findings/masshatter_training_speed_2026_09_17.md#optional-gpu-candidate-construction-2026-09-17-in-progress).
  Exact pointer jumping/larger tiles raised the measured gain to 7.02% before GPU selection.
- B2: exact GPU ranking, direct device CSR and live scoring pass complete candidate/model
  differentials. Full GPU construction is 20.57% faster than improved CPU and 12.66% faster
  than the optimized hybrid locally, passing the stage and aggregate GPU gates.
- B3: explicit `cpu`/`cuda` setting, legacy config/checkpoint migration, resolved W&B logging,
  a narrow inference override and three opt-in CUDA configs are implemented. CPU is the default;
  supported CUDA modes and native bounds are in the benchmark guide. Public API/checkpoint tests
  pass. The final public CUDA matrix improves throughput by 18.97% at 128×2 and 23.44% at 256×1
  over matched CPU; 512 runs out of memory on the local 8 GB GPU. CUDA source packaging and
  per-user native build caches are prepared; actual L40S compilation/execution is pending.
- Final validation: CPU full suite **3,422 passed, 93 skipped, 2 xfailed**; separate CUDA suite
  **118 passed**. These counts overlap and must not be summed. Ruff passed on all 25 changed
  Python files; resolved CUDA configs differ from their CPU counterparts only by backend and
  W&B labels/tags. Documentation links and whitespace checks passed.
- Timing boundary: `candidate_topology_assembly_batch_ms` replaces the two old topology/assembly
  metrics and measures one coordinator dispatch/join interval. Output-transfer/packing time now
  includes serial index remapping. Compare full-window throughput across this boundary.

No scheduler jobs or W&B runs were created. The CUDA option is locally validated and opt-in;
production promotion remains gated on matched L40S results. CPU stays the default.
