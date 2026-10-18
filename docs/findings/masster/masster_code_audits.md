# MASSter code audits: simplification and current code path

> **Status:** FINDING · **Updated:** 2026-10-18
> Two code audits: the 2026-10-12 simplification audit (items 1-4 landed, 5-6 open) and the 2026-10-16 current code-path audit (tests repaired, lint open).

Merged 2026-10-18 from 2 findings on one topic. Each part below is the original
document unchanged except for heading levels; its own status line keeps its original date.
Former file names are listed so old links and `git log -- <former path>` can be traced.

| Part | Updated | Verdict | Former file |
| --- | --- | --- | --- |
| [MASSter code simplification audit](#masster-code-simplification-audit) | 2026-10-18 | Items 1-4, export fix and `path_selection/` move landed in `7e73b384`; 5-6 open | `masster_simplification_audit_2026_10_12.md` |
| [MASSter current code-path audit](#masster-current-code-path-audit) | 2026-10-16 | Config-inventory tests repaired, 177 PL tests pass; package lint open; speed audit | `masster_code_path_audit_2026_10_16.md` |

## MASSter code simplification audit

> **Status:** FINDING · **Updated:** 2026-10-18
> Items 1–4, the export fix and the `path_selection/` move committed in `7e73b384`; 5–6 open.

MASSter has substantial opportunities for behavior-preserving simplification. Start with
diagnostic plumbing and duplicated CUDA loading, then separate model stages. Removing experimental
features is a separate compatibility decision; a closed research arm is not proof of dead code.

### Outcome (2026-10-12)

Items 1–4 were implemented in separate worktrees, each checked by a behavior-equivalence and a
rules/simplicity reviewer, then applied together to the working tree on top of `6b359293`.
Full suite afterwards: the 7 item-1 failures pass and nothing new fails; the 21 remaining
failures (export schema, the postcut-gate debug script, runner config) fail identically at
`6b359293`. CUDA parity tests ran on the local GPU. Nothing is committed.

| Item | Result |
| --- | --- |
| 1. Test baseline | Fixtures enable `masster_merge_same_mz_rows`. The guard already had its own test (`tests/test_masster_merge_same_mz_rows.py`). Repairing the CLI case exposed an export bug: annotated export untransformed each pathway (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_export_merge_untransform_2026_10_12.md`), fixed with it. |
| 2. Diagnostic names | Three module-level tuples (30 mask, 6 timing, 9 ion-channel names) drive reads, casting and `loss_names`; one edit per new diagnostic instead of four. A before/after script over 21 configurations matched bitwise except wall-clock timings. |
| 3. CUDA loading | One `build_cuda_library` lifecycle; per-stage locks, `lru_cache`, ABI and flags stay with the callers. Cache filenames are byte-identical, so existing builds stay valid. About +22 net lines; the gain is one lifecycle, not fewer lines. |
| 4. Backend dispatch | `_cuda_full` had **no callers** (the audit said "duplicates"); deleted, and its error now names `candidate_backend='cuda'`. The topology prototype moved behind `build_masster_candidates_cuda_topology` through a private CPU core with one hook. Cost: the 14 keyword parameters are now listed in three places, about +60 net lines. |

The package was then consolidated: candidate generation (builder, Cython `core.pyx`, mask
generator, SMARTS templates, CUDA wrappers, `.cu` kernels and the nvcc build) moved into
`src/fragnnet/models/masster/path_selection/` with short module names, and
`_expand_topology_views` moved from the benchmark prototype to its production user `cuda_full`.
The extension was rebuilt for cp311 and cp312; the stale `masster_core` build is in
`output/path_selection_quarantine/`. The CPU benchmark's `--reference-dir` accepts both the new
`core`/`candidates` layout and the pre-move `masster_core`/`masster_candidates` layout.

### Scope and baseline

Reviewed the Python package structure, model and loss orchestration, candidate backend dispatch,
CUDA loading, attention bucketing, callers and focused tests on `refactor/fragnnet-masster-reorg`.
This is a maintainability audit, not a complete scientific or native-kernel correctness review.

The package has 16,219 Python source lines, including comments and docstrings. Three files account
for 9,958 (61%): `masster_model.py` (4,625), `masster_pl.py` (2,759), and
`bond_fragment_heads.py` (2,574). Generated C and compiled binaries are excluded.

Line numbers and paths below are as of `85459676`, before the cleanup and the `path_selection/`
move described under Outcome.

### Findings in recommended order

#### 1. Restore the test baseline before refactoring

Seven cases in [test_masster_pl.py](../../../tests/test_masster_pl.py) fail at model construction:
one proposal-target transform case, two prediction-only pickle cases, and four CLI export cases.
All request `ints_transform="sqrt"` without enabling `masster_merge_same_mz_rows`, so they hit
the explicit guard in [masster_pl.py](../../../src/fragnnet/models/masster/masster_pl.py)
(line 990) before testing their intended behavior. The
[reorganization baseline](../../history/repo_reorg_plan_log.md) already records seven failures in
this file. Update these fixtures to the supported contract and verify their assertions; keep
dedicated coverage of the guard itself. Do not weaken the guard to make the suite pass.

#### 2. Give diagnostic names one owner

In [masster_pl.py](../../../src/fragnnet/models/masster/masster_pl.py), the nested `loss_fn`
(1661–2205) has 42 explicitly named parameters, mostly prediction diagnostics. The same names
reappear in detached-value copying (1999 onward) and `loss_names` (2210 onward), with further
timing and channel lists. Every additional diagnostic requires synchronized edits.

Use small tuples for mandatory diagnostic groups and reuse them for required dictionary reads,
detachment/conversion, and registration. Preserve the current distinction between timing values
and values cast to the prediction dtype. Keep losses and computed metrics explicit; no generic
metric plugin system is needed. Preserve flat output keys, enabled/disabled metric sets, FastGNN
renames and per-spectrum reduction. The existing candidate-feature spec table demonstrates how
a single local declaration can already serve multiple consumers.

#### 3. Share the CUDA build and load mechanics

The `_load_native_library` implementations in
[selection](../../../src/fragnnet/models/masster/masster_cuda_selection.py) (60–126) and
[topology](../../../src/fragnnet/models/masster/masster_cuda_topology.py) (107–164) repeat compiler
validation, `/tmp` cache validation, compiler-version lookup, source hashing, temporary-file
compilation, atomic replacement, cleanup and `ctypes.CDLL` loading.

Extract that common lifecycle into the existing
[masster_cuda_utils.py](../../../src/fragnnet/models/masster/masster_cuda_utils.py).
Keep ABI declarations and stage-specific flags in their callers: selection deliberately adds
`--fmad=false` and `--ftz=false`. Preserve lazy compilation, cache identity, locking and errors.
This is a real reduction in duplicated logic with a smaller compatibility surface than model
restructuring. Verify both native backends when CUDA is available.

#### 4. Separate benchmark dispatch from production backend selection

[build_masster_candidates](../../../src/fragnnet/models/masster/masster_candidates.py)
(542–1358) accepts both `candidate_backend` and private `_cuda_full` / `_cuda_topology` switches,
plus two statistics channels. `_cuda_full` duplicates the production CUDA selection condition
at line 586; topology-only branches also interrupt the CPU path at lines 655, 733 and 907.

Call-site inspection places the private topology switch in benchmarks/tests. Move its adapter
behind an explicit benchmark entry point, leaving production dispatch as CPU or CUDA. Update
benchmark callers and parity tests together. Do not delete
[masster_cuda_candidates.py](../../../src/fragnnet/models/masster/masster_cuda_candidates.py)
wholesale: production CUDA imports its `_expand_topology_views` helper. The native topology
implementation is also part of production CUDA.

#### 5. Split orchestration at actual data boundaries

[MassterModel.forward](../../../src/fragnnet/models/masster/masster_model.py) is 1,504 lines
(3122–4625); its constructor has 123 arguments excluding `self`. The forward path combines
encoding, shared proposals, candidate generation, pathway marginalization, channel selection,
intensity scoring, diagnostics, isotope expansion and annotation assembly.

[MassterPL._setup_loss_fn](../../../src/fragnnet/models/masster/masster_pl.py) is 982 lines
(1357–2338), combining validation, a 545-line closure, auxiliary objectives and metric registration.
[bond_fragment_heads.py](../../../src/fragnnet/models/masster/bond_fragment_heads.py) combines
attention layers, prediction heads, graph membership operations and probability reductions.

Extract cohesive stages with explicit inputs and outputs: proposal preparation, fragment/channel
scoring, spectrum emission and detached diagnostics; separate objective calculation from metric
bookkeeping. Move pure graph/pooling functions and attention layers into focused modules only
where that clarifies ownership. Avoid dozens of tiny helpers with large argument lists or a
mutable dictionary passed through every stage. Moving lines alone will not slim total code.

Preserve module attribute names and state-dict keys when extracting methods/functions. Grouped
internal configuration may eventually help, but retain the YAML schema and explicit required
hparam reads. Test proposal gradient detachment, shared-CE behavior, unique and ordinary modes,
SMARTS, both intensity outputs, nominal/isotope emission and annotations across each extraction.

#### 6. Simplify attention bucketing only after profiling

In [bond_fragment_heads.py](../../../src/fragnnet/models/masster/bond_fragment_heads.py)
(1225 onward), token counts are copied to the host and grouped into buckets, then each bucket
rediscovers its membership with `isin`, `nonzero` and `searchsorted`. Counts plus contiguous
sorted spans already describe that layout. A simpler layout calculation may also remove device
synchronizations, but no speedup was measured here.

This is an existing proposed work item (removed 2026-10-13; `git show e65339ed:docs/plans/masster/masster_intensity_head_bucketing_sync_plan.md`).
Its atom-decoder discussion is historical: that decoder is now retired. Reprofile the current
attention path and retain ordering, padding and gradient parity before changing it.

### Boundaries and next step

Keep CPU/CUDA numerical implementations and independent reference oracles where they test each
other. Preserve chemistry validity, candidate ordering, forced-mask prefixes, gradient flow,
checkpoint keys and flat model outputs. Do not infer feature retirement from default-off flags
or null experiments; schema removal requires a separate decision and migration.

Items 1–4, the export fix and the `path_selection/` move were committed in `7e73b384`.
Orchestration extraction (item 5) is next on a green baseline;
bucketing (item 6) waits for a profile.

### Validation

Using `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`:

- `-m ruff check src/fragnnet/models/masster --output-format concise`: passed.
- `-m pytest tests/test_masster_model.py tests/test_masster_pl.py tests/test_bond_fragment_heads.py tests/test_masster_candidates.py tests/test_masster_backend_compatibility.py tests/test_reorg_boundary.py -q`:
  **490 passed, 7 failed, 7 skipped**, 54.85 seconds. Failures are described above.
- CUDA was unavailable to the test process; native GPU parity and performance were not validated.
  The full test suite and model evaluation were not run. No source, test or config files changed.

## MASSter current code-path audit

> **Status:** FINDING · **Updated:** 2026-10-16
> Config-inventory tests repaired; all 177 MASSter PL tests pass. Package lint findings remain open.

### Scope

Reviewed HEAD `81c50c43` and the live working tree: MASSter hooks and dataset filtering,
model proposal sharing, rewire expansion and proposal isolation, channel selection,
spectrum emission, Lightning loss/metric integration, checkpoint migration and annotation
export. This is a targeted audit, not exhaustive verification of the native kernels or all
experiment configurations.

The working tree already contained the rewire prediction-row sorting fix in
`src/fragnnet/models/masster/masster_model.py`. Its regression test in
`tests/test_masster_rewire_model.py` appeared during the audit and was also exercised.
Neither change was authored by this audit. The sort keeps exact-mass, unmerged predictions
grouped by spectrum, as the vectorized CE loss requires; annotation rows follow the same
permutation. The mixed-batch versus singleton loss regression passes.

### Findings

1. **P2 — retired config inventories still break the MASSter PL test suite.**
   `tests/test_masster_pl.py:174` expects eight independent-distance configs, but only
   one remains. Line 197 expects three proposal-warmup configs, but none remain.
   Both tests fail before exercising model behavior. Replace the retired-file inventories
   with supported configuration fixtures and keep assertions of the actual runtime behavior.
   The previous project-state statement that the CPU suite passes does not describe this
   checkout's selected suite.
2. **P3 — the MASSter package fails its lint check.** Ruff reports unsorted imports in
   `masster_model.py:32` and `masster_pl.py:15`, an unused `n_atoms` assignment in
   `rewire_expansion.py:425`, and an unnecessary identity dictionary comprehension at
   `rewire_expansion.py:521`. These are maintenance defects; no runtime failure was
   demonstrated from them.

No further runtime correctness finding was confirmed in the inspected paths. Existing hub
descriptions of many changes as uncommitted are stale: the initial Git status contained only
the model change above, followed later by its test change.

### Validation

Interpreter: `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`, from the repository root.

- `-m pytest tests/test_masster_model.py tests/test_masster_pl.py tests/test_masster_rewire_model.py tests/test_masster_rewire_pl_export.py tests/test_masster_merge_same_mz_rows.py tests/test_masster_nominal_isotopes.py -q`
  — **418 passed, 2 failed, 10 skipped**. Failures are finding 1.
- `-m pytest tests/test_masster_candidates.py tests/test_masster_heads.py tests/test_masster_dataset.py tests/test_masster_backend_compatibility.py tests/test_masster_inference_backend.py tests/test_inference_masster_annotation.py tests/test_inference_masster_peak_aggregation.py tests/test_masster_rewire_expansion.py tests/test_masster_rewire_parity.py tests/test_masster_smarts_pathway.py tests/test_masster_intensity_reliability.py tests/test_masster_rt_ccs.py -q`
  — **312 passed, 5 skipped**.
- `-m ruff check src/fragnnet/models/masster --output-format concise`
  — **4 violations**, finding 2.

CUDA was unavailable to these processes. Native GPU correctness, performance, a real-checkpoint
inference run and scientific validation-set evaluation were not verified. The full repository
suite was not run. No implementation, test or configuration edits were made by this audit.

Next: repair the obsolete config tests and lint baseline, then validate the native GPU path
and real-checkpoint inference when the device is available.


### Unit-test repair (2026-10-16)

Finding 1 is repaired in `tests/test_masster_pl.py`. The retained merged JSS config is
resolved through `runner.load_config`, including the template's atom-decoder default.
The retired warm-up config inventory is replaced with parameterized runtime checks at
epochs 0, 14, 15 and 16: the shared encoder stays trainable and the intensity head becomes
trainable after the inclusive epoch-15 boundary. Existing direct-loss and warm-up gradient
and loss tests remain in place.

Validation with the configured interpreter: `-m pytest tests/test_masster_pl.py -q`
passed **177 tests**; `-m ruff check tests/test_masster_pl.py --output-format concise`
and `git diff --check` passed. The full repository suite was not rerun. No production code
or experiment configs changed. Finding 2 and GPU/real-checkpoint validation remain open.


### Batch and padding coverage (2026-10-16)

Added `tests/test_masster_batch_padding.py` with 14 cases covering:

- Three attention/feed-forward layers under induced, full and no-attention modes, with
  pre- and post-normalization. Unpadded references must match zero padding and large finite
  positive/negative padding values; real-token input gradients must match and padding
  gradients must be exactly zero.
- Shuffled candidate rows with per-spectrum counts 1, 3, 4, 5, 8 and 9. This spans shared
  buckets and power-of-two boundaries. Batched outputs and input gradients must match
  singleton references, with exactly zero input gradient in other spectra.
- End-to-end singleton versus mixed/permuted spectrum predictions for supported legacy and
  unique-fragment attention combinations. Inputs include a bondless sodium atom, ethanol,
  a larger aromatic molecule, both precursor types and distinct collision energies.

The deterministic comparisons disable dropout and use evaluation mode; the head tests
still exercise backward propagation. They do not claim training-mode batch invariance for
BatchNorm or stochastic regularizers, or validation of every optional attention bias.

Validation: `-m pytest tests/test_masster_batch_padding.py tests/test_masster_heads.py tests/test_masster_attention_and_budget.py tests/test_masster_rewire_model.py -q`
passed **168 tests, 3 skipped**, using the configured interpreter. Ruff on the new file
and `git diff --check` passed. No runtime defect was exposed. GPU checks remain unverified;
no production code changed.

### Speed audit of the current path (2026-10-16)

Inspected HEAD `b8546dd2` plus the live working tree. The four pre-existing modified files
are `candidates/rewire_null.py`, `candidates/rewire_rules.py`, `masster_pl.py` and
`tests/test_masster_rewire_pl_export.py`, under their current MASSter/test directories.
These supersede the initial checkout description above. No production changes were made
in this speed audit. Findings below identify concrete work, not measured current speedups.

Resolved `i1`, `i4` and `a0_rewire_after_selection_ob_obf` through `runner.load_config`:
all use CUDA candidates, FP32 with TF32 enabled, microbatch 512, loss chunks capped at 64,
induced candidate attention, ranked mask budget 4096, diagnostics/base-CE logging off,
and proposal sharing off. The largest historical switches are therefore already active.
See [earlier measurements](masster_training_speed_2026_09_17.md); their timings are not a
profile of this revision.

#### Opportunities, in implementation order

1. **Rewire: remove avoidable device/host round-trips.**
   `rewire_expansion.py:424` downloads atom pointers and atomic numbers and compares
   downloaded bond topology with the host sidecar on every active forward. At line 592,
   per-spectrum ordinary-fragment counts go GPU → NumPy → GPU solely to compute
   `min(max_extra, floor(fraction * count + 1e-9))`. Keep that cap calculation on-device
   and reuse immutable host chemistry/validation from graph preparation. Preserve the
   float64 rounding and epsilon policy of `extra_limits`, including boundary cases.
   Additional per-chunk scalar checks and component-label convergence checks
   (`label_components`, line 132) synchronize CUDA. Use the existing asynchronous
   validation helper for suitable checks; convergence needs a separate kernel/design,
   not removal of its correctness condition. This affects rewire arms only.

2. **Rewire: avoid repeatedly sorting the entire growing product pool.**
   The local `absorb` at `rewire_expansion.py:485` first reduces each chunk and then
   concatenates it with the full accumulated pool and reduces again. With many chunks,
   early products are sorted repeatedly; continuation cuts add more absorbs. Benchmark
   a bounded hierarchical merge or a final reduction if memory permits. Keep exact atom-set
   deduplication, rule-indicator union, parent ranking and deterministic tie behavior.
   Small workloads may not benefit. The final extra cap is applied after validity and
   deduplication, so simply capping intermediate products is not equivalent.

3. **Base path: benchmark candidate packing and compilation of fixed-shape layers.**
   `_contextualize_by_molecule` (`masster_candidate_transformer.py:1159`) sorts rows,
   downloads counts, constructs host buckets and performs `isin`/`nonzero` compaction for
   every bucket, then launches each layer separately per bucket. A GPU-built layout or
   reuse of producer counts could reduce packing overhead. Compile the fixed-shape
   attention/feed-forward blocks separately before attempting whole-model compilation:
   native candidate calls, host lists and ragged outputs remain graph breaks. Attention
   already uses SDPA/`need_weights=False`, and validation already uses asynchronous CUDA
   assertions; neither should be proposed as a missing optimization. An earlier packing
   rewrite failed its end-to-end gate (linked speed finding), so repeat only with evidence
   from the current workload. Mixed precision needs compatibility work because the native
   CUDA candidate path currently requires FP32.

4. **Base path: reduce padding in the spectral CE loss.**
   `loss/cross_entropy.py:559` creates per-chunk `[spectra, max_true, max_pred]` matching
   tensors. `loss/padding.py:72` already adaptively splits oversized chunks, but it processes
   spectra in their original order. Internally grouping similar true/predicted peak counts
   could reduce padded work while preserving the training sampler, with the inverse
   permutation restoring per-spectrum outputs and tolerances. First measure padding waste
   and actual chunk count. Increasing `loss_batch_size` alone need not improve throughput.

5. **Conditional: proposal sharing needs a favorable batch, and is not a free switch.**
   `_shared_proposal_layout` (`masster_model.py:3414`) can share precursor encoding and
   construction across repeated groups, but itself builds/validates a GPU layout. Measure
   repeated-group multiplicity first. Current configs do not enable sharing; changing the
   sampler changes batch composition, and the PL shared-proposal loss merges targets by
   group (`masster_pl.py:670`). Treat enabling this as a matched training experiment,
   rather than a behavior-preserving speed repair.

The working-tree transfer optimization in `masster_pl.py:2650` already excludes molecule
binary bytes from Lightning's recursive device transfer. Preserve it; its comment's timing
was not independently benchmarked here. CPU candidate threading, cut-bond deduplication,
unique-fragment transformer tokens and persistent loader workers also already exist.

#### Validation and next measurement

Configured interpreter: PyTorch `2.4.1+cu121`; `torch.cuda.is_available()` returned false.
`tests/test_masster_rewire_pl_export.py` and `tests/test_masster_batch_padding.py` passed
**49 tests** on CPU. These are regression checks, not performance evidence. No training,
checkpoint evaluation, GPU benchmark or full-suite run was performed.

Next: use the [full-training benchmark](../../../scripts/debug/bench/benchmark_masster_full_training.py)
on a supported GPU with fixed batches and source identity, compare base versus rewire stage
costs, and profile synchronized forward/backward plus optimizer throughput. Prioritize
the inexpensive rewire cap/host-metadata changes, then the measured dominant base stage.
Apply the [benchmark guide's](../../guides/benchmark_masster_cpu.md) end-to-end gain gates;
do not rank work by historical candidate timers alone.
