# MASSter code simplification audit

> **Status:** FINDING · **Updated:** 2026-10-12
> Items 1–4 implemented in the working tree 2026-10-12 (uncommitted); 5–6 open. One export bug found.

MASSter has substantial opportunities for behavior-preserving simplification. Start with
diagnostic plumbing and duplicated CUDA loading, then separate model stages. Removing experimental
features is a separate compatibility decision; a closed research arm is not proof of dead code.

## Outcome (2026-10-12)

Items 1–4 were implemented in separate worktrees, each checked by a behavior-equivalence and a
rules/simplicity reviewer, then applied together to the working tree on top of `6b359293`.
Full suite afterwards: the 7 item-1 failures pass and nothing new fails; the 21 remaining
failures (export schema, the postcut-gate debug script, runner config) fail identically at
`6b359293`. CUDA parity tests ran on the local GPU. Nothing is committed.

| Item | Result |
| --- | --- |
| 1. Test baseline | Fixtures enable `masster_merge_same_mz_rows`. The guard already had its own test (`tests/test_masster_merge_same_mz_rows.py`). Repairing the CLI case exposed an export bug: [annotated export untransformed each pathway](masster_export_merge_untransform_2026_10_12.md), fixed with it. |
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

## Scope and baseline

Reviewed the Python package structure, model and loss orchestration, candidate backend dispatch,
CUDA loading, attention bucketing, callers and focused tests on `refactor/fragnnet-masster-reorg`.
This is a maintainability audit, not a complete scientific or native-kernel correctness review.

The package has 16,219 Python source lines, including comments and docstrings. Three files account
for 9,958 (61%): `masster_model.py` (4,625), `masster_pl.py` (2,759), and
`bond_fragment_heads.py` (2,574). Generated C and compiled binaries are excluded.

Line numbers and paths below are as of `85459676`, before the cleanup and the `path_selection/`
move described under Outcome.

## Findings in recommended order

### 1. Restore the test baseline before refactoring

Seven cases in [test_masster_pl.py](../../../tests/test_masster_pl.py) fail at model construction:
one proposal-target transform case, two prediction-only pickle cases, and four CLI export cases.
All request `ints_transform="sqrt"` without enabling `masster_merge_same_mz_rows`, so they hit
the explicit guard in [masster_pl.py](../../../src/fragnnet/models/masster/masster_pl.py)
(line 990) before testing their intended behavior. The
[reorganization baseline](../../history/repo_reorg_plan_log.md) already records seven failures in
this file. Update these fixtures to the supported contract and verify their assertions; keep
dedicated coverage of the guard itself. Do not weaken the guard to make the suite pass.

### 2. Give diagnostic names one owner

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

### 3. Share the CUDA build and load mechanics

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

### 4. Separate benchmark dispatch from production backend selection

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

### 5. Split orchestration at actual data boundaries

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

### 6. Simplify attention bucketing only after profiling

In [bond_fragment_heads.py](../../../src/fragnnet/models/masster/bond_fragment_heads.py)
(1225 onward), token counts are copied to the host and grouped into buckets, then each bucket
rediscovers its membership with `isin`, `nonzero` and `searchsorted`. Counts plus contiguous
sorted spans already describe that layout. A simpler layout calculation may also remove device
synchronizations, but no speedup was measured here.

This is an existing [proposed work item](../../plans/masster/masster_intensity_head_bucketing_sync_plan.md).
Its atom-decoder discussion is historical: that decoder is now retired. Reprofile the current
attention path and retain ordering, padding and gradient parity before changing it.

## Boundaries and next step

Keep CPU/CUDA numerical implementations and independent reference oracles where they test each
other. Preserve chemistry validity, candidate ordering, forced-mask prefixes, gradient flow,
checkpoint keys and flat model outputs. Do not infer feature retirement from default-off flags
or null experiments; schema removal requires a separate decision and migration.

Next: review and commit the items 1–4 working-tree changes, the export fix and the
`path_selection/` move. Orchestration extraction (item 5) is next on a green baseline;
bucketing (item 6) waits for a profile.

## Validation

Using `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`:

- `-m ruff check src/fragnnet/models/masster --output-format concise`: passed.
- `-m pytest tests/test_masster_model.py tests/test_masster_pl.py tests/test_bond_fragment_heads.py tests/test_masster_candidates.py tests/test_masster_backend_compatibility.py tests/test_reorg_boundary.py -q`:
  **490 passed, 7 failed, 7 skipped**, 54.85 seconds. Failures are described above.
- CUDA was unavailable to the test process; native GPU parity and performance were not validated.
  The full test suite and model evaluation were not run. No source, test or config files changed.
