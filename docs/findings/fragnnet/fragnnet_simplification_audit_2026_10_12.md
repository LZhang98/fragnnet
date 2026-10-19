# FraGNNet simplification audit

> **Status:** FINDING · **Updated:** 2026-10-18
> Audit complete. The cc-mask padding leak (finding 1) is fixed with a regression test (`e0197d8f`); the staged cleanup (findings 2–6) is not started.

## Scope and conclusion

The requested `src/fragnnet/models/fragnnetr` does not exist; this audit covers
`src/fragnnet/models/fragnnet/`. Its 18 Python/Cython source files contain 10,641 lines,
including comments and docstrings. Detailed review concentrated on the model, Lightning
wrapper, dataset, samplers, and Python DAG assembly. The Cython search algorithms were
inventoried, not exhaustively audited.

The package can be substantially easier to maintain without redesigning the architecture.
Start with dead branches and duplicated operations. Splitting long functions helps navigation,
but does not itself remove code. Retiring experiment options is a separate compatibility decision.

Line references in findings 2–6 describe the source before the scatter-import migration
(`7a87d64e`) and the finding 1 fix. In `fragnnet_model.py`, lines before about 1413 now sit six
lines earlier and later lines about two lines earlier; search by symbol.

## Findings, in recommended order

### 1. cc-mask padding bits leaked across molecules (fixed in the working tree)

**Priority: high correctness issue. It affected `cc_interstage_use_rest=True` and the
`cc_corruption_frac` diagnostic; the default path was unaffected.**

The cc mask is `MASK_SIZE` = 128 bits wide; bits at or beyond the molecule's atom count are
padding. Two paths in
[the model](../../../src/fragnnet/models/fragnnet/fragnnet_model.py) set padding bits, added
each molecule's global atom offset, and then filtered only against the batch's total atom
count:

- **Rest pooling** pooled `~frag_node_mask`, the fragment's complement, padding included.
- **The corruption diagnostic** flipped every bit with probability `cc_corruption_frac`,
  padding included, before the cc path's equally global bounds check.

Padding positions of molecule *i* therefore indexed atoms of molecules *i+1, …*. With a
four-atom first molecule and a six-atom second, local positions 4–9 of the first molecule
are padding but pass the global `< 10` test. Padding of the last molecule falls beyond the
total and is dropped, so the leak depends on batch position. The true cc mask and
`boundary_cc`/`boundary_pair` set only real atom bits and were not affected.

The audit's CPU probe compared `CCCO` alone with `CCCO` at batch position 0 next to
`CCCCCC` in one eval-mode model. Maximum absolute peak-probability difference on randomly initialised test models: about
2e-8 to 4e-8 with rest pooling disabled, 2e-3 to 1e-2 with it enabled, depending on the
seed.

**Fix.** Right after the mask is decoded, the model builds `frag_atom_bits`: true for bit
positions below the owning molecule's atom count. The complement and the corruption flips
are both intersected with it. Rest pooling now passes `dim_size`, because every
full-molecule fragment has an empty complement and a scatter without it drops trailing
empty rows. The rest path's batch-wide bounds filter is removed as unreachable.
[`tests/test_fragnnet_mask_padding.py`](../../../tests/test_fragnnet_mask_padding.py) checks
that a molecule's prediction is the same alone and in a batch, for rest pooling (mean and
max pool, both batch orders) and for `cc_corruption_frac=1.0`. All five cases failed before
the fix and pass after it. Two further cases put a one-atom molecule last in the batch, so its
only fragment is the root with an empty complement in the last row; they fail if `dim_size` is
removed.

**Consequences.**

- Checkpoints trained with `cc_interstage_use_rest=True` (the `*_rest.yml` pooling
  ablations, two debug configs) learned on the leaking features and will predict
  differently now. No finding cites those ablations.
- The [cc-corruption probe](cc_corruption_probe_finding.md) ran on the leaking code. A
  NIST20 test-split spectrum has a median of 19 heavy atoms (p10 11, p90 34), so about 83% of
  the 128 bits are padding and most flipped bits were padding:
  they added atoms from other molecules in the batch (batch size 4), or nothing for the last
  molecule. The probe's `frac=1.0` row is a deterministic inversion, not a random mask. The
  dose-response measures cross-molecule contamination as much as loss of membership
  information; the finding carries a caveat.

### 2. Remove unreachable loss diagnostics and unreferenced memory helpers

**Priority: first behavior-preserving cleanup.**

[FraGNNetPL](../../../src/fragnnet/models/fragnnet/fragnnet_pl.py) builds `loss_names`
in `_setup_loss_names` (lines 407–568), but never registers the names guarding these loss-factory branches:

- Spectrum entropy: lines 1072–1080 (`spec_e`, `spec_ne`).
- Mutual information: lines 1154–1161 and 1296–1304.
- Hydrogen diagnostics: lines 1169–1180 (`h_mean`, `h_e`, `h_ne`).

These branches are unreachable through ordinary setup. No subsequent production mutation
of the list was found in the model/base wrapper. Remove the branches and their now-unused
local plumbing; do not remove model output fields that inference still consumes.

[The dataset](../../../src/fragnnet/models/fragnnet/fragnnet_dataset.py), lines 867–916,
also holds `get_batch_memory` and `find_largest_batch`. Search across `src`, `tests`, and
`scripts` found only their definitions and the call from one helper to the other.
They are deletion candidates, subject to checking any external notebook use. The streaming
`IterableSpecMolFragDataset` immediately above them is actively used by inference and stays.

### 3. Consolidate the two output heads' shared conditioning

**Priority: high maintainability benefit.**

`FraGNNetModel.__init__` spans 735 lines; `forward` spans 1,530 lines
(1235–2764). In the `formula` and `node_formula` branches, candidate-ion projection,
factorization, and mode/CE/adduct/stratum scaling are repeated (roughly 2056–2107 and
2216–2257). A private method for that common sequence removes duplication while retaining
the existing registered modules and parameter names.

Keep head-specific normalization, CMF handling, competition, and depth-bias placement explicit.
In particular, depth bias belongs on node logits for `node_formula`; shifting all conditional
formula logits equally cancels under softmax. The
earlier fixes (removed 2026-10-13; `git show e65339ed:docs/plans/fragnnet/fragnnet_audit_fixes.md`) and current regression tests
are reasons not to merge the heads into one generic conditional pipeline.

After deduplication, extract a few coherent stages: fragment feature assembly, formula-index
preparation, and terminal explainability. Keep the top-level forward readable and avoid a
large mutable context object or a new hierarchy of single-use classes. Moving existing
modules beneath new `nn.Module` owners would rename state-dict keys; plain private methods
avoid that compatibility cost.

### 4. Reduce repeated loss bookkeeping with two small helpers

**Priority: high maintainability benefit.**

`_setup_loss_fn` spans 861 lines (570–1430). The repeated entropy blocks (1082–1304)
perform the same weighting and logging around two mathematical operations: marginal and
conditional entropy. Keep their tensor selection explicit, and share the weighting/logging
operation. A compact explicit term table can also drive registration and the explanation-skip
guard, which currently repeat the same term families.

The cosine/JSS/cross-entropy pairwise MSE branches (1313–1403) repeat target computation under
`no_grad`, prediction computation, MSE, weighting, and per-sample logging. One local helper
parameterized by the existing similarity function is sufficient. Preserve the historical
`pairwise_cossine_*` metric spelling unless a separate metric migration is intended.

`training_step` (1471–1557) likewise repeats common-step execution, scalar checks, debug handling,
and result updates. Share only the common pieces. Dynamic manual optimization deliberately
uses accumulated sample weights; fixed accumulation and automatic optimization have different
step boundaries. Do not replace them with one superficially similar averaging rule.

### 5. Merge duplicate DAG root-child insertion mechanics

**Priority: clear, bounded code reduction.**

[DAG builder](../../../src/fragnnet/models/fragnnet/dag/builder.py), lines 309–500:
`_inject_smarts_pairs_into_dag` and `_inject_nl_masks_into_dag` duplicate mask lookup,
minimum-size filtering, node/edge deduplication, depth construction, and metadata append.
The former iterates both masks in each pair; the latter iterates product masks directly.
Apart from that loop the bodies are identical, so the SMARTS wrapper can call the
neutral-loss function on the flattened pairs; no new helper is needed.

Keep the chemistry-facing wrappers and pass an ordered mask iterable into one internal
insertion helper. Preserve insertion order, root-edge deduplication, node depth 2, edge depth 1,
and mutation behavior. Test byte-for-byte array/metadata equivalence, including empty inputs
and existing fragments. Do not merge these conventions with the Cython multi-cut seed path:
its depth semantics differ, as documented in the
[prepass reference](../../reference/frag_gen_prepass.md).

### 6. Small follow-ups; avoid a generic sampler or conditioning framework

[Sampler code](../../../src/fragnnet/models/fragnnet/fragnnet_data_sampler.py) repeats
cached/fallback node-edge metadata loading at lines 134, 326, and 402. A small shared reader
could help, but preserve the dual-group path's reuse of an already-loaded sample when group
metadata is absent. Keep the three distinct batching policies intact, especially the prepared
CE-context experiment (removed 2026-10-18; `git show 95341fce:docs/plans/fragnnet/ft_contender_pair_minimal_training_plan.md`).

`mol_context_input_only` at model lines 1389–1397 runs the full molecular GNN and then
discards its result in favor of `input_project`. An early branch can skip wasted work for
this diagnostic. Check eval behavior and any training-side RNG or normalization effects
before calling that change equivalent in every mode.

Experimental options should be inventoried separately for retirement. The
[hub](../../workstreams/fragnnet.md#settled--read-before-re-proposing) records the CE-depth
prior as having no headroom and competition as never successfully trained, yet options and
tests remain. Neither an unsuccessful experiment nor a default-off flag proves code is dead.
Removing those options requires explicit decisions about old configs and checkpoint loading.
No feature retirement is part of this audit.

## Validation and next change

Audit, run with the configured `FRAGNNET-GPU` interpreter on the pre-migration source:

- `python -m ruff check src/fragnnet/models/fragnnet --output-format concise`: passed.
- `python -m pytest tests/test_fragnnet_audit_fixes.py tests/test_fragnnet_audit_regressions.py tests/test_fragnnet_formula_support.py tests/test_fragnnet_pretrained_weights.py tests/test_skip_train_explainability.py tests/test_cc_atom_dropout.py tests/test_reorg_boundary.py -q`: **84 passed**, 7 warnings.
  None of these tests caught the padding leak.

Finding 1 fix, run on CPU on the current working tree:

- `tests/test_fragnnet_mask_padding.py`: the 5 batch-independence cases failed before the
  fix; all 7 cases pass after it. Mutation runs: removing either mask intersection, using `<=`
  in `frag_atom_bits`, or dropping `dim_size` each fails at least one case.
- The audit test list above plus every `tests/*fragnnet*`, `*boundary*`, `*pool*` and
  `*interstage*` file: **141 passed**.
- `ruff check` on the changed files: passed.

No training, scientific evaluation, full-suite run, Cython rebuild, performance benchmark,
or real-checkpoint validation was performed. There are no claimed speedups or measured
post-refactor line savings.

Next: remove dead branches (finding 2), then deduplicate DAG insertion and head conditioning
in independently reviewable changes. For each refactor, compare outputs, gradients, and
state-dict keys, including both heads, CMF/null handling, explanation flags, and optional
conditioning. Keep source and config behavior unchanged.
