# MASShatter Testing and Scale-Validation Plan

> **Status:** IN PROGRESS &nbsp;·&nbsp; **Updated:** 2026-08-14
>
> This document tracks verification only. Model architecture and probability
> semantics remain in `directional_bond_break_probability_plan.md`.

## 1. Scope

Validate that the MASShatter one-pass bond-fragment path is:

- mathematically correct for independent bond-mask ranking and marginalization;
- correctly grouped when several cut masks are batched together;
- sparse in component membership rather than quadratic in components and atoms;
- differentiable through all learned probability terms;
- compatible with eager and compiled CPU/CUDA execution;
- bounded by explicit candidate, memory, and runtime limits.

Coverage percentage is a diagnostic, not the success criterion. Reference
comparisons, invariants, gradients, and integration behavior are stronger gates.

## 2. Current verified baseline

As of 2026-08-14:

- full repository suite: 2,000 passed, 14 skipped, 2 expected failures;
- focused bond-fragment suite: 58 passed;
- statement coverage: mask generator 98%, neural heads 82%, combined 89%;
- branch coverage: mask generator 96%, neural heads 74%, combined 84%;
- 1,000 randomized exhaustive top-K comparisons passed;
- a CPU smoke test with 40,000 components and 320,000 sparse memberships
  completed the core sparse operations in approximately 0.06 seconds;
- a 100-bond, 500-mask generation smoke test completed in approximately
  0.075 seconds.

Timing results are local observations, not portable pass/fail thresholds.

## 3. Correctness gates

### 3.1 Cut-mask enumeration

- Compare exact top-K output with exhaustive `2**B` enumeration for small `B`.
- Include positive, negative, zero, tied, and extreme finite logits.
- Verify the empty mask can be excluded whether it is the MAP state or appears
  later in heap order.
- Verify all cut counts can occur without a D3/D4 depth restriction.
- Exercise normal target termination, iterator exhaustion, and the hard
  `max_enumerated_masks` safety cap independently.

### 3.2 Graph translation and fragment marginals

- Compare connected components with an independent graph reference on chains,
  rings, fused rings, disconnected test fixtures, and multi-cut masks.
- Verify PyG `(2, num_bonds)` endpoint layout through the complete head-to-
  translator interface.
- Verify ring openings that do not disconnect the graph are skipped by default.
- Compare duplicate-fragment `logsumexp` aggregation with direct probability
  sums.
- Verify deterministic ordering for equal fragment probabilities.

### 3.3 Charge retention

- Require one component-to-cut-mask group index per component.
- Compare segmented normalization with independent dense per-group softmax.
- Verify every cut-mask group sums to one when many groups are concatenated.
- Test both `logsumexp` and `logmeanexp` atom-to-component pooling.
- Test extreme atom logits such as `-1000` and `+1000` for finite outputs and
  gradients.
- Reject mismatched, wrong-dtype, negative, and out-of-range sparse indices.
- Reject components with no atom memberships.

### 3.4 Sparse component and boundary pooling

- Compare sparse `index_add` sum/mean pooling with a small dense reference.
- Verify gradients with repeated atom embeddings across many cut masks.
- Verify components with no boundary bonds receive a zero boundary embedding.
- Reject missing component memberships for component pooling.
- Define sparse membership pairs as unique; validate this at construction time
  rather than adding an expensive uniqueness check inside every model forward.

### 3.5 Ion-mode and hydrogen-transfer head

- Verify output shape and gradients for several ion-mode/H-shift dimensions.
- Verify normalization is per fragment across the joint channel cross product.
- Verify invalid channels have exactly zero probability.
- Exercise empty fragment batches and malformed embedding, context, cut-count,
  and validity-mask inputs.
- Compare formula/mass conversion with existing CRF/CMF utilities once Stage 3
  integration is implemented.

## 4. Compilation and device matrix

Run the learned heads and sparse probability operations under:

| Execution | CPU | CUDA |
|---|---:|---:|
| Eager float32 | required | required when available |
| `torch.compile`, full graph | required | required when available |
| Forward and backward | required | required when available |
| Mixed precision | optional initially | required before training rollout |

Compiled tests must detect graph breaks caused by tensor-to-Python conversions.
CUDA tests should be skipped explicitly when CUDA is unavailable, not silently
treated as passing CUDA coverage.

## 5. Scale and performance gates

Test representative batches near the planned upper range:

- 100 and 500 evaluated cut masks per molecule;
- approximately 40,000 components per batch;
- approximately 1,280 precursor atoms per batch;
- sparse atom/component and boundary-bond/component membership pairs;
- forward and backward peak memory measured separately.

Record:

- evaluated masks and unique fragments;
- membership-pair count;
- wall time for enumeration, graph translation, pooling, and learned heads;
- peak CPU resident memory and peak CUDA allocated memory;
- compile time separately from steady-state execution.

Do not add a hard timing threshold until the benchmark hardware and expected
batch composition are fixed. The structural gate is that memory scales with
membership pairs, not `num_components * num_batch_atoms`.

## 6. Coverage gaps to close

High-value remaining cases:

1. hard enumeration safety-cap termination;
2. extreme segmented charge logits and gradients;
3. malformed sparse charge-membership indices;
4. empty bond, component, fragment, and ion-channel batches;
5. joint ion/H head shape and context validation;
6. CUDA eager/compiled forward and backward when a CUDA runner is available.

Low-value uncovered statements are primarily constructor guards for impossible
configuration dimensions and redundant malformed-shape checks. Add those tests
only when they protect a public integration boundary or a reproduced failure.

## 7. Commands

Use the configured FRAGNNET interpreter:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m pytest -q \
  tests/test_bond_fragment_heads.py \
  tests/test_bond_mask_generator.py \
  tests/test_compute_frags.py

/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m pytest -q

/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m ruff check \
  src/fragnnet/frag/bond_mask_generator.py \
  src/fragnnet/model/bond_fragment_heads.py \
  tests/test_bond_mask_generator.py \
  tests/test_bond_fragment_heads.py

/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m compileall -q \
  src/fragnnet/frag/bond_mask_generator.py \
  src/fragnnet/model/bond_fragment_heads.py

git diff --check
```

For coverage, use direct `coverage run` with an `/tmp` data file. The equivalent
pytest-cov command currently triggers an early PyTorch import collision in this
environment.

## 8. Completion gate

This testing plan is complete when:

- every high-value gap in Section 6 has a deterministic regression test;
- focused tests, the full suite, Ruff, compile checks, and diff checks pass;
- CPU eager and full-graph compiled forward/backward pass;
- CUDA verification passes on an available CUDA runner;
- scale measurements demonstrate sparse rather than dense membership growth;
- Stage 3 formula/mass integration matches the existing chemistry utilities.
