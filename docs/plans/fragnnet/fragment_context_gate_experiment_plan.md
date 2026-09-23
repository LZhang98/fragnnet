# Per-fragment context routing and final-logit gate experiment plan

> **Status:** PROPOSED — ZERO-CODE ROUTING ABLATION FIRST; GATE NOT IMPLEMENTED
> &nbsp;·&nbsp; **Updated:** 2026-08-11
>
> First test whether the existing per-fragment context path has held-out value. Implement a
> final-logit multiplicative gate only if that zero-code test is positive.

## Corrected premise

The existing `context_embedding` path already supports fragment-specific nonlinear context
responses when `context_insert_location: mol,mlp`:

```text
context c_s is repeated for every fragment f
frag_embed_f = [h_f; c_s]
z_f = formula_module([h_f; c_s])
```

`formula_module` is a two-hidden-layer nonlinear MLP. It can therefore learn a different context
response for fragments A and B in the same spectrum. The previously proposed residual

```text
delta_z_f = W_o((W_h h_f) * (2 * sigmoid(W_c c_s) - 1))
```

is a restricted explicit parameterization of an interaction the existing MLP can already learn
over the bounded context domain. It is not a qualitatively new fragment × context capability and
must not be justified as extra model capacity.

The residual has two potentially testable differences only:

1. it routes context directly to final candidate logits, after the shared output scalers;
2. it can start as an exact zero residual, which may change optimization and attribution.

The plan is therefore an ablation of **routing and trainability**, not a capacity experiment.

## Phase 0 — decisive zero-code ablation

Test whether the existing per-fragment context route helps at all:

```text
mol-only: context_insert_location: mol
mol+mlp:  context_insert_location: mol,mlp
```

Use the existing NodeMLP context configuration as `mol+mlp` and its matched mol-only counterpart:

| Arm | Config | Per-fragment context path |
| --- | --- | ---: |
| mol-only | [`..._context_embedding_mol_only.yml`](../../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_context_embedding_mol_only.yml) | no |
| mol+mlp | [`..._context_embedding.yml`](../../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_context_embedding.yml) | yes |

The files differ in the model/training configuration only at `context_insert_location`. Keep seed,
context dimension, CE representation, output scalers, sampler, optimizer, and training schedule
matched. This NodeMLP screen is appropriate because the directional defect was measured on
NodeMLP and requires no model code.

Run one matched seed first. If the direction is promising, confirm with three matched seeds. Report:

- natural-distribution `cos_hun_inst_tol`;
- FT switch-only delta cosine;
- both-endpoint base-mass correctness and switch recall;
- supported base-mass accuracy by CE bin;
- stable-transition false-switch rate;
- formula retrieval and train/validation gap.

### Phase 0 decision

- **No held-out switch improvement:** stop. Do not implement the gate. A second, more restricted
  copy of a null per-fragment context interaction has no evidential basis.
- **Held-out switch improvement without safety regression:** reproduce the same `mol` versus
  `mol,mlp` comparison on the selected matched GINE-1 CE baseline before writing the gate.
- **Training improves but validation does not:** treat this as a generalization failure, not a
  request for another context module.

The formal go requires a three-seed directional improvement whose molecule-cluster interval
excludes zero and no material regression in natural-distribution cosine, stable false switches, or
formula retrieval.

## Phase 1 — GINE-1 routing confirmation

Activate only after Phase 0 passes. From the CE representation selected in
[`ft_contender_pair_minimal_training_plan.md`](ft_contender_pair_minimal_training_plan.md), create
two GINE-1 context configs that differ only at:

```text
context_insert_location: mol
context_insert_location: mol,mlp
```

Use the same Phase 0 metrics and decision rule. If the GINE-1 `mol,mlp` arm is null, stop. The gate
is not rescued by the NodeMLP result because the intended training baseline is GINE-1.

## Phase 2 — optional final-logit reparameterization

Activate only if both routing tests show held-out value. The question is now:

> Is a direct, zero-initialized multiplicative route to final logits easier to optimize or better
> placed than additive context concatenation at the formula MLP input?

Compare one per-fragment context route per arm:

```text
concat control:
    context_insert_location: mol,mlp
    fragment_context_gate: false

gate replacement:
    context_insert_location: mol
    fragment_context_gate: true
```

Do not use `mol,mlp + gate`; that would add a redundant second route and a null result would be
uninformative.

### Candidate gate

Let `h_f` be the final fragment representation and `c_s` the existing combined context vector:

```text
r_f       = W_h h_f
g_s       = 2 * sigmoid(W_c c_s)
delta_z_f = W_o(r_f * (g_s - 1))
z_f       = z_base_f + delta_z_f
```

Apply `delta_z_f` after the active shared output scalers and immediately before flattening and
spectrum-level `scatter_logsoftmax`. This ordering is the structural hypothesis; it must be named
in the result rather than described as new capacity.

Initialize `W_c.weight = 0` and `W_c.bias = 0`, giving `g_s = 1` and `delta_z_f = 0`. The gate arm
therefore begins as the same mol-only model before learning its replacement per-fragment route.
Construct optional modules after existing baseline modules so enabling the gate does not perturb
their random initialization sequence.

Choose the rank once by parameter matching, not validation:

```text
rank = nearest positive integer making
gate_parameter_count ~= parameters added by context concatenation to formula_module
```

Record both counts and require them to differ by no more than 5%. Do not sweep rank in v1.

## Configuration contract

Only Phase 2 adds required template keys:

```yaml
fragment_context_gate: false
fragment_context_gate_rank: null  # required and parameter-matched when the gate is enabled
```

Requirements:

- access both keys explicitly; no fallback defaults in model code;
- `fragment_context_gate: true` requires `context_embedding: true`;
- the gate replacement requires `context_insert_location: mol`, not `mol,mlp`;
- rank must be a positive integer and recorded in the checkpoint config when the gate is enabled;
- v1 supports the active `mlp_output_format: formula` path only and fails on another format;
- when disabled, construct no optional parameters so old checkpoint state dictionaries remain
  unchanged.

## Batching

No sampler change is required in any phase. The existing ragged batch already carries one context
vector per spectrum and fragment batch indices. Neither the context route nor the soft pairwise
endpoint loss requires low- and high-CE spectra in the same minibatch.

## Phase 2 verification

Tests must show:

- the gate residual is exactly zero at initialization;
- the disabled path leaves parameters, outputs, and strict checkpoint loading unchanged;
- activated gate weights produce different residuals for different fragments under one context;
- changing context changes residuals for a fixed fragment;
- fragment batch indices cannot leak context between spectra;
- gradients reach the gate, context embedding, and fragment representation;
- CPU and CUDA forward/backward preserve shapes and finite values;
- added memory and runtime are linear in fragment count;
- gate and concat arms satisfy the declared parameter-count match.

Log optimization-specific diagnostics because optimization is the hypothesis:

- training and validation loss curves at the same optimizer steps;
- steps required to reach fixed spectrum-loss and switch-error thresholds;
- gradient norm entering the context embedding;
- RMS `delta_z` relative to base-logit RMS;
- residual alignment with observed A-to-B switch direction;
- fraction of candidate and spectrum argmaxes changed by the residual.

## Run sequence

1. NodeMLP mol-only versus mol+mlp, one seed.
2. If promising, confirm NodeMLP routing over three seeds.
3. If passed, GINE-1 mol-only versus mol+mlp, one seed then three if promising.
4. If passed, implement parameter-matched concat versus gate replacement without pairwise loss.
5. Only after the soft pairwise loss independently works, repeat concat versus gate replacement
   with the already-selected loss weight and confidence mask.

Never tune CE representation, gate rank, pairwise weight, or confidence mask jointly.

## Promotion and stop rules

Promote the final-logit gate only when, relative to the parameter-matched concat control, it:

- improves three-seed natural-distribution cosine by at least `+0.005` or provides a statistically
  supported switch-direction gain without a cosine regression;
- improves convergence or held-out switch direction, consistent with the routing/optimization
  hypothesis;
- does not increase stable false switches or regress formula retrieval;
- improves multiple major FT adduct/CE strata;
- uses a bounded residual rather than changing only entropy or global sharpness.

Stop immediately when either zero-code routing ablation is null. After implementation, stop when
the gate merely duplicates concat, improves training only, concentrates gains in a thin stratum, or
changes only sharpness. Do not interpret a null gate as evidence that fragment-specific context is
useless—the preceding `mol` versus `mol,mlp` result answers that question.

## Relationship to other mechanisms

- [`conditioning_refactor_plan.md`](../conditioning_refactor_plan.md) defines how context metadata are
  consolidated; its `mol,mlp` route is already fragment-specific.
- `GINE_GATED` gates molecular bond messages upstream and answers a different bond-lability
  question.
- Shared CE/mode/stratum scalers cannot re-rank same-channel fragments.
- The set-level competition head compares candidates with the full spectrum set. Use it only if
  per-fragment conditioning works but global candidate comparison remains the diagnosed defect.
