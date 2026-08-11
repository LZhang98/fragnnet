# Minimal FT contender-pair training plan

> **Status:** PROPOSED — CACHE-FREE CONTEXT-BATCHED IMPLEMENTATION
> &nbsp;·&nbsp; **Updated:** 2026-08-11
>
> Add one online auxiliary loss to the existing flat GINE-1 model. Do not change the model
> architecture or write a persistent ranking-target cache.

## Goal

Test the smallest loss that directly addresses the measured FT failure:

> At collision-energy winner switches, the model changes the spectrum by about the right amount
> but moves intensity toward the wrong masses.

The experiment asks whether explicit supervision of the two observed winner masses improves
held-out spectra beyond a matched flat GINE-1 baseline.

## V1 decision

Train:

```text
selected flat GINE-1 CE baseline
+ existing whole-spectrum cross-entropy
+ online high-confidence FT soft pairwise ranking
```

V1 is deliberately narrow:

- FT `[M+H]+` and `[M-H]-` only;
- observed winner-switch transitions only;
- observed soft intensity ratios (`gamma = 1`);
- no persistent target builder, table, or fingerprint;
- CE-context batching with a matched weight-zero sampler control;
- no global base-peak margin, stable-anchor, teacher, contrastive, or trajectory loss;
- no new model capacity.

The ordinary spectrum loss still trains every peak. Stable transitions remain a safety metric, not
an auxiliary target: their lowest-change quartile has essentially no measured headroom
(`+0.002 [-0.097, 0.098]`).

## Baseline 0 — choose the CE representation

First compare the two already-defined, otherwise matched baselines:

| Config | CE representation |
| --- | --- |
| [`..._gine1.yml`](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_gine1.yml) | independent rounded CE tokens |
| [`..._gine1_ce_lin.yml`](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_gine1_ce_lin.yml) | affine projection of normalized continuous CE |

The `lin` arm makes neighboring CE values begin nearby instead of as unrelated lookup tokens.
Model width, insertion locations, output scaling, sampler, and optimization stay matched. Run one
seed first and confirm a promising result over three seeds. If `lin` does not improve held-out
behavior without harming QTOF or IT, keep `embed_clamped`; do not combine an unproven CE change with
the pair loss.

## `CEContextDynamicBatchSampler`

Online target construction requires the relevant CE levels in one autograd graph. Add a bounded
context sampler over:

```text
(molecule, precursor type, instrument type, fragmentation mode)
```

The sampler must preserve the baseline's deliberate group balancing. Select contexts with the same
inverse-spectrum-count weighting used by `simple_group_sampler`, targeting approximately five
spectra per group per epoch. Do not enumerate every spectrum from every long ladder: that would
reverse the current balancing and over-weight spectrum-rich contexts.

For each selected context:

1. sort distinct CE levels by the filled NCE value supplied to the model;
2. select a consecutive CE window containing at least two levels;
3. rotate the window start deterministically by epoch so every adjacent transition is eventually
   covered;
4. include one spectrum from every retained level, then allocate remaining replicate slots
   round-robin with deterministic epoch rotation;
5. enforce `ce_context_max_spectra_per_batch` as a hard per-context limit.

Use the existing `dynamic_batch_sampler_max` fragment-edge budget as the sole memory authority; do
not add `ce_context_max_frag_edges`. If a window is over budget, remove outer CE levels while
retaining complete adjacent transitions. If even a two-level transition is over budget, exclude and
count it. Never emit an over-budget batch, silently truncate one endpoint of a supervised
transition, or use “emit alone” as an OOM fallback.

The batch carries grouping metadata, not persistent supervision:

```text
ce_context_ptr
ce_level_ptr
ce_level_value
ce_level_spec_indices
```

The collator validates CE order, context boundaries, the hard spectrum cap, the edge budget, and at
least one selected spectrum per retained level.

### DDP and training-budget contract

Construct the epoch's atomic-window list deterministically. Drop the rotating tail needed to make
its length divisible by world size, then shard equal window counts across ranks. Every rank executes
the same number of optimizer steps; rotate and report dropped-window coverage across epochs.

Match the sampler control and pair arms on optimizer steps, not epochs or nominal spectra seen.
Also report spectra, contexts, and eligible transitions seen so the changed training estimand stays
visible.

The new estimand is the inverse-group-weighted distribution of bounded consecutive CE windows,
with deterministic coverage rotation. It is not the natural spectrum distribution. Retain a
sequential validation loader for primary metrics; the grouped validation loader is diagnostic only.

## Online target construction

Inside the loss under `torch.no_grad()`:

1. merge observed peaks within each selected same-CE replicate group;
2. average normalized observed intensities across selected same-CE replicates;
3. form adjacent transitions within each retained ordered window;
4. retain transitions whose consensus base mass changes;
5. set `A` to the lower-CE winner and `B` to the higher-CE winner;
6. require A and B to have raw-DAG support at both endpoints;
7. compute the endpoint targets and confidence/reliability weights.

The initial population is FT `[M+H]+` and `[M-H]-`. `[M+Na]+` remains excluded because its paired
directional interval crosses zero; `[M+Cl]-` remains excluded for insufficient coverage.

Use the frozen detection-limit floor when one contender is unobserved:

```text
q_e = normalize(max(I_e(A), L_A_e), max(I_e(B), L_B_e))
c_pair_e = abs(I_e(A) - I_e(B)) / max(I_e(A), I_e(B), L_A_e, L_B_e)
```

If observed mass groups are non-disjoint under the matching contract, skip and count the
transition. Fail only when the audited ambiguity rate exceeds `0.1%`; one pathological context must
not crash a late training step.

### Singleton reliability

At datamodule setup, use all training replicates to construct a small in-memory calibration table:

1. bin replicated endpoints by `(instrument, adduct, c_pair bin)` with fixed edges
   `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`;
2. estimate individual-replicate agreement with replicate-mean A/B ordering;
3. use the molecule-cluster 95% lower confidence bound as bin reliability;
4. pool adjacent bins until at least `100` replicate pairs are available;
5. mark bins still below coverage ineligible.

A replicated endpoint uses its observed consensus agreement. A singleton uses the frozen lower
bound for its bin. For singletons this is a per-bin constant, not endpoint-specific information; its
reliability gate acts as an additional confidence-bin cutoff.

Before transferring replicated reliability to singletons, compare agreement across replicated
strata with different replication rates within instrument/adduct/confidence bin. If adequately
covered strata differ by more than `0.05`, or the adverse molecule-cluster interval excludes zero,
exclude singleton targets from that bin. Record and broadcast the frozen table and checksum; do not
refit it on validation.

Confidence and reliability floors are nullable until this audit freezes them. `0.2` and `0.7` are
audit candidates, not schema defaults or validation-tuned values.

## Pair loss

Use the same FT mass-matching boundary as `inst_type_loss_tol`: a candidate at mass `m` matches when
its absolute error is strictly below `1.0e-5 * max(m, 200.0)` Da. Apply it to observed merging,
support, prediction aggregation, and evaluation.

Aggregate predicted **peak-intensity log probabilities** after isotope/peak expansion, not formula
probabilities:

```text
log_P_e(A) = logsumexp(log_peak_intensity_i for candidates matching A)
log_P_e(B) = logsumexp(log_peak_intensity_i for candidates matching B)
d_e = log_P_e(A) - log_P_e(B)
y_e = q_e(A)
ell_pair,e = BCEWithLogits(d_e, y_e)
```

This is the two-mass KL gradient in stable binary form. Train with BCE; report entropy-free
`KL(q_obs || p)` and TV against the observed `gamma = 1` ratios. Do not add BCE and KL together.

For transition `t` in context window `c`, with `R_t,e` selected replicates at endpoint `e`:

```text
a_t = min(c_pair_low, c_pair_high) * r_low * r_high
omega_t,e,r = a_t / (2 * R_t,e * sum(a_u for eligible transitions u in c))
```

For a minibatch of `B` spectra, including ineligible spectra:

```text
L_pair = sum(omega * BCEWithLogits(d, y) for eligible entries) / B
L = L_spectrum_ce + lambda_pair * L_pair
```

Keep the fixed `/B` denominator. Do not average over only eligible endpoints, transitions,
contexts, or auxiliary weight: that makes effective `lambda_pair` depend on variable batch
composition. Return differentiable zero for no eligible transition. Log eligible entries, summed
weight, selected/available context spectra, dropped levels/transitions, and the
auxiliary-to-primary fragment-logit gradient ratio.

## Minimal implementation

1. Choose the matched CE baseline.
2. Implement the bounded context sampler and weight-zero sampler control. Verify inverse-group
   selection, CE-window rotation, caps, edge fallback, and equal-step DDP behavior.
3. Implement online target construction and the training-only reliability audit. Emit reports, not
   target artifacts.
4. Apply the coverage gate: at least `5,000` retained training switches, at least `500` validation
   molecules, no molecule above `1%` of pair weight, and ambiguity below `0.1%`.
5. Implement peak-intensity alias aggregation and soft `BCEWithLogits`; verify empty batches and
   gradients to fragment-intensity logits.
6. Calibrate auxiliary weights from a fixed-subset gradient measurement and run a tiny overfit.
7. Run the pilot and confirmation matrix before broader alternatives.

Required configuration is explicit and cache-free:

```yaml
ce_context_batching: false
ce_context_max_spectra_per_batch: 16
ce_pair_loss_weight: 0.0
ce_pair_min_endpoint_confidence: null
ce_pair_min_endpoint_reliability: null
ce_pair_subset: switches
ce_pair_eligible_inst_types: [FT]
ce_pair_eligible_prec_types: ["[M+H]+", "[M-H]-"]
```

Do not add a second edge-budget key. Existing configs need a compatibility path for new required
keys; training code must not supply fallback hyperparameters. A positive pair-loss weight must fail
setup when either threshold is null.

## Run matrix

### Stage A — one-seed pilot

First reproduce the original-sampler control. Then use identical seeds, optimizer-step budget, and
model for the context-sampler arms. Measure the unscaled auxiliary-to-primary fragment-logit
gradient ratio `g` on a fixed subset and derive:

```text
lambda_weak = 0.05 / g
lambda_medium = 0.20 / g
```

| Arm | Context sampler | Pair weight | Purpose |
| --- | :---: | ---: | --- |
| original control | no | `0` | historical matched GINE-1 |
| sampler control | yes | `0` | isolate batching and estimand changes |
| pair-weak | yes | `lambda_weak` | approximately 5% initial gradient ratio |
| pair-medium | yes | `lambda_medium` | approximately 20% initial gradient ratio |

The pair arms must beat the sampler control; comparison with the original control reports the net
deployable effect. Log the realized gradient ratio throughout training.

### Stage B — confirmation

Confirm the sampler control and selected pair arm over three matched seeds. Retain matched original
controls if Stage A shows a nontrivial sampler effect. Only after the real arm passes promotion,
run an online within-context shuffled-A/B control.

## Evaluation and decision

Primary outcome:

- `cos_hun_inst_tol` on the sequential natural FT validation distribution.

Mechanism and safety checks:

- both-endpoint base-mass accuracy, switch recall, and CE delta cosine;
- endpoint two-mass KL/TV against observed `gamma = 1` ratios;
- whether a remaining wrong argmax lies inside or outside `{A,B}`;
- results by CE gap, `||delta q||_2`, adduct, singleton/replicated status, and window coverage;
- stable-transition retention and false-switch rate;
- high-CE supported base accuracy, `cos_sim_0.01`, formula retrieval, and CE-bin performance;
- sampler-control versus original-control differences.

Never pool stable and switch metrics or substitute grouped diagnostics for natural validation.

Promote only when the three-seed pair arm beats the sampler control by at least
`+0.005 cos_hun_inst_tol`, has paired-spectrum and molecule-cluster intervals excluding zero,
improves switch recall or both-endpoint correctness, causes no material safety regression, improves
over the original control after sampler effects, and beats the shuffled-target control if run.

Before Stage A, compute the eligible natural-validation fraction `f`; the `+0.005` overall bar
requires at least `0.005 / f` endpoint gain if all improvement comes from that subset. Stop if this
exceeds available headroom, the sampler itself cannot be made safe, the tiny subset cannot overfit,
or real and shuffled targets perform alike.

## Escalation after a diagnosed result

Only after v1:

1. add reliable stable anchors if switches improve but false switches rise;
2. add online top-k contender KL if the wrong argmax remains outside `{A,B}`;
3. run the trajectory redundancy gate if ranking improves but CE direction stays weak;
4. reuse the bounded context sampler for trajectory KL only if that gate passes;
5. add model capacity only when the useful loss fits training windows but fails to generalize.

The context gate is not the next default: existing `context_insert_location: mol,mlp` already gives
the fragment scorer nonlinear fragment-by-context conditioning.
