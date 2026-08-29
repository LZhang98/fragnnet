# MASShatter — Non-Probabilistic Intensity Plan

> **Status:** PROPOSED &nbsp;·&nbsp; **Updated:** 2026-08-29
>
> Replace MASShatter's Gaussian probability-density objective with direct, tolerance-matched
> peak-list training, compare cosine and Jensen-Shannon objectives with optional entropy matching,
> then test whether independent nonnegative peak scores outperform the current
> candidate-normalized intensity probabilities. Fragment proposal remains a separate problem.

## Decision

Move away from probabilistic modeling for **peak intensity**.

MASShatter may continue to use probabilities for bond breaking, cut-mask ranking, charge
retention, and candidate proposal. Those quantities describe discrete structural choices. Final
peak intensity is different: the target is relative signal magnitude, and the model is selected by
binned spectral similarity rather than calibrated likelihood.

The abandoned binned-background mixture is not part of this plan. Do not add a background head,
an OOS intensity class, a learned background mass, or an m/z density. Those mechanisms improve the
accounting of a likelihood objective that is no longer the desired training target.

## Current limitation

For each selected fragment `f` and valid ion/H channel `c`, current MASShatter constructs

```text
P(f | selected candidates) * P(c | f)
```

using two normalizations:

1. a grouped softmax over selected fragments in each spectrum;
2. a masked softmax over valid ion/H channels within each fragment.

The resulting candidate probabilities sum to one. The primary sparse Gaussian CE then evaluates
only true peaks that have a candidate within tolerance. Unmatched true peaks contribute exactly
zero. Consequently, the objective is a candidate-conditional, matched-subset density score rather
than a full-spectrum intensity objective.

The main defect is the loss support: unmatched observed intensity has no primary training effect,
and a truncated Gaussian density is optimized even though model selection uses spectral
similarity.

The two softmaxes are not a representational-capacity constraint under a scale-invariant cosine or
JSD objective. Any nonnegative fragment-by-channel intensity matrix, up to one global scale, can be
factorized into a fragment marginal and channel-conditional distributions. Probability and
independent outputs therefore span the same relative intensity shapes on fixed valid support. They
have different optimization parameterizations: softmax couples every candidate through
normalization, whereas independent Softplus scores receive local output gradients and begin with a
different dependence on valid-channel and duplicate-candidate counts. The controlled comparison
therefore tests the complete output parameterization--normalization coupling, initialization, and
cardinality behavior--not whether independent mode can represent spectra that the softmax mode
fundamentally cannot.

The existing CE can still fit relative intensity on its covered support. The claims to test are
whether a full-support spectral objective improves held-out similarity and whether independent
output optimization improves convergence or stability under that objective.

## Scope

The first implementation applies to the lead MASShatter path:

```text
mass_shatter_unique_fragment_mode: True
mass_shatter_proposal_selection_only: True
```

Legacy pathway-token modes keep their existing probability semantics. Extending independent
intensity scoring to them is out of scope until the lead-path result is positive.

Candidate enumeration, chemical channel validity, proposal supervision, and the hard selected
fragment budget remain unchanged in every arm.

## Proposed intensity representation

The intensity transformer continues to produce one fragment score and one ion/H adjustment per
valid pathway. In independent mode, combine them as unconstrained scores:

```text
z[f, c] = fragment_score[f] + ion_h_score[f, c]
intensity[f, c] = softplus(z[f, c])
```

Invalid chemical channels remain masked out. There is no fragment softmax and no ion/H softmax.
Several channels from one fragment may all receive high intensity when the target supports them;
false channels are penalized through the full-spectrum similarity objective.

`softplus` is the initial activation because it is nonnegative, differentiable everywhere, and
does not create dead outputs. Do not add activation choices in the first experiment. A different
activation is a later ablation only if score saturation or scale drift is measured.

The independent output is an intensity, not a probability. Expose that distinction explicitly in
the model/Lightning contract rather than silently storing unnormalized values under a probability
name. Compatibility code must preserve the existing `pred_logprobs` path for the legacy arm.

Use this concrete boundary contract:

- probability mode returns `pred_logprobs`, `pred_mzs`, and `pred_batch_idxs` exactly as today;
- independent mode returns `pred_intensities`, `pred_mzs`, and `pred_batch_idxs`, where
  `pred_intensities` is the normal-space Softplus output;
- `SpectrumPL._common_step` branches on the required `mass_shatter_intensity_output` value. The
  probability branch keeps the existing `log_in=True`, `log_out=True` preprocessing bit-for-bit;
  legacy CE consumes those tensors directly, while a new aligned loss receives normal-space copies
  produced by `exp()` after preprocessing. The independent branch preprocesses
  `pred_intensities` with `log_in=False` and `log_out=False` and sends those normal intensities to
  the new aligned losses;
- the adapter may derive `safelog(pred_intensities)` under the semantically neutral local name
  `pred_logintensities` when an existing evaluation utility requires log-space input. It must not
  expose that tensor from the model as `pred_logprobs` or send it to Gaussian CE;
- evaluation always starts from normal-space intensity, applies the existing evaluation
  normalization, and may then construct `eval_pred_logprobs` solely for compatibility with current
  metric helpers.

The new aligned loss receives explicitly named normal-space `true_intensities` and
`pred_intensities`. For probability-output arm B, these are the exponentiated post-preprocessing
log tensors; for independent arms, predicted intensity remains in normal space throughout the
training path. Legacy CE continues to receive `true_logprobs` and `pred_logprobs`. Proposal
supervision retains its own fragment-probability tensors and does not consume either spectral-output
field.

## Direct peak-list objectives

Do not bin spectra for the training loss. Keep the original predicted and observed m/z values and
construct a shared sparse support in two stages:

1. Canonicalize numerical exact masses with `round_aggregate_peaks(decimals=4)`. Use `agg="lse"`
   for log intensities or `agg="sum"` for normal-space intensities. This 1e-4-Da rounding prevents
   floating-point summation order from splitting chemically identical masses; it is numerical
   exact-mass coalescing, not an instrument-resolution bin or a license to merge nearby formulas.
2. Assign each coalesced predicted peak to its nearest observed peak inside that sample's configured
   mass-error tolerance, using m/z distance only and deterministic tie breaking. Multiple predicted
   peaks may map to the same observed peak and their intensities are summed. Retain unmatched
   observed peaks as target-only coordinates and unmatched predicted peaks as prediction-only
   coordinates.

Do not add bounded-diameter or resolution clustering in the first implementation. A physical
peak-resolution merge requires a separately justified instrument rule; the existing loss tolerance
can include mass-calibration and preprocessing error and must not silently become a merge radius.

Proposal supervision remains different: every structurally distinct fragment with at least one
channel inside the matching tolerance remains a valid proposal target, with each unique fragment
counted once. Exact-mass coalescing happens only after fragments are expanded into observable ion/H
peaks; it never merges structural fragment identities.

This many-to-one alignment matches the current CE and proposal-loss convention that all compatible
candidates may contribute to one observed peak, avoids an arbitrary losing fragment among
near-degenerate candidates, and can be implemented with sorted/searchsorted tensor operations
instead of a cubic assignment solve. It is not identical to the 0.01-Da evaluation metric because
the boundaries differ, so both training-aligned and historical metrics must be logged.

The alignment is discrete and need not be differentiable with respect to m/z because MASShatter's
predicted m/z values are fixed chemical outputs. The aligned union makes unmatched true peaks and
false predicted peaks affect every primary loss without imposing a fixed m/z grid.

### Gradient contract through aggregation

Let `c(i)` be the fixed aligned-union coordinate for predicted candidate `i` after exact-mass
coalescing and nearest-observed assignment, and let `q_i = softplus(z_i)`. Construct aligned
predicted intensity only with differentiable summation:

```text
p[k] = sum_{i: c(i)=k} q_i
dL/dz_i = dL/dp[c(i)] * sigmoid(z_i)
```

Compute exact-mass groups and cross-spectrum alignment from m/z tensors only. Those indices may be
detached because m/z is not learned. Never detach, convert to NumPy, call
`.item()`, or rebuild a tensor from the candidate intensities before `scatter_add`/summation. Do not
use `scatter_max` in the first implementation because it sends intensity gradient only to the
winning member. Keep the loss arithmetic in float32 under mixed-precision training.

All predicted candidates assigned to one aligned coordinate therefore receive gradient through its
summed intensity whenever the coordinate-level loss has a nonzero derivative. An unmatched
predicted peak receives a suppression gradient from cosine or JSD when the spectrum also contains
aligned signal. If the predicted and observed supports are completely disjoint, both cosine
distance and normalized JSD
are at their constant maximum and provide zero intensity gradient: changing intensity alone cannot
create an m/z match. Proposal coverage is therefore responsible for retrieving matching fragments
before the intensity objective can assign their relative signal. Likewise, an unmatched observed
peak with no generated candidate cannot create a gradient to a nonexistent intensity score. Return
the disjoint maximum with a graph-connected zero such as `predicted_intensities.sum() * 0.0`, not a
detached constant.

`softplus` has derivative `sigmoid(z_i)`, so extremely negative logits can still have numerically
tiny gradients. Initialize the independent intensity logits near zero and avoid hard clamping, but
do not claim that this matches the initial function of the probability arm: near-zero Softplus
scores give roughly equal intensity to every valid channel, so initial fragment and exact-mass-group
intensity can grow with channel and candidate multiplicity. During the fixed-batch overfit, log the
valid-channel logit distribution, gradient norms, valid channels per fragment, exact-mass-group
cardinality, and the correlation of both counts with predicted intensity. Treat these initialization
and cardinality effects as part of the parameterization being tested.

### Cosine distance

Let the tolerance-aligned vectors for spectrum `s` be `p_s` and `y_s`:

```text
L_cos[s] = 1 - dot(p_s, y_s) / (||p_s||_2 * ||y_s||_2)
```

Requirements:

- use the full true spectrum, including peaks with no predicted candidate;
- include every predicted peak in the predicted norm, including false peaks;
- sum candidate intensities that align to the same observed peak;
- use raw, non-sqrt intensity in the first experiment;
- keep the aligned union sparse rather than constructing a dense m/z representation;
- add only numerical epsilon handling required to avoid division by zero.

Cosine is scale-invariant. That is intentional because the task evaluates relative intensity. Do
not add an L1-normalization penalty, total-ion-current target, background mass, or amplitude
regression in the first experiment. Add scale regularization only if an actual numerical failure is
observed.

### Jensen-Shannon divergence

L1-normalize the aligned intensities inside the loss only:

```text
P_s = p_s / sum(p_s)
Y_s = y_s / sum(y_s)
M_s = 0.5 * (P_s + Y_s)
L_js[s] = (0.5 * KL(P_s || M_s) + 0.5 * KL(Y_s || M_s)) / log(2)
```

`L_js` is bounded in `[0, 1]` and remains finite for disjoint spectra. Reuse the numerical
conventions of the existing sparse JSS calculation on the aligned peak union and optimize
`1 - JSS`. Do not use Gaussian matching or intensity-dependent correspondence.

JSD interprets L1-normalized intensity shape as a distribution inside the loss. This does not
require the model head to emit probabilities: independent Softplus scores remain the model output,
and no likelihood or density over continuous m/z is introduced.

### Entropy matching auxiliary

For the same L1-normalized, tolerance-aligned vectors, compute Shannon entropy in nats:

```text
H(P_s) = -sum_i P_s[i] * log(P_s[i])
H(Y_s) = -sum_i Y_s[i] * log(Y_s[i])
L_ent[s] = (H(P_s) - H(Y_s))^2
```

Use raw entropy rather than dividing each spectrum by its own `log(active_peak_count)`. Predicted and
observed spectra have different supports, so separate support normalization can obscure the
concentration mismatch being measured.

Entropy matching is scale-invariant and target-aware. It must be auxiliary to cosine or JSD, never
a standalone primary loss: entropy contains no m/z alignment information, and spectra with peaks in
completely different locations can have identical entropy. The combined per-spectrum loss is:

```text
L_total[s] = L_primary[s] + lambda_ent * L_ent[s]
```

Do not precommit to `lambda_ent = 0.05`. On a small frozen calibration set spanning the configured
instrument types and representative observed-peak and candidate counts, choose one coefficient that
puts the weighted entropy term and its gradient norm at approximately 10% of the primary term near
initialization. Use the median ratio across calibration batches rather than one batch. Freeze and
record the calibration sample IDs, seed, coefficient, and resulting per-batch ratios before full
runs; do not launch a broad weight sweep initially.

## Experimental arms

Use the same candidate-generation architecture, data split, preprocessing, batch construction, and
optimization settings in all arms.

| Arm | Intensity output | Primary loss | Entropy | Question answered |
|---|---|---|---:|---|
| A — legacy | fragment/channel softmax | sparse Gaussian CE | 0 | Historical baseline |
| B — softmax cosine | fragment/channel softmax | many-to-one cosine | 0 | Effect of replacing CE support/objective |
| D — independent cosine | independent Softplus scores | many-to-one cosine | 0 | Effect of the output parameterization under cosine |
| E — independent JSD | independent Softplus scores | many-to-one JSD | 0 | Cosine versus JSD with independent output |
| F/G — entropy survivor | winner of D/E | same as winner | calibrated | Does entropy matching improve the winning primary objective? |

Stage the runs. First compare B, D, and E with paired seeds. Reuse an existing A run only after
confirming identical data, model, preprocessing, optimization, and seed configuration; the reported
`ef7fl5c6` result (`0.7515`) is the candidate baseline and should not be rerun merely to duplicate
it. Then add entropy only to the D/E survivor. Add the deferred softmax-JSD arm C only if the first
stage leaves a specific JSD/output-geometry interaction unresolved.

B and D have the same relative-intensity hypothesis class under scale-invariant loss. Their
comparison tests the full output parameterization, including gradient coupling, initialization,
candidate-cardinality sensitivity, convergence, and numerical stability; it does not test
representational capacity alone. Use two paired seeds for the initial rejection screen and treat
them as preliminary evidence, not a promotion-quality uncertainty estimate.

Score each run on mean validation `cos_sim_0.01` over the last 20 epochs. Do not use the maximum
over epochs. Report candidate weighted recall alongside cosine so a change in hard support is not
mistaken for better intensity assignment.

Before launching any arm, record one canonical anchor YAML, its fully resolved configuration and
hash, dataset artifact/version, split identifiers, code revision, and paired seeds in the run table.
Do not infer the anchor from a configuration family name: relevant unmerged and merged MASShatter
configs currently use different epoch budgets. The reported `ef7fl5c6` run becomes arm A only after
its resolved metadata is shown to match that anchor; otherwise run a matching A.

Use the plateau mean to compare arms. Within a promoted run, retain the checkpoint selected by the
existing maximum checkpoint metric and report that checkpoint's epoch and all co-primary metrics.
Inherit the training budget from the recorded anchor rather than hard-coding 100 or 200 epochs. If
the last-20-epoch convergence curve still has a material slope at that cutoff, do not reject the
slower objective; extend the paired arms by the same additional budget or mark the comparison
inconclusive.

Decision rules:

- Choose cosine versus JSD from the paired D/E comparison, not from one run's best epoch.
- Adopt independent intensity scoring only if D improves on B beyond seed variation or has a clear
  convergence/stability advantage; do not describe the result as increased output capacity.
- Keep entropy matching only if the entropy survivor improves on its no-entropy parent. Do not
  retain it merely because predicted and true entropy become closer while cosine stays flat or
  worsens.
- If entropy does not improve the JSD parent, treat it as redundant for that objective.
- Any arm with numerical collapse, materially worse candidate coverage, or clearly worse
  validation stability fails even if one seed has a higher best epoch.

## Implementation plan

### 1. Add explicit compatibility modes

Declare the required hyperparameters in `config/template.yml`; do not use fallback defaults:

```yaml
mass_shatter_intensity_output: probability  # [probability, independent]
mass_shatter_intensity_entropy_match_weight: 0.0
```

Use the existing required `loss_type` field rather than introducing a second MassShatter-specific
loss selector. Extend its validated vocabulary with explicit many-to-one tolerance-aligned cosine
and JSD values; do not reuse names whose existing FraGNNet behavior means binned or Hungarian loss.
Initially reject independent output with cross entropy because CE assumes normalized mixture
weights. Do not add a peak-merge hyperparameter in the first implementation.

`MassShatterPL._setup_loss_fn` currently hard-raises for any `loss_type` other than `cross_entropy`
and, in the same guard, requires `sparse_cosine_similarity` and `sum_ints`
(`src/fragnnet/pl_model/mass_shatter_pl.py`). Add the new values to that guard's accepted set; do
not relax the guard as a whole, because the `sparse_cosine_similarity`/`sum_ints` invariants and the
`binned_loss = False` assignment below it must survive unchanged in every arm.

Targets carry `ints_transform: none` (`config/template.yml`). The raw, non-sqrt convention in the
loss depends on that; keep the transform identical across all arms, since a sqrt or log transform
changes what a cosine or JSD objective is measuring.

### 2. Preserve the legacy output path

When `mass_shatter_intensity_output: probability`, leave fragment normalization, masked ion/H
normalization, output tensors, and historical loss behavior unchanged. This is the bit-for-bit
regression arm.

### 3. Add independent pathway intensities

For the unique-fragment, proposal-selection-only path:

1. retain the transformer and its candidate context exactly;
2. use the current second intensity-head output as the fragment score and `ion_h_logits` as channel
   scores. In the unique-fragment path that tensor is `fragment_residuals`, not the legacy
   pathway-token branch's `mask_residuals`; it is already gated by `use_mask_residuals`, which
   `proposal_selection_only` requires to be True;
3. mask invalid channels;
4. apply `softplus` to valid pathway scores;
5. return explicitly named intensity tensors for the spectral-loss path.

Update both `SpectrumPL._common_step` and the MassShatter loss signature to implement the boundary
contract above. The new loss signature accepts `pred_intensities`; it must not require a positional
`pred_logprobs` argument in independent mode. Keep the existing signature and preprocessing path
unchanged for legacy CE. Reject independent mode when an enabled auxiliary still assumes spectral
log-probabilities--initially `mass_shatter_base_ce_weight`, `ce_pair_loss_weight`, or
`cross_ev_fragment_loss_weight`--unless that auxiliary is explicitly adapted and tested against a
normalized diagnostic copy with preserved semantics.

Proposal probabilities stay out of both the transformer input and the intensity score, unchanged
from the current lead path: `proposal_selection_only` raises when `use_base_probability_feature` is
True, and that flag is exactly what gates the proposal-logprob token feature
(`src/fragnnet/model/bond_fragment_heads.py`). The lead arm's
`corrected_fragment_logits = fragment_residuals` already excludes proposal log-probabilities from
the score. Do not change either fact in a controlled arm; exposing them is a later ablation.

Concretely, independent mode replaces only the `_segment_logsumexp` fragment normalization and the
masked ion/H normalization with `softplus`. Everything upstream of those two lines is untouched.

Keep a separately normalized fragment-log-probability copy for
`unique_fragment_effective_count`, `unique_fragment_top1_mass_fraction`, and other diagnostics that
assume a distribution. Do not compute those quantities from raw independent scores. Initially
reject independent mode when `cross_ev_fragment_loss_weight` is nonzero unless the cross-CE loss is
explicitly updated to consume the normalized diagnostic copy with preserved semantics.

### 4. Add shared exact-mass aggregation and alignment

Reuse `round_aggregate_peaks(decimals=4)` to coalesce numerical exact masses while preserving
intensity gradients. Add one focused tensorized helper that assigns each predicted exact-mass group
to its nearest observed peak within the existing global or per-instrument tolerance and sums all
assigned predicted intensities. Matching must depend only on m/z. Build cosine, JSD, and entropy on
that shared aligned union so they cannot silently use different correspondence rules. Do not use
the existing Hungarian losses: their assignment scores depend on intensity, and their fallback
executes cubic SciPy assignment work per spectrum on CPU. Return one primary loss and one entropy
value per spectrum so `SpectrumPL` retains ownership of weighting and batch reduction.

### 5. Add JSD and entropy matching

Reuse the numerical conventions of the existing `jss_helper` and `sparse_entropy_fn`, but keep the
training helper differentiable with respect to independent intensities. Validate nonnegative finite
intensities and a positive total intensity for both spectra before normalization. Apply the entropy
weight with the same per-spectrum weighting and final batch reduction as the primary intensity
loss.

### 6. Keep proposal supervision separate

Continue the existing proposal loss unchanged for all arms. A true peak whose required
fragment is absent from the selected candidate set still cannot provide a direct intensity gradient
to that omitted fragment. Improving hard candidate retrieval requires supervision over a larger
pre-selection candidate pool and is a separate plan.

## Verification

Add focused tests for these invariants:

1. Legacy probability mode reproduces current `pred_logprobs` and sparse CE exactly.
2. Independent mode emits finite, strictly nonnegative intensities for every valid channel and no
   values for invalid channels.
3. Independent intensities are not constrained to sum to one.
4. Values that round to the same four-decimal m/z coalesce, and their intensities sum before each
   objective.
5. Two distinct predicted exact-mass groups inside one observed peak's tolerance map many-to-one and
   contribute their summed intensity to that observed coordinate.
6. A predicted peak inside two observed tolerances maps to the nearest observed m/z with documented
   deterministic tie breaking.
7. Known sparse examples match analytical cosine, JSD, and entropy values.
8. Identical spectra have zero cosine distance, zero JSD, and zero entropy-matching loss.
9. Spectra with no aligned peaks have normalized JSD equal to one without NaN.
10. Adding unmatched true intensity worsens the primary spectral loss without producing NaN.
11. Adding false predicted intensity worsens the primary spectral loss.
12. Two spectra with identical entropy but disjoint peaks demonstrate why entropy matching is not a
    standalone loss.
13. Multiplying all predicted intensities by a positive constant leaves cosine, JSD, and entropy
    matching unchanged within tolerance.
14. Gradients from each enabled loss reach the fragment score, ion/H score, intensity transformer,
    precursor encoder, and experiment context.
15. A spectrum with zero predicted or true total intensity raises an explicit validation error or
    follows one documented epsilon policy; it must not silently emit NaN.
16. Backpropagation through a many-to-one aligned coordinate gives finite, nonzero gradients to
    every contributing candidate logit on a deliberately non-optimal test spectrum.
17. In a spectrum with at least one aligned peak, an unmatched predicted peak receives a finite
    gradient that pushes its intensity down.
18. Exact-mass groups and nearest-observed assignments are unchanged when only predicted
    intensities change.
19. Aggregated predicted intensities retain a `grad_fn`, and backward produces the expected member
    gradients on CPU and CUDA.
20. A fully disjoint spectrum returns the documented maximum loss and a graph-connected zero
    intensity gradient; proposal-loss gradients remain available for candidate retrieval.
21. Probability mode still exercises the original `pred_logprobs`/`log_in=True` path, while
    independent mode exercises `pred_intensities`/`log_in=False`; neither mode silently consumes the
    other mode's field.
22. For finite positive test intensities above `log_min`, evaluation metrics computed directly agree
    with the same metrics computed after an explicit `safelog`/`exp` round trip within numerical
    tolerance.

Run the focused MassShatter model and Lightning suites after implementation. Run a one-batch CPU
smoke test and a short fixed-batch overfit before launching the staged training runs.

## Metrics and interpretation

Primary:

- plateau-mean validation `cos_sim_0.01`;
- validation `jss_0.01`;
- validation many-to-one, instrument-tolerance cosine and JSS using the same alignment as training;
- train/validation primary-loss and entropy-matching curves;
- candidate weighted recall.

Useful diagnostics already available or cheap to derive:

- predicted base-peak accuracy;
- matched-peak intensity rank correlation;
- predicted spectrum L1/L2 norm before metric normalization;
- target-aligned predicted-minus-observed entropy, explicitly labeled as the entropy used by the
  training loss;
- prediction-only entropy after exact-mass coalescing and before target alignment, so the logged
  value remains a property of the prediction rather than of the target-dependent correspondence;
- `cos_hun_inst_tol` and `jss_hun_inst_tol` as one-to-one secondary diagnostics;
- predicted precision and the number/fraction of exact-mass groups carrying material intensity;
- exact-mass group cardinality and its correlation with predicted intensity.

Do not use Gaussian CE slope, background calibration, background-bin entropy, or OOS mass as
success criteria for the new intensity path.
Treat materially worse predicted precision or material-peak count as a failure even if cosine
improves; cosine penalizes false intensity only through an L2 norm and can tolerate diffuse false
signal more readily than CE's normalized log penalty.

## Non-goals

- Calibrated probability or likelihood over m/z.
- Binned spectra or a fixed m/z grid in the training loss. Historical binned metrics remain for
  evaluation comparability only.
- Modeling contaminant, isotope, or otherwise unexplained intensity with a background density.
- Making hard top-K fragment selection differentiable.
- Changing candidate enumeration or the 256-fragment budget.
- Adding ranking, MSE, Wasserstein, or other hybrid losses before the cosine/JSD/entropy comparison
  is known.
- Extending the independent output to legacy pathway-token MASShatter or FraGNNet in the first
  experiment.

## Related documents

- [masshatter_plan.md](masshatter_plan.md) — current MASShatter architecture and historical
  probability semantics.
- [masshatter_candidate_pathway_refactor.md](masshatter_candidate_pathway_refactor.md) — separate
  candidate-generation simplification.
- [global_fragment_intensity_ranking_experiment_plan.md](global_fragment_intensity_ranking_experiment_plan.md)
  — broader FraGNNet intensity-ranking work; it does not define this MASShatter output change.
