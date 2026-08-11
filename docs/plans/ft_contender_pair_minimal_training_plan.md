# Minimal FT contender-pair training plan

> **Status:** PROPOSED — MEASUREMENT GATES BEFORE FIRST IMPLEMENTATION
> &nbsp;·&nbsp; **Updated:** 2026-08-11
>
> Add one small auxiliary loss to the existing flat GINE-1 model. Do not change the model
> architecture in this experiment.

## Goal

Test the smallest training change that directly addresses the measured FT failure:

> At collision-energy winner switches, the model changes the spectrum but moves intensity toward
> the wrong masses.

The first experiment should answer one question:

> Does explicit supervision of the two observed winner masses improve held-out spectra beyond the
> matched flat GINE-1 baseline?

## V1 decision

Train:

```text
selected flat GINE-1 CE baseline
+ existing whole-spectrum cross-entropy
+ high-confidence FT soft pairwise ranking
```

Keep the existing fragment scorer, CE inputs, DAG, sampler, and spectrum head unchanged. The only
new training hyperparameter that needs selection is the auxiliary loss weight.

This is intentionally narrower than the full
[`global_fragment_intensity_ranking_experiment_plan.md`](global_fragment_intensity_ranking_experiment_plan.md):

- FT only; exclude QTOF because its replicate coverage is inconclusive;
- switch transitions only in the auxiliary loss;
- observed soft intensity ratios only (`gamma = 1`);
- no global base-peak margin loss;
- no stable-anchor auxiliary loss in v1;
- no new competition head, teacher, contrastive loss, or CE insertion path;
- no target-sharpness, confidence-threshold, or architecture sweep.

The ordinary spectrum loss still trains every spectrum, including stable CE transitions. Stable
transitions are therefore a safety evaluation for v1, not an additional target. This scope cut is
also measured: on stable pairs in the lowest change-vector quartile, the replicate-versus-model
gap was only `0.002 [-0.097, 0.098]`, so the strongest evidence for recoverable headroom is on
switches. If the auxiliary loss improves switches but increases false switches on stable
transitions, add high-confidence stable pairs in v2.

## Baseline 0 — test continuous CE first

Before adding the pairwise loss, run one matched CE-featureization comparison:

| Config                                                                                                       | CE representation                             | Pairwise loss |
| ------------------------------------------------------------------------------------------------------------ | --------------------------------------------- | ------------: |
| [`..._gine1.yml`](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_gine1.yml)               | independent rounded CE tokens                 |             0 |
| [`..._gine1_ce_lin.yml`](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_gine1_ce_lin.yml) | affine projection of normalized continuous CE |             0 |

The second configuration changes only `ce_insert_type: embed_clamped` to
`ce_insert_type: lin`; model width, CE insertion location, dual NCE/ACE inputs, CE output scaler,
sampler, and optimization remain matched. With `lin`, the input difference between neighboring CE
values is proportional to their numeric difference, so nearby energies cannot begin as unrelated
lookup tokens. Downstream nonlinear layers can still learn a nonlinear response.

Run one matched seed first. Compare the natural-distribution spectrum cosine, switch-only delta
cosine, both-endpoint winner accuracy, stable false-switch rate, and CE-bin performance, with every
metric also reported separately for FT, QTOF, and IT. Confirm a promising result over three seeds
before selecting it as the baseline for pairwise training; do not accept an FT gain that materially
harms QTOF or IT. If the continuous version does not improve held-out behavior, retain
`embed_clamped`; do not combine an unproven representation change with the pair loss. A rare
both-missing CE is represented differently: `embed_clamped` has a dedicated token whereas `lin`
passes raw `-1` to its affine map, near the low-CE range. The loader normally fills NCE from ACE
and ACE from NCE, so audit the both-missing count rather than changing this ablation.

## Target construction

Use adjacent CE levels within a fixed:

```text
(molecule, precursor type, FT instrument type, fragmentation mode)
```

Average same-CE replicates before constructing a transition. For a transition whose consensus
base mass changes:

- `A` is the lower-CE winner;
- `B` is the higher-CE winner;
- both masses must exist in the raw DAG support at both endpoints;
- the initial population is FT `[M+H]+` and `[M-H]-`; `[M+Na]+` remains excluded because its
  paired directional interval crosses zero, and `[M+Cl]-` remains excluded for insufficient
  coverage;
- replicated endpoints use their observed A-versus-B consensus agreement, while singleton
  endpoints use the frozen training-only reliability proxy defined below;
- `min(c_pair_low, c_pair_high)` must pass one threshold frozen from the training-only replicate
  audit; `0.2` is the initial candidate, not a validation-tuned value.

Store the observed two-mass ratios at each endpoint:

```text
q_low  = normalize(I_low(A),  I_low(B))
q_high = normalize(I_high(A), I_high(B))
```

Use the frozen detection-limit floor for an unobserved contender. The cache always stores the
observed ratios; it does not store hard winner labels.

Define endpoint reliability without assigning singleton spectra artificial agreement `1.0`:

1. On the training split only, collect replicated endpoints by `(instrument, adduct,
   c_pair-confidence bin)` using initial fixed edges `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`, and measure
   the fraction of individual replicates whose A-versus-B ordering agrees with the replicate-mean
   consensus.
2. Use the molecule-cluster 95% lower confidence bound of that agreement as the bin reliability.
   Pool adjacent confidence bins within the same instrument/adduct until there are at least `100`
   replicate pairs; if the threshold still cannot be met, that bin is ineligible.
3. For an endpoint with at least two replicates, set `r_endpoint` to its observed consensus
   agreement. For a singleton, set `r_endpoint` to the frozen lower-bound reliability of its
   training bin. This is a conservative weighting proxy, not an asserted probability that the
   singleton winner is correct.
4. Require `min(r_low, r_high)` to pass one training-frozen reliability threshold; `0.7` is the
   initial audit candidate. Store the complete calibration table and pooling decisions in the cache
   metadata and apply them unchanged to validation.

Report singleton and replicated coverage and outcomes separately. The double-replicated subset
remains the cohort on which the measurement-noise comparison is identified, but it is not the only
population allowed to supply a soft `gamma = 1` training target. Target sharpening remains
forbidden until the relevant population passes its replicate gate.

## Loss

Use one named `mass_matching_contract`: for FT, a candidate at mass `m` matches when its absolute
mass error is strictly less than `1.0e-5 * max(m, 200.0)` Da, exactly the FT entry in
`inst_type_loss_tol`. Use this identical boundary rule for observed-peak merging, raw-DAG support,
cache construction, training-time alias resolution, and evaluation. Aggregate aliases in log
space so every matching DAG candidate contributes:

```text
log_P_e(A) = logsumexp(log_probability_i for i matching A)
log_P_e(B) = logsumexp(log_probability_i for i matching B)
```

Define one predicted A-versus-B logit and one observed soft target:

```text
d_e = log_P_e(A) - log_P_e(B)
y_e = q_e(A) = I_e(A) / (I_e(A) + I_e(B))
```

For endpoint `e`, optimize the numerically stable soft pairwise ranking loss:

```text
ell_pair,e = BCEWithLogits(d_e, y_e)
```

This is the implementation form of the two-mass KL. Since

```text
sigmoid(d_e) = P_e(A) / (P_e(A) + P_e(B)),
```

`BCEWithLogits(d_e, y_e) = KL(q_e || p_e) + H(q_e)`. The target entropy `H(q_e)` is constant with
respect to model parameters, so the BCE and explicit two-entry KL have identical gradients. Use
`BCEWithLogits` for training; report the entropy-free two-mass KL and TV for evaluation so arms are
directly comparable. Do not add both BCE and KL—they would duplicate the same objective.

For transition `t` in molecular context `c`, define the raw confidence weight:

```text
a_t = min(c_pair_low, c_pair_high)
      * r_endpoint_low
      * r_endpoint_high
```

Let `T_c` be the eligible transitions in context `c`, let `R_t,e` be the number of replicate
spectra at endpoint `e`, and attach the same replicate-averaged target to each such spectrum with
fixed cache weight:

```text
omega_t,e,r = a_t / (2 * R_t,e * sum(a_u for u in T_c))
```

Thus every context has total cached endpoint weight one in the complete cache, each transition is
weighted by confidence and reliability, and neither its two endpoints nor replicate multiplicity
duplicate its influence. The unchanged v1 `simple_group_sampler` still defines which cache entries
are seen: the training estimand is therefore the existing sampler-weighted distribution with
context normalization, not an equal-context or natural-spectrum average. Do not claim that cache
weights undo sampler inclusion probabilities. Natural-distribution validation remains sequential
and unmodified.

For a batch with
`B` total spectra, including ineligible spectra, use the single denominator:

```text
L_pair_batch = sum(omega_t,e,r * BCEWithLogits(d_t,e,r, y_t,e)
                   for eligible cache entries in the batch) / B
L = L_spectrum_ce + lambda_pair * L_pair_batch
```

Do not take another mean over eligible endpoints or renormalize by their batch weight; either would
make effective `lambda_pair` depend on the highly nonuniform eligible count produced by the group
sampler. Log eligible count, summed auxiliary weight, and auxiliary-to-primary gradient norm per
batch so the selected loss weight has an auditable scale. Any future sampler change requires a
sampler-only control. Both endpoints may be trained as ordinary per-spectrum cache entries; they do
not need to appear in the same minibatch. Keep `transition_id` so evaluation can reconstruct the
pair.

This should be easy to optimize because it uses probabilities already produced by the model, keeps
the observed target soft, adds no parameters, and introduces only one selected loss weight.

## Generalization beyond two winner masses

The pair arm is a focused diagnostic, not a claim that only two peaks matter. The ordinary
whole-spectrum cross-entropy continues to train every observed peak. If the pair arm validates the
focused-competition signal but its gain is confined to base-peak switches, expand the target before
adding model capacity.

### Top-k contender-set KL: generic extension without batching changes

For each eligible CE context, let `C` be the union of the observed top-k mass groups across its
adjacent endpoints; `k = 5` is the first fixed candidate. At each endpoint, aggregate every model
alias matching a mass in `C`, then condition both distributions on that set:

```text
q_e_C = normalize(q_e[C])
p_e_C = normalize(p_e[C])
L_contender,e = KL(q_e_C || p_e_C)
```

This trains several globally competitive peaks without constructing all `O(k^2)` hard pairs. It
also preserves soft observed intensity ratios. Each endpoint can remain an independent cache entry,
so the existing dynamic sampler is unchanged. Do not enable this objective in the first pair-BCE
arm: it answers a broader question and would make the initial result unattributable.

Activate the contender-set extension when the pair arm improves switch identity but held-out top-k
ordering or spectrum cosine remains weak. Keep the ordinary spectrum loss and report whether the
focused loss merely duplicates it by comparing gradient norms and mass-wise error changes inside
and outside `C`.

### Mass-trajectory conditional loss: generic cross-CE extension

A direct ladder objective groups an ordered sequence within one fixed context:

```text
(molecule, precursor type, FT instrument type, fragmentation mode)
```

It does not group unrelated molecules that happen to share one collision energy. Average same-CE
replicates first and sort the remaining consensus spectra by the exact model CE value. Let `C` be
the union of the top-k observed mass groups across a ladder window. Write the conditioned ladder as
matrices `Q[e,m]` and `P[e,m]`, where rows are CE levels and columns are masses in `C`. The ordinary
spectrum loss operates across each row, `q(m | e)` versus `p(m | e)`. The trajectory loss
transposes the conditioning direction. For every retained mass, normalize its intensity across the
CE levels:

```text
q(e | m) = Q[e,m] / sum_j Q[j,m]
p(e | m) = P[e,m] / sum_j P[j,m]
L_trajectory = sum_m w_m * KL(q(. | m) || p(. | m)) / sum_m w_m
L = L_spectrum_ce + lambda_trajectory * L_trajectory
```

For exactly two CE levels, this column KL has the stable binary form

```text
y_m = Q[high,m] / (Q[low,m] + Q[high,m])
d_m = log P[high,m] - log P[low,m]
L_m = BCEWithLogits(d_m, y_m)
```

because `sigmoid(d_m) = P[high,m] / (P[low,m] + P[high,m])`. For three or four levels, apply soft
cross-entropy to `log P[:,m]` with target `q(. | m)`; do not turn the normalized trajectory dot
product into a BCE logit. A bounded cosine is not an unrestricted logit, and positive-only dot-product
BCE would merely reparameterize trajectory similarity without a calibrated target.

With uniform CE weight and `w_m = mean_e Q[e,m]`, the KL chain rule gives

```text
mean_e KL(Q[e,:] || P[e,:])
  = KL(q_mass_marginal || p_mass_marginal) + L_trajectory
```

so trajectory KL is a deliberate reweighting of the row-wise target, not independent supervision.
Before implementing ladder-aware batching, manually collate a fixed diagnostic ladder set and
compare trajectory KL with top-k row CE using loss correlation, fragment-logit gradient cosine, and
a mass-activity-reweighted row-CE control. Follow Alternative J0 in the full plan and stop if the
gradient cosine exceeds `0.95`, the reweighted control is equivalent, and neither gradient enriches
the measured crossing errors.

Use the model-independent mass activity weight

```text
w_m = mean_e Q[e,m]
```

Normalize mass weights within each window. Use the minimum frozen top-versus-runner endpoint
reliability across its levels as a window-level reliability factor. The top-k union and activity
weight keep tiny trajectories from receiving equal influence. Apply the frozen detection-limit and
raw-DAG support contract at every required level.

The ordinary spectrum loss anchors the absolute distribution at each level; the column loss asks
whether each important mass places its intensity at the correct CE levels. It imposes no monotonic
fragment-intensity assumption and includes winner switches as one special trajectory. It does not
add a new label matrix: row CE and column trajectory KL share the same optimum. Its purpose is to
reorganize the gradient around coherent per-mass CE profiles, so a null result means this
optimization bias does not add held-out signal.

Unlike endpoint pair BCE or top-k endpoint KL, this loss requires the adjacent predictions in the
same autograd graph. The current samplers do not provide that contract:

- `GroupSampler` samples within `group_id` and then globally shuffles the selected indices;
- `SpecMolFragDynamicBatchSampler` only packs the resulting stream by node or edge budget;
- `DualGroupDynamicBatchSampler` can co-locate multiple adduct-plus-molecule samples but does not
  preserve an ordered, instrument-specific CE ladder.

Only after the redundancy gate passes, implement the deterministic target and batching contract in
Alternative J1 of the full plan. It partitions each sorted context into consecutive,
non-overlapping, balanced windows of two to four CE levels; no endpoint is duplicated, and boundary
transitions are reported. Keep each window intact and pack complete windows under the existing
fragment-edge budget. Do not create a dense peak-by-peak or mass-by-mass matrix. After sparse
contender aggregation, the loss is `O(L * k)`.

An adjacent change-vector arm remains a separate, noisier alternative:

```text
delta_q_e = q_(e+1)_C - q_e_C
delta_p_e = p_(e+1)_C - p_e_C
L_delta = weighted_mean_e SmoothL1(delta_p_e, delta_q_e)
```

It is more directly aligned with the measured delta-cosine defect but amplifies endpoint noise.
Test it only if trajectory KL improves CE-profile calibration while held-out change direction
remains weak. Do not combine trajectory KL, delta `SmoothL1`, and trajectory cosine in the first
ladder arm, and do not normalize the delta by CE gap in v1.

Recommended escalation:

1. run the two-mass pair BCE as the cheapest causal test;
2. if it works narrowly, run top-k contender-set KL with no batching change;
3. if absolute contender ranking improves but held-out CE response direction remains weak, run the
   trajectory redundancy gate; add the mass-trajectory loss and batching only if it passes;
4. if trajectory calibration improves but adjacent direction remains weak, compare the separate
   ladder-delta arm;
5. add architecture capacity only after the corresponding loss fits training ladders but fails to
   generalize.

## Minimal implementation

1. **Reproduce the control.** Train the existing matched
   [`fraggnn_d3_ma_mi_nist20_amolc_all_gine1.yml`](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_gine1.yml)
   and export unfiltered predictions.
2. **Build one FT cache and apply the coverage gate.** Implement only the switch-pair fields needed
   by this loss. Report counts, molecule coverage, raw-DAG support, confidence, CE gap,
   `||delta q||_2`, replicate counts, and removals by eligibility rule. Proceed only with at least
   `5,000` retained switches over at least `500` validation molecules and no molecule contributing
   more than `1%` of final pair weight. Otherwise stop or revise the declared population before
   spending the pair-loss pilot runs.
3. **Add one loss path.** Load the cache by spectrum ID, aggregate mass aliases with `logsumexp`,
   compute `BCEWithLogits(log_P_A - log_P_B, q_A)`, and return zero for a batch with no eligible
   endpoints.
4. **Run a tiny overfit check.** On a small fixed subset, verify that soft pairwise BCE, evaluation
   pair KL, and endpoint winner error decrease and that gradients reach fragment logits.
5. **Run the pilot and confirmation matrix below.** Do not build later alternatives before this
   decision.

Required configuration is limited to:

```yaml
global_ranking_target_fp: null
ce_pair_loss_weight: 0.0
ce_pair_min_endpoint_confidence: 0.0
ce_pair_min_endpoint_reliability: 0.0
ce_pair_subset: switches
ce_pair_eligible_inst_types: [FT]
ce_pair_eligible_prec_types: ["[M+H]+", "[M-H]-"]
```

These keys must be explicit in the required config schema; training code must not supply fallback
defaults. The target cache records its named mass-matching contract (including instrument map,
relative tolerance, m/z floor, and strict boundary rule), split, DAG, and input fingerprints. The
builder and loader must fail if any fingerprint or resolved alias set differs.

## Small run matrix

### Stage A — one-seed pilot

Use the same seed and training budget:

| Arm       | `lambda_pair` | Purpose                              |
| --------- | --------------: | ------------------------------------ |
| control   |               0 | matched GINE-1 baseline              |
| pair-low  |            0.01 | test a weak auxiliary gradient       |
| pair-high |             0.1 | test a meaningful auxiliary gradient |

Choose between the two nonzero weights using validation ranking metrics declared below, not the
training loss. If neither improves the switch metrics without a stable-transition regression, stop
rather than adding more weights.

### Stage B — confirmation

Run the control and the selected real arm for three matched seeds. Only if the real arm passes the
promotion criteria, run a shuffled-pair control at the selected weight to verify that the gain came
from the target identities rather than generic auxiliary regularization.

The selected baseline's matched-seed run is also the Stage A control; Stage A therefore adds only
the two nonzero runs. Reuse that same control seed and budget in Stage B, which adds two control
seeds and three real-arm seeds, not a duplicate three-seed control. Thus confirmation adds at most
five unique runs before the causal control. It avoids a Cartesian sweep.

## Evaluation

Primary outcome:

- `cos_hun_inst_tol` on the natural FT validation distribution.

Mechanism checks, reported on switch transitions:

- both endpoints have the correct base mass;
- switch recall;
- model-versus-observed CE delta cosine;
- endpoint two-mass `KL(q_obs || p)` and TV, always against the observed `gamma = 1` ratios;
- among endpoints whose global winner is still wrong, whether the predicted argmax lies inside or
  outside `{A, B}`;
- all switch metrics stratified by CE gap and `||delta q||_2`, because the measured replicate
  ceiling covers mostly gap-5, larger-change transitions and must not be extrapolated silently.

Safety checks:

- stable-transition base-mass retention and false-switch rate;
- overall and high-CE supported base-mass accuracy;
- `cos_sim_0.01`;
- formula retrieval;
- performance by FT adduct and CE bin.

Do not pool stable and switch metrics. The measured switch headroom is the efficacy target; stable
transitions test whether the intervention became flip-happy.

## Promotion and stop rules

Promote the v1 change only when the three-seed result:

- improves mean `cos_hun_inst_tol` by at least `+0.005` over the matched control;
- has paired-spectrum and molecule-cluster bootstrap intervals excluding zero;
- improves switch recall or both-endpoint correctness;
- does not materially worsen stable false-switch rate, formula retrieval, or major FT strata;
- beats the shuffled-pair control if that control is activated.

Before Stage A, compute the fraction `f` of natural-distribution FT validation spectra that are
eligible switch endpoints. Under the optimistic assumption that all overall cosine improvement
comes from those endpoints, the `+0.005` promotion bar requires at least `0.005 / f` mean endpoint
gain. Report this implied effect size and compare it with the cache's available endpoint headroom;
do not run the pilot if the bar is unattainable on the supervised subset.

Stop this line when the target lacks adequate raw-DAG coverage, the tiny subset cannot overfit, the
real target performs like shuffled targets, or ranking improves without spectrum-cosine improvement
across three seeds.

## Add complexity only after a diagnosed result

- **Switches improve but stable false switches rise:** add eligible stable-pair anchors.
- **Two-mass calibration improves but winners still do not cross, and the wrong argmax is A or B:**
  test target sharpening (`gamma > 1`) with the identical eligibility mask.
- **Two-mass calibration improves but the wrong argmax lies outside `{A, B}`:** add the global
  top-ranking target; sharpening the A/B ratio cannot outrank a third mass.
- **Switch metrics improve but global base-peak accuracy remains weak:** add the global top-ranking
  target.
- **The loss fits training but not validation:** audit structural identity and target noise before
  adding model capacity.
- **The target is useful but the flat scorer cannot fit it:** test the
  [`fragment_context_gate_experiment_plan.md`](fragment_context_gate_experiment_plan.md) before the
  more expensive set-level competition head.

Until one of these outcomes is observed, those additions solve hypothetical problems and make the
first result harder to interpret and train.
