# Global fragment intensity ranking experiment plan

> **Status:** PROPOSED — FINDINGS MEASURED; TARGETS AND LOSS NOT IMPLEMENTED
> &nbsp;·&nbsp; **Updated:** 2026-08-11
>
> Train the model to select the correct high-intensity fragment globally and to follow reliable
> winner changes across collision energy. Start with peak-mass losses on the existing flat head;
> add a set-level competition head only if the supervised target has held-out signal that the flat
> scorer cannot realize.

## Decision

The first experiment is:

```text
matched flat GINE-1
+ ordinary spectrum cross-entropy
+ confidence-weighted global base-mass ranking
+ confidence-weighted adjacent-CE contender-pair supervision
```

This is not a hard base-peak classifier. The supervised object is an observed **peak-mass group**,
which aggregates every DAG node/formula candidate matching that mass. This preserves ambiguity:
multiple fragments may explain one peak, but the spectrum metric cares about their summed intensity
at the observed mass.

Do not add another CE insertion path in the first test. The `ce_insert_location: mol,mlp`
experiment changed little, so lack of CE access is not the demonstrated defect. Do not add a
competition head initially either. First test whether a direct global-ranking objective changes the
held-out error.

The empirical basis is in
[`fragment_intensity_ranking_findings.md`](../findings/fragment_intensity_ranking_findings.md).
Complementary-cut supervision is a separate, lower-priority experiment specified in
[`complementary_cut_supervision_experiment_plan.md`](complementary_cut_supervision_experiment_plan.md).

## What the model must learn

For candidate peak-mass groups `k` in spectrum `s`, the model produces pre-normalization scores
`z_sk` and probabilities:

```text
p_sk = exp(z_sk) / sum_j exp(z_sj)
```

The required behavior has two levels:

1. **Global top ranking:** the observed dominant mass must outrank candidates arising from every
   cut, parent, and depth in the DAG—not only its siblings or complementary partner.
2. **CE-dependent winner switching:** within a fixed molecule/adduct/instrument/fragmentation
   context, if mass A dominates at lower CE and mass B dominates at higher CE, the prediction must
   reproduce that reversal.

Global softmax is necessary for a probability distribution but is not itself a competition model:
`argmax(softmax(z)) == argmax(z)`. The loss must train the pre-softmax ordering.

## Evidence that motivates the experiment

The provisional NodeMLP validation analysis found:

- base-peak accuracy 57.4% over 96,153 spectra;
- 84.1% of base-peak misses retained the observed mass in exported prediction support;
- supported misses had mean DAG-oracle cosine 0.979 but model cosine 0.578;
- within 7,519 fixed FT contexts, base-peak accuracy fell from 92.4% at the lowest available CE to
  42.3% at the highest;
- conditional on the observed mass remaining in support, FT accuracy fell from 92.8% to 50.8%;
- on adjacent FT CE transitions where the observed winner changed, the model changed its winner
  only 38.5% of the time and got both endpoints right only 13.1%;
- predictions captured 85.8% of observed FT variation around the fixed-context CE mean and were
  closer to the CE-specific target than the leave-one-CE-out average in 79.8% of cases, rejecting a
  pure static-average explanation;
- despite that continuous response, predicted FT variation at winner-changing adjacent steps was
  only 0.205 TV versus 0.298 observed, and the top node changed in only 35.9% of such transitions;
- the metric-consistent L2-optimal uniform gain at winner-changing steps was 1.010
  `[0.996, 1.025]` for FT and 0.847 `[0.760, 0.936]` for QTOF, so uniform amplification is not the
  missing fix; switch-only delta cosine was only 0.593 for FT and 0.478 for QTOF;
- the observed winner was stable in 71.9% of adjacent FT transitions, motivating stable-winner
  contender pairs as anchors rather than training only on reversals;
- adding CE directly at `mol,mlp` changed performance very little.

These are provisional because the checkpoint is NodeMLP and the inference export applied
`min_peak_int=1.0`. Repeat every gate below on an unfiltered matched GINE-1 export.

## Shared peak-mass contract

Both targets use one matching contract shared with the primary spectrum loss and evaluation:

- the same CRF/CMF, H-shift, isotope, and adduct mass conventions;
- the same instrument-relative tolerance and minimum-m/z tolerance floor;
- the same target preprocessing and precursor-peak policy;
- unfiltered candidate masses before `min_peak_int` inference pruning;
- summed predicted probability for all candidates matching one mass group.

If two observed peaks overlap under the instrument tolerance, merge them before identifying the
winner. If one candidate matches multiple merged observed groups, fail the builder: the mass groups
are not disjoint under the stated contract.

The builder must distinguish:

```text
raw DAG support      candidate mass exists before model scoring
export support       candidate survived inference intensity filtering
observed evidence    target spectrum contains intensity at that mass
```

Only raw DAG support determines whether a supervised target is realizable. Export support is a
diagnostic of model suppression, not a target eligibility rule.

## Step 0 — model-free replicate noise floors

Run [`debug_ce_replicate_noise_floors.py`](../../scripts/debug/debug_ce_replicate_noise_floors.py)
before building targets. Report coverage before either statistic, separately by instrument and
adduct:

1. **Winner-flip floor:** the fraction of same-CE replicate pairs whose observed base masses differ
   under instrument tolerance.
2. **Replicate delta cosine:** for adjacent CE levels with at least two replicates at both endpoints,
   form two mass-aligned \(\Delta q\) estimates from disjoint replicate choices and measure their
   cosine on the same 0.01-Da grid used for the model diagnostic.

The second statistic measures replicate reliability, not the literal ceiling for a
model-versus-observed cosine. Under independent additive errors, replicate-to-replicate agreement
is approximately the square of single-observation reliability. Compare the model and replicate
statistics only on the identical double-replicated transition cohort with a paired
molecule-cluster bootstrap. If their paired difference is indistinguishable from zero, the apparent
directional defect may be replicate noise: stop the contender-pair line before building a cache.
Do not transport the replicate estimate to a different cohort or instrument.

The provisional validation cohort gives:

| Instrument | Same-CE pairs / molecules | Winner-flip floor | Double-replicated switches / molecules | Replicate cosine | Matched model cosine | Paired difference | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FT | 4,519 / 1,202 | 0.171 `[0.156, 0.186]` | 572 / 386 | 0.885 `[0.856, 0.913]` | 0.765 `[0.737, 0.793]` | 0.120 `[0.091, 0.149]` | continue |
| QTOF | 160 / 10 | 0.200 `[0.103, 0.319]` | 15 / 7 | 0.035 `[-0.193, 0.302]` | not interpreted | not interpreted | inconclusive coverage |

Thus the kill criterion does not fire for FT on a valid cohort-matched comparison. The full-cohort
model value of 0.593 is not used in this decision because the double-replicated subset is easier.
The approximate independent-error correction gives \(\sqrt{0.885}=0.941\) for single-observation
reliability, but same-lab replicate correlation can bias 0.885 upward; treat both effects as
unquantified sensitivity considerations, not a corrected endpoint. QTOF cannot be admitted to the
target population on this evidence; obtain more replication or declare the first contender-pair
experiment FT-only.

The FT reliability estimate covers only 572 of 20,573 switching transitions (2.8%). It is not a
representative ceiling for the full target population: the matched transitions have median
\(\lVert\Delta q\rVert_2=0.396\) and CE gap 5, versus 0.235 and CE gap 10 for unmatched
transitions, and 57.0% of matched transitions lie in the largest-change quartile. The model reaches
0.765 on the matched cohort and 0.588 on the unmatched cohort. A shared-molecule restriction does
not remove the gap, while replicate averaging can denoise the matched target. Treat the Step 0 go
as evidence for high-confidence, well-measured FT crossings only; do not extrapolate it to weak
transitions without a measured ceiling.

The paired switch gap nevertheless persists after stratifying by full-population
\(\lVert\Delta q\rVert_2\): Q1 0.177 `[0.068, 0.279]`, Q2 0.106
`[-0.031, 0.234]`, Q3 0.109 `[0.061, 0.155]`, and Q4 0.113 `[0.084, 0.143]`.
Thus the FT switch go is not solely a large-change artifact; Q2 is underpowered rather than showing
a reversed point estimate. Stable anchors are different: their Q1 gap is 0.002
`[-0.097, 0.098]`, with measurable headroom appearing only in Q3 and Q4. Confidence eligibility
must therefore apply to stable anchors from v1, and stable and switch outcomes must be evaluated
separately.

## Target A — confidence-weighted global base-mass ranking

For spectrum `s`, let `b` be the merged observed mass group with largest normalized intensity
`I_b`. Let `I_2` be the second-largest observed group intensity, or zero for a one-peak spectrum.

Define top-winner confidence:

```text
c_top = max(0, (I_b - I_2) / max(I_b, I_lod(s, b)))
```

`I_lod` follows the frozen training-split detection-limit contract used by the cut builder. This
confidence approaches zero for near ties and one for a separated winner. Do not turn a near tie
into an equally strong hard label.

Let `p_b` be summed predicted probability inside the base-mass tolerance window. Let `p_k` be the
probability of every other disjoint predicted mass group. Define a smooth strongest-competitor
score:

```text
r_other = tau * logsumexp(log(p_k + eps) / tau for k != b)
```

The initial auxiliary loss is:

```text
L_global_top = c_top * softplus(rank_margin - log(p_b + eps) + r_other)
```

Use `rank_margin = 0` in the first experiment. `tau` smooths the maximum for optimization; declare
it in configuration and audit sensitivity rather than hiding a default. `eps` is numerical
stabilization inside the loss, not an observational floor or target probability.

Skip the auxiliary group when:

- the observed spectrum is empty;
- the base mass is outside raw DAG support;
- the winner is not separated from the runner-up after preprocessing;
- the mass group is not disjoint under the matching contract;
- target or predicted mass aggregation is non-finite.

The ordinary spectrum cross-entropy remains the primary loss. This auxiliary changes only the
ordering pressure near the global winner.

## Target B — adjacent-CE contender-pair supervision

### Controlled context

Group spectra by:

```text
(molecule, precursor type, instrument type, fragmentation mode)
```

Within a context, order distinct energy levels by the filled NCE value actually supplied to the
model. Record whether that value was raw NCE or converted from ACE. Conversion is monotonic within
the fixed context, so it preserves adjacent ordering.

Work at CE-level consensus before forming transitions. Replicate spectra at the same energy must
not create quadratic duplicates.

### Replicate consensus

For every mass group and CE level, average normalized observed intensity across same-CE replicates.
Record the fraction of replicates whose base mass agrees with the consensus winner. The primary
cache requires at least 0.8 winner agreement; audit thresholds 0.6, 0.8, and 1.0 on training before
freezing 0.8 for validation.

Same-CE replicate flips are a noise floor, not CE signal. Report them by instrument and adduct.

### Adjacent transitions only

Pair only adjacent distinct CE levels within a context. All-pairs construction overweights long CE
series and creates highly dependent examples.

Let A be the lower-energy consensus winner. Construct one contender pair per adjacent transition:

- if the higher-energy winner differs from A, let B be that winner and set `is_switch = true`;
- if A remains the higher-energy winner, let B be the non-A mass with the largest observed
  intensity at either endpoint and set `is_switch = false`.

Choose B only among masses with raw DAG support at both endpoints. Resolve exact ties
deterministically under the shared mass-group contract. Stable-winner pairs teach when not to
reverse and anchor the already-strong low-CE ordering; switch pairs remain the high-gradient subset
and retain their own evaluation metrics and switch-only ablation.

Retain the pair only when:

- A and B are distinct under instrument tolerance;
- A and B both exist in raw DAG support at both endpoints;
- both endpoint consensus winners pass replicate agreement;
- the A-versus-B intensity margin is positive at both endpoints;
- all four endpoint intensities `I_low(A)`, `I_low(B)`, `I_high(A)`, and `I_high(B)` are measurable
  or can be assigned the frozen detection-limit floor.

Define pair-specific endpoint confidence:

```text
c_pair_e = abs(I_e(A) - I_e(B)) / max(I_e(A), I_e(B), L_A_e, L_B_e)
```

This replaces Target A's `c_top`: B need not be the global runner-up, so the global
winner-versus-runner-up margin does not measure confidence in the supervised A-versus-B direction.

Define soft endpoint targets:

```text
q_low  = normalize(max(I_low(A),  L_A_low),  max(I_low(B),  L_B_low))
q_high = normalize(max(I_high(A), L_A_high), max(I_high(B), L_B_high))
```

Control target sharpness with one exponent:

```text
q_gamma_low  = normalize(q_low(A)^gamma,  q_low(B)^gamma)
q_gamma_high = normalize(q_high(A)^gamma, q_high(B)^gamma)
```

Use `gamma = 1` for the primary arm, which preserves the observed two-mass intensity ratios. As
`gamma` increases, the target concentrates on the endpoint winner; in the unique-winner limit
`gamma -> infinity`, it becomes one-hot. Implement this stably as `softmax(gamma * log(q))` after
the frozen detection-limit floor, rather than exponentiating very small values directly.

Define the predicted two-mass distributions from aggregated mass probability:

```text
p_low  = normalize(P_low(A),  P_low(B))
p_high = normalize(P_high(A), P_high(B))
```

The contender-pair loss is:

```text
L_ce_pair = w_pair * (
    KL(q_gamma_low || p_low) + KL(q_gamma_high || p_high)
) / 2
```

Use:

```text
w_pair = min(c_pair_low, c_pair_high) * replicate_agreement_low * replicate_agreement_high
```

Normalize pair weights within each molecular context so molecules with long CE series do not
dominate. Do not weight by CE gap in v1; adjacent gaps already reflect the available acquisition
grid, and gap weighting would favor particular datasets and instruments.

For the first FT arm, confidence is also an eligibility condition, not only a multiplier. Before
training, use the training-split replicate audit to freeze a minimum
`min(c_pair_low, c_pair_high)` threshold; `0.2` is the initial candidate, not a value selected from
validation performance. Apply it to stable anchors and switches, and use the same frozen mask for
the real arm and every control. Keep excluded low-confidence transitions for natural-distribution
evaluation, but do not let arbitrarily small weights silently admit the region where no replicate
ceiling has been measured. Because endpoint margin and \(\lVert\Delta q\rVert_2\) are different
quantities, the required audit must demonstrate that the threshold enriches for larger, more
reliable changes rather than assuming it does.

This single loss contains the proposed two-mass hard-negative objective. When the sharpened target
is one-hot on A,

\[
\operatorname{KL}([1,0]\parallel p)=-\log p_A,
\]

which is the two-candidate InfoNCE/cross-entropy term at prediction temperature 1. Therefore do not
add a second contender-pair InfoNCE loss, mixing weight, or prediction temperature in v1. Those
would reparameterize target hardness and logit scale without adding a distinct supervision signal.

The two endpoint terms can train independently after the offline cache fixes their shared
transition identity. Joint endpoint batching is not required for the real-target v1 arm. Preserve
`transition_id` and `is_switch` so evaluation can reconstruct paired metrics and compare all-pair
training with a switch-only ablation.

## Phase 1 — Build and audit the targets

Implement `scripts/eval/build_global_ranking_targets.py`. Emit one cache containing every train and
validation spectrum ID, including spectra with zero eligible groups.

The cache schema is:

```text
version
spec_ids
top_base_mass
top_confidence
top_eligible
ce_pair_spec_group_ptr
ce_pair_transition_ids
ce_pair_mass_a
ce_pair_mass_b
ce_pair_q_obs             # two positive entries per endpoint
ce_pair_endpoint_confidence  # unweighted A-vs-B c_pair at both endpoints
ce_pair_endpoint_ce       # exact model CE values needed by the CE-swap control
ce_pair_is_switch
ce_pair_group_weights
stats
metadata                 # matching, LOD, split, DAG, and input fingerprints
```

`ce_pair_q_obs` always stores the observed, detection-limit-floored ratios at `gamma = 1`. The
training loss applies `softmax(gamma * log(q_obs))` at runtime. Target sharpness and its runtime
confidence mask do not change the cache contents and are not part of the target-cache fingerprint;
they remain explicit run hyperparameters recorded with the checkpoint. This permits a gamma pilot
to reuse exactly the same observed target cache.

Mass values are target identifiers, not floating lookup keys. The loader must resolve each cached
mass through the same tolerance matcher used by the builder and fail if the match set changes.
Prefer storing resolved candidate formula indices plus the mass for audit when dataset indices are
stable and fingerprinted.

### Required audit

Report separately for train and validation:

- spectra and molecules with raw-DAG-supported base peaks;
- base support by NCE, instrument, fragmentation mode, adduct, mass, and molecule size;
- top-confidence histogram and counts above 0.2, 0.5, and 0.8;
- same-CE replicate base-flip rate and consensus agreement;
- controlled contexts, CE levels, adjacent transitions, stable-winner pairs, and winner switches;
- retained all-pair and switch-only counts and molecule-cluster Kish ESS by instrument and adduct;
- CE-gap, winner-mass-change, exact mass-delta, and depth-change distributions;
- on switch pairs, the alias-aware structural relation between A and B: B descendant of A, A
  descendant of B, both/ambiguous, or unrelated branches;
- for descendant cases, the minimum and distribution of directed path length after aggregating all
  candidate aliases matching A and B;
- endpoint target confidence and target TV distributions;
- replicate-ceiling coverage and model delta cosine by endpoint-confidence bin and
  \(\lVert\Delta q\rVert_2\) quartile;
- retained counts, ceiling coverage, and change-vector distributions above each candidate
  minimum-confidence threshold;
- maximum groups per spectrum and per molecular context;
- train/validation target and acquisition-stratum drift;
- the fraction of transition endpoints where A or B maps to multiple candidates;
- the fraction removed by every eligibility rule.

### Builder verification

- deterministic output for a fixed seed and inputs;
- no train-derived statistic is refit on validation;
- merged observed mass groups and candidate matching use one tolerance contract;
- unsupported base masses create no top-ranking gradient;
- near-tied winners have low weight at `gamma = 1` and are excluded by the runtime confidence floor
  in every `gamma > 1` arm;
- same-CE replicates are aggregated before adjacent transitions are built;
- only adjacent CE levels are paired;
- every `is_switch` pair reverses the consensus winner from A to B;
- every stable pair keeps A as the consensus winner and selects B by the frozen challenger rule;
- both endpoint `q_obs` vectors are positive and sum to one;
- mass aliases aggregate probability rather than assigning a node label;
- context-normalized weights prevent long CE series from dominating;
- cache and dataset fingerprints are checked before training;
- zero-group batches return zero auxiliary loss without error.

## Phase 2 — Measurement gates before training

Gates 1, 3, and 4 use the reproduced matched flat GINE-1 checkpoint and an unfiltered prediction
export. Gate 2 is model-free and uses the Step 0 replicate analysis.
The control config is
[`fraggnn_d3_ma_mi_nist20_amolc_all_gine1.yml`](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_gine1.yml).
Relative to the current NodeMLP baseline it changes only the fragment encoder to one-layer
bidirectional GINE and switches dynamic batching from node count to a 100,000-edge budget. For
tabular inference, pass `--min_intensity 0`; prediction filtering is not a training hyperparameter.

### Gate 1: realizable coverage

- raw-DAG base-mass support is at least 90% overall;
- support is at least 80% in every major instrument/adduct stratum with at least 500 spectra;
- at least 5,000 retained adjacent winner switches span at least 500 validation molecules;
- stable-winner and switch-pair coverage are both reported rather than assuming the cache grows by
  the inverse provisional switch rate;
- no single molecule contributes more than 1% of final contender-pair weight.

If coverage fails, fix candidate generation or narrow the declared target population before adding
a ranking loss.

### Gate 2: transition signal above replicate noise

Within every included instrument/adduct stratum, compare adjacent cross-CE winner flips with the
same-CE replicate flip floor. The molecule-cluster bootstrap interval for their difference must
exclude zero, and the absolute difference must be at least 0.05. The provisional FT point gaps are
0.089 to 0.138 across its four adducts; complete the joint molecule bootstrap before declaring the
formal gate passed. Recompute both sides with the same instrument-relative tolerance, winner
aggregation level, and eligible cohort; bootstrapping the existing fixed-0.02-Da consensus flip
rates against raw-spectrum replicate floors would not be a valid paired estimator. Current QTOF
coverage is insufficient and does not pass by extrapolation.

Declare Gate 2 only for the frozen confidence-eligible population. Report how much of that
population has an observed replicate ceiling and its distribution over
\(\lVert\Delta q\rVert_2\), CE gap, and adduct. If the confidence rule does not align the trained
population with the measured, higher-reliability region, either strengthen the eligibility rule or
leave the uncovered strata out of the first arm; a small continuous weight is not evidence that a
noisy target is safe.

Also compare the switch-only model delta cosine with the Step 0 replicate delta cosine. If the two
are statistically indistinguishable on the same eligible cohort, stop: the direction target would
fit measurement noise rather than recover model headroom.

Keep `ce_pair_target_gamma = 1.0` unless this gate passes for every instrument included in the
training population. A value above 1 puts more trust in winner identity, so it is not allowed while
any included instrument's same-CE replicate flip floor is unresolved. Do not infer safety from
confidence weighting alone: `w_pair` reduces the contribution of uncertain examples, whereas
`gamma` changes the target ratio for every retained example.

### Gate 3: matched-baseline defect

On raw-DAG-supported targets, report:

- global base-mass accuracy and confidence-weighted margin loss;
- base-mass accuracy conditional on support by CE bin;
- adjacent-switch recall;
- correct winner at both transition endpoints;
- stable-pair retention accuracy and false-switch rate;
- endpoint two-mass `KL(q_obs || p)` and `TV(q_obs, p)`, with `q_obs` fixed at `gamma = 1`.

The matched model must remain materially worse than the same-CE replicate ceiling. Use a
molecule-cluster interval; do not infer headroom only from a large raw spectrum count.

### Gate 3b: CE-response direction versus uniform gain

Before Gate 4, measure adjacent mass-aligned change vectors separately for stable and switching
transitions and for each instrument:

\[
\Delta q=q_h-q_l,\qquad \Delta p=p_h-p_l,
\qquad
\alpha^*=\frac{\sum \Delta p\cdot\Delta q}{\sum\lVert\Delta p\rVert_2^2}.
\]

Report switch-restricted delta cosine and a molecule-cluster bootstrap interval for \(\alpha^*\).
Do not estimate this gain by dividing cosine by the reported TV variation ratio: cosine and the
least-squares gain use L2 geometry, whereas TV is L1-derived. If \(\alpha^*\leq 1\), uniform CE
residual amplification and response-temperature fixes are ruled out for that stratum; the remaining
problem is response direction or contender identity. A stratum-specific magnitude intervention is
eligible only when the matched GINE-1 interval supports \(\alpha^*>1\).

### Gate 4: offline oracle specificity

Freeze the formula decoder and edit only candidate logits under the proposed top and contender-pair
losses.
Use the same optimizer, step budget, and regularization for:

1. observed targets;
2. within-context shuffled contender-pair identities;
3. model-direction targets with observed confidence;
4. exact current-model targets as a no-op check.

Rebuild spectra and measure absolute `delta_true`, not only true-minus-damaging-control contrasts.
Proceed only if:

- molecule-cluster 95% interval for absolute `delta_true` excludes zero;
- absolute mean gain is at least `+0.005 cos_hun_inst_tol`;
- observed targets outperform the model-direction control;
- gain is present in multiple CE and adduct strata;
- current-model targets are numerically a no-op.

## Phase 3 — Loss-only training arms

Add explicit required config keys to `config/template.yml`:

```yaml
global_ranking_target_fp: null
global_top_rank_loss_weight: 0.0
global_top_rank_temperature: 0.1
global_top_rank_margin: 0.0
ce_pair_loss_weight: 0.0
ce_pair_target_gamma: 1.0
ce_pair_min_endpoint_confidence: 0.0
ce_pair_shuffle_targets: false
ce_pair_swap_ce_control: false
```

No model or experiment hyperparameter may use a fallback default. A positive loss weight requires a
target path and matching cache metadata. For the first FT experiment, a positive
`ce_pair_loss_weight` also requires the frozen nonzero confidence floor selected by the Phase 1
coverage audit; `0.0` is valid only when the pair loss is disabled or for an explicitly declared
diagnostic ablation.

Run the following arms with identical seeds, sampler budget, and primary loss:

| Arm                     | Global top | CE pair | Pair subset | Purpose                                      |
| ----------------------- | ---------: | ------: | :---------- | -------------------------------------------- |
| control                 |          0 |       0 | —           | matched flat GINE-1                          |
| top only                |   selected |       0 | —           | isolate global winner supervision            |
| pair only               |          0 | selected | all         | learn CE-conditioned ordering with anchors   |
| switch-only ablation    |          0 | selected | switches    | measure the value and risk of stable anchors |
| top + pair              |   selected | selected | all         | test complementary global signals            |
| shuffled pair           |          0 | selected | all         | control gradient noise and regularization    |
| CE-swapped pair         |          0 | selected | all         | test whether correct CE assignment matters   |

Report stable-anchor and winner-switch losses and metrics separately for every arm. Never use a
pooled all-pair improvement as the go criterion: low-magnitude stable anchors can serve a
regularization role while having no recoverable directional headroom.

For each real loss, select from weights `{0.01, 0.1}` in a one-seed pilot using held-out ranking
metrics, then run the selected arms for three seeds. Do not launch a full Cartesian sweep.

Select the contender-pair loss weight with `ce_pair_target_gamma = 1.0` and the frozen eligibility
floor. Only after Gate 2 passes, and only if the soft target improves endpoint calibration without
enough winner crossing, keep the selected weight and eligibility mask fixed and pilot `gamma` in
`{2.0, 4.0}`. Compare against a `gamma = 1.0` arm with the identical mask, so a gamma effect is not
confounded with removing near ties. Re-run Gate 2 on the restricted population before training. Do
not jointly sweep loss weight, target sharpness, and the confidence floor.

The CE-swapped control keeps the same A/B pair and endpoint targets but exchanges the two endpoint
CE values. The ordinary spectrum loss must still use the correct CE. Compute this control through a
separate auxiliary-only forward (or an exactly equivalent detached feature replay); do not train
the primary spectrum objective on deliberately incorrect CE. This control costs extra compute and
is run only at the selected pair-loss weight.

The existing whole-spectrum pairwise similarity losses are not substitutes for these arms. They
match spectrum-to-spectrum similarity matrices and do not encode the identity reversal of two
specific global contenders.

## Phase 4 — Conditional competition head

Do not implement this phase unless loss-only training shows that the target is useful but the flat
scorer cannot fit it without harming the primary spectrum loss.

The minimal head is an `O(Nd)` CE-query attention pool over fragment nodes:

```text
a_f = attention_mlp([h_f, e_ce])
c_s = sum_f softmax_within_spectrum(a_f) * value(h_f)
delta_f = residual_mlp([h_f, e_ce, c_s, h_f * project(c_s)])
node_logit_f = base_node_logit_f + delta_f
```

Requirements:

- pool separately by spectrum using fragment batch indices;
- add one scalar residual per fragment node before formula-channel normalization;
- zero-initialize the residual output so the initial model equals the baseline;
- retain the ordinary flat score as a residual path;
- support CPU and CUDA;
- avoid full `O(N^2)` fragment self-attention;
- expose attention weights and residual magnitudes for audit.

Compare head-only, loss-only, and head-plus-loss. A head win without held-out ranking improvement is
not evidence of meaningful competition.

## Deferred alternatives and activation criteria

Do not combine these ideas into one large experiment. Start with the Phase 3 contender-pair KL at
`gamma = 1`, inspect which diagnostic remains weak, and activate at most one alternative at a time.
This keeps the result attributable and avoids hiding a failed target behind added capacity.

| Alternative | Activate when | What it tests | Main risk |
| --- | --- | --- | --- |
| 2-by-2 CE-assignment contrast | The model changes, but assigns the right spectra to the wrong CE endpoints | Whether each endpoint response is more compatible with its own CE than the neighboring CE | Requires paired endpoint forwards |
| CE change-vector direction matching | CE residual cosine remains too low after contender-pair training | Whether the direction of the whole-spectrum change is underfit | Sensitive to noisy small changes |
| Dynamic hard-negative ranking | The smooth global margin is diluted by many irrelevant candidates | Whether the current false top prediction is the useful competitor | Model-selected negatives can reinforce errors |
| Structured continuous CE basis | Losses expose the right target but the generic CE MLP still underfits it | Whether the architecture needs an explicit fragment-specific CE-response curve | Extra capacity can overfit sparse CE grids |
| Replicate-aware soft transition targets | Confidence filtering removes too many transitions or same-CE winners are unstable | Whether uncertainty should be learned instead of discarded | Replicate counts can be too small |
| Joint-trajectory teacher and distillation | Direct pair targets remain noisy after replicate aggregation | Whether a teacher that sees the CE sequence can provide smoother supervision | Leakage or teacher bias if splits are mishandled |
| Censored visibility/intensity factorization | Winner changes are dominated by peak appearance or disappearance | Whether detectability and conditional intensity need separate mechanisms | More labels and calibration choices |
| Listwise top-k ranking | Base-peak accuracy improves but top-k ordering or spectral similarity does not | Whether supervision must cover several globally competitive masses | Can duplicate the existing spectral loss |
| Transition curriculum and balanced sampling | Valid winner switches are rare relative to ordinary spectra | Whether the signal is currently diluted at the batch level | Changes the effective training distribution |

### Alternative A: 2-by-2 CE-assignment contrast

Let \(p_l\) and \(p_h\) be the model spectra at two CE endpoints and let \(q_l\) and \(q_h\) be
their observed targets. With a declared spectrum compatibility score \(S\), require the matched
assignment to beat the crossed assignment:

\[
\mathcal{L}_{\mathrm{assign}} =
\operatorname{softplus}\left[
m-
\left(
S(p_l,q_l)+S(p_h,q_h)-S(p_l,q_h)-S(p_h,q_l)
\right)
\right].
\]

This directly tests CE specificity: predicting two different spectra is insufficient if each is
more compatible with the other endpoint. Use neighboring supported CEs as the hard negative and
include a shuffled-CE control. This arm requires both endpoints in the same training item or an
extra forward pass.

### Alternative B: CE change-vector direction matching

Define mass-aligned response vectors

\[
\Delta q=q_h-q_l,\qquad \Delta p=p_h-p_l.
\]

Supervise change direction:

\[
\mathcal{L}_{\Delta}=
1-\cos(\Delta p,\Delta q).
\]

This targets which masses rise and fall without assuming that a uniformly larger response is
better. Gate out pairs with negligible observed \(\lVert\Delta q\rVert\), and report the loss
separately by CE gap. Add a magnitude term such as
\(\operatorname{Huber}(\Delta p-\Delta q)\) only for an instrument stratum whose matched Gate 3b
interval supports \(\alpha^*>1\); otherwise it can oppose the direction objective or reward further
shrinkage. This is the most direct alternative if held-out CE residual cosine remains low even when
winner classification improves.

### Alternative C: dynamic hard-negative global ranking

For each spectrum, select the model's highest-scoring incorrect mass group \(n\) against the
observed base mass \(b\), with the selection detached from gradient computation:

\[
\mathcal{L}_{\mathrm{hardneg}}=
\operatorname{softplus}\left[m-s_b+s_n\right].
\]

Prefer the neighboring-CE winner as the negative when it is available; otherwise use the current
false top prediction. This focuses optimization on the actual global competitor. It is only worth
testing if the smooth log-sum-exp base-mass margin is demonstrably diluted. Track negative identity
and refresh it online without differentiating through the argmax. A fixed or stale mined-negative
cache is not preferred.

### Alternative D: structured fragment-specific CE response

If the loss target works but the existing `mol,mlp` CE insertion cannot express enough crossing,
give each fragment a small continuous response basis:

\[
r_f(E)=\sum_{k=1}^{K}u_{f,k}\,\phi_k(E),
\qquad u_f=g(h_f),
\]

where \(h_f\) is the fragment representation and \(\phi_k\) are fixed normalized RBF or spline
basis functions over CE. Add \(r_f(E)\) to the fragment score before global normalization. Keep
\(K\) small, zero-initialize the residual path, and center the basis response across the supported
CE grid so it learns CE-dependent deviations rather than replacing the base fragment score.

This makes the missing interaction explicit:

\[
\text{fragment structure}\times\text{continuous CE}\rightarrow
\text{fragment-specific score change}.
\]

It permits non-monotonic responses and crossings, unlike a monotonic depth prior. Test it only
after a loss-only arm shows that the target is learnable but capacity-limited.

### Alternative E: replicate-aware probabilistic transitions

When replicate spectra disagree, estimate an empirical winner distribution or pairwise win
probability instead of collapsing the target to one hard winner. For example, if \(A\) beats \(B\)
in \(r_A\) of \(R\) valid replicates, use a smoothed target

\[
q(A\succ B)=\frac{r_A+\alpha}{R+2\alpha}
\]

in a binary cross-entropy or two-class KL loss. This recovers information currently discarded by a
strict confidence gate while preserving uncertainty. First verify that replicate count and coverage
are sufficient; otherwise the apparent soft probability is just a noisy estimate from one or two
measurements.

### Alternative F: joint-trajectory teacher and single-CE student

A teacher can process all supported CE queries for one molecular context jointly and learn a
smooth, mass-aligned CE trajectory. The production student still receives one CE at a time and is
distilled toward the teacher's per-CE distribution and response differences. The teacher objective
should include the same observed spectra plus explicit transition losses; simply distilling the
current model would reproduce its winner inertia.

Required leakage discipline:

- split by molecule before teacher fitting;
- train the teacher only on training molecules;
- never provide an observed held-out spectrum as a teacher input;
- evaluate the student directly against observed held-out spectra, not only teacher targets;
- compare against an equally trained student without distillation.

Use this only if direct targets remain unstable after replicate-aware aggregation. A teacher is a
denoising and representation mechanism, not evidence by itself that the learned CE response is
correct.

### Alternative G: censored visibility plus conditional intensity

If analysis shows that CE transitions are mostly peak appearance and disappearance, factor the
prediction into

\[
P(\text{visible}_f\mid h_f,E)
\quad\text{and}\quad
P(I_f\mid \text{visible}_f,h_f,E).
\]

Treat below-LOD observations as censored rather than exact zero intensity. Combine the two factors
before the global mass softmax. This gives the model explicit machinery for CE-dependent onset while
retaining global competition. Do not introduce it unless an audit shows visibility transitions,
rather than redistribution among already visible peaks, explain a substantial part of the missed
winner changes.

### Alternative H: listwise top-k ranking

If optimizing the base peak fixes top-1 identity but degrades the next strongest peaks, add a small
listwise loss over the observed top-k mass groups. Use the observed normalized intensities as a soft
ordering target rather than converting every pair into a hard comparison. Report base-mass
accuracy, top-k recall, Spearman rank correlation, and spectral similarity together. This is an
escalation from base-only supervision, not a default addition to it.

### Alternative I: transition-balanced sampling and curriculum

This changes signal delivery rather than model semantics. Ensure that a declared fraction of each
batch contains eligible same-context CE pairs, and begin with high-confidence, larger-margin
transitions. Include adjacent near-crossings only after a replicate audit supports expanding the
frozen eligibility population; curriculum order is not a substitute for target reliability.
Preserve an unmodified validation
distribution and log both sampled and natural transition frequencies. This can accompany one
transition objective, but it must have its own sampler-only control so an improvement is not
misattributed to the loss.

### Recommended activation order

1. Run the adjacent-CE contender-pair KL arm at `gamma = 1` with stable anchors, balanced transition
   sampling, its switch-only ablation, and the shuffled and CE-swapped controls.
2. If Gate 2 passes and calibration improves but crossings do not, keep the loss weight fixed and
   test target sharpening with `gamma` in `{2.0, 4.0}` after applying the frozen confidence floor;
   include a `gamma = 1.0` arm with the same floor.
3. If endpoint identity is confused, test 2-by-2 CE assignment; if response direction remains weak,
   test cosine change-vector matching instead. Do not add uniform gain unless Gate 3b supports it.
4. If the correct loss fits training transitions but held-out identities remain wrong, use the
   structural switch audit to decide whether to test the structured CE-response basis: prioritize
   it for recurrent parent-to-descendant handoffs, and deprioritize it for unrelated-branch switches.
5. If target coverage or reproducibility is the bottleneck, test replicate-aware targets before a
   teacher. Use a trajectory teacher only when simpler aggregation is inadequate.
6. Test the censored visibility factorization only after a visibility audit, and listwise top-k only
   after a base-only loss exposes a top-k tradeoff.
7. Move to the Phase 4 competition head only when the remaining failure specifically requires
   comparing the candidate set, rather than another CE-response target.

### Ideas not prioritized

- Generic same-molecule contrastive learning is misaligned because it would pull spectra at
  different CEs together, while the present failure is insufficient separation.
- Predicting CE from the output spectrum can be satisfied by generic CE signatures without fixing
  which fragment wins, so it is not a sufficient auxiliary target.
- A monotonic fragmentation-depth prior is not supported as the next move: fragment response can be
  non-monotonic, and the model-free CE-extent headroom analysis was a closed null result.
- Full self-attention over all DAG nodes is not the first architectural step. The proposed linear
  competition head is a cheaper test of whether explicit set-level comparison is needed.
- Adding all objectives simultaneously would prevent causal interpretation and make loss weighting
  the dominant experiment.

## Evaluation and promotion

Track:

- `cos_hun_inst_tol` and `cos_sim_0.01`;
- raw-DAG-supported and overall base-mass accuracy;
- base accuracy by NCE, instrument, mode, adduct, and molecule-size quartile;
- adjacent-switch recall and both-endpoint accuracy;
- stable-pair retention accuracy, false-switch rate, and all-pair versus switch-only results;
- endpoint two-mass `KL(q_obs || p)` and `TV(q_obs, p)`, always evaluated against the stored
  `gamma = 1` observed ratios regardless of the training gamma, reported separately for stable and
  switching pairs;
- matched-peak Spearman and predicted-minus-observed entropy;
- formula retrieval, including correct- and wrong-candidate scores;
- training versus validation ranking loss and accuracy;
- results on replicate-stable versus unstable targets.

The training-target KL at `gamma > 1` may be logged as an optimization diagnostic, but it must be
named separately and cannot be used for cross-arm comparison or promotion.

Promote only when:

- three-seed mean gain is at least `+0.005 cos_hun_inst_tol` over the matched control;
- paired spectrum bootstrap and molecule-cluster bootstrap intervals exclude zero;
- supported base-mass accuracy improves overall and at high CE;
- adjacent-switch and both-endpoint accuracy improve beyond both shuffled-pair and CE-swapped
  controls without degrading stable-pair retention;
- `cos_sim_0.01` agrees directionally;
- formula retrieval does not materially regress;
- improvement spans multiple CE and adduct strata.

## Stop rules

Stop or redesign when:

- raw-DAG support or reliable-transition coverage fails Gate 1;
- cross-CE switch signal does not exceed replicate noise;
- shuffled-pair or CE-swapped controls match real-target improvements;
- the oracle edit cannot produce `+0.005` absolute cosine;
- ranking metrics improve but spectrum cosine remains unchanged across three seeds;
- training targets fit but held-out transitions do not, indicating memorization or target noise;
- high-CE gains come only from suppressing unsupported spectra or one molecule/class;
- formula retrieval regresses materially;
- a competition head changes sharpness without improving winner identity.

If top ranking improves but cosine does not, inspect `P(formula | node)`, isotope/peak aggregation,
and false-peak intensity before adding encoder depth. If support fails at high CE, return to DAG
coverage rather than forcing the ranking head to solve an unrealizable target.

## Implementation sequence

0. Run the model-free replicate floors and freeze the eligible instrument/adduct population.
1. Build and audit `build_global_ranking_targets.py`.
2. Add focused tests for mass aggregation, ties, support, replicate consensus, adjacency,
   stable-challenger selection, pair confidence, structural relation, CE swapping, and context
   weighting.
3. Implement the peak-mass top-margin and endpoint contender-pair KL losses.
4. Add explicit configuration keys and cache validation.
5. Run the four offline-oracle controls.
6. Run loss-only training arms.
7. Implement the competition head only if Phase 4 is activated by the loss-only result.

## Relationship to existing work

- [`fragment_intensity_ranking_findings.md`](../findings/fragment_intensity_ranking_findings.md)
  supplies the empirical defect, CE-transition statistics, and limitations.
- [`model_improvement_plan.md`](model_improvement_plan.md) supplies the three-seed promotion and
  retrieval-protection contract.
- [`pairwise_loss.md`](../reference/pairwise_loss.md) documents existing samplers and explains why
  the shipped whole-spectrum pairwise losses are a different objective.
- [`complementary_cut_supervision_experiment_plan.md`](complementary_cut_supervision_experiment_plan.md)
  specifies the lower-priority local cut experiment.
- [`sibling_ranking_target_plan.md`](sibling_ranking_target_plan.md) specifies parent-local node
  competition; this plan addresses competitors anywhere in the DAG.
