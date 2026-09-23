# Complementary-cut supervision experiment plan

> **Status:** PROPOSED — KL SUBSYSTEM AND CUT TARGET BUILDER EXIST; MATCHED GINE-1 GATES PENDING
> &nbsp;·&nbsp; **Updated:** 2026-08-10
>
> Reuse the implemented sibling-ranking KL and cache machinery. First prove that
> detection-limit-softened complementary-pair evidence improves a matched flat GINE-1 model; do not
> add conserved flow or a cut-specific head.
>
> Empirical results and reproduction details live in
> [`fragment_intensity_ranking_findings.md`](../../findings/fragnnet/fragment_intensity_ranking_findings.md).

## Decision

The experiment is:

```text
one-layer fragment GINE
+ flat node-probability head
+ ordinary spectrum loss
+ raw-intensity, detection-limit-softened cut target
+ responsibility-weighted cut-pair KL
```

The target is called `q_cut_evidence`. It estimates the relative observable evidence for the two
sides of one cut. It is not a ground-truth bond-breaking probability, fragment population, charge
retention probability, or edge-`beta` label.

Keep the flat head for this test. It can assign observed intensity directly to any candidate node,
whereas conserved flow requires mass to survive every ancestor. This isolates whether cut-focused
supervision helps without confounding the result with conservation.

## Existing implementation to reuse

The ranking subsystem is already implemented on branch `research/sibling-ranking-loss`, commit
`a30e92d4`:

- `src/fragnnet/model/loss.py::sibling_rank_kl` implements ragged, group-renormalized KL and
  per-spectrum weight normalization;
- `src/fragnnet/pl_model/fragnnet_pl.py` loads a ragged cache, batches it, and maps local DAG node
  indices to global batch indices;
- `config/*_template.yml` defines the loss weight, target path, and shuffled-target control;
- `tests/test_sibling_rank_loss.py` covers the KL, empty groups, batching offsets, and shuffling;
- `scripts/eval/build_sibling_rank_targets.py` is the reference cache builder.

The loader requires every cached `q_obs` entry to be strictly positive. This independently rules out
the old one-hot `(1, 0)` construction: such a cache fails at startup.

A complementary pair is just a two-entry target group, so `sibling_rank_kl` accepts it without new
loss mathematics. Do not copy a second scatter implementation. The earlier draft's loss snippet was
wrong: it referenced an undefined group index, counted pair weights twice, omitted per-spectrum
normalization, and computed cross-entropy rather than the logged KL.

The sibling-ranking branch, the current `feiw_v2` branch, and the FluxMPNN experiment branch have
diverged. Before training, create an integration line containing the existing ranking subsystem and
the matched control named below. This plan does not authorize a merge or branch switch.

For an initial cut-only pilot, the existing `sibling_rank_*` slot can load a pair cache verbatim, but
it will log the loss as `sibling_rank_kl`. Stamp the cache with `builder: cut_pair`, include its
fingerprint in run metadata, and require `cut_pair` in the W&B run name to prevent provenance
confusion. To run sibling and cut losses together, add only a second cache/loss slot that calls the
same `sibling_rank_kl` function:

```yaml
cut_pair_loss_weight: 0.0
cut_pair_target_fp: null
cut_pair_shuffle_targets: false
```

The attribution-weight rule is made by the offline builder and recorded in cache metadata:
responsibility is primary and strict is audit-only. It is not a training hyperparameter.

## Matched control

The attributable control is on branch `research/flux_mpnn`:

```text
config/exp_fraggnn_ma_mi/
fraggnn_d3_ma_mi_nist20_amolc_all_fluxmpnn_stage3_off.yml
```

That configuration uses one-layer GINE, reverse encoder edges, directed physical DAG edges,
`boundary_pair`, no fragment edge attributes, and edge-budgeted batching. It keeps the flat node
head and sets `fluxmpnn_stage3: False`. The supervised arms must differ from this control only in the
target cache and auxiliary loss settings. The older NodeMLP baseline is historical context, not the
control for this experiment.

With one GINE layer, complementary children A and B are not adjacent and receive the same immediate
parent message. Their ordering is distinguished by their own `boundary_pair`, `base_formula`, depth,
and H-formula features. A null result therefore means those per-node features could not learn the
two-side preference from this supervision; it does not establish that all cut-focused modeling is
useless.

The term “V3” refers only to the architecture and three-arm experiment documented on
`research/flux_mpnn`; it is not present on `feiw_v2`.

## Evidence prerequisite

The current evidence is a soft negative for immediate cut-loss training: responsibility weighting
concentrates on pairs the provisional NodeMLP already predicts well, and the absolute cut oracle
recovers only 3.37% of exact current-DAG intensity headroom. Global base-peak and CE-dependent
intensity ordering remain higher-priority. See
[`fragment_intensity_ranking_findings.md`](../../findings/fragnnet/fragment_intensity_ranking_findings.md) for
measurements, controls, limitations, and reproduction commands.

This plan remains useful as the specification for a matched GINE-1 decision. Do not begin Phase 3
unless the replicate, matched-baseline, specificity-control, and relative-headroom gates below pass.

## Complementary-pair definition and precondition

For parent fragment `P`, child nodes `A` and `B` form a complementary two-entry subset when:

```text
atoms(A) intersect atoms(B) = empty
atoms(A) union atoms(B) = atoms(P)
```

Deduplicate by `(molecule, parent, cut, unordered child pair)` so repeated DAG paths do not create
duplicate facts.

Both products exist only when DAG generation retains small fragments. The builder must fail closed
unless the loaded DAG's generation metadata records `min_frag_atoms == 0`. If current artifacts do
not carry that metadata, add or require a generation manifest before building targets; a path-name
assumption is not sufficient. The builder must also verify complement completeness from the loaded
atom masks. A nonzero value silently removes exactly the small counter-fragments for which charge
retention may be most asymmetric.

## Soft target and group weight use separate uncertainty channels

Peak absence is not evidence of zero fragment probability. A partner may be below detection, may
not have formed, or may be hidden by another candidate. Therefore the unobserved side must be
floored at the spectrum's detection limit rather than assigned zero.

Use exactly the same formula/H-shift, adduct, CMF, and instrument-tolerance matching contract as the
spectrum loss. For each peak `j` that matches only one side within the pair, let `I_j` be its raw
per-spectrum normalized intensity and let `n_j` be the number of all DAG nodes that match it.

First compute undiscounted directional intensity for the target:

```text
I_A = sum_j I_j  over directional A peaks
I_B = sum_j I_j  over directional B peaks
```

Separately compute responsibility-discounted evidence for the group weight:

```text
primary: c_j = 1 / n_j
strict audit: c_j = 1 when n_j = 1, else 0

w_A = sum_j I_j * c_j  over directional A peaks
w_B = sum_j I_j * c_j  over directional B peaks
```

Do not compare `w_A` or `w_B` with a detection threshold. They have already been reduced for
attribution ambiguity and are no longer on the observed-intensity scale. For example, a 20%-intensity
peak with confidence 0.1 and a clean 2%-intensity peak both give weighted evidence 0.02, but they are
not equally close to the detection limit. Raw intensity determines target softness; responsibility
determines how strongly the pair trains.

Let the candidate channels for side A be `C_A` and define its side-level floor as:

```text
L_A = min over k in C_A of I_lod(s, m/z_Ak)
L_B = min over k in C_B of I_lod(s, m/z_Bk)
```

`I_lod(s, m/z)` is the smallest intensity that could have been retained for spectrum `s` at that
candidate m/z, on the same scale as `I_j`. If preprocessing has an explicit relative cutoff, use it.
Otherwise estimate an instrument- and m/z-stratified floor on the training split only, freeze it
before validation, and audit sensitivity. `L_A` and `L_B` must be finite and strictly positive; fail
the builder otherwise. Do not substitute an arbitrary numerical epsilon.

Drop the pair unless at least one side has undiscounted directional evidence strictly above its
floor:

```text
max(I_A / L_A, I_B / L_B) > 1
```

Sub-LOD evidence is not converted into a fake near-tie.

```text
u_A = max(I_A, L_A)
u_B = max(I_B, L_B)

q_cut_evidence(A) = u_A / (u_A + u_B)
q_cut_evidence(B) = u_B / (u_A + u_B)
```

For example, with `L_A = L_B = 0.01`, a one-sided 2%-intensity peak gives
`q_cut_evidence = [0.67, 0.33]`, while a one-sided 20%-intensity peak gives approximately
`[0.95, 0.05]`, regardless of attribution confidence. Their confidence still changes their relative
training influence through:

```text
w_pair = w_A + w_B
```

The intensities are already normalized per spectrum, and `sibling_rank_kl` divides by the sum of
group weights in each spectrum. Another per-spectrum divisor would cancel exactly. No additional
confidence shrinkage of `q_cut_evidence` is used in v1; ambiguity remains an explicit, separate
weight channel. A true `[0.5, 0.5]` is allowed only when both sides have equal above-LOD directional
evidence. Unidentified and sub-LOD pairs are excluded.

### Same-mass and aliased pairs

If the candidate-ion envelopes of A and B overlap under the actual matching contract, an observation
in that channel cannot identify a side. Exclude the pair in the first builder rather than treating
the overlap as evidence. This includes exact same-mass fragments and collisions introduced by
instrument tolerance, adduct conversion, CMF, or H-shift channels.

Do not emit `[0.5, 0.5]` for an unidentified pair: “unknown” is not “equal.” Also do not cache a
`cut_pair_shared_intensity` field initially. Complementary disjoint formulas make this channel
equivalent to the mass-alias exclusion and it is expected to be nearly empty. Count its occupancy in
the audit; add a stored field only if the measurement shows a non-negligible, actionable channel.

## Phase 1 — Build and audit the second target cache

Implement `scripts/eval/build_cut_pair_targets.py` by reusing the sibling builder's peak matching,
cache layout, deterministic shuffling, and validation conventions. Emit the same training schema:

```text
version
spec_ids
spec_group_ptr
group_child_ptr       # every group has exactly two entries
child_nodes           # local DAG indices
q_obs                 # serialized name; semantically q_cut_evidence
q_obs_shuffled
group_weights          # responsibility- and overlap-adjusted w_pair
stats
metadata               # extension: builder=cut_pair, LOD contract, fingerprints, min_frag_atoms
```

The reference sibling cache already emits `version` and `stats`; `metadata` is an intentional
extension. The current loader ignores unknown keys, so the stamp is safe. Detailed diagnostics
belong in a sidecar report rather than enlarging the training cache.

### Overlap control

The scratchpad probe found a median of about 20 pairs per parent and a tail to 875. Per-spectrum
normalization does not correct this within-spectrum dependence. Apply operations in this order:

```text
deduplicate
-> exclude mass aliases
-> skip zero raw evidence
-> skip sub-LOD evidence
-> cap retained pairs
-> compute node multiplicities and final group weights
```

The first cache caps at 32 retained pairs per `(spectrum, parent)`, sampled uniformly with a fixed
seed. The cap must come after evidence filters; otherwise sampling mostly selects the roughly 80.7%
of candidates for which neither side was observed. Do not select top-intensity pairs, which would
create label-dependent selection bias.

A per-parent cap does not bound per-node multiplicity because nodes may have multiple parents. Let
`m_A` and `m_B` be the number of final retained pairs containing each node in that spectrum. Apply a
group-level correction expressible by the existing loss:

```text
group_weights = w_pair / ((m_A + m_B) / 2)
```

This is a group-level approximation to inverse entry multiplicity. It requires no change to
`sibling_rank_kl`; do not add unsupported per-entry weights in v1. Report multiplicity and weights
before and after this correction.

### Required audit

Report separately for train and validation:

- candidate, mass-aliased, zero-evidence, sub-LOD, retained, and capped pairs per spectrum;
- directional explained-intensity coverage under one common definition for cut and sibling targets;
- raw target distributions and strict-versus-responsibility group-weight distributions;
- fraction of retained pairs with `max(q_cut_evidence) > 0.99`, plus q histograms by evidence/LOD
  ratio;
- explicit train/validation counts, molecule counts, Kish ESS, and stratum counts for the
  `max(q_cut_evidence) >= 0.8` subset used by both Phase 2 gates;
- same-mass/overlap occupancy under each H-shift, CMF, adduct, and tolerance channel;
- p50/p95/max pairs per parent and node multiplicity before and after the cap;
- coverage by NCE, adduct, instrument, fragmentation mode, bond type, and fragment depth;
- retained-versus-discarded composition and mass distributions;
- Kish effective sample size, `(sum w)^2 / sum(w^2)`, before and after overlap adjustment;
- structural counts: distinct `(spectrum, parent, cut)` facts, distinct molecules, and distinct
  directional observed peaks.

Kish ESS and structural counts answer different questions; neither should be reported as a stand-in
for the other.

### Builder verification

- fail unless `min_frag_atoms == 0` is verifiable;
- deterministic pair construction, deduplication, capping, and shuffled targets;
- two positive target entries per retained pair summing to one;
- q uses raw intensity while responsibility changes only the group weight;
- sub-LOD and mass-aliased pairs produce no target or gradient;
- a supra-LOD one-sided peak yields a positive soft target, never `(1, 0)`;
- stronger evidence moves the target monotonically away from `0.5`;
- valid local node indices and stable input/cache fingerprints;
- `metadata.builder == cut_pair` and the run name/cache fingerprint preserve pilot provenance;
- one cache contains every train and validation `spec_id`, including spectra with zero groups;
- a batch of cached spectra with zero usable pair groups returns zero loss without error;
- cache-versus-dataset coverage is checked before training, because a wholly uncached batch raises;
- the cache loads through the existing ranking path without changing the KL implementation.

## Phase 2 — Measurement gates before supervised training

### Matched-control prerequisite

Reproduce the matched Stage-3-OFF control named above first, and save both its checkpoint and
node-probability inference artifact. The baseline-headroom and offline-oracle gates depend on that
trained artifact. If an existing checkpoint is proposed instead, verify its branch, full config,
data fingerprints, and metrics against the named control before use.

### Replicate gate

Match the same `(molecule, parent, cut, adduct, instrument, fragmentation mode)` across replicate
spectra. Phase 1 must explicitly size the high-confidence subset, fixed before these measurements as
`max(q_cut_evidence) >= 0.8`—approximately a 4:1 evidence-to-LOD ratio in the one-sided equal-LOD
case.

1. responsibility-weighted mean same-NCE TV must be at most `0.15`;
2. dominant-side flip is reported as a diagnostic, not used as the gate;
3. a within-molecule, within-context permutation of cut identities supplies the null, rather than a
   trivial cross-molecule shuffle.

TV has the same `[0, 1]` scale for binary and multi-entry groups, unlike dominant-side mismatch. If
the absolute TV floor exceeds 0.15, stop or narrow the target strata. Merely beating a shuffled
control is insufficient.

### Baseline-headroom gate

For the matched flat control, compute:

```text
q_cut_model(A) = P(A) / (P(A) + P(B))
q_cut_model(B) = P(B) / (P(A) + P(B))
```

Measure weighted KL, TV, and dominant-side mismatch against `q_cut_evidence` on the identical
high-confidence subset. The model's responsibility-weighted mean TV must exceed the same-NCE
replicate TV floor by at least `0.15`, with a paired bootstrap confidence interval excluding zero.
Dominant-side mismatch remains descriptive. Otherwise there is no demonstrated ranking defect large
enough to train against.

### Relative offline-oracle gate

Freeze the trained formula decoder and jointly edit only per-spectrum node logits toward all cut
targets, because pairs overlap. Rebuild the predicted spectrum after the edit. Run the identical
optimizer, budget, and regularization with fixed within-pair shuffled targets. Also run a
model-direction matched-confidence control: preserve each true target's confidence but place its
larger probability on the side currently preferred by the model. An exact current-model-ratio
target is an expected no-op and is diagnostic only; it does not supply a sharpening control.

For each spectrum define:

```text
delta_true     = cos(edited with true targets)     - cos(model)
delta_shuffled = cos(edited with shuffled targets) - cos(model)
delta_model_direction = cos(edited with model-direction targets) - cos(model)
delta_specific = delta_true - delta_shuffled
delta_direction = delta_true - delta_model_direction

DAG intensity headroom = cos_oracle - cos(model)
headroom fraction       = sum(delta_true) / sum(DAG intensity headroom)
```

Use `scripts/eval/dag_oracle_decomposition.py` to obtain the current-DAG intensity headroom. The
earlier decomposition attributed roughly 84% of the total cosine gap to intensity, so an absolute
gain without this denominator is not informative.

Proceed only if molecule-cluster bootstrap 95% intervals for `delta_true` and `delta_direction`
exclude zero and the absolute true edit recovers at least 5% of current-DAG intensity headroom.
Report `delta_specific` as a stress-control contrast, but do not use a damaging shuffled arm to
inflate the practical effect size. This gate can fail even though targets come from the observed
spectrum, because a statistically identifiable cut channel can still be too small to justify a
training objective.

## Phase 3 — Training arms

With the reproduced control fixed, select the sibling-only weight using its own validation sweep and
the cut weight from `{0.01, 0.1, 1.0}`. Then run the informative arm set at identical seeds and
budgets:

| Arm                | Sibling KL |        Cut-pair KL | Purpose                                                 |
| ------------------ | ---------: | -----------------: | ------------------------------------------------------- |
| control            |          0 |                  0 | attributable flat GINE baseline                         |
| sibling            |   selected |                  0 | test whether any parent-local ranking supervision helps |
| cut-pair           |          0 |           selected | isolate complementary-cut supervision                   |
| sibling + cut-pair |   selected |           selected | test complementarity                                    |
| shuffled cut-pair  |          0 | selected, shuffled | control gradient noise and regularization               |

The sibling arm is mandatory. Both losses supervise renormalized `P(node)` over subsets of one
parent's children; without it, a cut-pair win is not attributable to the cut construction.

Track `cos_hun_inst_tol`, `cos_sim_0.01`, spectrum cross-entropy, cut and sibling KL/TV/dominant-side
mismatch, train/validation gaps, formula retrieval, and results by target confidence, NCE, adduct,
instrument, bond type, and depth.

## Phase 4 — Promotion and stop rules

Promote only when:

- three-seed mean gain is at least `+0.005 cos_hun_inst_tol` over the matched control;
- paired per-spectrum bootstrap 95% confidence interval excludes zero;
- `cos_sim_0.01` agrees directionally;
- formula retrieval does not materially regress;
- held-out cut ranking improves beyond the sibling-only arm;
- shuffled cut targets do not reproduce the cosine improvement;
- gains occur across multiple NCE/adduct strata.

Stop when any decisive gate fails: inadequate high-confidence coverage, replicate TV above 0.15,
baseline TV within 0.15 of the replicate floor, less than 5% relative oracle headroom, shuffled
supervision matching the real target, or held-out ranking improving without spectral cosine.

If pair ranking improves but cosine does not, the remaining error is downstream of node ranking,
most likely `P(formula | node)` or ion-response modeling. Do not respond by adding GINE depth or an
undefined cut head.

## Deferred explicitly

No cut-specific residual head is specified here. A node may participate in many cuts and have
multiple parents, so `node_logit += cut_score` has no defined aggregation. Revisit architecture only
after the parameter-free auxiliary loss passes promotion, with a separate document that defines the
scored object and overlap aggregation.

FluxMPNN is also outside the first test. Only after flat GINE plus cut supervision passes should the
same pair KL be applied to the final FluxMPNN node marginal. The attributable comparison would then
be `(FluxMPNN + cut KL) - (flat GINE + cut KL)`.

## Relationship to existing work

- `fragment_bond_set_representation.md` supplies the complementary-pair definition, scratchpad
  priors, overlap warning, and `min_frag_atoms` caveat.
- `sibling_ranking_target_plan.md` describes the node-marginal ranking object; its implementation is
  on `research/sibling-ranking-loss` at `a30e92d4`, newer than the copy on `feiw_v2`.
- `model_improvement_plan.md` supplies the evaluation and promotion contract.
- `fluxmpnn_plan.md` on `research/flux_mpnn` defines the exact matched GINE-1 OFF control.
- `scripts/eval/dag_oracle_decomposition.py` supplies the relative intensity-headroom denominator.
- [`fragment_intensity_ranking_findings.md`](../../findings/fragnnet/fragment_intensity_ranking_findings.md)
  records the provisional NodeMLP measurements, controls, limitations, and reproduction commands
  that motivate the gates in this plan.
- [`global_fragment_intensity_ranking_experiment_plan.md`](global_fragment_intensity_ranking_experiment_plan.md)
  specifies the higher-priority global base-mass and adjacent-CE contender-pair experiment.
