# Candidate correspondence and transfer of fragmentation competition

> **Status:** STOPPED · **Updated:** 2026-09-29
> Gate B complete and negative: ion reference corrections select zero; loss alignment degrades
> held-out cosine. No new competition loss or full training run. Existing cm0–2 staging preserved.

## Objective and distinction

Test whether structurally corresponding fragmentation outcomes provide useful supervision for
MASSter's intensity allocation across molecular identities. The first experiment must establish
incremental information beyond frozen `i1`, rather than merely show that related spectra overlap.

The [whole-spectrum pilot](masster_cross_molecule_supervision_gate.md) supervises two similarities
through pooled precursor/context representations. Its census is positive, but it does not locate
which candidates carry the transferable signal. This proposal uses candidate correspondences
and ultimately supervises the actual output allocation. It is not independent chemical data.

The [post-cut descriptor null](../../findings/masster/masster_i1_postcut_information_gate_2026_09_29.md)
lowers the prior for another structural description alone. Here the additional evidence is
measured behavior of training-library reference molecules. The separately developed
[retained-fragment screen](../../findings/masster/masster_fragment_connectivity_local_screen_2026_09_29.md)
is now negative; this design does not depend on that branch.

## Literature premise and its limit

[MARASON, ICML 2025](https://arxiv.org/html/2502.17874v2) aligns target/reference fragments and
uses measured reference intensities during prediction. Its matching incorporates fragment and
fragmentation-DAG embeddings. Table 3 reports cosine 0.739 without retrieval, 0.737 for naive
concatenation, and 0.757 for its best neural matching variant. Table 4 shows weaker gains at low
Tanimoto. These results motivate alignment but do not establish a benefit from training-only
pair supervision, or gains for MASSter. The proposed objective below is our hypothesis.

## Refined unit: observable outcomes, then competition

A measured peak need not identify one retained atom set: multiple candidate structures and
ion/H channels can share its mass. Formula equality also does not identify a mechanism.

1. Generate candidate pathways independently of measured query peaks. Retain fragment atom
   membership, ion mode, H shift, mass, and the link to the final predicted probability.
2. Group pathways into experimentally indistinguishable candidate peaks using the existing
   evaluation convention. Supervise the sum of predicted probabilities in each group.
3. Attach observed intensity once to each resolvable group. Keep structural alternatives as
   alternatives; do not assign the full peak intensity to every structure or use `i1`'s largest
   structural share as ground-truth identification.
4. Establish a high-confidence cross-molecule correspondence only where the plausible
   structures support it consistently. Audit excluded ambiguous mass matches and retained
   intensity coverage. A strict subset is useful for the first screen, not representative of
   all fragmentation chemistry.

Start with deterministic chemistry-based correspondence rules: retained subgraph plus boundary
context, and a separate view for neutral-loss composition plus cut environment. Formula alone
is insufficient. Exact and analogous matches must be reported separately. Keep an unmatched
state. Do not start with unconstrained learned all-to-all attention or assume one-to-one mapping.
Fingerprint/graph rules must be fixed on training data; query intensities must not choose the
reference, structural mapping, or candidate pair. Label availability can restrict supervision,
but its selection bias and coverage must be reported explicitly.

## Gate A: correspondence coverage, with bounded cost

Reuse the existing train-only neighbor/control manifest for an initial bounded extraction,
including globally unfamiliar queries. It supplies spectrum pairs, not fragment correspondences.
Preserve exact acquisition matching and same-identity exclusions. Record its precursor-mass
window as a limitation: this first screen will miss some larger structural changes.

Cache each unique spectrum's annotated `i1` output once. Existing `return_annotation` in
[MassterModel](../../../src/fragnnet/model/masster_model.py) exposes fragment membership and
pathway ion/H identities. The [post-cut probe](../../../scripts/debug/masster_memorization/debug_masster_postcut_information_gate.py)
already merges channels into candidate peaks and maps measured targets, but its saved
probability-weighted descriptor averages are not a correspondence ground truth. New group and
mapping tables are needed. No new GNN or DFT is needed for this gate.

Report counts of distinct molecular identities, donors and structural families, groups matched,
ambiguity rate, retained target-intensity coverage, and the number of pairs with TWO distinct,
non-overlapping matched outcomes. Split by molecular novelty and exact versus analogous matches.
Also report absent/weak peaks; selecting only four strong observed peaks would exclude many
channel-switch cases. Freeze matching and intensity-reliability rules before the next gate.
If two-outcome support is scarce or confined to close analogues, stop the proposed competition
objective rather than forcing correspondences. Set the minimum useful cohort from precision
and compute requirements before inspecting any correction gains.

## Gate B: does aligned experimental evidence correct `i1`?

Use frozen `i1` and a small correction on candidate-group logits, evaluated on molecular
identities excluded from correction fitting. References always come from training identities.
An efficient exploratory design follows the prior post-cut gate: two molecule-disjoint
validation folds, with early stopping inside the fit fold. Neither evaluation-fold labels nor
test labels enter fitting or retrieval. This remains validation-fitted diagnostic evidence,
not a deployable trained model or a final held-out benchmark. Historical validation reuse also
limits its independence; any promising training variant needs a fresh confirmatory evaluation.

Compare equal-capacity corrections with the same query cases and mappings:

| Control / arm | Evidence available to correction |
| --- | --- |
| Context control | `i1` probabilities, acquisition context, query candidate descriptors |
| Reference-structure control | Above plus reference structure/match confidence, no measured reference intensities |
| Aligned evidence | Above plus measured reference group intensities and competing-outcome ratios |
| Shuffled evidence | Same reference and features; reference intensities reassigned among compatible groups |

Keep shuffle eligibility/marginals, missingness and coverage matched; record unshufflable
strata. Do not use query residuals or query labels to choose donor evidence. An effect over the
context arm alone is insufficient if the reference-structure or shuffled control explains it.

Read molecule-paired production spectral cosine, candidate ranking/base-peak agreement, and
unfamiliar-molecule slices (nearest full-training Tanimoto <=0.4). Include unmatched cases with
zero correction in the overall metric, so coverage cannot inflate the headline. Bootstrap by
query identity and audit donor concentration; shared donors make observations dependent.
Require a positive incremental effect over both structure-only and shuffled controls in each
fold, with an interval excluding zero on the prespecified unfamiliar slice, before prioritizing
novel-molecule transfer. This is a screening criterion, not a promised effect size. A narrow
analogue-only effect supports an analogue-retrieval application instead.

## Conditional training design: learn changes in competition

Only after the gates, test a train-only auxiliary objective on matched outcome pairs. Suppose
query Q has outcomes a and b corresponding to reference R outcomes a' and b'. Let y denote
observed group intensities and p the model's FINAL aggregated group probabilities.

    observed_change = log((y_Qa + eps) / (y_Qb + eps))
                    - log((y_Ra' + eps) / (y_Rb' + eps))
    predicted_change = log((p_Qa + eps) / (p_Qb + eps))
                     - log((p_Ra' + eps) / (p_Rb' + eps))
    pair_loss = reliability_weight * Huber(predicted_change - observed_change)

Normalize spectra consistently first. The numerical floor, reliability policy and Huber scale
must be fixed from training-only intensity/noise distributions, with a sensitivity check.
Missing detected peaks are censored observations, not proof of chemical impossibility. The
initial high-confidence read can restrict positive intensities, but must separately quantify
what it misses about presence/absence transitions. Confidence must not be a freely learned
weight that can erase difficult examples.

This directly teaches the direction and size of a competition change. Example (illustrative):
a:b changes from 4:1 in R to 1:2 in Q; the desired change is log(1/8), not equality of the two
spectra. The primary spectral loss still identifies each spectrum's absolute allocation.
Gradients pass through existing decoder outputs; no separate similarity head can satisfy the
objective while leaving these outputs unchanged. No reference input is needed at inference.

This loss adds no new labels beyond the two spectra. Algebraically it couples their allocation
errors; the model can satisfy the difference with shared errors. Near-perfect training fit could
also leave little useful signal. It is a proposed inductive bias, not a mechanism learner.
Compare to matched within-spectrum ranking to distinguish correspondence from extra emphasis
on bright-peak ratios. Supervise bounded numbers of outcomes per molecule, normalize by
molecule/pair rather than raw match count, and preserve exposure across all arms.

## Decision sequence and controls

1. Coverage audit and frozen rules; no full training yet.
2. Incremental-information correction gate, with the controls above.
3. If positive, pilot the direct competition-change objective against the same sampler without
   it, shuffled correspondences, and matched within-spectrum ranking. Match training budget,
   outcome exposure, initialization and loss-scale selection procedure.
4. Judge production validation predictions and novelty slices, not auxiliary training loss;
   confirm a promising result on a second seed and independent evaluation.

A positive retrieval-evidence gate does not prove that training-only supervision will work.
If retrieval helps but the auxiliary objective does not, reference-conditioned prediction and
subsequent teacher distillation are separate future hypotheses. Neither is part of this pilot.
Do not train a flexible matcher jointly with the first auxiliary loss: it could choose easy
matches. Learned matching and DAG context become later ablations after fixed mappings work.

## Current state and next check

Gate A established formula-match coverage. Gate B is now complete and does not establish
incremental predictive value. On 239 validation queries (119 unfamiliar), ion-aligned reference
corrections select epoch zero in both folds; loss alignment is −0.00985 cosine versus raw i1 and
−0.00341 versus structure-only. See the
[canonical result](../../findings/masster/masster_reference_information_gate_2026_09_29.md).
Do not implement the proposed competition objective or launch a larger run from this evidence.
The separate cm0–2 pilot remains an independent, untested hypothesis. Reopening this plan requires
new chemical alignment evidence and a fresh gate; the bounded null is preserved below.


## Deterministic outcome-matching implementation (2026-09-29)

The first implementation deliberately tests observable chemical compositions rather than
structural graph equivalence. The earlier structural matching rules remain possible extensions.
The [audit script](../../../scripts/debug/masster_memorization/debug_masster_candidate_correspondence.py) and
[standalone config](../../../config/analysis/masster_candidate_correspondence.yaml) use:

- Frozen i1 epoch 59, CPU annotated inference, existing training-only census. Select 60 query
  identities with global donor similarity <=0.4 and 60 above it, seeded without spectral labels.
  One condition per query (lowest spectrum ID), its nearest eligible neighbor, and the first
  recorded mass/acquisition-matched random control. This is a bounded balanced sample, not an
  estimate weighted to the full training population; its one-condition selection may skew CE.
- Complete elemental composition reconstructed from retained atoms, implicit/explicit H,
  production CRF/CMF composition deltas and H shifts. Verify each against predicted monoisotopic
  ion mass (maximum allowed discrepancy 2e-5 Da). Only singly charged [M+H]+ is supported.
- Group predicted channels at six mass decimals; sum their probabilities. Retain all structural
  alternatives. Reject mixed-formula groups, overlapping instrument-tolerance windows, and
  candidates within 1.5 Da below the measured precursor or above it.
- Match uniquely occurring ion formulas, or nonnegative nonzero precursor-ion minus product-ion
  formulas. These are outcome correspondences, not proof of shared structure or mechanism.
  Also report the union, excluding conflicting endpoints to make its matches disjoint.
- Measured peaks enter only after mapping rules are fixed. Assign a peak only if exactly one
  candidate is within instrument tolerance; normalize intensity over the full eval-preprocessed
  target. Never copy one peak's intensity into several groups. Missing assignments remain zero
  observations, not chemical impossibility labels.
- Report both-positive matches, matches with >=1% total processed intensity at both endpoints,
  two such distinct outcomes, and covered target intensity. The 1% cutoff is a prespecified
  descriptive coverage threshold, not a validated noise model or training-loss hyperparameter.
  Save zeros and weaker matches too, allowing later presence/absence analysis.

Nine tests passed, including condition sensitivity preserving query identities and actual-model ion formula reconstruction and mass verification,
water-loss matches between different product formulas, ambiguous formula/mass cases,
structural alternatives, precursor/negative-loss rejection, union conflicts, label-independent
matching/selection, and reliability coverage. Ruff passed. Initial artifacts are under
`output/candidate_correspondence_2026_09_29/` and `high_nce/`; results and limitations are in
the linked finding. Both runs completed, with no fitting or model-quality evaluation.


## NCE-band continuation (2026-09-29; complete)

The user requested coverage beyond the two endpoint selections. Freeze five observed-NCE bands
before inspecting correspondence outcomes: [0,20), [20,40), [40,60), [60,100), [100,201).
Retain all 120 original query identities and, unlike the earlier high-NCE sensitivity, fix each
query's original neighbor AND random-control molecule across bands. Preserve instrument,
adduct and fragmentation mode. Among NCEs measured for all three molecules in a band, choose
the value nearest its midpoint (lower NCE breaks ties); lowest spectrum ID resolves replicate
conditions. Select from the full training metadata rather than the original census's three
conditions per query. No peak intensity enters selection and no CE interpolation is used.

[Selector/summarizer](../../../scripts/debug/masster_memorization/debug_masster_correspondence_nce.py),
[config](../../../config/analysis/masster_candidate_correspondence_nce.yaml). The existing
candidate audit accepts a preselected pair CSV and validates training IDs and exact acquisition
matching before annotated inference. Formula matching and the >=1% reliability cutoff are
unchanged. Reuse neither a different reference identity nor a different instrument to fill gaps.

Availability for the 60 unfamiliar queries is respectively 36, 51, 43, 45 and 34 across bands;
for the other 60, 32, 50, 50, 41 and 39. Fifteen queries per novelty group support ALL five bands.
The 842 selected spectrum pairs require 1,250 unique spectrum evaluations on CPU. Report both
available-case coverage with explicit denominators and the all-bands-common cohort; also retain
instrument-specific summaries. The common cohort has only 15 unfamiliar queries, so its curve
is descriptive and cannot justify precise population-level energy effects.

Twelve focused tests passed (nine correspondence tests and three band-selection/aggregation
tests). Ruff passed. The CPU run and provenance are under
`output/candidate_correspondence_2026_09_29/nce_bands/`. CPU audit complete in 392.8 seconds;
results are in the linked finding. At [40,60), 12/43 unfamiliar neighbor pairs versus 3/43
random pairs have two strong ion matches; at [60,100), 15/45 versus 6/45. The fixed cohort
of 15 unfamiliar queries shows 3 versus 1 and 6 versus 1 respectively. Extreme NCE has much
higher random-reference coverage. Next: bounded incremental-information check on moderate-NCE
ion matches, with loss matches separate; no new training objective until that check.

## Gate B frozen protocol (2026-09-29; before correction results)

The user authorized the bounded diagnostic. Coverage queries were training molecules, so this
read uses fresh validation queries and training-only references. The label-blind selector samples
120 identities per novelty stratum before retrieval, computes novelty against all eligible
training identities, and selects an exact acquisition match at observed NCE [40,100), preferring
60. Same formula/identity references are excluded. Random references exclude the three closest
eligible donors and match precursor-distance within 2 Da. Of 240 selected identities, 239 have a
usable neighbor/control condition (119 unfamiliar); the missing case remains in the manifest.
This balanced cohort is not a population-weighted estimate. No test labels are accessed.

Freeze two molecular folds, stratified by novelty. Within each fit fold reserve 15% of identities
for early stopping. Fit a width-32 residual MLP on frozen i1 log probabilities and query context
(log probability, mass, precursor mass, NCE, instrument, elemental composition, ambiguity counts).
Freeze that context correction for all reference arms. Add an equal-width, zero-initialized
reference residual for structure-only, aligned intensities, shuffled intensities, and random
reference intensities. Structure-only zeroes the measured-label feature slots. Shuffle intensities
among the matched groups of the same donor; preserve zeros and marginal values. References supply
match flags, similarity, mass context and ambiguity counts; no measured labels enter the structure
control. Unmatched groups get zero reference residual but retain the common context correction.
Fit ion-formula correspondence as the primary view; repeat loss-formula correspondence separately.
No strong-peak cutoff selects the fitting/evaluation cohort or correspondence.

Use full-spectrum candidate cross entropy (normalize uniquely assigned target intensity over
covered groups) for fitting; Adam, learning rate 0.001, weight decay 0.001, 60 epochs, patience 10,
16 spectra per batch, seed 397623. Fit feature normalization on the fit identities only. Include
epoch zero in early stopping. Select epochs on inner-fold production cosine, then report only
outer-fold predictions. Preserve complete target spectra, including uncovered peaks, in production
0.01-Da binned cosine; retain precursor and unmatched predicted groups. Also report base-peak and
strong-outcome ranking metrics. Bootstrap paired differences by query identity (2,000 replicates),
with fold signs, novelty and NCE slices and reference reuse counts. Gains over structure and shuffle
in both folds, plus positive unfamiliar-slice confidence intervals, are required to promote novel
transfer. Cross-fitted validation corrections are diagnostic; they are neither untouched test
results nor evidence that a training-only auxiliary loss works. No hyperparameter search or new
MASSter training run is part of this gate.

Sources: [selector](../../../scripts/debug/masster_memorization/prepare_masster_reference_gate.py),
[config](../../../config/analysis/masster_reference_information_gate.yaml).
Artifacts: `output/reference_information_gate_2026_09_29/`.

## Gate B execution and disposition (complete)

Implemented and ran the frozen protocol above. The [result](../../findings/masster/masster_reference_information_gate_2026_09_29.md)
records all controls, confidence intervals, NCE slices, source/artifact provenance and limitations.
Sixteen focused tests, Ruff, split/fold/formula audits and production-metric comparisons passed.
The ion corrections learned on fit data but did not beat epoch zero on either inner fold. Loss
alignment failed to beat structure in either outer fold. Promotion criteria fail; this candidate
transfer recipe is stopped. No new full-model training, submission, staging or history changes.

## Structural alignment audit (2026-09-29; frozen protocol, now complete)

The formula-correction null above remains closed. The user authorized a bounded structural
screen, without a new GNN or correction fit. Use the existing training-only fixed-reference
pairs at NCE [40,100), preserving both neighbor and random-reference identities. This avoids
using the previous validation gate to select a new matcher. Novelty here retains the coverage
census definition (maximum similarity to its donor pool), not the later full-training definition.

Retain all pathway atom sets, ion modes and H shifts in a separate annotated inference cache.
Verify atom numbering, hydrogens and chemical bonds against the source RDKit molecule. Construct
canonical exact retained heavy-atom connectivity signatures including elements, formal charges
and aromatic/bond types, excluding stereochemistry and parent hydrogen counts. A stricter key
adds oriented radius-one local environments around both ends of every boundary bond (retained
versus discarded roles), bond type/ring/conjugation flags, ion mode and H shift. These are
parent-graph environments, not computed ionic structures or proven reaction mechanisms.

Compare nested ion-formula matches: formula only; formula plus unambiguous retained connectivity;
formula plus connectivity and unambiguous cut/channel context. A peak with multiple distinct
signatures is excluded from the corresponding strict tier; do not choose the highest-probability
or best-label explanation. Equivalent atom sets can collapse to one signature. Save all
alternatives and ambiguity counts. This first screen does not implement approximate graph
matching, fragmentation DAGs or mass-shifted analogies.

Report coverage (matches, query intensity, two >=1% outcomes) on all fixed pairs, including empty
strict matches. On pairs with at least two matches and nonzero measured norms, report measured
intensity cosine and signed residual cosine, where residual = measured group fraction minus
frozen i1 group fraction. Shuffle donor intensity/residual values together within the matched
set (20 seeded permutations) and subsample formula edges to the strict edge count (20 repeats).
The latter controls raw edge count, not all chemical/label-selection differences. Average
conditions within query molecular identity before 2,000-replicate paired bootstrap; report
neighbor/random and novelty strata separately. Reference residual agreement is exploratory on
training molecules whose i1 fit is near perfect; it is not validation correction gain.

A follow-up correction gate requires at least 20 unfamiliar query identities with usable strict
matches, positive unfamiliar residual-agreement advantage over both shuffled and count-matched
formula controls (95% lower bounds >0), and retained coverage reported. Otherwise record the
screen as insufficient and stop. Any passing screen requires a separately frozen fresh-validation
protocol before fitting; no automatic full training. Config:
[structural audit](../../../config/analysis/masster_structural_alignment_audit.yaml).

## Structural audit disposition (2026-09-29)

The [structural audit finding](../../findings/masster/masster_structural_alignment_audit_2026_09_29.md)
records 358 training-only acquisition pairs, 110 query identities and 532 spectra. For 54 unfamiliar
identities, exact connectivity reduces measured query coverage from 38.00% to 0.85%; exact
cut/channel context gives zero. Neither strict tier retains two strong matched outcomes in any
of the 88 unfamiliar neighbor acquisition pairs.

The predeclared residual criterion mechanically passes connectivity (36 defined unfamiliar
identities), but a post-hoc sanity check exposes a confound: moving donor residuals also moves
the donor model predictions. Actual labels do not beat a zero-label reference or label-only
shuffle with predictions fixed. Original positive residual-screen outputs and post-hoc controls
are both preserved; the criterion is not retroactively called failed. This is insufficient
evidence for the conditional fresh-validation correction. Stop this exact-matching recipe.
No GNN, quantum calculation, correction fitting or full training was run. Twenty-three focused
tests, Ruff and artifact checks passed. A future approximate matcher needs a distinct rationale
for recovering measured coverage and label-specific evidence before another correction gate.
