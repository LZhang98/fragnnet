# Rare chemistry augmentation for NIST20 and MassSpecGym

> **Status:** PROPOSED · **Updated:** 2026-10-10
> Proposed pilot: learn edit effects from measured molecular pairs, test transfer to rare chemistry, then augment with partial intensity constraints. No implementation or training started.

## Objective and scope

Improve spectrum prediction for underrepresented scaffolds and chemical classes on unseen
molecules, without materially reducing overall accuracy. The user confirmed this definition
of rare on 2026-10-10. Start with MASSter on NIST20 tandem MS and the existing MassSpecGym
1.5 simulation benchmark subset. Preserve each dataset's current split, preprocessing,
adduct scope and unmerged targets. NIST23 EI is outside this plan.

MassSpecGym's benchmark split already enforces structural novelty. Expect held-out Murcko
scaffolds to be dominated by the unseen bin, potentially nearly all; this is an expectation
to quantify, not a measured scaffold census or a guarantee of scaffold disjointness. A2 is
therefore unlikely to discriminate held-out MSGYM rarity usefully. Prioritize the class arm
A3 there; retain A2 only if the census establishes a meaningful training-weight contrast and
an interpretable evaluation cohort. NIST keeps both scaffold and class arms eligible.

The first question is whether rare chemistry lacks training exposure, rather than whether
low-scoring spectra should be repeated. Repetition changes exposure; it creates no new
chemical information. Augmented observations are a separate experimental factor.

## Established evidence and constraints

- The [NIST generalization hub](../../workstreams/masster_generalization.md) records covered
  ranking failures at the prediction floor and negative noise/masking experiments. These
  results do not establish that rare-class sampling helps; do not relaunch generic input
  corruption under an augmentation label.
- The [MassSpecGym hub](../../workstreams/msgym.md) records the benchmark-subset-only decision,
  training overfit and the completed arm screen. Do not add the full MassSpecGym dataset or
  import NIST spectra into its training set in this experiment.
- The [official MassSpecGym repository](https://github.com/pluskal-lab/MassSpecGym) defines
  separate simulation and retrieval challenges with predefined splits and metrics. Preserve
  the local benchmark subset and report its scope when evaluating retrieval.
- The [reliability plan](masster_intensity_reliability_plan.md) explains why same
  molecule and nominal CE are insufficient replicate provenance; instrument identity and
  acquisition settings matter. Recompute eligibility on training data rather than reuse
  historical all-split replicate counts.
- [Current sampler code](../../../src/fragnnet/dataset/data_sampler.py) supports molecule/group
  sampling without replacement and adduct/instrument (optionally fragmentation-mode) balancing
  with replacement, without CE in that balancing key. It has no scaffold-aware mode.
  Its molecule mode also changes draw count; it is not a drop-in
  implementation of the fixed-budget controls below.
- [Architecture](../../architecture.md) and [batch reference](../../reference/dataloader_data.md)
  describe dataset/model boundaries. Augmentation must preserve input/output shape and the
  existing normalization, target intensity transform and candidate supervision semantics.

## 1. Define rarity using training molecules

Create a census separately for each dataset after its existing filters. Count unique
standardized molecules, not spectra or collision energies. Retain the existing split identity
and detect standardized identity overlap; report any conflicts before proceeding.

| View | Proposed definition | Use |
| --- | --- | --- |
| Scaffold | Existing `get_murcko_scaffold(mol, include_chirality=False)`; fixed 2D policy, record RDKit version | NIST sampling arm; MSGYM diagnostic unless census supports A2 |
| Chemical class | A frozen, versioned functional-group SMARTS panel applied identically to all molecules | Separate sampling arm and overlapping evaluation strata |
| Acquisition | Adduct, instrument, fragmentation mode, recorded CE type and range, missing metadata | Diagnose confounding; do not conflate with chemical rarity |
| Familiarity | Nearest-training-molecule fingerprint similarity, with training self-matches excluded | Diagnose novelty within and across rarity strata |

Reuse [get_murcko_scaffold](../../../src/fragnnet/utils/data_utils.py), which already defaults
to no chirality. The fixed 2D scaffold policy matches the NIST `inchikey_s` (first 14 characters,
2D connectivity) split described in the [preprocessing README](../../../preproc_scripts/README.md).
Do not introduce a second scaffold helper or silently switch to stereochemical scaffolds.

Proposed bins: zero training molecules = unseen; 1–5 = very rare; 6–20 = rare;
21–100 = intermediate; over 100 = common. Freeze these before reading validation outcomes.
Report scaffold and class bins separately. Acyclic molecules have an empty Murcko scaffold:
give them a separate category and use functional-group classes for their rarity analysis,
rather than treating the empty scaffold as a chemically coherent class. Keep unmatched
classes explicit. For multi-label classes, deduplicate molecules within each class.

Deliver counts of molecules, spectra, acquisition groups and eligible replicate groups;
inspect whether rare classes are concentrated in one instrument or CE range. Measure training
coverage and held-out candidate coverage separately from intensity accuracy. No synthetic
target can repair absent candidate support by itself.

**Census gate:** proceed only after identity checks, class-definition spot checks and adequate
evaluation support. Report groups with fewer than 20 held-out molecules descriptively, with
counts; do not make individual-class promotion claims from them. If virtually all scaffolds
are singletons, scaffold inverse-frequency weighting has little discrimination: prioritize
the class arm and novelty analysis instead.

## 2. Separate balancing from augmentation

**A0 is uniform per-spectrum shuffling, without replacement within an epoch, at fixed train
and evaluation batch size 512 and gradient accumulation 1.** Both the
[NIST i1 config](../../../config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml)
and [MSGYM control](../../../config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_k1_ema999_fourier.yml)
disable dynamic, group and simple-group sampling; other optional samplers default off in the
template. Both use `train_sample_weight: none` and `eval_sample_weight: none`.
The [runner](../../../src/fragnnet/runner.py) uses `RandomSampler(ds, False)` for this training
path and sequential sampling for evaluation.

A1 samples training molecules uniformly, then one observed acquisition key uniformly within
the molecule, then a real spectrum within that key. This changes both molecule exposure and
within-molecule condition exposure; report both instead of attributing its entire effect to
spectrum multiplicity.

**New acquisition key required:** `(molecule_identity, adduct, instrument, fragmentation_mode,
recorded_CE_signature)`. The CE signature retains source units/types, single versus stepped
or ramped acquisition, all recorded values and explicit missingness; never convert NCE/eV
or collapse different CE schedules to a mean for grouping. Preserve distinct instrument models
where available, including ORB versus ORB_F. The existing dataset `group_id` covers compound
plus precursor context, without instrument, fragmentation mode or CE
([definition](../../../preproc_scripts/README.md)); the existing `group` sampler does NOT implement
this key. Construct a separate training-only key for A1–A3 and the replicate operator; retain
the original `group_id` unchanged. Unknown-condition records remain sampleable in A1–A3 under
separate per-spectrum keys, but are ineligible for replicate mixing. Replicate eligibility
additionally requires matching available acquisition/resolution provenance; a shared key alone
is not proof of an independent repeat.

For rare-scaffold sampling A2, assign molecule weight
`w(m) = min(4, max(1, sqrt(20 / n(scaffold(m)))))`, where `n` counts distinct training
molecules. For the class alternative A3, use the maximum weight over the molecule's matched
classes, capped at 4; do not add weights across labels. Empty scaffolds and unmatched classes
receive weight 1 in their respective arms.

Draw molecules from `q(m) = 0.5 / M + 0.5 * w(m) / sum(w)`, with replacement, then draw
acquisition group and spectrum as for A1. These constants are proposed pilot choices, not
measured optima. The uniform mixture preserves common-molecule exposure and limits extreme
weights. Log realized molecule/condition exposure, duplicates and effective sample size.
Do not add inverse-frequency loss weighting on top of the sampler.

Keep total spectrum draws, fixed batch size 512, accumulation 1, optimizer updates,
learning-rate schedule in update units, validation cadence, final-batch handling and checkpoint
selection matched across arms. Dynamic batching is disabled in both controls and stays disabled
in this pilot. Log actual draw and step counts to verify the comparison.

## 3. Add plausible spectral variation only after the sampling screen

For forward spectrum prediction the spectrum is the target. Random target corruption can
change the learning objective; it is not automatically a label-preserving augmentation.

Known NIST baseline from the [reliability plan](masster_intensity_reliability_plan.md):
about **1,900 strict training pairs** under the earlier molecule/NCE/eV key (user-reported
2026-09-27, not independently rerun in that plan), and **81 of 11,654 same-molecule/same-NCE
groups** mixing instruments in the later instrument audit. These have different definitions
and denominators; neither is the count of eligible rare-chemistry replicates under our new key.
Start the skip/eligibility gate from those measurements, then recount on the filtered training
subset with the explicit acquisition key above. Do not assume that all 1,900 pairs qualify or
that few mixed-instrument groups justify omitting instrument identity.

1. **Real replicate views first.** For oversampled rare molecules, cycle or randomly select
   distinct measured training replicates at identical standardized structure, adduct,
   instrument, fragmentation mode and recorded acquisition settings. Match CE units and
   values, and available resolution metadata. Keep uncertain or missing-condition groups
   ineligible. A1–A3 already expose real records; evaluate an explicit replicate-view arm
   only if its policy changes exposure beyond those controls.
2. **Replicate-calibrated intensity variation, conditional pilot.** Estimate peak-presence
   and intensity variation from eligible training replicate pairs, stratified by acquisition
   and intensity. A concrete first synthetic operator uses two aligned replicate spectra
   `y1`, `y2` from the same new acquisition key and forms `y' = lambda*y1 + (1-lambda)*y2`, with
   `lambda` uniform on [0,1], on their tolerance-aligned peak union. Use fixed representative
   masses and the existing normalization before the recipe's target transform. Mix only
   replicates of the same molecule and condition; preserve the original target on half of
   eligible draws. This operator smooths replicate variability and may hurt sharp peaks:
   test that possibility, do not describe it as a new physical measurement.
3. **Gate synthetic views.** Split eligible training replicate groups by molecule into
   development and confirmation sets. Compare synthetic-to-real versus real-to-real cosine,
   peak count, dominant-peak identity, intensity distribution and candidate-explained
   intensity. Freeze acceptance envelopes from the development pairs before confirmation.
   Skip the arm if rare molecules lack suitable replicates or the confirmation check fails.
   Do not transfer a noise model from common chemistry without a separate validity test.
4. **CE interpolation stays deferred.** It would need same-molecule, same-instrument CE
   ladders and a training-only withheld-intermediate-energy reconstruction check. Never
   jitter CE while retaining an unchanged target, pool NCE and eV, or extrapolate beyond
   measured conditions. Missing-CE samples stay under the baseline's existing policy.

Arbitrary bond/atom deletion, unchanged spectra assigned to modified molecules, random m/z
shifts and peak deletion are not justified as label-preserving operators. Randomized SMILES
are not extra chemistry for a graph model. The molecular-network branch below explores
structured substitutions and aligned donor combinations as hypotheses to validate. External
measured analogues remain outside the benchmark-subset scope. The cm0–2 cross-molecule training
integration, configs, launcher, manifest preparation and pilot tests were **removed/reverted
at the user's request on 2026-09-30**, as recorded in the
[cross-molecule plan](masster_cross_molecule_supervision_gate.md). Census code and
findings remain; this exploration does not restore the removed training implementation.

## Molecular networking exploration added 2026-10-10

An atom-mapped edit gives a testable correspondence hypothesis. The current preferred design
is the edit-effect pilot below; the broader transfer options remain conditional alternatives.
Its usefulness for pseudo-label generation remains unmeasured.

### Refined pilot using measured edit effects

**Hypothesis:** measured examples of the same local edit can teach how relative fragment
intensities change, allowing a rare molecule's measured spectrum to anchor supervision for
nearby, unmeasured structures. The evidence comes from experimental training spectra. The
student model's predictions do not manufacture or approve their own labels.

**Expected benefit:** vary molecular context around a rare motif while retaining supported
fragmentation relationships. This could weaken a shortcut from one training molecule's identity
to its spectrum. Peripheral substitutions normally preserve its Murcko scaffold: this pilot
expands within-scaffold context, not automatically the number of represented scaffolds. Report
rare-class, seen-scaffold and unseen-scaffold results separately. Recombining rare motifs onto
new cores is a more ambitious extension requiring independent cross-core evidence.

#### Learn a change rather than copy a spectrum

For one edit family `e`, collect measured pairs `A -> A'` across several molecular families,
with compatible acquisition settings. Keep an independent rare parent `R` with its measured
spectrum and construct a valid edited structure `R'`. Predict the effect of `e` on `R`, using
the measured edit pairs, then transport `R`'s observed fragments into `R'`.

This forms a proposed analogy: `A -> A'` supplies the edit effect and `R -> R'` is its transfer
test. Calibration uses cases where `R'` was actually measured but its spectra are hidden from
fitting and donor selection. Hold out all identities in each confirmation family from edit
fitting; R's spectra remain available only as explicitly designated inference-time references,
and R' never serves as a reference. Report direct-parent references separately from broader
neighbour retrieval so their easier task is visible.

Molecular squares, when available, strengthen this test: two edits connect four molecules;
hide one corner and compare predictions through the two measured neighbouring corners. Use
complete measured squares to detect context-dependent edit effects rather than enforce cycle
consistency as truth. Such squares test nonadditivity in
[molecular property research](https://link.springer.com/article/10.1186/s13321-021-00525-z);
using them for fragmentation is our proposed diagnostic, not a result of that paper. Do not
require squares if the training network has too few; repeated matched pairs across families
are the minimum design. Never propagate through previously generated spectra in this pilot.

Start with a simple edit/context table and robust interval estimates, not a new graph network.
Separate edit site, attachment environment, affected-fragment role, instrument and CE. Do not
pool contexts merely to obtain counts. A learned correction is a later option if the table
has useful signal but insufficient context resolution.

#### Supervise uncertain relative intensities

At the observable peak-group level, both fragment groups `j` and `k` must be present with
positive measured intensity and resolved mappings in **both donor-pair endpoints A and A'
and the rare parent R**. They must also map to two distinct candidate groups in R'; no measured
R' intensity may enter label construction. Define `r(A) = log(I_j(A) / I_k(A))` and learn the
distribution of `delta_e = r(A') - r(A)` within supported contexts. This avoids treating a
change in total spectral normalization as the change of a single fragment. The first operator
does not infer ratios for absent/undetected peaks; report this censoring and the lost coverage.

For `R'`, the label is an interval `[l, u]` for `r(R) + delta_e`, calibrated on separate
molecules. A proposed auxiliary penalty for predicted grouped probabilities is

`L_pair = c * max(0, l - log(p_j/p_k), log(p_j/p_k) - u)^2`.

Here `c` is a frozen confidence weight determined from calibration, not the student's agreement
with the label. Use stable grouped log probabilities in a future implementation; aggregate all
pathways sharing an observable peak before forming ratios. Cap the number of pairs per parent
and average within parent so a dense fragment set cannot dominate. Pick a bounded set of pairs
by donor evidence and mapping before seeing any hidden target spectrum; include pairs crossing
edited and conserved regions when supported, not only trivial conserved fragments.

Illustrative arithmetic, not data: if a parent's two fragment intensities have ratio 4 and the
edit changes that ratio by a factor between 0.5 and 0.8, supervise the analogue's ratio within
2–3.2. Do not insist on ratio 4, invent exact intensities, or suppress unmapped peaks.

This objective constrains ratios only. It cannot recover total intensity assigned to the mapped
subset, new fragment channels, or missing candidate support. Shared parameters can still change
other predictions, so full-spectrum evaluation remains essential. Existing
`conditional_in_support_cross_entropy` in [loss.py](../../../src/fragnnet/model/loss.py) rescales
the measured target contribution; it does not implement this masked ratio objective. Passing
partial targets through that existing loss would not supply the intended semantics.

#### Choose edits for reliability and useful diversity

| Edit category | Pilot decision | Reason to test or defer |
| --- | --- | --- |
| One peripheral substitution with repeated measured analogues | First candidate, if the census supports it | Bounded atom map; can vary context around a rare motif |
| Edit touching an inferred charge site or conjugated fragmentation region | Separate high-change stratum | Small structural distance does not guarantee a small spectral change |
| Ring opening, scaffold replacement or several simultaneous edits | Deferred | Larger mapping and fragmentation uncertainty |
| Isomeric edit with no precursor mass change | Separate later test | Precursor-delta matching cannot identify changed pathways |

Rank edit families using training-only support, mapped outcome coverage, calibration error and
unique parent families. Select ONE family initially; do not assume a named chemistry is safe.
Require adequate held-out predictive precision and interval width together: an interval covering
every possible ratio is not useful. Start with at most two accepted one-edit analogues per rare
parent; this is a proposed cost cap. Deduplicate products and cap each parent's total weight.
Count real training parents separately from synthetic structures; do not reclassify a rare
class as common because of its own generated descendants.

Require generated contexts to fall inside the measured rule's supported context range. If a
rare motif changes the edit response and there are no calibration examples for that context,
abstain. Donor count, inverse-frequency weighting and student uncertainty cannot establish
chemical validity. Several consistent references can narrow uncertainty only when calibration
shows that doing so improves reliability; shared family/parent evidence is correlated.

#### Minimal experiment and decisions

1. **Census:** report rare-parent counts with valid measured edit neighbours, independent edit
   families, matching conditions, two-outcome maps and available molecular squares. Keep
   unmappable/isolated rare parents in the denominator. Apply the preregistered coverage gate
   below before fitting any edit effects. No pseudo-spectra are needed yet.
2. **Hidden-analogue test:** fit the edit table on development families, freeze confidence and
   interval thresholds on calibration families, then reveal confirmation targets once. Compare
   transported parent ratios unchanged, edit-adjusted ratios, context-matched shuffled edit
   effects and donor-label-ablated predictions. Use the SAME accepted targets and mapped pairs
   across controls; report interval width, coverage and prediction error, plus abstention.
   Compare edit effects against an out-of-family forward predictor when available. Without
   that comparator this is a feasibility result, not evidence of additional model information.
3. **Training pilot, only if supported:** compare rare sampling, sampling measured network
   neighbours, and sampling plus edit-based partial labels. Include a synthetic arm with the
   same structures/maps but unchanged donor ratios to isolate learning edit effects. Keep the
   real-data recipe fixed and match updates and compute accounting; report real versus synthetic
   exposure explicitly. Begin with auxiliary weights `{0, 0.05, 0.1}` after separately averaging
   real and synthetic losses; these are pilot choices, not calibrated equivalents of label trust.
   Select using validation; no synthetic examples enter validation or test.
4. **Confirmation:** carry forward the existing seed and rare-cohort gates below. A reconstruction
   gain alone does not promote augmentation. Require improved real held-out rare molecules and
   no material common-cohort loss. Label any benefit confined to same-scaffold analogues accurately.

#### Preregistered mapped coverage gate

The [structural-alignment audit](../../findings/masster/masster_structural_alignment_audit_2026_09_29.md)
retained **38.00%** of measured query intensity with formula matching but only **0.85%** with
unambiguous connectivity on its unfamiliar-query cohort. Edit mapping is unmeasured and is
not mathematically bounded by those numbers: it changes the correspondence rule and may
change the cohort. Requiring two observable outcomes can reduce usable support further.

Freeze the following proposed engineering thresholds now, before the census or any fit.
Use a development-only, structure-selected rare-parent cohort fixed before spectral inspection;
reserve calibration and confirmation targets for their later roles. Evaluate each dataset and
edit family independently. A family proceeds only if:

- Mean usable mapped intensity is **at least 10%** across ALL cohort parent molecules. Sum
  intensity over unique mapped groups participating in eligible ratios, never once per pair.
  Divide by all measured non-precursor intensity, including unmapped peaks; do not normalize
  only over explained candidates. Average conditions within a molecule, then molecules equally.
  Give unmappable parents and spectra with no non-precursor intensity zero coverage.
- **At least 20% of cohort parents, and at least 50 distinct parents**, supply two distinct
  mapped peaks each contributing at least 1% of measured non-precursor intensity. Require that
  same two-peak support in both measured donor endpoints and R. At least one peak in an accepted
  ratio must retain the edited region; pairs of trivially conserved peaks alone cannot pass.
- Those usable parents span **at least five nonempty 2D Murcko scaffolds**; report acyclic
  parents separately rather than count the empty scaffold as a family. A smaller cohort may
  support a descriptive within-family study, but does not pass this transfer pilot.

Exclude precursor-matched peaks with the existing dataset's instrument-tolerance rule; record
the exact rule and fixed condition selection in the census manifest. No top-peak-only denominator
or filtering to successful alignments is allowed. These cutoffs define minimum pilot utility,
not measured feasibility or a statistical power guarantee. Failure stops this family before
fitting; do not lower thresholds after seeing the census. Carry the fixed mapper and thresholds
into confirmation, reveal target intensities only for scoring, and report missing target peaks
as failures rather than dropping their ratio predictions. Recheck coverage on the entire
confirmation cohort; development coverage alone does not establish transfer coverage.

Freeze the donor library and pseudo-label manifest before student fitting; store parent and edit
provenance, atom maps, grouped outcomes, intervals and abstentions. Do not refresh labels from
student predictions during this pilot. For any fold-based teacher/comparator, exclude confirmation
target identities and their spectra from its own training and retrieval pool as well.

If mass maps pass but intensity ratios fail, retain the maps only as diagnostics: peak-presence
augmentation needs its own measured gate. If edit effects predict held-out analogues but synthetic
training adds nothing, test retrieval-conditioned prediction as a separate experiment, motivated
by [MARASON](https://proceedings.mlr.press/v267/wang25dg.html). Neither outcome justifies restoring
the stopped formula-alignment or cm0–2 training recipes unchanged.

### Evidence and the new hypothesis

- [ModiFinder](https://pmc.ncbi.nlm.nih.gov/articles/PMC11540723/) uses shifted and unshifted
  peaks between two measured spectra to localize structural modifications. It supports an
  alignment intuition, not accurate intensity generation when the target spectrum is absent.
- [GNPS2 transitive-alignment documentation](https://wang-bioinformatics-lab.github.io/GNPS2_Documentation/transitive/)
  describes using network topology to connect indirectly related molecules when pairwise
  modified-cosine alignment misses multiple modifications. This is network construction,
  not evidence that arbitrary network paths can synthesize spectra.
- [MARASON](https://proceedings.mlr.press/v267/wang25dg.html) is a direct precedent for
  retrieval-conditioned forward spectrum prediction using neural graph matching. It supports
  testing learned donor use; it does not validate our proposed augmentation recipe or its
  expected gain on the current NIST20/MSGYM splits.
- Our [cross-molecule census](../../findings/masster/masster_cross_molecule_supervision_gate_2026_09_29.md)
  found soft measured relationships; strongest-outcome agreement was weak. Its training pilot
  was removed at user request, never measured as a training null.
- [Formula-aligned corrections](../../findings/masster/masster_reference_information_gate_2026_09_29.md)
  did not improve frozen i1, and [exact structural matching](../../findings/masster/masster_structural_alignment_audit_2026_09_29.md)
  retained little useful measured intensity. That audit explicitly left mass-shifted analogies
  untested. The new hypothesis is edit-conditioned correspondence across changed formulas,
  rather than another exact-match or naive intensity-copying experiment.

### How augmentation could work

1. Build a separate training-only structure network for each dataset. Nodes are standardized
   molecules with their measured conditions; edges are atom-mapped matched molecular pairs
   with one localized substitution. Retrieve by structure first. Measured training spectra
   can characterize edit reliability, but a target's hidden spectrum must never select its
   donors, edges, shifts or acceptance confidence.
2. Catalogue edits observed in training pairs and their attachment environments. Prioritize
   edits around rare training classes, preserving the defining rare motif. An alkyl homologue
   or O-methylation is a candidate edit family to test, not a guaranteed safe transformation.
   Record atom maps, element/formula balance, charge/adduct and valid molecular graphs.
3. For donor A and proposed analogue B, map candidate fragment explanations through the edit.
   A fragment excluding the changed site may keep its mass; one retaining the changed group
   may change formula and mass; an incompatible fragment remains unknown. Recompute the ion
   formula, charge and allowed H changes on B. Do not simply shift every peak by precursor
   delta. For an otherwise conserved singly charged O-methylated fragment, the net CH2 edit
   predicts about +14.01565 Da; a fragment excluding that site predicts no shift. Rearrangements
   and changed charge retention can invalidate either correspondence.
4. Treat intensities as uncertain. First evaluate donor intensity copying as a baseline;
   then, if data support it, learn edit/context/CE-dependent intensity ratios or presence
   probabilities using disjoint training pairs. Retain ambiguity over fragment explanations,
   or abstain when their mapped masses disagree. Matching a formula is not proof of mechanism.
5. Several donors may be transported into the candidate coordinates of the SAME B and then
   combined with confidence weights. This defines a possible pseudo-target for B. Directly
   averaging unaligned spectra of A and C does not define a target for a unique intermediate
   molecular graph. High network degree or donor agreement alone is not independent evidence;
   donors can share the same bias.

Start with partial supervision: only mapped peaks or mapped within-spectrum intensity ratios
are supported. Unmapped peaks are unknown, not zero. A possible auxiliary loss compares
conditional distributions over the mapped candidate subset, confidence-weighted and disabled
when fewer than two distinct candidates are mapped. It cannot supervise total mapped mass or
unknown channels. Do not feed a renormalized partial spectrum into the existing full-spectrum
loss as if complete. Full pseudo-spectra require a separate completeness gate or an explicitly
audited teacher completion, with a teacher-only control.

### First experiment before any synthetic training

Use existing measured molecules as simulated unmeasured analogues. Partition original training
molecular identities into disjoint donor/fit, calibration and confirmation sets; keep all spectra
of each identity together. Freeze separate scaffold-family holdouts to test transfer beyond
near-duplicate series. If network components are too large for component-disjoint splits,
report shared donor dependence rather than claiming component independence.

Hide every target spectrum from retrieval, parameter fitting and target selection. Reconstruct
it from its structure, matched-condition donors and the learned edit rule, then reveal its
measured spectrum for scoring only. A forward-model comparator must also exclude these hidden
identities from training; an existing model fitted on them is not a valid comparator.

Compare no-shift donor copying, whole-spectrum precursor-delta shifting, atom-mapped transfer,
mass/acquisition-matched random donors, label-only shuffled donor intensities and a structure-only
predictor. A learned transfer model also needs a donor-label-ablated control. Keep mapped support
fixed for the label-shuffle comparison. If costs preclude a held-out forward predictor initially,
call this a correspondence screen only, not evidence of incremental model information.

Report mapped-peak precision and measured-intensity coverage jointly, peak/intensity errors,
full-spectrum cosine when full predictions exist, rare-cohort availability, abstention and
per-edit/CE/scaffold outcomes. Do not score only successful alignments or only precursor peaks.
Require positive incremental donor-label value against matched controls on independent
confirmation molecules; keep the mapped-coverage thresholds above fixed, and freeze prediction-precision thresholds
from calibration before confirmation. Passing a close-analogue check supports only that scope.

Only after this gate: enumerate a bounded number of valid analogues from rare training nodes,
using edit families validated above; exclude validation/test identities and enforce the
benchmark's structural split constraints. Freeze generation before exclusion checks so held-out
structures do not become a design target. If augmented structures violate the benchmark regime,
report a separate expanded-data experiment, not a like-for-like benchmark result. Audit teacher
training provenance too. Keep NIST and MSGYM donor pools separate.

Compare rare sampling alone, network-guided sampling of REAL neighbours, and the same training
budget plus partial synthetic supervision. Use a small synthetic auxiliary weight selected on
validation, and retain real measured targets as the main objective. The existing three-seed,
rare-cohort and overall-performance gates still apply. A retrieval-conditioned architecture is
a separate alternative if donors help reconstruction but hard pseudo-labels are unreliable.

**Next concrete check:** count matched-edit edges incident to rare training molecules, their
condition compatibility and scaffold diversity. Select one adequately supported edit family
for the hidden-spectrum screen. No benefit can be assumed for isolated rare classes with no
relevant donors. This branch is proposed; no network, pseudo-labels or trainer has been built.

## 4. Controlled experiment sequence

| Arm | Change | Main comparison |
| --- | --- | --- |
| A0 | Uniform per-spectrum shuffle; fixed batch 512; optional samplers off | Reference |
| A1 | Uniform molecule and acquisition-group sampling | A1 vs A0: spectrum multiplicity |
| A2 | Capped rare-scaffold mixture; NIST first, MSGYM conditional on census | A2 vs A1: scaffold exposure |
| A3 | Capped rare-class mixture | A3 vs A1: class exposure |
| A4, conditional | Winning sampling policy plus validated replicate interpolation | A4 vs its exact sampling parent: synthetic variation |

NIST starts from `i1`, the workstream's designated control; a confirmed winner is then checked
on `i4`. MassSpecGym starts from the k1/EMA/Fourier screen control; a winner is subsequently
checked on `pow075` if retrieval remains a target. Resolve exact configs and freeze them at
implementation time using the linked workstream plans. Do not compare an augmented new
recipe only against an older historical checkpoint.

Screen A0–A3 on NIST and A0/A1/A3 on MSGYM with one matched seed; add MSGYM A2 only if its census
gate passes. Replicate A0, A1 and the selected rare-sampling
winner with two additional matched seeds before promotion. A4 is optional and follows its
own validity gate; confirm it with the same three seeds if it screens positively. Reserve
test evaluation until the recipe and retrieval calibration are frozen. No job submission is
authorized by this plan.

## 5. Evaluation and acceptance

Use untouched validation and test spectra, with no weighting or augmentation at evaluation.
Freeze rare cohorts from training counts before model comparison; do not define them by the
new model's errors. Previously inspected validation floors are diagnostic, not independent
confirmation. Report unseen scaffolds/classes separately: repeating known molecules cannot
directly supply examples of an unseen class.

- Primary endpoint: molecule-averaged cosine within the predeclared rare cohort, with both
  scaffold-rare and class-rare cohorts reported. Name the arm's target cohort before fitting.
  For NIST use instrument-tolerance `cos_hun_inst_tol`; also retain the workstream's 0.01-Da
  historical screen metric. MassSpecGym uses the established `cos_sim_0.01` protocol.
- Report overall spectrum and molecule means, common-cohort mean, class macro mean,
  bottom-decile score, fraction below the predeclared failure threshold, dominant-peak
  accuracy and candidate coverage. Keep overlapping class metrics explicitly multi-label.
- Stratify by instrument/CE where supported. For MassSpecGym also report formula-track
  retrieval hit@1 and its rare subset with the same candidates and validation-calibrated
  scoring rule; do not choose that rule on test.
- Pair model comparisons by molecule, cluster-bootstrap molecules rather than spectra,
  and report each seed's difference and uncertainty. For scaffold transfer additionally
  bootstrap scaffold clusters where there are enough clusters.

Proposed confirmation gate, to freeze before runs: rare-cohort gain at least +0.01 absolute
cosine, positive in at least two of three matched seeds, with the paired 95% interval excluding
zero. Overall and common-cohort lower confidence bounds must exceed -0.005 cosine; candidate
coverage must not drop by more than 0.005. These are new pilot tolerances, not existing measured
rare-cohort noise levels. If intervals cannot resolve them, label the outcome inconclusive.
Preserve the NIST workstream's historical +0.005 single-seed mean screen as a separate read;
an intervention passing only the rare-cohort gate is a targeted tradeoff, not an overall winner.

Stop augmentation if gains come only from more updates, memorized training molecules or
acquisition confounding. If A2/A3 do not beat A1, retain the simpler balancing result and
close the rarity-weighting hypothesis for this pilot. If A4 does not beat its parent,
retain measured spectra. Report each dataset independently.

## Implementation handoff and current state

- Completed: reviewed workstream constraints, current sampler semantics and related
  reliability work; specified the census, arms, validity checks and evaluation gates. Refined
  the network branch to measured edit effects, family-held-out tests and partial ratio intervals.
- Not started: data census, sampler changes, configs, augmentation or training.
- First concrete check: training-only scaffold/class census, matched-edit coverage and replicate
  eligibility. For the network branch, select one supported edit family for hidden-analogue tests.
- Future code belongs in the dataset/sampler layer and focused analysis scripts; expose
  default-off config keys only after reading the Python/ML rules. Tests must verify training-only
  counts, split isolation, reproducible draws, actual update budgets and unmodified evaluation.
- Follow the [session workflow](../../workflow/session.md) and [artifact rules](../../rules/artifacts.md)
  for provenance. Record dataset/split identities, SMARTS panel, scaffold policy, counts,
  resolved config, seed and sampler exposure alongside future experiment results.
- Documentation validation: local links and whitespace check; model tests are unnecessary
  for this planning-only change.
- Citation recheck (2026-10-10): eScholarship returned a bot challenge and the OSTI PDF fetch
  timed out; replaced that supporting citation with successfully read official GNPS2 docs.
  The Springer nonadditivity article was successfully retrieved through the web tool, including
  its double-transformation-cycle method; retained as molecular-property evidence only.
