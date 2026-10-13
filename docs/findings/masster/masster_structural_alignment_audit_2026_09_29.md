# Exact structural alignment loses most of the useful measured coverage

> **Status:** CLOSED NULL · **Updated:** 2026-09-29
> Training-only audit complete. Exact connectivity retains little measured intensity on unfamiliar
> queries; exact cut context retains none. An apparent residual signal fails label-specific sanity
> controls. No fresh validation correction or new training run is justified by this screen.

## Result

The user authorized the richer-alignment audit after the
[formula-reference correction null](masster_reference_information_gate_2026_09_29.md).
We implemented exact retained-connectivity and cut-environment matching and ran both on 358
fixed query/reference acquisition pairs: 110 training query identities, 532 unique spectra,
observed NCE [40,100). References and random controls retain the identities from the earlier
[fixed-reference coverage audit](masster_candidate_correspondence_2026_09_29.md).

For the **54 unfamiliar query identities**, neighbor coverage is:

| Matching rule | Mean matched groups | Mean measured query intensity retained | Acquisition pairs with two strong matched outcomes |
| --- | ---: | ---: | ---: |
| Ion formula | 200.7 | 38.00% | 27/88 |
| Formula + unambiguous retained connectivity | 22.0 | 0.85% | 0/88 |
| Above + unambiguous cut/channel context | 0.87 | 0.00% | 0/88 |

Means first average acquisitions within each query identity, then give identities equal weight.
The last column counts acquisition pairs, with >=1% measured intensity at both ends of each of
two different matched outcomes. No peak-strength condition selects matches or the cohort.
Connectivity retains only about 2.2% of the formula tier's measured coverage (0.85/38.00).

Random-reference intensity coverage is respectively 25.27%, 0.27% and zero. On the 56 other
query identities, neighbor coverage is 47.48%, 4.83% and 2.60%; two-strong-outcome support falls
from 35/91 acquisition pairs to 7/91 and 3/91. These rules therefore favor closer chemistry and
leave very little support for transferring competition on unfamiliar molecules.

## Why the initial residual signal was misleading

The predeclared residual screen **mechanically passes for connectivity**, and that result is
preserved. For each matched outcome, residual = assigned measured intensity fraction − frozen
model probability. Across 36 unfamiliar identities with defined residual cosines, connectivity
has mean residual cosine 0.3780, beating shuffled residuals by +0.2609 (95% interval
[+0.0973, +0.4192]) and formula edges subsampled to the same count by +0.3137
[+0.1326, +0.4955]. Cut context has only one defined unfamiliar identity and fails the screen.

However, that residual shuffle moves the donor's **model prediction together with its label**.
It can reward matched model errors even if the measured donor label adds nothing. In the simple
case where both assigned target intensities are zero, both residuals are negative model
probabilities; they can align perfectly without any positive measured reference evidence.
A regression test now captures this failure mode.

We therefore added explicitly **post-hoc sanity controls**, preserving the original results and
not changing the predeclared threshold retroactively:

- Keep donor predictions aligned but replace all matched donor measured intensities with zero.
  Residual cosine is **0.4628**, higher than the real-label 0.3780. Real minus zero-reference is
  **−0.0848**, 95% interval **[−0.2501, +0.0952]**.
- Shuffle only measured donor intensities while keeping donor predictions at their correct
  matched groups. Real minus label-only shuffle is **−0.0426**, with interval
  **[−0.1487, +0.0693]**; it does not establish an incremental label contribution.

Only 13/88 unfamiliar connectivity acquisition pairs have nonzero assigned measured norms on
both sides, spanning 10 query identities. A measured cosine of 0.4653 on those 10 identities is
not directly comparable with formula cosine 0.1572 on 48 identities: the evaluable subset changed.
Zero assignment also need not mean chemical impossibility; it can reflect weak/undetected peaks
or limitations of candidate/peak assignment.

The predeclared residual criterion was insufficient as an information gate. We do **not** claim
that it failed its numerical thresholds. Low measured coverage, absent two-outcome support and
the failed label-specific controls are the reasons not to proceed to a fresh correction fit.
This result constrains exact structural matching; it does not rule out approximate alignment,
mass-shifted analogies or retrieval-conditioned models with richer pathway context.

## What was implemented

The existing extractor optionally saves all candidate pathway atom sets in molecule-local
numbering, ion modes, H shifts and group membership. Graph atomic numbers, parent hydrogen
counts and chemical-bond endpoints are checked against source RDKit molecules before signatures
are constructed. The original frozen prediction/target group aggregation is retained.

The [audit](../../../scripts/debug/masster_memorization/debug_masster_structural_alignment.py) uses three nested rules:

1. Existing unique eligible ion-formula correspondence, with precursor and unresolved mass
   groups excluded as in the original audit.
2. The same formula, plus a single common canonical retained-atom connectivity signature on
   both peaks. Signatures preserve elements, formal charges and aromatic/bond types; parent H
   counts, stereochemistry and isotope annotations are omitted from this structural descriptor.
3. The same connectivity, plus one common context signature. Keep the retained graph and extend
   it to radius-one neighborhoods around both endpoints of all boundary bonds. Mark retained,
   discarded and boundary-anchor roles, preserving attachment positions within the retained
   graph. Include parent H counts, boundary bond type/ring/conjugation flags, ion mode and H shift.

These are descriptors of retained atoms and their **parent-molecule environments**, not relaxed
ionic structures, reaction barriers, proven fragmentation steps or a fragmentation DAG. A broken
aromatic ring remains a graph descriptor; the code does not sanitize it into a claimed product.

Every pathway explanation contributes to the signature sets for its measured candidate group.
Symmetry-equivalent atom sets may collapse to one graph signature. Multiple different structures
or channel/context explanations remain ambiguous and are excluded at the corresponding strict
tier. No highest-probability explanation or best matching measured label is selected. Across all
532 spectra, 313,847 groups are formula-eligible; 160,600 have one retained-connectivity signature
and 144,026 have one complete context signature. Unique within-molecule explanations do not imply
that another molecule has the same explanation.

This is an **exact refinement of formula matching**. It does not yet align different formulas,
substituted scaffolds or analogous reactions. The coverage collapse explains why simply making
formula matching stricter is insufficient for the original generalization objective.

## Controls, provenance and checks

All queries/references are training spectra. The 54 unfamiliar identities retain the original
census novelty definition: maximum Morgan Tanimoto <=0.4 against that census's donor pool. This
is not the full-training novelty definition used by the later validation correction gate.
Each molecular pair retains exact observed instrument/adduct/fragmentation-mode/NCE agreement.
No new reference search, model fitting, validation-label use, test-label use or quantum calculation.

All fixed pairs, including empty strict matches, contribute to coverage. Undefined cosines remain
missing, with denominators reported. Controls use 20 seeded residual permutations or formula-edge
subsamples of the same size. Count matching controls only edge count, not every chemical or label
selection effect. Acquisition-level paired differences are averaged within query molecular
identity before 2,000-replicate bootstrap. Confidence intervals condition on the fixed donor
library and existing fitted i1. Training residuals, where i1 already fits closely, are exploratory;
none of these cosine numbers is a validation prediction improvement.

Sources and artifacts:

- [Audit config](../../../config/analysis/masster_structural_alignment_audit.yaml),
  [extractor](../../../scripts/debug/masster_memorization/debug_masster_candidate_correspondence.py),
  [tests](../../../tests/test_debug_masster_structural_alignment.py).
- `output/structural_alignment_audit_2026_09_29/`: frozen pair manifest/selection hashes;
  annotated extraction cache; all structural signature sets and signature catalog;
  individual matches, pair results, identity-weighted summaries, ambiguity counts and provenance.
- `initial_residual_screen/` preserves the original script, results and provenance before adding
  label-specific sanity checks. `label_sanity_source.py` and its outputs record the first explicit
  post-hoc check; the controls are then incorporated in the main audit for reproducibility.
- Inference took 397.8 seconds including serialization and correspondence summaries. Initial
  structural audit took 92.6 seconds on CPU. Source/config/checkpoint/input hashes, exact commands,
  Git revision and dirty diff, versions, seeds and completion state are recorded in provenance.
- The older NCE extraction is not bit-identical: one of 532 spectra changes its candidate set;
  maximum group-probability L1 difference is 0.00181, mean 0.00000564. Assigned target intensities
  agree. Batch cohort and source snapshot differ; the cause was not isolated. Every audit arm
  uses the same **new** cache, with no mixture of older predictions. See
  `prior_inference_comparison.csv` and `inference_comparison.json`.
- Twenty-three focused tests pass, including seven structural tests and the new shared-absence
  regression. Ruff passes. Tests cover atom-renumbering invariance, open
  aromatic fragments, same-formula/different-connectivity cases, different leaving environments,
  ambiguous/channel alternatives, label-independent mapping, empty matches, count controls and
  equal query-identity weighting. No full test suite or main-model training run was required.

## Disposition

Stop this exact-matching recipe. Preserve the original formula-correction null and the residual
screen's apparent positive result alongside the sanity check explaining its limitation. A future
approximate matcher needs evidence that it restores **useful measured outcome coverage** while
preserving chemical relevance; larger matching networks alone are not justified by this audit.
The separate cm0–2 whole-spectrum pilot remains unchanged and is not evaluated here.
