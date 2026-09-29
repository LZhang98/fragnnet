# Aligned reference intensities do not improve frozen i1 in the bounded gate

> **Status:** CLOSED NULL · **Updated:** 2026-09-29
> Two-fold validation-query diagnostic complete. Ion-formula corrections select zero residual;
> loss-formula corrections degrade held-out cosine. Do not promote this recipe to full training.

## Result and decision

The [coverage advantage](masster_candidate_correspondence_2026_09_29.md) did not translate into
incremental predictive value in this correction recipe. This is a negative result for a small
formula-aligned reference correction, not proof that references contain no useful information.
No candidate competition loss or full MASSter training run was launched.

Production 0.01-Da spectral cosine, averaged once per held-out query identity:

| Correction | All 239 queries | 119 unfamiliar queries |
| --- | ---: | ---: |
| Frozen i1 | 0.63853 | 0.54237 |
| Query context only | 0.63660 | 0.53881 |
| Ion-aligned reference intensities | 0.63660 | 0.53881 |
| Loss structure only | 0.63210 | 0.53996 |
| Loss-aligned reference intensities | 0.62868 | 0.53497 |
| Loss shuffled intensities | 0.62949 | 0.53291 |
| Loss random-reference intensities | 0.63871 | 0.54066 |

All four ion reference arms (structure, aligned, shuffled, random) selected **epoch zero in both
folds**, so each exactly equals its frozen context baseline. Their zero paired differences and
zero bootstrap intervals describe the selected zero estimator; they do not tightly bound the
amount of chemical information in a reference. Optimization ran: ion-aligned fit cross entropy
fell from 3.2055 to 3.1907 and 3.2898 to 3.2746 across ten epochs, while inner cosine never beat
epoch zero. Context selected epoch 11 in fold 0 and zero in fold 1. Its small apparent inner gain
did not transfer to the outer fold.

For loss alignment, paired cosine changes and 95% query-bootstrap intervals are:

| Comparison | All queries | Unfamiliar queries |
| --- | ---: | ---: |
| Aligned − raw i1 | −0.00985 [−0.01906, −0.00184] | −0.00741 [−0.01645, +0.00023] |
| Aligned − context | −0.00792 [−0.01664, −0.00090] | −0.00384 [−0.01157, +0.00092] |
| Aligned − structure | −0.00341 [−0.01080, +0.00260] | −0.00499 [−0.01794, +0.00543] |
| Aligned − shuffled | −0.00080 [−0.00626, +0.00348] | +0.00206 [−0.00099, +0.00484] |
| Aligned − random | −0.01003 [−0.01878, −0.00298] | −0.00569 [−0.01348, −0.00025] |

Aligned-minus-structure is negative in both outer folds, overall (−0.00055, −0.00625) and on
unfamiliar molecules (−0.00079, −0.00912). The unfamiliar advantage over shuffled is positive in
both folds but has an interval crossing zero and does not beat structure or i1. The preregistered
promotion criterion therefore fails.

Base-peak accuracy changes from 44.77% for i1 to 46.44% for both context and aligned loss; this
is not an incremental alignment gain. On unfamiliar queries it changes from 35.29% to 37.82%
for both. Strong-outcome inversion error worsens from 0.30348 to 0.30771 overall and from 0.36659
to 0.37102 on unfamiliar queries. This auxiliary ranking metric compares unequal target groups
with at least 1% of total measured intensity; it is not the earlier ranking-decomposition metric.

## Cohort and leakage boundaries

The previous coverage sample used training molecules. This gate instead selected validation
queries without examining their peak intensities and used only training references. The selector
samples 120 identities in each novelty stratum before retrieval; 119 unfamiliar and 120 other
queries have feasible matched conditions. One unfamiliar query has no eligible reference/control
condition and remains marked unselected in `cohort.csv`; the reported score is conditional on
reference availability, 239/240. No query is filtered by peak strength or number of correspondences.

Novelty is maximum Morgan radius-2, 2,048-bit Tanimoto against all **12,668 eligible training
identities**, not just the mass/acquisition-matched reference pool. Of 4,100 eligible validation
identities at moderate NCE, 526 meet the <=0.4 unfamiliar threshold. Train/validation first-block
InChIKey identities are disjoint. Reference selection excludes same identity and molecular formula.
It matches instrument, adduct ([M+H]+), fragmentation mode and exact observed NCE; the precursor
window is max(10 Da, 5%). The random reference excludes the three closest eligible donors and
matches absolute precursor distance within 2 Da.

There are 27 queries at NCE [40,60) and 212 at [60,100); most are NCE 60 (127) or 65 (78), reflecting
the label-blind preference for a feasible condition nearest 60. Instruments are OB (169) and OB_F
(70). This balanced novelty sample is not a full-validation mean, and its raw 0.63853 cosine must
not be compared directly to a full-validation W&B score.

The 27-query lower-NCE loss slice beats its structure and shuffled corrections by +0.00522 and
+0.00805 respectively. However, aligned loss remains **−0.00714 below raw i1**, and the shuffled
comparison changes sign between folds. This small secondary slice does not rescue the gate.

## Method frozen before fitting

See the [protocol](../../plans/masster/masster_candidate_transfer_plan.md#gate-b-frozen-protocol-2026-09-29-before-correction-results)
for selection and fitting details. Each query appears once, in one of two outer folds stratified
by novelty. Each training fold reserves 15% of identities (18 queries) for early stopping. Query
feature normalization and gradients use only the fit identities; outer labels enter scores only.
Inner-fold cosine selects epochs, including epoch zero. This is validation cross-fitting, not an
untouched test evaluation or a deployable model trained only on the original training split.

A width-32 residual MLP first corrects frozen i1 log probabilities using query chemistry/context.
Each reference arm adds an equally sized residual to that same frozen context baseline. Reference
structure features include similarity, masses and ambiguity counts. The aligned arm also sees
measured reference group intensity and its floored log; structure-only zeroes those label slots.
The shuffled arm permutes labels among the same donor's matched groups, preserving values and
zeros. A derangement changes endpoints when there is more than one; identical zero values can
remain unchanged. Eleven ion and 41 loss references have no label values changed by shuffling,
so that control is weaker on those queries. Random references have different coverage, explicitly
reported; shuffled references supply the same-coverage control.

Formula matching identifies observable compositions, not shared structures or mechanisms. Ion
and loss views are fitted independently. Weak and absent assigned peaks are retained. Unmatched
groups receive zero reference logit correction but stay in the full softmax and evaluation.
Cross entropy fits the distribution over uniquely assigned covered target groups. Evaluation uses
all preprocessed target peaks, including uncovered peaks, and all in-range predicted groups,
including precursor and unmatched groups; native mass dtypes preserve production bin boundaries.

Coverage remains positive in this new cohort: 22/119 unfamiliar queries have two strong ion matches
with neighbors versus 8/119 random controls; for loss matches, 19/119 versus 4/119. Mean matched
query intensity is 34.59% versus 17.94% for ion matches and 30.67% versus 10.71% for losses. This is
exactly why coverage alone was not a sufficient acceptance criterion.

## Execution, verification and limitations

- Frozen checkpoint: `saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt`.
  CPU inference: 703 unique spectra, 234.0 seconds; maximum formula/mass error 9.13e-11 Da.
  CPU correction fitting: about 20 seconds. No cluster job or main-model optimization.
- Sources: [selector](../../../scripts/debug/prepare_masster_reference_gate.py),
  [annotated extraction](../../../scripts/debug/debug_masster_candidate_correspondence.py),
  [correction](../../../scripts/debug/debug_masster_reference_information_gate.py),
  [config](../../../config/analysis/masster_reference_information_gate.yaml).
- Artifacts: `output/reference_information_gate_2026_09_29/` contains the cohort/pair manifests,
  selection provenance, extraction cache/provenance, `audit.json`, and `correction/` with outer
  per-query scores, summaries, selected epochs, fit histories, fitted states and source snapshots.
  Provenance pins source/config/checkpoint/data/split hashes, Git revision/dirty diff, seed,
  command, versions, device and completion time. The selector's executed source snapshot precedes
  a subsequent lint-only edit. A trace-logging rerun reproduced all outer scores exactly; the
  initial artifacts remain under `correction_initial/`.
- Sixteen focused tests pass, covering production scoring, label-independent mapping/features,
  shuffled value preservation, held-label isolation, unmatched residuals, reference exclusion,
  existing formula reconstruction and NCE selection. Ruff passes. Artifact audit confirms split,
  fold, novelty and NCE constraints and formula correspondences. All 239 grouped raw cosine scores
  match the production bin/cosine helpers within 2.02e-7.
- Reference reuse is limited: 234 distinct neighbor molecules and 235 random-reference molecules;
  each serves at most two queries. Intervals resample query identities and condition on this
  fixed reference library and fitted folds; they do not include training-procedure/library uncertainty.
- The 18-query inner sets and small correction class limit sensitivity. The CE fitting target and
  binned cosine selection criterion differ. Formula correspondence can join different mechanisms.
  These limits preclude a universal claim that useful alignment is impossible. No tuning was done
  after seeing outer results, and no larger architecture should be justified by this failed gate.

Keep the separate cm0–2 whole-spectrum relationship pilot distinct: its hypothesis is different
and remains untested here. Reopening candidate-level transfer requires a new chemical alignment
argument or independent evidence, followed by a fresh gate, rather than relabeling this result.
