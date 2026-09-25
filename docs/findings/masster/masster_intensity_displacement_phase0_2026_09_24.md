# Where does MASSter's misplaced intensity go?

> **Status:** FINDING · **Updated:** 2026-09-24
> Phase 0 of the [intensity-displacement plan](../../plans/masster/masster_intensity_displacement_plan.md),
> offline on the f0/f1 (`3bej6fq7`/`6h67zps9`) exports of the
> [target-cleaning cross-eval](masster_target_cleaning_cross_eval_2026_09_23.md); no training
> and no new inference.
>
> - **Most of it is shared by the two runs.** 72.5% of the misplaced-intensity error on val is
>   common to f0 and f1 (test 72.3%). After removing the replicate target-noise floor, the
>   reducible shared part is 67% (val) and 63% (test). Adjacent checkpoints of one run share 89%.
> - **It is not neighbour confusion.** Against a naive null, the misplaced mass looks local:
>   one-neutral-loss moves come out at 1.75x chance. Against a null that keeps the mass on the
>   real peaks and only shuffles their order, every class is within 0.90-1.08x. The model's wrong
>   base-peak winner is a real peak 94% of the time. It is one loss from the true base peak no
>   more often than a random real peak is (30.8% vs 32.5%).
> - **The structured part is below NCE 60.** There the model over-holds the heavier member of
>   nested pairs: 0.64 of that mass sits on the heavier member against 0.19 for the null at
>   NCE ≤ 20, fading to no bias above NCE 60. This is the precursor/parent fallback.
> - **A formula-only learner is complementary there.** On the precursor-versus-contender decision
>   at NCE ≤ 30, stacking it with the model lifts pairwise AUC from 0.798 to 0.829 on held-out
>   val molecules, and from 0.807 to 0.838 on test.
>
> Gate G0 passes its first condition and fails its second against the proper null. The
> class-targeted Phase 2 arms have no premise; see [Consequences](#consequences-for-the-plan).

## Method

Script: [`scripts/eval/intensity_displacement.py`](../../../scripts/eval/intensity_displacement.py),
with tests in [`tests/test_intensity_displacement.py`](../../../tests/test_intensity_displacement.py).
Driver and outputs are in `output/cross_eval_mh_ft_clean_2026_09_23/analysis/displacement/`
(local, Git-ignored):

- `run_phase0.sh` runs every stage.
- `summary.json` and `support_null/summary.json` hold every number below.
- The raw targets were copied from a session scratchpad under `/tmp` into
  `output/cross_eval_mh_ft_clean_2026_09_23/targets/`.

Everything lives in the candidate space of the
[ranking decomposition](masster_ranking_decomposition_2026_09_23.md): exact-m/z-merged
candidates, target peaks mapped to the nearest candidate within 10 ppm (200 Da floor), and here
the union of both runs' candidates. The task is unmerged `[M+H]+`/FT on
`data/split/nist20_inchikey`. The main comparison uses the raw target (52,020 val spectra), where
f0 and f1 score 0.7564 and 0.7566 against the ranking decomposition's 0.7565.

- **Shared versus run-specific.** Per spectrum, with `p_k` the L2-normalized prediction and
  `t = T_covered / |T_all|`, the gap splits exactly as `1 − cos_k = |p_k − t|²/2 + (1 − |t|²)/2`:
  displacement plus coverage. With `ē` the mean and `d` half the difference of the two runs'
  error vectors:
  - the shared part is `|ē|² − |d|²` and the run-specific part is `2|d|²`, both halved into
    cosine units;
  - the floor is replicate target noise, `|t_a − t_b|²/4`, measured on val spectra of one
    molecule at the same NCE (1,315 pairs on val, 1,293 on test). The shared part contains it,
    since both runs face the same noisy target.
- **Transport.** Excess `(q − y)+` travels to deficit `(y − q)+` for unit-L1 predictions and
  covered targets. The solver is exact assignment on 500 quantized mass units: it agreed with
  HiGHS to 0.2% on class shares at 1,000 units and ran 300 times faster. Each unit of moved mass
  is classed by the two candidates' ion formulas, assigned as exact-mass subformulas of the
  precursor ion (5e-5 Da; 0% of candidates ambiguous):
  - precursor at either end;
  - H shift (same heavy atoms);
  - one-loss (heavy atoms nested, difference in a 37-loss vocabulary, H within 2);
  - other nested;
  - sibling (not nested).

  The ground cost follows that class order, with a small m/z tie-break.
- **Controls.** Three nulls, plus a direction measure:
  - an m/z-distance ground cost, to test cost dependence;
  - a naive null that permutes the prediction over all candidates;
  - a **support-preserving null** that permutes it within the target-supported candidates and
    within the rest separately, on a 10k-spectrum val sample. It keeps "mass sits on real peaks"
    and destroys only the order among them;
  - for nested pairs, the share of moved mass whose donor is the heavier formula.
- **Precursor versus contender.** At NCE ≤ 30, the precursor against the strongest covered target
  peak below it (filtered targets, f1 and f0 exports). There are 7,543 train-sample pairs, 15,109
  val and 15,357 test. A gradient-boosted classifier on formula-only inputs (loss and precursor
  composition, loss mass and fraction, NCE, precursor m/z) is fitted on the train pairs. Its
  logit is stacked with the model's log-ratio on half the val molecules, then scored on the other
  half and on test.

## Result

### Two thirds of the displacement is shared bias

| pair, split | displacement | shared | run-specific |
| --- | ---: | ---: | ---: |
| f0 vs f1, raw val | 0.2288 | 0.1658 (72.5%) | 0.0630 (27.5%) |
| f0 vs f1, raw test | 0.2246 | 0.1624 (72.3%) | 0.0622 (27.7%) |
| f0 vs f1, filtered val | 0.2291 | 0.1662 (72.5%) | 0.0630 (27.5%) |
| f1 best vs last checkpoint, raw val | 0.2285 | 0.2035 (89.1%) | 0.0249 (10.9%) |
| f0 best vs last checkpoint, raw val | 0.2287 | 0.2024 (88.5%) | 0.0263 (11.5%) |

Coverage is 0.0147 on both splits. On the replicate subset (val) the displacement is 0.1616, the
shared part 0.1214 and the target-noise floor 0.0132. That leaves 0.108, or **67.0%** of the
displacement, as reducible shared error (test: floor 0.0140, 62.7%). Adjacent checkpoints differ
by 0.025, about 40% of the between-run 0.063, which matches the cross-eval's split of the
ensemble gain.

The shared fraction falls from 0.81 at NCE ≤ 20 to 0.68 above 60, while the shared amount rises
from 0.129 to 0.179 and the run-specific part from 0.031 to 0.083. By nearest-train Tanimoto the
displacement triples, from 0.116 (> 0.8) to 0.363 (≤ 0.4), but the shared fraction stays at
0.71-0.73 in every bin. Novel molecules raise both parts in proportion.

### The misplacement looks local only against the wrong null

Share of displaced mass by class, raw val. D is the displaced (TV) mass per spectrum.

| predictor | D | precursor | H shift | one-loss | other nested | sibling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f0 | 0.366 | 0.208 | 0.062 | 0.409 | 0.209 | 0.100 |
| mean of f0, f1 (shared) | 0.347 | 0.210 | 0.063 | 0.408 | 0.207 | 0.098 |
| f0 against f1 (disagreement) | 0.261 | 0.192 | 0.073 | 0.449 | 0.189 | 0.085 |
| naive null | 0.986 | 0.181 | 0.038 | 0.233 | 0.378 | 0.159 |
| support-preserving null (10k) | 0.751 | 0.207 | 0.065 | 0.377 | 0.231 | 0.109 |

| ratio to null | precursor | H shift | one-loss | other nested | sibling |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared, against the naive null | 1.16 | 1.67 | 1.75 | 0.55 | 0.62 |
| f0 with the m/z-distance cost, against the naive null under the same cost | 1.14 | 2.45 | 2.33 | 0.73 | 0.72 |
| **shared, against the support-preserving null** | **1.03** | **0.96** | **1.07** | **0.90** | **0.90** |

Test agrees within 0.01 in every cell. Unmatched (pruned) mass is 1.1-1.3%.

The naive null's enrichment is the structure of real spectra. Observed peaks are connected by
common losses, so any wrong order among them moves mass along one-loss edges. Once that is held
fixed, the model's misplacement has the class mix of a random reordering of the real peaks. The
disagreement between runs has the same mix, so the run-specific part is not a different kind of
error either.

Base-peak misses (the base peak is covered but not the model's top) are 41.5% of spectra. In the
10k sample, 94.0% of the model's wrong winners carry target intensity, and 39.5% are the true #2
peak. Class of the relation to the true base peak:

| wrong winner | precursor | H shift | one-loss | other nested | sibling |
| --- | ---: | ---: | ---: | ---: | ---: |
| model's top | 0.228 | 0.045 | 0.308 | 0.266 | 0.152 |
| true #2 peak | 0.207 | 0.053 | 0.370 | 0.231 | 0.140 |
| intensity-weighted random real peak | 0.187 | 0.051 | 0.325 | 0.275 | 0.162 |
| random candidate | 0.100 | 0.012 | 0.100 | 0.476 | 0.311 |

So the earlier "one neutral loss away" readings (31% here, and 66% within ±60 Da on FraGNNet)
describe the spectra more than the model. The loss list is bidirectional:
the model's top sits one H₂O above the true base in 615 misses and one H₂O below in 423, followed
by ±HCN, ±H, ±C₂H₂ and ±CO.

### The one directional component: the fallback below NCE 60

The kind of displacement changes with collision energy. Novelty changes only its amount:

| NCE | spectra | D (shared) | precursor | one-loss | mass on the heavier nested member: model / support null |
| --- | ---: | ---: | ---: | ---: | --- |
| ≤ 20 | 9,304 | 0.273 | 0.67 | 0.15 | 0.64 / 0.19 |
| 20-30 | 5,814 | 0.325 | 0.43 | 0.26 | 0.57 / 0.26 |
| 30-40 | 6,204 | 0.347 | 0.29 | 0.34 | 0.55 / 0.30 |
| 40-60 | 9,916 | 0.358 | 0.14 | 0.43 | 0.54 / 0.34 |
| > 60 | 20,777 | 0.380 | 0.02 | 0.53 | 0.48 / 0.52 |

Base-peak misses with the precursor at one end (10k sample):

| NCE | misses | model's top | true #2 | random real peak |
| --- | ---: | ---: | ---: | ---: |
| ≤ 20 | 502 | 0.787 | 0.677 | 0.624 |
| 20-40 | 879 | 0.443 | 0.419 | 0.364 |
| > 40 | 2,773 | 0.059 | 0.054 | 0.052 |

Displacement by novelty goes from 0.229 (> 0.8) to 0.476 (≤ 0.4), with the one-loss share at
0.39-0.44 and the precursor share at 0.19-0.23 in every bin.

Below NCE 60 the model leaves mass on the precursor or parent of a nested pair far more often than
a random reordering would. Above 60 there is no direction bias, and the error is misordering among
real peaks. On the training sample (f1, filtered target) the displaced mass is 0.136 against
0.363 on val. Precursor moves are at chance there (share 0.174, 0.95x the naive null), while on
val they are 0.216 (1.19x). This fits the
[shortcut audit's](masster_overfit_readout_2026_09_22.md#7-intensity-memorization-threat-model)
reading that the precursor prior is a fallback used only on unseen molecules.

### Precursor versus contender: formula-level loss identity is complementary

At NCE ≤ 30 the precursor wins 54.0% of val pairs. The model (f1) calls the winner correctly in
0.754 (AUC 0.798). By outcome, it is right in 0.881 of pairs the precursor wins and 0.605 of pairs
the fragment wins; f0 gives 0.763 / 0.894 / 0.608. On train it is right in 0.941.

The formula-only learner reaches 0.714 (AUC 0.783), with 0.751 when the precursor wins and 0.670
when the fragment wins. The per-NCE-bin population rate reaches 0.627. Alone, the learner trades
precursor wins for fragment wins, as the rejected FraGNNet global precursor bias did. Stacked:

| held out | model AUC | stacked AUC | accuracy: model / stacked | fragment-wins accuracy: model / stacked |
| --- | ---: | ---: | --- | --- |
| val, other half of molecules | 0.798 | **0.829** | 0.758 / 0.760 | 0.674 / 0.716 |
| test | 0.807 | **0.838** | 0.762 / 0.768 | 0.676 / 0.709 |

The model's accuracy uses a logistic recalibration fitted on the same half. The identity of the
lost neutral therefore carries signal on this decision that the model does not use on unseen
molecules. The gain is in ordering (AUC); accuracy at a single threshold moves by only
0.002-0.006.

## Addendum: bond-type check, ICEBERG-style (2026-09-24)

**Question.** ICEBERG-2 reports that its aggregate bond-type response to CE is chemically
sensible (Extended Data Fig. 3a; see the
[literature note](../../notes/intensity_displacement_literature_2026_09_24.md)). Does MASSter
get the aggregate trends right too? And is its transfer failure *between* bond types or
*within* them, that is, which bond of a type breaks in this molecule?

**Method.**

- Script: [`scripts/eval/bond_type_ce_allocation.py`](../../../scripts/eval/bond_type_ce_allocation.py),
  tests in [`tests/test_bond_type_ce_allocation.py`](../../../tests/test_bond_type_ce_allocation.py).
- It hooks the f1 checkpoint (`6h67zps9`, its own filtered target) to record each retained
  fragment's severed bonds, and runs it in eval mode on 20,480 train and 10,229 val spectra
  (40 batches each).
- Each fragment's mass (summed over ion/H channels) and its matched target mass are split
  equally over its cut bonds; the intact precursor goes to a pseudo-bond. Bonds are typed by
  element pair and order.
- Per spectrum, bond-level TV splits exactly into a precursor term, a between-bond-type term
  and a within-type term.
- Target mass at an m/z shared by several candidate fragments is split two ways: equally
  (model-free), or by the model's own channel mass ("responsibility").
- Outputs: `output/cross_eval_mh_ft_clean_2026_09_23/analysis/displacement/bond_type/`.

**The aggregate trend is right on both splits.** Under the responsibility split, pooled
bond-type shares match the target to about 0.01 on train and on val, in every NCE bin. For
example:

- the precursor falls from 0.63 (predicted) / 0.61 (target) at NCE ≤ 20 to 0.02 / 0.02 above 60
  on val;
- aromatic C:C rises from 0.002 / 0.004 to 0.187 / 0.189.

Under the equal split, aromatic C:C and C=O look about 2x under-predicted at every CE on
*both* splits, but the responsibility split removes that. It comes from isomeric fragments
sharing an m/z, which the spectrum metric cannot see either. So, like ICEBERG, MASSter
learned the population-level bond-type-by-CE behaviour, and on novel molecules too.

**Per molecule, the transfer failure is mostly between bond types.** Responsibility split, mean
per-spectrum TV:

| component | train | val | val − train | share |
| --- | ---: | ---: | ---: | ---: |
| precursor vs fragments | 0.012 | 0.031 | +0.019 | 16% |
| between bond types | 0.041 | 0.112 | +0.071 | 59% |
| within a type | 0.015 | 0.046 | +0.031 | 26% |
| bond level, total | 0.068 | 0.189 | +0.121 | |

- **By NCE.** The between-type gap sits at every NCE, from +0.06 to +0.09: train is flat at
  0.04 and val runs 0.10-0.13. The precursor gap is at low CE (+0.051 at NCE ≤ 20, +0.003 above
  60). The within-type gap grows with CE, from +0.015 to +0.037.
- **Top-type hit rate.** The dominant bond type is right in 0.931 of train spectra and 0.829 of
  val.
- **Within the dominant type**, the right bond is picked 0.939 on train and 0.825 on val, against
  chance of about 0.3.
- **The equal split agrees on the ordering:** between types 45%, precursor 29%, within 26%, on a
  higher, isomer-inflated floor.

**Reading.**

- **The model knows the average behaviour of bond types at each CE, even on unseen molecules.**
  A type-level CE response or bond-type prior would add nothing.
- **What fails to transfer is the ranking of different kinds of bonds inside one molecule.**
  Whether this molecule's C–N, C–O or C–C bond goes first accounts for 59% of the gap. Choosing
  among bonds of the same kind is 26%, and precursor survival 16%.
- **This narrows 2A's requirement.** A local lability prior must put bonds of different types
  on one comparable scale, such as the threshold/slope curve of the literature note.
  Discriminating bonds within a type is not enough on its own.

**Limits.**

- Bond typing is coarse (element pair and order only).
- Multi-cut fragments split their mass equally over their cut bonds.
- The responsibility split uses the model's own isomer choice, so it understates target-side
  ambiguity. The equal split overstates it, which is why both are reported.
- One checkpoint. The train sample comes from the train loader in eval mode.

## Reading

- **Displacement is mostly systematic, not noise.** Two thirds of it survives the replicate floor
  and is common to both runs, in every novelty bin. Variance reduction (EMA, averaging) can reach
  at most the 28% run-specific share. The
  [ranking-transfer plan's](../../plans/masster/masster_ranking_transfer_plan.md) `u2` EMA
  arm (+0.0094) sits inside that share. The rest needs information or inductive bias, not
  smoothing.
- **It is an ordering problem among the real peaks, not a locality problem.** The model almost
  always puts its mass on peaks that exist. What it gets wrong is which of them win, and the wrong
  winner is no closer in the fragmentation lattice than chance. Any fix aimed at "neighbouring"
  candidates has no premise: sibling or neighbour attention biases, H-shift channel work, or
  one-loss pairwise terms. This is consistent with the exact null of the pairwise mass bias
  (−0.0002).
- **The one structured part is the fallback below NCE 60.** On unseen molecules the model keeps
  the precursor or parent when the molecule's labile step should have fired. A molecule-free
  formula learner already recovers +0.03 AUC of that decision. That is direct evidence for the
  plan's lability hypothesis at low energy, measured at formula level. Structure-level features
  should do better if the hypothesis holds.
- **High CE is where most of the error is, and it has no structure these classes can see.** The
  20,777 spectra above NCE 60 carry the largest displacement (0.380) with no direction bias.
  Formula classes cannot separate a lability error there from any other misordering.

## Consequences for the plan

1. **G0:**
   - Condition 1 (shared share at least half after the floor) passes: 67% val, 63% test.
   - Condition 2 (a class at 1.5x or more its chance share) passes only against the naive null.
     It fails against the support-preserving null (at most 1.08x), which is the right control.
2. **Dropped:**
   - arm 2C (H shift): 6% of the mass, 0.96x the support null;
   - the H2/H3 split (one-loss against sibling);
   - Phase 1.2's fragment-level classes: the class mix already matches a random reordering, so
     finer classes would not change the reading.
3. **Kept, re-aimed:**
   - Phase 1.4's bond-local baseline now targets real-peak ordering in general, not a pre-selected
     pair type.
   - It starts with the precursor-versus-contender decision, where the formula-level learner
     already stacks for +0.03 AUC. The test is whether bond-local structure adds more than
     formula identity.
   - Phase 1.3 (environment support) and 1.5 (conditional sharpness) are unchanged.
4. **Arm 2A's premise narrows to lability ordering.** The question is no longer "neighbour
   confusion" but whether an identity-blind local rule orders real peaks on unseen molecules
   better than the model. G1 still decides it.

## Limits

- f0 and f1 share a seed and differ in training target. The first inflates the shared share and
  the second deflates it. A pure seed pair is still unmeasured.
- One lineage: unmerged `[M+H]+`/FT. The merged `c4a_base` lineage and MassSpecGym are
  unmeasured.
- Formula-level classes only. Nesting is by heavy-atom containment, which is looser than
  fragment-level containment. The support-preserving null uses the same classes, so the
  comparison is fair but coarse.
- The support-preserving null and the base-miss references ran on a 10k val sample. Shares match
  the full split within 0.005 for the predictors both contain.
- About 0.1% of spectra (50 of 52,020) map a target peak to a candidate only one run holds in the
  union space. The mean cosine shifts by 1e-4.
- Precursor pairs use the target's strongest non-precursor peak, chosen with the target in hand,
  so they test ordering on a given pair, not contender discovery. The stack weight is fitted on
  val molecules and confirmed on test.
