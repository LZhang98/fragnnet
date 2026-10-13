# GLACIER contrastive fine-tuning: wrong-spectrum (swap) test on MassSpecGym

> **Status:** FINDING · **Updated:** 2026-10-03
> On MassSpecGym formula retrieval, the released GLACIER checkpoint (with contrastive fine-tuning,
> CF) shows no spectrum-blind structure prior beyond MASSter (no CF). With a wrong spectrum of
> a different formula, both rank the truth at chance. With a wrong spectrum of a same-formula
> molecule, both rank above chance, and MASSter ranks higher. The NIST'20 random-split shortcut
> argument in [the GLACIER note](../../notes/glacier_object_detection.md) is still unmeasured.

## Question

GLACIER's CF uses measured positives and PubChem/mutation negatives. The note argues that this
can teach a "looks measured" prior that ranks candidates without reading the spectrum. The test
from the MassSpecGym-in-the-wild audit (§5.2): score each test query's candidates against a
*different* test spectrum, then compare hit@k with chance and with a model trained without CF.

## Setup

- Test fold, all 9,954 `[M+H]+` queries, our candidate list, the saved candidate predictions
  of the released GLACIER MassSpecGym checkpoint and MASSter `ll04i00m` ep060
  ([retrieval finding](masster_msgym_formula_retrieval_full_2026_10_03.md)). No new inference.
- **Donors** (same instrument, closest CE, random among ties, seed 0):
  - `sameF`: a test query whose truth has the same formula and a different 2D InChIKey. Exists
    for 1,114 queries (221 of 2,318 molecules).
  - `near`: the nearest precursor m/z with a different 2D InChIKey, for all queries
    (median |Δm/z| 0.015 Da, p90 1.03 Da; 11% are the same formula).
- Under a swapped spectrum, candidates sharing the donor's 2D InChIKey are removed. The score is
  the expected hit@k (random ties) of the query's own truth, so it measures ranking by structure
  alone. Chance is the mean of min(k, n)/n over the remaining candidates.
- Scores: ms-pred binned 0.01 Da sqrt entropy (`ent`, GLACIER's native ranking) and the frozen
  calibrated query log-likelihood (`ll`). Spectrum-free rankers: the negative entropy of each
  candidate's predicted spectrum (sharpness) and its support size.
- Intervals: 95% bootstrap over molecules, paired by query.

## Results

The true-spectrum rows reproduce the earlier reads (GLACIER `ent` 0.4965, MASSter `ll` 0.5284).

| spectrum | n | chance @1 / @5 / @20 | GLACIER `ent` | GLACIER `ll` | MASSter `ent` | MASSter `ll` |
|---|---:|---|---|---|---|---|
| true | 9,954 | 0.025 / 0.105 / 0.270 | 0.497 / 0.826 / 0.942 | 0.490 / 0.829 / 0.951 | 0.485 / 0.818 / 0.949 | 0.528 / 0.853 / 0.965 |
| `near` (wrong) | 9,954 | 0.025 / 0.105 / 0.271 | 0.030 / 0.117 / 0.304 | 0.030 / 0.113 / 0.302 | 0.032 / 0.130 / 0.319 | 0.031 / 0.122 / 0.315 |
| true, `sameF` subset | 1,114 | 0.014 / 0.072 / 0.192 | 0.433 / 0.692 / 0.873 | 0.440 / 0.688 / 0.867 | 0.425 / 0.727 / 0.938 | 0.481 / 0.738 / 0.929 |
| `sameF` (wrong) | 1,114 | 0.016 / 0.074 / 0.195 | 0.076 / 0.205 / 0.443 | 0.079 / 0.205 / 0.446 | 0.105 / 0.311 / 0.562 | 0.107 / 0.281 / 0.537 |

| contrast, hit@1 | `near` | `sameF` |
|---|---|---|
| GLACIER `ent` − chance | +0.005 [−0.001, +0.011] | +0.060 [+0.029, +0.101] |
| MASSter `ll` − chance | +0.005 [−0.002, +0.013] | +0.091 [+0.054, +0.132] |
| GLACIER − MASSter, `ent` | −0.002 [−0.009, +0.005] | −0.029 [−0.060, +0.008] |
| GLACIER − MASSter, `ll` | −0.000 [−0.007, +0.006] | −0.028 [−0.056, +0.006] |

At @5 and @20, under `sameF`, MASSter leads GLACIER by 0.076-0.119 (`ent` @5: −0.106 [−0.209, −0.009]).

Spectrum-free rankers, hit@1 (chance 0.025): GLACIER sharpness 0.019, MASSter sharpness 0.019,
GLACIER support size 0.025, MASSter support size 0.016. Prediction shape carries no
"measured-like" signal.

When the donor's own truth is in the query's candidate list (about 600 queries per arm), the swapped
spectrum puts it first 18-24% of the time, about the same for both models.

## Reading

1. **Neither model has a spectrum-blind structure prior on MassSpecGym.** With a wrong spectrum
   of nearby mass, both sit within +0.005 of chance, and the GLACIER CF checkpoint is level with
   MASSter.
2. **The above-chance `sameF` hits come from reading the spectrum, not from a prior.** Same-formula
   test molecules are often close analogs, so the donor's spectrum partly resembles the truth's.
   MASSter, which has no CF, gains more from it than GLACIER does. A CF prior would push the
   other way.
3. **Scope.** This tests the released MassSpecGym CF model on the MCES-split benchmark, where the
   paper's CF gain is only +1.4 pt. The shortcut argument is about the NIST'20 random split, where
   CF gives +0.135. That case is not tested: no NIST'20 GLACIER checkpoint is released, and
   retraining one means running ms-pred's MAGMa labelling pipeline on NIST'20 first (judged
   not worth it, user, 2026-10-03). The policy of comparing against w/o-CF rows still
   holds, on the argument alone.

## Artifacts

`output/msgym_swap_test_2026_10_03/`: `build_donors.py` → `donors.csv`; `score_swap.py <shard>`
→ `test/swap_XX.pkl`; `analyze_swap.py` → `swap_summary.csv`, `swap_paired.csv`,
`swap_donor_truth.csv`.
