# Wrong base-peak choices of `i1`: a broad transfer deficit, not one missing distinction

> **Status:** FINDING · **Updated:** 2026-09-28
> Inference only, on the local RTX 2070 SUPER: `i1_inst_ob_obf` (`i50ig7rf`) at its best epoch
> (59), on the full val split (52,020 spectra, 4,222 molecules) and 4,000 random train molecules
> (49,205 spectra). This is step 1 of the three-step plan the user proposed on 2026-09-28: build
> the wrong-peak table, look for one repeated missing distinction, then add one representation
> branch aimed at it.
>
> - **The base peak is missed in 40% of val spectra, and it is a candidate in 94% of those.** On
>   train, 15%. The ranking gap is 0.104 on val and 0.010 on train, as for `f1`.
> - **Matched pairs show strong chemistry biases on val.**
>   - Favored: the intact precursor, shallower fragments, fragments keeping every aromatic ring.
>   - Disfavored: extra cuts, ring cleavage, H transfer, and C–C, aryl–N and heteroatom–heteroatom
>     cleavages.
>   - Calibrated: amide, acyl, amine and anomeric cleavages.
> - **But the model is calibrated in aggregate, so these biases are hedging, not a systematic
>   offset.**
>   - Over all candidates it predicts 1.84 cuts per spectrum against 1.93 measured.
>   - A cross-fitted reweighting on eight structure attributes adds +0.0004 cosine over `P^γ`
>     (+0.0009 per NCE bin). Its fitted per-cut weight is −0.003.
>   - Where the model cannot tell candidates apart on a new molecule, it falls back on the prior.
> - **Discrimination falls on val for every kind of decision, by similar amounts.**
>   - Balanced pair accuracy is 0.71–0.79 across every step type (train 0.86–0.92) and 0.63–0.76
>     across cut classes (train 0.75–0.91).
>   - It is flat across NCE and rises with similarity to training (nested pairs 0.657 → 0.810).
> - **By volume, the failing decisions are about how far fragmentation goes, then which bond
>   breaks.**
>   - Nested pairs, one candidate further along the other's path: 69% of val top-two decisions,
>     60% of base-peak misses.
>   - Overlapping or disjoint alternatives: 26% of decisions, 34% of misses.
>   - Which side of one cleavage keeps the charge: 1.5% of decisions, 1.3% of misses.
> - **For step 3:**
>   - Charge-related features have the least support.
>   - A post-cut connectivity branch aims at the decision types that fail. But the table shows no
>     input the model lacks: it makes the same decisions correctly on training molecules.
>   - Such a branch tests whether a different inductive bias transfers. Read it on nested and
>     sibling pairs at novelty < 0.6, against `i1`.

## Question

The user's step 1 (2026-09-28): tabulate `i1`'s wrong peak choices, comparing the predicted and
the measured strongest peak where the base peak is covered. Group them by cut environment,
fragment size, neutral loss, instrument and energy. Keep every candidate structure of an
ambiguous peak. Step 2 asks for one repeated missing distinction, checked against matched
successes so common chemistry does not look like a failure. Step 3 would add one representation
branch aimed at it, such as a description of the retained fragment connectivity and the newly
exposed cut sites. The table should say whether to invest in fragment connectivity,
charge-related features or something else.

## Method

- **Stage 1, GPU:**
  [`debug_masster_base_peak_errors.py`](../../../scripts/debug/masster_memorization/debug_masster_base_peak_errors.py).
  - One annotated forward per batch in eval mode (`return_explainability_tensors`). Each output
    channel carries its m/z, probability, fragment, H shift and ion mode, plus the fragment's atom
    memberships.
  - Channel m/z equals the fragment's atoms and their H, plus the H shift, plus a proton, to
    5e-7 Da (checked on 228 spectra). `i1` has charge-retention channels only.
  - Channels are merged by m/z into candidate peaks. The eval-convention target is used: relative
    filter, normalized, no transform. Each target peak maps to its nearest candidate within
    10 ppm (200 Da floor).
  - Per spectrum it records the ranking decomposition's coverage/ranking/calibration split
    ([ranking decomposition](masster_ranking_decomposition_2026_09_23.md)) and the base-peak
    outcome.
    - `hit`: the model's top candidate is within tolerance of the measured base peak.
    - `under_ranked`: the base peak is a candidate, but not the model's top.
    - `uncovered`: the base peak is not a candidate.
  - It keeps every structure behind the model's and the target's top-3 peaks, and structure
    attributes for every candidate. 107-119 candidates per spectrum are saved, holding at least
    99.8% of predicted mass (median 99.97%).
  - About 5 min per split.
- **Stage 2, CPU:**
  [`debug_masster_base_peak_tables.py`](../../../scripts/debug/masster_memorization/debug_masster_base_peak_tables.py).
  Each structure is annotated with:
  - its neutral loss;
  - its cut bonds, in 19 functional classes plus element fallbacks, and whether each is a ring
    bond;
  - whether it keeps a whole match of the molecule's most basic N/P site class;
  - whether it keeps, loses whole or breaks aromatic rings;
  - how much of the largest conjugated system it keeps.
- **Ambiguous peaks.** A peak's structures all share one formula, so its loss is exact. Its cut
  sites often are not: a median of 2 structures per peak. Structure-level attributes take the
  prediction-weighted mean. A peak is flagged ambiguous when structures holding ≥ 5% of its share
  disagree, and every pairwise result is repeated on unambiguous peaks or on peaks with a single
  structure.
- **Matched pairs.** Take the measured #1 and #2 candidates of each spectrum, with #2 ≥ 5% of #1.
  The model is right when it predicts #1 above #2.
  - For an attribute on exactly one side, bias = accuracy when it is on #1 minus accuracy when it
    is on #2.
  - A joint least-squares fit regresses the log-ratio residual, `log(p1/p2) − log(t1/t2)`, on
    the attribute differences.
- **No rank selection.** Two checks that do not select pairs by target rank:
  - calibration in the large, the predicted against the target mean of each attribute over all
    candidates;
  - a cross-fitted reweighting, `p' ∝ p^γ exp(β·x)`, fitted to maximize cosine on one half of the
    molecules (by `mol_id` parity) and scored on the other.
- **Other settings.** Novelty is the maximum Morgan-r2 Tanimoto to another train molecule (train)
  or to any train molecule (val), from `output/i1_memorization_2026_09_28/meta.pkl`. Intervals are
  molecule-clustered bootstraps.
- **The candidate-space cosine is tolerance-matched** (0.7878 on val, against W&B's binned
  0.7622).
  - The median spectrum agrees exactly. The mean gap comes from about 1% of spectra, and 90% of
    those have the base peak and its matched candidate in different 0.01 Da bins.
  - This is the bin splitting that the
    [residual-structure finding](masster_i1_residual_structure_2026_09_28.md) puts at 9% of
    the logged val deficit.

## Result

### 1. Outcome

| split | spectra | cos | coverage | ranking | calibration | hit | under-ranked | uncovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 49,205 | 0.959 | 0.015 | **0.010** | 0.016 | 0.848 | 0.133 | 0.019 |
| val | 52,020 | 0.788 | 0.020 | **0.104** | 0.088 | 0.603 | 0.372 | 0.025 |

`f1` on its cleaned target read 0.121 against 0.011 and a hit rate of 0.567 against 0.845.

### 2. Hit rate by stratum (covered spectra)

| stratum | train | val |
| --- | ---: | ---: |
| Elite (`OB`) / Lumos (`OB_F`) | 0.865 / 0.863 | 0.631 / 0.584 |
| NCE ≤ 20 / 30–40 / > 60 | 0.922 / 0.899 / 0.807 | 0.764 / 0.658 / 0.525 |
| novelty < 0.4 / 0.4–0.6 / 0.6–0.8 / ≥ 0.8 | 0.836 / 0.871 / 0.873 / 0.833 | 0.461 / 0.577 / 0.684 / 0.755 |
| base peak is the precursor / a fragment | 0.930 / 0.845 | 0.872 / 0.548 |
| base peak has 1 / 2 / 3 / ≥ 4 cuts | 0.922 / 0.846 / 0.800 / 0.669 | 0.681 / 0.521 / 0.431 / 0.317 |
| base peak cuts a ring bond: no / yes | 0.903 / 0.765 | 0.694 / 0.420 |
| basic site kept / lost / no basic site | 0.894 / 0.840 / 0.853 | 0.657 / 0.552 / 0.620 |

- **Cut class of the measured base peak.** The val hit rate runs from 0.22-0.26 (N–O, N–N)
  through 0.37-0.44 (aryl–O, benzylic C–C, aryl–N, alkyl C–C, alkyl–O) to 0.62-0.63 (alkyl–N,
  amide C–N, anomeric C–O). Train runs 0.73-0.89 over the named classes.
- **Named losses.** The val hit rate runs from 0.41 (CO) and 0.45 (CO2) through 0.58 (H2O) to
  0.70 (NH3) and 0.81 (hexose). Train is 0.88-0.97 throughout.

### 3. Anatomy of the misses

The model's top candidate against the measured base peak, in under-ranked spectra (dominant
structures):

| relation | train | val |
| --- | ---: | ---: |
| model's top contains the base peak (stopped short on the same path) | 0.300 | **0.345** |
| base peak contains the model's top (went too far) | 0.309 | 0.257 |
| overlapping (a different bond broke) | 0.211 | 0.197 |
| disjoint, not complementary | 0.125 | 0.141 |
| same atoms, different H | 0.048 | 0.047 |
| complementary (the other side kept the charge) | 0.007 | 0.013 |

In val misses the base peak is the model's #2 in 38% and ranks sixth or lower in 32%. On train
it is #2 in 71%.

### 4. Matched pairs: apparent biases

49,221 val and 46,645 train pairs. Pair accuracy is 0.739 on val and 0.881 on train.

| attribute | val bias [95%] | train bias |
| --- | ---: | ---: |
| intact precursor | +0.310 [+0.291, +0.332] | +0.011 |
| keeps every aromatic ring | +0.261 [+0.236, +0.284] | +0.063 |
| larger fragment | +0.135 [+0.123, +0.146] | +0.032 |
| keeps the basic site | +0.084 [+0.049, +0.117] | +0.038 |
| H transfer | −0.168 [−0.186, −0.148] | −0.023 |
| loses a whole aromatic ring | −0.180 [−0.211, −0.150] | −0.017 |
| breaks an aromatic ring | −0.264 [−0.289, −0.239] | −0.089 |
| ring bond cut | −0.279 [−0.304, −0.256] | −0.092 |
| more than one cut | −0.328 [−0.351, −0.307] | −0.101 |
| amide C–N cut, anomeric C–O cut, NH3, HCOOH or hexose loss | −0.04 to +0.04 | −0.01 to +0.04 |

The same pattern holds jointly, in the log-ratio fit.
- **Per extra cut:** −0.420 [−0.474, −0.380] on val, against −0.097 on train. That is −1.37 at
  NCE ≤ 20 and −0.27 above NCE 60; −0.57 at novelty < 0.4 and −0.25 at ≥ 0.8.
- **Per transferred H:** −0.250, against −0.047.
- **Losing a whole aromatic ring:** −0.433, against +0.018.
- **Keeping the basic site, holding depth fixed:** −0.207, against +0.023.
- **Cut classes on top of the per-cut term.** Amide, acyl, amine, acid and anomeric cleavages are
  within their intervals of zero. Alkyl C–C (−0.64), benzylic (−0.80), aryl–N (−0.81),
  biaryl (−0.88), C–S (−0.61) and N–N/N–O/P–O/O–S (−0.83 to −1.42) sit below.
- **Train:** every class is within ±0.26.
- **Robustness:** unambiguous peaks give the same signs.

### 5. Without rank selection, the model is calibrated in aggregate

Predicted minus target mean over all candidates, per spectrum:

| attribute | val: predicted / target | val difference [95%] | train difference |
| --- | --- | ---: | ---: |
| cuts | 1.837 / 1.930 | −0.092 [−0.100, −0.084] | −0.029 |
| transferred H | 0.880 / 0.917 | −0.037 | −0.002 |
| intact precursor | 0.195 / 0.179 | +0.016 | −0.005 |
| ring bond cut | 0.307 / 0.327 | −0.020 | −0.006 |
| breaks / loses an aromatic ring | | −0.015 / −0.010 | −0.005 / +0.001 |
| keeps the basic site (27,915 spectra with one) | 0.607 / 0.600 | +0.007 | +0.004 |

The cut deficit is −0.10 to −0.11 in every NCE bin up to 60 and −0.07 above. It is −0.16 at
novelty < 0.4. The precursor excess is +0.051 at NCE ≤ 20 and +0.002 above 60, the low-energy
precursor default of the
[ranking decomposition](masster_ranking_decomposition_2026_09_23.md#the-low-ce-precursor-default-is-still-there).

Cross-fitted reweighting, held-out mean cosine gain over the raw 0.7878:

| reweighting | val gain [95%] | train gain |
| --- | ---: | ---: |
| `P^γ` alone (γ = 0.79 on val, 0.93 on train) | +0.0052 [+0.0048, +0.0056] | +0.0006 |
| `P^γ` and eight attribute weights | +0.0056 [+0.0052, +0.0061] | +0.0010 |
| the same, fitted per NCE bin | +0.0060 [+0.0056, +0.0065] | +0.0012 |

Fitted on the whole val split, the attribute weights are small: −0.003 per cut, −0.024 per
transferred H, +0.15 on the precursor. The precursor term is consistent with the
[memorization read's](masster_i1_memorization_read_2026_09_28.md) null precursor rescale.

So the section 4 biases are not an offset a global correction can remove. Boosting deep candidates
boosts the wrong deep candidates too. Pairs chosen by target rank show a bias toward the prior
whenever the model cannot tell two candidates apart: the size of each bias follows how often that
attribute's candidate wins its pair. Amide cleavage wins 47% of its pairs and shows −0.04; a
multi-cut candidate wins 34% and shows −0.33; the precursor wins 67% and shows +0.31. The
exceptions are the well-learned losses: NH3 wins 41% but shows +0.007, at 0.78 accuracy on both
sides.

### 6. Where discrimination fails

Pair accuracy by how the measured #1 and #2 relate:

| relation | share of val pairs | train | val | val − train |
| --- | ---: | ---: | ---: | ---: |
| nested, the measured winner is the deeper candidate | 27.1% | 0.865 | **0.638** | −0.227 |
| nested, the measured winner is the shallower candidate | 41.6% | 0.910 | 0.835 | −0.075 |
| overlapping (which bond breaks) | 14.7% | 0.833 | 0.681 | −0.152 |
| disjoint, not complementary | 11.4% | 0.872 | 0.716 | −0.156 |
| same atoms, different H | 3.8% | 0.861 | 0.691 | −0.169 |
| complementary (which side keeps the charge) | 1.5% | 0.933 | 0.738 | −0.194 |

Restricted to pairs of single-structure peaks (8,372 val pairs), the ordering is the same:
nested with a deeper winner 0.670, overlapping 0.731, complementary 0.778 (239 pairs).

Nested pairs, balanced accuracy (the mean over which side the target favors):

| | NCE ≤ 20 | 20–30 | 30–40 | 40–60 | > 60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0.919 | 0.910 | 0.904 | 0.897 | 0.851 |
| val | 0.713 | 0.737 | 0.747 | 0.737 | 0.720 |
| val asymmetry (shallower-winner minus deeper-winner accuracy) | 0.404 | 0.257 | 0.187 | 0.188 | 0.053 |

| | novelty < 0.4 | 0.4–0.6 | 0.6–0.8 | ≥ 0.8 |
| --- | ---: | ---: | ---: | ---: |
| train | 0.876 | 0.894 | 0.891 | 0.856 |
| val | 0.657 | 0.712 | 0.773 | 0.810 |

- **Energy sets the direction of the hedge, not how well the model discriminates.** Val
  discrimination is flat across NCE. The asymmetry is the prior: stay shallow at low energy,
  balanced above NCE 60.
- **By the step between the two nested candidates,** val balanced accuracy is 0.706 (CH3) to
  0.790 (HCl), with H2O at 0.717, CO 0.759 and NH3 0.779. Train is 0.864 to 0.917.
- **By cut class (pairs where the class is cut on one side only, ≥ 300 val pairs),** val is
  0.633 (N–O) to 0.755 (amide C–N), and train 0.75 to 0.91. The deficit is −0.10 to −0.24 for
  every class but one: an unmatched aromatic C–O class (434 pairs), −0.06, which is weak on
  train as well (0.75).

## Reading

- **No single repeated missing distinction.**
  - Every attribute tested shows the same pattern: calibrated on train, calibrated in aggregate on
    val, biased toward the prior in pairs.
  - The val discrimination deficit is similar across decision types, step types and bond classes.
  - The strongest covariate is similarity to training.
  - This is the ranking-transfer problem of the
    [ranking decomposition](masster_ranking_decomposition_2026_09_23.md), not a chemistry class
    the model cannot represent.
  - It agrees with the [residual-structure finding](masster_i1_residual_structure_2026_09_28.md)
    on the same checkpoint. There, 64% of the val deficit is how intensity is split among
    covered candidates within a depth band, and predictions follow similarity to training
    molecules, not a rule that transfers.
- **Charge-related features: least supported.**
  - Which side of a cleavage keeps the charge decides 1.5% of val top-two pairs and 1.3% of
    misses.
  - Basic-site retention is calibrated in aggregate, and reweighting it recovers nothing.
  - The token already carries the Gasteiger charge sum (`c_charge`), and the model has a charge
    head.
- **Fragment connectivity: aimed at the right decisions, not at a demonstrated gap.**
  - Nested decisions (does fragmentation go one step further?) are 69% of val pairs and 60% of
    misses. Sibling decisions (which bond breaks) are another 26% and 34%.
  - A branch describing each candidate after cutting would target these. But on training
    molecules the same decisions are made correctly: balanced 0.85–0.92 nested, 0.83–0.87
    siblings.
  - The oriented cut sites are already in the token: `CutBondEncoder` on the (retained, lost)
    endpoints, per [feature audit §11](../../reference/masster_feature_set_audit.md#11-fixes-applied-2026-09-20).
    What the branch would add is message passing on the retained subgraph.
  - It is a test of inductive bias for transfer, and its gate should be on val decisions, not
    train fit.
- **Global corrections are exhausted.** `P^γ` is the only cheap gain, and it is the free +0.005
  already known. Attribute priors add +0.0004 to +0.0009.

## Evaluating a step 3 arm against `i1`

The two scripts give every quantity the plan names, for any MASSter checkpoint. They take about
5 min per split for stage 1 and 3 min for stage 2 on this GPU:

- base-peak hit rate, overall and at novelty < 0.6 (`i1` val: 0.603 over all spectra; over
  covered spectra, 0.461 below 0.4 and 0.577 at 0.4–0.6);
- the ranking gap (`i1` val 0.104);
- calibrated cosine, cross-fitted `P^γ` (`i1` val 0.7929 in candidate space);
- the decision the branch aims at: nested balanced accuracy, 0.657 / 0.712 at novelty < 0.4 /
  0.4–0.6, and overlapping-pair accuracy, 0.681.

A +0.003 cosine effect needs two seeds per arm
([memorization audit](masster_memorization_attempts_audit_2026_09_27.md)). Runtime is not
measured here.

## Limits

- The CUDA candidate builder swaps tied isomers between calls even in eval mode
  ([dropout-view finding](masster_dropout_view_disagreement_2026_09_27.md#side-findings)). A
  rerun of both stages reproduced Tables 1, 2, 4, 6 and 7 exactly. It moved the
  dominant-structure relations of Tables 3 and 8 by up to 0.003; the complementary class went
  from 761 to 753 val pairs. The numbers here are from the rerun, the recorded artifacts.
- One checkpoint, one seed. The train sample is 4,000 random molecules. Nothing here was trained.
- Cut classes, basic sites and aromatic rings come from SMARTS and RDKit perception on
  standardized molecules. The candidates are MASSter's cut fragments with H shifts, so
  rearrangement products appear under whatever cut set yields their formula.
- For peaks with several structures, the target's attribute mean splits intensity by the model's
  structure shares. That is 13-52% of target mass, depending on the attribute (none for loss,
  size and precursor). The pairwise results were repeated on unambiguous peaks.
- The reweighting uses attribute means per candidate peak and the candidates saved (prediction
  ≥ 1e-5 or target intensity). A richer per-candidate correction was not tried.

## Artifacts

Local and Git-ignored, under `output/base_peak_errors_2026_09_28/`:
- the extraction run (`run_extract.sh`);
- per-split spectrum, peak and candidate tables, with provenance JSON (git revision, diff hash,
  input md5s);
- `tables/tables.txt` with the CSVs behind every table.

Tests: `tests/test_debug_masster_base_peak_errors.py` (26).
