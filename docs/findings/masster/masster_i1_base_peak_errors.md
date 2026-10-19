# `i1` wrong base peaks: error table, post-cut gate and `fc1` screen

> **Status:** FINDING · **Updated:** 2026-10-18
> The user's three-step plan on `i1`'s wrong base-peak choices: step 1 error table, step 2 post-cut information gate, step 3 `fc1` local screen. All three negative for a single missing input.

Merged 2026-10-18 from 3 findings on one topic. Each part below is the original
document unchanged except for heading levels; its own status line keeps its original date.
Former file names are listed so old links and `git log -- <former path>` can be traced.

| Part | Updated | Verdict | Former file |
| --- | --- | --- | --- |
| [Wrong base-peak choices of `i1`: a broad transfer deficit, not one missing distinction](#wrong-base-peak-choices-of-i1-a-broad-transfer-deficit-not-one-missing-distinction) | 2026-09-28 | Step 1: base peak missed in 40% of val (train 15%); hedging, no single missing distinction | `masster_i1_base_peak_errors_2026_09_28.md` |
| [Post-cut information gate on `i1`: explicit fragment and charge descriptors add nothing](#post-cut-information-gate-on-i1-explicit-fragment-and-charge-descriptors-add-nothing) | 2026-09-29 | Step 2: connectivity/charge descriptors add <= +0.001 over the `P^gamma` baseline | `masster_i1_postcut_information_gate_2026_09_29.md` |
| [Fragment-connectivity branch `fc1`: local screen against `i1` is not positive](#fragment-connectivity-branch-fc1-local-screen-against-i1-is-not-positive) | 2026-09-29 | Step 3: `fc1` +0.0050 mean, below the 0.0076 control spread; no Fir run | `masster_fragment_connectivity_local_screen_2026_09_29.md` |

## Wrong base-peak choices of `i1`: a broad transfer deficit, not one missing distinction

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

### Question

The user's step 1 (2026-09-28): tabulate `i1`'s wrong peak choices, comparing the predicted and
the measured strongest peak where the base peak is covered. Group them by cut environment,
fragment size, neutral loss, instrument and energy. Keep every candidate structure of an
ambiguous peak. Step 2 asks for one repeated missing distinction, checked against matched
successes so common chemistry does not look like a failure. Step 3 would add one representation
branch aimed at it, such as a description of the retained fragment connectivity and the newly
exposed cut sites. The table should say whether to invest in fragment connectivity,
charge-related features or something else.

### Method

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
    (ranking decomposition (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_ranking_decomposition_2026_09_23.md`)) and the base-peak
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

### Result

#### 1. Outcome

| split | spectra | cos | coverage | ranking | calibration | hit | under-ranked | uncovered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 49,205 | 0.959 | 0.015 | **0.010** | 0.016 | 0.848 | 0.133 | 0.019 |
| val | 52,020 | 0.788 | 0.020 | **0.104** | 0.088 | 0.603 | 0.372 | 0.025 |

`f1` on its cleaned target read 0.121 against 0.011 and a hit rate of 0.567 against 0.845.

#### 2. Hit rate by stratum (covered spectra)

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

#### 3. Anatomy of the misses

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

#### 4. Matched pairs: apparent biases

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

#### 5. Without rank selection, the model is calibrated in aggregate

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
ranking decomposition (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_ranking_decomposition_2026_09_23.md`).

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

#### 6. Where discrimination fails

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

### Reading

- **No single repeated missing distinction.**
  - Every attribute tested shows the same pattern: calibrated on train, calibrated in aggregate on
    val, biased toward the prior in pairs.
  - The val discrimination deficit is similar across decision types, step types and bond classes.
  - The strongest covariate is similarity to training.
  - This is the ranking-transfer problem of the
    ranking decomposition (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_ranking_decomposition_2026_09_23.md`), not a chemistry class
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

### Evaluating a step 3 arm against `i1`

The two scripts give every quantity the plan names, for any MASSter checkpoint. They take about
5 min per split for stage 1 and 3 min for stage 2 on this GPU:

- base-peak hit rate, overall and at novelty < 0.6 (`i1` val: 0.603 over all spectra; over
  covered spectra, 0.461 below 0.4 and 0.577 at 0.4–0.6);
- the ranking gap (`i1` val 0.104);
- calibrated cosine, cross-fitted `P^γ` (`i1` val 0.7929 in candidate space);
- the decision the branch aims at: nested balanced accuracy, 0.657 / 0.712 at novelty < 0.4 /
  0.4–0.6, and overlapping-pair accuracy, 0.681.

A +0.003 cosine effect needs two seeds per arm
([memorization audit](masster_memorization_audits.md#audit-of-every-attempt-to-close-massters-memorization-gap)). Runtime is not
measured here.

### Limits

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

### Artifacts

Local and Git-ignored, under `output/base_peak_errors_2026_09_28/`:
- the extraction run (`run_extract.sh`);
- per-split spectrum, peak and candidate tables, with provenance JSON (git revision, diff hash,
  input md5s);
- `tables/tables.txt` with the CSVs behind every table.

Tests: `tests/test_debug_masster_base_peak_errors.py` (26).

## Post-cut information gate on `i1`: explicit fragment and charge descriptors add nothing

> **Status:** FINDING · **Updated:** 2026-09-29
> Inference and offline fitting only, on the local RTX 2070 SUPER. `i1_inst_ob_obf`
> (`i50ig7rf`) at epoch 59, all 52,020 val spectra (4,222 molecules). This is step 2 of the
> user's three-step plan, after the
> [wrong base-peak table](#wrong-base-peak-choices-of-i1-a-broad-transfer-deficit-not-one-missing-distinction) found no single repeated
> missing distinction.
>
> - **Question.** Does an explicit description of each candidate after cutting carry information
>   `i1` does not use? Two descriptions are tested: its retained connectivity and exposed cut
>   sites, and its relation to the charge site.
> - **Test.** A per-candidate correction `logit' = log p + g(x)`.
>   - It is fitted on half the val molecules and scored on the other half, by `mol_id` parity.
>   - The context-only correction (`log p`, NCE, instrument, precursor m/z) reproduces the known
>     `P^γ` gain, +0.0053 cosine. Each descriptor family is read by its gain over that baseline.
> - **Neither family adds more than about +0.001 cosine on held-out molecules, and neither
>   stands out.** Over the baseline, in three fitting settings:
>   - post-cut connectivity (80 columns): −0.0008 (MLP), −0.0004 (linear), +0.0010 (regularized
>     MLP);
>   - charge (20 columns): +0.0002, +0.0001, +0.0009;
>   - the earlier eight structure attributes: +0.0005, +0.0002, +0.0008.
>   - The regularized baseline itself fits 0.0004 worse. Against the best baseline, no family
>     adds more than +0.0006.
> - **For step 3:** hand-built post-cut and charge descriptors cannot predict `i1`'s errors on
>   new molecules. This lowers the prior for a branch that describes the candidate after cutting,
>   without ruling it out: a learned branch trained on training molecules sees different features
>   and a different objective.

### Question

Step 1 found that `i1`'s wrong base-peak choices come from a broad transfer deficit. They are not
a systematic bias a global correction removes: eight attribute weights add +0.0004. A
per-attribute linear reweighting cannot use the chemistry of each candidate jointly, though.
Step 2 asks whether a richer, explicit description of each candidate after cutting predicts the
errors on molecules the correction never saw. If it does, the information exists and transfers
across molecules, and a representation branch has something to learn. If it does not, the
description adds nothing the model does not already use.

### Method

[`debug_masster_postcut_information_gate.py`](../../../scripts/debug/masster_memorization/debug_masster_postcut_information_gate.py)
has two subcommands.

- **`extract`** runs the annotated eval-mode forward of the
  [base-peak probe](#method-1). Channels are merged into
  candidate peaks and the eval-convention target is mapped at 10 ppm.
  - For every candidate peak with predicted share ≥ 1e-5 or target intensity (119 per spectrum),
    it saves descriptors computed per fragment from its membership row. They are averaged over
    the peak's structures with the model's shares.
  - 7 min for the val split.
- **Descriptor families:**

  | family | columns | contents |
  | --- | ---: | --- |
  | basics (in every family) | 6 | H shift, its magnitude, intact precursor, heavy atoms, fraction of precursor atoms, cut count |
  | existing | 4 | ring bond cut, aromatic ring lost whole or broken, basic site kept (the eight step-1 attributes, with the basics) |
  | post-cut connectivity | 74 | retained bonds and the fragment's cyclomatic number; SSSR and aromatic rings kept and broken; conjugation kept; ring, aromatic and multiple bonds cut; retained and lost cut-site atom types (12 each); cation-stabilizing retained and lost cut sites (acyl, benzylic, α-heteroatom, allylic, tertiary carbon, heteroatom); cut-bond classes (20); element counts, H count and RDBE of the neutral species |
  | charge | 14 | basic site kept, kept matches, its class (7); graph distance from the retained cut sites to the kept basic site (and a missing flag); Gasteiger charge sum, minimum and maximum of the fragment, and the complement's sum |

- **`fit`** fits `logit' = log p + g([log p, context, x])` by listwise softmax cross-entropy
  against the target over the saved candidates.
  - `g` is zero at start: a two-layer MLP (width 128, dropout 0.1), or a linear map.
  - It is fitted on val molecules of one `mol_id` parity, with early stopping on 10% of those
    molecules held out, and scored on the other parity; then the folds are swapped.
  - Every spectrum is scored by a correction that never saw its molecule.
  - Reported: held-out mean cosine (candidate space, tolerance-matched, as in step 1), top
    candidate agreement, and molecule-clustered 95% intervals.
- The context-only family is the baseline. Its gain is the calibration gain, which
  [step 1](#5-without-rank-selection-the-model-is-calibrated-in-aggregate)
  measured as +0.0052 for `P^γ`.

### Result

Held-out mean cosine gain over the raw 0.7878, and over the context-only baseline:

| family | MLP gain | MLP over baseline [95%] | linear over baseline [95%] |
| --- | ---: | ---: | ---: |
| context only | +0.0053 [+0.0050, +0.0057] | — | — (linear gain +0.0049) |
| existing | +0.0059 | +0.0005 [+0.0001, +0.0009] | +0.0002 [−0.0001, +0.0004] |
| post-cut connectivity | +0.0045 | −0.0008 [−0.0015, −0.0001] | −0.0004 [−0.0009, −0.0001] |
| charge | +0.0055 | +0.0002 [−0.0003, +0.0007] | +0.0001 [−0.0002, +0.0003] |
| post-cut + charge | +0.0035 | −0.0018 [−0.0026, −0.0010] | −0.0004 [−0.0009, −0.0001] |
| all | +0.0036 | −0.0017 [−0.0027, −0.0006] | −0.0005 [−0.0009, −0.0001] |

- **Top-candidate agreement does not move:** 0.614 raw, 0.614 for the baseline, and 0.610-0.615
  for every family across the three fits.
- **The inner early-stopping loss agrees.** It is the listwise cross-entropy on unseen molecules
  inside the fit half. The best inner loss is 2.3003 / 2.3382 for the baseline and
  2.2981 / 2.3474 for post-cut (the two folds). No family ends below the baseline.
- **Regularized MLP** (width 32, dropout 0.3, weight decay 1e-3). The baseline gain is
  +0.0049 [+0.0046, +0.0052]. Over it:
  - existing: +0.0008 [+0.0005, +0.0010];
  - post-cut: +0.0010 [+0.0005, +0.0015];
  - charge: +0.0009 [+0.0006, +0.0012];
  - post-cut + charge: +0.0006 [−0.0000, +0.0011];
  - all: +0.0006.

  The total gains, 0.0055-0.0059, sit within +0.0006 of the default MLP's baseline (+0.0053).

### Reading

- **The explicit description of the candidate after cutting carries almost no information `i1`
  fails to use on new molecules.**
  - The post-cut family covers retained connectivity, exposed cut sites and their chemistry,
    cation-stabilizing sites, bond classes and ion composition.
  - Learned on 2,100 val molecules, it predicts `i1`'s errors on the other 2,100 at most about
    0.001 better than `log p` and context do.
  - That small amount is not specific to post-cut connectivity: the eight basic attributes and
    the charge family reach the same.
  - A linear map and two MLP settings agree, so the null is not a fitting artifact.
- **Charge-site relations add no more,** which agrees with step 1: charge-side decisions are 1.5%
  of val top-two pairs.
- **What this does and does not say about step 3.**
  - The gate asks whether hand-built descriptors predict how `i1` errs on new molecules. A branch
    trained on training molecules faces a different problem: it must change what the model learns,
    where the model already fits the training molecules.
  - A gain here would have shown the information exists and transfers. The null says these
    descriptors do not carry it. A learned post-cut encoding could still find features they
    miss, but the prior for it is lower than before.
  - The screen is the direct test.

### Limits

- One checkpoint, one seed; the correction is fitted on 2,100 molecules per fold.
- Descriptors of peaks with several structures are averaged with the model's structure shares
  (median 2 structures per peak).
- The correction sees only the saved candidates (≥ 99.8% of predicted mass per spectrum).
- The objective is listwise cross-entropy; the reported metric is cosine.

### Artifacts

`output/postcut_gate_2026_09_29/` (Git-ignored):
- `run_gate.sh`;
- `i1_e059_val.npz`, 2.5 GB, with provenance;
- `fit*.csv`, `fit*.log` and per-spectrum tables.

The script's `--hidden 0` (linear) option and its Black formatting came after the extraction and
the default fit. The extraction path and the MLP are unchanged, so the provenance source hash of
the extraction differs from the file only by those edits.

Tests: `tests/test_debug_masster_postcut_information_gate.py` (8).

## Fragment-connectivity branch `fc1`: local screen against `i1` is not positive

> **Status:** FINDING · **Updated:** 2026-09-29
> Step 3 of the user's three-step plan: a local screen on the RTX 2070 SUPER, with two seeds per
> arm on a quarter of the molecules. It is not comparable to Fir numbers. The read rule was fixed
> before any `fc1` result was seen.
>
> - **The branch.** Each candidate gets one message-passing layer over its own retained bonds,
>   from raw atom and bond features plus two per-atom counts: its retained degree and the cut
>   bonds it keeps. It is pooled over all atoms and over the cut-site atoms, and added to the
>   intensity token as a zero-initialized residual.
>   - It costs 21,504 parameters, +12% peak memory at batch 128, about 3% of step time, and 2-3
>     min per 1-hour run.
>   - Default off; `i1` reproduces bit for bit.
> - **The primary read passes neither bar.** The val tail (epochs 19-23) gains +0.0088 and
>   +0.0013 at the two seeds, a mean of +0.0050. The rule required a mean above the control pair's
>   own spread, 0.0076.
> - **Calibrated cosine erases most of it.** After cross-fitted `P^γ`, `fc1` is +0.0025 and
>   −0.0009 (mean +0.0008).
>   - Base-peak hit: +0.0057 and −0.0044.
>   - Ranking gap: −0.0037 and −0.0008.
> - **The decisions it targets do not move consistently.** At novelty < 0.4 nested balanced
>   accuracy is +0.002 and −0.018. Overlapping pairs are +0.031 and −0.009.
> - **It lowers train fit slightly**, like a weak regularizer: the eval-mode gap narrows by 0.012
>   and 0.004.
> - **Per the rule, the matched pooled control `fc1p` was not run.** This agrees with steps 1 and
>   2: describing the candidate after cutting does not recover the transfer deficit, at this
>   screen's resolution of about 0.01.

### Question

Does a small branch that describes each candidate after cutting help `i1` on validation
molecules? It covers the retained-fragment connectivity and the newly exposed cut sites, and it
supplements the pooled precursor atom embeddings. The prior was low:

- [step 1](#wrong-base-peak-choices-of-i1-a-broad-transfer-deficit-not-one-missing-distinction) found a broad transfer deficit, not a
  missing distinction;
- [step 2](#post-cut-information-gate-on-i1-explicit-fragment-and-charge-descriptors-add-nothing) found hand-built post-cut
  descriptors worth at most about +0.001 cosine on held-out molecules;
- another session's
  retained-fragment plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_hierarchical_fragment_graph_plan.md`)
  deferred the branch on that evidence.

The user chose on 2026-09-29 to build and screen it anyway, as the direct test.

### The branch

Keys: `masster_fragment_connectivity_layers` (0 = off),
`masster_fragment_connectivity_hidden_dim` (32) and `masster_fragment_connectivity_messages`
(true). They are declared in `config/template.yml` and read by `MassterModel`
([`_fragment_connectivity_embeddings`](../../../src/fragnnet/model/masster_model.py)).

- **Graph.** Nodes are each candidate's (fragment, atom) membership rows. Edges are the precursor
  bonds with both endpoints in that fragment (`fragment_internal_edges` in
  [bond_fragment_heads.py](../../../src/fragnnet/model/bond_fragment_heads.py)). The same atom
  gets a different state in every candidate that cuts differently.
  - This differs from `r1_local_cut`, which was null: that arm re-sourced the cut endpoints from a
    k-hop view of the intact molecule.
- **Inputs.** Raw atom features (never the shared encoder's embeddings), plus the atom's retained
  degree in that fragment and the number of the fragment's cut bonds it keeps. Bond features come
  from `chemical_bond_attr`.
- **Encoder.** One plain GINE layer, width 32, with the model's precursor normalization and
  dropout.
- **Readout.**
  - Mean over all retained atoms, concatenated with the mean over cut-site atoms; the intact
    precursor gets zeros.
  - Projected to the token width and added after `input_projection`, before token dropout.
  - The projection is zero-initialized, so at step 0 the arm equals `i1` exactly (verified
    bit-identical in train and eval mode).
- **Matched control.** `messages: false` runs the same module, with the same parameters and
  features, on an empty edge set (`fc1p`).
- **Checks.**
  - 22 tests.
  - Against a brute force on real data (100,322 memberships), internal edges, degrees, counts and
    pools match exactly, including an atom in 136 fragments and 687 single-atom fragments.
  - With the new code, `i1` still reproduces Table 1 of step 1 exactly.
  - The 1,732 related MASSter tests pass. The full suite has one failure, in `test_runner_config`,
    from another session's in-progress relationship-loss code (`KeyError` on
    `masster_relationship_pairs_fp`), not this branch.
- **Cost.**
  - Parameters: 9,215,701 → 9,237,205.
  - At batch 128, peak memory 6.14 → 6.92 GB allocated (7.00 → 7.82 GB reserved).
  - Step time: the branch is about 13.8 ms of about 450 ms (join 3.2, GINE 7.3, pooling 3.3).
  - Wall time per 24-epoch run: 60-61 min, against 58 min for the uncontended control.
  - At batch 512 on Fir, the extrapolation is about +3 GB.
- **Arm configs:**
  - [`fc1_fragment_connectivity.yml`](../../../config/masster/mh_ft_memorization_route_2026_09_24/fc1_fragment_connectivity.yml);
  - [`fc1p_fragment_pooled.yml`](../../../config/masster/mh_ft_memorization_route_2026_09_24/fc1p_fragment_pooled.yml);
  - both `i1` plus the keys. Neither is in the Fir submitter.

### Setup

- **Recipe.** That of the [2026-09-27 local screens](../../history/masster_memorization_route_plan_log.md#local-screen-of-am1de1-2026-09-27):
  - a molecule-level subset of `nist20_inchikey`, 3,000 train and 1,000 val molecules;
  - `test_ids.csv` holds 1,000 train molecules, so the post-fit test pass is the eval-mode
    train fit;
  - batch 128, 24 epochs, on the relabelled `OB` / `OB_F` data.
- **Controls.** `i1_ctrl` (seed 397623) and `i1_ctrl_s2` (811237) were run fresh, because earlier
  local controls used `FT` labels. The arms are `fc1` and `fc1_s2` at the same seeds.
- **Probes.** Both base-peak probes ran on each run's best checkpoint, on the local val split and
  the train-molecule test split.
  - Novelty is taken against the 3,000 local train molecules, a mean nearest-neighbour Tanimoto of
    0.48.
  - Calibrated cosine is the tables' cross-fitted `P^γ`.

### Result

Training logs (val `cos_sim_0.01`; gap = best checkpoint's train-eval minus its val):

| run | tail 19-23 | tail 12-17 | best (epoch) | train-eval | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `i1_ctrl` | 0.6586 | 0.6546 | 0.6675 (21) | 0.8914 | 0.2239 |
| `fc1` | 0.6674 | 0.6603 | 0.6705 (23) | 0.8829 | 0.2123 |
| `i1_ctrl_s2` | 0.6662 | 0.6592 | 0.6702 (23) | 0.8954 | 0.2253 |
| `fc1_s2` | 0.6675 | 0.6626 | 0.6708 (23) | 0.8920 | 0.2212 |

- Paired tail 19-23: **+0.0088 and +0.0013**, mean +0.0050.
- Tail 12-17: +0.0057 and +0.0034.
- Gap: −0.0116 and −0.0041.
- The control pair's own tail spread is 0.0076.

Probes on the best checkpoints (val unless marked):

| read | `i1_ctrl` | `fc1` | `i1_ctrl_s2` | `fc1_s2` |
| --- | ---: | ---: | ---: | ---: |
| candidate-space cosine | 0.6877 | 0.6910 | 0.6909 | 0.6921 |
| calibrated cosine (cross-fitted `P^γ`) | 0.6939 | **0.6964** | 0.6980 | **0.6971** |
| ranking gap | 0.1696 | 0.1659 | 0.1653 | 0.1645 |
| base-peak hit | 0.488 | 0.494 | 0.498 | 0.494 |
| hit, covered, novelty < 0.4 / 0.4-0.6 | 0.419 / 0.523 | 0.424 / 0.535 | 0.423 / 0.535 | 0.425 / 0.529 |
| nested, deeper winner / shallower winner | 0.549 / 0.798 | 0.559 / 0.791 | 0.576 / 0.802 | 0.535 / 0.814 |
| overlapping / disjoint pairs | 0.614 / 0.658 | 0.645 / 0.659 | 0.630 / 0.645 | 0.621 / 0.667 |
| nested balanced, novelty < 0.4 / 0.4-0.6 | 0.632 / 0.670 | 0.634 / 0.684 | 0.643 / 0.691 | 0.625 / 0.678 |
| train-molecule cosine | 0.9249 | 0.9162 | 0.9286 | 0.9247 |
| train-molecule ranking gap | 0.0261 | 0.0289 | 0.0240 | 0.0255 |

### Reading

- **By the rule fixed in advance, the screen is not positive.**
  - Both seeds gain on the tail, but the mean (+0.0050) is below the control pair's spread
    (0.0076).
  - Calibrated cosine, the metric the plan names, gains +0.0008 on average.
  - The raw gain is mostly the flattening that cross-fitted `P^γ` already gives.
- **The decisions the branch aims at do not move consistently.** Nested pairs at low novelty,
  the largest failing population in step 1, are +0.002 and −0.018. Overlapping pairs are +0.031
  and −0.009. No read moves beyond the seed pair's spread in the same direction at both seeds.
- **What it does do: fit training molecules a little less.**
  - Train-molecule cosine falls by 0.009 and 0.004, and the train ranking gap rises.
  - The eval-mode gap narrows by 0.012 and 0.004, with val flat to up. That is `am1`'s pattern at
    a smaller size, which fits a weak regularizer better than a new description of the fragment.
- **The three steps agree.** `i1`'s wrong base-peak choices are a transfer failure spread across
  chemistry (step 1). Explicit post-cut descriptors do not predict them (step 2). A learned
  post-cut encoding does not fix them at this resolution (step 3).
- **The matched pooled control `fc1p` was not run.** The rule runs it only after a positive read.
  Its config and key exist.

### Limits

- The screen resolves about 0.01. The tail difference between the two control seeds is 0.0076, so
  a true +0.003-0.005 effect is invisible here.
- Settling that needs `fc1` and `i1` at two seeds each on Fir (64 epochs, batch 512, about +3 GB
  per run). That is not recommended on this evidence, and not submitted.
- One design point: one layer, width 32, raw-feature inputs. Wider, deeper or
  embedding-seeded variants were not tried. The other plan treats deeper layers as a later
  hypothesis, not an automatic response to a null.
- The controls' best checkpoints are epochs 21 and 23, and both arms' are 23. The probe reads
  compare best checkpoints, as the plan's local screens do.

### Artifacts

`output/local_postcut_branch_2026_09_29/` (Git-ignored):
- `run_one.sh`, `run_controls.sh`, `run_arms.sh`, `run_probes.sh`, `parse_logs.py`,
  `prep_meta.py` / `meta_local.pkl`, `logs/`;
- `probe/`: per-run probe outputs and tables.

Checkpoints: `ckpts/local_{i1_ctrl,i1_ctrl_s2,fc1,fc1_s2}/`.

Tests: `tests/test_masster_fragment_connectivity.py` (22).
