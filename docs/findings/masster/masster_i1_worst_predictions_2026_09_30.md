# `i1`'s worst val predictions: bin-edge artifacts, then a ranking failure of both heads on new molecules

> **Status:** FINDING · **Updated:** 2026-10-01 · Additive cut-probe follow-up linked below.
> Inference only: `i1_inst_ob_obf` (`i50ig7rf`) at epoch 59, all 52,020 val spectra and the
> 4,000-molecule train sample (49,205 spectra). Stored predictions from the
> [residual-structure](masster_i1_residual_structure_2026_09_28.md) and
> [base-peak](masster_i1_base_peak_errors_2026_09_28.md) reads, plus two local GPU re-runs
> (fragment-level internals; a larger candidate budget). The worst set is ranked by the
> instrument-tolerance cosine `cos_hun_inst_tol` (10 ppm, 200 Da floor), per the user
> (2026-09-30), not by the logged `cos_sim_0.01`.
>
> - **Most of the "worst" spectra under the logged metric are good predictions split by a 0.01 Da
>   bin edge.**
>   - 497 of the 903 val spectra with `cos_sim_0.01` < 0.05 score ≥ 0.5 at instrument tolerance,
>     and so do 76 of the worst 100.
>   - The measured peak sits 1–4 ppm from the exact candidate m/z.
>   - The train sample has as many (856 against 854), and the loss never sees the bins.
> - **At instrument tolerance the floor (< 0.05) is 330 spectra of 119 molecules; the train sample
>   has 4.**
>   - The tail (< 0.2) is 36.0 per 1,000 spectra, against 0.87 on train.
>   - Ranking failures with the candidates present are 32.9 against 0.06 per 1,000.
>   - The tail is 15% of the whole-val inst-tol deficit.
> - **What is missing, by size:**
>   - **A ranking of the measured cut that transfers.** In 76% of floor spectra with candidates, the
>     measured base-peak fragment is outside the bond head's top 3 (median proposal rank 20 of
>     ~250). The intensity head ranks it low as well (median 13), and its top pick is right in 5%.
>     In `i1` the two heads score independently and share only the encoder. The bond head, trained
>     only by a 0.1-weighted m/z-coverage loss on detached inputs, picks shallower fragments than
>     measured.
>   - **Molecule-specific fragmentation that analogues do not show.** Floor targets depart from
>     their nearest train analogues 3–7× more often than train molecules at the same Tanimoto,
>     while `i1`'s prediction resembles the analogues. About 20% are transfer failures: a top-20
>     analogue shows the dominant fragment. In about 45%, no similar train molecule shows the
>     pattern.
>   - **Candidates, for 12% of the floor (40 spectra).** Of the uncovered intensity:
>     - about 50% needs a rearrangement no bond cut gives;
>     - 18% is a selection miss that a 4× budget recovers;
>     - 9% needs more than ±4 H or a radical ion;
>     - 23% does not come from this molecule.
> - **Not missing:**
>   - **A conditioning input.** Nitro and sulfonamide molecules fail twice as often, but atom
>     charges are already inputs and these groups are common in train.
>   - **The energy scale.** A same-molecule rung shift rescues 5.6%, the same rate as on good
>     spectra.
>   - **Clean data.** Bad targets are about 3% of the floor: the D5 siloxane and sucrose.
> - **The failures belong to the molecule under a condition, not to one checkpoint.**
>   - `c0` fails on the same molecules (molecule-level r = 0.77).
>   - Yet a floor molecule fails on a median 17% of its own spectra, and its measured spectrum one
>     NCE step away matches the floor target at cosine 0.975.
> - **Follow-up (2026-09-30).**
>   - **Not the encoder.** Classifiers on the frozen encoder embeddings predict cut sites on new
>     molecules as well as `i1`'s bond head, and better on the floor (AUC 0.79 against 0.73). So
>     the failure is the heads' use of the representation and multi-cut composition.
>   - **No tested arm fixes the floor.** On a floor chosen by an independent checkpoint, `fa1`
>     (full attention) and `i3` (EMA + dropout) are level with `i1`. They raise whole val by +0.006
>     and +0.010 beyond `i1`@63; averaging `i1`'s epoch-59 and -63 spectra raises it by +0.014.
>   - **About half the floor set changes between `i1`'s epochs 59 and 63.**
>   - **Keep selection-only.** Adding the bond head's scores to the final ranking lowers the
>     cosine at every strength, on the floor as well (−0.007 at α = 0.5).
>   - **Nitro and sulfonyl failures run through the group's own heteroatom bonds** (N–O; S–N,
>     S–C, S–O).
>     - Their measured base peaks are often radical cations. The model over-predicts radicals for
>       nitro (51% of its picks against 28% measured) and under-predicts them for sulfones (10%
>       against 32%).
>     - 29% of sulfonamide tail spectra lack the candidate: SO2 extrusion and S–O
>       rearrangements.
>   - **Remaining levers**, none built:
>     - a training signal phrased as transferable cut preferences that reaches the intensity
>       head;
>     - more varied training molecules.
>
>     See [Next steps](#next-steps-not-built).

## Question

The user (2026-09-30): check `i1`'s predictions and, for the worst ones, find what is missing.
Mid-analysis the user set the ranking metric to instrument tolerance instead of 0.01 Da bins.

## Method

- **Inputs.** `output/i1_worst_predictions_2026_09_30/prepare.py` joins the existing per-spectrum
  artifacts into compact tables (docstring lists the sources):
  - predictions and targets from `collect_masster_preds.py`;
  - support bounds, base-peak outcomes and top-3 peak structures;
  - campaign fields, novelty (max Morgan-r2 Tanimoto to train), SMILES and raw peaks.
- **Metric.** [`score_inst_tol.py`](../../../output/i1_worst_predictions_2026_09_30/inst_tol/score_inst_tol.py)
  recomputes `cos_hun_inst_tol` per spectrum with the repo's own functions
  (`round_aggregate_peaks`, `calculate_match_mzs`, `batch_cos_hun_helper`):
  - m/z rounded to 4 decimals, intensities summed (`sum_ints: True`);
  - tolerance 1e-5 · max(m/z, 200) per target peak (OB/OB_F `inst_type_loss_tol`);
  - one-to-one assignment, then cosine.
  - Val 0.7880 (logged binned 0.7622); train sample 0.9589 (0.9245).
- **Classes** (`cls3`, [`summary_tables.py`](../../../output/i1_worst_predictions_2026_09_30/inst_tol/summary_tables.py)):
  - `M_metric`: logged cos < 0.2 but inst-tol ≥ 0.5.
  - `U_uncovered`: inst-tol < 0.2 and most target norm has no candidate within 10 ppm
    (`cos_support_tol` < 0.5).
  - `R_ranking`: inst-tol < 0.2 otherwise, i.e. candidates present.
  - Floor < 0.05, tail < 0.2.
- **The base-peak table's nearest-candidate cosine is not used for ranking.** It agrees on 329 of
  330 floor spectra, but it puts 25 spectra near zero that the inst-tol and binned cosines score at
  0.88–0.97. The nearest candidate to a target peak can be a zero-share isobaric candidate while the
  model's mass sits on another candidate within tolerance. Example: train mol 32385 at NCE 95–180
  reads 0.10–0.17 against 0.97–0.995.
  - Some verification reads used that cosine. Their populations differ from the inst-tol floor by at
    most 25 spectra, and the model-internals read is recomputed on the inst-tol floor.
- **Procedure.** Eight parallel analyses, each testing one explanation. Then eight adversarial
  re-derivations on the corrected floor: independent code, a completeness critic, and two GPU
  re-runs under a lock. Intervals are 95% molecule-clustered bootstraps.

## Result: the logged-metric floor is mostly a binning artifact

| worst set by logged `cos_sim_0.01` | spectra | inst-tol ≥ 0.5 | inst-tol < 0.05 |
| --- | ---: | ---: | ---: |
| cos < 0.05 | 903 | 497 (55%) | 327 |
| worst 500 | 500 | 354 (71%) | 121 |
| worst 100 | 100 | 76 | 23 |

- **Mechanism.** The measured peak sits a mean −0.34 mDa from the exact candidate m/z: median
  −2.3 ppm, negative in 73% of spectra.
  - Across an edge of the 0.01 Da floor bins, the binned cosine sees no overlap.
  - The model's precursor candidate equals theoretical [M+H]+ to 5e-7 Da, and the `prec_mz` header
    agrees to 0.03 mDa. Only the measured peak list is offset.
  - Offsets are larger at low NCE (−0.49 mDa at ≤ 20). They correlate with the spectrum's
    `mzdiff_ppm` field (r = 0.30).
- **The support bound can hide it.** Spec 575913 has its target peak at 146.05969 (bin 14605) and
  the model's 27% candidate at 146.06004 (−2.4 ppm, bin 14606). An unrelated zero-share candidate
  in bin 14605 makes that bin count as covered.
- **The two sides use different bin grids, a real defect but a minor one.** Edges follow each side's
  dtype (`spec_utils.py:565-571`): targets are float32, predictions float64. One float64 grid for
  both sides would resolve 51 of 850 artifact spectra (6%) and move whole-val by +0.0001. The
  logged cosine is reproduced exactly (0 of 1,137 sampled spectra off by more than 1e-6).
- **Metric only.** The rate is the same on train (0.9×). The loss is a tolerance Gaussian
  cross-entropy with every bin-based term at weight 0. The bins reach only checkpoint selection
  (`val_cos_sim_0.01_epoch/mean`). Snapping each target peak to its matched candidate raises the
  whole-val binned cosine by about +0.033.

## Result: the real floor is a generalization failure

| per 1,000 spectra [95% CI] | val (spectra / molecules) | train sample | ratio |
| --- | --- | --- | ---: |
| `M_metric` | 16.4 [14.3, 18.8] (854 / 312) | 17.4 [15.1, 19.8] (856) | 0.9× |
| `U_uncovered` | 3.1 [2.2, 4.1] (162 / 71) | 0.8 [0.4, 1.3] (40) | 3.8× |
| `R_ranking` | 32.9 [29.7, 36.5] (1,710 / 518) | 0.06 [0, 0.14] (3) | ~540× |
| tail (< 0.2) | 36.0 [32.4, 39.6] (1,872 / 561) | 0.87 [0.51, 1.32] (43) | 41× |
| floor (< 0.05) | 6.3 [4.9, 7.8] (330 / 119) | 0.08 [0, 0.22] (4) | ~80× |

- **Where the tail falls.**
  - By novelty: 87 per 1,000 below Tanimoto 0.4, falling to 13 at ≥ 0.8.
  - By NCE: peaks at 40–60 (52 per 1,000), with 25 at ≤ 20 and 22 above 80.
  - By instrument: Lumos 40, Elite 35.
- **Novelty does not explain the train gap.** Restricting val to train's 5–95% novelty band barely
  moves the ranking-failure rate (31.3 against 33.5 per 1,000).
- **The floor is spread thin.**
  - A floor molecule fails on a median 17% of its own val spectra; 2 of 119 fail on all.
  - Its measured target one NCE step away matches the floor target at 0.975 (random-molecule null
    0.04).
  - The targets are smooth along the ladder. Adjacent-rung self-consistency equals train's (0.924
    against 0.928).
- **Stable across checkpoints.** `c0` also fails on `i1`'s floor (binned cos mean 0.13, against
  0.78 on a random control) with candidate coverage intact. Across all val molecules, `i1`'s and
  `c0`'s molecule-mean scores correlate at r = 0.77.

## Result: where in the model the floor fails

Fragment-level tensors from a direct `MassterModel` forward: 289 of the 290 inst-tol floor spectra
with candidates (105 molecules; the target fragment is found in 275, the base of the percentages),
against 293 NCE-matched good spectra (cos ≥ 0.6)
([`01_extract.py`](../../../output/i1_worst_predictions_2026_09_30/verify_component/01_extract.py),
[re-read](../../../output/i1_worst_predictions_2026_09_30/inst_tol/component_on_inst_tol_floor.log)).
The "target" is the fragment behind the measured base peak.

**How `i1` wires the two heads** (`masster_proposal_selection_only: true`):
- The bond head only selects the ~256 candidates. Its probabilities do not enter the final
  distribution: the final fragment logits are the intensity head's residuals alone
  (`masster_model.py:3642-3646`).
- The bond head's inputs are detached from the shared encoder (`detach_proposal_inputs`,
  `masster_model.py:3033`). It is trained only by the proposal coverage loss at weight 0.1, which
  credits every candidate matching a peak's m/z.
- The shared encoder is therefore trained only through the intensity head.
- A fragment's proposal score is a product of independent per-bond break probabilities
  (`masster_candidates.py:500-539`). The bond head sees no fragment-level size, depth or ring
  descriptor; those reach only the intensity head.

| | floor | control |
| --- | ---: | ---: |
| target not in the bond head's top 3 (proposal rank > 3) | 76.0% (median rank 20) | 17.1% (median 11) |
| target in the bond head's top 3, but ranked lower by the intensity head | 18.2% | 14.0% |
| the model's top channel is the target | 5.1% | 74.3% |
| intensity head's #1 fragment = bond head's #1 | 46.2% | 66.4% |
| target's H shift: median share of its fragment's mass | 0.48 | 0.999 |
| target's H shift is not its fragment's top H shift | 45% | 5% |

- **Both heads rank the target low, independently.**
  - The bond head's median rank for it is 20 and the intensity head's 13; the two ranks correlate
    at Spearman 0.71.
  - Since the final distribution does not inherit the proposal score, the agreement comes from two
    heads learning similar preferences. The only component they share is the encoder, and the
    bond head reads it detached.
  - The intensity head disagrees with the bond head's #1 more often on the floor than on control,
    but rarely in favour of the target.
- **The bond head picks shallower fragments on the floor**
  ([log](../../../output/i1_worst_predictions_2026_09_30/fix_bondhead/01_bondhead_read.log)).
  - Its top fragment needs 1.75 cuts against the target's 2.25; on control, 1.54 against 1.62.
  - It cuts a ring bond in 27% against the target's 39%; on control, 24% against 25%.
  - The target's proposal rank correlates with its cut count at r = 0.35 on the floor. This is the
    direction the per-bond product favours on a molecule with no memorized per-bond preference.
- **The H-shift split is suggestive, not established.** The target fragment carries little mass on
  the floor, so its internal H split is weakly determined.
  - A clear case: mol 1580 has the right cut, but the model predicts 200.083 against the measured
    201.091, one H short.
  - The H-shift channels are the ±4 ladder; `i1` has one ion mode (`masster_include_cmf: false`).
- **Every diagnostic is continuous in the inst-tol cosine**: top share, fragment-only cosine,
  depth gaps, coverage and the target's rank. There is no break at 0.05 or 0.2, so the floor is the
  far end of the ordinary ranking deficit, not a separate mode.
  - The model's top share is U-shaped: about 0.51 at the floor, 0.33 at 0.3–0.8 and 0.64 at the
    top.
- **The depth bias is small.** Against controls matched on instrument × NCE decile × true-precursor
  quintile, a residual of +0.04 to +0.05 in precursor fraction remains (CI excludes 0). About a
  third of the raw gap is where floor spectra sit on the ladder.
- **Cut classes all degrade.** On ranking failures, the model's top cut matches the measured
  base-peak cut class in:
  - 2–11% of acyl, alkyl and benzylic C–C cases (43–80 spectra, 17–23 molecules). The base-peak
    table's val-wide hit rates for alkyl and benzylic C–C base peaks are 0.37–0.44.
  - 25% of amide C–N cases (36 spectra, 17 molecules), against a val-wide 0.62.

## Result: the floor departs from its analogues

Top-20 Morgan-r2 train neighbours; a tolerance-clustered neutral-loss and fragment cosine
([`verify_analogues/`](../../../output/i1_worst_predictions_2026_09_30/verify_analogues/)).

- **Floor molecules are less familiar, but most have an analogue.** Mean novelty is 0.50 against
  0.59 for val, and 31% sit below 0.4.
- **Target–analogue resemblance on the floor is lower at every similarity level.** Against
  control, it is 0.32–0.83 lower in every nearest-neighbour Tanimoto bin (CIs exclude 0).
- **`i1`'s prediction resembles the analogues** at 0.758, against 0.136 for 30 unrelated
  molecules. The floor targets reach only 0.247. Predictions are also flatter than targets (log
  effective peaks +0.22, against −0.04 on control), but that explains little of the gap.
- **Train negative control.** At Tanimoto 0.3–0.7, where floor molecules sit, 7–20% of train
  molecules depart from their own nearest analogues, against 50–64% of the floor. Train molecules
  that do depart are still predicted well (median 0.97), by memorization.
- **Split of the floor, at 20 neighbours:**

| category | spectra [95% CI] |
| --- | --- |
| transfer failure: a top-20 analogue shows the dominant fragment or loss | 19.8% [13.7, 26.4] |
| the pattern appears in a less similar train molecule (Tanimoto ≥ 0.3) | 4.8% [1.7, 9.0] |
| no similar train molecule shows the pattern | 44.9% [36.7, 53.5] |
| unresolved (target and prediction equally related to the pool) | 30.5% [22.8, 38.3] |

  - **The rule.** "Transfer failure": the target resembles one of its 20 nearest analogues' spectra
    above 0.30 (tolerance cosine in fragment or neutral-loss space), and beats the prediction by
    more than 0.10. "No similar molecule shows it": the target resembles none of the 20 above 0.30,
    and no train molecule at Tanimoto ≥ 0.3 has the target's base-peak ion or loss as its own base
    peak. Matching is by mass, on the base peak only.
  - This is about the fragmentation, not the molecule. These molecules have analogues about as
    close as other val molecules do (mean novelty 0.50); those analogues fragment differently.
  - The transfer-failure rate is stable from 5 to 20 neighbours. A threshold sweep keeps it at
    12–21%.
- **A mass-only "present somewhere in train" check is uninformative.**
  - A random mass also matches 32–51% of all 5.6 M train peaks, and 9–11% of train base peaks.
  - Matching train molecules have a median Tanimoto of 0.185 to the query.

## Result: chemistry

The measure is the share of molecules with at least one inst-tol tail spectrum
([log](../../../output/i1_worst_predictions_2026_09_30/inst_tol/chem_rates_inst_tol.log)). The
three siloxane/sucrose molecules are excluded.

| group | molecules | tail rate | rest of val | ratio |
| --- | ---: | ---: | ---: | ---: |
| sulfonamide | 174 | 29.3% | 12.5% | 2.3× |
| sulfonyl | 241 | 26.6% | 12.4% | 2.1× |
| nitro | 197 | 25.9% | 12.6% | 2.1× |

- **Nitro and sulfonamide/sulfonyl survive** adjustment for novelty and size (logistic, p < 1e-7
  on the tail). Nitro survives on the floor too.
- **They cover a minority.** These groups are in 10% of val molecules and 20% of tail molecules
  (114 of 561). Half of the tail carries none of the 12 flags tested.
- **Train has them in proportion**: 545 sulfonamide and 572 nitro molecules.
- **The floor is not one ClassyFire class.** Its 119 molecules span 11 superclasses, 42 classes and
  100 direct parents, as many as random draws of 119 val molecules give. No superclass or class is
  enriched. Only benzenesulfonamides (direct parent) are: 9 floor molecules, 4.1× their val share
  (BH q = 0.03); in the tail 25, 2.4×
  ([log](../../../output/i1_worst_predictions_2026_09_30/classes/floor_classes.log)).
- **Duplicates and non-survivors.**
  - The `[NX3+][OX1-]` "N-oxide" pattern matches every nitro group (197 of 200 molecules).
  - 212 of 245 "charged" molecules carry charge only through nitro or N-oxide, and the rest is not
    enriched.
  - Azo, carbamate, fused rings, esters and organosilicon do not survive adjustment or the
    contaminant exclusion.
- **Not a missing input.** `i1`'s node features already include formal charge (`a_formal`) and
  Gasteiger charge (`a_charge`).
- **The mechanism reading is partial.**
  - In tail spectra of these groups, the measured base peak is an S–N, S–O, N–O, N–N or P–O
    cleavage in 19.5% of cases, against 11.7% in their good spectra.
  - When it is, the model's top pick is also a heteroatom–heteroatom cleavage in 51%.
  - This matches the base-peak table's N–O/N–N hit rates (0.22–0.26). It covers about a fifth of
    these failures.

### Why the nitro and sulfonyl groups fail (follow-up, 2026-09-30)

These are the measured base peak and the model's top peak for these groups' tail spectra (inst-tol
< 0.2) against their good spectra (≥ 0.6). Contaminants are excluded, and the structure is the
model's dominant candidate at that m/z
([log](../../../output/i1_worst_predictions_2026_09_30/sn_groups/sn_groups.log)).

| group (molecules) | tail per 1,000 | missing candidates in tail | het–het cut, tail / good | odd-electron base peak, tail (model's pick) | base-peak cuts, tail (model's pick) |
| --- | ---: | ---: | --- | --- | --- |
| nitro (197) | 74.0 | 1 of 183 | 0.25 / 0.07 | 0.28 (0.51) | 2.37 (1.52) |
| sulfonamide (162) | 58.0 | 36 of 125 | 0.41 / 0.21 | 0.15 (0.10) | 1.90 (1.40) |
| sulfone (74) | 73.4 | 8 of 68 | 0.22 / 0.11 | 0.32 (0.10) | 2.18 (2.04) |
| other (3,786) | 31.8 | 99 of 1,476 | 0.05 / 0.01 | 0.12 (0.11) | 2.23 (1.67) |

- **Their bad spectra are the ones where fragmentation runs through the group's heteroatom bonds.**
  - Nitro: the N–O bonds.
  - Sulfonyl: S–N, S–C and S–O. The measured base peak cuts an S–O bond in 22% of sulfonamide
    tail spectra, and the model's top pick almost never does.
- **Many of these base peaks are radical cations, and the model gets the electron parity wrong in
  both directions.**
  - For nitro compounds it predicts an odd-electron ion in 51% of tail spectra against 28%
    measured. It over-applies the radical NO•/NO2• losses.
  - For sulfones it predicts 10% against 32% measured.
- **Nitro failures are deep fragmentation.** Half of the measured base peaks also cut an aromatic
  ring C–C bond (model 17%), at 2.37 cuts against the model's 1.52.
- **Sulfonyl failures also lack candidates.** 29% of sulfonamide tail spectra are uncovered (other
  molecules 7%), with mean uncovered intensity 0.30 against 0.13. These are SO2-extrusion and S–O
  rearrangement products that no bond cut produces (see the next section).
- **Near misses.** The model's top peak is the measured base peak in 8.8% of sulfonamide tail
  spectra, against under 1.5% elsewhere: the right region, the wrong side or H count.
- **Reading.** These groups fragment through a few energy-sensitive pathways: nitro–nitrite
  rearrangement and radical losses; S–N cleavage against SO2 extrusion. Which of these wins depends
  on neighbouring substituents. Train holds enough of them to memorize each molecule (545
  sulfonamide and 572 nitro molecules), but the model does not learn the choice as a rule that
  carries over.

**What is actionable in it** ([log](../../../output/i1_worst_predictions_2026_09_30/sn_groups/sn_levers.log)):
- **Parity errors mostly come from picking a different fragment, not a different H.**
  - The model's top has the right atoms with the wrong H in only 3–6% of these groups' tail
    spectra.
  - Nitro: in 50% the model's pick is a larger fragment containing the measured one, so it stops
    early (other molecules 35%).
  - Sulfones: 43% cut a different bond.
- **Parity is learned per molecule, not as a rule.** Radical/even-electron agreement between the
  measured base peak and the model's top, train against val: nitro 94% / 80%, sulfone 96% / 87%,
  sulfonamide 96% / 89%, other 97% / 93%.
- **`i1` has no electron-parity input.**
  - Its candidate features are raw cut count, log size, log mass and summed Gasteiger charge.
    `c_cut_bonds`, which carries the cut bonds' order, ring and conjugation, exists but is off.
  - The H shift is an output slot (−4 to +4) of the intensity head, not an input.
  - On 8,705 single-structure candidates the ion is odd-electron exactly when cut count + H
    shift is odd in 92.6% of cases; the exceptions are double, aromatic and ring cuts. So the
    head must learn an alternating slot pattern plus bond-order exceptions it cannot see, and it
    can memorize the right slot per molecule instead.
- **Sulfonamides' missing ions are one rearrangement family.**
  - 59% of their uncovered intensity (117 peaks, 21 molecules) is ions that lost the S and some of
    the O.
  - The implied losses are X–N–S–O units: HNOS 16%, C3H7NOS 6%, CH3BrNOS 5%, C2H9N2OS 4%, H3NOS
    and HBrNOS 2–3%. This is the sulfonyl O-transfer (aryl keeps one O; R2N–S=O leaves), as in
    sulfanilamide's 156 → 108.
  - The registry has `primary_aryl_sulfonamide_o_transfer` (`[M+H]+`, gate passed), but not the
    secondary, tertiary or non-aryl cases, and `i1` runs with templates off.
  - 28% of the intensity has no composition within 10 ppm.

**Two follow-up tests (2026-09-30).**

*Odd-electron bias, offline* ([`parity/`](../../../output/i1_worst_predictions_2026_09_30/parity/)).
- **Setup.** One `i1`@59 forward over all 52,020 val spectra, with each channel's exact electron
  parity from its member atoms. Each channel's m/z, rebuilt from atoms, own H and H shift, matches
  the model's to 2e-8 Da.
- **Bias.** β × odd-electron is added to the channel logits: on all odd channels, only on
  fragments cutting a het–het bond, or only on the others. β is fitted per group on half the
  molecules and scored on the other half.
- **Null.** The parity labels are permuted within each spectrum.
- **Result: null.**
  - Whole val +0.0000 [−0.0000, +0.0001].
  - Nitro +0.0008 [−0.0007, +0.0024].
  - Sulfonamide and sulfone 0. The fitted β is 0 or −0.25 almost everywhere.
  - The one positive cell, nitro het–het on its 183 tail spectra (+0.0057 [0.0014, 0.0121]), is
    on a set selected by `i1`@59.
- **Why.** The model's odd-electron mass is already calibrated per group: predicted 0.30 against
  a measured base-peak rate of 0.25 (nitro), 0.17 against 0.16 (sulfonamide), 0.17 against 0.18
  (sulfone). So no group-level bias can help. The error is which candidate carries the parity in
  each spectrum, the same "calibrated in aggregate, wrong per spectrum" pattern as the base-peak
  table.
- **Not tested.** Whether a parity input would help the model rank candidates within a spectrum.
  That needs training.

*Generalized sulfonyl O-transfer, model-free gate*
([`smarts_o_transfer/`](../../../output/i1_worst_predictions_2026_09_30/smarts_o_transfer/)).
- **Setup.** The product is the carbon side of the C–S bond plus one sulfonyl O. Scoring uses
  `debug_masster_smarts_rule_gate.py`'s channel scorer, its rule-only criterion and its random
  disconnected null, on val and train-sample `[M+H]+` spectra.

| rule | val molecules | rule-only intensity over null, val [95% CI] | train sample |
| --- | ---: | --- | --- |
| registered `primary_aryl_sulfonamide_o_transfer` | 30 | +1.61% [0.75, 2.64] | +1.96% |
| aryl sulfonamide O-transfer, any N | 145 | +1.86% [0.94, 2.68] | +2.21% [1.62, 2.89] |
| alkyl sulfonamide O-transfer | 18 | +1.67% [0.00, 5.02] | 0.00% |
| aryl sulfone O-transfer | 25 | +0.31% [0.03, 0.69] | +0.63% |

- **The any-N aryl rule replicates on train** and reaches far more of the bad spectra than the
  registered primary-only rule.
  - On the 151 sulfonyl tail spectra: rule-only share 6.4% against 0.9%, and 26 against 4 spectra
    at ≥ 10%.
  - On the 37 sulfonyl spectra missing candidates: 19.7% against 2.1%, and 17 against 3 spectra.
- **The alkyl and sulfone variants are too weak to admit.** Four val molecules carry the alkyl
  signal, and it is zero on train.
- **Registered (2026-09-30)** as `substituted_aryl_sulfonamide_o_transfer`: secondary and
  tertiary aryl sulfonamides, `[M+H]+`, appended as vocabulary index 23. The primary rule keeps the
  NH2 case.
  - Its product needs the new `NeutralLossRule.nl_side_map_nums`, where the whole N side leaves;
    matches whose N side is fused to the aryl ring are skipped.
  - On the molecules the primary rule does not match: val 115 molecules +1.81% [0.70, 2.80], train
    sample 104 +2.33% [1.64, 3.20]
    ([catalog](../../reference/masster_smarts_rules.md#rule-added-from-the-i1-worst-prediction-read-2026-09-30)).
  - Templates stay off by default. Its effect is confined to about 150 sulfonyl tail spectra, so
    read a templates-on arm on that subset, not on whole val.

## Result: uncovered candidates

`U_uncovered` is 162 spectra of 71 molecules, and 40 spectra of the floor. Every uncovered peak of
at least 5% of its base was classified (1,304 peaks, 92.5% of the uncovered intensity), with all
compositions within 10 ppm and a connected-subgraph search under `i1`'s H bound (±4, valence)
([`verify_uncovered/`](../../../output/i1_worst_predictions_2026_09_30/verify_uncovered/)).

| cause, by intensity | share [95% CI] | what would reach it |
| --- | --- | --- |
| rearrangement: no connected atom set has the composition | 49.6% [35.3, 63.4] | a mechanism rule |
| selection miss: connected, H-feasible, not proposed | 18.3% [10.0, 27.6] | a larger mask budget |
| more than ±4 H, or a radical ion | 9.3% [3.9, 16.5] | a wider H ladder |
| not this molecule: isotope, above precursor, background | 22.8% [13.4, 34.7] | nothing |

- **Rearrangements with no rule anywhere.** Phosphonate → H4PO4⁺ (e.g. spec 52257, m/z 98.9843,
  86% of the spectrum) and siloxane methyl migration have no rule in the vocabulary or the
  literature backlog. The registered sulfonamide SO2 rules reach 0.66% of this class's uncovered
  intensity. This is consistent with the
  [SMARTS outlier read](masster_smarts_outlier_reach_2026_09_29.md).
- **Selection misses are recoverable.**
  - Raising `masster_unique_fragment_target` / `masster_ranked_mask_budget` from 256 / 4096 to
    1024 / 16384 lifts candidate coverage on 9 sampled selection-miss spectra from 0.23 to 0.71
    (all 9 improve). The rest of the subset moves from 0.79 to 0.82.
  - Mask ranking takes the NCE through the context encoder, so the selection is CE-dependent.
- **On the floor subset** the mix shifts slightly toward selection misses (22%) and away from the H
  limit (5%).

## Result: not the energy scale, not the data

- **Energy.** Scoring a floor target against the model's prediction for the same molecule at
  another measured NCE rescues 5.6% [2.8, 9.3] of spectra. On good spectra of the same molecules
  it is 5.5%.
  - The model's top m/z stays the same between adjacent rungs in 71% of pairs for floor molecules
    and 72% for others.
- **Data.**
  - Precursor headers match theoretical [M+H]+ within 0.22 ppm everywhere.
  - No two molecules share a peak list.
  - Bad targets are 12 of the 330 floor spectra (3.6%). These are the D5 siloxane (7) and sucrose
    (5: a 342.2–343.2 Da cluster with no peak at [M+H]+). D4 is not in the floor; in the tail the
    three molecules hold 20 of 1,872 spectra.
  - Also flagged:
    - glycosides fragment completely at NCE ≤ 10 (expected chemistry);
    - 4 floor spectra have at most 2 peaks, which makes the cosine fragile.
- **One possible mislabelled structure, not checked further.** Spec 399730 is labelled
  N-(4-nitrobenzyl)benzamide.
  - `i1` predicts m/z 105 (benzoyl) and 136 (nitrobenzyl), the ions that structure gives.
  - The measured spectrum is 94% m/z 91, like its train regio-isomer N-benzyl-4-nitrobenzamide.

## Result: encoder or head (follow-up, 2026-09-30)

The user asked whether a bond-head failure means the GNN encoder is at fault. Two tests on `i1`@59,
each re-derived by an adversarial check
([`enc_trainval/`](../../../output/i1_worst_predictions_2026_09_30/enc_trainval/),
[`enc_probe/`](../../../output/i1_worst_predictions_2026_09_30/enc_probe/)).

**The encoder carries transferable cut information; the bond head does not use it on the floor.**
- **Setup.** Simple classifiers predict which bonds the measured base-peak fragment cuts. They are
  trained on train-sample molecules and scored on held-out val molecules. Inputs are either the
  frozen, context-free encoder atom embeddings (128 PCA components of the 384-d GINE_GSA output) or
  hand-made bond features, plus NCE and instrument.
- **Labels.** The fragment's cut bonds, on spectra whose base peak is a fragment with one dominant
  structure: 31,117 val and 30,721 train spectra.

| bond-level AUC [95% CI] | val (31,117 spectra) | val floor (198 spectra, 87 molecules) |
| --- | ---: | ---: |
| `i1`'s own bond head (`bond_break_logits`) | 0.901 [0.898, 0.904] | 0.733 [0.675, 0.790] |
| classifier on frozen encoder embeddings (GBM) | 0.895 | 0.789 |
| classifier on hand-made bond features (GBM) | 0.878 | 0.822 |
| both (GBM) | 0.896 | 0.805 |
| bond-class base rate | 0.739 | 0.671 |

- **On ordinary spectra the encoder is not the limit.** Embedding classifiers beat hand-made
  features in every novelty band, and `i1`'s head matches them. The head's own train-to-val gap is
  small (0.926 in-sample against 0.901).
- **On the floor the head is the worst non-trivial scorer.** All six classifier variants on the same
  frozen inputs beat it, 4 of 5 significantly on a paired molecule-clustered bootstrap. The
  information is there; the head, trained by the 0.1-weighted m/z-coverage loss on detached inputs,
  does not extract it for these molecules.
- **Caveat in the head's favour.** The classifiers are fitted to exactly this label, while the head
  is trained to cover all peaks.
- **The floor is harder for every scorer** (AUC 0.87–0.90 drops to 0.73–0.82).
- **Untested for the deepest floor spectra.** The single-structure filter drops 130 of the 330
  floor spectra, and those are deeper (mean 2.5 against 2.0 cuts) and at higher NCE (57 against
  46).

**Test A: the bond head also ranks worse on val than on train at every depth.** A sample matched on
instrument × NCE decile × depth, 800 spectra per split, with no molecule overlap by InChIKey:

| | train | val |
| --- | ---: | ---: |
| measured fragment in the bond head's top 3 | 80.2% | 66.0% |
| ranked first by the bond head | 61.3% | 45.4% |
| ranked first by the intensity head (final) | 68.8% | 48.2% |
| bond head misses (rank > 3) but final rank 1 | 4.6% | 3.0% |

- **Depth.** Top-3 rates by the measured fragment's cut count are 92/89% (one cut), 81/63% (two)
  and 68/46% (three or more).
- **One-cut fragments.** The top-3 gap is not significant there, but rank-1 differs by
  +12.6 points [5.3, 20.0] and final rank-1 by +14.1 [7.3, 21.1].
- **Both heads carry train-specific fit.** The intensity head seldom rescues a bond-head miss, on
  either split. When the bond head does have the fragment in its top 3, the intensity head puts it
  first 92% of the time on train against 80% on val (one-cut spectra).
- **Reading.** Per-bond discrimination transfers on ordinary spectra. What fails on new molecules
  is:
  - the bond head's scoring on the hardest molecules;
  - the composition of per-bond scores into multi-cut fragments;
  - the intensity head's final choice.

  None of these is a missing encoder representation.

## Result: `fa1` on the floor (follow-up, 2026-09-30)

`fa1` (`0vn8m2gr`) is `i1` plus full intensity-transformer attention; its best epoch is 63, logged
0.7674, reproduced to 7e-7. It is compared on `i1`@59's floor with `i1`@63 as the control:
selecting the worst spectra of one checkpoint guarantees any other checkpoint looks better there
([`fa1/summary.md`](../../../output/i1_worst_predictions_2026_09_30/fa1/summary.md), checked in
`fa1/check/`).

| inst-tol cosine | `i1`@59 | `i1`@63 | `fa1`@63 | `fa1`@63 − `i1`@63 |
| --- | ---: | ---: | ---: | --- |
| whole val (52,020) | 0.7880 | 0.7874 | 0.7934 | +0.0060 [0.0036, 0.0082] |
| train sample (49,205) | 0.9589 | 0.9581 | 0.9576 | −0.0005 |
| `i1`@59 floor (330) | 0.028 | 0.094 | 0.117 | +0.024 [−0.004, 0.057] |
| `i1`@59 tail (1,872) | 0.112 | 0.186 | 0.224 | +0.037 [0.023, 0.051] |
| `i1`@59 good (≥ 0.6, 41,452) | 0.893 | 0.879 | 0.880 | +0.000 |

- **`fa1` does not fix the floor.**
  - 84% of `i1`'s floor stays below 0.2 under `fa1`.
  - `fa1`'s own increment there is inconclusive. It is +0.024 with the spectrum-weighted bootstrap,
    and +0.030 [0.001, 0.058] with molecule weighting.
  - On the same 275 floor spectra, `fa1`'s intensity head ranks the measured fragment worse than
    `i1`@63 in 135 spectra and better in 112; the rank ≤ 3 rate is 17.6% against 19.9%.
- **It helps the milder tail.**
  - Its tail gain is entirely in ranking failures: +0.041 on `R_ranking`, +0.001 on
    `U_uncovered`.
  - 64% of its whole-val gain over `i1`@59 comes from spectra at 0.05–0.2, 11% from the floor and
    25% from the rest.
  - It fits train slightly worse (−0.0013 against `i1`@59).
- **Epoch-to-epoch variance is large at the spectrum level.**
  - Four more epochs of `i1` lift its own floor's mean from 0.028 to 0.094, and cost the good
    spectra −0.014.
  - The floor keeps its size (330 to 279), but the sets overlap only at Jaccard 0.46; `fa1`'s floor
    overlaps `i1`@59's at 0.44.
  - So about half of which spectra sit on the floor is checkpoint-specific. The molecules are more
    stable (`c0` r = 0.77, above).
- **Correction (after the `i3` read): the floor and tail increments above are inflated.**
  - `i1`@63 is a same-run neighbour of the selector `i1`@59. Their per-spectrum inst-tol cosines
    correlate at 0.900, against 0.861–0.867 between separate runs. So on `i1`@59's worst spectra,
    `i1`@63 regresses less than a separate run does.
  - On floors chosen by `i3`@63, `fa1` − `i1`@63 is −0.014 [−0.053, +0.022] (floor) and −0.006
    (tail).
  - `fa1` has no floor- or tail-specific gain that survives an independent selector. Its whole-val
    gain (+0.006) involves no selection and stands
    ([log](../../../output/i1_worst_predictions_2026_09_30/i3/check/06_symmetric_selectors.log)).

## Result: `i3` on the floor (follow-up, 2026-09-30)

`i3` (`xypkkfti`) is `i1` plus dropout 0.2 on the boundary encoder and on the intensity inputs, and
EMA 0.999. The EMA average is the checkpoint's `state_dict`. Epoch 63 reproduces the logged 0.771194
to 6e-6 ([`i3/summary.md`](../../../output/i1_worst_predictions_2026_09_30/i3/summary.md), checked in
`i3/check/`).

| inst-tol cosine | `i1`@59 | `i1`@63 | `fa1`@63 | `i3`@63 | average of `i1`@59 and `i1`@63 spectra |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole val | 0.7880 | 0.7874 | 0.7934 | 0.7972 | 0.8012 |
| train sample | 0.9589 | 0.9581 | 0.9576 | 0.9589 | |
| floor chosen by `fa1`@63 (290) | 0.069 | 0.079 | (selector) | 0.072 | 0.075 |
| tail chosen by `fa1`@63 (1,605) | 0.189 | 0.200 | (selector) | 0.189 | 0.203 |

- **Whole val.** `i3`@63 − `i1`@63 is +0.0098 [0.0077, 0.0122], at no train cost. It is larger than
  `fa1`'s +0.0060 and narrows the train/val gap most of the four checkpoints.
- **No floor-specific gain.** On `i1`@59's floor, `i3`@63 − `i1`@63 reads +0.029 [0.003, 0.058].
  That is inflated by the same-run control, as above. On floors chosen by `fa1`@63 it is −0.007
  [−0.031, +0.013], and −0.011 on the tail.
  - Each checkpoint has its own floor of 279–330 spectra, and any two floors overlap at Jaccard
    0.36–0.46.
- **The ranking mechanism is unchanged.** On the 299 floor spectra with candidates, `i3`'s intensity
  head ranks the measured fragment better than `i1`@63 in 123 and worse in 111 (sign test p = 0.47).
  The bond head reads 134 against 111.
- **Averaging two `i1` checkpoints' predicted spectra beats both arms on whole val.** It is +0.0138
  [0.0128, 0.0149] over `i1`@63, against `i3`'s +0.0098. It has no floor effect either (−0.003 on
  the `fa1`-chosen floor).
  - So the whole-val gains of `i3` and of ensembling look like variance reduction on ordinary
    spectra, not a fix for the worst ones.

## Result: adding the bond head's scores to the final ranking (follow-up, 2026-09-30)

Should the bond head's scores enter the final spectrum (undoing selection-only)? This is an offline
test on `i1`@59
([`mixing/summary.md`](../../../output/i1_worst_predictions_2026_09_30/mixing/summary.md), checked in
`mixing/check/`).
- **Mixing rule.** Per spectrum, from one forward pass, the fragment logits become the intensity
  head's residual + α × the bond head's proposal log-probability over the same pool. The H-channel
  split is unchanged, and each α is scored with the inst-tol cosine.
- **α = 1 is exactly the joint-mode readout** of `masster_model.py:3646-3648` at frozen weights. The
  two fragment terms differ by a per-spectrum constant that renormalization removes.
- **Reproduction.** α = 0 reproduces the model's own spectra to 1e-15 on all 52,020 val and 49,205
  train-sample spectra.
- **Controls.** A shuffled-proposal null permutes each spectrum's own proposal scores (same
  magnitudes, no information). An entropy-matched temperature control sharpens the residuals alone.

| inst-tol cosine, change from α = 0 [95% CI] | α = 0 | α = 0.5 | α = 1.0 |
| --- | ---: | ---: | ---: |
| floor (330) | 0.028 | −0.007 [−0.009, −0.005] | −0.007 [−0.010, −0.003] |
| tail (1,872) | 0.112 | −0.017 [−0.021, −0.014] | −0.020 [−0.026, −0.013] |
| good, ≥ 0.6 (41,452) | 0.893 | −0.030 [−0.031, −0.028] | −0.061 [−0.064, −0.059] |
| whole val | 0.788 | −0.031 [−0.032, −0.029] | −0.061 [−0.063, −0.059] |
| train sample | 0.959 | −0.033 | −0.063 |

- **Every α from 0.1 to 1.0 lowers the cosine on every set.**
  - Choosing α on half the molecules and scoring on the other half picks α = 0 in both folds.
  - The result holds with medians, at 5 ppm, and with precursor-base-peak spectra excluded.
- **The bond head's scores reinforce the intensity head's wrong picks on the floor.**
  - The target's median rank improves (13 to 10 at α = 1), but mass spreads to other wrong
    fragments.
  - Real mixing is worse than the shuffled null on the floor (−0.027 [−0.047, −0.012] at α = 1) and
    the tail (−0.014). The real scores sharpen the prediction four times as much as the null
    (entropy −1.06 against −0.27 on the floor), so they add confidence to the same wrong answer.
  - They still beat pure sharpening there (0.021 against 0.014), so they carry a little information.
    Noise alone helps the floor and ruins good spectra, which is the overconfidence the `P^γ`
    tempering in the base-peak read already corrects.
- **No single-cut / multi-cut asymmetry.** Single-cut targets are not helped either (floor −0.037
  real − null at α = 0.5), so the per-bond product penalty is not the specific obstacle here.
- **Limit.** The heads were trained separately. This shows the current bond head adds no useful
  ranking signal where the intensity head fails; it cannot say what joint training would learn. The
  one joint-training run on record lost 0.0126.

## Next steps (training arms not built)

**2026-10-01 follow-up:** the proposed frozen cut-to-intensity check is now complete and
[CLOSED NULL](masster_cut_intensity_probe_2026_10_01.md). Both ambiguity-preserving train-only
probes select zero correction on separate tuning molecules; no training arm is justified by
that additive gate. The original options below remain hypotheses, not established remedies.

The floor is a ranking failure of both heads on new molecules. The encoder carries the needed cut
information (probe), and four things were tested without fixing it: full attention (`fa1`), EMA
with dropout (`i3`), checkpoint averaging, and adding the bond head's scores. What remains is a
training signal that states cut preferences in terms of local chemistry, which recurs across
molecules, instead of per-molecule spectra, which the heads can memorize. It has to reach the
intensity head, because in selection-only mode the bond head only changes which candidates exist.

| option | what it trains | evidence for | evidence against | status |
| --- | --- | --- | --- | --- |
| cut-level targets | a per-bond break score. Labels are the bonds cut by the fragments behind the measured peaks, intensity-weighted, with soft credit where structures are ambiguous (noisy-OR or responsibility, not hard m/z-match labels). Applied as an auxiliary loss on the shared encoder, or as a per-candidate feature for the intensity head | frozen-embedding classifiers trained on exactly such labels beat `i1`'s bond head on the floor (AUC 0.79–0.82 against 0.73) | the probe drops the deepest 130 floor spectra; bond-level AUC does not guarantee fragment-level ranking | not built |
| pairwise preferences within a spectrum | "fragment A over fragment B" on competing candidates, not absolute intensities | the failure is comparative: nested and sibling decisions in the base-peak table | FraGNNet's sibling-ranking target passed its coverage gate but was never implemented; no MASSter code | not built |
| cross-molecule relationship targets (`cm1`) | a soft auxiliary target saying related training molecules share losses | the census advantage survives for unfamiliar queries (+0.050 fragment, +0.037 loss cosine at Tanimoto ≤ 0.4); about 20% of the floor are transfer failures | about 45% of the floor has no similar molecule showing the pattern, and `i1` already predicts what analogues do (0.76 against the targets' 0.25), so the target may push that part the wrong way; the retrieval-style correction was negative (0.6287 against 0.6385) | built and smoke-tested, then reverted at the user's request before any run; not a measured null |
| more varied molecules (NIST23) | the same model on ~19,900 more [M+H]+ Orbitrap molecules | the tail rate falls with similarity to train (87 to 13 per 1,000) | the projected tail drop is at most 21–25%, and not specific to the floor; it needs a new preprocessing build | out of scope by an earlier user decision |

- **The cheap check before building the cut-level option (completed; result linked above):**
  - add the probe's per-bond scores, summed over each fragment's cuts, to the intensity head's
    logits offline, exactly like the bond-head mixing test;
  - if a transferable cut score does not help even when added this way, training it into the model
    is unlikely to.
- **The `cm0`–`cm2` code is kept only in `/tmp/frag_gnn_cm_revert_20260929/`** (`cm_only.patch`,
  `before.patch`, removed files). It is not in Git or a stash, and `/tmp` is a small volume on this
  workstation.

## Round-1 claims corrected

- **"Train ranking failures are a capacity ceiling"** (mol 26765, 29233) is refuted. They are
  hidden bin splits: 49 of 55 score ≥ 0.95 at tolerance. The val/train ranking gap grows from 34×
  to about 540×.
- **"The floor is a distinct, more confident mode"** is refuted: the model's top share is
  U-shaped in the inst-tol cosine.
- **"Amide, ester and anomeric cleavages stay intact"** rested on 2–4 spectra. At scale, amide
  C–N falls to 25%.
- **"`i1` lacks a functional-group or charge input"** is refuted: charges are inputs, and "N-oxide"
  and "charge" were the nitro group again.
- **"D4's recorded precursor is 6,700 ppm off"** mislabels the base peak (299.06) as the precursor.
  The header is exact.
- **"The model is CE-insensitive on floor molecules"** is a general property: 71% against 72%.
- **"A new failure mode on memorized molecules"** (mol 32385) is the nearest-candidate cosine
  artifact from the method section.

## Limits and open checks

- The main reads use one checkpoint (`i1`@59). The follow-ups add `i1`@63, `fa1`@63 and `i3`@63.
  At the spectrum level about half of the floor changes between epochs (Jaccard 0.46).
- A floor gain must be judged on a floor chosen by an independent checkpoint. A same-run
  control inflates it (see the `fa1` correction).
- No joint per-molecule table of causes: chemistry flag, analogue category, uncovered cause.
- The analogue categories use threshold rules; a sweep moves the transfer-failure share over
  12–21%.
- The model-internals read covers 289 of the 290 inst-tol ranking-floor spectra; the target
  fragment is found in 275. The H-shift statistic is not controlled for fragment mass.
- The analogue, chemistry-adjustment, energy and depth-bias reads ran on the nearest-candidate
  floor (354 spectra, 124 molecules). It holds 329 of the 330 inst-tol floor spectra.
- The train sample is 4,000 of 12,668 molecules, so its tail counts are small (43 spectra).
- Artifacts: `output/i1_worst_predictions_2026_09_30/`. `prepare.py`, `inst_tol/` (metric and
  tables), round-1 folders and `verify_*` folders each hold scripts with logs. `round1/` and
  `round2/` hold the structured results.
