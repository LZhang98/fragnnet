# Residual structure on `i1`: where the val deficit is, and whether recorded metadata is still missing

> **Status:** FINDING · **Updated:** 2026-09-28
> Inference only; no training. `i1_inst_ob_obf` (`i50ig7rf`) at epoch 59, on all 52,020 val
> spectra, with `c0`@42 and a 4,000-molecule train sample for contrast. One checkpoint per run.
> This follows the instrument result: the probes read the missing instrument as memorization
> ([`i1` memorization read](masster_i1_memorization_read_2026_09_28.md)). Is any other missing
> input left?
>
> - **The deficit is fragment allocation, not conditioning.** Of `i1`'s val deficit
>   (1 − 0.7622 = 0.238):
>   - 18% at most can be fixed by reweighting intensity between precursor, mid and deep bands
>     (the most any energy-like input can do);
>   - 64% is how intensity is split among covered bins within a band;
>   - 8% has no candidate within 10 ppm;
>   - 9% has a candidate within 10 ppm that the 0.01 Da bins split from its target peak.
> - **What conditioning could still fix is molecule-specific and unpredictable.**
>   - Knowing each val molecule's energy scale is worth +0.010 cosine (a per-molecule ladder
>     shift; train +0.0006). Two thirds of molecules want no shift, and the rest split about
>     evenly between +1 and −1.
>   - No recorded variable predicts that shift on held-out molecules (R² ≤ 0). Applying the
>     predicted shift costs cosine.
>   - A shared instrument × NCE curve explains 0.4-6% of the within-molecule variation.
> - **No recorded acquisition field adds cosine.** Campaign, `Nreps`, m/z error, solvent, vial
>   and eV make held-out residual predictions no better or worse (MSE up to 6% higher).
>   Chemistry descriptors cut the MSE of the loss and of the precursor fraction by 4-6%. No stack
>   turns that into cosine: rescaling the predicted bands to any stacked estimate loses
>   0.004-0.007.
> - **One real acquisition effect, worth no cosine.** Lumos campaigns differ in precursor
>   survival beyond chemistry (p < 0.001, 3% of residual variance; up to +0.11 precursor fraction
>   in the breakdown region). Late-2019 campaigns fragment more than the model expects. Elite
>   campaigns do not differ (p = 0.52).
>   - Cross-fitted per-campaign corrections lose cosine (−0.0008 Lumos).
>   - Train spectra of the off campaigns show no extra identity-route damage.
> - **The prediction's sharpness flips with novelty.** Unfamiliar molecules (nearest-train
>   Tanimoto ≤ 0.4) get flatter spectra than measured, 1.31× the effective peaks on fully
>   fragmented spectra. Close analogues (> 0.8) get sharper ones. This is prediction by
>   similarity to training molecules, not by a rule that transfers.
> - **Side finding: `cos_sim_0.01` depends on the device.** CPU and CUDA build different float32
>   bin grids, so a CPU rescoring of the same predictions reads 0.7611 against the logged 0.7622.

## Question

The memorization probes read `c0`'s missing instrument input as memorization: identity routes
carried the instrument on train, and val had only the NCE value
([`i1` memorization read](masster_i1_memorization_read_2026_09_28.md)). Giving the model the
instrument narrowed the gap while train fit rose. So before constraining capacity further, is
another input missing?

A missing input should leave errors that repeat across molecules under the same conditions and
that a recorded variable predicts on molecules the model has not seen. Four questions:

1. Where in the spectrum is the val deficit, and how much of it could a conditioning input fix?
2. Is the error shared across molecules along the energy axis, or molecule-specific?
3. Does any recorded variable (acquisition metadata, chemistry) predict the residual on held-out
   molecules beyond the model's own prediction, and is that worth cosine?
4. Is the NIST acquisition campaign a missing input, as the instrument was?

The instrument and energy readouts by NCE, m/z and extent are already in the memorization read's
follow-up: no Lumos-specific energy or shape error is left, and the leftover precursor bias costs
no cosine ([follow-up](masster_i1_memorization_read_2026_09_28.md#is-an-energy-or-shape-error-left-follow-up-2026-09-28)).
This finding starts there.

## Method

- **Predictions.** `collect_masster_preds.py`'s pickles for `i1`@59 and `c0`@42:
  - every val spectrum;
  - the 4,000-molecule train sample (seed 0).
- **Cosine.** Recomputed from the stored spectra on the model's floor binning. The bin grid is
  built by `th.arange` on CUDA in each side's dtype (float64 predictions, float32 targets). This
  reproduces the logged `cos_sim_0.01` to 2 × 10⁻⁶ on every spectrum (see the side finding).
- **Instrument.** The raw NIST20 instrument label: Elite, Lumos or Velos.
- **Acquisition fields** are parsed from the raw MSP `Notes`:
  - campaign: `Data_source` (`Met_2017_06`, `Met_Lumos2_2019_11`, …), the older
    `Metabolite_*_ID`, the 2015 peptide library, or legacy `Acetonitrile/Water` records;
  - `Nreps`, m/z error in ppm, solvent, `Vial_ID`, and the `_FTMSn` suffix.
  - Only 38 val spectra have no campaign.
  - 0.78% of (molecule, instrument) series span two campaigns, so campaign is a molecule-level
    variable, like the instrument was.
- **Chemistry** per molecule:
  - heavy atoms, element counts, ring and aromatic ring counts, Fsp3, rotatable bonds, HBD/HBA,
    TPSA and Crippen logP;
  - a basic-site class and a basic-amine count;
  - the 166 MACCS keys.
- **Novelty.** The maximum Morgan-r2 Tanimoto to any train molecule, from the memorization read.
- **Bounds.** Per spectrum, the binned cosine is split by band b as Σ_b a_b / √(Σ_b n_b), with
  a_b = p_b · t_b and n_b = |p_b|² for the L2-normalized prediction p and target t.
  - Scaling band b by w_b gives Σ w_b a_b / √(Σ w_b² n_b). This is maximized at w_b = a_b / n_b,
    so the best value is √(Σ a_b² / n_b).
  - The **3-band** scheme (precursor, mid ≥ precursor/2, deep < precursor/2) bounds what an
    energy-like input can do.
  - The **21-band** scheme adds 20 bands of 5% of the precursor m/z. For a 300 Da precursor these
    are 15 Da wide, so it mixes energy with fragment identity.
  - "Oracle" rescales each band to the target's L1 mass instead.
  - The **support bound** is the cosine of predicting the target exactly on the bins the
    prediction's candidates cover. It is split by tolerance: a target peak with a candidate
    within 10 ppm (200 Da floor, as in the
    [ranking decomposition](masster_ranking_decomposition_2026_09_23.md)) but in another bin is
    proposed but split by the binning; a peak with none is not proposed.
- **Energy offset.** A series is one molecule's spectra on one instrument, ordered along its NCE
  ladder. A shift of s steps scores each target against the prediction s rungs higher, clipped at
  the ends. The per-series best s in [−3, 3] is the molecule-specific energy-scale oracle.
  - NCE50 is where the precursor fraction falls through 0.5, linear between rungs.
- **Held-out tests.** Gradient boosting (scikit-learn `HistGradientBoostingRegressor`) predicts a
  target from:
  - the model's own predicted shape (precursor, deep, centroid, effective peak count);
  - instrument, NCE, log precursor m/z and novelty;
  - plus a covariate group.
- **Held-out design.** Val molecules are split 50/50: a discovery half (5-fold `GroupKFold` by
  molecule) and a confirmation half (fit on discovery, score on confirmation). The tables report
  confirmation.
- **Uncertainty.** Intervals are 95% molecule-cluster bootstraps.

## Result: where the deficit is

| | cos | best 2-band (precursor / rest) | oracle 3-band | best 3-band | oracle 21-band | best 21-band | support, bins | support, bins or 10 ppm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `i1` val | 0.7622 | 0.7859 | 0.7933 | 0.8058 | 0.8566 | 0.8804 | 0.9588 | 0.9810 |
| `i1` val Elite | 0.7699 | 0.7955 | 0.8016 | 0.8143 | 0.8633 | 0.8863 | 0.9600 | 0.9826 |
| `i1` val Lumos | 0.7423 | 0.7618 | 0.7727 | 0.7848 | 0.8406 | 0.8662 | 0.9570 | 0.9769 |
| `c0` val | 0.7292 | 0.7645 | 0.7768 | 0.7894 | 0.8490 | 0.8732 | 0.9580 | 0.9802 |
| `i1` train sample | 0.9245 | 0.9302 | 0.9276 | 0.9343 | 0.9329 | 0.9479 | 0.9612 | 0.9852 |

The bounds are nested per spectrum (best 3-band ≤ support, bins ≤ support, bins or 10 ppm), so
the `i1` val deficit splits:

| part of the deficit | cosine | share |
| --- | ---: | ---: |
| band reweighting, best case | 0.044 | 18% |
| allocation among covered bins, within bands | 0.153 | 64% |
| a candidate within 10 ppm, in another 0.01 Da bin | 0.022 | 9% |
| no candidate within 10 ppm | 0.019 | 8% |
| total | 0.238 | |

- **A precursor-only correction is worth at most +0.024**, even with cosine-optimal weights per
  spectrum.
- **`i1` already took part of the band error.** `c0`'s best 3-band gain is 0.060 (Lumos 0.077);
  `i1`'s is 0.044 (Lumos 0.043).
- **The band part sits at low and middle NCE.** The best 3-band gain is 0.04-0.08 at NCE ≤ 60 and
  0.010-0.016 above 100. At high NCE the precursor is gone and the deficit is which fragments.
- **The support bound is the same on train and val (0.961 against 0.959).** Train's remaining
  deficit is 51% outside the bins (20% no candidate, 32% split bins); val's is 17% (8% and 9%).
  Memorization cannot pass the support, and the gap between train and val is allocation.
- **No candidate within 10 ppm: 6.0% of val target intensity** (Elite 5.3%, Lumos 7.6%).
  - This is the ranking decomposition's coverage share on f0/f1 (6% of the gap).
- **Split bins: 0.022 cosine** that no allocation can reach under `cos_sim_0.01`.
  - The candidate is within 10 ppm, but the target peak sits across a 0.01 Da edge.
  - This is the metric's edge sensitivity, not a missing fragment.
- 12.4% of val spectra have less than 80% of their target intensity in covered bins.

## Result: shared or molecule-specific

Variance of each residual over `i1` val (Elite and Lumos, 50,558 spectra), split into between
(molecule, instrument) series and within them. The last column is the share of the within part
explained by an instrument × NCE curve fitted on the other half of the molecules.

| residual | between series | within series | shared curve, of within |
| --- | ---: | ---: | ---: |
| 1 − cos | 0.456 | 0.544 | 0.057 |
| 1 − fragment-only cos | 0.585 | 0.415 | 0.004 |
| precursor-fraction bias | 0.319 | 0.681 | 0.031 |
| deep-fraction bias | 0.471 | 0.529 | 0.009 |
| centroid bias | 0.447 | 0.553 | 0.028 |

- **The energy-axis error is not shared.** A common distortion along NCE would show in the last
  column. It explains 0.4-6% of the within-molecule variation.
- The precursor error is mostly within series (68%): it changes along a molecule's energy series
  in a way specific to that molecule.

### The molecule-specific energy scale

Elite and Lumos series with at least 3 NCE values:

| | series | cosine | per-molecule shift oracle gain | best shift −1 / 0 / +1 | ±2 or ±3 |
| --- | ---: | ---: | --- | --- | ---: |
| `i1` val | 4,081 | 0.7605 | **+0.0103** [+0.0095, +0.0111] | 0.13 / 0.66 / 0.16 | 0.05 |
| `c0` val | 4,081 | 0.7265 | +0.0264 [+0.0249, +0.0277] | 0.18 / 0.51 / 0.17 | 0.14 |
| `c0` val Lumos | | 0.6814 | +0.0448 | 0.05 / 0.34 / 0.36 | 0.25 |
| `i1` train sample | 3,886 | 0.9243 | +0.0006 | 0.01 / 0.96 / 0.02 | 0.00 |

- **Positive** means the target is matched by a higher-energy prediction: the model
  under-fragments that molecule.
- **`c0`'s Lumos error was a shared shift.** 60% of Lumos series wanted a positive one, which is
  the missing-instrument signature. On `i1` the shifts are symmetric.
- **`i1` fits each train molecule's energy scale** (96% no shift) and misses val molecules' scale
  by a molecule-specific amount.
  - NCE50 error: median |prediction − target| is 2.97 NCE on val against 1.12 on train.
  - On val, 62% of molecules are predicted too stable. The median is +1.0 to +1.9 NCE in every
    m/z bin except Lumos above m/z 450 (+0.2, 19 series).
  - `c0`'s mass-dependent Lumos offset is gone: it was +12.5 below m/z 200 and +2.2 above 450,
    and `i1` reads +1.4 and +0.2.
- The per-spectrum best rung (optimistic: a maximum over the ladder) gains +0.043.

Can a recorded variable predict the molecule-specific shift? 5-fold `GroupKFold` over val series:

| covariates | best shift: held-out R² | applying the predicted shift | ΔNCE50: held-out R² |
| --- | ---: | --- | ---: |
| instrument, m/z, novelty, ladder length | −0.030 | −0.0012 [−0.0016, −0.0008] | +0.002 |
| + chemistry and MACCS | −0.025 | −0.0010 | **+0.061** |
| + acquisition fields | −0.029 | −0.0017 | +0.010 |
| + both | −0.016 | −0.0010 | +0.046 |

- **Chemistry explains 6% of the NCE50 error. Acquisition fields explain none beyond it.**
- Most (94%) of the molecule-specific energy error is not predicted by these descriptors.

## Result: do recorded variables predict the residual?

Confirmation-half R² of the base stack, and the MSE change when a covariate group is added
(positive = better), 26,125 spectra:

| target | base R² | + acquisition | + chemistry | + chemistry + MACCS | + all |
| --- | ---: | --- | --- | --- | --- |
| 1 − cos | 0.202 | −3.7% [−5.3, −1.8] | +4.4% [+2.6, +6.3] | +5.6% [+3.7, +7.6] | +0.9% [−1.4, +2.9] |
| 1 − fragment-only cos | 0.170 | −6.1% [−8.4, −3.6] | +3.0% | +5.2% [+2.8, +7.4] | −1.4% |
| true precursor fraction (logit) | 0.896 | −0.1% | +2.6% | +4.2% [+2.2, +6.2] | +2.1% |
| true deep fraction (logit) | 0.867 | −4.0% | +0.6% | +0.7% [−1.4, +2.9] | −2.3% |
| true centroid | 0.912 | −3.9% | −0.2% | +0.4% | +0.2% |

- **Acquisition fields hurt on held-out molecules.** They let the booster localize molecules
  (campaign, vial) without a signal that transfers.
- **Chemistry helps a little**, most on the loss magnitude and the precursor fraction.

In cosine, on the confirmation half, the prediction's three bands are rescaled to the stacked
precursor and deep estimates:

| | cosine | gain |
| --- | ---: | --- |
| model | 0.7595 | |
| oracle: target band masses | 0.7912 | +0.0317 [+0.0292, +0.0341] |
| best band weights | 0.8041 | +0.0446 |
| stack, base | 0.7549 | −0.0046 [−0.0056, −0.0037] |
| stack, + chemistry + MACCS | 0.7560 | −0.0035 |
| stack, + acquisition | 0.7529 | −0.0066 |
| stack, + all | 0.7535 | −0.0060 |

- **No stack beats the model's own band masses.** The MSE gains do not carry into cosine, as for
  the per-group precursor rescale in the memorization read's follow-up (+0.0002 on Lumos).

## Result: campaign

The heterogeneity statistic is the spectrum-weighted between-campaign variance of the precursor
residual after the base + chemistry + MACCS stack, out of fold by molecule. The null permutes
campaign labels across molecules within an instrument (1,000 draws); campaigns need ≥ 100
spectra (≥ 50 in the breakdown region).

| residual | instrument | campaigns | observed | null mean | p | excess share of residual variance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| logit, all spectra | Elite | 43 | 0.0124 | 0.0127 (95th pct 0.0179) | 0.52 | 0.000 |
| logit, all spectra | Lumos | 29 | 0.0530 | 0.0137 (95th pct 0.0205) | < 0.001 | 0.026 |
| probability, all spectra | Lumos | 31 | 0.00024 | 0.00008 | < 0.001 | 0.014 |
| probability, predicted pf 0.05-0.95 | Elite | 43 | 0.00030 | 0.00040 | 0.88 | −0.003 |
| probability, predicted pf 0.05-0.95 | Lumos | 25 | 0.00194 | 0.00072 | < 0.001 | 0.033 |

- **Lumos campaigns differ; Elite campaigns do not.** The Lumos effect holds on the probability
  scale and where the predicted precursor fraction is 0.05-0.95. That region is selected on the
  input side, so it is not a noise-floor artefact.
- **It follows the date.** In the breakdown region, mean true minus stacked precursor fraction:
  - `Met_Lumos2` August and September 2019: +0.114 and +0.058;
  - `Met_Lumos` July 2019: +0.086; April 2019: +0.065;
  - October to December 2019: −0.038 to −0.042 on both lines (`Met_Lumos2` October, November,
    December and `Met_Lumos` November).
  - These are less and more fragmentation than the model expects at the recorded NCE.
- **It is worth no cosine.** Cross-fitted over 2 folds by molecule and 5 repeats:

  | correction | Elite | Lumos |
  | --- | ---: | ---: |
  | precursor rescale per instrument × NCE bin | +0.0000 | +0.0002 |
  | precursor rescale per campaign × NCE bin | −0.0013 | −0.0008 |
  | ladder shift per instrument | +0.0000 | +0.0000 |
  | ladder shift per campaign | −0.0000 | −0.0005 |

  - Every Lumos campaign's in-sample best ladder shift is 0: the offsets are under one rung.
- **No identity-route signature on train.** On the memorization probe's train spectra, identity
  excess is set-message (or global-attention) damage beyond what cc-pool damage predicts. Across
  campaigns, it does not rise with the size of the campaign's val residual:
  - Lumos, 24 campaigns: set message ρ = −0.27 (permutation p = 0.20), global attention
    ρ = +0.02 (p = 0.93);
  - Elite, 41 campaigns: ρ ≈ 0.
  - For scale: with the instrument missing, `c0`'s Lumos train spectra took +0.028 (set message)
    and +0.042 (global attention) more damage than Elite ones
    ([memorization read](masster_i1_memorization_read_2026_09_28.md#result-by-instrument)).

## Result: sharpness by novelty (added 2026-09-28)

Spread is the log effective peak count, exp(entropy) of the binned spectrum. The bias is
predicted minus measured, in nats; positive means the prediction is flatter. Rows are `i1` val
(Elite and Lumos) by nearest-train Tanimoto. The fragment-pattern columns keep only spectra
whose measured precursor fraction is below 0.05.

| Tanimoto | spectra | cos | spread bias | fragment pattern: spectra | fragment cos | spread bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤ 0.4 | 6,409 | 0.634 | +0.107 | 4,025 | 0.569 | **+0.269** |
| 0.4-0.5 | 9,573 | 0.692 | +0.068 | 5,918 | 0.638 | +0.173 |
| 0.5-0.6 | 11,127 | 0.768 | +0.025 | 6,898 | 0.734 | +0.102 |
| 0.6-0.7 | 11,553 | 0.803 | −0.006 | 7,376 | 0.773 | +0.043 |
| 0.7-0.8 | 8,002 | 0.840 | −0.040 | 5,279 | 0.821 | −0.028 |
| > 0.8 | 3,910 | 0.851 | −0.088 | 2,664 | 0.842 | **−0.077** |

- **The sign of the error flips with novelty.**
  - On unfamiliar molecules the prediction is flatter than the spectrum: 1.31× the effective
    peaks at Tanimoto ≤ 0.4.
  - On close analogues it is sharper.
  - The [ranking decomposition](masster_ranking_decomposition_2026_09_23.md)'s "slightly too
    sharp" on f0/f1 is the average over these.
- **This is the pattern of prediction by similarity to training molecules.** With a close
  analogue the model commits to that analogue's pattern, too firmly. Without one it spreads its
  intensity over the competing channels, because it has no rule for which one wins.
- The measured-precursor restriction conditions on the target. The all-spectra column shows the
  same flip.

## Side finding: `cos_sim_0.01` depends on the device

- **Mechanism.** `batched_bin_func` builds its bin edges with
  `th.arange(res, mz_max + res, res, dtype=mzs.dtype)` on the tensors' device. The CPU and CUDA
  float32 grids differ at 56,022 of 150,000 edges, by up to 1.2 × 10⁻⁴ Da; float64 grids at
  19,971.
- **Why it matters.** A target peak recorded to two decimals sits on an edge.
  - Example: m/z 74.06, stored as 74.05999756, falls in bin 7405 on the CPU grid and 7406 on the
    CUDA one.
  - The predicted m/z (74.0600x) is in 7406, so the peak matches only on the GPU.
- **Size on `i1` val.**
  - A CPU rescoring of the same predictions averages 0.7611 against the logged 0.7622 (−0.0011).
  - 1.1% of spectra differ by more than 10⁻³ (0.99% lower on the CPU, 0.15% higher) and 0.6% by
    more than 0.01, the largest by 0.99.
  - A float32 cast of the float64 predictions adds further mismatches.
- **Consequences.**
  - Any CPU rescoring of exported predictions reads slightly low against W&B.
  - The memorization follow-up's reimplemented cosine ("within 10⁻³ on 98% of spectra") has these
    same spectra.
  - Built on CUDA in each side's dtype, as in `build_table.py`, the reimplementation is exact.
- **Not changed.** The metric's edge sensitivity is already recorded
  ([dropout-view finding](masster_dropout_view_disagreement_2026_09_27.md)). Nothing in `src/` is
  changed.

## Reading

- **The instrument was the recorded input that probes read as memorization. Recorded metadata
  holds no second one of cosine value.**
  - Campaign is the only acquisition field with a real effect, and only for Lumos.
  - It is under one ladder rung, 3% of the residual variance, worth nothing when corrected, and
    invisible in the train identity routes.
  - A NIST23 model would weigh it more: Lumos is 35% of NIST23 `[M+H]+`, so this check is worth
    repeating there.
- **The remaining val deficit is structural.** 64% is how intensity is shared among fragments
  the model already proposes, within an extent band. Another 8% is fragments no candidate
  reaches, and 9% is lost to the metric's bins.
  - This agrees with the [ranking decomposition](masster_ranking_decomposition_2026_09_23.md) on
    f0/f1: candidate ordering is half the gap, coverage 6%, and the ranking error nearly vanishes
    on training molecules.
  - That split is fixed by molecular structure, and the model reads the structure. So the deficit
    is a failure to transfer the structure → branching map to new molecules, not missing input.
- **The energy response transfers per molecule, not per condition.** Train molecules have their
  energy scale fitted; val molecules miss theirs by about 3 NCE in either direction. That error
  is a molecular property (stability), and the descriptors tested predict 6% of it.
- **Where that leaves the memorization levers.** They act on the per-molecule lookup, which is
  where the remaining gap is:
  - `i3` (`k1` + EMA on `i1`) and `i4`
    ([plan](../../history/masster_memorization_route_plan_log.md#stacking-on-the-instrument-input-i3-and-i4-built-2026-09-28-not-submitted));
  - inductive bias for fragment ranking;
  - more molecules (NIST23).
  - Candidate coverage is a separate lever for the 8%.
  - The 9% in split bins is the metric's. A tolerance-matched score does not count it. On test,
    `cos_hun_inst_tol` is 0.7909 against `cos_sim_0.01` 0.7656
    ([instrument finding §8](../nist_orbitrap_instrument_ce_scales_2026_09_28.md#8-training-with-the-instrument-input-i1-and-i2-running-read-2026-09-28));
    that difference also includes the Hungarian matching.
- **What residuals cannot see.** An acquisition variable NIST never recorded would look like
  molecule-specific error. The same-instrument, cross-campaign NIST23 null bounds it: 0.946 on
  0.01 Da bins
  ([instrument finding §3.1](../nist_orbitrap_instrument_ce_scales_2026_09_28.md#31-at-the-same-nce)).

## Limits

- One checkpoint per model, one seed, NIST20 `[M+H]+` only.
- **The band bounds are upper bounds for reweighting** at 3 and 21 bands. A conditioning input
  could also move intensity among fragments inside a band, which these bounds count as
  allocation.
- **The shift oracle uses the model's own predictions at other rungs**, clipped at the ladder
  ends. It is a proxy for re-predicting at a continuous energy offset.
- **The chemistry descriptors are generic.** A physics feature they do not span, such as a
  computed proton affinity or bond dissociation energy, is not ruled out; they explain 6% of the
  NCE50 error.
- **Campaign and chemistry are confounded at the collection level.** Each campaign measured its
  own compounds, and within-molecule cross-campaign pairs are too rare in NIST20 to separate
  them. The campaign residual is after chemistry and MACCS.
- **The support bound uses this model's candidates**, at 0.01 Da bins and at 10 ppm, not the DAG.

## Artifacts

Everything is in `output/i1_residual_structure_2026_09_28/` (local, Git-ignored). Inputs are
`output/ev_ladder_census_2026_09_28/{i1,c0}_{val,train}.pkl`.

- `build_table.py`: per-spectrum descriptors, parsed `Notes` and chemistry. Writes
  `{tag}_val_resid.pkl` and `mol_feats.pkl`.
- `build_bands.py`: band terms, `{tag}_{split}_bands.pkl`.
- `support_bound.py` and `support_bound_log.txt`: support bounds in bins and at 10 ppm,
  `{tag}_{split}_support.pkl`.
- `analyze.py` and `analyze_log.txt`: band bounds, variance split, stacking, campaign table.
- `energy_shift.py` and `energy_shift_log.txt`: shift and NCE50 oracles, predictability, and
  `{tag}_{split}_series_shift.pkl`.
- `campaign_check.py` and `campaign_check_log.txt`: heterogeneity, cross-fitted corrections,
  robustness.
- `route_by_campaign.py` and `route_by_campaign_log.txt`: train identity-route damage by
  campaign.
