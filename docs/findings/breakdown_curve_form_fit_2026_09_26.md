# Does a two-parameter breakdown-curve form fit NIST20's collision-energy series?

> **Status:** FINDING · **Updated:** 2026-09-26
> Offline and model-free, on NIST20 `[M+H]+`/FT train targets (`data/split/nist20_inchikey`,
> the raw target `c0` trains on). Each fragment channel of a molecule gets two numbers, and its
> spectrum at any NCE is `softmax_i(a_i − e_i·β(NCE))`: the Arrhenius form of RRKM/QET channel
> competition, with one β curve shared by every molecule.
>
> - **It fits.** Channel cosine 0.995 (0.988 on the full raw target), 98.4% of the CE variation a
>   CE-blind spectrum misses. That is above `c0`'s own eval-mode train cosine, 0.922.
> - **β comes out monotone without a constraint** and matches `1/(NCE + 200)` (R² 0.999).
> - **On a held-out interior energy** it beats linear interpolation between the neighbouring
>   energies (+0.0050, SE 0.0002) but loses to log-space interpolation (−0.0049). A second shared
>   component, three numbers per channel, closes that to −0.0007.
> - **Precursor survival at fixed NCE falls steeply with molecule size**, the opposite of the
>   dilution-over-modes argument. Within chain homologs the effect nearly vanishes.
> - **Perfect precursor survival is worth +0.027 val cosine** on `u2` (0.7394 → 0.7665), 83% of
>   it at NCE ≤ 40. The model over-calls the precursor most on novel molecules, and a
>   population survival prior is worse than the model (−0.027 to −0.036)
>   ([section](#precursor-survival-ceiling-on-val-2026-09-26)). A post-hoc correction from
>   inference-time context recovers only 12-15% of it. With `P^0.75` that is +0.009 on test,
>   with no training ([section](#how-much-of-it-a-post-hoc-correction-recovers)).
> - **The rest is one onset number per molecule that nothing here predicts.** A per-molecule
>   curve recovers 90-97% of the ceiling. Structure learners trained on all train molecules are
>   no better than the model, and stacked they add +0.0014. It is epistemic, so a
>   survival-specific head is not supported
>   ([section](#where-the-rest-lives-one-number-per-molecule-and-structure-learners-do-not-find-it)).
>
> The form is expressive enough to be an intensity head's CE law, with two components. Nothing
> here shows that it transfers better to new molecules.

## Question

The [intensity-displacement literature note](../notes/intensity_displacement_literature_2026_09_24.md)
proposes a CE-dependent breakdown curve. Before building a head whose CE dependence is restricted
to such a law, check whether measured CE series obey it:

1. **Capacity.** Fitted freely per molecule, would the form cap the training fit?
2. **Structure.** Does it predict a spectrum at an energy it was not fitted on?

## Method

Scripts: `output/qet_form_fit_2026_09_26/` (local, Git-ignored). Run `build_channels.py` from the
repository root. Then, from that directory, run `fit_forms.py channels.pkl .`,
`fit_rank2_global.py`, `analyze.py` and `extra_checks.py`. Each fit takes about 2.5 min on the
local RTX 2070 SUPER.

- **Data.** 12,570 train molecules with at least 3 distinct NCEs (of 12,668), 157,516 spectra,
  all single-energy HCD. There are 42 distinct NCE values, 2 to 195, and a median of 12 per
  molecule.
- **Target.** Raw peaks, unit sum, linear intensity: the `c0` recipe has no filters and
  `ints_transform: none`.
- **Channels.**
  - A peak is kept if a subformula of the `[M+H]+` precursor matches it within 10 ppm, with a
    200 Da floor (`_subformula_masses` in `clean_spectra_utils.py`).
  - Kept peaks are pooled over the molecule's spectra and single-linkage clustered at the same
    tolerance, so one ion keeps one identity across energies. That gives 1,087,056 channels, a
    median of 58 per molecule.
  - Channels hold 97.5% of target intensity (L1) and 99.3% of its L2 norm. The full-target
    cosine is `cos_ch × ‖T_ch‖/‖T‖`, which approximates the metric's 0.01 Da cosine.
- **Objective.** The mean per-spectrum cross-entropy on channel-normalized targets, the model's
  own objective. All molecules are fitted jointly, with Adam and cosine decay for 2,500 steps.
  5,000 steps change M1g by 1e-5.
- **Forms.**

  | form | logit of channel i in spectrum k | parameters |
  | --- | --- | --- |
  | M0 | `a_i` (the molecule's mean spectrum) | 1 per channel |
  | M1g | `a_i − e_i·β(NCE_k)`, β one global piecewise-linear curve, knots every 10 NCE | 2 per channel, 21 shared |
  | M1p | the same, β the standardized `1/(NCE + n0)` at a fixed n0 | 2 per channel |
  | M1f | the same, β free per spectrum | 2 per channel, 1 per spectrum |
  | M2g | `a_i + b_i·u(NCE_k) + c_i·v(NCE_k)`, u and v global curves | 3 per channel, 42 shared |
  | M2f | the same, u and v free per spectrum | 3 per channel, 2 per spectrum |

  β is identified only up to an affine map: a shift folds into `a_i` and a scale into `e_i`. Each
  molecule's energy scale, including any size dependence, is therefore absorbed into its `e_i`.
  Only the curve's shape is shared.
- **Held-out energy.**
  - Covers the 12,339 molecules with at least 5 distinct NCEs. Every spectrum at one randomly
    chosen interior NCE is removed (12,670 spectra, seed 0), and each form is refit on the rest.
  - Channels seen only in the removed spectra cannot be predicted. They cap the held-out cosine
    at 0.9999 or above for 99% of the removed spectra.
  - Baselines, all built from the same remaining spectra: the nearest-NCE spectrum, linear and
    log-space interpolation between the bracketing energies, and the mean spectrum. Log-space
    uses a 1e-4 floor, the best of 1e-5 to 1e-2.
  - Interpolation uses the molecule's measured neighbouring spectra, which a model predicting a
    new molecule never has. It is a within-molecule oracle, not a rival model.

## Result

### In-sample

| form | channel cosine | full-target cosine | KL (nats) |
| --- | ---: | ---: | ---: |
| M0 | 0.6827 | 0.6787 | 0.729 |
| M1p, n0 = 200 | 0.9950 | 0.9884 | 0.0234 |
| M1g | 0.9950 | 0.9884 | 0.0232 |
| M1f | 0.9967 | 0.9901 | 0.0179 |
| M2g | 0.9985 | 0.9919 | 0.0076 |
| M2f | 0.9996 | 0.9930 | 0.0044 |
| the target's own channels | 1 | 0.9934 | 0 |

- **M1g recovers 98.4%** of the CE variation M0 leaves. Its per-spectrum 1st, 5th and 50th
  percentiles are 0.952, 0.980 and 0.998. It is at least 0.993 in every NCE bin, and falls slightly
  with size: 0.9965 at up to 15 heavy atoms, 0.990 at 40-60.
- **Freeing β per spectrum adds only +0.0017.** Where a spectrum sits on its molecule's curve is
  set by NCE, through one curve shared by all molecules.
- **β(NCE) is monotone decreasing over NCE 10-190**, though nothing constrains it to be.
  - Its shape is `1/(NCE + n0)` with n0 near 200: R² 0.9989, against 0.95 at n0 = 50, 0.78 at
    n0 = 10, and 0.977 for a straight line.
  - With n0 fixed at 200, the one-parameter curve fits as well as the free curve (0.9950); at 50
    it gives 0.9944.
  - A fit with n0 free stalls (9.6 or 65 depending on the start, still improving), so no fitted
    n0 is reported as a measurement.
  - Read as an effective temperature, `kT ∝ NCE + 200` rises only 1.5x from NCE 20 to 130. On
    this data NCE is not proportional to a thermal temperature. n0 is a shape parameter, not a
    calibrated internal energy.
- **The precursor needs no special treatment.** Fitted as one more channel, its share matches the
  target to a mean absolute error of 0.024 at NCE ≤ 15 and 0.012 at NCE 25-35, with correlation at
  least 0.992 in every NCE bin.
- **A second component is real but small.** Rank 2 with shared curves (M2g) adds +0.0035 channel
  cosine over M1g and cuts KL three-fold.

### Held-out interior energy

| predictor | channel cosine | full-target cosine |
| --- | ---: | ---: |
| M2g, 3 numbers per channel | 0.9955 | 0.9884 |
| M1g, 2 numbers per channel | 0.9913 | 0.9842 |
| log-space interpolation | 0.9962 | 0.9891 |
| linear interpolation | 0.9863 | 0.9794 |
| nearest NCE | 0.9576 | 0.9511 |
| mean spectrum | 0.6719 | 0.6677 |

Paired over the 12,670 spectra:

- M1g − linear interpolation: +0.0050 (SE 0.0002).
- M1g − log-space interpolation: −0.0049 (SE 0.0001); M1g is better on 16% of spectra.
- M2g − log-space interpolation: −0.0007 (SE 0.0002); M2g is better on 46%.
- M2g − M1g: +0.0042 (SE 0.0002).

By the gap between the bracketing energies (channel cosine):

| gap (NCE) | M2g | M1g | log-space | linear | spectra |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤ 10 | 0.9971 | 0.9944 | 0.9974 | 0.9932 | 3,675 |
| 10-20 | 0.9956 | 0.9914 | 0.9955 | 0.9859 | 2,910 |
| 20-40 | 0.9956 | 0.9894 | 0.9964 | 0.9828 | 5,227 |
| > 40 | 0.9881 | 0.9896 | 0.9930 | 0.9800 | 858 |

- **The form carries structure beyond smoothness.** Its lead over linear interpolation grows with
  the gap, from +0.001 at 10 NCE or less to +0.010 above 40.
- **Most of that structure is that intensity moves exponentially with energy.** Log-space
  interpolation captures the same thing and beats one curve per channel.
- **A second shared component supplies the missing curvature.** M2g matches the log-space oracle
  up to a 40 NCE gap and falls behind beyond it (0.9881, 858 spectra), where the second component
  extrapolates poorly.

### Molecule size and precursor survival

Mean precursor share of the raw target:

| NCE | ≤ 10 heavy atoms | 10-15 | 15-20 | 20-25 | 25-30 | 30-40 | 40-60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤ 15 | 0.670 | 0.613 | 0.594 | 0.587 | 0.560 | 0.472 | 0.312 |
| 15-25 | 0.542 | 0.450 | 0.397 | 0.393 | 0.369 | 0.228 | 0.122 |
| 25-35 | 0.403 | 0.284 | 0.218 | 0.210 | 0.177 | 0.089 | 0.042 |
| 35-45 | 0.336 | 0.215 | 0.139 | 0.114 | 0.056 | 0.026 | 0.010 |
| 45-60 | 0.248 | 0.130 | 0.072 | 0.043 | 0.014 | 0.007 | 0.001 |
| 60-80 | 0.100 | 0.030 | 0.012 | 0.004 | 0.001 | 0.000 | 0.000 |

- **At a fixed NCE, larger molecules keep far less precursor.** At NCE 35-45 the share is 0.34 at
  up to 10 heavy atoms and 0.01 at 40-60. That reverses the argument that a larger ion is colder
  per mode at the same NCE. The argument rests on Thermo's NCE scaling lab energy with m/z, which
  makes the centre-of-mass energy per collision nearly independent of size.
- **Chemistry changes with size too**: glycosides, lipids and peptides are both large and labile.
  Within acyclic families that differ only in CH₂ count (same heteroatoms, same H − 2C), the
  effect nearly vanishes.
  - Over 47 families (478 molecules), the NCE at which the precursor falls below half the
    spectrum moves −0.35 per added carbon (median −0.27; 38% of families positive).
  - The largest clean series, mostly saturated acylcarnitines (30 molecules), moves +0.73 per
    carbon, the dilution direction.
  - The family key is only a proxy for a homologous series.
- **So the population trend is mostly about which bonds large molecules have**, not how energy is
  deposited. This is tentative.
- **For a head:** a network would have to learn any size dependence of the energy scale from
  structure. This fit absorbs each molecule's scale into its `e_i`, so it says nothing either way.

## What this means for a breakdown-curve head

- **Capacity is not the obstacle.** Fitted freely per molecule, M1g reproduces training spectra
  at 0.988 on the full target, against `c0`'s eval-mode train cosine of 0.922
  ([Step 1 table](../history/masster_memorization_route_plan_log.md#eval-mode-trainval-gap-on-the-step-1-checkpoints-2026-09-25);
  a different sample and metric pipeline, so the comparison is approximate).
  - A head that outputs `(a_i, e_i)` per candidate from a CE-blind candidate token, with CE
    entering only through one shared β(NCE), would not cap the training fit.
  - With two components, it is within 0.0015 of the target's own channels.
- **It ties the energies together.** A molecule's ~12 spectra constrain two or three numbers per
  candidate instead of 12 free allocations, and the candidate token no longer sees CE. That
  removes per-energy freedom from the lookup. Whether it helps new molecules is a training
  question this test cannot answer.
- **It does not supply which channels are easy in a new molecule.** That is still learned from
  structure. The form restricts how intensity moves with CE, while the val gap is ranking on new
  molecules at a given CE.
- **Possible compute gain.** With a CE-blind token, the encoder and candidate stack could run once
  per molecule, and each energy would cost one softmax. This is not checked, including whether
  MASSter's candidate set can be shared across energies (`share_proposal_across_ev` exists).

## Limits

- **In-sample and within-molecule only; no model was trained.** The held-out test interpolates
  between measured energies. It does not test extrapolation beyond the measured range, or
  transfer to new molecules.
- **NIST20 `[M+H]+`/FT (Orbitrap HCD) only.** Untested: QTOF collision energies in eV, stepped
  acquisitions such as Enveda-180, and other adducts. A stepped spectrum is an average over its
  steps' β, not one β.
- **Channels are subformula clusters, not MASSter candidates.** The full-target cosine
  approximates the 0.01 Da metric.
- **Channels seen in one spectrum have weakly determined parameters.** The held-out test is what
  guards against reading in-sample fit as structure.

## Precursor-survival ceiling on val (2026-09-26)

The gate for a survival-coupled head: how much cosine would a correct precursor share buy on new
molecules?

- **Setup.** `u2_ema999_c8hh9j1b.ckpt` (EMA weights) on all 52,020 val spectra, scored in the
  model's own `cos_sim_0.01` space. The baseline reproduces the model's per-spectrum
  `cos_sim_0.01` to within 5.4e-7.
- **Oracle.** The predicted precursor share, the L1 share of the 0.01 Da bins within 10 ppm of
  the precursor m/z, is set to the target's. The other predicted bins are rescaled to fill the
  rest, so the model's ranking among fragments is untouched.
- **Priors.** For comparison, the model's precursor share is replaced by the mean train share at
  that NCE, or at that NCE and heavy-atom bin.
- **Files.** `output/precursor_oracle_2026_09_26/` (`precursor_oracle.py`, `analyze_oracle.py`,
  `u2_val_analysis.txt`).

| NCE | spectra | model | perfect survival | gain | share of the total gain | mean predicted − true share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤ 20 | 9,305 | 0.8081 | 0.8744 | +0.066 | 44% | +0.069 |
| 20-30 | 5,816 | 0.7639 | 0.8173 | +0.053 | 22% | +0.023 |
| 30-40 | 6,206 | 0.7407 | 0.7805 | +0.040 | 17% | +0.018 |
| 40-60 | 9,916 | 0.7285 | 0.7475 | +0.019 | 13% | +0.009 |
| 60-80 | 5,577 | 0.6953 | 0.7011 | +0.006 | 2% | +0.006 |
| > 80 | 15,200 | 0.7107 | 0.7118 | +0.001 | 1% | +0.001 |
| all | 52,020 | 0.7394 | 0.7665 | **+0.027** (SE 0.0004) | | |

- **The ceiling is +0.027 overall**, 10.4% of the gap to 1. 83% of it sits at NCE ≤ 40. Perfect
  survival helps 45% of spectra and hurts 12%, where the rescaled fragments were themselves
  misallocated.
- **The model over-calls the precursor, most on novel molecules.** At NCE ≤ 30 its predicted
  share exceeds the true one by +0.11 for val molecules with nearest-train Tanimoto ≤ 0.4, and
  by +0.015 above 0.8. The ceiling there is +0.071 against +0.040, and the absolute error 0.22
  against 0.11. This is the fallback on unseen molecules found in the
  displacement Phase 0 (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_intensity_displacement_phase0_2026_09_24.md`).
- **A population prior is much worse than the model.** Swapping in the train mean at that NCE
  costs −0.036, and at that NCE and size −0.027. The model's survival already carries
  molecule-specific information, so a better survival term has to come from structure, not
  from a better curve.
- **The fragments are the weaker part.** With precursor bins removed from both sides, cosine is
  0.7188, below the full 0.7394.

This passes the rule set before running it: at least +0.02 overall, or +0.05 at NCE ≤ 30.

### How much of it a post-hoc correction recovers

A gradient-boosted regressor predicts the true precursor share from what inference already has:
the model's share, NCE, heavy atoms, precursor m/z, nearest-train Tanimoto, and the predicted
spectrum's entropy, top fragment share and peak count. Its prediction replaces the model's share.

- **Val** is cross-fitted, 5-fold GroupKFold by molecule.
- **Test** uses a regressor and `P^γ` fitted on all of val and applied once.
- **Scoring is exact.** `survival_sums.py` stores per-spectrum sums that give `cos_sim_0.01` in
  closed form for any precursor share and power. It reproduces the model's per-spectrum value to
  within 1.2e-6.
- **Files.** `recalibrate.py`, `recalibrate_log.txt`.

| change from the model | val (cross-fitted) | test (fitted on val) |
| --- | ---: | ---: |
| `P^0.75` (chosen in all 5 folds) | +0.0068 | +0.0065 |
| survival correction alone | +0.0032 | +0.0042 |
| `P^0.75` then survival correction | **+0.0090** (SE 0.0002) | **+0.0090** (SE 0.0002) |
| perfect survival | +0.0271 | +0.0283 |

- **Only 12-15% of the ceiling is recoverable** from the model's own share and inference-time
  context. On top of `P^0.75` the correction adds +0.0022 (val). It all comes at NCE ≤ 30:
  +0.011 at NCE ≤ 20 on val, +0.013 on test.
- **Novelty, size and spectrum shape add nothing.** The model's share plus NCE alone gives
  +0.0035, against +0.0032 with every feature. The over-call on novel molecules is not a
  correctable offset.
- **So about 85% of the survival ceiling, roughly +0.023, is not in inference-time context.** The
  next section tests whether structure learners find it.
- **Free now:** `P^0.75` plus the survival correction adds +0.009 on test with no training,
  fitted on val.

### Where the rest lives: one number per molecule, and structure learners do not find it

- **Oracles.** Each val spectrum's precursor share is replaced by a per-molecule logistic curve,
  `logit S = a_m − b·z` with `z` the standardized `1/(NCE + 200)`, fitted by BCE to that
  molecule's own true shares. The slope is either shared or per molecule.
- **Structure learners, fitted on val folds.** Q + NCE plus heavy atoms, the 85 RDKit `fr_*`
  functional-group counts, ring and rotatable-bond counts, and 1,024 Morgan count bits. They are
  cross-fitted like the correction above.
- **Structure learners, fitted on train targets.** The same features are fitted on the 157,678
  train spectra of 12,668 molecules. They use targets only, so the model's memorization of train
  cannot leak in. Train `T` comes from raw peaks: r = 1.0000 and MAE 0.0001 against the metric's
  `T` on val. Two predictors, each scored alone and stacked with Q + NCE:
  - `struct_gbm`, a gradient-boosted model on those features;
  - `knn_onset`, the Tanimoto-weighted mean onset `a_m` of the 5 nearest train molecules.
- **Label-free projection.** The model's own shares are projected onto a per-molecule curve,
  with no targets used.
- **Files.** `structure_probe.py`, `structure_probe_train.py` and their `_log.txt` files.

| survival used (val, 52,020 spectra) | cosine | gain | share of ceiling | gain at NCE ≤ 40 |
| --- | ---: | ---: | ---: | ---: |
| model | 0.7394 | | | |
| true share per spectrum | 0.7665 | +0.0271 | 100% | +0.055 |
| per-molecule curve, shared slope | 0.7638 | +0.0244 | 90% | +0.050 |
| per-molecule curve, own slope | 0.7656 | +0.0262 | 97% | +0.053 |
| model's shares projected onto a per-molecule curve | 0.7400 | +0.0006 | 2% | +0.001 |
| Q + NCE (val-fitted) | 0.7429 | +0.0035 | 13% | +0.008 |
| + functional groups (val-fitted) | 0.7431 | +0.0037 | 14% | +0.008 |
| + functional groups + Morgan (val-fitted) | 0.7429 | +0.0036 | 13% | +0.009 |
| functional groups + Morgan, no Q (val-fitted) | 0.7295 | −0.0099 | −36% | −0.014 |
| `struct_gbm` alone (train-fitted) | 0.7363 | −0.0031 | −12% | −0.002 |
| `knn_onset` alone (train-fitted) | 0.7219 | −0.0175 | −65% | −0.026 |
| Q + NCE + `struct_gbm` | **0.7443** | **+0.0049** (SE 0.0002) | 18% | +0.011 |
| Q + NCE + `struct_gbm` + `knn_onset` | 0.7440 | +0.0046 | 17% | +0.010 |

- **The headroom is one number per molecule.** A per-molecule onset with a shared slope recovers
  90% of the ceiling, and a per-molecule slope 97%. Per-spectrum scatter around a molecule's
  curve is worth under 0.003.
- **The model's curve is already self-consistent across energies.** Projecting its shares onto
  a per-molecule curve gains +0.0006. Its error is a wrong onset for the whole molecule, not
  energy-to-energy noise.
- **Structure learners trained on the same labels are no better than the model.** A
  fingerprint GBM trained on all train molecules comes within −0.0031 of the 9M-parameter
  model's survival (MAE 0.088 against 0.076). A nearest-neighbour onset is much worse. Stacked
  with the model, the GBM adds +0.0014: it behaves like a second ensemble member, not a missing
  information source.
- **So the remaining ~+0.022 is epistemic**, like the rest of the val gap
  (variance is epistemic (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_variance_novelty_2026_09_24.md`)). It is not something
  a survival-specific parameterization or post-hoc stage can extract from these 12.7k molecules.

## Next

- **The breakdown-curve head's gates ran**
  ([finding](masster/masster_qet_projection_candidate_union_2026_09_26.md)).
  - Projecting `u2`'s full predicted spectra onto the form gains +0.0019 val, so the output law
    has little to correct.
  - One CE-blind candidate set per molecule is feasible.
  - MASSter's NCE input is a categorical code in which NCE 76 and 180 collide. The smooth energy
    axis this fit measured is what that input lacks.
- **A survival-specific head is not supported.** In MASSter the precursor is already the intact
  (empty-mask) candidate, in the same softmax as every fragment. Its share already competes with
  the total fragment mass, which is the RRKM coupling. With that softmax the loss factorizes
  exactly into BCE on the precursor share plus `(1 − T)`-weighted fragment cross-entropy. The
  onset is a molecule-level generalization error that neither a curve constraint nor
  fingerprints fix.
- **What would test the epistemic reading.**
  - Decompose a two-checkpoint ensemble's gain into survival and fragments, with `u2` + `k1` or
    `c0` in `saved_ckpts/` and the same sums.
  - The NIST23-molecule data arm.
  - An onset feature that does not come from these labels, such as computed thresholds for the
    cheapest neutral losses (for example GFN2-xTB reaction energies of the protonated ion).
- **Free and deployable:** `P^0.75`, then the Q + NCE survival correction, +0.009 on test.
  Stacking `struct_gbm` adds about +0.0014 on val, not yet run on test.
