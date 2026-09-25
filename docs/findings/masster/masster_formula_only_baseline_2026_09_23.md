# How much of MASSter's validation cosine can formula statistics alone explain?

> **Status:** FINDING · **Updated:** 2026-09-23
> Offline, on the `6h67zps9` (f1) exports of the
> [ranking decomposition](masster_ranking_decomposition_2026_09_23.md); no training of
> MASSter. A predictor that sees only formulas (ion, neutral loss, precursor) and NCE, fitted
> on training molecules, reaches **0.38-0.40** val cosine as a lookup table and **0.46** as a
> learned model, against the model's **0.76**. Its base-peak hit rate is 25-29% against 57%.
> Almost all of its signal is at low NCE. Mixing it into the model adds **+0.005** (test-confirmed,
> SE 0.0004), and the gain is largest above NCE 60 (+0.009). The model already carries nearly all
> the transferable formula-level signal, so its val performance and its failures are structural.

## Question

The [GLACIER note](../../notes/glacier_object_detection.md) shows GLACIER building its fragment
token from formula terms (fragment formula, neutral-loss formula, broken-bond count), while
MASSter's token has no formula term at all. The question is whether formula-level statistics
carry transferable signal that MASSter is not using, especially given its train/val gap
(ranking error 0.011 on train against 0.121 on val).

## Method

Scripts, cached tables and per-spectrum results live in
`output/cross_eval_mh_ft_clean_2026_09_23/analysis/formula_baseline/` (local, Git-ignored), next
to `ranking_decomp.py`. Run them from that directory: `build_prior.py`, `run_baseline.py`,
`features.py`, then `gbm.py`. Targets are the f1 cleaned targets (`filter_subformula_peaks`,
above-precursor filter, centroiding), dumped from the dataset as in `dump_targets*.py`, with
copies in `output/cross_eval_mh_ft_clean_2026_09_23/targets/`. Task: unmerged `[M+H]+`/FT,
`data/split/nist20_inchikey`. Train has 157,678 spectra from 12,668 molecules, val 52,010 from
4,221, test 52,621 from 4,223.

1. **Formula assignment.** Each target peak is mapped to the closest-mass ion formula that is a
   subformula of the precursor ion `M+H`, within 10 ppm (200 Da floor). Ion m/z is
   `mass(formula) − m_e`, which reproduces every `prec_mz` and maps 100% of the model's
   candidate m/z to a subformula within 2e-5 Da. All filtered-target peaks are assignable. The
   neutral loss is `(M+H) − ion`, and the empty loss is the precursor.
2. **Training tables.** Per formula `k` and NCE bin `b` (edges 10, 20, 30, 40, 50, 60, 80, 100,
   130), `S[k, b]` sums the unit-sum-normalized training intensity assigned to `k`. `N[k, b]`
   counts training spectra whose precursor contains `k` element-wise, i.e. where `k` was
   feasible. There are 71,541 ion formulas and 166,729 neutral-loss formulas. Formulas that occur
   in at least two training molecules hold 94% (ion) and 97% (neutral loss) of the training
   intensity.
3. **Lookup-table prior.** `E[k, b] = (S + 5·E_all[k]) / (N + 5)` for the ion formula and for the
   neutral-loss formula, and `score = 0.5·E_nl + 0.5·E_ion`. The weight was the best of
   0/0.25/0.5/0.75/1 on val, and test agrees. It is scored on two supports:
   - **own**: every subformula whose ion or neutral-loss formula was seen in training (oracle
     coverage 1.000);
   - **model**: the f1 model's own candidate m/z, so it competes with the model on identical
     support.
4. **Learned formula-only model.** `HistGradientBoostingRegressor` on 26 formula-only features per
   model candidate: the lookup estimates, log feasibility counts, ion and loss masses, mass
   fraction, RDBE of ion, loss and precursor, loss element counts, ion N/O/S counts, and NCE. It
   is fitted on the 25.6k-spectrum train sample with squared error on the L2-normalized target,
   and evaluated on 15k-spectrum val and test samples. The model is compared on the same ids.
5. **Train side.** On the model's 25.6k train sample, the lookup statistics are
   **leave-one-molecule-out**: the molecule's own intensity and feasibility counts are removed.
6. **Metric.** The candidate-space cosine of `ranking_decomp.py`: true peaks map to the nearest
   candidate within tolerance, and `cos = P·T / (|P| |T_all|)`. It reproduces the model's
   0.7618 on val exactly.

An independent audit re-derived the leave-one-molecule-out estimates (exact to 3e-18), the
assignment (400/400 against brute-force subformula enumeration), 20 feasibility counts (exact),
and the model and prior cosines on 200 val spectra (exact to 1e-9). It confirmed no molecule or
2D-InChIKey overlap across splits and no twin molecules inside train. Its scripts are in
`formula_baseline/audit/`.

## Result

| | train (25.6k) | val | test |
| --- | ---: | ---: | ---: |
| **f1 model** | 0.9607 | 0.7618 | 0.7657 |
| lookup prior, own support | 0.3814 (LOO) | 0.3787 | 0.3814 |
| lookup prior, model support | 0.4048 (LOO) | 0.4017 | 0.4043 |
| model + 20% lookup prior | 0.9581 | 0.7649 | 0.7682 |
| **learned formula-only** (15k samples) | 0.5436 (in-sample) | **0.4624** | **0.4619** |
| model on the same 15k | 0.9607 | 0.7638 | 0.7637 |
| model + 30% learned prior | 0.9574 | **0.7694** | **0.7691** |
| base-peak hit: model / lookup / learned | 0.845 / 0.251 / 0.365 | 0.567 / 0.250 / 0.294 | 0.568 / 0.254 / 0.293 |

The mixture is `(1 − α)·P/ΣP + α·R/ΣR`. `α` was chosen on val from 0.1, 0.2, 0.3 and 0.5, and test
confirms it. Paired gains, with standard errors clustered by molecule:

| blend | val | test |
| --- | ---: | ---: |
| model + 20% lookup (full splits) | +0.0031 ± 0.0003 | +0.0025 ± 0.0002 |
| model + 30% learned (15k samples) | +0.0056 ± 0.0004 | +0.0054 ± 0.0004 |

By NCE, on the 15k val sample (the lookup column is the full val split):

| NCE | n | model | learned formula-only | lookup, model support | model + 30% learned | hit: model / learned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤ 20 | 2,696 | 0.845 | 0.722 | 0.699 | 0.850 (+0.005) | 0.728 / 0.659 |
| 20-30 | 1,675 | 0.784 | 0.511 | 0.480 | 0.789 (+0.005) | 0.616 / 0.405 |
| 30-40 | 1,838 | 0.771 | 0.391 | 0.343 | 0.773 (+0.002) | 0.598 / 0.239 |
| 40-60 | 2,849 | 0.756 | 0.336 | 0.251 | 0.757 (+0.001) | 0.575 / 0.184 |
| > 60 | 5,942 | 0.723 | 0.414 | 0.336 | 0.732 (**+0.009**) | 0.479 / 0.167 |

The lookup prior has no train/val gap (0.381 against 0.379), since its train side is
leave-one-molecule-out. The learned version fits train better than val, 0.544 against 0.462.

## Reading

- **Formula statistics explain about half of the model's val cosine, and mostly at low NCE.**
  Below NCE 20 they reach 0.70-0.72, most of it the precursor's CE-dependent share. From NCE 30
  to 60 they reach 0.25-0.39, with a base-peak hit rate of 18-24% against the model's 58-60%.
  Which fragment dominates a new molecule's spectrum is not predictable from formula statistics.
  The model's validation performance is overwhelmingly structural.
- **The model already carries nearly all of the transferable formula signal.** The best blend adds
  +0.005, against a train/val gap of 0.20 and a val ranking gap of 0.12. This holds although
  MASSter's intensity token has no formula feature, since pooled embeddings and `c_mass`
  evidently recover most of it.
- **For point 4 of the GLACIER comparison** (a formula-first token without the boundary block),
  formula features are not a large untapped information source. That arm's case rests on
  removing the input that memorizes, the boundary block
  ([memorization probe](masster_memorization_route_probe_2026_09_23.md)), not on the formula
  supplying the answer. The formula terms alone are worth roughly the blend gain, about +0.005.
- **Where the formula prior does help is where the model is weakest.** Above NCE 60 the gain is
  +0.009. High-energy spectra are dominated by generic small ions whose formulas recur across
  molecules, and the model underweights that prior in the regime where its ranking error peaks
  (0.151).
- **A free blend exists.** Mixing the learned formula prior into exported predictions adds
  +0.005 with no retraining. It needs the formula tables at inference. Whether it stacks with the
  `P^0.75` rescaling (+0.007) is untested.

## Limits

- One checkpoint on one task: f1, unmerged `[M+H]+`/FT, cleaned targets. The merged `c4a_base`
  lineage is not measured.
- The model-support variant scores the model's own candidate list, which comes from structure.
  "Formula-only" describes the scoring, not the support. The own-support variant has no
  structural input at all, and scores 0.02 lower.
- Assignment takes the closest-mass subformula. The audit found a second subformula within
  10 ppm for 10% of peaks, and ¹³C isotope satellites are not de-isotoped, so they are assigned
  to the nearest wrong subformula. Both affect the model's and the baseline's targets equally,
  and put a little noise into the tables.
- The learned model is one untuned setting on simple features. A stronger formula-only learner
  might do better; the step from lookup (0.40) to learned (0.46) is the only evidence on
  headroom.
- `α` and the neutral-loss weight were chosen on val. Test confirms both.
