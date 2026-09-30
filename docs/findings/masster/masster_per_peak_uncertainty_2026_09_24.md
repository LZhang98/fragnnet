# Can an ensemble give MASSter a per-peak uncertainty?

> **Status:** SUPERSEDED · **Updated:** 2026-09-30 (was FINDING, 2026-09-24)
> **Retired 2026-09-30 with the `f0`/`f1` runs (user decision).** They trained on the old `FT`
> instrument labels, where identity routes stood in for the missing instrument input, so these
> numbers are not current evidence.
> No `i1` rerun; the [intensity reliability plan](../../plans/masster/masster_intensity_reliability_plan.md)
> carries the question.
>
> Partly. Offline, on the four raw-val exports of the
> [target-cleaning cross-eval](masster_target_cleaning_cross_eval_2026_09_23.md): f0 epochs 61
> and 63, f1 epochs 62 and 63. The members' spread at a peak does track that peak's error, and
> the members' votes predict whether a predicted peak exists in the target (76% when all four
> agree, 23% when one does). Most of that signal is spectrum-level, though. Inside one spectrum,
> at matched predicted intensity, the higher-spread peak is the worse one only **58-60%** of the
> time. Per-peak spread adds **+0.007** presence AUC over intensity plus spectrum-level
> confidence (0.793 → 0.800). The raw spread is too narrow: μ ± 2σ covers 76% of true
> intensities, and 80% coverage needs 2.3σ. Three limits bound what any per-peak interval can
> say. 14% of the squared error sits in target peaks the model does not show. Measurement noise
> is about 10% of the error. And an uncertainty head trained on training residuals would learn
> the fit, not the error.

## Method

Scripts: `output/cross_eval_mh_ft_clean_2026_09_23/analysis/per_peak_uncertainty/`
(`per_peak.py`, `analyze_per_peak.py`, `baselines.py`, `replicate_floor.py`; local, Git-ignored).
Run them from the repository root.

- **Members and set.** Unmerged `[M+H]+`/FT, `data/split/nist20_inchikey` val, raw target, 52,010
  spectra from 4,221 molecules. The four members are two runs × two adjacent checkpoints, so
  there are about two independent solutions, not four.
- **Peaks.** Per spectrum, every member's unit-sum candidates and the target are pooled and
  single-linkage clustered at the FT tolerance (10 ppm, floor at m/z 200). One cluster is one
  peak, so there are no fixed bin edges.
- **Units.** The cosine frame: the target and the ensemble mean are each L2-normalized, and the
  members share the ensemble's scale. Per-peak squared errors then sum to 2(1 − cos). The
  ensemble cosine in this frame is 0.782. Per peak: μ is the member mean, σ the member sd, and
  the error is |y − μ|.
- **Peak sets.** Predicted peaks are those at ≥ 1% of the ensemble's base peak (24.5 per
  spectrum). "Present" means the target has ≥ 1% of its own base peak in that cluster.
  Missed peaks are target peaks at ≥ 1% of its base whose μ is below 1% of the predicted base
  (7.8 per spectrum).
- **Presence models.** Logistic regressions are fitted on even `mol_id` and scored on odd, so
  fit and score use disjoint molecules.

## Result

**Where the error sits.** 85.7% of the squared error is on predicted peaks and 14.3% on missed
target peaks. Of the missed peaks, 37% have no candidate in any member. Only 19.5% reach 1% of
the base in even one member. Their median is 2.5% of the target base, and 12% are ≥ 10%.

**Spread ranks error beyond intensity.** Pooled over all spectra, inside each predicted-intensity
bin:

| μ (fraction of predicted base) | peaks | Spearman σ vs error | median relative error, low / mid / high cv tercile | present, low / mid / high cv |
| --- | ---: | ---: | --- | --- |
| 0.01-0.03 | 515,698 | 0.36 | 1.00 / 1.00 / 1.00 | 44% / 35% / 24% |
| 0.03-0.1 | 385,319 | 0.38 | 0.72 / 0.85 / 1.00 | 71% / 60% / 43% |
| 0.1-0.3 | 212,530 | 0.40 | 0.57 / 0.71 / 0.89 | 89% / 81% / 65% |
| 0.3-1.0 | 162,585 | 0.40 | 0.20 / 0.48 / 0.74 | 98% / 94% / 84% |

A relative error of 1.00 is a predicted peak that is absent from the target.

**Most of that is spectrum-level.** With spectrum fixed effects, log error on log μ has R² 0.454,
and adding log σ raises it to 0.463. Among peak pairs in one spectrum and one intensity bin
(15.6M pairs), the higher-σ peak has the larger error 59.8% of the time: 60.2%, 59.6%, 58.4% and
58.6% across the four bins. The between-run difference |μ_f0 − μ_f1| alone gives 57.5%, so the
extra checkpoints sharpen the estimate. For presence:

| features | AUC (held-out molecules) |
| --- | ---: |
| log μ | 0.755 |
| + spectrum-level: intensity-weighted cv, NCE, novelty | 0.793 |
| + per-peak: cv, member votes | 0.800 |

Presence by member votes (how many of the four put the peak at ≥ 1% of their own base): 4/4
76%, 3/4 45%, 2/4 31%, 1/4 23%.

**Calibration.** μ ± 1σ covers 45% of true intensities, μ ± 2σ 76%, and the member range 36%.
|z| reaches 2.28 at its 80th percentile and 3.74 at its 90th. Coverage of ±2σ by stratum:

| stratum | ±2σ coverage | \|z\| 80th pct | median cv |
| --- | ---: | ---: | ---: |
| Tanimoto < 0.4 | 78% | 2.11 | 0.77 |
| Tanimoto ≥ 0.8 | 74% | 2.46 | 0.46 |
| NCE < 20 | 69% | 3.13 | 0.48 |
| NCE ≥ 60 | 77% | 2.23 | 0.64 |
| μ 0.01-0.03 | 78% | 2.17 | 0.79 |
| μ ≥ 0.3 | 71% | 2.56 | 0.32 |

The spread scales with novelty about the right amount: coverage is flat across novelty while cv
nearly doubles. It is worst at low NCE and on large peaks, where the members agree and are
wrong together. That matches the known low-CE precursor default.

**Measurement noise.** Val has 1,314 molecule/NCE groups with two or more spectra at identical
NCE (1,504 pairs). Replicate cosine is 0.970 mean and 0.989 median. For replicates y₁ = t + n₁
and y₂ = t + n₂, half the replicate squared error estimates the noise share of the model's
error. On those spectra that share is 9.8%, and 19% at NCE < 20. These spectra are easier than
average (model 0.844 in this frame), so the share over all of val is likely lower.

## Reading

- **Per-peak intervals are feasible, but they must be calibrated.** The raw member spread is
  about 2.3× too narrow at 80%, and the shortfall varies by NCE and peak size. Held-out
  calibration of a normalized score, stratified by NCE and intensity, is the minimum. Split
  conformal on val is valid for test, because both are random 2D-InChIKey splits.
- **With two runs, a per-peak interval mostly tells you which spectrum to doubt.** A
  spectrum-level confidence plus an intensity-conditioned interval recovers most of the same
  information. The per-peak part is real but weak: 58-60% concordance and +0.007 AUC.
- **More members might sharpen it.** Going from the between-run difference to four checkpoints
  moved concordance from 57.5% to 59.8%. Whether five independent seeds do better is unmeasured.
- **The uncertainty has to come from disagreement or held-out residuals, not from training.**
  Measurement noise is about 10% of the error, so almost all of it is model error. Training fits
  the molecules almost exactly (train cosine 0.96), so a heteroscedastic or Dirichlet head fitted
  on training residuals would learn the fit and be overconfident on novel molecules.
- **Missed peaks need a separate output.** 14% of the error is in peaks the model does not show,
  and 37% of those have no candidate. A per-peak interval cannot flag a peak it does not list.
  Covering them needs a per-spectrum estimate of unexplained intensity, or upper bounds on
  low-intensity candidates.

## Next

- **A seed set of one config** (K = 4-5, starting from the planned seed pair), with this probe
  re-run at K = 2…5. It shows whether per-peak concordance and the AUC gain over spectrum-level
  features grow with K.
- **A held-out error model.** Fit on half of val (by molecule) and score on test. Its features
  would be μ, member spread and votes, spectrum-level confidence, NCE, novelty, and candidate
  descriptors (cut count, H shift, ion mode, fragment size). Its targets would be presence and
  quantiles of y, calibrated by conformal on the other val half. The export does not yet carry
  candidate descriptors.
- **Test replication.** Raw-test exports exist for the two best checkpoints only.

## Limits

- Two runs × two adjacent checkpoints, not independent seeds, and the members differ in
  training target. This is val only.
- "Present" is thresholded at 1% of each spectrum's base peak. A small true peak below the
  threshold counts as absent.
- The replicate groups are whatever NIST20 holds at identical NCE; their instruments and
  sources are not checked.
- Unmerged `[M+H]+`/FT only.
