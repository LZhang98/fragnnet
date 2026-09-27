# Does MASSter still have the ranking problem?

> **Status:** FINDING · **Updated:** 2026-09-23
> Yes. Offline decomposition of the `6h67zps9` (f1) and `3bej6fq7` (f0) predictions exported by
> the [target-cleaning cross-eval](masster_target_cleaning_cross_eval_2026_09_23.md); no new
> inference. Coverage is 6% of the cosine gap. Candidate ordering that no order-preserving
> rescale can fix is 51%, and the rest is the ceiling for rescaling. The base peak is missed in
> 43% of spectra, and 97% of those misses are peaks the candidate set already holds. The same
> low-CE precursor default FraGNNet showed is present. What differs from FraGNNet: MASSter is
> slightly *too sharp*, so a global power of 0.75 on the predictions adds **+0.007** for free,
> fixed on val and confirmed on test. **On training molecules the ranking error nearly vanishes**
> (ranking 0.011 against 0.121, base-peak hit 84.5% against 56.7%). The ranking problem is a
> generalization problem, so a ranking loss on the training set has little left to fit.

## Method

`run_save_inference_eval.py` exports MASSter's **full candidate set**: about 1,900 rows per
spectrum, or about 570 distinct m/z once rows sharing a formula are merged. `--min_peak_int` is a no-op for this model (`script_utils.py:2236` filters only when
`frag_formula_peak_idxs` exists). So coverage can be read from the export directly, unlike the
FraGNNet [dominant-peak decomposition](../fragnnet/dominant_peak_outcome_decomposition.md), which
had to rebuild support from DAG files.

Per spectrum, candidates sharing an exact m/z are merged. Each target peak is mapped to its
nearest candidate within the FT tolerance (10 ppm, 200 Da floor). With `T` the target intensity on
candidates and `P` the prediction, the gap `1 − cos` splits into three non-negative parts:

| part        | definition                    | what fixes it                                        |
| ----------- | ----------------------------- | ---------------------------------------------------- |
| coverage    | `1 − oracle`, where `oracle = ‖T_covered‖ / ‖T‖` | more candidates                 |
| ranking     | `oracle − iso`                | a different *order* of candidates                    |
| calibration | `iso − cos`                   | an order-preserving rescale of the model's scores    |

`iso` is the cosine of the isotonic projection of `T` onto `P`'s rank order. That projection
maximizes cosine over all vectors that preserve the model's order, so `iso` is the ceiling for any
temperature, power or monotone recalibration. It is chosen per spectrum with the target in hand,
which makes the calibration share an upper bound. The model cosine in this space reproduces the
pipeline's `cos_hun` to within 0.0006 in every variant.

Script: `output/cross_eval_mh_ft_clean_2026_09_23/analysis/ranking_decomp.py` (local, Git-ignored,
about 40 s per variant). Per-spectrum tables sit beside it.

## Result

f1 on its cleaned target (the target it was trained and selected on):

| split | cos_hun | coverage     | ranking          | calibration      | base hit  | under-ranked | uncovered |
| ----- | ------: | ------------ | ---------------- | ---------------- | --------: | -----------: | --------: |
| val   |  0.7618 | 0.014 (6%)   | **0.121 (51%)**  | 0.104 (44%)      |     0.567 |    **0.416** |     0.017 |
| test  |  0.7657 | 0.014 (6%)   | **0.117 (50%)**  | 0.103 (44%)      |     0.568 |    **0.414** |     0.018 |

The raw target, the f0 control, the two-checkpoint weight average and the two-run ensemble all give
the same split to within 0.01. Target cleaning does not touch it.

- **Base peak: ranking beats coverage 24:1.** FraGNNet `pqx7ywc9` on the NIST20 test set measured
  0.573 / 0.367 / 0.052 (7:1). The hit rate is essentially the same, and the difference is that
  MASSter's candidate set misses less. That comparison is loose: FraGNNet's set spans four
  adducts and both instruments, while this one is `[M+H]+`/FT.
- **Most misses are near misses.** When the base peak is missed, it is the model's #2 candidate in
  37% of cases (test 38%) and in its top 3 in 52% (test 54%). Its median predicted height is 0.30
  of the predicted maximum. 72% of misses land within 60 Da of the true base peak, but only 19% are
  exactly one common neutral loss away (H₂O 5.2%, CO 2.6%, NH₃ 2.1%).
- **Order is only moderately right.** Spearman on covered target peaks is 0.636, against FraGNNet's
  0.613.

### Ranking is the part that grows with collision energy

f1, filtered val (test within 0.01 in every cell):

| NCE     | spectra | cos_hun | coverage | ranking   | calibration | base hit | under-ranked | uncovered |
| ------- | ------: | ------: | -------: | --------: | ----------: | -------: | -----------: | --------: |
| ≤ 20    |   9,299 |   0.838 |    0.003 | **0.065** |       0.095 |    0.714 |        0.283 |     0.003 |
| 20–30   |   5,814 |   0.786 |    0.007 |     0.100 |       0.107 |    0.616 |        0.377 |     0.007 |
| 30–40   |   6,204 |   0.768 |    0.010 |     0.114 |       0.108 |    0.604 |        0.384 |     0.012 |
| 40–60   |   9,916 |   0.755 |    0.013 |     0.127 |       0.104 |    0.578 |        0.408 |     0.015 |
| > 60    |  20,777 |   0.722 |    0.022 | **0.151** |       0.105 |    0.471 |        0.499 |     0.030 |

Calibration is flat across energy. Ranking more than doubles.

### The low-CE precursor default is still there

At NCE ≤ 20 the precursor is the true base peak in 64% of spectra and the predicted top in 71%
(test 64% and 73%). The base-peak hit rate is 0.88 when the precursor is the base peak and 0.42
when it is not (test 0.89 and 0.42). In 69% of the second group's misses, the model predicts the
precursor. The same over-call, about ×1.1, runs through 20–60 NCE. FraGNNet showed the same
pattern: 0.951 vs 0.362 ([dominant-peak decomposition](../fragnnet/dominant_peak_outcome_decomposition.md)).

### Calibration: slightly too sharp, and a global temperature pays

Entropy over covered peaks is 0.15 nats *below* the target's (−0.12 for f0). FraGNNet was 0.60
nats too flat, and `γ = 1` was optimal for it. Power-rescaling MASSter's predictions, `P^γ`, with
`γ` read off a val grid of 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2 and 3:

| variant                   | γ = 1  | γ = 0.75 | gain        |
| ------------------------- | -----: | -------: | ----------- |
| f1, filtered val          | 0.7618 |   0.7694 | +0.0076     |
| f1, filtered **test**     | 0.7657 |   0.7728 | **+0.0071** |
| f0, raw **test**          | 0.7604 |   0.7666 | **+0.0062** |
| f1 weight average, raw val| 0.7600 |   0.7673 | +0.0073     |
| f1 + f0 ensemble, raw val | 0.7772 |   0.7814 | +0.0042     |

`γ = 0.75` is the best grid point in every variant. It stacks with weight averaging. The ensemble
already flattens (entropy −0.08) and gains less. With both calibrated, the ensemble still leads
the single model by +0.017, so the ensemble gain is mostly not flattening.

Temperature is a small slice of the calibration ceiling. A global γ gets +0.007, the per-spectrum
best γ +0.039 (FraGNNet +0.035), and the per-spectrum isotonic ceiling +0.104.

### Averaging does not solve ranking

The two-run ensemble cuts ranking from 0.120 to 0.110 and calibration from 0.103 to 0.095, and
moves the base-peak hit rate from 0.562 to 0.582. The weight average moves ranking by 0.002. Both
leave a ranking gap of about 0.11.

### Train versus validation: the ranking error is a generalization error

The same decomposition was run on a train-split sample: f1 on its cleaned target, eval mode, the
first 100 batches of 256 from the train loader (25,600 spectra), same snapshot and command.

| f1, cleaned target  | train (25.6k) | val (52.0k) |
| ------------------- | ------------: | ----------: |
| cos_hun             |         0.961 |       0.762 |
| coverage            |         0.009 |       0.014 |
| **ranking**         |     **0.011** |   **0.121** |
| calibration ceiling |         0.019 |       0.104 |
| base-peak hit       |         84.5% |       56.7% |
| under-ranked        |         14.3% |       41.6% |
| Spearman            |          0.85 |        0.64 |
| entropy excess      |         −0.12 |       −0.15 |

It holds within every stratum. The train ranking gap is 0.003-0.019 across the five NCE bins
(val 0.065-0.151) and 0.004-0.024 across five precursor-mass bins (val 0.108-0.156). Candidate
and peak counts match: median 563 vs 568 distinct candidate m/z and 18 vs 18 target peaks. The
train sample comes from a seeded `RandomSampler` with no size-based selection. (*Corrected
2026-09-27:* this line earlier said the train loader skips contexts too large for its batch
budget. That applies only to the dynamic samplers, which this export does not use;
[audit](masster_memorization_attempts_audit_2026_09_27.md).) No sample selection could produce a
tenfold gap that holds bin by bin.

So the model ranks training molecules nearly correctly and new molecules poorly. The
over-sharpness is the same on both splits, so the temperature fix is a separate, split-independent
calibration effect.

## Reading

- **The ranking problem carries over from FraGNNet to MASSter nearly unchanged.** The
  candidate set is not the limit: coverage is 0.014, and the base peak is uncovered in under 2% of
  spectra. The model fails to put the right candidate on top. It fails most at high CE, and at low
  CE it defaults to the precursor.
- **A ranking loss on the training set is unlikely to help.** Half the validation gap is out of
  reach for any order-preserving transform, but on training molecules the order is already right
  (ranking 0.011). A pairwise or listwise term fitted on train has almost nothing to push on, and
  the cross-entropy already gets the train ranking right. It would join the objective-reshaping
  arms that have all landed at or below control (JSS, KL, Hellinger, direct cosine at −0.028).
  The [sibling-ranking plan](../../plans/fragnnet/sibling_ranking_target_plan.md) should be read
  in that light before it is built for either model. No ranking loss exists in `src/`.
- **The lever is making the ranking transfer.** Candidates, in order of evidence:
  - limiting the per-molecule allocation lookup the
    [shortcut audit](masster_overfit_readout_2026_09_22.md#7-intensity-memorization-threat-model)
    leaves on this lineage, at the allocator as a whole. Blocking only the boundary block the
    [memorization probe](masster_memorization_route_probe_2026_09_23.md) located can reroute it,
    because identity reaches every head input;
  - variance reduction: the two-run ensemble cuts val ranking from 0.120 to 0.110, and EMA, SWA or
    LR decay target the same noise;
  - information that generalizes across molecules, such as bond-lability priors not learned from
    this data, or more training molecules.

  Judge each on the val ranking gap and base-peak hit rate, with the train side measured too.
  The first two are prepared as one-key arms in the
  [ranking-transfer plan](../../plans/masster/masster_ranking_transfer_plan.md).
- **A free +0.007 is available now.** Exporting `P^0.75` instead of `P` raises cosine on both
  checkpoints and both splits, and it stacks with weight averaging. It reflects a
  metric-versus-objective mismatch, since cross-entropy training does not optimize cosine. It does
  not fix the ranking.

## Limits

- One architecture, unmerged `[M+H]+`/FT, two checkpoints from one seed. The merged `c4a_base`
  lineage is not measured.
- Only a coarse γ grid was tried; the optimum lies between about 0.6 and 0.9.
- The FraGNNet comparison crosses splits, scopes and metrics (FraGNNet used `cos_hun_inst_tol`
  on the test set), so it is qualitative.
