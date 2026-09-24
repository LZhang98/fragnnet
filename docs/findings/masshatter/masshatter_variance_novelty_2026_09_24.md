# Is MASShatter's run-to-run variance epistemic?

> **Status:** FINDING · **Updated:** 2026-09-24
> Yes. Offline, on the f0/f1 (`3bej6fq7`/`6h67zps9`) val exports of the
> [target-cleaning cross-eval](masshatter_target_cleaning_cross_eval_2026_09_23.md). The two runs
> disagree most, and averaging them gains most, on val molecules unlike anything in training.
> Agreement falls from 0.94 to 0.75 and the ensemble gain rises from +0.005 to +0.034 as
> nearest-train-neighbour Tanimoto drops from above 0.8 to below 0.3. The effect holds inside every
> NCE bin, and novelty is uncorrelated with NCE. Adding the 17,006 NIST23-only `[M+H]+`/FT
> molecules to training would raise val molecules' mean nearest-neighbour similarity from 0.590 to
> 0.626. Mapped through the observed similarity curve, that is **+0.018** single-model cosine, a
> correlational estimate.

## Method

Scripts: `output/cross_eval_mh_ft_clean_2026_09_23/analysis/variance_novelty/{novelty,nist23_coverage}.py`
(local, Git-ignored). Run them from the repository root.

- **Per-spectrum inputs.** The cross-eval's `per_spectrum_val.pkl`: f0 and f1 single-model cosine,
  their two-run ensemble cosine (0.01 Da binned, raw target), and prediction agreement (cosine of
  f0's predicted spectrum with f1's). The task is unmerged `[M+H]+`/FT, `data/split/nist20_inchikey`,
  52,010 val spectra from 4,221 molecules.
- **Novelty.** The maximum Tanimoto of each val molecule (Morgan r=2, 2048 bits) against the
  12,668 train molecules.
- **NIST23 coverage.** NIST23 `[M+H]+`/FT molecules whose 2D InChIKey appears nowhere in NIST20,
  i.e. 17,006 molecules and 229,898 spectra. Each val molecule's novelty is recomputed against
  NIST20 train plus these molecules. The shift is mapped through today's single-model cosine per
  similarity bin.

## Result

| nearest-train Tanimoto | spectra | molecules | agreement | single | ensemble | gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤ 0.3 | 1,095 | 95 | 0.752 | 0.571 | 0.605 | +0.034 |
| 0.3-0.4 | 5,426 | 456 | 0.801 | 0.607 | 0.635 | +0.028 |
| 0.4-0.5 | 9,692 | 808 | 0.838 | 0.659 | 0.683 | +0.024 |
| 0.5-0.6 | 11,359 | 928 | 0.873 | 0.728 | 0.748 | +0.020 |
| 0.6-0.7 | 11,924 | 953 | 0.894 | 0.771 | 0.789 | +0.018 |
| 0.7-0.8 | 8,207 | 638 | 0.920 | 0.819 | 0.835 | +0.016 |
| 0.8-0.9 | 2,982 | 234 | 0.938 | 0.836 | 0.847 | +0.011 |
| > 0.9 | 1,325 | 109 | 0.941 | 0.823 | 0.828 | +0.005 |

53% of val spectra have a nearest training neighbour at or below 0.6.

Within NCE bins (cells are means):

| agreement / gain | < 0.4 | 0.4-0.6 | 0.6-0.8 | > 0.8 |
| --- | --- | --- | --- | --- |
| NCE ≤ 20 | 0.913 / +0.014 | 0.936 / +0.010 | 0.947 / +0.010 | 0.959 / +0.005 |
| NCE 20-40 | 0.814 / +0.025 | 0.881 / +0.018 | 0.923 / +0.014 | 0.946 / +0.009 |
| NCE 40-60 | 0.777 / +0.031 | 0.855 / +0.021 | 0.906 / +0.016 | 0.948 / +0.009 |
| NCE > 60 | 0.730 / +0.037 | 0.811 / +0.029 | 0.874 / +0.023 | 0.919 / +0.012 |

Spectrum-level Spearman correlations: novelty with NCE 0.001. Agreement with single-model cosine
0.62, and with ensemble gain −0.84.

Adding NIST23-only molecules to training:

| nearest-train Tanimoto | share of val now | with NIST23 |
| --- | ---: | ---: |
| ≤ 0.5 | 31.1% | 21.2% |
| 0.5-0.7 | 44.7% | 49.0% |
| > 0.7 | 24.0% | 29.8% |

Mean similarity rises from 0.590 to 0.626, and 14.1% of val molecules gain 0.1 or more. The
curve-mapped single-model cosine goes from 0.7319 to 0.7498, i.e. +0.018.

## Reading

- **The variance is epistemic.** Runs trained on the same data extrapolate differently where the
  data do not pin the answer down, which is on molecules unlike the training set and at high NCE.
  These two drivers are independent, and they add up: at NCE above 60 with novelty below 0.4,
  agreement is 0.73 and the single model scores 0.546.
- **Averaging recovers only a slice of the novelty deficit.** The most novel bin gains +0.034 but
  stays about 0.23 below the familiar bins. The remainder is bias on unfamiliar chemistry, and it
  needs information or a better inductive bias.
- **Disagreement is a usable uncertainty estimate.** Its correlation with single-model error is
  0.62. An ensemble's spread could serve as a per-spectrum confidence at no extra training cost.
- **Distillation needs extra molecules.** The members fit training molecules almost exactly
  (train cosine 0.96), so an ensemble's predictions on train are close to the labels, and a
  student distilled on train alone has little to gain. The ensemble knows more than one model only
  where members disagree, which is on novel molecules. Distilling onto unlabeled structures that
  cover that region, such as NIST23-only molecules or PubChem, is the version with something to
  transfer.
- **More molecules target the same region.** The NIST23 estimate (+0.018) is about the size of
  the two-run ensemble gain. Unlike the ensemble, it would also reduce the bias term, if the
  similarity curve is causal.
- **The optimizer-noise share still applies.** The cross-eval puts about 40% of the two-run gain
  in within-run noise at constant LR. That part is what LR decay (`d3`), EMA (`t2`) and
  weight averaging act on.

## Next

- **A pure seed pair of one config** prices the between-run share without the target confound.
- **Run-to-run agreement under `n1`.** If the boundary block memorizes arbitrary per-molecule
  keys, each run keys differently, so removing it should raise agreement between seeds.
- **A mixture of experts, later.** An MoE could build this spread into a single model. The gain
  measured here comes from averaging members whose errors are largely independent. A hard gate
  that sends each input to one specialized expert splits the training molecules, which could
  raise variance on novel ones. Designs that keep averaging keep that property, e.g. several
  heads over a shared encoder whose outputs are averaged or softly weighted by a learned gate
  (inputs could include novelty and NCE). Disagreement among the heads would also give the
  per-spectrum confidence noted above.

## Limits

- One pair of runs, and not seed replicates: f0 and f1 share a seed and differ in target
  cleaning. A pure seed pair is still unmeasured.
- The cosine here is 0.01 Da binned, not `cos_hun`, and the bin edges add per-spectrum noise.
- The NIST23 estimate is correlational. Novel molecules may be intrinsically harder (a bias the
  curve would carry), and extra data can also help familiar molecules (a gain the curve
  misses). It is a planning number, not a prediction.
- Unmerged `[M+H]+`/FT only. The merged `c4a_base` lineage is unmeasured.
