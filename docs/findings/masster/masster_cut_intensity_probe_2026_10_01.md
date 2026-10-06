# Train-only cut scores do not improve frozen intensity allocation

> **Status:** CLOSED NULL · **Updated:** 2026-10-01
> Full offline screen complete.
> This closes the tested additive boundary-score correction, not all cut-supervision designs.

## Decision

Do not build a cut-supervision training arm on the strength of this probe. Tuning selected
**zero correction** for both the encoder and handmade-feature probes. Every tested positive
strength lowers overall confirmation cosine. Even the smallest nonzero strength slightly harms
the independently selected floor. Existing proposal-score mixing also selects zero.

The positive control is a modest fragment-temperature adjustment: multiplying the frozen
fragment logits by 0.9, preserving the conditional ion/H distribution, adds 0.00284 confirmation
cosine. This is below the predeclared 0.005 practical screen threshold, and supplies no evidence
for useful cut-specific correction.

## Design and population

The [frozen protocol](../../plans/masster/masster_cut_intensity_probe.md) was recorded before
fitting and the full scoring read. Model: `i1` (`i50ig7rf`) epoch 59, NIST20 `[M+H]+` Orbitrap.
No MASSter weights, training targets, original datasets or original checkpoints were modified.

- Training labels are weak boundary-cut frequencies over **all distinct stored compatible
  atom sets** at the measured base-peak mass, with uniform credit. Duplicate H/ion channels
  and predicted shares do not weight labels. This includes 38,414 spectra, 27,764 with multiple
  atom sets; 10,789 pure no-cut base groups are omitted. These are not identified reaction sites.
- Probe fitting: 3,177 training molecules, 9,408 capped spectra and 188,161 bond rows.
  Separate calibration: 794 training molecules, 2,357 spectra and 47,031 bond rows. Both
  feature families use fixed 100-iteration gradient boosting and scalar logistic calibration.
- Validation tuning: 26,052 spectra / 2,111 molecules. Locked confirmation: **25,968 spectra /
  2,111 molecules**. Identity overlap is zero between train and validation, fit and calibration,
  and tuning and confirmation, checked with InChIKey first blocks.
- Confirmation includes 2,949 unfamiliar spectra / 245 molecules (nearest-training Tanimoto
  <=0.4), and 137 floor spectra / 52 molecules selected by **`fa1` epoch 63**, independent
  of the corrected `i1`. The full independent selector has 290 spectra.
- This validation set has been inspected in earlier research. Confirmation is held out from
  this correction's selection, not historically untouched. Canonical test spectra were unused.

Fresh annotated CPU inference covers every one of the **52,020 validation spectra**. Each
fragment's correction is the sum of calibrated per-bond log odds over its original-molecule
boundary bonds. Every emitted fragment and ion/H channel stays available; no target-dependent
candidate filtering or target-selected structure enters the correction. Matching uses summed
4-decimal masses, instrument tolerance (10 ppm, 200 Da floor), complete unmatched-peak norms,
and one-to-one maximum-product assignment.

## Confirmation results

Unchanged confirmation instrument-tolerance cosine is **0.791994**; base-peak hit rate is
**60.9481%**. The table's nonzero strengths are fixed points on the predeclared grid, not arms
selected after seeing confirmation. Intervals are 2,000 paired molecule-cluster bootstrap draws;
effects are spectrum-weighted.

| Correction | Strength | Confirmation cosine delta [95% CI] |
| --- | ---: | ---: |
| Selected cut probe | 0 | 0.000000 [0, 0] |
| Encoder cut probe | 0.01 | -0.000082 [-0.000142, -0.000024] |
| Handmade cut probe | 0.01 | -0.000104 [-0.000168, -0.000045] |
| Encoder cut probe | 0.10 | -0.003410 [-0.004004, -0.002825] |
| Handmade cut probe | 0.10 | -0.003816 [-0.004458, -0.003209] |
| Independently tuned cut-count control | +0.05 | -0.000082 [-0.000270, +0.000105] |
| Independently tuned fragment-temperature control | -0.10 (logits x 0.9) | +0.002836 [+0.002543, +0.003129] |

At strength 0.1, base-peak hit rates fall by 0.5314 percentage points for encoder scores and
0.5276 points for handmade scores. Molecule-weighted overall deltas have the same sign
(-0.003087 and -0.003481). Proposal mixing and both per-bond shuffled controls select zero.
Shuffling preserves each spectrum's per-bond score distribution and fragment cut counts.

| Confirmation subset | Encoder alpha 0.1 | Handmade alpha 0.1 | Temperature control |
| --- | ---: | ---: | ---: |
| Unfamiliar molecules | -0.002443 [-0.004004, -0.000936] | -0.002407 [-0.004105, -0.000784] | +0.004517 [+0.003525, +0.005445] |
| Independent floor | -0.002187 [-0.004132, -0.000626] | -0.002759 [-0.004999, -0.000970] | +0.005091 [+0.003229, +0.007321] |

The smallest strength 0.01 also lowers floor cosine: encoder -0.000235 and handmade -0.000294,
with both intervals below zero. The temperature control changes no floor base-peak hits and
slightly lowers the overall hit rate. Its benefit is spectral shape, not demonstrated recovery
of the missing dominant fragment.

## Validation, artifacts and limits

13 diagnostic tests and 10 existing fragment-comparison tests passed; Ruff and
`git diff --check` passed. Tests cover ambiguous-label weighting, identity aliases, one-to-one
matching, unmatched norms, summed channel winners, fixed ion/H ratios, zero correction,
calibration, molecule bootstrap, and tuning-only arm selection. Full model tests were not run:
production model and training behavior did not change.

The independent artifact audit confirms complete unique spectrum coverage, finite cosines in
[0,1], binary hits, and exact zero-strength agreement for every spectrum. Direct production-helper
agreement on 32 real spectra has maximum error 6.66e-16. All fresh/cached bond orders and emitted
fragment memberships were checked. CPU baseline mean is 0.788038 versus cached GPU 0.788036;
paired mean difference is +1.47e-6, mean absolute difference 7.72e-6 and the 99th percentile
absolute difference 7.30e-5. Backend variation is far smaller than the failed alpha-0.1 effects.

Local ignored artifacts are under `output/cut_intensity_probe_2026_10_01/`:

- `result.json`, `locked_choices.json`, `sweep.csv`, `scores.pkl`, `artifact_audit.json`;
- `split_manifest.json`, `validation_manifest.pkl`, `bond_log_odds.pkl`, and both
  `probe_*.joblib` files;
- `provenance.json`, `score_audit.json`, `environment.json`, source/config snapshots and logs.

Source revision: `ab0adfb46f237bb1ab408370b660c7655b046dfa` plus this task's uncommitted
diagnostic. Input and source hashes are recorded. Fit and score source hashes differ because
scoring audit additions and formatting followed the fit; the fitting algorithm did not change.
Interpreter: FRAGNNET-GPU Python 3.11.13, torch 2.4.1+cu121, scikit-learn 1.7.2. Full CPU
scoring took 1,028.4 seconds. No cluster run was submitted.

Labels inherit a rounded-mass, model-generated candidate table, not independent structural
identification. The encoder probe uses 128 PCA components retaining 87.8% of sampled training
variance. The result concerns two fixed regressors, positive additive strengths and one
checkpoint; it does not measure a learned interaction over multi-cut fragments, another label
definition, or a cut auxiliary's training effect. No such training gain is established.
