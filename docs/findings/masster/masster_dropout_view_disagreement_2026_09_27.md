# Dropout-view disagreement: pricing R-Drop and a logit penalty for MASSter

> **Status:** FINDING · **Updated:** 2026-09-28
> Inference only; no training. The first gate of the
> [generalization-methods note](../../notes/masster_generalization_methods_2026_09_27.md) (§1 and
> its second cheap check), on four weight sets with 4,096 seeded-random spectra per split. Two
> passes of the allocator under its own training dropout move 7-8% (`c0`) to 13-16% (`k1` family)
> of predicted intensity, so R-Drop would have something to penalize. But the disagreement is
> almost as large on training molecules as on validation ones, and it follows collision energy
> rather than novelty. Logit spread is the same on both splits. Neither proposal has the
> train-specific target its rationale assumes.
> *Many-view follow-up (2026-09-28, [below](#many-view-follow-up-the-mc-dropout-mean-2026-09-28)):*
> the first read, that dropout views gain on validation only what flattening gains, came from
> two-view averages, which keep half a single view's noise. Eight head views measure the mean
> directly. Flattened, it beats the best flattened deterministic prediction on val by +0.0031
> on the new control `i1` and +0.0049 on `k1_ema`, and it costs nothing on train. R-Drop's own
> target, the gap between the deterministic net and its dropout mean, is therefore real but
> small on `i1`.

## Answer

- **Magnitude passes.** Two allocator passes on one candidate set move 7.3% of intensity on
  training spectra under `c0` (symmetric KL 0.031 nats) and 13-14% under `k1`/`k1_ema` (0.10-0.12).
  The views agree on the top peak 82-89% of the time.
- **It is not specific to unseen molecules.** Validation disagreement exceeds training by only
  8-15%. Within each split it tracks error to a similar degree (Spearman 0.55-0.58 on train,
  0.49-0.51 on val), and it doubles from NCE below 30 to NCE 100 and up.
- **Novelty barely moves it.** On validation, two dropout views agree at cosine 0.92-0.97 (`k1`
  family) or 0.98-0.99 (`c0`) from the least to the most familiar molecules. Two independently
  seeded runs fall from 0.94 to 0.75 across the same range
  ([variance finding](masster_variance_novelty_2026_09_24.md)). Dropout noise is not the
  epistemic variance that ensembles exploit.
- **Two-view averages: the gain looked like flattening (corrected by the many-view follow-up).**
  - The average of two views beats the deterministic prediction on val by +0.003 to +0.005 and
    does not help on train.
  - `P^0.75` of the deterministic prediction gains +0.005 to +0.007 on val.
  - Once flattened, the two-view average adds nothing over flattened eval for the `k1` family, and
    +0.002 for `c0`.
  - *Corrected:* a two-view average still carries half a single view's noise. The eight-view mean,
    flattened, beats flattened eval on val by +0.0049 on `k1_ema` and +0.0031 on `i1`
    ([follow-up](#many-view-follow-up-the-mc-dropout-mean-2026-09-28)).
- **The best-known levers do not act through this quantity.** `k1` beats `c0` by +0.0084 on val
  while doubling the disagreement. EMA beats its own training weights by +0.0119 and lowers the
  disagreement by only 7-9%.
- **The logit penalty has no train-specific target.** The centered spread of the allocation
  log-probabilities is the same on train and val in all four weight sets. About half (51-55%) of
  the proposed unweighted penalty comes from the ~80% of candidates that together hold under 1%
  of the predicted intensity.

## Setup

**Probe.** [`debug_masster_dropout_view_disagreement.py`](../../../scripts/debug/masster_memorization/debug_masster_dropout_view_disagreement.py),
with unit tests in `tests/test_debug_masster_dropout_view_disagreement.py`. Each batch runs five
conditions, paired:

| condition | what is stochastic | candidate set |
| --- | --- | --- |
| `eval` | nothing: the production prediction | eval |
| `head` ×2 | the cut-bond encoder and the candidate allocator in train mode, each view its own dropout draw; every proposal module in eval mode | the eval set, for both views |
| `full` ×2 | every dropout layer; BatchNorm stays on its running statistics, which are never updated | redrawn per view |

- The model's own training flag stays off, so every condition uses the production output.
  The eval-time channel-selection thresholds are 0 in these configs.
- Views are compared on the dense output: every chemically valid (fragment, ion mode, H shift)
  pathway, which is what the training loss scores.
- Pathways closer than 0.5 mDa are summed into one peak group. Isomeric fragments and equivalent
  H-shift channels therefore count as one peak, since the spectrum constrains only their total.
- Group-level and 0.01-Da total variation agree to 1e-4 on average (95% of spectra within 2e-4).
  So the disagreement is between separated peaks, not near-isobars.

**Weight sets.** `[M+H]+`/FT, `data/split/nist20_inchikey`, unfiltered targets:

| label | run | epoch | dropout live in `head` | val cosine: this sample / logged |
| --- | --- | ---: | --- | --- |
| `c0` | `vcpho5qm` | 42 | allocator 0.1 | 0.7318 / 0.7292 |
| `k1` | `wg0nxymi` | 61 | allocator 0.1, token 0.2, cut-bond encoder 0.2 | 0.7407 / 0.7365 |
| `k1_ema` average | `j4u25qfd` | 50 | as `k1` | 0.7434 / 0.7427 |
| `k1_ema` training weights | `j4u25qfd` (`ema_training_state_dict`) | 50 | as `k1` | 0.7311 / not logged |

`full` adds the encoder's dropout (0.1) and the bond and charge heads' (0.1) to each set's head
sites. The epochs differ, and each run's disagreement is read against its own eval prediction.

**Sample and statistics.**

- 4,096 spectra per split, drawn with seeds 20260927 (train) and 20260928 (val), the same
  spectra for every weight set. Contrasts between weight sets are therefore paired by spectrum.
- 0.8-1.4% of spectra per split are dropped: there a budget tie gave a head view another isomer
  (side finding 1).
- Intervals are 95% molecule-clustered bootstraps.
- The dense cosine reproduces the logged `cos_sim_0.01` to 2e-6 per spectrum (side finding 2).

**Provenance.**

- Revision `394dbab4`, with other sessions' uncommitted `src/` edits present (SMARTS CUDA port,
  restricted allocators; `git diff -- src` hash `92465c4e`). All are off or bit-identical in these
  configs. The logged val metrics above reproduce.
- Torch 2.4.1+cu121 on the local RTX 2070 SUPER.
- Checkpoint SHA-256 prefixes: `41e923a3` (`c0`), `7b599ae7` (`k1`) and `eea40c09` (`k1_ema`,
  the pinned file of the [reliability plan](../../plans/masster/masster_intensity_reliability_plan.md)).
- That plan's rule against enabling dropout concerns estimates of *measurement* noise. This probe
  measures model noise on purpose.

## Results

### How much two passes disagree

Per-spectrum means. TV is the share of predicted intensity moved between the two views. The last
column is the eval prediction's own 0.01-Da TV to the target: the model's error in the same units.

| weights | split | cosine | head TV | head sym. KL (nats) | head top-1 agree | full TV (0.01 Da) | full churn | eval TV to target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c0` | train | 0.9171 | 0.073 | 0.031 | 89% | 0.123 | 0.30% | 0.208 |
| `c0` | val | 0.7318 | 0.079 | 0.036 | 89% | 0.144 | 0.43% | 0.407 |
| `k1` | train | 0.9129 | 0.136 | 0.106 | 83% | 0.170 | 0.32% | 0.214 |
| `k1` | val | 0.7407 | 0.155 | 0.136 | 80% | 0.193 | 0.50% | 0.398 |
| `k1_ema` average | train | 0.9194 | 0.131 | 0.098 | 82% | 0.161 | 0.24% | 0.203 |
| `k1_ema` average | val | 0.7434 | 0.150 | 0.126 | 82% | 0.191 | 0.38% | 0.393 |
| `k1_ema` training weights | train | 0.9054 | 0.144 | 0.115 | 82% | 0.178 | 0.33% | 0.230 |
| `k1_ema` training weights | val | 0.7311 | 0.162 | 0.145 | 78% | 0.205 | 0.48% | 0.407 |

- `full` churn is the intensity on peak groups present in only one view. At 0.2-0.5% of
  intensity, whole-model dropout changes which candidates exist but almost none of the predicted
  intensity rides on the change. So the note's worry about candidate churn for whole-model
  R-Drop is small in mass terms.
- A single `full` view costs train cosine 1.6 to 4 times what a `head` view costs. The encoder's
  dropout matters more for fit than the allocator's.

### Where the disagreement sits

- **Collision energy.** Head TV roughly doubles from NCE below 30 to NCE 100 and up, alike on both
  splits: 0.045 to 0.102 (train) and 0.045 to 0.107 (val) for `c0`, 0.087 to 0.179 and 0.095
  to 0.193 for `k1`. Higher-energy spectra spread intensity over more peaks, so more of it can move.
- **Error.** Spearman correlation of head TV with 1 - cosine: 0.55-0.58 on train, 0.49-0.51 on
  val. That makes it a difficulty signal on both splits, not an unseen-molecule signal.
- **Novelty.** Validation dropout-view agreement (cosine between the two head views' peak
  distributions) and gains, by maximum Morgan-2 Tanimoto to any training molecule:

| max Tanimoto (val) | ≤ 0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.8 | > 0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c0`: dropout-view agreement | 0.983 | 0.984 | 0.988 | 0.989 | 0.991 |
| `k1`: dropout-view agreement | 0.929 | 0.940 | 0.957 | 0.963 | 0.969 |
| `k1_ema` average: dropout-view agreement | 0.932 | 0.945 | 0.958 | 0.965 | 0.973 |
| `k1_ema` training weights: dropout-view agreement | 0.921 | 0.933 | 0.952 | 0.957 | 0.965 |
| `c0`: `P^0.75` gain | +0.0098 | +0.0079 | +0.0052 | +0.0026 | +0.0014 |
| `c0`: head-average gain | +0.0033 | +0.0055 | +0.0045 | +0.0044 | +0.0023 |
| `k1_ema` average: `P^0.75` gain | +0.0119 | +0.0094 | +0.0066 | +0.0026 | +0.0011 |
| `k1_ema` average: head-average gain | +0.0138 | +0.0028 | +0.0027 | +0.0038 | +0.0013 |
| val spectra | 80 | 1,124 | 1,823 | 678 | 340 |

The [variance finding](masster_variance_novelty_2026_09_24.md)'s two-seed agreement runs from
0.75 to 0.94 over the same range (f0/f1, 0.01-Da cosine of the two predictions). The flattening
gain rises steadily with novelty; the head-average gain has no consistent trend outside the
80-spectrum first bin.

### Cosine of dropout views, their average and flattened predictions

Dense 0.01-Da cosine to the target, minus the deterministic eval prediction's, paired per
spectrum. "Average" is the mean of the two views' distributions (two-sample MC dropout). `P^γ`
flattens exact peak groups and renormalizes.

| weights | split | one head view | head average | one full view | full average | `P^0.875` | `P^0.75` | head average `^0.875` | head average `^0.75` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c0` | train | −0.0023 [−0.0028, −0.0017] | +0.0000 [−0.0005, +0.0006] | −0.0091 [−0.0101, −0.0082] | −0.0032 [−0.0040, −0.0022] | −0.0004 [−0.0007, −0.0001] | −0.0064 [−0.0070, −0.0058] | −0.0021 [−0.0027, −0.0014] | −0.0104 [−0.0113, −0.0096] |
| `c0` | val | +0.0028 [+0.0017, +0.0038] | +0.0046 [+0.0036, +0.0056] | −0.0009 [−0.0026, +0.0008] | +0.0052 [+0.0035, +0.0068] | +0.0043 [+0.0038, +0.0048] | +0.0053 [+0.0041, +0.0064] | +0.0077 [+0.0065, +0.0089] | +0.0070 [+0.0052, +0.0088] |
| `k1` | train | −0.0117 [−0.0127, −0.0107] | −0.0043 [−0.0052, −0.0034] | −0.0189 [−0.0203, −0.0176] | −0.0075 [−0.0087, −0.0062] | +0.0002 [−0.0000, +0.0005] | −0.0048 [−0.0053, −0.0042] | −0.0061 [−0.0071, −0.0052] | −0.0144 [−0.0155, −0.0133] |
| `k1` | val | −0.0046 [−0.0063, −0.0028] | +0.0026 [+0.0009, +0.0044] | −0.0103 [−0.0125, −0.0079] | +0.0006 [−0.0017, +0.0028] | +0.0046 [+0.0040, +0.0051] | +0.0061 [+0.0050, +0.0072] | +0.0055 [+0.0037, +0.0074] | +0.0044 [+0.0023, +0.0064] |
| `k1_ema` average | train | −0.0092 [−0.0100, −0.0083] | −0.0023 [−0.0030, −0.0015] | −0.0144 [−0.0155, −0.0134] | −0.0041 [−0.0052, −0.0031] | +0.0000 [−0.0002, +0.0003] | −0.0051 [−0.0057, −0.0046] | −0.0045 [−0.0053, −0.0037] | −0.0131 [−0.0141, −0.0121] |
| `k1_ema` average | val | −0.0036 [−0.0054, −0.0019] | +0.0030 [+0.0013, +0.0048] | −0.0061 [−0.0084, −0.0039] | +0.0045 [+0.0023, +0.0067] | +0.0047 [+0.0042, +0.0052] | +0.0064 [+0.0052, +0.0075] | +0.0061 [+0.0043, +0.0079] | +0.0050 [+0.0029, +0.0070] |
| `k1_ema` training weights | train | −0.0124 [−0.0135, −0.0113] | −0.0042 [−0.0053, −0.0031] | −0.0200 [−0.0215, −0.0186] | −0.0077 [−0.0091, −0.0065] | +0.0004 [+0.0001, +0.0007] | −0.0043 [−0.0049, −0.0037] | −0.0061 [−0.0072, −0.0050] | −0.0144 [−0.0157, −0.0132] |
| `k1_ema` training weights | val | −0.0047 [−0.0066, −0.0028] | +0.0030 [+0.0011, +0.0049] | −0.0087 [−0.0112, −0.0063] | +0.0036 [+0.0011, +0.0061] | +0.0049 [+0.0043, +0.0054] | +0.0068 [+0.0056, +0.0079] | +0.0061 [+0.0041, +0.0080] | +0.0052 [+0.0030, +0.0075] |

- On train, every perturbation costs fit or is neutral. `P^0.875` stays within 0.0004 of zero
  there.
- On val, a single `k1`-family view is worse than the deterministic prediction. A `c0` view is
  better, by what a milder flattening gives.
- The two-view average gains about what flattening gains. Flattening the average does not beat
  flattened eval for the `k1` family (best +0.0055-0.0061 against +0.0061-0.0068). `c0` keeps
  +0.0024 beyond the best eval flattening on this grid.
- The flattening gain grows with novelty (table above). It is a hedge on allocations the model
  gets wrong, which are concentrated on unfamiliar molecules.

### Paired contrasts

Same spectra, weight set a minus weight set b:

| contrast | split | spectra | cosine | head TV | full TV (0.01 Da) | peak-group logit std |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `k1` − `c0` | train | 4,018 | −0.0044 [−0.0060, −0.0029] | +0.0624 [+0.0605, +0.0642] | +0.0469 [+0.0446, +0.0492] | −0.186 [−0.196, −0.177] |
| `k1` − `c0` | val | 4,008 | +0.0084 [+0.0034, +0.0138] | +0.0761 [+0.0737, +0.0784] | +0.0491 [+0.0465, +0.0519] | −0.206 [−0.218, −0.195] |
| `k1_ema` average − training weights | train | 4,022 | +0.0140 [+0.0127, +0.0153] | −0.0129 [−0.0149, −0.0110] | −0.0176 [−0.0200, −0.0153] | −0.009 [−0.014, −0.005] |
| `k1_ema` average − training weights | val | 4,014 | +0.0119 [+0.0092, +0.0145] | −0.0115 [−0.0138, −0.0093] | −0.0132 [−0.0160, −0.0105] | −0.003 [−0.008, +0.003] |

- `k1`'s generalization gain comes with twice the dropout disagreement and a narrower logit range,
  on both splits.
- EMA's gain, the largest single-model lever in the record, comes with 7-9% less disagreement.
  That is too little to attribute the gain to dropout stability.

### Logit spread and the proposed centered-logit penalty

`z_f` is the allocation logit. Under `masster_proposal_selection_only` it is the head's
`mask_residual_head` output, and the valid-fragment log-probabilities equal `z_f` less a
per-spectrum constant.

| weights | split | std of `log P(f)` | `Var_P(z)` | peak-group std | candidates holding 99% of mass | share of penalty from the rest |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `c0` | train | 2.827 | 1.80 | 3.394 | 21% | 55% |
| `c0` | val | 2.843 | 1.85 | 3.422 | 21% | 55% |
| `k1` | train | 2.884 | 1.76 | 3.207 | 20% | 54% |
| `k1` | val | 2.887 | 1.81 | 3.213 | 21% | 54% |
| `k1_ema` average | train | 2.769 | 1.72 | 3.126 | 19% | 51% |
| `k1_ema` average | val | 2.777 | 1.77 | 3.138 | 20% | 51% |
| `k1_ema` training weights | train | 2.788 | 1.74 | 3.134 | 20% | 52% |
| `k1_ema` training weights | val | 2.789 | 1.78 | 3.141 | 20% | 51% |

- In every weight set the two standard deviations differ between train and val by at most 0.03,
  with overlapping per-split intervals. The mass-weighted variance differs by at most 0.05 and is
  2-3% larger on val. Nothing here shows training spectra solved by pushing dominant scores
  further apart.
- About 228 candidates reach the allocator per spectrum, and about a fifth of them carry 99% of
  the predicted intensity. The rest contribute 51-55% of `mean_f (z_f - mean z)^2`. Much of the
  penalty's gradient would raise logits of candidates the target gives no intensity.

## Many-view follow-up: the MC-dropout mean (2026-09-28)

**Why.** A two-view average is itself noisy. Averaging `n` independent views leaves `1/n` of one
view's noise variance, so to second order the expected cosine of the `n`-view average is
`S_inf - k/n`, with `S_inf` the cosine of the MC-dropout mean. A two-point extrapolation from the
first run's pickles, `S_inf ≈ 2 · (two views) − (one view)`, put the `k1`-family mean +0.003 to
+0.004 above the best flattened eval on val. The gap between a deterministic network and the mean
of its dropout sub-networks is the R-Drop paper's own rationale, so it was measured directly.

**Setup.**

- The probe's new `--head_views 8`: eight head views per spectrum, same head scope as above, with
  the cosine of the average of the first `n` = 1, 2, 4, 8 views. The one-view point is the mean
  over all eight.
- Same 4,096 spectra per split (sample seed 20260927). The dropout seed stride changed, so views 0
  and 1 are new draws. Spectra where any of the eight views drew another tied isomer are dropped:
  1.4-1.8%.
- `i1_inst_ob_obf` (`i50ig7rf`), epoch 59: best val (0.7622 logged; tail over epochs 54-63
  0.7607). It is the new NIST20 `[M+H]+` control on the `OB` / `OB_F` labels
  ([instrument finding §8](../nist_orbitrap_instrument_ce_scales_2026_09_28.md#8-training-with-the-instrument-input-i1-and-i2-running-read-2026-09-28)).
  Its head dropout is allocator 0.1 alone, as `c0`'s. File
  `saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt`, SHA-256 prefix `959efa48`.
- `k1_ema` average, epoch 50, as above. The spectra are now labelled `OB` / `OB_F`, and the
  checkpoint has a one-row (`FT`) instrument table. A copy with that row in both rows gives every
  spectrum the embedding `FT` gave it. Eval cosine matches the first run to 1e-4 on 8,141 of 8,192
  spectra. The other 51 are the builder's tie swaps (side finding 1).

**The curve.** Cosine minus the deterministic eval prediction, paired per spectrum:

| weights | split | 1 view | 2 views | 4 views | 8 views | `S_inf`, fit over `n` | `S_inf`, 2·y(8) − y(4) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `i1` | train | −0.0016 | +0.0003 | +0.0014 | +0.0020 | +0.0024 [+0.0020, +0.0029] | +0.0027 |
| `i1` | val | +0.0021 | +0.0040 | +0.0048 | +0.0051 | +0.0057 [+0.0049, +0.0064] | +0.0055 |
| `k1_ema` average | train | −0.0095 | −0.0023 | +0.0010 | +0.0027 | +0.0045 [+0.0039, +0.0052] | +0.0043 |
| `k1_ema` average | val | −0.0036 | +0.0042 | +0.0069 | +0.0086 | +0.0105 [+0.0091, +0.0119] | +0.0103 |

**Against flattening.** Powers chosen on val (eval `P^0.75`, eight-view `P^0.875`), applied to
both splits:

| weights | split | eval cosine | eval `P^0.875` | eval `P^0.75` | eight-view `P^0.875` | eight-view `P^0.875` − eval `P^0.75` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `i1` | train | 0.9259 | +0.0003 | −0.0046 | +0.0007 [+0.0002, +0.0012] | n/a: flattening costs train |
| `i1` | val | 0.7622 | +0.0041 | +0.0050 | +0.0081 [+0.0071, +0.0090] | **+0.0031 [+0.0024, +0.0039]** |
| `k1_ema` average | train | 0.9193 | +0.0000 | −0.0051 | +0.0001 [−0.0006, +0.0007] | n/a |
| `k1_ema` average | val | 0.7431 | +0.0047 | +0.0064 | +0.0112 [+0.0098, +0.0127] | **+0.0049 [+0.0037, +0.0060]** |

- **The curve is linear in `1/n`.** The fit over all `n` and the last-two estimate agree within
  0.0003. The two-point extrapolation from the first run (`k1_ema`: +0.0046 train, +0.0097 val)
  matches the direct measurement.
- **The dropout mean beats the deterministic network on train too** (+0.0024 `i1`, +0.0045
  `k1_ema`), where flattening gains nothing. That is the weight-scaling mismatch R-Drop is argued
  to close.
- **On val, averaging adds to flattening.** Eight views at `P^0.875` beat the best flattened eval by
  +0.0031 (`i1`) and +0.0049 (`k1_ema`), paired on one checkpoint. The unflattened limit alone is
  only +0.0007 [−0.0004, +0.0018] above eval `P^0.75` on `i1`: the two act together.
- **The mismatch scales with head dropout.** `i1`'s allocator-only noise (two-view TV 0.069 train,
  0.075 val; symmetric KL 0.029 / 0.035; top-peak agreement 91%, all as `c0`) gives about half of
  `k1_ema`'s gap.
- **Limits.** One checkpoint each and 4,096 spectra per split. Both sides' flattening powers were
  chosen on the same val sample. The probe reruns the encoder and candidate builder for every view,
  so it does not price an inference path that reuses them.

## What this means for the proposals

- **R-Drop (note §1).**
  - The magnitude gate passes: two head passes disagree by 0.03-0.12 nats on training spectra.
  - The disagreement it penalizes is not what separates training from validation.
  - *Corrected by the many-view follow-up:* its inference-side counterpart is not just
    flattening. The eight-view mean, flattened, beats flattened eval on val by +0.0031 on `i1`
    and +0.0049 on `k1_ema`. On `i1` that bounds what closing the weight-scaling gap alone can
    give, before any regularizing effect of training. The bound is below the 0.0042 seed spread.
  - The dropout that helps (`k1`) raised the disagreement. EMA lowered it only slightly.
  - If R-Drop is run:
    - `i1` is the control (user decision, 2026-09-28). Its head noise is allocator dropout
      alone, the kind `h1` found harmful to raise.
    - Keep the note's two-pass λ = 0 control.
    - Read validation cosine and ranking after applying the same `P^γ` to arm and control.
      Otherwise a calibration shift passes as a gain.
    - Also read against `i1` with eight-view head averaging at inference, which gains +0.0031
      without training.
    - A +0.003 effect needs two seeds per arm to resolve.
- **Centered-logit penalty (note §2).** Do not build it as specified.
  - Its premise, dominant training scores pushed to extremes, has no support: the spread is equal
    across splits.
  - The unweighted form spends half its weight on the zero-intensity tail.
  - The flattening it could deliver is available at inference: `P^0.875` gains +0.004-0.005 on val
    at no train cost in these runs, and `P^0.75` gained +0.006-0.007 on test in the
    [ranking decomposition](masster_ranking_decomposition_2026_09_23.md#calibration-slightly-too-sharp-and-a-global-temperature-pays).
  - A confidence penalty worth testing would be mass-weighted (`Var_P(z)`) or a peak-level entropy
    term, read against `P^γ`.
- **Late SAM (note §3).** Nothing here measures sharpness. Two constraints apply to the build:
  - `MassterPL` inherits `SpectrumPL`'s automatic-optimization-only `training_step`: manual
    optimization raises `NotImplementedError`. A closure-based SAM step would re-run candidate
    construction and the BatchNorm update on its second pass.
  - Even in eval mode, the candidate builder does not return the same set twice (side finding 1).
    Holding candidates fixed across the two sharpness passes therefore needs an explicit cache,
    not re-generation.
  - EMA's +0.012 over its training weights is four times the dropout average's gain, so a
    flatness lever would be read against EMA, as the note proposes.
- **Distillation from an EMA teacher (note §6).** The note omits the closest measured case.
  - `u3` (Switch EMA: the average fed back into the training weights every few epochs) lost
    −0.0089 against `u2`, with train loss 0.16 higher throughout
    ([ranking-transfer plan](../../plans/masster/masster_ranking_transfer_plan.md#switch-ema-arm-u3-built-2026-09-25-not-submitted)).
  - An EMA-teacher consistency term is a soft form of that feedback. `u3` is a prior against it,
    not proof.
  - The out-of-fold teacher in the note is unaffected.
- **Uncertainty.** Dropout disagreement tracks spectrum difficulty on both splits. It barely
  changes with molecular novelty, unlike seed disagreement. So it is a weak substitute for the
  independent-seed ensembles the
  [reliability plan](../../plans/masster/masster_intensity_reliability_plan.md) relies on.

## Side findings

1. **The CUDA candidate builder is not reproducible between calls, even in eval mode.**
   - Two eval passes on one batch return the same fragments in a different order.
   - In about 1% of spectra per pair of calls, an exact proposal-score tie at the 256-fragment
     budget resolves to another isomer. The pathway counts and m/z values match, but their
     multiplicities differ.
   - Run-to-run nondeterminism was recorded before
     ([regularization audit](masster_regularization_audit_2026_09_19.md)). The practical point is
     new: any two-pass method (R-Drop, SAM, consistency to a teacher) must reuse one candidate set.
     Aligning two passes by candidate row is invalid, so compare by m/z group.
2. **The logged 0.01-Da cosine is edge-sensitive for low-precision targets.**
   - Spectrum 525898's base peak is recorded at 110.99. MASSter predicts it at 110.9899, and the
     logged metric scores that near-perfect prediction 0.0035.
   - Conversely, a float64 `floor(mz / 0.01)` puts a target at exactly 141.01 one bin below the
     logged metric's `searchsorted` edges.
   - 2.6% of sampled spectra changed by more than 0.01 between those two bin conventions. The probe
     replicates the logged binning exactly.
   - Paired comparisons of models that predict the same exact masses are unaffected, since both
     predictions land in the same bins. Absolute cosines and comparisons against other models'
     masses carry this noise.
3. **Flattening is a hedge, not a split-independent calibration.**
   - The [ranking decomposition](masster_ranking_decomposition_2026_09_23.md#train-versus-validation-the-ranking-error-is-a-generalization-error)
     measured `P^0.75` on val and test only, and inferred a split-independent effect from equal
     entropy excess.
   - On training spectra `P^0.75` costs 0.004-0.006 cosine here, and its validation gain rises
     with novelty.
   - The free inference gain stands. The reading changes: it pays where the allocation is wrong.

## Reproduce

Local, Git-ignored outputs under `output/masster_dropout_views_2026_09_27/`. `run_probe.sh` runs
all four weight sets, about 5 minutes each on the RTX 2070 SUPER, into one per-spectrum pickle and
log per set. Then:

```bash
python scripts/debug/masster_memorization/debug_masster_dropout_view_disagreement.py --summarize \
  output/masster_dropout_views_2026_09_27/{c0_vcpho5qm_e042,k1_wg0nxymi_e061,k1_ema_j4u25qfd_e050_average,k1_ema_j4u25qfd_e050_training}.pkl
```

It prints the split means with intervals, the correlations, the NCE and novelty strata, the
variant table and the paired contrasts.

The first run's pickles used the earlier dropout seed stride. A rerun draws new views, so its
head numbers reproduce only within their intervals.

**Many-view follow-up.** `output/masster_dropout_views_mc_2026_09_28/run_probe_v8.sh` builds the
two-row `k1_ema` copy with `patch_inst_embedding.py` (same folder). It then runs `--head_views 8`
on `i1` and on that copy, about 8 minutes each on the RTX 2070 SUPER, and writes
`summary_v8.log`. Its "MC-dropout head curve" table is the curve above.
