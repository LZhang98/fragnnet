# Explicit peak-ranking loss: no gain in a bounded intensity-head continuation

> **Status:** CLOSED NULL · **Updated:** 2026-10-02 · Completed local CPU pilot. Tuning
> selects zero ranking weight; this does not rule out full-data or end-to-end training.

## Result

The user authorized testing explicit peak preference supervision after the
[covered base-peak ranking diagnosis](masster_i1_base_peak_errors_2026_09_28.md).
In this pilot, the auxiliary changes training ranking but adds no held-out spectral gain.
Do not promote either tested ranking-loss arm on this evidence.

All arms start from `i1` epoch 59 and fine-tune its existing intensity head for four epochs.
The encoder, proposals and dropout stay frozen. The matched control uses ordinary production
spectrum loss; arms add weight 0.1 or 0.3 times a measured-base-versus-strongest-incorrect-mass
preference objective. All channels at a mass are summed, and all base-compatible masses form
the positive group. Near-tied and uncovered targets are skipped only by the auxiliary.
See the [locked protocol](../../plans/masster/masster_peak_ranking_pilot_2026_10_02.md).

There are 765 training spectra / 256 identities, 762 tuning spectra / 256 identities and
763 confirmation spectra / 256 identities. Conditions are sampled before inspecting targets,
up to three per identity. Training and validation identities are disjoint, as are tuning and
confirmation identities. No canonical test targets were read.

### Selection and locked confirmation

Tuning cosine selects **weight zero**: 0.777962 for control, 0.777815 at 0.1 and 0.777366 at 0.3.

| arm | confirmation cosine | base-peak hit | mean compatible base rank |
| --- | ---: | ---: | ---: |
| unchanged checkpoint | 0.797310 | 61.337% | 4.533 |
| ordinary-loss continuation | **0.801505** | 61.206% | **4.450** |
| + ranking weight 0.1 | 0.801284 | 61.468% | 4.458 |
| + ranking weight 0.3 | 0.800747 | 61.599% | 4.478 |

Rank averages use covered base peaks only; lower is better. Cosine uses complete unmatched-peak
norms and production-equivalent one-to-one instrument-tolerance matching. Coverage is **exactly
identical per spectrum** across all four predictions, with confirmation base coverage 98.689%.

The selected ordinary-loss continuation improves cosine over the unchanged checkpoint by
0.004195, molecule-clustered 95% interval [0.002251, 0.006141]. That gain cannot be credited to
ranking supervision. Its base-hit difference is -0.131 percentage points [-1.699, +1.312].

For completeness, the unselected arms' descriptive confirmation contrasts against the matched
control are:

| ranking weight | cosine difference [95% interval] | hit difference, percentage points [95% interval] |
| --- | ---: | ---: |
| 0.1 | -0.000221 [-0.000418, -0.000026] | +0.262 [0.000, +0.658] |
| 0.3 | -0.000758 [-0.001288, -0.000228] | +0.393 [-0.131, +1.046] |

Intervals use 2,000 identity-clustered bootstrap samples. These arms were not selected by the
locked tuning procedure, and no correction was made for multiple descriptive comparisons.
The small base-hit movements do not reproduce on tuning: all three arms have hit rate 57.480%.

### Training and slices

- The training auxiliary is eligible on 676/765 spectra per epoch. Its final recorded
  pre-update batch-average loss falls from 0.12808 in control to 0.12249 / 0.11363 at weights
  0.1 / 0.3. Final training base hits rise from 90.327% to 91.242% / 91.503%.
  The signal is implemented and optimized; its incremental benefit does not transfer here.
- On tuning, both ranking weights lower cosine in every reported NCE band (≤20, 20–40,
  40–60 and >60). On confirmation, both slightly improve the 40–60 band but lower the other
  bands. No consistent energy-specific gain is established.
- The unfamiliar confirmation subset (nearest-train Tanimoto ≤0.4) is only 103 spectra /
  35 identities. Relative to control, cosine changes are +0.000061 / +0.000073 and base-hit
  changes are zero. This small descriptive subset establishes no useful novelty benefit.

## Limits and decision

This is one checkpoint, one sample/seed, a short continuation and an intensity-head-only update.
It does not test full-dataset training from scratch, changes to the shared encoder, alternate
ranking objectives, or explicit nested-path supervision. The validation identities have been
inspected historically, although confirmation is held out from this pilot's strength selection.
No nested-pair annotation or independent worst-spectrum-floor comparison was performed.

The broad ranking-loss hypothesis remains unresolved. The specific tested intervention fails
this pilot's acceptance gate. Preserve the null; a further experiment needs a distinct scope
or mechanism and a full matched control, rather than presenting this auxiliary as a fix.

## Validation and artifacts

All three arms completed with finite losses and clipped gradient norms in 417.9 seconds on CPU.
Strict checkpoint load passed. Six objective/sampling tests and thirteen inherited scorer/probe
tests pass; Ruff and whitespace checks pass. Independent artifact checks verify identity
separation, equal memberships, score uniqueness/bounds, identical coverage and run source/config
hashes. No production model, checkpoint original, dataset original or environment was modified.

Git-ignored artifacts: `output/peak_ranking_pilot_2026_10_02/` contains manifest, resolved model
config and provenance, sampled dataset copies, training history, final head weights, scores,
selection, summary, descriptive arm contrasts, molecule-weighted and energy/instrument/novelty
slices, audit results, audit scripts and run log. Parameters are in the
[analysis YAML](../../../config/analysis/masster_peak_ranking_pilot.yaml); the
[diagnostic runner](../../../scripts/debug/debug_masster_peak_ranking_pilot.py) refuses to overwrite
an existing output directory. Use a new output path in a copied analysis YAML for a fresh run.
