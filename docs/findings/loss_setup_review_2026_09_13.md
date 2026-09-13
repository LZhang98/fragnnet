# FraGNNet and MASShatter loss setup review

> **Status:** FINDING · **Updated:** 2026-09-13 · CPU counterexamples verified; implementation fixes and scientific comparisons remain pending.

## Scope and baseline

Reviewed clean revision `fe417a2f990a7041f80d63306cdc5f3f1512c051`, model/Lightning
loss wiring, shared loss helpers, representative checked-in configurations, and the
[model roadmap](../plans/model_improvement_plan.md) and
[MASShatter plan](../plans/masshatter_improvement_plan.md).
Configurations below establish available setups, not the identity of a running experiment.
No model code, configuration, data, environment, or training state was changed.

The prior [correctness audit](fragnnet_masshatter_audit_2026_09_12.md) is historical:
its three FraGNNet loss/explanation defects are fixed in current source and their
regressions pass. MASShatter now rejects unsupported precursor-pair supervision.
The project state's linked `wandb_screen_2026_09_11.md` is absent in this checkout;
its claimed run rankings were not independently verified here.

## Actual objectives

| Setup | Objective and behavior |
| --- | --- |
| [FraGNNet NIST20 baseline](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml) | Inherits Gaussian cross entropy from the template; explicitly includes out-of-structure (OOS) CE. Primary weight 1; standard auxiliary weights zero. |
| [FraGNNet NIST20 JSS/sqrt](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_jss_hun_sqrt.yml) | Vectorized Hungarian normalized JSD, square-root training targets, plus 0.01 node normalized entropy. OOS flags remain enabled, but JSS does not consume the OOS loss. |
| [FraGNNet formula-support arm](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_formula_support_hungarian.yml) | Gaussian CE plus 0.1 class-balanced formula-support BCE. |
| [MASShatter JSS warmup 0–4](../../config/masshatter/merged/warmup_duration/nist20_merged_independent_jss_warmup_e0_4_dense_subformula.yml) | After warmup: normalized JSD + 0.1 proposal loss + 0.1 class-balanced support BCE. Raw-intensity targets, mass-only Hungarian alignment, Gaussian support responsibilities, support gate enabled, entropy weight and selection thresholds zero. Epochs 0–4 train proposal only; joint training starts at 5. |

`SpectrumPL.preproc_spec` transforms training **targets**, leaves training predictions
in their learned output coordinates, and inverse-transforms predictions for evaluation.
Thus `ints_transform: sqrt` is a training-objective change despite comments describing
it as evaluation-only. Both models return per-spectrum loss vectors; shared sample
weighting performs the batch reduction.

## Verified defects to fix before interpreting new FraGNNet comparisons

### 1. Vectorization changes Hungarian JSS correspondence

[FraGNNetPL](../../src/fragnnet/pl_model/fragnnet_pl.py), lines 653–671, selects
scalar or vectorized JSS using `loss_vectorized`. The scalar implementation in
[loss.py](../../src/fragnnet/model/loss.py), approximately lines 1594–1647,
maximizes JS matching gain. The vectorized implementation calls
`batch_jss_hun_helper` in [spec_utils.py](../../src/fragnnet/utils/spec_utils.py),
whose assignment scores at lines 1098–1108 are instead `p_i + q_j` on feasible pairs.
This sum cannot distinguish different complete pairings over the same peak sets.

Counterexample, all pairs inside FT tolerance (10 ppm with minimum m/z 200):

- Observed masses `[100, 100.0005, 100.001]`, intensities `[0.8, 0.15, 0.05]`.
- Predicted masses `[100.0002, 100.0007, 100.0012]`, same intensities.
- Scalar JSD: **0.000000238**; vectorized JSD: **0.352213264**, using the local SciPy solver.

The vectorized assignment objective is wrong for scalar equivalence; tie outcomes
may depend on solver/backend. Use the same JS gain and invalid-pair handling in both
paths, with ambiguous-match value/gradient regression coverage. These inputs are
synthetic; prevalence in real spectra remains unmeasured.

### 2. Vectorized JSS backward fails through the SciPy assignment fallback

With predicted intensities requiring gradients, `batch_jss_hun_helper` passes its
attached score matrix through `_solve_batch_assignment` to
`scipy_linear_sum_assignment`. At `spec_utils.py:882`, `.cpu().numpy()` raises
`RuntimeError: Can't call numpy() on Tensor that requires grad`.

Reproduced both in the shared function and via FraGNNet's actual PL loss closure.
Detach the assignment-cost input while retaining gradients through selected original
intensities. The local environment lacks `torch_linear_assignment`; this result does
not establish a CUDA training failure. Existing no-gradient value tests pass.

### 3. JSS silently leaves the OOS head without primary supervision

At `fragnnet_pl.py:825–887`, `oos_loss` is consumed only by the cross-entropy branch.
JSS renormalizes represented peak intensities, cancelling their shared `(1 - p_oos)`
factor. Both the NIST20 JSS arm and
[SpectraVerse JSS/sqrt](../../config/spectraverse/fragnnet_jss_sqrt/cv0.yml) nevertheless
set `predict_oos=True` and `oos_loss=True`.

Actual PL closure probe: observed masses `[100,120,160]`, intensities `[0.6,0.2,0.2]`;
candidate masses `[100,120]` and conditional logits `[1,0]`.

| Objective | Loss at OOS probability 0.1 | Loss at 0.9 | OOS-logit gradient |
| --- | ---: | ---: | --- |
| Gaussian CE | -3.54649 | -2.22816 | -0.1 → +0.7 |
| Binned JSD | 0.108330607 | 0.108330607 | Approximately zero |
| Scalar Hungarian JSD | 0.108330846 | 0.108330846 | Approximately zero |

Gaussian density CE can be negative; its magnitude is not comparable to bounded JSD.
Either explicitly supervise OOS under the chosen coverage semantics, or reject the
unsupported combination and use a configuration that disables the unused head.
Adding a CE component is a scientific alternative, not an existing mixed-loss switch.

### 4. Invalid formula slots compete for valid formula-support targets

`_formula_support_targets` at `fragnnet_pl.py:49–89` marks every formula isotope peak
as valid during attribution. The real formula validity mask is applied only later,
inside BCE at lines 908–914. Unrepresented slots can therefore steal a target even
though their own BCE is masked out.

Observed mass 100; valid candidate formulas at 100.001 and 150; add an invalid formula
slot at 100. At zero support logits:

| Attribution | Valid matching formula's target before → after | Its loss gradient before → after |
| --- | --- | --- |
| Hungarian | 1 → 0 | -0.25 → +0.25 |
| Gaussian | 1 → 0.468825 | -0.25 → -0.163273 |

Hungarian training actively suppresses the real matching formula. Filter formula
isotope peaks by formula validity **before attribution**, then retain the BCE mask.
This concerns support-enabled variants; real-data incidence remains unknown.

## MASShatter: improvements supported by the setup review

### JSS gives weak recovery gradients to heavily suppressed correct peaks

Actual `sparse_peak_list_distance` probe: one observed unit peak at 100; predicted
peaks at 100 and 150 with independent intensities `softplus([z, 0])`. At `z=-10`,
the correct peak receives only `0.00006549` of normalized intensity:

| Objective | Loss | Absolute gradient on the correct peak's logit |
| --- | ---: | ---: |
| JSD | 0.999498 | 0.000455 |
| Existing smoothed KL, epsilon 0.0001 | 8.706624 | 0.395684 |
| Cosine distance | 0.999934 | 0.0000655 |

This isolates the spectral term, not the full support-gated model. It establishes
a mechanism for slow recovery, not a measured training failure or KL superiority.
The existing KL is itself capped by target-mixture smoothing and its gradients also
vanish in the extreme suppression limit. Coefficients/gradient scales must be considered.

First compare the existing JSS and KL arms with identical architecture, candidate
selection, matching, intensity transform, support terms, warmup, seed, and evaluation.
Measure weighted gradients on the shared encoder and intensity/support heads before
deciding whether a small KL auxiliary or schedule warrants implementation.

JSD penalizes total unmatched probability, not false-peak count: assigning 99% to the
correct peak and splitting 1% among either one or 1,000 unmatched peaks gives JSD
approximately **0.00501813** in both cases. Monitor unmatched intensity and thresholded
precision/recall together. This invariance alone is not evidence of defective training.

### Matching and support supervision encode different ambiguity assumptions

MASShatter's spectral Hungarian assignment is deliberately mass-only; Gaussian support
responsibilities can label several plausible channels. For observed mass 100 and
predicted masses `[100,100.001]` with intensities `[0.01,0.99]`, all in tolerance:

| Correspondence | JSD |
| --- | ---: |
| MASShatter mass-only Hungarian | 0.959531 |
| MASShatter closest, many-to-one | 0 |
| FraGNNet scalar JS-gain Hungarian | 0.00501817 |

This is an intentional [design choice](../plans/masshatter_nonprobabilistic_intensity_plan.md),
not an accidental MASShatter implementation bug. Diagnose the frequency/intensity of
ambiguous matches, then compare existing `closest` and `hungarian` settings if needed.
Fix correspondence across a JSS/KL comparison. Many-to-one matching also makes attribution
less identifiable, so better training loss alone is insufficient evidence of improvement.

### Auxiliary weights, proposal coverage, and warmup need separate interpretation

- **Support scores are not calibrated occurrence probabilities.** The shared
  [balanced BCE](../../src/fragnnet/model/candidate_support.py), lines 170–201,
  separately normalizes positive and negative mass. Two targets `[0.5,0.5]` have
  optimal sigmoids `[0.5,0.5]`; adding 98 zero-target channels shifts those same two
  optimal sigmoids to `[0.99,0.99]`. This known limitation is recorded in the
  [support plan](../plans/masshatter_ion_h_channel_selection_plan.md). Validate support
  and final-intensity threshold curves before selecting a floor or changing BCE.
- **A 0.1 coefficient does not establish a 10% gradient contribution.** Measure
  weighted per-term gradient norms and their agreement on shared parameters. Also
  report raw loss values, candidate counts, and positive/negative support mass.
- **Proposal loss is not candidate coverage.** `_proposal_coverage_loss` in
  [mass_shatter_pl.py](../../src/fragnnet/pl_model/mass_shatter_pl.py), lines 330–343,
  omits uncovered target peaks. For equal-intensity targets at 100 and 200, candidates
  at both masses with probabilities 0.5 each give loss 0.693; a sole candidate at 100
  gives loss 0 with 50% weighted recall; a sole candidate at 300 gives loss 0 and
  zero gradient with no coverage. Track candidate weighted recall and a support oracle.
- **No-warmup comparisons also change gradient routing.** Model initialization sets
  `proposal_input_gradients_enabled=False`. `_apply_staged_training` at PL lines
  970–979 returns immediately for `None`, but any configured warmup enables proposal
  gradients permanently into a shared encoder. Thus `None` versus epoch 0 changes
  joint-training routing as well as staging. The 0–0/0–2/0–4/0–9 family keeps routing
  matched. Expose or otherwise explicitly match routing before a no-warmup comparison.

## Recommended order and acceptance checks

**Live-evidence follow-up (2026-09-13):** the
[proposal review](masshatter_proposal_review_2026_09_13.md#recent-matched-loss-runs-do-not-establish-kl-superiority)
now verifies an existing matched JSS/KL pair in W&B. Its results are close, with no
established KL advantage. The synthetic gradient mechanism above remains valid, but
another JSS/KL training comparison is not the next default action. Preserve selection-only
and diagnose intensity allocation on fixed support before selecting a new intervention.

1. Correct FraGNNet's vectorized matching/backward contracts, invalid support-target
   competition, and OOS configuration contract. Require value/gradient counterexamples
   above to pass through the real PL path; preserve historical experiment provenance.
2. On a fixed MASShatter batch, inspect candidate weighted recall, ambiguous matching,
   per-term gradient norms, and suppressed-but-supported correct peaks. This selects
   whether the next isolated comparison targets matching, coverage, or intensity recovery.
3. When intensity recovery is the measured issue, run a matched existing JSS/KL comparison.
   Hold support, routing and warmup fixed. Evaluate raw-intensity spectrum quality,
   thresholded precision/recall, weak-peak recall, base-peak ranking, and retrieval under
   the existing roadmap's seed and validation gates. No run was launched here.

Do not revive generic false-peak or log1p-Huber additions without new evidence: the
[earlier plan](../plans/false_peak_loss_plan.md) records null or harmful outcomes under
its historical protocol. Likewise, the [spectral entropy paper](https://www.nature.com/articles/s41592-021-01331-z)
supports a library-search similarity method; it does not establish the optimal training
objective for either model.

## Validation and limits

Configured interpreter: `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`.

- `python -m pytest tests/test_loss.py -q`: **57 passed**.
- `python -m pytest tests/test_mass_shatter_pl.py -q`: **95 passed**.
- `python -m pytest tests/test_fragnnet_formula_support.py tests/test_fragnnet_audit_fixes.py tests/test_fragnnet_audit_regressions.py -q`: **43 passed**.
- New synthetic counterexamples called the actual helpers/PL loss closure. Scratch
  reproducers: `/tmp/loss_setup_probe_20260913.py` and
  `/tmp/fragnnet_loss_setup_review.py`; their essential inputs and outcomes are above.
- No CUDA, full-suite run, real-data gradient audit, trained-checkpoint evaluation,
  training, or fresh W&B query. Passing existing tests does not resolve the new counterexamples.

Only this report and its documentation navigation/handoff were changed. Next action:
implement the verified loss-contract fixes, then collect the bounded MASShatter diagnostics.
