# FraGNNet and MASSter loss setup review

> **Status:** FINDING · **Updated:** 2026-10-06
> CPU counterexamples reverified at `551a64ae`. A second pass audited the working-tree changes themselves and fixed four defects in them, including a vectorized-JSD NaN. A third pass fixed the mirror of the leak term's batch-size defect in three more functions and collapsed two duplicated guards. **`jss_hun*` is not comparable across this change.** OOS and formula-validity fixes remain pending.
>
> **Proposal coverage:** this review is canonical for the blind-spot counterexample; it is
> consolidated with three other reviews, and re-run against live code, in MASSter proposal loss — consolidated audit (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_proposal_loss_audit_2026_09_22.md`).

## Current assessment: tighter objectives (2026-09-21)

Audit started on clean revision `551a64ae`. The historical uncommitted-work descriptions in
project state do not describe this checkout.
The original counterexamples and their inputs below remain the canonical reproductions.

### Correctness before stronger supervision

All four FraGNNet defects below are still present and reproduced with the configured interpreter:

| Priority / affected path | Reproduction | Current state |
| --- | --- | --- |
| High: vectorized Hungarian JSD correspondence | Before the fix, scalar loss 0.000000238 versus vectorized 0.352213264 on the same ambiguous three-peak pair. | **Fixed in working tree:** scalar 0.000000238 versus vectorized 0 after clamping, within 1e-6. Batched and single helpers now maximize JS gain and use `p+q` only for union mass. |
| High: vectorized Hungarian JSD CPU backward | Before the fix, attached predicted intensities raised the SciPy `.numpy()` autograd error. | **Fixed in working tree:** assignment consumes detached gain scores; selected original values retain finite, nonzero gradients. |
| Medium: JSD with OOS prediction enabled | Actual FraGNNet PL closure gives the same 0.1083306 loss at OOS probabilities 0.1 and 0.9; OOS-logit gradients are below 1e-8. | Explicitly supervise coverage under agreed semantics, or reject/disable this unsupported configuration. |
| Medium: formula-support target attribution | Adding an invalid exact-mass slot changes the valid Hungarian target from 1 to 0 and its logit gradient from -0.25 to +0.25. | Apply formula validity before target attribution, not only inside BCE. |

These affect the named optional paths, not every FraGNNet configuration. The representative
FraGNNet MA/MI baseline inherits Gaussian CE; the representative MASSter
`ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64.yml` explicitly uses Gaussian CE
plus 0.1 proposal supervision. Neither is a JSD baseline. FraGNNet's model exports formula
isotope masses before filtering them by support validity, so the target helper's missing
validity input is not repaired upstream. Real-data incidence remains unmeasured.

### What is loose in MASSter

Follow-up implementation: `loss_type: conditional_cross_entropy` now selects the in-support CE
divided by represented target mass. The historical `cross_entropy` remains unchanged. Both paths
log `coverage_weighted_spec_ce` and `conditional_spec_ce`; `spec_ce` remains a compatibility alias
for the selected primary variant. A spectrum with zero represented target mass contributes zero
conditional CE because it contains no supported intensity ratio to fit. OOS CE remains separate:
MASSter has no learned OOS probability, so adding its currently computed zero term would have
no effect. This is an experiment path, not a promoted default.

MASSter direct losses and FraGNNet vectorized Hungarian JSD now share
`align_sparse_peak_support`. Its explicit policies are `closest`, `mass_hungarian`, and
`js_gain_hungarian`; the caller owns the downstream divergence and OOS semantics. This removes
duplicated padded correspondence code without conflating MASSter's m/z-based matching with
FraGNNet's JSD-gain matching. The older scalar FraGNNet JSD entry point remains for compatibility.

The direct peak-list objectives in `model/loss.py:sparse_peak_list_distance` retain the
previously documented behavior. Fresh CPU measurements with Softplus logits `[-10, 0]`,
one correct and one false predicted peak, give:

| Objective | Loss | Absolute correct-logit gradient |
| --- | ---: | ---: |
| Normalized JSD | 0.999498 | 0.000455 |
| Smoothed forward KL, epsilon 1e-4 | 8.706624 | 0.395684 |
| Cosine distance | 0.999934 | 0.0000655 |

This is a recovery-gradient weakness of these parameterized direct objectives, not evidence
that the current Gaussian-CE baseline needs a KL replacement. KL also saturates: at logit -20
its loss is 9.21031 and its gradient is only 0.0000297. Analytically, the implemented
`KL(q || (1-epsilon)p + epsilon q)` is bounded above by `-log(epsilon)`; it is not an
uncapped log penalty for missing peaks. A future recovery term should operate on stable
log intensities of supported mass groups and explicitly separate unreachable targets.

Matching semantics can dominate the choice of divergence. For an observed peak at 100 and
predictions at 100 and 100.001 with intensities 0.01 and 0.99, MASSter mass-only Hungarian
JSD is 0.959531, closest matching gives 0, and FraGNNet scalar JS-gain Hungarian gives
0.005018. These encode different correspondence assumptions. Do not treat their losses as
comparable measures or automatically replace one with another; first measure ambiguous
match prevalence under the relevant instrument tolerances.

The proposal objective still zeros contributions from uncovered peaks. It trains probabilities
over available explanations and cannot directly recover a fragment absent from its candidate
pool. Its value must be accompanied by candidate weighted recall and an oracle on the same
support. Likewise, balanced support BCE produces cost-sensitive scores, not calibrated
occurrence probabilities; increasing its weight does not fix its attribution ambiguity.

### Recommended tightening

1. Repair the four FraGNNet contracts before interpreting new JSD/support comparisons.
2. Keep mass tolerance tied to acquisition accuracy. Narrowing it can turn correct chemical
   predictions into unmatched peaks without improving intensity ordering. Keep target
   preprocessing fixed: `ints_transform: sqrt` changes training targets and inverse-transforms
   evaluation predictions; it is not just a reporting option.
3. Preserve the primary full-spectrum objective and test targeted soft ranking supervision
   where supported dominant peaks are misordered. The existing
   FT contender-pair plan (removed 2026-10-18; `git show 95341fce:docs/plans/fragnnet/ft_contender_pair_minimal_training_plan.md`) and shipped
   `ce_pair_loss_weight` path provide the narrow starting point for unmerged CE-resolved
   spectra. They do not provide a merged-spectrum solution. For merged spectra, the
   global mass-ranking proposal (removed 2026-10-18; `git show 95341fce:docs/plans/fragnnet/global_fragment_intensity_ranking_experiment_plan.md`)
   is an untested alternative, requiring its coverage/ambiguity gates first.
4. Use observed soft ratios between disjoint, supported mass groups: with predicted summed
   intensities `p_a, p_b`, target `t = q_a/(q_a+q_b)`, and
   `d = log(p_a)-log(p_b)`, the pair term is
   `t*softplus(-d) + (1-t)*softplus(d)`. Its derivative is `sigmoid(d)-t`, so an incorrectly
   suppressed supported winner gets strong correction in log-ratio space. Stable log-sum-exp
   aggregation avoids an arbitrary intensity floor. This derivative is not a guarantee of
   large encoder gradients through support gates or hard candidate selection. Whole-spectrum
   CE already supervises ratios; the auxiliary deliberately reweights hard competitions rather
   than adding observational information. Avoid unique-fragment labels for mass-aliased peaks.
5. Measure weighted per-term gradient norms/agreement, supported-but-suppressed true intensity,
   and candidate oracle quality on a fixed batch before choosing a coefficient. Compare at
   matched epochs and seeds; evaluate dominant-peak accuracy, full-spectrum similarity,
   weak-peak recall, false intensity, and retrieval. Use the existing roadmap's validation
   gates, with an identical sampler at zero pair weight for CE-context comparisons.

The campaign read-out (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_campaign_readout_2026_09_21.md`) reports loss-family
differences within replicate variation. That is historical recorded evidence, not a fresh
W&B query in this audit. It argues against another broad CE/JSD/KL sweep without a measured
mechanism. The older sibling-ranking plan is a distinct node-marginal proposal, not an
already implemented shared loss for both architectures. Generic false-peak penalties and
global sharpening remain unsupported by the linked negative findings.

### Second-pass audit of the working tree (2026-09-21)

Re-audited the uncommitted loss changes themselves rather than the pre-existing defects.
Four issues were found and fixed; three claims were checked and cleared.

**`jss_hun` is a reported metric, not only a FraGNNet loss helper.** This is the
compatibility item to carry forward. `batch_jss_hun_helper` also produces the logged
`jss_hun`, `jss_hun_sqrt`, `jss_hun_np`, and `jss_hun_inst_tol` auxiliary scores for both
models (`pl_model/spectrum_pl.py:878` and `:973`), and `jss_hun_helper` produces the oracle
scores in `scripts/eval/dag_oracle_scores.py:281`. Switching those helpers from additive-mass
matching to JS-gain matching therefore moves an evaluation metric. Measured on 300 random
6x6 padded spectra: **40% of values change, mean +0.0165, maximum +0.2653, and no value gets
worse.** The direction is expected, because the additive score `P_u + Q_v` is rank-one, so
every complete pairing of the same peaks ties and the choice fell to an arbitrary tie-break;
JS gain resolves it toward the pairing that actually minimizes JSD. **Do not compare
`jss_hun*` across this change.** `cos_hun` uses `batch_cos_hun_helper` and is unaffected.

| Fixed | Evidence |
| --- | --- |
| Vectorized Hungarian JSD returned NaN for a spectrum with observed but no predicted peaks. The emptiness guard is per chunk, so such a spectrum reached the normalizer with zero predicted mass. Before: `[0.0, nan]`; the scalar gives `[1.8e-07, 1.0]`. At `551a64ae` the same input returned `[0.0, 0.0]`, scoring an empty prediction as perfect. | Now clamps the normalizer and selects 1.0 per spectrum, matching the scalar. A `th.where` alone is not enough: a NaN branch still poisons the backward pass. |
| The infeasible-pair sentinel `-1e9` overflows float16, and SciPy cannot convert bfloat16, so gain matching raised under mixed precision. The replaced code upcast through `sub_mask.float()`. | `_solve_batch_assignment` now solves in float32 for non-float32/64 inputs, and the three gain sites promote before masking. `batch_jss_hun_helper` and `jss_hun_helper` also had a pre-existing half-precision `index_put` dtype error at `551a64ae`; both now score in float32 and agree with the float32 result to 1e-3. |
| The new leak term inferred `batch_size` from observed peaks, so a trailing spectrum without observed peaks shortened the result. Measured: two spectra returned shape `(1,)`, which then broadcasts silently onto a per-spectrum loss rather than raising. It also raised for a spectrum with no predicted peaks, which the primary loss tolerates. | `batch_size` is now a required argument supplied by both call sites, degenerate spectra are defined (no predicted intensity gives 0, no observed peaks gives 1, both with zero gradient), and out-of-range batch indices raise. |
| The same observed-only `batch_size` inference the leak term was fixed for survived in `sparse_hellinger_distance` and in both Hungarian JSD paths, in the mirror direction: a trailing spectrum holding predictions but no observed peaks. Measured on `true_batch_idxs=[0]`, `pred_batch_idxs=[0, 1]`: Hellinger raised a bare `AssertionError` from `scatter_logsumexp`, and the vectorized JSD returned shape `(1,)` for two spectra, silently dropping one from the per-spectrum loss. | All three now size the batch from both peak lists through `_spectra_in_batch`, which raises when both lists are empty. Such a spectrum shares no support, so it scores 1.0; Hellinger returns `[0.0, 1.0]` and both JSD paths return `[~0, 1.0]`. Scalar and vectorized JSD stay bit-identical (`max |diff| = 0`) over 40 random well-formed batches. `sparse_cosine_distance_hungarian` keeps the old inference: it predates this change and nothing here touches it. |
| The `spectral_leak_loss_weight`/`spectral_ce_auxiliary_weight` validation was byte-identical in `fragnnet_pl.py` and `masster_pl.py`, and the half/bfloat16 upcast guard was repeated at three sites in `spec_utils.py`. Either could be changed in one place only. | Collapsed into `validate_hellinger_auxiliary_weights` in `loss.py` and `_as_scoring_dtype` in `spec_utils.py`. The existing parametrized rejection tests for both models pass against the shared validator unchanged. |
| The MASSter CE auxiliary divided by `pred_totals[pred_batch_idxs]` and clamped the quotient, which cannot repair a NaN. Reachable only through a spectrum whose predicted intensities sum to zero, which `sparse_peak_list_distance` rejects first on the same branch, so the guard rested on a non-local invariant. | Clamps the divisor instead of the quotient, at the division site. |
| The leak term ran an unbatched per-spectrum Python loop with three to four `.item()` synchronizations per spectrum, about 1.5-2k device syncs per step at batch 512. The same pattern is what `masster_loss_audit_2026_09_19.md` (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_loss_audit_2026_09_19.md`) attributes the CE/JSS gap to. | Rewritten as the chunked padded form already used by the vectorized JSD, keyed on `loss_batch_size`; no per-spectrum synchronization remains. Chunk invariance is tested. |

Checked and cleared:

- **`align_sparse_peak_support` is bit-identical to the code it replaced.** Over 180 random
  spectra, all three objectives and both matching policies give `max |new - old| = 0`.
- **The vectorized JSD agrees with the scalar.** Maximum absolute difference 6.3e-07 over 40
  random multi-spectrum batches, and the implemented
  `0.5*sum(t log t) + 0.5*sum(p log p) - sum(m log m)` over log 2 is algebraically the
  scalar's `1 - sum(gain)/log 2`, because unmatched coordinates contribute zero gain.
- **`sparse_hellinger_distance` bin pairing is sound.** It matches a dense reference to 1.8e-07.
  `batched_bin_func` offsets bin ids by `batch_idx * num_bins` and `th.unique` returns them
  sorted, so the two `th.isin` selections are the same bin set in the same order. The runtime
  assertion now compares bin ids rather than only their spectra.
- **`conditional_in_support_cross_entropy` recovers represented mass stably.**
  `-expm1(oos_logprob)` is well conditioned where it matters and degrades to zero only below
  about 1e-8 represented mass, which is the correct value there. Sequential and vectorized CE
  paths agree to 1e-6, and MASSter forces `oos_loss` off, so no coverage-weighted OOS term
  mixes with the conditional in-support term.

**Both `hellinger_distance` paths are batched, and are now covered as such.** Neither was a
per-spectrum loop: MASSter's objective reduces inside the `loss_batch_size` chunk loop of
`sparse_peak_list_distance`, and FraGNNet's `sparse_hellinger_distance` has no Python loop at
all (scatter-based, 10 ms for a 1024-spectrum batch of 61k predicted peaks on CPU). The gap was
in the tests, not the code: `hellinger_distance` had been left out of the four objective
parameterizations in `tests/test_masster_peak_list_distance.py`, including the
`TestPeakListDistanceMatchesLoopReference` harness, so the chunked Hellinger reduction had never
been compared against the per-spectrum reference. The reference now carries its own Hellinger
branch, written as a plain `sqrt(p*t)` without the production zero-safe clamping so the two
cross-check rather than restate each other, and the objective is parameterized everywhere the
others are. Added on top: chunk invariance of Hellinger values, entropy, and gradients over
`loss_batch_size` in `{1, 2, 5, 64}` under both matching rules; batch-composition independence
for both paths; a closed-form three-spectrum case where every spectrum occupies m/z 100 so
shared-bin pairing cannot silently leak; and per-spectrum gradient isolation. The
`_random_binned_spectra` fixture draws from a small shared m/z grid, because its first version
produced no cross-spectrum bin collisions at all and therefore never exercised the property its
test claimed. Three mutations confirm the coverage bites: normalizing across the batch instead of
per spectrum (8 failures), pairing shared bins by position instead of global bin id (6 failures),
and reducing the Bhattacharyya coefficient over the chunk instead of per spectrum (29 failures).

Naming: `1 - BC` is the squared Hellinger distance `H^2`, not the literature's
`H = sqrt(1 - BC)`. The squared form is the right training choice because its gradient stays
bounded as the spectra agree, but `loss_type: hellinger_distance` values are not comparable
with published Hellinger numbers. Recorded in the docstrings and the config comment.

Not changed: `sparse_jensen_shannon_divergence_hungarian_vec` still derives its own batch size
from observed peaks, so a trailing spectrum with predictions but no observed peaks is dropped.
That convention is shared with the scalar entry point and `jss_helper`, so changing it is a
separate decision about the whole FraGNNet loss family.

### Follow-up validation and limits

- Re-ran `/tmp/loss_setup_probe_20260913.py` and `/tmp/fragnnet_loss_setup_review.py` after
  inspecting their contents: all four FraGNNet counterexamples and the MASSter gradient,
  matching, and clutter examples reproduced. These scratch scripts are not tracked artifacts;
  essential inputs and outputs are recorded here and in the original sections below.
- `python -m pytest tests/test_loss.py tests/test_masster_peak_list_distance.py
  tests/test_fragnnet_formula_support.py tests/test_fragnnet_audit_regressions.py -q`:
  **260 passed, 6 skipped** using `FRAGNNET-GPU`. Passing existing tests does not cover or
  negate the reproduced defects.
- After the conditional-CE and vectorized-JSD fixes, the focused loss, spectrum-helper,
  MASSter PL, formula-support, and FraGNNet audit suites report **251 passed**. Ruff and
  `git diff --check` pass. CUDA assignment was not available.
- After extracting shared sparse alignment, loss, spectrum-helper, MASSter peak-list, and
  MASSter PL suites report **407 passed, 6 skipped**. Ruff passes; CUDA assignment was not
  available.
- CUDA was unavailable to these processes. No CUDA assignment validation, full suite,
  checkpoint/batch gradient survey, training, or validation-set model comparison was run.
- MASSter's separately named conditional-CE path, FraGNNet's vectorized JSD matching/backward
  corrections, and focused regressions are in the working tree; no preprocessing or default-config
  behavior changed. Next concrete action: fix FraGNNet validity masking and settle its OOS contract
  before implementing dependent behavior. Evaluate scientific loss changes with matched controls
  before promotion.

## Scope and baseline

Reviewed clean revision `fe417a2f990a7041f80d63306cdc5f3f1512c051`, model/Lightning
loss wiring, shared loss helpers, representative checked-in configurations, and the
[historical model roadmap](../README.md#removed-historical-plans) and
MASSter plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_improvement_plan.md`).
Configurations below establish available setups, not the identity of a running experiment.
No model code, configuration, data, environment, or training state was changed.

The prior correctness audit (removed 2026-10-13; `git show e65339ed:docs/findings/fragnnet_masster_audit_2026_09_12.md`) is historical:
its three FraGNNet loss/explanation defects are fixed in current source and their
regressions pass. MASSter now rejects unsupported precursor-pair supervision.
The project state's linked `wandb_screen_2026_09_11.md` is absent in this checkout;
its claimed run rankings were not independently verified here.

## Actual objectives

| Setup | Objective and behavior |
| --- | --- |
| [FraGNNet NIST20 baseline](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml) | Inherits Gaussian cross entropy from the template; explicitly includes out-of-structure (OOS) CE. Primary weight 1; standard auxiliary weights zero. |
| [FraGNNet NIST20 JSS/sqrt](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_jss_hun_sqrt.yml) | Vectorized Hungarian normalized JSD, square-root training targets, plus 0.01 node normalized entropy. OOS flags remain enabled, but JSS does not consume the OOS loss. |
| [FraGNNet formula-support arm](../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_formula_support_hungarian.yml) | Gaussian CE plus 0.1 class-balanced formula-support BCE. |
| [MASSter JSS warmup 0–4](../../config/masster/merged/warmup_duration/nist20_merged_independent_jss_warmup_e0_4_dense_subformula.yml) | After warmup: normalized JSD + 0.1 proposal loss + 0.1 class-balanced support BCE. Raw-intensity targets, mass-only Hungarian alignment, Gaussian support responsibilities, support gate enabled, entropy weight and selection thresholds zero. Epochs 0–4 train proposal only; joint training starts at 5. |

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

## MASSter: improvements supported by the setup review

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

MASSter's spectral Hungarian assignment is deliberately mass-only; Gaussian support
responsibilities can label several plausible channels. For observed mass 100 and
predicted masses `[100,100.001]` with intensities `[0.01,0.99]`, all in tolerance:

| Correspondence | JSD |
| --- | ---: |
| MASSter mass-only Hungarian | 0.959531 |
| MASSter closest, many-to-one | 0 |
| FraGNNet scalar JS-gain Hungarian | 0.00501817 |

This is an intentional [design choice](../plans/masster/masster_nonprobabilistic_intensity_plan.md),
not an accidental MASSter implementation bug. Diagnose the frequency/intensity of
ambiguous matches, then compare existing `closest` and `hungarian` settings if needed.
Fix correspondence across a JSS/KL comparison. Many-to-one matching also makes attribution
less identifiable, so better training loss alone is insufficient evidence of improvement.

### Auxiliary weights, proposal coverage, and warmup need separate interpretation

- **Support scores are not calibrated occurrence probabilities.** The shared
  [balanced BCE](../../src/fragnnet/model/candidate_support.py), lines 170–201,
  separately normalizes positive and negative mass. Two targets `[0.5,0.5]` have
  optimal sigmoids `[0.5,0.5]`; adding 98 zero-target channels shifts those same two
  optimal sigmoids to `[0.99,0.99]`. This known limitation is recorded in the
  support plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_ion_h_channel_selection_plan.md`). Validate support
  and final-intensity threshold curves before selecting a floor or changing BCE.
- **A 0.1 coefficient does not establish a 10% gradient contribution.** Measure
  weighted per-term gradient norms and their agreement on shared parameters. Also
  report raw loss values, candidate counts, and positive/negative support mass.
- **Proposal loss is not candidate coverage.** `_proposal_coverage_loss` in
  [masster_pl.py](../../src/fragnnet/pl_model/masster_pl.py), lines 330–343,
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
proposal review (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_proposal_review_2026_09_13.md`)
now verifies an existing matched JSS/KL pair in W&B. Its results are close, with no
established KL advantage. The synthetic gradient mechanism above remains valid, but
another JSS/KL training comparison is not the next default action. Preserve selection-only
and diagnose intensity allocation on fixed support before selecting a new intervention.

1. Correct FraGNNet's vectorized matching/backward contracts, invalid support-target
   competition, and OOS configuration contract. Require value/gradient counterexamples
   above to pass through the real PL path; preserve historical experiment provenance.
2. On a fixed MASSter batch, inspect candidate weighted recall, ambiguous matching,
   per-term gradient norms, and suppressed-but-supported correct peaks. This selects
   whether the next isolated comparison targets matching, coverage, or intensity recovery.
3. When intensity recovery is the measured issue, run a matched existing JSS/KL comparison.
   Hold support, routing and warmup fixed. Evaluate raw-intensity spectrum quality,
   thresholded precision/recall, weak-peak recall, base-peak ranking, and retrieval under
   the existing roadmap's seed and validation gates. No run was launched here.

Do not revive generic false-peak or log1p-Huber additions without new evidence: the
[earlier plan](../plans/fragnnet/false_peak_loss_plan.md) records null or harmful outcomes under
its historical protocol. Likewise, the [spectral entropy paper](https://www.nature.com/articles/s41592-021-01331-z)
supports a library-search similarity method; it does not establish the optimal training
objective for either model.

## Validation and limits

Configured interpreter: `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`.

- `python -m pytest tests/test_loss.py -q`: **57 passed**.
- `python -m pytest tests/test_masster_pl.py -q`: **95 passed**.
- `python -m pytest tests/test_fragnnet_formula_support.py tests/test_fragnnet_audit_fixes.py tests/test_fragnnet_audit_regressions.py -q`: **43 passed**.
- New synthetic counterexamples called the actual helpers/PL loss closure. Scratch
  reproducers: `/tmp/loss_setup_probe_20260913.py` and
  `/tmp/fragnnet_loss_setup_review.py`; their essential inputs and outcomes are above.
- No CUDA, full-suite run, real-data gradient audit, trained-checkpoint evaluation,
  training, or fresh W&B query. Passing existing tests does not resolve the new counterexamples.

Only this report and its documentation navigation/handoff were changed. Next action:
implement the verified loss-contract fixes, then collect the bounded MASSter diagnostics.

## `spec_ce` is a Gaussian density: negative by design (from local memory, verified in `loss.py` 2026-10-06)

- `loss_type: cross_entropy` with `output_distribution: gaussian` is a mixture-of-Gaussians density
  over m/z (`mog_ce_fn` in `src/fragnnet/model/loss.py`). Each predicted peak has
  sd = clamp(pred_mz, min=`loss_tolerance_min_mz`) x `loss_tolerance_rel` (10 ppm floored at m/z
  200, so sd >= 0.002 Da). The log-density is positive (about 5), so `spec_ce` is negative and
  falls as the model sharpens. It is not a sign bug.
- `oos_tolerance_multiple = -1` means no truncation (every true peak counts as in-support);
  `oos_loss=False` makes `primary_loss` equal `ios_ce` alone; `jss_distance` is on another scale.
- So run families optimize different objectives: `s3gy1ro4`, `ef7fl5c6`, `8v0i0fq7` use
  `oos_loss` False / -1 (the last with `jss_distance`); `qnjttwod`, `d1iv8151`, `jnokddr9`,
  `s3iimtqi`, `a6s5it2m` use True / 1. **Never compare losses across runs; compare
  `val_cos_sim_0.01_epoch/mean`.**
- `loss = primary_loss_weight * primary_loss` (+ 0.1 x proposal loss for MASSter). Check on
  `s3gy1ro4`: -3.0830 + 0.1 x 2.0299 = -2.8800.
