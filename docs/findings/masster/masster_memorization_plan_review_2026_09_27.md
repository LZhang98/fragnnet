# Review of MASSter's memorization diagnosis and next experiments

> **Status:** FINDING · **Updated:** 2026-09-27
> Document and source review; no new training, inference, or W&B refresh. Run numbers below
> are the recorded measurements, not a new read of running jobs. Recommendations are hypotheses.

## Main conclusion

The experiments establish a large allocation generalization gap. They do not establish that
erasing molecule identity will recover that gap. Most attempted remedies remove or perturb
information without establishing a better rule for predicting unseen chemistry. The results
fit this explanation: removal usually does not help; modest noise sometimes helps; averaging
helps consistently; a large residual error remains on unfamiliar chemistry.

Some methods work. The [attempts audit](masster_memorization_attempts_audit_2026_09_27.md)
records replicated key-dropout gains (+0.0074 and +0.0048), replicated EMA gains, and merged
LR-decay gains. These are useful performance improvements even when the train/val gap persists.
The [latest local screen](../../history/masster_memorization_route_plan_log.md#local-screen-allocator-size-standard-attention-and-mz-encoding-2026-09-27)
also has promising full-attention and atom-mask results, under a different data/batch/horizon
recipe. They are not confirmed main-recipe successes or failures.

## What the current reasoning misses

1. **Dependence is not a removable cause of generalization error.** The route probes intervene
   on trained activations. Greater damage on train establishes train-specific dependence under
   that intervention. It does not show that training without the route improves unseen-molecule
   predictions. Nor are the route damage scores additive: the encoder, boundary block and set
   attention form interacting stages. The change in `r2`'s set-message sensitivity supports
   redistribution of dependence, but does not uniquely identify an internal lookup algorithm.
   The [audit](masster_memorization_attempts_audit_2026_09_27.md#what-the-attempts-show-together)
   already notes the poor predictive record of the route map.

2. **Molecule identity and useful chemistry have not been separated.** A structure-rich
   representation can distinguish molecules while supporting transfer. A local encoder followed
   by whole-fragment pooling can still distinguish whole molecules. Conversely, removing global
   context can discard genuine differences between otherwise similar cuts. The plan's statements
   that `r1` cannot name a molecule and `nl2` has no whole-molecule representation are guarantees
   its architecture does not provide. The still-unbuilt cross-molecule collision census is a
   necessary check before calling a representation shared or identity-blind.

3. **The strongest evidence points to limited chemical transfer, not solely excess capacity.**
   In the parts analysis (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md`),
   27.7% of fragment-explained validation intensity has no matching radius-2 break environment
   in training; captured intensity is 0.488 there versus 0.699 for environments seen in 10–99
   other molecules. This is a diagnostic association, not an irreducible-error bound. It does
   explain why destroying training fit need not improve validation. The local learner already
   tested is only modestly complementary and contributes almost nothing at high CE
   (finding (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_bond_local_learner_2026_09_24.md`)); “use local chemistry” alone is not a
   sufficient replacement plan.

4. **The acceptance criteria mix usefulness with a mechanistic theory.** The
   [plan](../../plans/masster/masster_memorization_route_plan.md#objective-and-acceptance)
   requires every route's asymmetry to avoid increasing and the largest gain to occur below
   Tanimoto 0.6. Neither is necessary for a useful predictor. Treat replicated held-out quality,
   coverage, cost and important subgroup regressions as promotion criteria. Treat gap closure,
   ranking decomposition and route sensitivity as evidence for a separate anti-memorization
   claim. A smaller gap without better validation is not the objective.

5. **The statistical precision is overstated.** A control pair differing by 0.0042 is a warning
   about seed sensitivity, not a reliable estimate of the entire seed distribution. The audit's
   +0.006 rule is a practical screening margin, not a calibrated two-standard-error guarantee;
   its false-pass probabilities should likewise remain illustrative. Ten adjacent epochs are
   not ten independent replicates. Parameter-free stochastic arms preserve initialization but
   need not preserve later dropout draws or data order unless RNG streams are isolated.
   Existing prefix bias, unmatched checkpoint epochs and missing raw gap outputs compound this.

6. **Peak supervision does not identify individual fragment explanations.** Multiple candidate
   fragments and ion/H channels can explain the same observed peak. The spectral loss constrains
   their aggregate intensity; it does not observe the true split between isomeric explanations.
   The bond-local finding (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_bond_local_learner_2026_09_24.md`) explicitly demonstrates
   attribution sensitivity. Therefore fragment/bond “memorization” conclusions need peak-level
   corroboration. Smoothing uniformly over candidates would also weight peaks by their number
   of explanations; define any such prior at the observed peak/formula level first.

7. **The sink does not isolate shared and specific learning.** In
   [the head](../../../src/fragnnet/model/bond_fragment_heads.py), the sink correction is
   `E[m] · W h`, where `h` is the shared contextualized token and is not detached. Sink loss can
   train the encoder and allocator to serve molecule-specific corrections. There is no
   identifiability constraint forcing shared chemistry into one branch and memorization into
   the other. Sink dropout restores unsunk training exposures, but does not prove separation.
   A sink-on/off eval comparison measures reliance; by itself it cannot distinguish gradient
   starvation from transferable information being absorbed. Raw embedding norm is also only a
   proxy: reciprocal rescaling of `E` and `W` preserves the correction. Measure prediction/loss
   changes with the sink on/off and shared-path gradients before proposing more sink variants.

8. **Some literature analogies change the problem.**
   [Goldfish](https://arxiv.org/abs/2406.10209) masks next-token training positions to interrupt
   verbatim sequence reproduction. Fixed peak masking in a normalized spectrum is a different
   objective. It can leave the retained peaks fully memorized and the excluded allocation
   underconstrained; it does not force chemical transfer. Likewise
   [MemSinks](https://arxiv.org/abs/2507.09937) provides motivation for isolation, not evidence
   that this bilinear spectral sink implements it. The literature note's categorical statements
   about identity removal, 1/K dilution by shard averaging, and guaranteed transfer from masked
   peaks should be treated as hypotheses.

## A smaller next-step sequence

1. **Finish measurement on existing runs before another broad arm batch.** Use `c0`, `k1`, EMA,
   and their available seed counterparts, plus completed new arms. Evaluate the full validation
   split and a fixed molecule-balanced train sample at a common checkpoint epoch, all in eval
   mode. Save per-spectrum outputs and immutable config/checkpoint identities. Report both the
   production spectrum-weighted metric and molecule-balanced summaries, with molecule-clustered
   paired uncertainty and seed effects separately. Complete the existing ranking and novelty
   gates, including rarity × CE strata. A pure same-config seed ensemble avoids f0/f1's
   target-cleaning confound.

2. **Separate performance improvements from mechanism tests.** Confirm the promising local
   atom-mask/full-attention screens on the intended recipe. Read the already-running `sm1` and
   `sd1` before extending that family. Test the already-built `s1` schedule as a bounded next
   training experiment; do not assume its benefit stacks with EMA. These are recommendations,
   not authorization or a record of submission.

3. **Run the collision census before building another identity-removal architecture.** Count
   cross-molecule sharing of complete proposed local signatures, with target-intensity weighting,
   at radii 0/1/2 and by CE. Validate whether sharing holds for actual predicted peak groups,
   not just individual bonds. Stop a tied representation if almost every useful signature is
   unique or coarse signatures merge outcomes that need different intensities.

4. **Use a fixed-candidate allocation experiment to test the resulting hypothesis.** Generate
   candidates with a frozen training-derived proposal and use the same sets for each allocator.
   Compare a positionwise scorer with the set allocator, preserving the other token inputs and
   optimizer recipe. This measures the value of candidate exchange, not identity removal. Only
   if the census supports sharing, test a tied/backoff local scorer under the same protocol.
   Report peak-level ranking and novel-chemistry performance, then confirm any gain end to end.
   The proposed graph mask needs this control: global information already exists in its input
   tokens, so restricting edges alone cannot guarantee removal of molecule identity.

5. **Protect a confirmation split from further tuning.** The
   bond-local stack (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_bond_local_learner_2026_09_24.md`) fits validation weights on
   part of test and test weights on part of validation. That is a useful symmetric diagnostic,
   but not an untouched final-test result. Future calibration and arm selection should use
   training folds/validation only, with a reserved molecule-level confirmation set.

Success is a replicated improvement on unseen molecules at acceptable coverage and cost.
Failure of a route-removal arm should not automatically trigger another removal: first ask
whether it tested a shared predictive rule and whether the data support that rule.

## Review scope and working-tree caveat

Reviewed the memorization literature note, route probe, route map, energy/parts finding,
attempts audit and memorization-route plan; cross-checked ranking-transfer, displacement,
regularization and main MASSter planning records and the relevant model/head/loss source.
No claim is made to have re-audited every external paper or rerun the recorded measurements.

The working tree already contained documentation and isotope-probe changes. During this review,
the head source gained active edits for positionwise (`attention: none`) and mass-window
allocation. Those are not changes made or validated by this review. Their appearance means the
recorded candidate-graph planning state is not a complete account of current implementation work.
