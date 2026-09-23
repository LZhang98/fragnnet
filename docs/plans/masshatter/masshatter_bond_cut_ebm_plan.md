# MASShatter bond-cut energy model

> **Status:** PROPOSED · **Updated:** 2026-09-13
> First stage of the requested bond-cut plus spectrum EBM program; design only, no implementation or runs.

## Objective and sequence

Test whether learned interactions between cut decisions improve selection of chemically useful
fragment support at a fixed budget. The user selected both EBM directions, with bond-cut modeling
first. Follow with the MASShatter spectrum-reranking stage in the
[EBM program and setup delta](masshatter_energy_based_model_plan.md). Keep the two interventions
separate until each has been evaluated, then compare their combination.

This document owns the bond-cut experiment. The
[shared spectrum-reranking design](../energy_based_spectrum_reranking_plan.md) owns the later
intensity experiment; its intensity-perturbation oracle is not the bond-cut gate below.
The [proposal review](../../findings/masshatter/masshatter_proposal_review_2026_09_13.md) remains the canonical
source for coverage, redundancy, budget, and mixture findings. Its recorded E4096 gains are
modest and setup-dependent; underfilling U256 alone is not evidence that an EBM will help.

## Coverage evidence and revised justification

The [verified subformula-filtered D4 comparison](../../findings/subformula_filter_ft_qtof.md#saved-subformula-filtered-d3d4-comparison-verified-2026-09-13)
places the recorded MASShatter coverage in a stronger context. This is an unmatched reference,
not a measurement of recoverable EBM headroom. Do not use all uncovered intensity, nominal
fragment-budget underfilling, or the raw difference from D4 to justify a new cut model.

Keep the requested bond-cut-first order as a bounded diagnostic. First distinguish useful masks
that are under-ranked from missing chemistry/search support and intensity-assignment errors.
If current-support oracle cosine is already high relative to learned cosine, retain the current
selector and advance to the spectrum stage. Alternatively, test a separately declared efficiency
objective: less mask processing at non-inferior coverage and downstream quality. Fix those
non-inferiority margins and the required runtime saving before measuring; it is not an escape
criterion chosen after a failed coverage experiment.

## Established constraints

- A cut mask represents a net set of absent precursor bonds, not a known fragmentation event
  sequence. Observed masses do not uniquely label cuts or mechanisms.
- The [mask model and enumeration proof](masshatter_plan.md#2-probability-of-a-complete-cut-mask)
  use additive independent-bond flip costs for exact heap enumeration. Pairwise energies break
  that guarantee. Existing Bernoulli mixtures already express some cut dependence.
- The [candidate builder](../../../src/fragnnet/frag/mass_shatter_candidates.py) selects mask
  identities using detached scores and subsequently computes live Bernoulli/mixture mask
  probabilities. Arbitrary energy scores cannot silently replace those probability tensors.
- Unique-fragment stopping, repeated component pathways, charge retention, ion/H validity, and
  SMARTS mechanisms all affect reachable masses. Freeze these choices across arms and record
  the active source/configuration; a different support set must not be attributed solely to EBM.

## First model: conditional pairwise residual energy

Let `x` contain the molecule and acquisition conditions and `c_b` indicate whether chemical bond
`b` is cut. Use frozen baseline bond/context features for the initial experiment:

```text
E_base(c | x) = -log P_base(c | x)
E_unary(c | x) = E_base(c | x) + sum_b a_theta(h_b, x) * c_b
E_pair(c | x) = E_unary(c | x)
               + sum_{(b,d) in A} J_theta(h_b, h_d, x) * c_b * c_d
```

`P_base` is the actual checkpoint's Bernoulli or mixture distribution. `A` initially contains
unordered pairs of chemical bonds sharing an atom, each counted once. Make `J` symmetric using
symmetric inputs such as feature sums and absolute differences. Both signs are allowed: pair
terms can favor or discourage joint cuts. These terms model local interactions only; failure
would not rule out nonlocal or higher-order cut models. Condition consistently on the baseline's
CE, adduct, instrument, and fragmentation-mode inputs. Energy is a learned compatibility score,
not a physical barrier or calibrated mask probability.

Zero-initialize residual outputs for baseline ranking at initialization. Fit small unary and
pair heads while freezing the molecular/proposal encoder, baseline logits, and spectral head.
Keep the unary head identical in the unary-only and pair arms; also report a parameter-matched
wider unary control so added capacity is not credited automatically to interactions.

## 1. Build an auditable fixed mask pool

Select one recoverable checkpoint and molecule-disjoint train/validation/test split, preserving
preprocessing and merged/unmerged target identity. Start from its current E/U budgets rather
than assuming older unmerged comparisons transfer to filtered merged data.

Generate a deterministic, deduplicated pool `C_pool(x)` from the baseline enumerator, with an
explicit maximum mask count. Include the current baseline's selected masks. Materialize this pool
before stopping at U unique fragments; merely increasing E with early U stopping may expose no
new masks. Retain the existing chemistry and invalid-mask handling. Record true mask count,
cut-count distribution, unique fragments, reachable masses, stopping reasons, and processing cost.

If baseline-ranked expansion has inadequate coverage, investigate a separate, bounded pool of
one/two-bond flips with fixed seeds. Match its size, cut-count strata, and chemical validity to
random perturbations. Keep generation identical at train and inference and independent of spectra.
Treat exploration as a separate arm, not an implicit benefit of the energy head.

Rerank only this finite pool. Call the result pool top-K, never globally exact EBM top-K. Use a
shared deterministic conversion from ordered masks to at most U unique fragments, with explicit
rules for a mask yielding multiple components at the boundary. Apply the same mask-processing
ceiling, deduplication, fragment order/tie breaking, and charge/channel construction in all arms.
Record realized fragment counts; mass redundancy is not equivalent to structural redundancy.

## 2. Headroom gate before training

Compare current selection, baseline-score selection from the expanded pool, and a target-informed
coverage selector from that same pool at the same U and processing ceiling. Define reachable
support using chemically valid channels before learned intensity pruning. Count each observed
peak once under the declared mass tolerance, regardless of how many masks/channels explain it.
Report intensity-weighted recall and support-oracle cosine separately; weighted recall is not
cosine and target-informed mask selection is not deployable.

Use a greedy marginal-coverage selector if exact budgeted set selection is infeasible. Label its
result an achievable target-informed reference, not an exact maximum. Include the baseline as a
fallback when reporting its attainable score. Compare with equally budgeted random mask-order
selectors over the same pool, matched by cut-count strata and repeated with predeclared seeds.
Report both raw and excess diagnostic gains over random selection. A positive target-informed
maximum alone does not demonstrate usable cut structure or learnability.

For the coverage objective, before training require practical headroom beyond the matched random
reference and the expanded pool's baseline ranking, with paired molecule-bootstrap uncertainty and consistent seed direction.
Predeclare useful recall/oracle gains and runtime limits. If widening baseline enumeration already
captures the benefit, prefer that control. If the union has weak support, inspect chemistry and
search first. Do not train a new head merely because nominal U is underfilled.

## 3. Weak supervision from coverage, not invented cut labels

For training only, let `S(c)` be the reachable mass support of a mask after the frozen chemistry
mapping. Define its utility relative to a target-independent reference set of other masks `R`:

```text
Q(c ; R, observed) = weighted_recall(S(R) union S(c), observed)
                    - weighted_recall(S(R), observed)
```

Draw `R` from the fixed proposal pool using predeclared baseline/random contexts and exclude the
compared masks. Use the same `R` for each comparison. This marginal utility discourages repeatedly
rewarding explanations of an already covered mass. Match comparison pairs on cut count and,
where feasible, component count; report unmatched strata rather than silently excluding them.
Deduplicate equivalent supports for label accounting while preserving structural alternatives.
No match means no incremental utility under this target; it is not proof a cut is chemically false.

For `Q(c_good) > Q(c_bad)` beyond a fixed tie tolerance, train:

```text
L_pair = w * softplus(E(c_good | x) - E(c_bad | x))
```

Use bounded gap weights, cap comparisons per spectrum, and average within spectrum then across
spectra. Observed masses, intensities, matches, and `Q` never enter the energy inputs or inference
selector. Freeze labels/pools across model arms. Do not train on validation/test labels.

The initial scorer does not consume `R`; this objective averages potentially conflicting
marginal-utility preferences across contexts. Evaluate selected-set coverage, not just pair
accuracy. If redundancy persists, a set-conditioned selector is a separate future design change.
Use validation selected-support quality for early stopping. This is discriminative energy ranking,
not likelihood or contrastive-divergence training; the combinatorial partition function is unused.

## 4. Controls and compatibility

| Arm | What it tests |
| --- | --- |
| Original checkpoint selection | Current support, predicted quality, and cost |
| Expanded pool, original scores | Search-budget effect without a learned energy |
| Expanded pool, random scores matched by cut count | Selection null |
| Unary residual and parameter-matched wider unary | Additional supervised scoring/capacity without explicit pairs |
| Unary plus pair residual | Added value of local cut interactions |
| Existing mixture alternative at matched budgets | Whether existing dependency modeling suffices; train under matched routing/data rather than assume a new mixture is calibrated |
| Target-informed selector | Diagnostic headroom, never inference |

First measure support with a frozen spectrum head. Changed support may be outside that head's
training distribution, so those spectrum scores are diagnostics. For a final scientific comparison,
train the same intensity architecture under each fixed selector using matched seeds, budgets,
losses, and routing, and report total training/inference compute.

Integrate energies into selection only in the first implementation. Preserve original mask
probabilities wherever downstream features/losses require them; never write an energy into
`mask_log_probabilities` or use it for retained probability/OOS accounting. Verify whether the
anchor consumes proposal probabilities in intensities and report that dependency. Any later
replacement of probability semantics needs separate compatibility work and an explicit objective.

## Validation, promotion, and next stage

Before implementation runs, test zero-residual ordering parity, symmetric pair scores, atom/bond
relabeling invariance, finite gradients confined to intended heads, pool deduplication, deterministic
budget stopping, and absence of target inputs in inference. On tiny graphs enumerate every mask
to verify energy sorting and distinguish pool truncation from a scoring bug; do not claim the
baseline heap enumerates exact pair-energy order. Verify the unchanged probability-output contract.

Report matched-budget candidate weighted recall, distinct masses, support-oracle cosine, missed
intensity, emitted-spectrum quality, and end-to-end latency/memory. Stratify by CE, molecular size,
cut count, and represented acquisition types. Use paired molecule bootstrap and multiple training
seeds, preserving all spectra of a molecule together. Thresholds, checkpoint, pool sizes, and
compute budget must be fixed before measurements; none are established in this document.

For coverage, promote only if gains beat search-only, unary, random, and relevant mixture controls
beyond paired variation without unacceptable runtime or spectral regression. For a predeclared
efficiency experiment, require the agreed runtime reduction and non-inferior coverage and spectral
quality against those controls; a failed coverage arm cannot be relabeled as an efficiency success. Otherwise record the
bounded failure and retain the baseline selector. Then evaluate the spectrum EBM with the chosen
selector frozen using its own null-adjusted Gate A. A later 2-by-2 comparison (baseline/bond EBM
selector crossed with baseline/spectrum EBM output) separates individual gains from interactions.
Both methods being requested does not imply both must be adopted if their gates fail.

## Current state and handoff

Completed: source-informed bond-cut design and user-requested ordering of the two EBM stages.
No implementation, configuration, checkpoint evaluation, or training for this plan. Unrelated
SMARTS source/documentation edits in the working tree were preserved.

The per-fragment efficiency variant (rank the enumerated pool by closed-form release probability,
no pair terms) is implemented and untrained: see
[fragment-marginal selection](masshatter_fragment_marginal_selection_plan.md) (2026-09-23).

Next: identify the anchor checkpoint and fixed validation subset, align target/metric definitions,
and measure current-support oracle versus predicted cosine. Establish finite-pool headroom or a
predeclared efficiency opportunity before implementing the pair head. Follow the
[session workflow](../../workflow/session.md) for revision/dirty-diff, environment, dataset, resolved
configuration, seeds, commands, and artifacts. Documentation validation checks links, anchors,
and whitespace; scientific thresholds and results remain pending.
