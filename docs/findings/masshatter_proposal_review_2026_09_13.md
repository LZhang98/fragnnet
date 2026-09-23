# MASShatter fragment proposal model review

> **Status:** FINDING · **Updated:** 2026-09-13 · Source and live W&B review complete; no new implementation or training.
>
> **Proposal coverage loss:** the loss term itself, across this and three other reviews, is
> consolidated in [MASShatter proposal loss — consolidated audit](masshatter_proposal_loss_audit_2026_09_22.md). This document remains canonical for the proposal *branch* — budgets,
> selection, diversity and mixtures.

## Scope and conclusion

This review concerns the fragment proposal model: bond-cut scoring, mask enumeration,
unique-fragment construction, and proposal coverage supervision. It does not review the
separate energy-based spectrum reranking proposal.

The first priority is to measure and improve recovery of missing fragments before enlarging
the encoder. The current loss can be low despite poor candidate coverage, and deterministic
selection limits direct supervision for fragments outside the selected support. Whether
these mechanisms materially limit trained-model quality remains unmeasured.

This review complements the [MASShatter improvement plan](../plans/masshatter/masshatter_improvement_plan.md)
and its support-oracle gate. It does not replace the existing experiment queue. The
[loss setup review](loss_setup_review_2026_09_13.md#auxiliary-weights-proposal-coverage-and-warmup-need-separate-interpretation)
is the canonical source for previously reproduced proposal-loss and gradient-routing
counterexamples; those probes were not rerun for this review.

## Current implementation

- [MassShatterModel](../../src/fragnnet/model/mass_shatter_model.py) predicts bond-cut
  logits and optional molecule-level Bernoulli-mixture weights. Proposal inputs can come
  from shared molecular features or a separate proposal encoder.
- [Candidate construction](../../src/fragnnet/frag/mass_shatter_candidates.py),
  `build_mass_shatter_candidates`, selects mask identities from detached logits on CPU,
  then recomputes their probabilities from live tensors. Selected cut/intact decisions
  retain gradients; the discrete selection operation itself does not.
- Unique-fragment enumeration retains a prefix reaching the requested number of distinct
  atom sets, subject to the enumeration budget and forced-mask policy. Pathways producing
  the same atom set are marginalized before the intensity transformer.
- [Proposal supervision](../../src/fragnnet/pl_model/mass_shatter_pl.py),
  `_proposal_coverage_loss`, rewards probability assigned to fragments whose valid ion/H
  channels match observed peaks. A fragment counts at most once per target peak.

## Prioritized improvements

### 1. Address the coverage-loss blind spot

**Established behavior:** uncovered target peaks contribute zero to the proposal loss.
The existing loss review shows that zero coverage can therefore produce zero loss and
zero proposal gradient. Proposal loss is not a standalone measure of candidate quality.

**Proposed improvement:** log intensity-weighted candidate recall independently and test
proposal supervision over a broader training candidate pool. Missing peaks can provide
useful ranking supervision only when the pool contains compatible fragments. Adding a
constant uncovered-peak penalty alone would not teach the model which cuts to select.

**Check:** report coverage, support-oracle quality, and proposal gradients alongside loss.
Measure whether a larger pool actually introduces explanations for previously missed peaks.
Distinguish search failures from peaks unreachable under the available chemistry.

### 2. Test controlled exploration beyond deterministic selection

**Established behavior:** the candidate builder uses deterministic ranked enumeration.
The forward comment mentioning training-only exploration is not evidence that sampled
exploration is implemented in that builder.

**Hypothesis:** repeatedly training on a narrow selected set can reinforce early proposal
preferences. Live probability gradients can still change future rankings, so this is not
a claim that candidate identities can never change.

**Proposed improvement:** reserve a small training budget for sampled or perturbed cut masks,
with deduplication and reproducible randomness. Initially keep inference deterministic.

**Check:** compare against deterministic selection at matched candidate budgets and measured
compute. Track newly covered target intensity, validation coverage, oracle quality, and
runtime. Training-pool gains alone are insufficient if deterministic inference loses them.

### 3. Separate warmup duration from gradient routing

**Established behavior:** `proposal_input_gradients_enabled` starts false. In
`MassShatterPL._apply_staged_training`, a null warmup returns immediately, whereas a
configured warmup enables proposal gradients into the shared encoder after warmup as well.
Thus warmup versus no warmup changes both staging and joint-training gradient routing.

**Proposed improvement:** expose routing as an independent choice with compatibility for
existing configurations, then compare warmup durations with identical joint-training routing.
Resolve this confound before attributing gains to a shared versus separate proposal GNN.

**Check:** verify per-term encoder gradients during warmup and joint training, and compare
matched schedules, seeds, context, losses, and candidate budgets.

### 4. Measure useful diversity within the fragment budget

**Established behavior:** distinct precursor atom sets define unique fragments. Distinct
atom sets can still generate overlapping masses, so unique-fragment count does not directly
measure spectral coverage.

**Proposed improvement:** measure distinct reachable masses, duplicate pathways, and marginal
coverage as the budget grows. If redundancy is substantial, test selection that balances
proposal score with additional reachable support. Preserve structurally distinct alternatives
that the intensity model may need; mass overlap alone does not make a fragment useless.

**Check:** report coverage and oracle curves versus fragment count and runtime, using the
same tolerance and mass aggregation as evaluation. Measure actual learned quality as well.

### 5. Test correlated cut patterns only when diagnostics justify them

**Established behavior:** a single Bernoulli component factorizes cut decisions conditional
on molecular features. Existing mixture support can express dependencies through a shared
latent component; adding this capability does not require a new proposal architecture.

**Proposed improvement:** compare one versus two mixtures only if missed support suggests
that alternative multi-cut patterns are a bottleneck. Hold retained-fragment and enumeration
budgets fixed, and measure full-step cost.

**Check:** inspect mixture usage/collapse, candidate overlap, and coverage on difficult
multi-cut molecules. Adopt additional capacity only for reproducible quality or coverage gains.

## First diagnostic experiment

Select one recoverable checkpoint and a fixed validation subset. Record code, configuration,
dataset/split identity, evaluation IDs, and tolerances under the
[session workflow](../workflow/session.md). Preserve preprocessing and acquisition conditioning.

Compare current candidate support with a larger enumeration pool while keeping proposal
weights fixed. Record actual retained fragment counts: a larger enumeration ceiling need
not expand the pool if the unique-fragment stopping target is reached early. If needed,
vary the retained-fragment budget as a separate diagnostic.

Measure intensity-weighted candidate recall, best attainable cosine on each support,
actual predicted cosine, and runtime. Define the support oracle under the same mass
aggregation, matching tolerance, and metric as the learned prediction; weighted recall
alone is not oracle cosine. Any approximate oracle must be labeled accordingly. Evaluate
relevant collision-energy and molecular-complexity slices, and acquisition strata where
the selected data contain variation.

| Observation | Next action |
| --- | --- |
| Larger pool improves support oracle | Investigate proposal search, exploration, and ranking. |
| Current support oracle is strong but predicted cosine is weak | Prioritize intensity learning under fixed support. |
| Both support oracles are weak | Inspect chemistry limitations and search-budget saturation before choosing an extension. |

Larger-support intensity predictions can be outside the checkpoint's training distribution;
use them as diagnostics, not as proof that retraining would improve quality. Use the existing
improvement plan's matched-seed and validation gates before promoting a model change.

## Live W&B follow-up: U256 is usually not reached

Read-only API inspection on 2026-09-13 scanned the full logged validation histories of
three recent merged runs. All use U256, ranked-mask budget E1024, zero relative mask
probability floor, and no forced single-bond cuts. Latest logged epoch means follow;
these are not best-checkpoint measurements.

| Run | Epoch / W&B state | Unique fragments | U256 reached | Selected masks | Duplicate component pathways |
| --- | --- | ---: | ---: | ---: | ---: |
| [JSS warmup 0–4, subformula](https://wandb.ai/frag-gnn/frag-gnn/runs/8v0i0fq7) | 166 / crashed | 170.00 | 5.50% | 977.68 | 95.03% |
| [CE warmup 0–15, subformula](https://wandb.ai/frag-gnn/frag-gnn/runs/7abal2ru) | 199 / finished | 169.20 | 5.88% | 976.47 | 95.07% |
| [Full metadata, GNN context + heads](https://wandb.ai/frag-gnn/frag-gnn/runs/clmrbaz3) | 199 / finished | 174.59 | 6.99% | 974.73 | 94.75% |

Across all 167, 200, and 200 logged validation epochs respectively, the epoch-mean
U256-reached rates ranged from 5.09–13.24%, 4.67–15.87%, and 2.23–18.12%.
Underfilling is therefore not confined to the final epoch. A crashed run's logged
diagnostics do not establish completion of its intended training schedule.

Metric names are `val_unique_component_count_epoch/mean`,
`val_unique_fragment_target_reached_epoch/mean`, `val_selected_mask_count_epoch/mean`,
and `val_duplicate_component_pathway_fraction_epoch/mean`. Current source defines
target-reached as the indicator that unique components reach U, and duplicate pathway
fraction as one minus unique components divided by component pathways. The latter is
not the fraction of duplicate emitted spectral peaks. Raw per-molecule stopping reasons
were not retrieved.

**Correction to the recommendation:** raising U alone is not the first intervention.
Start with E1024 versus E4096 at fixed U256, using the same checkpoint and validation IDs.
Measure additional unique fragments, newly covered observed intensity, support-oracle
quality, and runtime before introducing a separate supervision-pool selection rule.
Near-E1024 selected-mask counts and high redundancy motivate this diagnostic; they do
not prove useful extra coverage or that every underfilled molecule exhausted E. Small
graphs can have fewer possible masks. Older unmerged E4096 summaries showed higher
target-reached rates, so this finding is specific to the recent merged E1024 setups.

The proposal loss's uncovered-peak blind spot remains valid. What changes is the working
assumption that U256 truncation is the main practical cause. Exploration remains deferred.

Scratch extracts are `/tmp/masshatter_u256_wandb_review.json` and
`/tmp/masshatter_u256_wandb_history.json`; run links and essential measurements above are
the portable evidence. No remote runs were modified, and no training or checkpoint
evaluation was performed. This live budget check does not verify the missing historical
W&B screen report's quality rankings.

## Existing E1024 versus E4096 runs

A follow-up read-only query found 92 U256 runs with logged E1024 or E4096 configurations.
Use `mass_shatter_ranked_mask_budget`, not run names: some sweep names contain enum1024
but their configured budget is 4096. Two unmerged pairs differ in logged configuration
only by enumeration budget, W&B name, and tags. All four use seed 397623.

Compare the latest common epoch within each pair:

| Pair / epoch | E1024 → E4096 runs | Unique fragments | U256 reached | Candidate weighted recall | Binned cosine 0.01 |
| --- | --- | --- | --- | --- | --- |
| Standard / 8 | [msqh275h](https://wandb.ai/frag-gnn/frag-gnn/runs/msqh275h) → [3oqibfwg](https://wandb.ai/frag-gnn/frag-gnn/runs/3oqibfwg) | 192.86 → 235.47 | 22.64% → 75.76% | 0.92174 → 0.92849 | 0.67955 → 0.68192 |
| Learned OOS / 15 | [3z9y6yu9](https://wandb.ai/frag-gnn/frag-gnn/runs/3z9y6yu9) → [er846qd7](https://wandb.ai/frag-gnn/frag-gnn/runs/er846qd7) | 190.50 → 236.28 | 21.91% → 76.65% | 0.92266 → 0.92921 | 0.68262 → 0.68522 |

E4096 adds roughly 43–46 unique fragments but only 0.66–0.67 percentage points of
candidate weighted recall. Binned cosine increases by 0.00236 and 0.00261 at the matched
epochs. Selected-mask means roughly double: 934.02 → 1879.65 and 937.53 → 1871.02.
Mask counts are not wall-clock runtime measurements.

These existing runs already support a modest coverage benefit from larger enumeration
in the older unmerged setup. They do not justify assuming that filling U256 resolves the
uncovered-peak blind spot. Inspect the remaining unexplained intensity and support oracle
before a larger supervision-pool implementation or another training-budget comparison.
A same-checkpoint budget replay would isolate support effects from different training
trajectories, if needed; it is not a prerequisite for acknowledging the existing evidence.

Limits: all four runs report crashed, with unequal final training epochs. Comparisons use
common logged epochs, not matched wall time or converged models. Logged configurations
match within each pair, but code revision and dataset/cache fingerprints were not verified;
the retrieved metadata did not provide a Git revision. These runs predate subsequent
chemistry/audit changes and are not directly comparable to the recent merged runs above.
No support oracle, raw missed-peak overlap, or current-code replay was measured. Two
same-seed pairs are not independent multi-seed confirmation.

Scratch extracts: `/tmp/masshatter_e1024_e4096_runs.json` and
`/tmp/masshatter_e_budget_matched_histories.json`. Values are validation epoch means from
the same budget metrics above plus `val_candidate_wrecall_epoch/mean` and
`val_cos_sim_0.01_epoch/mean`.

## Subformula-filtered run check

A read-only query of `config.spec_params.filter_subformula_peaks=True` on 2026-09-13
returned 14 MASShatter unique-fragment runs in `frag-gnn/frag-gnn`. All configure U256/E1024;
no E4096 counterpart with this flag enabled was found. The older unmerged E comparison
above therefore does not establish the E4096 benefit on subformula-filtered targets.

Representative latest validation epoch means:

| Run | Epoch | Candidate weighted recall | Unique fragments | U256 reached |
| --- | ---: | ---: | ---: | ---: |
| [JSS warmup 0–4](https://wandb.ai/frag-gnn/frag-gnn/runs/8v0i0fq7) | 166 | 93.81% | 170.00 | 5.50% |
| [JSS warmup 0–15](https://wandb.ai/frag-gnn/frag-gnn/runs/38ugaxen) | 170 | 93.97% | 171.51 | 6.30% |
| [CE warmup 0–15](https://wandb.ai/frag-gnn/frag-gnn/runs/7abal2ru) | 199 | 93.87% | 169.20 | 5.88% |
| [KL warmup 0–15](https://wandb.ai/frag-gnn/frag-gnn/runs/kj17tksj) | 164 | 93.89% | 169.18 | 5.71% |

About 6% of filtered target intensity remains outside candidate coverage in these logged
aggregates. This is not the original unfiltered intensity denominator. Subformula filtering
tests compositional compatibility; it does not guarantee that MASShatter's selected
structural fragments and ion/H channels can generate every retained peak. The remaining
gap may reflect search, chemistry/channel limits, or matching details and cannot be
assigned to the proposal objective alone from these metrics.

The finding that U256 is usually inactive persists with filtering enabled. Next inspect
missed filtered peaks and their reachable explanations; do not transfer the older unfiltered
E4096 gain directly or describe roughly 7% as the measured filtered deficit. Query scope
is this project and the explicit logged filter flag; historical externally prefiltered data
without that flag were not identified. Scratch extract:
`/tmp/masshatter_subformula_wandb_runs.json`. No training or checkpoint replay ran.

## Selection-only versus joint training: live W&B comparison

A fresh read-only API query on 2026-09-13 returned 155 unique-fragment MASShatter runs
in `frag-gnn/frag-gnn`: 121 explicitly enable selection-only, and 34 disable it or
predate the logged field. Eight selected runs' full validation histories were scanned,
requesting the epoch, step, runtime, candidate weighted recall, and binned cosine together.
The comparisons below use common logged epochs, not unequal final-summary epochs.

### Selection-only improves coverage in the closest historical pair

Both runs use merged NIST20, U256/E1024, seed 397623, Gaussian spectral CE, proposal
anchoring, and forced single cuts. Excluding W&B labels, their logged configurations
differ only in the selection-only flag, proposal loss weight, and base-CE weight:
the joint arm uses base CE weight 0.1; the selection-only arm replaces it with
H-independent proposal coverage loss weight 0.1 and removes proposal probabilities
from final intensity scoring. This comparison is against joint **plus base CE**, not
a pure single-loss joint baseline.

| Setup | Run | Candidate weighted recall at epoch 142 | Binned cosine 0.01 at epoch 142 |
| --- | --- | ---: | ---: |
| Joint/anchored + base CE | [vu5nwu9q](https://wandb.ai/frag-gnn/frag-gnn/runs/vu5nwu9q) | 0.894504 | 0.728915 |
| Selection-only + proposal coverage loss | [p0fk89de](https://wandb.ai/frag-gnn/frag-gnn/runs/p0fk89de) | 0.909926 | 0.741526 |

At the latest common epoch, coverage increases by **1.54 percentage points** and
cosine by **0.01261**. Over common epochs 133–142, mean paired improvements are
**1.63 percentage points** and **0.01041**. The direction also holds at inspected
epochs 20, 50, 98, and 100. These temporal means describe one training pair, not
independent samples or a confidence interval.

The run summary for selection-only reports epoch 141, but its scanned history includes
validation epoch 142; the table uses the actual history records. Both runs report
crashed. Neither retrieved run metadata supplies a Git commit; dataset/cache identities
were not verified. This is supportive same-seed evidence, not a fully controlled causal
estimate or multi-seed confirmation. The older target/chemistry protocol must not be
numerically pooled with the recent subformula-filtered runs.

### An existing intensity-head comparison supports preserving the separation

Another pair retains selection-only, separate proposal encoder, U256/E1024, no forced
single cuts, identical logged data settings, and seed 397623. Its only non-label logged
configuration changes are the intensity-transformer dimension, number of layers,
feedforward dimension, and inducing-token count:

| Intensity head | Run | Candidate weighted recall at epoch 98 | Binned cosine 0.01 at epoch 98 |
| --- | --- | ---: | ---: |
| Dim 256, 2 layers, FF 512, 64 inducing tokens | [lcwsng2h](https://wandb.ai/frag-gnn/frag-gnn/runs/lcwsng2h) | 0.896286 | 0.734168 |
| Dim 384, 3 layers, FF 1024, 128 inducing tokens | [ef7fl5c6](https://wandb.ai/frag-gnn/frag-gnn/runs/ef7fl5c6) | 0.893599 | 0.743135 |

Cosine improves by **0.00897** despite candidate weighted recall decreasing by
0.27 percentage points. Over common epochs 89–98, average cosine gain is 0.00872
and coverage change is -0.16 percentage points. This supports an intensity-model
contribution beyond additional covered mass, although candidate identities were not
held fixed or replayed. It does not identify which of the four head changes helped.
The small-head run crashed at epoch 98; the larger-head run finished at 199. Comparison
is at equal epochs, not equal wall time; code/data provenance remains unverified.

### Recent matched loss runs do not establish KL superiority

The filtered JSS and KL warmup 0–15 arms differ in logged configuration only by
`loss_type` and W&B labels. Both retain selection-only, independent output, Gaussian
support BCE/gating, the same seed, and the same warmup:

| Loss | Run | Coverage at common epoch 164 | Cosine at epoch 164 | Best logged cosine over its own history |
| --- | --- | ---: | ---: | ---: |
| JSS | [38ugaxen](https://wandb.ai/frag-gnn/frag-gnn/runs/38ugaxen) | 0.939339 | 0.797770 | 0.799603 at epoch 153 |
| KL | [kj17tksj](https://wandb.ai/frag-gnn/frag-gnn/runs/kj17tksj) | 0.938880 | 0.796526 | 0.798366 at epoch 135 |

Both report crashed; JSS has history through epoch 170 and KL through 164. Over
common epochs 155–164, coverage differs by only about 0.002 percentage points and
JSS cosine averages 0.000962 higher. The different gradient behavior reproduced in
the [loss review](loss_setup_review_2026_09_13.md) is therefore not sufficient reason
to claim KL is the better loss or repeat this existing comparison without a new question.

The shorter-warmup [JSS run](https://wandb.ai/frag-gnn/frag-gnn/runs/8v0i0fq7) has best
logged cosine 0.800989 at epoch 140 and coverage 0.936876 at that same epoch. Its only
non-label configuration difference from the JSS arm above is warmup ending at 4
instead of 15. A [selection-only CE arm](https://wandb.ai/frag-gnn/frag-gnn/runs/7abal2ru)
has best logged cosine 0.798527, but differs from JSS in output parameterization and
support supervision/gating as well as loss, so it does not isolate CE versus JSS.

### Implication for the intensity model

The evidence supports retaining selection-only as the working proposal baseline.
Proposal scores can optimize useful fragment coverage while a separate intensity
objective learns the relative signal over retained fragment/ion/H candidates.
Do not reconnect proposal probabilities as mandatory intensity multipliers merely to
reduce the number of losses; this would change the separation supported by the pair above.

The [existing intensity head](../../src/fragnnet/model/bond_fragment_heads.py) already
uses candidate-set attention and optional atom cross-attention. Its candidate tokens are
fragment-level; ion/H logits are emitted by a final linear head. An explicit ion/H
candidate decoder is a possible extension, not an implemented or validated improvement.

Before selecting another head or objective, use one recoverable selection-only checkpoint
and fixed candidate sets to measure whether missed intensity is primarily between-fragment
ranking or within-fragment ion/H allocation. Record a support oracle under the evaluation
matching rule and raw-intensity rank/base-peak diagnostics. No oracle metric was found
in the inspected recent run summaries. Candidate weighted recall near 94% and cosine
near 0.80 are different quantities and cannot be subtracted to claim a 14-point oracle gap.

For a subsequent intensity comparison, freeze or cache the exact selected support and
hold it fixed across heads; if molecular features are shared, simply freezing proposal
head weights does not freeze candidate identities. Start with the current transformer as
the control. A channel-aware decoder is justified only if within-fragment allocation is
the measured limitation. This informs the intensity stage of the existing program; no
new architecture, training, or scheduler run was started.

Scratch records: `/tmp/masshatter_selection_wandb_index.json`,
`/tmp/masshatter_selection_wandb_histories.json`, and read-only query script
`/tmp/masshatter_selection_wandb_review.py`. Run links, query scope, matching rules,
configuration differences, and essential values above are the portable evidence.

## D4 reference and revised coverage interpretation

The [saved subformula-filtered D4 comparison](subformula_filter_ft_qtof.md#saved-subformula-filtered-d3d4-comparison-verified-2026-09-13)
was verified from its per-spectrum artifact. It supports viewing the filtered MASShatter coverage
above as strong, while its separate cleaning, population, and target construction prevent a matched
gap estimate. The canonical finding records the metrics, exclusions, and provenance limits.

The remaining uncovered intensity is not all demonstrated proposal-scoring headroom. Before
another architecture or budget intervention, compare current-support oracle cosine with predicted
cosine on identical targets. Prioritize intensity assignment if that gap dominates; pursue cut
modeling only for demonstrated support gains, efficiency at preserved coverage, or downstream
quality at matched coverage. Earlier search/coverage recommendations in this review are conditional
on those diagnostics, not on U256 underfilling alone.

## Validation and handoff

This was a source and documentation review. No model code, configuration, data, or environment
was changed, and no new model tests, training, or checkpoint evaluation ran. The referenced
W&B screen report is missing from this checkout; the fresh, separately scoped API checks
above provide the current comparison evidence without relying on that report.

The document and navigation links were added while preserving pre-existing documentation
edits. Next check: select the checkpoint and validation subset, then run the bounded support
diagnostic before implementing a proposal architecture change.
