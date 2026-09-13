# Energy-based spectrum reranking experiment

> **Status:** PROPOSED · **Updated:** 2026-09-13
> Shared design of record for FraGNNet and MASShatter reranking; no implementation or runs.

## Objective and rationale

Test whether a learned energy function can select better whole-spectrum intensity patterns
from a small candidate pool produced around a frozen baseline prediction. Separate proposal
quality from ranking quality before considering a generative EBM or iterative inference.

This document owns the common export contract, proposal/null pools, gates, ranking objective,
controls, and decisions. The initial anchor below is FraGNNet; the
[MASShatter delta](masshatter_energy_based_model_plan.md) changes only model-specific setup and
records conditional refinement. The MASShatter program now evaluates
[bond-cut modeling first](masshatter_bond_cut_ebm_plan.md), then applies this spectrum protocol
with the selected proposal model frozen. Apply the spectrum gates separately to each anchor.

FraGNNet already normalizes fragment/formula logits in
[fragnnet_model.py](../../src/fragnnet/model/fragnnet_model.py). Calling negative logits
energies does not change that model. This experiment instead scores a proposed spectrum:

```text
E_theta(G, c, y) -> scalar compatibility energy; lower is better
```

Here `G` supplies molecular/fragment embeddings, `c` supplies acquisition conditions, and
`y` supplies proposed peak intensities. Energy is a learned compatibility score, not a physical
bond-dissociation energy. Ranking training does not yield calibrated spectrum probabilities.

The [fragment ranking findings](../findings/fragment_intensity_ranking_findings.md) and
[dominant-peak decomposition](../findings/dominant_peak_outcome_decomposition.md) motivate
testing intensity allocation within existing support. Their measurements are not evidence
that an EBM will improve performance. Compare successful results with the
[minimal FT ranking-loss plan](ft_contender_pair_minimal_training_plan.md) and the existing
[fragment competition head](../../src/fragnnet/model/fragment_competition.py).

### Prior output-correction null

The FormulaBagCorrectionHead sweep is relevant negative evidence, not a result to omit when
motivating another fixed-support intervention. A historical report can be read with:

```bash
git show baaaa61ec00ea864861442ce492fd7ad1fc2e893:docs/formula_bag_ce_response_curve_diagnostic.md
```

That report records the 2026-07-14 CE-response sweep as null within seed variation, including
high-NCE slices, and diagnoses nearly uniform per-bag corrections that cancel under softmax
(runs `mirlys0d` and `ykh2hts0`). It reports the CE-blind head as near-neutral. This evidence
was inspected in Git history; its source document and implementation are absent from the current
checkout. The broader conclusion “the ceiling isn't the output distribution” is user-reported;
the inspected diagnostic establishes this particular correction failure, not a universal ceiling.
Recover the full sweep/provenance and reconcile its comparison conditions before training.

The proposed reranker consumes complete alternative intensity assignments and learns their
relative metric quality, whereas that head directly emitted per-bag logit corrections. This is
a different optimization and inference test, but it shares the risk that frozen representations
cannot support useful correction. Require the null-adjusted pool gate and a matched direct
correction control; a positive raw oracle does not rebut the prior null. The FraGNNet ranking
finding also does not establish that MASShatter has the same error decomposition.

## 1. Freeze the baseline and data contract

- Select one trained FraGNNet checkpoint and preserve its molecule-disjoint train/validation/test
  split. All collision energies and replicates of a molecule must stay in the same split.
- Start with FT spectra, `[M+H]+` and `[M-H]-`, consistent with the minimal ranking pilot.
  Preserve instrument, adduct, fragmentation mode, and the baseline's CE convention.
- Freeze the encoder and prediction head. Use training molecules to fit the reranker,
  validation molecules to select settings, and the locked test split for final evaluation.
- Export with `scripts/run_inference.py --min_intensity -1` plus the resolved checkpoint/data
  arguments. The CLI default is `1.0` on the 0–1000 scale (0.1% of the base peak); never use that
  default for support/oracle analysis. Its help says peaks at or below the threshold are removed,
  which would exclude exact zeros at `0`. However, the current `filter_by_peak_intensity` and
  `batch_filter_by_peak_intensity` functions in
  [script_utils.py](../../src/fragnnet/utils/script_utils.py) bypass filtering for
  `min_peak_int <= 0`, so zero currently also disables this filter. Use explicit `-1` in the
  experiment manifest rather than depending on the ambiguous help. This disables the export
  intensity filter only: audit upstream model pruning, zero handling, and serialization to verify
  complete retained chemical support. Inspect the
  [correctness audit](../findings/fragnnet_masshatter_audit_2026_09_12.md) and reconcile
  relevant checkpoint, export, and metric issues against current code before measuring.
- Aggregate candidate ions/fragments into predicted peak-mass groups using the existing mass
  convention. Pool all associated fragment embeddings for a mass; do not assign an observed
  peak to one arbitrary latent fragment. This grouping must not depend on observed peaks.
- Preserve the full target spectrum for evaluation, including peaks outside prediction support.
  Record normalization, intensity transform, mass tolerance, matching algorithm, isotope handling,
  and treatment of any out-of-support probability. Use the same choices in every arm.

The first check is a small export audit confirming split identity, complete support, grouping,
finite normalized intensities, and agreement with the baseline evaluator. Existing exports
must not be assumed to contain frozen embeddings; inspect their schema before choosing online
encoding or a separate experiment cache.

## 2. Generate a bounded candidate pool

Proposed starting budget: 32 spectra per input, including the unchanged baseline. These are
pilot settings, not measured optima. Use only inference-available inputs to propose candidates:

1. Transfer intensity between pairs among the top 16 baseline masses.
2. Sharpen or flatten the baseline intensity distribution.
3. Promote plausible lower-ranked masses already present in the candidate support.

Keep masses fixed, intensities nonnegative, and total intensity normalized. Preserve support
outside the edited masses. Deduplicate candidates; report actual pool size for small supports.
Record proposal-family counts, transfer fractions, sharpening exponents, lower-rank selection,
and random seeds in a resolved experiment configuration before the first oracle measurement.
Candidate generation must be identical in training and inference and independent of targets.
Do not inject the observed spectrum or an observed winner into evaluation proposals.

### Matched random-perturbation null

For every structured pool, generate an equal-size null pool containing the same baseline plus
31 label-independent random perturbations (or the same realized unique count on small supports).
Use the same fixed masses, normalization, proposal-family counts, perturbation magnitudes,
number of edited masses, and feasibility rules. Randomize donor/recipient or promoted-mass
identities within the same baseline-rank/eligibility strata, removing chemical/context preferences.
Shared global sharpening candidates remain identical in both pools. Record realized L1/L2
perturbation sizes and donor mass limits; clipping, duplicate removal, or unequal pool sizes must
not silently give one pool a larger search budget. Reduce both pools to a common feasible count
or report an unmatched stratum separately without using it to pass the gate.

Both generators may use baseline predictions and valid support, but neither may use labels.
Freeze several null/proposal seeds and all settings before measuring. If a proposed “structured”
family is already random under these matching constraints, declare it equivalent to the null;
do not manufacture a weaker null to get a positive difference.

### Gate A: oracle gain beyond matched random search

For each validation spectrum and paired proposal seed, use the frozen primary cosine metric:

```text
structured_gain = max_k cosine(structured_k, observed) - cosine(baseline, observed)
null_gain       = max_k cosine(random_k, observed) - cosine(baseline, observed)
excess_gain     = structured_gain - null_gain
```

The observed spectrum is used only for diagnostic selection and training labels. Both raw gains
are nonnegative up to numerical tolerance because each pool includes the baseline. A positive
raw gain is expected even from target-independent noise selected after seeing the target; it is
not evidence of informative proposals. Gate A uses only the excess as evidence of structured-pool
headroom beyond matched random search. This is a finite-pool comparison, not the DAG support oracle.

Report raw gains and paired excess, their distributions and molecule-bootstrap intervals,
proposal-family winners, actual pool sizes, and sensitivity across predeclared random seeds.
Average paired seed results per spectrum before molecule bootstrap; do not count perturbations
or replicate spectra as independent molecules. Before measuring, declare a positive minimum
useful excess and a latency budget. Require the lower bound of a 95% paired molecule-bootstrap
interval for mean excess to exceed that minimum, with the direction consistent across seeds.
No numerical practical threshold is established here; it must be fixed before launch.

If excess is negligible, negative, or unresolved, stop before implementing/training the energy
head. A larger raw oracle does not override this gate. Revise proposal structure only on a
validation-development subset; confirm the frozen gate on a reserved molecule-disjoint validation
subset after tuning. Do not use test-oracle results for development.

Positive excess is necessary for this structured-pool pilot, but does not prove that the ordering
is learnable from available features. A failed scorer after this gate still requires checking
feature signal, training, and proposal distribution shift; it is not automatically an energy-head
capacity defect. Random pools may also contain learnable improvements; rejecting this pilot is
not a claim that all random-search rerankers or EBMs are impossible.

## 3. Fit a small energy head

Proceed only after Gate A and the prior-null reconciliation pass. For each candidate mass,
provide pooled frozen fragment embeddings, mass, candidate intensity,
baseline intensity, and acquisition context. Use a small permutation-invariant set scorer
(for example token MLP, pooled summary, scalar MLP) for the first pilot. Cache molecule features
once where practical and batch candidate scoring. Candidate intensities must enter the scorer;
otherwise it cannot distinguish proposals for the same input.

For training candidates `a` and `b`, compute their primary-metric similarities `s_a` and `s_b`
to the observed spectrum. If `s_a > s_b` by the configured tie tolerance, minimize:

```text
L_pair = w_ab * softplus(E_theta(G, c, y_a) - E_theta(G, c, y_b))
```

Use a bounded weight based on the similarity gap, downweight or omit near-ties, and average
pairs within each spectrum before averaging spectra. Fix pair sampling, tie tolerance, energy
head size, optimizer, regularization, and training budget on validation. Labels and observed
intensities must never enter the scorer inputs. Do not train only on easily recognized
synthetic corruptions: include baseline-like alternatives with incorrect dominant peaks and
plausible intensity ratios from the actual inference proposal generator.

Use early stopping on validation selected-spectrum cosine, not just pair classification
accuracy. A successful corruption discriminator is insufficient evidence of useful reranking.

## 4. Inference and controls

```text
y_selected = argmin_k E_theta(G, c, y_k)
```

Break exact ties in favor of the baseline when it is tied. Baseline inclusion does not
guarantee improvement: the learned scorer can still select a worse candidate. This pilot uses
finite candidate ranking; no partition-function estimation or iterative sampling is needed.

| Arm | Purpose |
| --- | --- |
| Frozen anchor model | Baseline performance and cost |
| Validation-tuned global sharpening/flattening | Simple intensity-calibration control |
| Matched direct intensity-correction network | Same frozen features, training data, tuning and reported parameter/compute budget; tests ordinary output correction |
| Learned energy reranker on structured pool | Test candidate-dependent whole-spectrum scoring |
| Same reranker architecture trained/evaluated on matched random pools | Tests whether proposal structure adds value beyond generic perturbation search |
| Best structured and best random candidate using the observed spectrum | Paired diagnostic upper bounds, never deployable |

If the reranker passes, compare with the ordinary ranking-loss and fragment-competition
experiments using matched data, checkpoint provenance, tuning effort, and reported compute.
An EBM-specific advantage is not established by beating only an unchanged baseline.

## 5. Evaluation and decision

Primary outcome: paired change in mean spectral cosine under the exact baseline metric.
Secondary outcomes: JSS, candidate and emitted-spectrum weighted recall, false-peak intensity,
peak count, base-peak accuracy, adjacent-CE winner-switch accuracy, stable-winner
accuracy, regressions on previously correct spectra, and inference latency/peak memory.
For CE transitions, hold molecule, adduct, instrument, and fragmentation mode fixed; preserve
the existing CE-ordering convention and report replicate handling and transition counts.

Report all pilot spectra and separately the subset whose observed base peak is in support;
do not silently restrict the headline result to supported or oracle-improvable cases. Use
paired bootstrap resampling by molecule, retaining its spectra and transitions together.
Report multiple reranker seeds for confirmation and the exact test sample counts.

| Result | Interpretation and next step |
| --- | --- |
| Positive raw oracle, negligible or unresolved excess over null | Gate A fails; no demonstrated headroom beyond matched random search; revise proposals or stop before trainer work |
| Robust excess, no selected-spectrum gain | Learnability unresolved; inspect feature signal, training and proposal shift before attributing failure to the energy head |
| Calibration, direct correction, or random-pool reranking improves equally | No demonstrated advantage for the structured EBM treatment |
| Reranker improves beyond controls within the latency budget | Confirm across seeds and compare ranking/competition baselines |

Lock proposals, checkpoint selection, scorer settings, and practical success criteria before
final test evaluation. A positive test result needs improvement beyond calibration and matched
learned controls,
paired uncertainty supporting the gain, and acceptable accuracy regressions and runtime.
Fixed support cannot recover absent masses. Whole-spectrum relationships are hypotheses to
learn; do not impose mutual exclusion on alternative fragmentation paths, since a spectrum
aggregates many events.

## Deliverables and handoff

Future implementation should produce a resolved configuration/provenance manifest, proposal
oracle/null report, trained energy head (only after Gate A), and per-spectrum comparison table
with baseline, selected, and both oracle scores, excess gain, proposal/null seeds and identities,
metadata, and timing. Record dataset/split/checkpoint identities, Git revision and dirty diff, environment, seeds, and actual artifact paths.

Completed: documented this bounded pilot and linked it in the project indexes. No model code
or complete executable experiment commands were added; no scientific results have been measured.
Documentation validation consists of local-link checks and `git diff --check`.

Next concrete step: select the baseline checkpoint and inspect a small unfiltered export, then
reconcile the historical correction null, resolve matched structured/null settings, and pass Gate A
before implementing the energy-head trainer. The checkpoint, practical excess threshold, and
latency budget remain unresolved; this documentation update is not a gate result.
