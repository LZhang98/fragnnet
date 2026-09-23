# MASShatter EBM program and spectrum setup delta

> **Status:** PROPOSED · **Updated:** 2026-09-13
> Bond-cut EBM first, spectrum EBM second; spectrum setup delta below. No implementation or runs.

## Objective and decision

The user selected **both bond-cut and spectrum-level EBM modeling, with bond-cut modeling first**.
The [bond-cut EBM plan](masshatter_bond_cut_ebm_plan.md) owns the first experiment: correlated
cut scoring over a bounded proposal pool, with search, unary, random, and mixture controls.
After its evaluation, freeze the selected proposal model and run the spectrum experiment.
A negative first-stage result retains the baseline selector; it does not silently cancel stage two.

The [energy-based spectrum reranking plan](../energy_based_spectrum_reranking_plan.md) remains the
**design of record for spectrum reranking**: export contract, structured/random intensity pools,
null-adjusted Gate A, ranking objective, controls, metrics, and decisions. The spectrum sections
below specify only MASShatter setup differences and conditional refinement; they do not duplicate
that protocol. A later factorial comparison tests the combination of both EBM interventions.

This sequence updates the priority within the requested EBM program. Existing
[MASShatter improvement](masshatter_improvement_plan.md) and
[intensity](masshatter_nonprobabilistic_intensity_plan.md) evidence and gates still apply.
No performance advantage is assumed, and documenting this order does not launch training.

## D4 evidence update

The [verified D4 comparison](../../findings/subformula_filter_ft_qtof.md#saved-subformula-filtered-d3d4-comparison-verified-2026-09-13)
supports treating current MASShatter intensity coverage as strong. Its numerical proximity to D4
is informative context, but differences in targets and evaluation prevent a matched ceiling claim.
The uncovered fraction is not demonstrated cut-scoring headroom.

Preserve the user's bond-cut-first order as a diagnostic priority. A pair head is conditional on
matched support gains or a predeclared efficiency opportunity. If current support already permits
much better spectra than the learned predictor produces, keep the baseline proposal model and
proceed to spectrum/intensity modeling under the shared gate. This does not assume that a spectrum
EBM will outperform ordinary intensity learning, and it does not promote either EBM without controls.

## Established implementation and alternatives

The inspected [model](../../../src/fragnnet/model/mass_shatter_model.py) already supports mixtures
of factorized bond-cut proposals, candidate interactions through an intensity transformer,
normalized probability output, and independent Softplus intensities with optional support gating.
An EBM must add a testable mechanism beyond renaming negative logits as energies.

| Energy target | Potential benefit | Main limitation |
| --- | --- | --- |
| Bond-cut configuration | Model interacting cuts beyond factorized proposal components | Combinatorial search and no directly observed pathway labels |
| Individual fragment/channel | Rank chemically valid candidates | May be equivalent to existing softmax scoring |
| Complete spectrum | Score joint support and intensity assignments | Negative construction and inference cost |

Investigate cut-configuration energies first under the linked bond-cut plan. Use complete-spectrum
scoring next for intensity/clutter, preserving the selected support during that second experiment.
Neither a reranker nor an intensity refiner can recover masses absent from its fixed support.

## Stage two: MASShatter spectrum setup delta

- Replace the canonical FraGNNet anchor with a recoverable MASShatter checkpoint. Freeze its
  proposal generator, encoder, and intensity predictor; record probability versus independent
  output and support-gating settings. Select the actual supported acquisition strata and existing
  molecule-disjoint split rather than assuming the FraGNNet FT/adduct subset transfers unchanged.
- Retain valid fragment/ion/H candidates and pool their features into equal-mass groups before
  energy scoring. Do not assign a unique structure to an ambiguous peak. Candidate enumeration,
  hard selection budget, and chemistry masks remain fixed across arms.
- Follow the [canonical export contract](../energy_based_spectrum_reranking_plan.md#1-freeze-the-baseline-and-data-contract):
  explicitly set `--min_intensity -1`; verify upstream channel/fragment pruning and zero handling.
  This flag alone cannot restore candidates already removed by the model.
- Measure a MASShatter-specific support oracle and coverage using the
  [improvement plan's gate](masshatter_improvement_plan.md#ordered-experiments). The FraGNNet
  dominant-peak ranking/coverage ratio is not a measured MASShatter result.
- Apply the [prior-null reconciliation](../energy_based_spectrum_reranking_plan.md#prior-output-correction-null)
  to this anchor. The historical FormulaBagCorrectionHead null argues against assuming that a
  frozen-output correction will help, even though scoring complete alternatives differs from
  direct per-bag logit correction. Do not duplicate or broaden that evidence here.
- Run the canonical matched random-pool Gate A before any energy-head trainer work. Include all
  canonical calibration, direct-correction, random-pool, and oracle controls. Thresholds and
  runtime budgets must be recorded for this anchor before measuring.

The current intensity transformer already models candidate interactions. The proposed energy
must consume the complete candidate intensity assignment so it can distinguish alternative
outputs. Learned compatibility energy has no automatic physical-energy interpretation; the
initial ranking objective does not estimate a partition function or calibrated likelihood.

## Conditional extension: iterative refinement

Only after the canonical reranker passes held-out quality and compute gates, and alternative-pool
headroom remains, test a small fixed number of gradient steps on intensities with nonnegativity
and normalization enforced. Start from the baseline and compare against a direct refiner under
the same compute budget. Keep masses, chemical validity, and candidate support fixed.

Ranking training constrains energies only on its sampled alternatives. Optimization can exploit
unreliable energies elsewhere; test whether lower optimized energies correspond to better spectra
on held-out molecules. A ranking result alone does not establish useful generation. Validate
finite gradients, candidate-order invariance, equal-mass aggregation, and valid normalized outputs
before a learning probe. Full generative training, MCMC, partition-function estimation, physical
energy supervision, and joint proposal training remain deferred.

## Current state and handoff

Completed: architecture assessment, review corrections, and bond-cut-first program ordering.
No implementation, checkpoints evaluated, or scientific measurements for this proposal.
Follow the canonical plan's provenance and validation requirements and the
[session workflow](../../workflow/session.md); model tests are unnecessary for this documentation edit.

Next: select the MASShatter checkpoint and split and execute the bond-cut plan's support/pool
diagnostics. Reconcile the prior output-correction null and apply the shared intensity-pool gate
before the later spectrum stage.
Checkpoint, numerical gates, proposal settings, and compute budget remain unspecified. Keep the
shared protocol in the canonical plan when revising either anchor; this delta must not redefine it.

## Literature basis

- [Structured Prediction Energy Networks (Belanger and McCallum, 2016)](https://proceedings.mlr.press/v48/belanger16.html)
  defines learned energies over structured outputs and prediction through output optimization.
  It motivates this architecture, not a demonstrated mass-spectrum improvement.
- [End-to-End Learning for Structured Prediction Energy Networks (Belanger et al., 2017)](https://proceedings.mlr.press/v70/belanger17a.html)
  is a reference for a later differentiable iterative-refinement experiment.
- [Learning Proposals for Practical Energy-Based Regression (Gustafsson et al., 2022)](https://proceedings.mlr.press/v151/gustafsson22a.html)
  motivates learned proposals if normalized probabilistic EBM training is pursued later; its
  additional machinery is outside the initial ranking experiment.
