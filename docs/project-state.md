# Current project state

> **Status:** REFERENCE · **Updated:** 2026-09-15

## Current handoff

The shared agent workflow is established: start at [AGENTS.md](../AGENTS.md), follow the
[session workflow](workflow/session.md), and use the [architecture map](architecture.md).
Existing AGENTS and Claude rules are organized into [engineering](rules/engineering.md),
[Python and ML](rules/python.md), and [documentation](rules/documentation.md) rules.
The agent entry point selects the applicable files by task; the original rules are preserved.
This includes import aliases, formatting/docstrings, testing conventions, and documentation
placement. Where legacy Claude constraints differed, the original AGENTS policy is retained:
preprocessing changes require explicit authorization and compatibility work; PR validation runs
when practical, with omitted checks reported.
The adoption changes remain uncommitted for user review.

Validation: shared-rule preservation, relative documentation links, and Git whitespace checks
passed. The configured interpreter provides pytest and Ruff. No model code changed; no training,
scientific evaluation, or full test suite was run for this documentation change.

MASShatter CMF alignment (2026-09-14): MASShatter now emits the full configured H-transfer
ladder for chemically valid, non-degenerate CMF adducts, matching FraGNNet. It preserves
reachable CRF/CMF duplicate suppression and in-source-loss exclusion; the pre-batch support
filter shares the same contract. Existing checkpoints are diagnostic-only because their newly
valid CMF-H channels were not trained. Validation: 3,032 passed, 21 skipped, 2 xfailed; the
CMF plan records the required oracle-coverage rebaseline and matched-control training.

## Research navigation

- [Energy-based spectrum reranking](plans/energy_based_spectrum_reranking_plan.md) is a proposed
  shared design of record with a FraGNNet anchor (2026-09-13). Next: reconcile the prior correction
  null, verify export with `--min_intensity -1`, and require oracle excess over matched random pools
  before implementing/training an energy head. Documentation only; no runs or implementation.
- [MASShatter improvement plan](plans/masshatter_improvement_plan.md) records the proposed
  unmerged sequence: corrected CE control, proposal-encoder comparison, Gaussian support-gated
  JSS, conservative output pruning, and measured throughput improvements. Five merged warmup
  configs are prepared for a CE/JSS/KL loss and intensity-transform screen. Next: verify checkpoint
  persistence before launching; resolve the unmerged anchor/warmup policy before its matched
  control. The plan includes [Vulcan generation and submission commands](plans/masshatter_improvement_plan.md#vulcan-commands-for-the-five-config-screen)
  for all five configs with a three-hour allocation. No runs were launched for this plan.
- [MASShatter code audit](findings/masshatter_code_audit.md) records three reproduced
  issues at `0b5ad3f4`: standalone SMARTS inference, charged-component masses, and
  SMARTS proposal gradient isolation. All three fixes and regression tests are present
  at `77fdbea2`. A continued audit reproduced two additional inference blockers:
  unconditional DAG-file filtering/dataset selection in the CLI and independent-output
  tabular export expecting `pred_logprobs`. Those fixes and the mass-feasibility
  alignment are now in the working tree, along with CLI directory/row-layout fixes.
  Validation: 781 passed, 8 skipped; six CLI regressions passed after adding Parquet
  serializer-boundary coverage. Real Parquet serialization was not verified because
  its optional engine is unavailable; checkpoint loading and dataset preprocessing
  are mocked in the CLI tests. Next check: a target-environment real-checkpoint CLI
  smoke run. Review the linked compatibility
  notes and re-evaluate affected checkpoints before comparing experiment metrics.
  A third audit pass found two remaining API issues: prediction-only
  `run_inference_pkl` rejects its existing `pred_ints` output, and direct CPU bfloat16
  calls with multiple proposal mixtures fail the normalization check. Standard CPU
  runner fallback avoids the latter. The pickle API fix is now in the working tree:
  90 tests passed, including real-model prediction-only regressions for both output
  modes and batch offsets. The bfloat16 issue is deferred by user because it is not
  used in the current workflow. Separate failure reproductions remain in the audit.
  The checkout was clean at audit start, so the
  uncommitted-work description in the workflow-adoption handoff above is historical.

- [Model improvement roadmap](plans/model_improvement_plan.md) remains the canonical source
  for research priorities and acceptance criteria. Its banner is dated 2026-07-24; reconcile
  its proposed next action with current code and artifacts before resuming it.
- [Documentation index](README.md) lists live, proposed, landed, and closed work, including
  MASShatter plans. Those are recorded statuses, not verification of current execution.
- Read linked [findings](findings/) before reviving a rejected or closed experiment.

No training processes, scheduler jobs, checkpoints, or external run statuses were audited during
workflow adoption. No new research priority or scientific conclusion is inferred here.

## Next task

Candidate attention topology audit (2026-09-13): the
[audit section](findings/masshatter_code_audit.md#candidate-attention-topology-audit-2026-09-13)
records seven findings on the uncommitted `full` candidate-attention arm, all fixed in the
working tree: a hyperparameter fallback default plus a missing `get_pl_hparams` migration,
corrected A/B parameter counts, an unstated capacity confound, a missing unique-fragment guard,
a pre-existing CUDA bfloat16 `index_add` crash, two documented degenerate head inputs, and a
per-layer host sync in `_efa_linear_attn`. Bucketed attention masking was verified exact for both
topologies, and `full` is cheaper only up to about 256 candidates. Validation: full suite plus
274 focused tests, Ruff check and format on changed files, and construction of both A/B arms.
A pre-rebuild full-suite run gave 92 failed, 2,873 passed; every failure was the then-stale
multi-cut Cython extension, all carrying `compute_ccs_multi_cut()` unexpected-keyword errors
for `max_ring_cut_steps` or `ring_open_mask`. Both extensions were rebuilt in place at
2026-09-13 18:12, and `mass_shatter_core` was force-rebuilt at 22:38 to a byte-identical
binary, confirming the 18:12 build already matched its source. All four extensions are now
current and the full suite passes: 2,975 passed, 1 skipped, 2 xfailed.
Next: decide whether to remove the degenerate `component_residual_head` and constant mask-count
feature, which would invalidate existing MASShatter checkpoints and so needs explicit
authorization. No training ran and no run was launched for this arm.

D4 oracle verification (2026-09-13): the
[subformula-filter finding](findings/subformula_filter_ft_qtof.md#saved-subformula-filtered-d3d4-comparison-verified-2026-09-13)
records reaggregation of the saved D3/D4 artifacts. MASShatter's roughly 94% coverage is strong
relative to that unmatched D4 reference; the remaining 6% is not established cut-scoring headroom.
No preprocessing, DAG generation, or model inference reran. Next: compare current MASShatter
support-oracle and predicted cosine on identical targets. Bond-cut-first remains the requested
diagnostic order, with training conditional on useful support or efficiency headroom.

MASShatter H-support ablation (measurement 2026-09-14; code clarification 2026-09-15): the
[ion/H channel plan](plans/masshatter_ion_h_channel_selection_plan.md#h-support-ablation-measurement-2026-09-14)
now records the matched 2026-09-08 gated / supervision-only / no-support arms read from W&B. No
trained arm exercises channel selection: retained channels equal valid channels in every arm
(about 8.5 per fragment against roughly 1.02 attributed), so the hard-selection stage has never
been measured. The soft support gate shows no spectral or weighted-precision benefit in either
data variant, and all dense cosine differences fall below one per-epoch standard deviation of a
single seed. The plan now records clone-initialized self-gating as an untested explanation for
the gate null, and clarifies that the six-run scope tests only H support (one valid ion mode,
at most nine CRF H channels). The support BCE does concentrate 95% of per-fragment channel mass into about 11%
fewer channels without moving cosine. Next: disable the gate in the H-support configs, verify the
`required_ion_h_top_k_95`/`_99` metric that reports identical means, and recover one cluster
checkpoint so the Phase 2 selection sweep can run; W&B holds no checkpoint artifacts for these
runs. Documentation only; no training, evaluation, or code change.

Joint selection objective (2026-09-13): the
[ion/H channel plan](plans/masshatter_ion_h_channel_selection_plan.md#joint-fragmentchannel-selection-loss-experiment-2026-09-13)
now records a proposed single selection loss that couples fragment proposal and
ion/H support probabilities while retaining the independent spectral intensity loss.
The current JSS + proposal loss + support BCE configuration is the control. Next:
check joint-loss gradients and the all-channels-open failure mode on synthetic cases,
then compare on matched validation support and intensity metrics. Documentation only;
no implementation, config change, or training run.

MASShatter SMARTS review (2026-09-13): the [set/literature review](findings/masshatter_smarts_review_2026_09_13.md)
confirms 26 rules, 19 legacy rules without enforced ionization guards, and a pyrimidine
gap in the carbamate family. Positive-ion oxygen-transfer and prazole-sulfone candidates
are supported literature leads, not measured coverage gains. Validation: 75 SMARTS tests
passed on CPU plus registry/matching probes. Next: measure existing rules' incremental
coverage on filtered validation peaks before adding narrowly mapped rules. No model edits
or training ran.

MASShatter proposal review (2026-09-13): the [proposal review](findings/masshatter_proposal_review_2026_09_13.md)
prioritizes support diagnostics before encoder expansion and separates established loss/routing
limitations from exploration, diversity, and mixture hypotheses. A fresh 155-run W&B query
and eight full validation histories support selection-only in a close same-seed historical
pair: +1.54 percentage points candidate weighted recall and +0.01261 cosine at epoch 142.
A separate larger-intensity-head pair gains 0.00897 cosine at epoch 98 despite slightly
lower coverage. Recent matched JSS/KL results are close and do not establish a KL advantage.
Next for the intensity stage: preserve selection-only and diagnose fragment-versus-H ranking
on fixed support before another head or loss comparison; see the review for provenance limits.
A live W&B check of three
recent merged U256/E1024 histories confirms only 5.5–7.0% reach U256 at their latest logged
validation epochs. Two existing unmerged pairs match in logged configuration except E and
labels: E4096 adds about 43–46 fragments but only 0.66–0.67 percentage points of candidate
weighted recall at common epochs. A subformula-filter query found 14 runs, all U256/E1024;
late-run coverage is about 94%, with no filtered E4096 counterpart found. No new tests or
experiments ran. Next: inspect remaining
missed intensity and support-oracle headroom before proposing another budget run. The proposal review
is present but untracked, as is its linked loss review; include both in the eventual documentation
change set with these indexes. No files were staged or committed.

EBM modeling proposal (2026-09-13): the user selected both approaches, with
[bond-cut EBM modeling first](plans/masshatter_bond_cut_ebm_plan.md), then spectrum reranking.
The [MASShatter program](plans/masshatter_energy_based_model_plan.md) records this ordering and
the spectrum setup delta. Next: select a checkpoint and validation subset, measure missed-intensity
and finite-mask-pool headroom, then evaluate pairwise cut energies against search-only, unary,
random, and mixture controls. Preserve probability contracts; the existing exact heap guarantee
does not extend to pairwise energies. No implementation or experiments started for this program.
The spectrum stage retains its separate shared protocol and null-adjusted gate.

Loss setup review (2026-09-13): the [loss review](findings/loss_setup_review_2026_09_13.md)
reproduces new FraGNNet vectorized-JSS correspondence/backward defects, ignored OOS
supervision under JSS, and invalid formula slots stealing support targets. MASShatter
probes identify weak JSS recovery gradients, correspondence/support ambiguity, proposal-loss
coverage limits, and a no-warmup gradient-routing confound. No implementation changes or
training ran; 195 focused tests passed on CPU. Next: fix the verified FraGNNet contracts,
then inspect MASShatter gradients/coverage before an isolated matched JSS/KL comparison.
The prior audit's three FraGNNet loss/explanation fixes and MASShatter precursor-pair
rejection are already present in the reviewed source; their earlier pending status below
is historical. The W&B screen report is missing from this checkout; its historical summary below
is not reverified evidence.

Correctness audit (2026-09-12): the [combined model audit](findings/fragnnet_masshatter_audit_2026_09_12.md)
records reproduced checkpoint-loading, HDF5 cutoff, FraGNNet loss/explanation, and shared
test-metric defects, plus silently ignored MASShatter precursor-pair supervision.
No fixes were applied. The existing local multi-cut extension is
stale; an isolated rebuild from current source yielded 2,876 passed, 21 skipped, and
2 expected failures for the full suite. Next: address checkpoint loading and test-metric
initialization, then the remaining counterexamples. Pre-existing SMARTS warmup edits
were preserved; CUDA and real-data scientific behavior remain unverified.

W&B review (2026-09-11; historical summary, source report unavailable here): the recorded screen
superseded older “no runs launched” statements. `docs/findings/wandb_screen_2026_09_11.md`
is absent from this checkout; recover the report before using these comparisons as evidence.
JSS warmup 0–4 leads the single-seed filtered screen at 0.80099 best validation binned
cosine; CE remains close and finished sooner. Support supervision helps early training;
an additional soft-gate gain is unresolved. GNN context completed, but its matched
heads-only comparison is missing from the inspected runs. SMARTS warmup failed at the
known construction guard. Next: matched-seed confirmation and the missing controls;
recover the missing report for metric definitions, run links, and provenance limitations.

SMARTS merged readiness (2026-09-09): the [readiness audit](findings/masshatter_code_audit.md#smarts-merged-run-readiness-2026-09-09)
passed 494 CPU tests (5 skipped) and a synthetic SMARTS + merged NCE + GNN-context
forward/backward probe. The current merged configs are blocked by their proposal-only
warmup, which explicitly rejects SMARTS. The requested no-warmup SMARTS config is now
prepared and linked in the audit; full model construction and intended-only resolved config
differences passed. Next: perform a real-data GPU smoke check. No runs launched.
The subsequently requested [SMARTS warmup config](../config/masshatter/merged/smarts/nist20_merged_full_metadata_context_gnn_heads_smarts_warmup_e0_15.yml)
is also prepared, with proposal-only epochs 0–15. The SMARTS/proposal-only-warmup
construction guard is now removed and validated (2026-09-11): see the
[guard-removal finding](findings/masshatter_code_audit.md#smarts-proposal-only-warmup-guard-removed-2026-09-11)
for the new PL-level gradient-isolation regression tests. Construction now succeeds.
Next: a real-data GPU smoke check before launching this arm.

Context-conditioning screen (2026-09-08): the user selected controlled experiments before a new
layer. The [conditioning plan](plans/conditioning_refactor_plan.md#current-controlled-screen-2026-09-08)
now selects the full-metadata, dual-CE MASShatter mask1024 config as the conditioning baseline:
the user reports that NCE + ACE + adduct + instrument + mode already helps in existing runs.
The prepared NCE-only variant is an optional diagnostic outside the next run queue. Two new
FraGNNet configs complete the existing full-metadata routing pair with output scalers disabled.
Follow-up: the user requested context at the merged MASShatter GNN input. The optional
`mass_shatter_context_in_gnn` path is now implemented, default off with legacy-checkpoint
migration. A matched full-metadata merged heads-only versus GNN+heads pair is prepared;
existing experiment configs are unchanged. Validation: 182 passed, 5 skipped, both merged
configs constructed, intended-only resolved diffs and Ruff passed. No runs launched.
The [pre-run design audit](findings/masshatter_context_design_audit.md) now confirms homogeneous
groups, but all selected rows are `[M+H]+`/FT/HCD. It found ACE saturation at 100 eV in 21.42% of
selected training groups and row-order-dependent ACE selection for same-NCE replicates. Actual
GINE-EFA forward/backward checks at epochs 15/16 were finite; no quality evaluation ran.
CE fixes are now implemented: the new pair uses ACE up to 300 eV and order-invariant weighted
CE aggregation, with both models consuming the weights and legacy behavior preserved. The
[audit follow-up](findings/masshatter_context_design_audit.md) records regression and source-data
checks. GNN initialization remains unchanged. Next: a real-data GPU smoke check before training.

FraGNNet formula-support crash follow-up (2026-09-07): W&B runs
[kvrlaxof](https://wandb.ai/frag-gnn/frag-gnn/runs/kvrlaxof) and
[71qtthvw](https://wandb.ai/frag-gnn/frag-gnn/runs/71qtthvw) both failed with infinite
loss on batch 0 before backward. Local reproduction shows that positive support targets for
unreferenced formula slots can overflow the binary loss: their logits retain the float32
minimum sentinel. The working-tree fix in `fragnnet_model.py` derives support validity from
valid joint-state contributors and zeros invalid support logits. Tensor shapes, parameters,
configs, and spectral intensity computation are unchanged. Regression coverage verifies both
attribution methods, finite gradients, max pooling, and empty support. Validation: 40 focused
tests passed on CPU; Ruff and whitespace checks passed. No full training or validation-set
evaluation ran. Next check: run a short cluster training smoke test with the patched package,
checking individual loss terms if failure persists. The exact failing remote batch was not
replayed, so this is a reproduced defect and candidate crash fix, not a confirmed remote cure.

For the next requested implementation or experiment, verify the relevant plan against code and
available artifacts. Link that task here when it spans sessions, and record its next check and
blockers in the plan. Do not duplicate the roadmap or copy its full backlog here.

## Local context caveat

The ignored `.claude/` entry point and six rule files now link to the shared guidance.
These local pointer changes are not included in Git diffs. It still contains a legacy architecture inventory;
its state directory had no files during this audit. Shared records above are the portable
entry points. Local permissions, commands, and skills were not migrated or audited for runtime
behavior. Their presence is not evidence that another checkout has them.

## Cluster submission knowledge

The [Fir/Vulcan submission guide](guides/submit_cluster_runs.md) now provides the canonical
procedure linked from INSTALL.md. Commands and profiles were checked against repository source;
no cluster connection, submission, or training validation was performed.
