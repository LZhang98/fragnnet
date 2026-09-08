# Current project state

> **Status:** REFERENCE · **Updated:** 2026-09-07

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

## Research navigation

- [MASShatter improvement plan](plans/masshatter_improvement_plan.md) records the proposed
  unmerged sequence: corrected CE control, proposal-encoder comparison, Gaussian support-gated
  JSS, conservative output pruning, and measured throughput improvements. Five merged warmup
  configs are prepared for a CE/JSS/KL loss and intensity-transform screen. Next: verify checkpoint
  persistence before launching; resolve the unmerged anchor/warmup policy before its matched
  control. No runs were launched for this plan.
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
