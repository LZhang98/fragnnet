# Development and experiment workflow

> **Status:** GUIDE · **Updated:** 2026-10-06

Adapted from the [Agentic Research Workflow](https://github.com/7FeiW/Incomplete_guide/blob/main/docs/15_Agentic_Workflow.md).
Read the [shared rules](../rules/engineering.md) for engineering, environment, compatibility,
and Git constraints. Existing [guides](../guides/) remain the runbooks for specific operations.

## 1. Orient and define

Read [project state](../project-state.md), the relevant [workstream hub](../workstreams/), its
plans and findings, and current code.
Inspect `git status --short` and the relevant `git diff`. Treat local memories and transcripts
as hints. Report stale assumptions; distinguish evidence, hypotheses, and unresolved questions.
State the objective, scope, compatibility constraints, and observable acceptance checks.
Resolve ambiguous scientific choices before implementing dependent behavior.

## 2. Plan and implement

Use a brief plan for multi-step work. For work spanning sessions, update the existing relevant
file under `docs/plans/`, or create a bounded task record and link it from its workstream hub.
Use these fields (adapt existing plans without wholesale reformatting):

- Status and update date.
- Objective and acceptance criteria.
- Established facts with source or artifact links.
- Plan and constraints.
- Completed work and current implementation/execution state.
- Validation evidence, unresolved questions, and next concrete check.

Make surgical changes and check each meaningful increment. Keep scientific changes, dependency
changes, and broad refactors separate. A plan or example command does not itself authorize a
training run, scheduler submission, dataset modification, or external publication; follow the
user's task scope and the shared rules. Keep instructions distinct from enforced sandbox,
operating-system, and CI controls.

## 3. Validate

Run from the repository root using the configured interpreter. On this workstation:

```bash
FRAGNNET_PYTHON=/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python
"$FRAGNNET_PYTHON" -m pytest tests/<relevant_test_file>.py -q
"$FRAGNNET_PYTHON" -m ruff check <changed_python_files>
# Before a PR, when practical:
"$FRAGNNET_PYTHON" -m pytest tests -q
```

Replace angle-bracket placeholders with actual paths. On another workstation resolve the
`FRAGNNET-GPU` interpreter from its workspace configuration; do not silently use system Python.
Pytest discovery and Ruff settings live in [pytest.ini](../../pytest.ini) and
[pyproject.toml](../../pyproject.toml). Tool availability does not imply the suite passes.
Report missing dependencies or data; do not install packages without the required confirmation.
For documentation-only changes, check local links and `git diff --check`; model tests are
unnecessary unless behavior or executable examples changed.

Validate scientific meaning separately: split leakage, dataset identity, units, adducts,
collision energies, mass tolerance, metric definitions, aggregation, and matched baselines.
Use the relevant research plan's gates rather than inventing a global promotion threshold.
Scientific interpretations remain provisional until supported and reviewed.

## 4. Record experiments when authorized

For Fir or Vulcan runs, read the [cluster submission guide](../guides/submit_cluster_runs.md)
before preparing a job.

Before a run, record in its task/experiment record:

- Git revision and dirty diff identity; environment/package identity.
- Input dataset, split, DAG/cache identity and preprocessing provenance.
- Resolved configuration, overrides, exact command, seeds, and determinism settings.
- Hardware/device and scheduler allocation, if applicable.
- Start time, run ID, and intended log/checkpoint/result paths.

Afterward record completion or failure, exit reason, actual artifact paths, metrics and their
aggregation, and unresolved checks. The runner seeds runs and supports W&B/checkpoints, but
this workflow does not assume it captures every provenance field automatically. Record missing
metadata at launch, not by reconstructing it from memory afterward. Keep large artifacts in
their documented stores; include accessible identities and links in the record.

## 5. Update state and hand off

At meaningful checkpoints, update the plan with observed results. Put supported durable
conclusions and negative results in `docs/findings/`; preserve evidence and uncertainty.
Append each new measurement as a row to its benchmark ledger in `docs/results/`.
Update the workstream hub's "Where it stands" and "Open threads", then **replace** that
workstream's block in [project state](../project-state.md) (no dated entries there; it stays
under about 200 lines). Keep useful closed plans and their status banners. Do not copy transcripts into docs.
The [history and summary workflow](history_and_summary.md) gives the handoff checklist and the
procedures for rotating plan logs, compacting project state, merging and retiring documents.

Report what changed, checks actually run and their outcomes, remaining limitations, and artifact
locations. Leave Git staging, history, and branch state to the user under the shared rules.

## Maintenance

Keep one canonical location per fact. Preserve the existing `guides/`, `reference/`, `findings/`,
and `plans/` layout. Check status and evidence before following older documents; a GUIDE banner
alone is not verification. Reusable procedures belong here or in `guides/`; add tool-specific
skill wrappers only when a concrete recurring task needs them. Periodically check stale plans,
links, commands, and local guidance for drift ([sweep](history_and_summary.md#sweep)). Runtime permissions and hooks need a separate
review of their actual configuration, not assumptions based on prose.
