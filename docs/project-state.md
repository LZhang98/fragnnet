# Current project state

> **Status:** REFERENCE · **Updated:** 2026-09-15

## Current handoff

The shared agent workflow is established on this branch: start at [AGENTS.md](../AGENTS.md),
follow the [session workflow](workflow/session.md), and use the [architecture map](architecture.md).
Rules are split into [engineering](rules/engineering.md), [Python and ML](rules/python.md), and
[documentation](rules/documentation.md); the agent entry point selects the applicable files by task.

These scaffolding files were ported from `research/masshatter-ion-h-selection`, which authored
them. Branch-specific content was not carried over: the cluster guide's launcher reference and the
documentation rules' curated-analysis paths were corrected against this checkout, and this
project-state file was written fresh rather than copied. The adoption changes are uncommitted and
left in the working tree for user review.

Validation performed for the adoption: every relative link in the ported documents resolves in this
checkout, and the GPU profiles, account choices, and wheel archives cited by the cluster guide were
checked against [setup_job_cc.py](../scripts/setup_job_cc.py). No model code changed; no training,
scientific evaluation, or test suite was run for this documentation change.

`docs/missing_peak_findings.md` and `docs/missing_peak_findings_m-h.md` moved to
[findings/](findings/), matching the source branch. The `/analyze-subclass` and
`/analyze-subclass-mh` command definitions and the path comment in
[smarts_prepass.py](../src/fragnnet/frag/smarts_prepass.py) were updated to the new locations.
These remain append-only curated analyses.

## Research navigation

- [Model improvement roadmap](plans/model_improvement_plan.md) is the canonical source for
  research priorities and acceptance criteria. Its banner is dated 2026-07-24; reconcile its
  proposed next action against current code and artifacts before resuming it.
- [Documentation index](README.md) lists live, proposed, landed, and closed work. Those are
  recorded statuses, not verification of current execution.
- Read the linked [findings](findings/) before reviving a rejected or closed experiment. Several
  plans here are closed nulls and are kept so the same line is not re-run.

## Next task

No task was carried across from the source branch, and no training process, scheduler job,
checkpoint, or external run status was audited during this adoption.

Recent branch work concerns subformula filtering and CMF candidate enumeration
(`d0393776`, `4977310a`, `2922ef7e`); see [subformula_filter_ft_qtof.md](findings/subformula_filter_ft_qtof.md)
for the measured context. Confirm the intended next experiment with the user, verify its plan
against code and available artifacts, and link that task here when it spans sessions, recording
its next check and blockers in the plan. Do not duplicate the roadmap or copy its backlog here.

## Local context caveat

The ignored `.claude/` entry point and `AGENTS.md` are not included in Git diffs on this branch
(`.gitignore:64,66`), and `.claude/` is one working tree shared by every branch in this checkout.
The `docs/` records above are the portable entry points. Local permissions, commands, and skills
were not audited for runtime behavior; their presence here is not evidence that another checkout
has them.
