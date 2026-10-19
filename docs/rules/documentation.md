# Documentation rules

> **Status:** REFERENCE · **Updated:** 2026-10-18
> Placement and maintenance conventions for project documentation. The 2026-10-06 consolidation
> added workstream hubs, the project-state size cap, and `docs/history/`.

- Put code documentation in docstrings and config documentation in YAML comments under `config/`.
- Create README files only at the repository root or `docs/README.md`, unless explicitly
  requested elsewhere. Preserve existing README files. Do not put Markdown alongside source code
  or under `src/`.
- Organize project documents by purpose:

  | Location | Purpose |
  | --- | --- |
  | `docs/workstreams/` | One hub per workstream: where it stands, open threads, settled conclusions, and every plan and finding in it |
  | `docs/guides/` | How to run a specific operation |
  | `docs/reference/` | Detailed current system behavior |
  | `docs/findings/` | Measurements, evidence, and conclusions: FraGNNet/DAG findings in `fragnnet/`, MASSter findings in `masster/`, cross-model and shared-data findings at top level |
  | `docs/plans/` | Proposed, active, completed, or stopped work: FraGNNet/DAG plans in `fragnnet/`, MASSter plans in `masster/`, cross-model plans at top level |
  | `docs/results/` | One results ledger per benchmark (`msgym.md`, `nist20.md`, …): every measured result as a table row linking its source |
  | `docs/history/` | Frozen verbatim logs moved out of live documents, and `retired_docs.md` (summaries of removed documents); never the canonical home of a current fact |
  | `docs/notes/` | Scratch and literature notes |
  | `docs/figures/` | Images and drawing sources (PNG, SVG, draw.io) referenced by documentation; no CSVs |
  | `docs/tables/<doc stem>/` | Small summary CSVs cited by one finding or plan; raw per-query outputs and logs stay in `output/` |
  | `docs/rules/` | Shared constraints |
  | `docs/workflow/` | Shared development and experiment procedures |

- Keep the architecture map, current-state entry point, and documentation index at `docs/`
  top level. The curated `findings/fragnnet/missing_peak_findings.md` and
  `findings/fragnnet/missing_peak_findings_m-h.md` analyses are append-only. Use their `/analyze-subclass` and
  `/analyze-subclass-mh` procedures when available; if unavailable, read the procedure and
  resolve how to preserve its append-only contract before changing the logs.

## Where each kind of fact goes

- **`docs/project-state.md` is a dashboard, capped at about 200 lines.** It holds one block per
  workstream (status, best result, prepared or uncommitted work, next action, hub link) and a
  list of uncommitted working-tree changes. Update a workstream by **replacing** its block, never
  by prepending a dated entry. Dated narrative, numbers, run IDs and decisions go in the plan or
  finding; project-state only points at them.
- **A workstream hub** (`docs/workstreams/<name>.md`) is a map, not a copy: its "Where it stands"
  and "Open threads" stay current, its "Settled" list gives one-line verdicts with links, and its
  "All documents" table lists every plan and finding in the workstream. A new workstream gets a
  new hub and a row in `docs/README.md` and project-state.
- **A plan** carries its own dated progress. When a plan passes about 600 lines, move its dated
  log sections verbatim to `docs/history/<plan-stem>_log.md` and keep a `## Current summary` in
  the plan ([procedure](../workflow/history_and_summary.md#rotate-a-plan-log)).
- **Findings** hold measurements and conclusions; one canonical location per fact. Link to it
  instead of restating numbers elsewhere.
- **A results ledger** (`docs/results/<benchmark>.md`) is the one exception: it copies each
  measured result's headline numbers verbatim into a table row, with run@epoch, split, setup and
  a link to the finding or plan section that stays canonical. Rows are appended in date order
  and never deleted; a corrected source means a corrected row, and a retracted result keeps its
  row marked as retracted. Rows without a measurement (prepared runs) do not belong there.
- **Paths in frozen records** (findings, plans, `docs/history/`) are as of `13b1599d`, before
  the reorganization moved FraGNNet, MASSter and the baselines into `src/fragnnet/models/` and
  `src/fragnnet/baselines/`; use git history to resolve them. Repoint links only in live
  documents (the architecture map, README files, `docs/reference/`, `docs/guides/`, hubs, notes).

## Status banners

Every project document needs a status banner directly below its H1:
`> **Status:** <STATUS> · **Updated:** YYYY-MM-DD`, then a one-line current state. Tool entry
points and thin rule pointers need not repeat it. Update the banner when a conclusion or task
status changes, and use only these statuses:

| Status | Meaning |
|--------|---------|
| `HUB` | Workstream map in `docs/workstreams/`. |
| `LEDGER` | Results table in `docs/results/`; numbers copied from linked sources. |
| `ACTIVE` | Current: someone is working on it and something happens next. |
| `PREPARED` | Configs or code ready; not submitted or not run. |
| `PROPOSED` | Written up, not started. |
| `DEFERRED` | Not pursued now; still open. Mark an `ACTIVE` plan idle for over a week `DEFERRED`. |
| `LANDED` | Shipped; the doc now describes production behavior. |
| `FINDING` | A measurement and its conclusion. |
| `CLOSED NULL` | Measured, no headroom. **Read before re-proposing the same mechanism.** |
| `STOPPED` / `SUPERSEDED` | Do not resume without reading why it stopped or what replaced it. |
| `GUIDE` / `REFERENCE` / `NOTE` | How to run something, how the system works, or scratch/literature; verify against current code. |
| `CURATED ANALYSIS` | Long-running append-only analysis. Append via its slash command rather than hand-editing. |

Keep `CLOSED NULL` documents in place with the reason they closed; do not archive or delete
them. Mark stopped or replaced work `STOPPED` or `SUPERSEDED` with the reason.

## Removing a document

Remove a document only when the user asks. Candidates are `SUPERSEDED` and `STOPPED` documents
and audits whose fixes have all landed; an audit with an open item stays. A removed document
leaves a summary in `docs/history/retired_docs.md`, a hub Settled line, and inbound links that
name a commit still holding it. Follow the
[retirement procedure](../workflow/history_and_summary.md#retire-a-document); merging several
findings into one topic document follows the
[merge procedure](../workflow/history_and_summary.md#merge-findings-into-a-topic-document).

## Adding, moving, or renaming a document

Add or update its row in its workstream hub (or, for guides, reference, notes, rules and hubs,
in `docs/README.md`). Search for affected paths in `src/`, `tests/`, `scripts/`, `config/`,
`slurm_scripts/`, and `INSTALL.md`, including `.yaml`, `.yml`, `.sh`, and `.pyx` files.
Check relative links from the document's actual directory; links from a subdirectory
to the repository root typically need `../../`, and to figures need `../figures/`.
