# History and summary workflow

> **Status:** GUIDE · **Updated:** 2026-10-18
> How to move dated material into `docs/history/`, merge or retire documents, and keep the summaries that replace them current.

The [documentation rules](../rules/documentation.md) say where each kind of fact belongs. This
guide gives the procedures that keep live documents short without losing the record. They
codify the operations already used: plan logs (2026-10-06, 2026-10-12), the project-state
handoff log (2026-10-06), and the finding merges and retirements (2026-10-18).

Two invariants hold for every procedure:

- **History is verbatim and frozen.** Text moved to `docs/history/` keeps its wording and
  heading text, so anchors still resolve by file-name substitution. Only relative links are
  rebased. Nothing in `docs/history/` is edited afterwards, except that a plan log receives
  later rotations from its own plan.
- **The summary stands alone.** A reader of the live document can act without opening the
  history: verdicts, the headline number with a link to where it is measured, what is open, and
  the next check. Numbers stay in their canonical finding or log; a summary links to them.

## When to run which procedure

| Trigger | Procedure | Needs the user? |
| --- | --- | --- |
| A plan passes about 600 lines, or dated logs outweigh its current content | [Rotate a plan log](#rotate-a-plan-log) | No |
| End of a session, or a conclusion or task status changes | [Summarize at handoff](#summarize-at-handoff) | No |
| `docs/project-state.md` passes about 200 lines | [Compact project state](#compact-project-state) | No |
| Several findings answer one question, or a topic is split across dated files | [Merge findings into a topic document](#merge-findings-into-a-topic-document) | Yes: it renames files |
| A `SUPERSEDED`, `STOPPED` or `CLOSED NULL` document, or an audit with every fix landed, is no longer read | [Retire a document](#retire-a-document) | Yes: only when asked |
| Starting documentation maintenance | [Sweep](#sweep) | Report; act per row above |

## Rotate a plan log

1. **Pick the sections.** Dated progress entries, measured-result subsections, review rounds and
   status-banner history move. Objective, acceptance criteria, design rationale still in force,
   open questions, and the next step stay.
2. **Create or extend `docs/history/<plan-stem>_log.md`.** A new log starts with:

   ```markdown
   # <Plan H1> — log

   > **Status:** REFERENCE · **Updated:** YYYY-MM-DD
   > Verbatim dated log moved from [the plan](../plans/<subdir>/<plan-stem>.md) on YYYY-MM-DD; the plan holds the current summary.
   ```

   Then one paragraph naming what the line references are relative to (commit or plan version)
   and that relative links were rebased. A later rotation appends below the existing sections,
   under a `## Moved YYYY-MM-DD` note line, and updates the banner date.
3. **Move the text verbatim.** Keep each heading's text identical. Rebase relative links from the
   plan's directory to `docs/history/` (from `docs/plans/masster/`, `../../` becomes `../`).
   An anchor that stays in the plan points back to the plan.
4. **Write `## Current summary`** directly under the plan's banner or objective: what was
   decided, the best measured result with a link to the log anchor or finding that holds it,
   closed branches as one line each, open questions, and the next concrete check. Replace an
   existing summary; do not append to it.
5. **Update the banner:** date, a one-line current state, and `Dated log: [history](...)`.
6. **Repoint inbound anchors** in live documents only (hubs, `docs/README.md`, project state,
   `docs/reference/`, `docs/guides/`, notes, other live plans, `config/`, `src/`, `scripts/`,
   `tests/`). Find them with `grep -rn '<plan-stem>.md#' docs config src scripts tests`.
   Frozen records keep their links; the moved heading resolves in `<plan-stem>_log.md`.
7. **Link the log from the hub:** in the plan's "All documents" row or under "Key artifacts".
8. **Validate** with the [checks below](#checks).

## Summarize at handoff

This is the summary half of [session workflow step 5](session.md#5-update-state-and-hand-off).

1. **Plan or finding first.** Record the dated result, run IDs and numbers where they belong; set
   the banner status and date. If the plan now passes about 600 lines, rotate it.
2. **Hub.** Rewrite "Where it stands" (current, not a diary) and "Open threads". When a thread
   closes, delete its row and add a one-line "Settled" verdict with a link. Keep the "All
   documents" table complete.
3. **Project state.** Replace the workstream's block: status, best result with link, prepared
   or uncommitted work, next action. Update its row in the workstream table. Never add a dated
   entry.
4. **Uncommitted work** goes in project state's uncommitted list with its paths, and is removed
   when committed.

## Compact project state

1. For each block line that is narrative rather than state, check that its fact exists in the
   plan, finding or hub it concerns; port it there first if not.
2. Delete the line from project state. Git keeps the old text; `docs/history/handoff_log.md`
   is frozen at 2026-10-06 and is not appended to.
3. Stop when the file is under about 200 lines and each block holds only the fields listed in
   [Summarize at handoff](#summarize-at-handoff).

## Merge findings into a topic document

Follow the pattern of [masster_smarts_findings.md](../findings/masster/masster_smarts_findings.md).

1. Name the topic file without a date (`<topic>.md`) in the same findings subdirectory.
2. Start it with H1, a `FINDING` banner whose line lists the parts, the sentence "Merged
   YYYY-MM-DD from N findings on one topic. Each part below is the original document unchanged
   except for heading levels; its own status line keeps its original date.", and a table:
   `| Part | Updated | Verdict | Former file |`, oldest first, each part linked by anchor.
3. Append each finding unchanged, demoting every heading one level, so anchors survive within
   the new file. Rebase relative links only if the directory changed.
4. Delete the former files. Add one row per former file to the "Merged" table in
   [retired_docs.md](../history/retired_docs.md#merged-masster-findings-2026-10-18) (start a
   new dated "Merged" section for a new batch).
5. Repoint inbound links in live documents to the topic file and anchor; replace the former
   rows in the hub's "All documents" table with one row for the topic file.
6. Validate with the [checks below](#checks).

## Retire a document

Only when the user asks ([rule](../rules/documentation.md#removing-a-document)).

1. **Record the commit** that still holds the file, before deleting it: the file must be
   committed and clean, then `git log -1 --format=%h -- <path>`.
2. **Move open consequences** (data to rebuild, validation not yet run) into the hub's "Open
   threads" or project state.
3. **Write the summary** in [retired_docs.md](../history/retired_docs.md) under the workstream
   heading, in this form:

   ```markdown
   - **<Title>** (<plan|finding>, <STATUS>, <banner date>) — `git show <commit>:<path>`
     <Question; verdict with key numbers and run IDs; what not to re-propose and why.>
     Open when retired: <open items, or omit the line if none>.
   ```

   A `CLOSED NULL` summary must keep the mechanism, the null result and its numbers: it is the
   in-tree reason not to re-propose it.
4. **Update the hub:** delete the "All documents" row; keep any "Settled" line, rewriting its
   link as `<text> (removed YYYY-MM-DD; git show <commit>:<path>)`; turn an "Open threads" row
   into a Settled line saying it was retired and the next step if revived. Keep or update the
   hub's "Retired YYYY-MM-DD: N documents" line.
5. **Replace inbound links** in live documents, `config/`, `src/`, `scripts/` and `tests/` with
   the text followed by (removed YYYY-MM-DD; `git show <commit>:<path>`); a bare path becomes
   `<commit>:<path>, removed YYYY-MM-DD`. Leave `docs/history/` unchanged.
6. Delete the file and validate with the [checks below](#checks).

## Sweep

Report candidates to the user; act on each per the [trigger table](#when-to-run-which-procedure).

```bash
# Plans over the rotation threshold
wc -l docs/plans/*.md docs/plans/*/*.md | sort -n | awk '$1 > 600 && $2 != "total"'
# ACTIVE documents not updated in over a week (mark DEFERRED)
grep -rE --include=*.md --exclude-dir=history 'Status:\*\* ACTIVE · \*\*Updated:\*\* [0-9-]+' docs \
  | sed -E 's/:.*Updated:\*\* ([0-9-]+).*/ \1/' | awk -v c="$(date -d '7 days ago' +%F)" '$2 < c'
# SUPERSEDED / STOPPED documents still in the tree (retirement candidates)
grep -rlE 'Status:\*\* (SUPERSEDED|STOPPED)' docs --include=*.md
```

Also check that every file under `docs/plans/` and `docs/findings/` appears in its hub's "All
documents" table.

## Checks

Run from the repository root after any procedure:

```bash
git diff --check
# Broken relative links in changed Markdown files (live documents and docs/history/)
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python - $(git diff --name-only -- docs; git ls-files -o --exclude-standard docs) <<'EOF'
import re, sys, pathlib
for name in sys.argv[1:]:
    p = pathlib.Path(name)
    if p.suffix != ".md" or not p.exists():
        continue
    text = re.sub(r"```.*?```|`[^`\n]*`", "", p.read_text(), flags=re.S)
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        path = target.split("#")[0]
        if path and "://" not in path and not (p.parent / path).exists():
            print(f"{p}: {target}")
EOF
```

For a rotation or merge, also confirm the move was verbatim: the lines removed from the source
(`git diff -U0 -- <source> | sed -n 's/^-//p'`) should match the lines added to the target,
apart from rebased links and demoted headings.
