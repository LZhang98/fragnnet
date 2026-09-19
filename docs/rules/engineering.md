# Engineering and research rules

> **Status:** REFERENCE · **Updated:** 2026-09-19

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Plan Mode Questions

- In Plan mode, do not put a time limit on the user's answer to a blocking question.
  Ask the question and wait for the user's response before proceeding.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### Step-by-Step Labels for Complex Processes

When code implements a complicated multi-stage process:

- Add numbered comments for the major phases, e.g. `# 1. Load data.`, `# 2. Build grid.`, `# 3. Write artifacts.`
- Keep the process flat and readable; avoid nested helper functions unless they clearly reduce complexity.
- Comment the workflow phases and non-obvious choices, not every obvious line.
- Use local numbering inside each function or workflow. Do not force a global numbering scheme across files.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Environment

- Use the `FRAGNNET-GPU` conda environment configured in `.vscode/settings.json`.
- Prefer the configured interpreter directly:
  `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`.
- Do not use system Python, assume an environment is active, install packages globally,
  or modify the conda environment without user confirmation.

- If the workspace interpreter setting is absent, ask which environment to use.
- For cluster work, inspect the actual environment and submission scripts before using them;
  do not assume a named setup script exists or that a job activates the correct environment.

- Narval compute nodes have no outbound internet. Generate Narval jobs with `w=offline`
  (`--script_kwargs w=offline`), then sync from a login node with
  `wandb sync wandb/offline-run-*`; the job symlinks `wandb/` back to the cluster checkout, so
  offline runs survive the job. `wandb agent` sweeps cannot run on Narval for the same reason.
  The venv build is unaffected: it installs with `pip install --no-index` from the local
  wheelhouse. Fir and Vulcan compute nodes do have internet, so `w=online` is correct there.


## 6. Compatibility and validation


**Do not change without explicit authorization and corresponding compatibility work:**

- Model input/output shapes without updating downstream consumers
- Config schemas without a compatibility or deprecation path
- Existing preprocessing behavior without addressing experiment reproducibility

**When applicable:**

- Save checkpoints with seed + config info
- Add tests for new architectures
- Update `config/` when adding hyperparameters
- Include docstrings for mathematical operations
- Run focused tests for changed behavior. Run the full test suite and validation before a PR
  when practical; report any checks that were not run.

## 7. Git Workflow

**Leave repository history and branch state to the user.**

- Do not run `git commit`, `git push`, `git merge`, `git rebase`, or `git reset --hard`.
- Do not switch branches or use `git stash`.
- Do not stage files with `git add` unless the user explicitly asks.
- Read-only inspection such as `git status`, `git diff`, `git log`, `git show`,
  `git blame`, and `git branch --list` is encouraged.
- Leave changes in the working tree and summarize what changed and what verification ran.

- Requests to "save" work mean write files, not commit them. Passing checks does not authorize
  staging or committing. Git revisions are experiment provenance and remain under user control.


## Task-specific rules

Read [Python and ML rules](python.md) for Python changes and
[documentation rules](documentation.md) when creating or editing project documentation.
