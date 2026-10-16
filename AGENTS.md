# Agent entry point

FraGNNet predicts tandem mass spectra from molecular structures; this repository also
contains MASSter and baseline models.

Before planning or editing:

1. Always read and follow [engineering rules](docs/rules/engineering.md).
   For Python changes, also read [Python and ML rules](docs/rules/python.md).
   For documentation changes, also read [documentation rules](docs/rules/documentation.md).
   For formal academic writing (such as theses and manuscripts), also read
   [formal writing rules](docs/writing_rule.md). These writing rules do not apply to
   plans, task records, progress updates, or routine project documentation.
   Before writing or citing checkpoints or run outputs, read [artifact storage rules](docs/rules/artifacts.md).
2. Read [current project state](docs/project-state.md), then the relevant
   [workstream hub](docs/workstreams/) and the plans it links.
3. Read the [architecture map](docs/architecture.md) and relevant detailed references.
4. Follow the [session workflow](docs/workflow/session.md), including validation and handoff.
5. Inspect current Git status and relevant code; report conflicts with recorded state.

Use the `FRAGNNET-GPU` interpreter configured in `.vscode/settings.json`.
Environment constraints are in the shared rules; validation commands are in the session workflow.
The [documentation index](docs/README.md) links research findings, guides, and historical plans.
Keep shared knowledge and task state in `docs/`; link to canonical facts instead of duplicating them.
