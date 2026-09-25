# Documentation rules

> **Status:** REFERENCE · **Updated:** 2026-09-23
> Placement and maintenance conventions for project documentation.

- Put code documentation in docstrings and config documentation in YAML comments under `config/`.
- Create README files only at the repository root or `docs/README.md`, unless explicitly
  requested elsewhere. Preserve existing README files. Do not put Markdown alongside source code
  or under `src/`.
- Organize project documents by purpose:

  | Location | Purpose |
  | --- | --- |
  | `docs/guides/` | How to run a specific operation |
  | `docs/reference/` | Detailed current system behavior |
  | `docs/findings/` | Measurements, evidence, and conclusions: FraGNNet/DAG findings in `fragnnet/`, MASSter findings in `masster/`, cross-model and shared-data findings at top level |
  | `docs/plans/` | Proposed, active, completed, or stopped work: FraGNNet/DAG plans in `fragnnet/`, MASSter plans in `masster/`, cross-model plans at top level |
  | `docs/notes/` | Scratch and literature notes |
  | `docs/figures/` | Figures referenced by documentation |
  | `docs/rules/` | Shared constraints |
  | `docs/workflow/` | Shared development and experiment procedures |

- Keep the architecture map, current-state entry point, and documentation index at `docs/`
  top level. The curated `findings/fragnnet/missing_peak_findings.md` and
  `findings/fragnnet/missing_peak_findings_m-h.md` analyses are append-only. Use their `/analyze-subclass` and
  `/analyze-subclass-mh` procedures when available; if unavailable, read the procedure and
  resolve how to preserve its append-only contract before changing the logs.
- Every project document needs a status banner directly below its H1. Include an update date
  and a short explanation of its current state. Tool entry points and thin rule pointers need
  not repeat project-state banners. Use the statuses in the [documentation index](../README.md),
  updating the banner when a conclusion or task status changes.
- Keep failed and negative-result plans in place with `CLOSED NULL`, `STOPPED`, or `SUPERSEDED`
  and the reason they stopped. Do not archive or delete them.
- When adding, moving, or renaming a document, add or update its row in `docs/README.md`.
  Search for affected paths in `src/`, `tests/`, `scripts/`, `config/`, `slurm_scripts/`,
  and `INSTALL.md`, including `.yaml`, `.yml`, `.sh`, and `.pyx` files.
  Check relative links from the document's actual directory; links from a subdirectory
  to the repository root typically need `../../`, and to figures need `../figures/`.
