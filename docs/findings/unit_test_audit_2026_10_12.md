# Unit test audit

> **Status:** FINDING · **Updated:** 2026-10-12
> Bounded repository test audit; all four findings resolved (export expectations and three false-positive guards). Full-suite and GPU validation remain open.

## Scope and validation

Reviewed discovery, fixtures, skip policies, configuration regression guards, preprocessing
assertions, export contracts and selected numerical/model tests. This is not an exhaustive
review of every test. Used `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`.

- `python -m pytest tests -q` collected 5,802 cases. Interrupted after approximately 15%
  progress; the complete suite result is unknown.
- `python -m pytest tests/test_export_schema.py tests/test_01_inf_prepare_mol_df.py
  tests/test_rearrangement_smarts.py tests/test_loss_numerical_contracts.py -q -o addopts=''`:
  **136 passed, 8 failed, 1 skipped**. All failures are export-schema expectations;
  the skip requires CUDA. CUDA was unavailable.
- A concurrent edit added `pred_rt` and `pred_ccs` to the export expectations. Rerun of
  `python -m pytest tests/test_export_schema.py -q -o addopts=''`: **29 passed, 1 skipped**.
- In-memory fault injection confirmed both filter assertions accept an empty DataFrame,
  and the SMARTS negative helper accepts a matching substrate when sanitization raises.
  No production files or golden records were altered for these probes.
- Initial Git status was clean. Project state still describes an in-progress isotope
  file move and some hubs describe committed changes as uncommitted; those statements
  do not describe the inspected working tree.

## Findings, ordered by priority

### 1. Resolved during audit: Export schema and frozen expectations disagreed

`tests/test_export_schema.py:26` defines `BASE_COLUMNS` without `pred_rt` and `pred_ccs`.
`src/fragnnet/inference/export_columns.py` inserts those columns immediately after
`pred_oos_prob`. Eight assertions fail, including the frozen column order and the
unrequested-column contract. This discrepancy is present in the clean checkout; it is
not contingent on the uncommitted export work mentioned in the loss consolidation plan.

Another session added the two columns to `BASE_COLUMNS` during this audit; all CPU export
schema tests now pass. This audit did not make that edit. The original failure remains useful
as evidence that the frozen expectations needed maintenance.

### 2. Resolved: Configuration regression guard checked only a failure count

`tests/test_config_templates.py:274` collects rejected paths and messages but ultimately
asserts only `len(failures) == 10`. A previously invalid config becoming accepted can hide a
previously valid config becoming rejected. Malformed YAML is also silently excluded.

Pin the exact known rejected paths and their expected diagnostic categories. Make any new
malformed config visible through an explicit known-invalid list or a separate YAML check.
Do not change the count alone to accommodate failures.

### 3. Resolved: Filter assertions passed when all molecules were discarded

`tests/test_01_inf_prepare_mol_df.py:75` and `:85` assert only that every retained molecule
has zero charge/radicals. Both pass on an empty result because `all([])` is true. Confirmed
by replacing `create_mol_df` in memory with an empty DataFrame carrying the two columns.

Assert the two expected neutral molecules survive and the charged/radical molecule is absent,
including retained identity rather than only a row count.

### 4. Resolved: SMARTS negative checks conflated nonmatches and invalid products

`tests/test_rearrangement_smarts.py:88` catches every sanitization/conversion exception;
`:96` interprets an empty sanitized set as a nonmatch. A SMARTS rule can match a forbidden
substrate and generate invalid products while its negative test still passes. Fault injection
on a known matching McLafferty substrate reproduced this behavior.

Check raw `RunReactants` results for true nonmatches and assert sanitization separately for
positive reactions. If invalid-product rejection is intended, give it a distinct assertion.
These constants are local research examples, not imports of production rearrangement rules;
their passing results do not establish production rule coverage.

## Resolution (2026-10-12)

- **2:** `tests/test_config_templates.py` pins `KNOWN_REJECTED_CONFIGS` (the 10 paths) and
  `KNOWN_MALFORMED_CONFIGS` (3 YAML files that do not parse), compares sets rather than counts,
  and requires each rejection to be an undeclared-key error.
- **3:** both filter tests assert the surviving SMILES are exactly `["CCO", "CC(=O)O"]`.
- **4:** `_no_match` now checks raw `RunReactants` output, so a match with unsanitizable
  products fails the negative test. All 12 negative cases produce zero raw products.
- The postcut-gate tests had a separate stale expectation: N-methylbenzenesulfonamide also
  matches `substituted_aryl_sulfonamide_o_transfer`, so the fixture now keeps only the
  `aryl_sulfonamide_so2` sides and the builder test pins both rule names.

Validation: those three files plus `test_export_schema.py` and
`test_debug_masster_postcut_information_gate.py`, **116 passed** (CPU, `-o addopts=''`).

## Handoff

This audit changed documentation only. Another session changed `tests/test_export_schema.py`
and `tests/test_debug_masster_postcut_information_gate.py` while it was running; those edits
were preserved. The guards were strengthened afterwards (see Resolution). Complete-suite and GPU
validation remain open. Documentation links and `git diff --check` passed.
