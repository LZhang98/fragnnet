# Preprocessing script audit

> **Status:** FINDING · **Updated:** 2026-10-12
> Seven correctness findings; preprocessing code and datasets unchanged.

## Scope and evidence

Reviewed the main preparation pipeline, inference preparation, dataset merger and migration,
and selected split/retrieval utilities. Ran Ruff over all `preproc_scripts` and focused tests.
This is not a full execution audit of every source format, experimental SMARTS script, or
large-data pipeline. Synthetic checks used the configured `FRAGNNET-GPU` interpreter and
temporary directories; no production data was modified. No training or DAG build was run.

The working tree was clean at entry, contrary to the project-state note about an in-progress
isotope utility move. Recorded experiment results were not used to infer current code validity.

## Findings

1. **P1 — Dataset merger breaks molecule references and group identity.**
   [`merge_datasets.py`](../../preproc_scripts/dataset_migration/merge_datasets.py), lines
   51–54 and 81–90, deduplicates molecule rows but retains their source-local `mol_id`, while
   assigning new IDs only to spectra. Source-local `group_id` values also remain unchanged.
   A subprocess reproduction with two sources, each containing molecule 0 and group 0 but
   different InChIKey blocks, exited successfully and wrote molecule IDs `[0, 0]`, spectrum
   molecule IDs `[0, 1]`, and spectrum group IDs `[0, 0]`. Molecule 1 has no referenced row;
   group 0 spans unrelated molecules. Apply one global molecule mapping to both tables and
   rebuild groups from global molecule/adduct/instrument identity before saving.

2. **P1 — Merged inference groups span different molecules.**
   [`02_inf_prepare_spec_df.py`](../../preproc_scripts/inference/02_inf_prepare_spec_df.py),
   lines 141–150, keys merged groups only by precursor and instrument. A two-molecule,
   two-energy reproduction with `merged_spectra=True` produced one group containing both
   molecules. Downstream merged-spectrum consumers use `group_id` to combine rows. Include
   `mol_id` in the group mapping.

3. **P2 — Inline predefined numeric split IDs silently select nothing.**
   [`04_prepare_split.py`](../../preproc_scripts/04_prepare_split.py), lines 220–260 and
   354–386, casts CSV IDs to the selected type but leaves inline arguments as strings.
   Integer `spec_id`, `mol_id` and `group_id` columns do not match these values in `isin`.
   The underlying check `pd.Series([1, 2]).isin(['1'])` returned `[False, False]`.
   Normalize all four inline ID lists with the same type conversion as file inputs.

4. **P2 — The advertised `ms+meta` input format always crashes.**
   [`01_prepare_df.py`](../../preproc_scripts/01_prepare_df.py), lines 269–270 and 332,
   references `ms_fp_list` without ever initializing it from `ms_dp`. Calling
   `preproc_ms_files` with a valid minimal metadata TSV and `num_entries=-1` raised
   `UnboundLocalError`; a positive limit fails earlier at the slice. Enumerate source `.ms`
   files before applying the limit.

5. **P2 — Lowercase NCE is silently converted to absolute energy.**
   [`02_inf_prepare_spec_df.py`](../../preproc_scripts/inference/02_inf_prepare_spec_df.py),
   lines 248–257, matches units case-insensitively but compares the captured unit
   case-sensitively. `get_config(..., ft_ces=['20nce'], inst_types=['ORB'], ...)` emitted
   `ce_unit='eV'`, placing normalized energy into the absolute-energy column. Normalize
   unit case before classification and require the entire string to match.

6. **P2 — Accepted inference instruments crash or inherit another instrument's energy.**
   [`02_inf_prepare_spec_df.py`](../../preproc_scripts/inference/02_inf_prepare_spec_df.py),
   lines 263–271, assigns `energy_values` only for Orbitrap and QTOF, although the CLI also
   accepts IT, Q, TOF and QQQ and validation lets them through. `inst_types=['IT']` raised
   `UnboundLocalError` in a direct check. After a supported instrument, an unsupported one
   instead reuses the previous loop's energy list. Explicitly implement the intended unit
   policy or reject these instruments before building combinations.

7. **P2 — Disabling tautomer canonicalization does not affect the initial structure pass.**
   [`02_prepare_proc.py`](../../preproc_scripts/02_prepare_proc.py), lines 111–117,
   calls `mol_from_smiles` with its default `canonicalize_tautomers=True` in the initial
   standardization pass. Only the second conversion, after standardized SMILES and molecule
   IDs have already been created, receives the requested flag (lines 255–264).
   Thus `--canonicalize_tautomers False` cannot preserve original tautomers unless
   standardization is also skipped. This finding follows directly from the two call sites
   and the defaults in `utils/mol_utils.py`; an end-to-end paired build was not run.
   Pass the flag consistently in both conversions and preserve provenance for changed builds.

## Validation and next action

- `python -m pytest tests/test_spec_utils_preprocessing.py tests/test_clean_spectra_utils.py
  tests/test_spectraverse_benchmark_split.py tests/test_split_filters.py -q`: **214 passed**.
- `python -m ruff check preproc_scripts --output-format concise`: **85 findings**, including
  the undefined `ms_fp_list` and undefined names in two experimental SMARTS scripts. Formatting
  findings are not counted as correctness bugs above.
- Small direct-function checks and a merger subprocess reproduced findings 1, 2, 4, 5 and 6;
  the pandas membership check supports finding 3. Finding 7 is a static call-path finding.
- No fixes applied. Next: repair identity corruption first, add regression coverage for these
  failures, then address split and parser behavior. Existing processed outputs and splits
  must not be regenerated in place as part of a code fix.
