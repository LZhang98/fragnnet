# Combined PCDL, NIST23, and Enveda-180 dataset

> **Status:** PREPARED · **Updated:** 2026-10-06
> A shared molecule-disjoint 80/10/10 split manifest exists and is verified; a directly loadable combined processed table does not (needs bounded-memory multi-source loading).

## Objective and acceptance

Combine the existing `data/proc/pcdl`, `data/proc/nist23`, and `data/proc/enveda_180` sources under one 80/10/10 molecule-disjoint split. Confirm that every source spectrum belongs to exactly one part and that no InChIKey first block crosses parts. A training-ready combined dataset additionally needs a loader that reads multiple processed sources without constructing one giant pickle.

## Current artifact

`data/split/pcdl_nist23_enveda_180_inchikey_811/manifest.json` records seed 42, source paths, and spectrum counts. Under each source directory, `train_ids.csv`, `val_ids.csv`, and `test_ids.csv` contain that source's original `spec_id`, `mol_id`, and `group_id`. IDs are source-local; `(source, spec_id)` is the spectrum identity. All source spectra are included. Model-specific precursor, collision-energy, and instrument filters have not been applied.

| Source | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| PCDL | 31,051 | 3,802 | 3,812 |
| NIST23 | 1,522,955 | 186,647 | 189,995 |
| Enveda-180 | 926,074 | 115,961 | 116,258 |

Across sources, 241,917 unique InChIKey first blocks form 193,529 train, 24,192 validation, and 24,191 test molecules. The three intersections are empty. Rebuild with `preproc_scripts/dataset_migration/build_combined_split_manifest.py --sources data/proc/pcdl data/proc/nist23 data/proc/enveda_180 --output data/split/pcdl_nist23_enveda_180_inchikey_811` after moving the existing artifact aside.

## Constraint and next check

The existing runner reads one `spec_df.pkl` and one `mol_df.pkl`. Two attempts to materialize a monolithic combined pickle were killed by memory pressure before writing output. The manifest is therefore a verified split definition, not yet a directly loadable training dataset. The next implementation should make the runner load per-source processed tables and remap IDs during loading, or otherwise build a bounded-memory combined store. It must normalize legacy `FT` labels in PCDL, apply supported-adduct filtering for model configs, and verify cross-source molecule/group merging. No model run has used this split.

Validation: Ruff passed for the builder; a full read of the three source molecule tables and nine split CSVs confirmed counts and zero InChIKey overlap. The unrelated pre-existing working-tree changes were not touched.
