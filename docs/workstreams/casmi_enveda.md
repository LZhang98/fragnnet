# CASMI 2026, Enveda-180 and shared data quality

> **Status:** HUB · **Updated:** 2026-10-16
> CASMI 2026 (Enveda) data release, the Enveda-180 dataset and its baseline config, the ZenoTOF CID/EAD libraries, the combined PCDL+NIST23+Enveda-180 split, and shared NIST data-quality / subformula / isotope-detection work.

## Where it stands

- **Shared preprocessing audit:** seven correctness findings (removed 2026-10-18; `git show 731fe493:docs/findings/preproc_scripts_audit_2026_10_12.md`), including broken merged molecule/group identities; fixed with regression tests and committed in `bbcdf00c` (2026-10-13). Rebuild merger, inference-merged or `--canonicalize_tautomers False` outputs made before it into new directories before use.

- **ZenoTOF CID + EAD libraries:** [MultiMS2 and MS-DIAL 5 standards preprocessed](../findings/zenotof_ead_libraries_2026_10_15.md) (steps 1-2, `data/proc/multims2`, `data/proc/msdial5_ead`) with a new `EAD` frag_mode (`ace` = electron kinetic energy) and `frag_mode` in the `group_id` key (uncommitted). CIeaD deferred: no structures in any export. Next: splits and an EAD-aware config.

- **RT/CCS:** `RTHead` and `CCSHead` with independent label sets: [GPU debug rerun completed](../plans/masster/masster_rt_ccs_plan.md#separate-head-debug-rerun--completed-2026-10-10): 10 epochs on small train/validation subsets, checkpoint retained. Full Enveda arm remains prepared.

- **CASMI 2026:** data extracted to `data/raw/casmi2026/`, nothing preprocessed. The visible `test.parquet` is a verbatim copy of `train.parquet` (Enveda-180 rows); the hidden natural-product test replaces it at Kaggle notebook scoring. Validate only on a molecule-disjoint split of train.
- **Train quality:** labels consistent; 166k HR spectra are poorly explained by their labelled molecule (70% RIKEN rows labelled plain `QTOF`/`Orbitrap`, mostly unmerged BMDMS-NP replicate scans); the Spectraverse dedup rule drops 17.6% of train; 2,022 multi-peak spectra are filed under two molecules.
- **Distribution bias:** train is synthetic-heavy (NP median -1.31; Enveda-180 is 66% of molecules at -1.49) against the NP proxy (+1.42); the NP proxy's nearest-neighbour Tanimoto to train has median 0.775. Quality filters fall on NP-like data (19.7% suspects, 33.7% near-duplicates).
- **Halogens:** [CASMI halogen census](../findings/casmi2026_halogen_census_2026_10_16.md): high-quality MSnLib + Enveda-180 hold 7,558 Br / 25,794 Cl molecules (NIST20 HR 1,837 / 3,662), almost all new, and 354 Br→Cl pairs (NIST20 train: 57); Enveda-180 supplies most.
- **MSnLib vs NIST:** nearly disjoint (6,432 shared molecules, 13.7% of MSnLib); NIST HR would add 34,295 molecules not in any CASMI train library.
- **Landed in `09e95e9a`:** finding sections 5.7-5.9 and scripts `scripts/debug/casmi_enveda/{casmi26_mol_distribution,msnlib_nist_coverage}.py`.
- **Enveda-180 baseline** (standalone, ACE only, no EMA; user decisions): `config/masster/enveda/enveda_180_inchikey_811_unmerged_mh_mmh_ace_fourier.yml`, not submitted, needs a second seed.
- **Combined split:** manifest exists (seed 42, 80/10/10, zero molecule overlap); no loader reads it as one dataset.
- **Single next action:** decide the CASMI train filter (drop suspects, near-duplicates, low-res rows, keep good BMDMS-NP scans), then preprocess to a split.

## Open threads

| Thread | Status | Next concrete step |
| --- | --- | --- |
| [ZenoTOF CID + EAD libraries](../findings/zenotof_ead_libraries_2026_10_15.md) | STEPS 1-2 DONE, UNCOMMITTED | Review the reader/group-key change; build splits; config with `frag_modes` incl. `EAD`; CIeaD structures from the authors |
| [CASMI train filter](../findings/casmi2026_dataset_2026_10_05.md) | OPEN | Choose filter (subformula suspects at ORB tier vs shifted-peak control, dedup, `riken` rule in section 5.9), then preprocess |
| CASMI external data (NIST) | OPEN | Competition rules on external data not checked; hidden test energy protocol unknown (section 6) |
| [RT/CCS heads](../plans/masster/masster_rt_ccs_plan.md) | DEBUG COMPLETE; full arm PREPARED | Full matched multitask/control evaluation; confirm CCS calibration before cross-dataset use |
| [Enveda-180 baseline](../findings/enveda_180_dataset_2026_09_25.md#10-baseline-config-and-run-state-2026-09-2728-from-project-state-handoff) | PREPARED | Second seed, then submit; re-check SMARTS rules on Enveda-180 (see generalization hub) |

## Settled — read before re-proposing

- **CASMI**
  - [Visible test is train](../findings/casmi2026_dataset_2026_10_05.md#3-the-visible-test-is-enveda-180-and-it-is-inside-train) — any score on `test.parquet` measures memorization.
  - [Subformula coverage check](../findings/casmi2026_dataset_2026_10_05.md#54-subformula-coverage) — judge only at the tight (ORB) tier against a shifted-peak control.
  - [`riken` noise](../findings/casmi2026_dataset_2026_10_05.md#59-what-the-riken-library-is-and-why-part-of-it-is-noisy) — BMDMS-NP unmerged replicate scans; the filter keeps 37,157 spectra / 2,469 molecules (untested on a model).
- **Enveda-180**
  - [Standalone, ACE only, no EMA](../findings/enveda_180_dataset_2026_09_25.md) — user decision; molecules effectively disjoint from NIST20/23, MSGYM and Spectraverse (26-66 shared).
  - [Isotopes](../findings/enveda_180_dataset_2026_09_25.md#9-isotopes-2026-09-25) — 13C satellites are 7.7% of positive-mode target; envelope gate fails (0.0047 vs 0.005), so no isotope modelling for cosine.
  - [`[M+Br]-` support](../findings/enveda_180_dataset_2026_09_25.md#8-mbr--support-2026-09-25-uncommitted) — added to the adduct tables; 25 Enveda and 224 NIST23 spectra; the split needs regenerating to include them.
  - [SMARTS gate on Enveda-180](../findings/masster/masster_smarts_enveda_gate_2026_09_29.md) — only S=O rules pass, 2-10x weaker than NIST20 (owned by the generalization hub).
- **Shared data quality**
  - NIST20 zero-coverage (removed 2026-10-18; `git show 95341fce:docs/plans/nist_data_quality_plan.md`) — only 255 spectra (0.069%); 16 are lipid precursor-annotation errors; UWPR flags are a sensitivity label, not ground truth.
  - [5 ppm subformula filter](../findings/subformula_filter_ft_qtof.md) — NIST20-HR census by FT/QTOF; the Cl/Na budget fix reaches MASSter.
  - Subformula isotope source (removed 2026-10-13; `git show e65339ed:docs/plans/subformula_isotope_detection_plan.md`) — matches the DAG source (F1 0.808 vs 0.805; 99.2% identical decisions); no DAG or `frag_dp` needed.
- **Retired 2026-10-18:** 2 documents of this workstream removed; summaries and open items in [retired documents](../history/retired_docs.md#casmi_enveda).
- Combined PCDL+NIST23+Enveda-180 — retired 2026-10-18 while PREPARED (`git show 95341fce:docs/plans/combined_pcdl_nist23_enveda_180_dataset.md`); if revived: bounded-memory multi-source loader and model-specific filters.
- NIST20 data-quality validation — retired 2026-10-18 while DEFERRED (`git show 95341fce:docs/plans/nist_data_quality_plan.md`); if revived: independent contaminant false-removal check before any peak stripping; precursor-mass sanity filter.
- Combined PCDL+NIST23+Enveda-180 split manifest: `data/split/pcdl_nist23_enveda_180_inchikey_811` (rebuild with `preproc_scripts/dataset_migration/build_combined_split_manifest.py`); no loader reads it yet (needs bounded-memory multi-source loading with source-local `spec_id` remapping).

## All documents

| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [CASMI halogen census](../findings/casmi2026_halogen_census_2026_10_16.md) | FINDING (2026-10-16) | Br/Cl/I carriers per library, halogen-aware quality filter, Br→Cl swap pairs vs NIST20 |
| [ZenoTOF CID + EAD libraries](../findings/zenotof_ead_libraries_2026_10_15.md) | FINDING (2026-10-15) | MultiMS2, MS-DIAL 5 standards, CIeaD: sources, contents, EAD frag_mode, group key, processed outputs |
| [CASMI 2026 data](../findings/casmi2026_dataset_2026_10_05.md) | FINDING (2026-10-06) | Release contents, placeholder test, train checks, MSnLib/NIST overlap, NP bias, `riken` |
| [MASSter RT/CCS heads](../plans/masster/masster_rt_ccs_plan.md) | PREPARED (2026-10-10) | RTHead and CCSHead, label audit, completed GPU debug run and full experiment |
| [Enveda-180 dataset](../findings/enveda_180_dataset_2026_09_25.md) | FINDING (2026-10-06) | Source, CE, adducts, preprocessing, split, `[M+Br]-`, isotopes, baseline config |
| [5 ppm subformula filtering, NIST20-HR](../findings/subformula_filter_ft_qtof.md) | FINDING (2026-10-06) | Census, instrument tolerance, Cl/Na fix |
| [NIST human review guide](../guides/nist_human_review.md) | GUIDE (linked only) | Manual spectral review procedure |
| [Isotope-peak detection tuning](../reference/detect_isotope_peaks_tuning.md) | REFERENCE (linked only) | Tuning of the isotope-peak detectors (`utils/isotope_cleanup_utils.py`: `detect_isotope_peaks_for_training` and its formula-aware variant; the naive `detect_isotope_peaks` was removed 2026-10-12) |
| [SMARTS gate on Enveda-180](../findings/masster/masster_smarts_enveda_gate_2026_09_29.md) | linked only | Per-class SMARTS gate |

Shared data quality (NIST20 diagnostics, subformula filter, isotope detection) is listed here because no other hub owns it; Spectraverse filter screens are in [the Spectraverse hub](spectraverse.md).

## Key artifacts

- CASMI: `data/raw/casmi2026/{train,test}.parquet`; `output/casmi26_train_check_2026_10_05/`, `output/casmi26_mol_distribution_2026_10_06/`, `output/msnlib_nist_coverage_2026_10_06/`; scripts `scripts/debug/casmi_enveda/{casmi26_train_stats,casmi26_train_duplicates,casmi26_mol_distribution,msnlib_nist_coverage}.py`; notebook `notebooks/data_notebooks/casmi26_train.ipynb`. pyarrow 25.0.1 is in `FRAGNNET-GPU` (user approved).
- Enveda-180: `data/raw/enveda-180/`, `data/df/enveda_180_df.csv`, `data/proc/enveda_180/` (1,158,293 spectra / 183,476 molecules), split `data/split/enveda_180_inchikey_811` (143,568 / 17,946 / 17,946 molecules; 640,475 / 79,881 / 80,152 spectra); config `config/masster/enveda/`. Step 2 peaks at 23.3 GB RSS plus swap locally.
- Combined split: `data/split/pcdl_nist23_enveda_180_inchikey_811/manifest.json` (241,917 molecules: 193,529 / 24,192 / 24,191); builder `preproc_scripts/dataset_migration/build_combined_split_manifest.py`.
- NIST20 quality: `data/eval/oracle_cos_nist20_default/zero_coverage_spectra.csv`, `data/uwpr/nist20_uwpr_hits_feasibility.csv`, `scripts/eval/{dag_oracle_scores,uwpr_feasibility}.py`.
- Neighbours: [MASSter generalization](masster_generalization.md), [MASSter core](masster_core.md), [MSGYM](msgym.md), [FraGNNet](fragnnet.md).
