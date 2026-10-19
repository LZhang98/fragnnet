# CASMI 2026 and Enveda-180 results

> **Status:** LEDGER · **Updated:** 2026-10-18
> Every measured model result on CASMI 2026, Enveda-180, ZenoTOF EAD and the RT/CCS heads, one row per run, checkpoint, fold, split and evaluation. Numbers are copied from the linked source, which stays canonical.

Add a row when a finding or plan records a new measurement; fix a row when its source is corrected. Hub: [casmi_enveda](../workstreams/casmi_enveda.md).

## Current best

- No full-data trained-model result exists for CASMI 2026, Enveda-180 or ZenoTOF EAD; the Enveda-180 baseline is prepared, not submitted ([hub](../workstreams/casmi_enveda.md#where-it-stands)).
- Only trained models: two RT/CCS debug runs on 247-spectrum subsets, selected epoch 8 in both. Separate-head run: RT MAE 40.76 s, CCS MAE 10.67 source units, CCS MAPE 6.08%, MS/MS cosine 0.3536. Source: [separate-head rerun](../plans/masster/masster_rt_ccs_plan.md#separate-head-debug-rerun--completed-2026-10-10).

## Comparability

- Both RT/CCS runs: local RTX 2070 SUPER, 10 epochs / 160 steps, seed 397623, W&B disabled (no run id), 256 spectra sampled per partition with seed 42 before adduct filtering (247 train / 247 val), GINE_GSA width 64; test evaluation disabled. Debug-subset measurements, not benchmark results, and not evidence of improvement over the MS/MS-only baseline ([source](../plans/masster/masster_rt_ccs_plan.md#local-gpu-debug-training--completed-2026-10-10)).
- The two runs are not a controlled comparison: the rerun's head weights initialize in a different order ([source](../plans/masster/masster_rt_ccs_plan.md#separate-head-debug-rerun--completed-2026-10-10)). CCS is in source units, with no cross-dataset calibration claimed.
- The isotope rows are model-free ceilings (perfect monoisotopic model plus best scalar envelope), not trained models; the Enveda-180 baseline (control `i0`) was never submitted.
- CASMI 2026 visible `test.parquet` is a verbatim copy of train rows, so any score on it measures memorization ([finding](../findings/datasets/casmi2026_dataset_2026_10_05.md#3-the-visible-test-is-enveda-180-and-it-is-inside-train)). ZenoTOF EAD and the halogen census are data-only (no model trained).

## RT/CCS head debug runs (Enveda-180, local GPU)

| Date | Model / arm | Run@epoch | Fold | val RT MAE (s) | val CCS MAE (source units) | val CCS MAPE (%) | val MS/MS cosine, 0.01 Da | val joint loss | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-10 | RT+CCS combined wrapper, MS/MS multitask | `enveda_rt_ccs_debug_s397623_20261010a`@0 (no W&B id) | Enveda-180 `inchikey_811` val subset (247 spectra; CCS valid on 235) | 80.2327 | 41.0389 | 22.6895 | 0.1287 | — | — | [RT/CCS plan](../plans/masster/masster_rt_ccs_plan.md#local-gpu-debug-training--completed-2026-10-10) |
| 2026-10-10 | RT+CCS combined wrapper, MS/MS multitask | `enveda_rt_ccs_debug_s397623_20261010a`@8 (selected) | Enveda-180 `inchikey_811` val subset (247 spectra; CCS valid on 235) | 37.4279 | 11.9031 | 6.4688 | 0.3780 | -1.30294 | — | [RT/CCS plan](../plans/masster/masster_rt_ccs_plan.md#local-gpu-debug-training--completed-2026-10-10) |
| 2026-10-10 | RT+CCS combined wrapper, MS/MS multitask | `enveda_rt_ccs_debug_s397623_20261010a`@9 (final) | Enveda-180 `inchikey_811` val subset (247 spectra; CCS valid on 235) | 41.3931 | 9.8066 | 5.3955 | 0.3395 | — | — | [RT/CCS plan](../plans/masster/masster_rt_ccs_plan.md#local-gpu-debug-training--completed-2026-10-10) |
| 2026-10-10 | separate `RTHead` + `CCSHead`, MS/MS multitask | `enveda_rt_ccs_separate_debug_s397623_20261010b`@8 (selected) | Enveda-180 `inchikey_811` val subset (247 spectra; CCS valid on 235) | 40.76 | 10.67 | 6.08 | 0.3536 | — | not a controlled comparison with run a | [RT/CCS plan](../plans/masster/masster_rt_ccs_plan.md#separate-head-debug-rerun--completed-2026-10-10) |

## Isotope-envelope ceiling on Enveda-180 (model-free)

| Date | Model / arm | Run@epoch | Fold | Subset | Cosine recovered | vs comparator | Source |
|---|---|---|---|---|---|---|---|
| 2026-09-27 | perfect monoisotopic model + best scalar envelope | — | Enveda-180 val (77,320 `[M+H]+`/`[M-H]-` spectra) | positive | 0.0047 (CI 0.0045-0.0050) | gate 0.005; seed spread 0.0042 | [Enveda-180 finding](../findings/datasets/enveda_180_dataset_2026_09_25.md#94-what-a-monoisotopic-prediction-cannot-reach); [retired isotope plan](../history/retired_docs.md#masster_generalization) |
| 2026-09-27 | perfect monoisotopic model + best scalar envelope | — | Enveda-180 val | negative | 0.0010 | gate 0.005 | [Enveda-180 finding](../findings/datasets/enveda_180_dataset_2026_09_25.md#94-what-a-monoisotopic-prediction-cannot-reach) |
| 2026-09-27 | perfect monoisotopic model + natural-abundance prior | — | Enveda-180 val | positive | 0.0000 | best scalar envelope 0.0047 | [retired isotope plan](../history/retired_docs.md#masster_generalization) |
| 2026-09-27 | perfect monoisotopic model + natural-abundance prior | — | Enveda-180 val | negative | costs 0.013 | best scalar envelope 0.0010 | [Enveda-180 finding](../findings/datasets/enveda_180_dataset_2026_09_25.md#94-what-a-monoisotopic-prediction-cannot-reach) |
