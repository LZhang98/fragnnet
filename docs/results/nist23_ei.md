# NIST23 EI results

> **Status:** LEDGER · **Updated:** 2026-10-19
> Every measured model result on nominal NIST23 EI and on held-out HR-EI spectra, one row per run, checkpoint, split and evaluation. Numbers are copied from the linked source, which stays canonical.

Add a row when a finding or plan records a new measurement; fix a row when its source is corrected. Hub: [nist23_ei](../workstreams/nist23_ei.md).

## Current best

- Nominal NIST23: e1 (natural isotopes) [`jeljdikj`](https://wandb.ai/frag-gnn/frag-gnn/runs/jeljdikj)@37, `val_cos_sim_1.0` 0.870173; full test (51,685 spectra) mean 1 Da cosine 0.869984, mean Stein 0.886845, 0.899167 on the 26,372 RASSP-eligible spectra ([source](../plans/masster/masster_ei_nominal_plan.md#stein-weighted-similarity-on-the-full-test-split-2026-10-01)).
- Held-out HR-EI (10 ppm, natural exact isotopes): e5 [`eugy8s9n`](https://wandb.ai/frag-gnn/frag-gnn/runs/eugy8s9n)@45, mean cosine 0.8497 HREI-MSDB, 0.7062 RECETOX Exposome, 0.5726 RECETOX Metabolome ([source](../findings/masster/masster_ei_evaluation.md#held-out-e5-ei-checkpoint-on-measured-hr-ei-spectra)).
- Isotope effect (one seed): mean `val_cos_sim_1.0` over epochs 54-63, e1 0.867738 vs e0 0.859883, a 0.007856 difference ([source](../plans/masster/masster_ei_nominal_plan.md#wb-completion-check-2026-09-30-session)).
- Support oracle (512 val spectra): 0.99615 mean Stein vs the model's 0.88703 ([source](../findings/masster/masster_ei_evaluation.md#masster-ei-performance-audit)).
- EI SMARTS (TMS first): CLOSED NULL, no rule admitted at the NIST23 gate ([source](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18)).

## Comparability

- Nominal NIST23 EI (`nist23_ei`, `[M]+`, 70 eV, integer m/z) is scored with 1 Da cosine; checkpoint selection metric is `val_cos_sim_1.0_epoch/mean`; `loss_tolerance_abs` 0.5; the split is random by 2D InChIKey, 154,981 / 51,700 / 51,685.
- Stein similarity: cosine after transforming each nominal peak to `m/z**3 * intensity**0.6` (RASSP metric). RASSP FormulaNet's 0.929 is on its own NIST17 set with its own filtering and split; training overlap, dataset, preprocessing and split are unmatched and RASSP has not been run on our test set, so the 0.030 difference is descriptive only.
- e0/e1/e5 each have one seed; run-to-run variance is unmeasured. e0/e1 W&B metadata has no Git revision, so inclusion of the local isotope-offset fix is unverified.
- HR-EI reference set: 962 mapping-consistent records (329 HREI-MSDB, 383 RECETOX Exposome, 250 RECETOX Metabolome); Castro excluded (no resolved structures). Counts are spectrum records, not unique molecules.
- HR-EI score: mean linear-intensity cosine with maximum-product one-to-one peak assignment; unmatched predicted and reference peaks stay in the norms. Headline tolerance is 10 ppm with a 1 mDa floor (user decision 2026-10-18), natural exact isotopes on. Floors for 5/10/20 ppm are 0.0005/0.001/0.002 Da. HREI-MSDB predictions are restricted to 40-500 m/z; others use 0-1500 m/z.
- Exposome 5 to 10 ppm sensitivity is a property of the reference masses or matching floor (peak annotations about 7 ppm above reported masses), not of the checkpoint.
- e1 is not a valid comparison for HR-EI (user decision 2026-10-19): it trained on 197 HREI-MSDB, 220 Exposome and 114 Metabolome overlapping records by the raw-audit flag, and possibly on native parents of others. Its HR-EI rows stay as an implementation record only; do not compare them with e5. e5 is trained on `nist23_ei_inchikey_hr_ei_excluded_2026_10_17` and is the HR-EI result.
- Rounded-mass control: the same monoisotopic exact predictions and measured masses rounded to integers and summed; not a forward pass of the nominal/natural-isotope configuration.
- HR-vs-NIST23 library comparisons are nominal linear cosines cut to the source window (40-500 HREI-MSDB, >=67 Exposome, >=70 Metabolome), over the 860 of 962 records with a NIST23 spectrum of the same 2D connectivity; e5 never trained on these.
- Metabolome: 243 of 250 records are TMS derivatives; the 149.045 ion is instrument-specific, and its reference isotope peaks look incomplete or merged.

## Nominal NIST23 (validation, selection metric)

| Date | Model / arm | Run@epoch | Best `val_cos_sim_1.0` (epoch) | `val_cos_sim_1.0` epoch 63 | Mean `val_cos_sim_1.0` epochs 54-63 | Final `val_wrecall` | vs comparator | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-30 | e0 (nominal, no isotopes) | `he3j5y3o`@57 | 0.861611 (57) | 0.859522 | 0.859883 | 0.974399 | — | [W&B completion check](../plans/masster/masster_ei_nominal_plan.md#wb-completion-check-2026-09-30-session) |
| 2026-09-30 | e1 (nominal, natural isotopes) | `jeljdikj`@37 | 0.870173 (37) | 0.866724 | 0.867738 | 0.996870 | Last-ten-epoch comparison favors e1 by 0.007856 cosine (one matched-seed pair) | [W&B completion check](../plans/masster/masster_ei_nominal_plan.md#wb-completion-check-2026-09-30-session) |
| 2026-10-18 | e5 (HR-EI-excluded split, natural isotopes) | `eugy8s9n`@45 | 0.86896 (45) | — | — | — | — | [e5 setup](../findings/masster/masster_ei_evaluation.md#setup) |

## Nominal NIST23 (test)

| Date | Model / arm | Run@epoch | Subset (n) | Mean 1 Da cosine | Mean Stein | Median Stein (p10 / p90) | vs comparator | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-01 | e1 | `jeljdikj`@37 | Full test (51,685) | 0.869984 | — | — | — | [Full test by size](../plans/masster/masster_ei_nominal_plan.md#full-test-performance-by-molecule-size-2026-10-01) |
| 2026-10-01 | e1 | `jeljdikj`@37 | Full test (51,685), Stein rerun | 0.8699833 | 0.886845 | 0.938095 (0.715053 / 0.988242) | RASSP FormulaNet 0.929 Stein on NIST17 (descriptive only) | [Stein on full test](../plans/masster/masster_ei_nominal_plan.md#stein-weighted-similarity-on-the-full-test-split-2026-10-01) |
| 2026-10-01 | e1 | `jeljdikj`@37 | <=64 total atoms (46,435) | — | 0.890374 | — | — | [Stein on full test](../plans/masster/masster_ei_nominal_plan.md#stein-weighted-similarity-on-the-full-test-split-2026-10-01) |
| 2026-10-01 | e1 | `jeljdikj`@37 | >64 total atoms (5,250) | — | 0.855634 | — | — | [Stein on full test](../plans/masster/masster_ei_nominal_plan.md#stein-weighted-similarity-on-the-full-test-split-2026-10-01) |
| 2026-10-01 | e1 | `jeljdikj`@37 | RASSP-eligible: H/C/N/O/F/P/S/Cl, <=48 total atoms, <=4096 formulae (26,372 spectra of 25,558 molecules) | — | 0.899167 | — | RASSP FormulaNet 0.929 Stein on NIST17 (descriptive only; RASSP not run on our test set) | [Stein on full test](../plans/masster/masster_ei_nominal_plan.md#stein-weighted-similarity-on-the-full-test-split-2026-10-01) |
| 2026-10-01 | e1 | `jeljdikj`@37 | Molecular-weight quintiles, smallest to largest bin | 0.8729 (32-200 Da), 0.8703 (371-1485 Da); range 0.8654-0.8754 | 0.9066 to 0.8495 | — | — | [Full test by size](../plans/masster/masster_ei_nominal_plan.md#full-test-performance-by-molecule-size-2026-10-01) |
| 2026-10-01 | e1 | `jeljdikj`@37 | Total-atom groups (incl. H), 2-16 to 65+ | 0.8730 (2-16), 0.8887 (65-209); range 0.8614-0.8887 | 0.9202 (2-16 atoms) to 0.8556 (>64 atoms) | — | — | [Full test by size](../plans/masster/masster_ei_nominal_plan.md#full-test-performance-by-molecule-size-2026-10-01) |
| 2026-10-06 | e1 | `jeljdikj`@37 | Replicate-bearing test spectra (2,245 replicate pairs) | — | 0.85360 | — | Pairwise Stein between replicates 0.91889 | [Audit, limits](../findings/masster/masster_ei_evaluation.md#what-limits-the-current-score) |
| 2026-10-18 | e5 | `eugy8s9n`@45 | e5 internal test, 0 TMS groups (6,000 sampled) | 0.868 | 0.889 | — | — | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |
| 2026-10-18 | e5 | `eugy8s9n`@45 | e5 internal test, 1 TMS group (696) | 0.846 | 0.865 | — | — | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |
| 2026-10-18 | e5 | `eugy8s9n`@45 | e5 internal test, 2 TMS groups (429) | 0.817 | 0.815 | — | — | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |
| 2026-10-18 | e5 | `eugy8s9n`@45 | e5 internal test, 3 TMS groups (165) | 0.794 | 0.772 | — | — | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |
| 2026-10-18 | e5 | `eugy8s9n`@45 | e5 internal test, >=4 TMS groups (146) | 0.806 | 0.708 | — | — | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |
| 2026-10-18 | e5 | `eugy8s9n`@45 | TMS penalty after reweighting non-TMS to TMS mass distribution (six exact-mass bins) | -0.035 [-0.046, -0.024] | -0.044 [-0.055, -0.034] | — | Within-bin TMS minus non-TMS cosine: -0.005 (150-250), -0.028 (250-350), -0.051 (350-450), -0.093 (450-600); 500 bootstrap resamples | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |

## Post-hoc ablations on e1 (512-spectrum validation sample)

| Date | Model / arm | Run@epoch | Ablation (fixed predictions, no retraining) | Mean 1 Da cosine | Mean Stein | vs comparator | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-01 | e1 | `jeljdikj`@37 | None (annotated inference, one spectrum per molecule, seed 20261001) | 0.86905 | — | W&B best validation 0.87017 | [Annotation read](../plans/masster/masster_ei_nominal_plan.md#e1-checkpoint-annotation-read-2026-10-01-4-used-6-unresolved) |
| 2026-10-01 | e1 | `jeljdikj`@37 | Remove H+/-4 rows (H+/-4 improves cosine in 63.7% of sampled spectra) | 0.86805 | — | vs 0.86905 unmodified | [Annotation read](../plans/masster/masster_ei_nominal_plan.md#e1-checkpoint-annotation-read-2026-10-01-4-used-6-unresolved) |
| 2026-10-01 | e1 | `jeljdikj`@37 | Remove all H+/-3/4 rows | 0.86284 | — | vs 0.86905 unmodified | [Annotation read](../plans/masster/masster_ei_nominal_plan.md#e1-checkpoint-annotation-read-2026-10-01-4-used-6-unresolved) |
| 2026-10-06 | e1 | `jeljdikj`@37 | Peak powers 0.6, 0.8, 1.2, 1.4 | — | All below 0.88703 | vs 0.88703 unmodified | [Audit, limits](../findings/masster/masster_ei_evaluation.md#what-limits-the-current-score) |
| 2026-10-06 | e1 | `jeljdikj`@37 | Multiply intensities by `m/z**alpha`, alpha -1, -0.5, +0.5, +1 | — | All lower than 0.88703 | vs 0.88703 unmodified | [Audit, limits](../findings/masster/masster_ei_evaluation.md#what-limits-the-current-score) |

## Ceilings and support oracles

| Date | Model / arm | Run@epoch | Sample | Evaluation | Value | vs comparator | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-06 | e1 | `jeljdikj`@37 | 512 validation spectra | Support oracle: predicted m/z support with measured intensities, mean Stein | 0.99615 | Model 0.88703 | [Audit, limits](../findings/masster/masster_ei_evaluation.md#what-limits-the-current-score) |
| 2026-10-06 | e1 | `jeljdikj`@37 | 512 validation spectra | Mean target Stein weighted squared norm covered by predicted m/z | 0.99321 | — | [Audit, limits](../findings/masster/masster_ei_evaluation.md#what-limits-the-current-score) |
| 2026-10-06 | e1 | `jeljdikj`@37 | >64-total-atom group of the sample (45 spectra) | Support oracle Stein / model Stein | 0.9902 / 0.8234 | Written model/oracle 0.8234/0.9902 in source | [Audit, limits](../findings/masster/masster_ei_evaluation.md#what-limits-the-current-score) |
| 2026-09-30 | e1 | `jeljdikj`@37 | Validation, epoch 63 | Final monoisotopic candidate weighted recall | 0.974254 | e0 0.974399 | [W&B completion check](../plans/masster/masster_ei_nominal_plan.md#wb-completion-check-2026-09-30-session) |
| 2026-09-30 | `i1` keys on EI data, exact masses (smoke) | — | 512-spectrum subsample, 1 epoch | `val_candidate_wrecall` / `val_wrecall` | 0.021 / 0.021 | — | [Implementation state](../plans/masster/masster_ei_nominal_plan.md#implementation-state-2026-09-30-working-tree-uncommitted) |
| 2026-09-30 | e0 (nominal, smoke) | — | 512-spectrum subsample, 1 epoch | `val_candidate_wrecall` / `val_wrecall` | 0.954 / 0.954 | — | [Implementation state](../plans/masster/masster_ei_nominal_plan.md#implementation-state-2026-09-30-working-tree-uncommitted) |
| 2026-09-30 | e1 (nominal + natural isotopes, smoke) | — | 512-spectrum subsample, 1 epoch | `val_candidate_wrecall` / `val_wrecall` | 0.954 / 0.983 | One-epoch cosines 0.215 (e0) and 0.208 (e1) say nothing about the arms | [Implementation state](../plans/masster/masster_ei_nominal_plan.md#implementation-state-2026-09-30-working-tree-uncommitted) |

## Candidate reach ceilings (model-free, 5,000 validation molecules)

| Date | Channels / H limit | L1 reach, no isotope offsets | L1 reach, +1/+2 reach ceiling | Mean per-spectrum cosine ceiling | vs comparator | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-30 | Max abs H shift 0 | 0.4972 | 0.7212 | 0.8971 | — | [H-transfer width screen](../plans/masster/masster_ei_nominal_plan.md#ei-h-transfer-width-screen-2026-09-30-session-retain-4-for-now) |
| 2026-09-30 | Max abs H shift 1 | 0.7735 | 0.8578 | 0.9630 | — | [H-transfer width screen](../plans/masster/masster_ei_nominal_plan.md#ei-h-transfer-width-screen-2026-09-30-session-retain-4-for-now) |
| 2026-09-30 | Max abs H shift 2 | 0.8520 | 0.9040 | 0.9744 | — | [H-transfer width screen](../plans/masster/masster_ei_nominal_plan.md#ei-h-transfer-width-screen-2026-09-30-session-retain-4-for-now) |
| 2026-09-30 | Max abs H shift 3 | — | 0.9339 | 0.9817 | — | [H-transfer width screen](../plans/masster/masster_ei_nominal_plan.md#ei-h-transfer-width-screen-2026-09-30-session-retain-4-for-now) |
| 2026-09-30 | Max abs H shift 4 (production) | 0.9108 | 0.9482 | 0.9847 | — | [H-transfer width screen](../plans/masster/masster_ei_nominal_plan.md#ei-h-transfer-width-screen-2026-09-30-session-retain-4-for-now) |
| 2026-10-01 | Max abs H shift 5 | 0.9228 | 0.9576 | 0.9870 | — | [+/-6 H screen](../plans/masster/masster_ei_nominal_plan.md#ei-6-h-screen-2026-10-01-retain-4) |
| 2026-10-01 | Max abs H shift 6 | 0.9289 | 0.9613 | 0.9878 | Mean per-spectrum cosine ceiling rises 0.00314 from +/-4 to +/-6 | [+/-6 H screen](../plans/masster/masster_ei_nominal_plan.md#ei-6-h-screen-2026-10-01-retain-4) |

| Date | Channels reaching the peak (Stage 0 probe) | L1 share | L2 share | Mean sqrt(L2), cosine bound | Source |
| --- | --- | --- | --- | --- | --- |
| 2026-09-30 | H shift 0 only | 0.497 | 0.643 | 0.771 | [Nominal reach](../plans/masster/masster_ei_nominal_plan.md#nominal-reach-of-the-m-candidate-space) |
| 2026-09-30 | abs(H) <= 1 | 0.773 | 0.883 | 0.932 | [Nominal reach](../plans/masster/masster_ei_nominal_plan.md#nominal-reach-of-the-m-candidate-space) |
| 2026-09-30 | abs(H) <= 2 | 0.852 | 0.926 | 0.958 | [Nominal reach](../plans/masster/masster_ei_nominal_plan.md#nominal-reach-of-the-m-candidate-space) |
| 2026-09-30 | abs(H) <= 4 | 0.911 | 0.952 | 0.973 | [Nominal reach](../plans/masster/masster_ei_nominal_plan.md#nominal-reach-of-the-m-candidate-space) |
| 2026-09-30 | abs(H) <= 4, plus +1/+2 isotopes | 0.948 | 0.973 | 0.985 | [Nominal reach](../plans/masster/masster_ei_nominal_plan.md#nominal-reach-of-the-m-candidate-space) |
| 2026-09-30 | Null: abs(H) <= 4 channels of the nearest-mass other molecule | 0.819 | 0.836 | 0.893 | [Nominal reach](../plans/masster/masster_ei_nominal_plan.md#nominal-reach-of-the-m-candidate-space) |

## SMARTS gate (CLOSED NULL, model-free)

NIST23 gate, train and val peaks scored, test structures counted only; decision rule `b and (c or e)` with >=20 val molecules.

| Date | Rule | Molecules (train+val) | (b) unreachable fraction | (c) rule - null added nominal coverage, 95% CI | Gate | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-18 | T1 `tms_siloxy_silylium` | 2,067 | 1.000 | -0.0018 [-0.0029, -0.0007] | Fails on train+val (e below), borderline | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |
| 2026-10-18 | T1b `tms_siloxy_silanol` | 1,826 | 0.801 | -0.0094 [-0.0107, -0.0080] | Fails (b, c) | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |
| 2026-10-18 | T2 `vicinal_otms_geminal` | 180 | 0.017 | +0.0024 [+0.0011, +0.0037] | Fails (b) | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |
| 2026-10-18 | T3 `silyl_to_ester_carbonyl` | 227 | 0.379 | +0.0024 [+0.0003, +0.0047] | Fails (b) | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |
| 2026-10-18 | ESI rewire `sulfonyl_so2_extrusion` under `[M]+` | 6,677 | 0.835 | +0.0071 (CI above 0) | Fails (b); no ESI rule reaches 0.95 | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |
| 2026-10-18 | ESI rewire `s_to_o_aromatic_C` under `[M]+` | 5,301 | 0.754 | +0.0061 (CI above 0) | Fails (b) | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |
| 2026-10-18 | ESI rewire `thiono_thiolo_aryl` under `[M]+` | 119 | 0.916 | +0.0050 (CI above 0) | Fails (b) | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |

| Date | T1 isotope consistency (e): share with observed 148/147 closer to C5H15OSi2 | Fold | T1 (>=2 TMS, 147 >=10% of base) | 1-TMS control | Median abs. error, Si2 vs competitor | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-18 | Molecule bootstrap | train+val | 0.739 [0.714, 0.764] (1,164 mol / 1,453 spectra) | 0.667 [0.597, 0.731] (201 / 221) | 0.0115 vs 0.0231 | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |
| 2026-10-18 | Molecule bootstrap | val | 0.780 [0.731, 0.826] (287 / 368) | 0.566 [0.434, 0.698] (53 / 60) | 0.011 vs 0.023 | [NIST23 gate](../plans/masster/masster_ei_smarts_plan.md#nist23-gate-2026-10-18) |

| Date | Earlier EI SMARTS screen (all 250,644 molecules; removed finding, summary in retired docs) | Added coverage | Source |
| --- | --- | --- | --- |
| 2026-10-06 | Anisoyl fluoride CO loss/F transfer (0/1/0 train/val/test matches) | Zero added | [retired_docs](../history/retired_docs.md#nist23_ei) |
| 2026-10-06 | para-hydroxyazobenzene N2 loss (3/3/0) | 0.000 added | [retired_docs](../history/retired_docs.md#nist23_ei) |
| 2026-10-06 | N-H aromatic-acyl arylsulfonamide carbonyl-O transfer (7/0/0) | +2.414% train; null-adjusted +1.034 pts CI [-0.010, +3.063]; no val example | [retired_docs](../history/retired_docs.md#nist23_ei) |

## Held-out HR-EI (10 ppm)

Isotope output "natural exact" is `masster_isotope_envelope: natural` (order 4, minimum ratio 0.001) with exact masses; "monoisotopic" is envelope `none`. Mean cosine is the 10 ppm value with a 1 mDa floor. Rows for e1 are not a valid HR-EI comparison (2026-10-19); see Comparability.

| Date | Model / arm | Run@epoch | Reference (spectra) | Isotope output | Mean cosine | Median cosine | Measured intensity with predicted match | vs comparator | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | HREI-MSDB (329) | monoisotopic | 0.8391 | — | — | Rounded-mass control 0.8491 | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Exposome (383) | monoisotopic | 0.6549 | — | — | Rounded-mass control 0.6902 | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Metabolome (250) | monoisotopic | 0.6009 | — | — | Rounded-mass control 0.6517 | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | HREI-MSDB (329) | natural exact | 0.8618 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Exposome (383) | natural exact | 0.7549 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Metabolome (250) | natural exact | 0.5973 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | HREI-MSDB (329) | natural exact | 0.8497 | 0.8712 | 0.973 | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | RECETOX Exposome (383) | natural exact | 0.7062 | 0.7772 | 0.844 | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | RECETOX Metabolome (250) | natural exact | 0.5726 | 0.5743 | 0.768 | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | HREI-MSDB (329) | monoisotopic | 0.8272 | — | — | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | RECETOX Exposome (383) | monoisotopic | 0.6127 | — | — | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | RECETOX Metabolome (250) | monoisotopic | 0.5764 | — | — | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |

## HR-EI other tolerances and controls

| Date | Model / arm | Run@epoch | Reference (spectra) | Tolerance | Isotope output | Mean cosine | Median cosine | Share of spectra cosine >=0.8 | Mean measured intensity with any predicted supporting mass | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | HREI-MSDB (329) | 5 ppm | monoisotopic | 0.8390 | 0.8662 | about 69.3% | 87.65% | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Exposome (383) | 5 ppm | monoisotopic | 0.6114 | 0.6997 | about 36.6% | 61.12% | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Metabolome (250) | 5 ppm | monoisotopic | 0.6009 | 0.6263 | about 28.4% | 57.94% | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | HREI-MSDB (329) | 20 ppm | monoisotopic | 0.8392 | — | — | — | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Exposome (383) | 20 ppm | monoisotopic | 0.6592 | — | — | — | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Metabolome (250) | 20 ppm | monoisotopic | 0.6025 | — | — | — | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | HREI-MSDB (329) | Rounded-mass control | monoisotopic, rounded to integers | 0.8491 | — | — | — | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Exposome (383) | Rounded-mass control | monoisotopic, rounded to integers | 0.6902 | — | — | — | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Metabolome (250) | Rounded-mass control | monoisotopic, rounded to integers | 0.6517 | — | — | — | [Frozen e1 results](../findings/masster/masster_ei_evaluation.md#results) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | HREI-MSDB (329) | 5 ppm | natural exact | 0.8617 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Exposome (383) | 5 ppm | natural exact | 0.6804 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Metabolome (250) | 5 ppm | natural exact | 0.5971 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | HREI-MSDB (329) | 20 ppm | natural exact | 0.8619 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Exposome (383) | 20 ppm | natural exact | 0.7615 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-17 | e1 (frozen, not a valid HR-EI comparison) | `jeljdikj`@37 | RECETOX Metabolome (250) | 20 ppm | natural exact | 0.5986 | — | — | — | [Natural exact follow-up](../findings/masster/masster_ei_evaluation.md#follow-up-natural-exact-mass-isotope-output) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | HREI-MSDB (329) | 5 ppm | natural exact | 0.8495 | — | — | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | RECETOX Exposome (383) | 5 ppm | natural exact | 0.6315 | — | — | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | RECETOX Metabolome (250) | 5 ppm | natural exact | 0.5724 | — | — | — | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |

## Held-out HR-EI isotope effect (e5, 10 ppm, paired, isotopes on minus off)

| Date | Model / arm | Run@epoch | Reference (spectra) | Cosine difference [95% CI] | Intensity recall change | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | HREI-MSDB (329) | +0.0225 [+0.0151, +0.0310] | +0.095 (cosine +0.022 in the isotope section) | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1), [isotope output](../findings/masster/masster_ei_evaluation.md#isotope-output-against-the-hr-references) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | RECETOX Exposome (383) | +0.0935 [+0.0759, +0.1104] | +0.196 (cosine +0.094 in the isotope section) | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1), [isotope output](../findings/masster/masster_ei_evaluation.md#isotope-output-against-the-hr-references) |
| 2026-10-18 | e5 (held out) | `eugy8s9n`@45 | RECETOX Metabolome (250) | -0.0038 [-0.0057, -0.0021] | +0.177 (cosine -0.004 in the isotope section) | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1), [isotope output](../findings/masster/masster_ei_evaluation.md#isotope-output-against-the-hr-references) |

## Held-out HR-EI by e1 training overlap (10 ppm, natural isotopes, paired e5 minus e1)

Retracted 2026-10-19: e1 is not a valid HR-EI comparison, and the finding no longer reports these rows (git history at `d57d4224`). Rows kept per ledger rules; do not cite.

| Date | Reference | e1 overlap | n | e1 (`jeljdikj`@37) | e5 (`eugy8s9n`@45) | Delta | 95% CI (2,000 bootstrap resamples) | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-18 | HREI-MSDB | no | 132 | 0.8501 | 0.8466 | -0.0036 | [-0.0137, +0.0067] | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | HREI-MSDB | yes | 197 | 0.8697 | 0.8518 | -0.0179 | [-0.0298, -0.0070] | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | Exposome | no | 163 | 0.7152 | 0.7119 | -0.0034 | [-0.0269, +0.0214] | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | Exposome | yes | 220 | 0.7844 | 0.7019 | -0.0824 | [-0.1028, -0.0622] | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | Metabolome | no | 136 | 0.5702 | 0.5467 | -0.0234 | [-0.0405, -0.0057] | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |
| 2026-10-18 | Metabolome | yes | 114 | 0.6296 | 0.6034 | -0.0262 | [-0.0481, -0.0029] | [e5 results](../findings/masster/masster_ei_evaluation.md#results-1) |

## Held-out HR-EI by molecular-ion m/z (e5 `eugy8s9n`@45, 10 ppm, natural isotopes)

Median cosine per bin (spectra in parentheses), 2026-10-18; source [by molecular-ion m/z](../findings/masster/masster_ei_evaluation.md#by-molecular-ion-mz-e5-10-ppm-isotopes-on).

| m/z bin | HREI-MSDB | Exposome | Metabolome |
| --- | --- | --- | --- |
| <150 | 0.93 (31), 0.91 (119) | 0.82 (10) | 0.82 (1) |
| 150-200 | 0.80 (102) | 0.85 (34) | 0.71 (3) |
| 200-250 | 0.82 (47) | 0.87 (69) | 0.67 (12) |
| 250-300 | 0.96 (17) | 0.77 (69) | 0.63 (41) |
| 300-400 | 0.94 (5) | 0.79 (142) | 0.64 (90) |
| 400-500 | 0.78 (8) | 0.54 (50) | 0.63 (57) |
| >=500 | — | 0.35 (9) | 0.41 (46) |

## RECETOX Metabolome diagnostics (e5 `eugy8s9n`@45, 10 ppm, natural isotopes)

| Date | Evaluation | Mean cosine | Spectra | vs comparator | Source |
| --- | --- | --- | --- | --- | --- |
| 2026-10-18 | Baseline | 0.5726 | 250 | — | [Why Metabolome is lowest](../findings/masster/masster_ei_evaluation.md#why-recetox-metabolome-is-lowest) |
| 2026-10-18 | Counterfactual: remove the measured m/z 149.045 peak | 0.5927 | 250 | vs 0.5726 baseline | [Why Metabolome is lowest](../findings/masster/masster_ei_evaluation.md#why-recetox-metabolome-is-lowest) |
| 2026-10-18 | Counterfactual: move predicted nominal-147 intensity to 147.0656 | 0.5896 | 250 | vs 0.5726 baseline | [Why Metabolome is lowest](../findings/masster/masster_ei_evaluation.md#why-recetox-metabolome-is-lowest) |
| 2026-10-18 | Counterfactual: both 149 removal and 147 move | 0.6172 | 250 | vs 0.5726 baseline | [Why Metabolome is lowest](../findings/masster/masster_ei_evaluation.md#why-recetox-metabolome-is-lowest) |
| 2026-10-18 | Nominal-control score (before exact-mass loss) | 0.638 | 250 | 10 ppm score 0.573 | [Why Metabolome is lowest](../findings/masster/masster_ei_evaluation.md#why-recetox-metabolome-is-lowest) |

| Date | Within Metabolome, by TMS groups | 0 | 1 | 2 | 3 | >=4 | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-18 | n | 12 | 43 | 71 | 52 | 72 | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |
| 2026-10-18 | e5 mean cosine, 10 ppm natural isotopes | 0.679 | 0.683 | 0.601 | 0.557 | 0.473 | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |
| 2026-10-18 | Nominal-control score minus 10 ppm score | 0.009 | 0.035 | 0.073 | 0.056 | 0.091 | [TMS vs non-TMS](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms) |

The >=4-TMS spectra score 0.806 nominal on NIST23 (see the test table) versus 0.564 nominal on RECETOX ([source](../findings/masster/masster_ei_evaluation.md#tms-vs-non-tms)).

## Library-to-library agreement and comparators (nominal linear cosine, source window)

Comparator rows are not model checkpoints except the e5 rows; 860 of 962 records (321/363/176 by source), 2026-10-18, [source](../findings/masster/masster_ei_evaluation.md#measured-hr-ei-vs-nist23-nominal-ei-for-the-same-molecules).

| Mean nominal cosine (window) | Model | HREI-MSDB | Exposome | Metabolome |
| --- | --- | --- | --- | --- |
| NIST23 replicate vs replicate | — | 0.955 (84) | 0.947 (59) | 0.947 (53) |
| HR library vs NIST23 | — | 0.904 | 0.908 | 0.759 |
| e5 vs NIST23 | `eugy8s9n`@45 | 0.934 | 0.846 | 0.882 |
| e5 vs HR library | `eugy8s9n`@45 | 0.862 | 0.778 | 0.676 |
| HR vs NIST23, without window | — | 0.884 | 0.874 | 0.750 |
| Metabolome with nominal m/z 149 removed: HR vs NIST23 / e5 vs NIST23 / e5 vs HR | `eugy8s9n`@45 | — | — | 0.794 / 0.882 / 0.704 |

## Held-out HR-EI vs chemical-space similarity (e5 `eugy8s9n`@45, 10 ppm, natural isotopes)

2026-10-18, [source](../findings/masster/masster_ei_evaluation.md#chemical-space-training-vs-references).

| Reference | Molecules | Median nearest-train Tanimoto | Share >=0.7 | Share <0.4 | Spearman (Tanimoto, e5 cosine) |
| --- | --- | --- | --- | --- | --- |
| HREI-MSDB | 278 | 0.667 | 42% | 6% | -0.02 (p 0.68) |
| Exposome | 350 | 0.640 | 30% | 9% | 0.31 (p 5e-9) |
| Metabolome | 217 | 0.706 | 52% | 3% | 0.03 (p 0.69) |
