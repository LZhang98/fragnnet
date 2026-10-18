# MASSter on NIST23 EI (nominal mass)

> **Status:** HUB · **Updated:** 2026-10-18
> MASSter on the NIST23 electron-ionization library (`[M]+`, 70 eV, unit resolution, natural isotope envelopes): plan, performance audit, screens.

## Where it stands

- **Held-out HR-EI result (e5):** [finding](../findings/masster/masster_ei_evaluation.md#held-out-e5-ei-checkpoint-on-measured-hr-ei-spectra): e5 `eugy8s9n`@45, trained on the [HR-EI-excluded split](../plans/masster/hr_ei_isotopes_holdout_plan.md). It is lower than e1 only on molecules e1 trained on (memorization); HREI-MSDB and Exposome are unchanged on unseen molecules. Same molecules: e5 matches NIST23 better than the HR libraries do on HREI-MSDB/Metabolome (0.934/0.882 vs 0.904/0.759), so most of the HR gap is the difference between libraries, a systematic m/z intensity bias. Exposome is real model error. Metabolome also has a 149.045 artifact and Si2 ions outside the candidate space. Next: decide on reference cleaning; e5 seed.
- **Frozen checkpoint HR-EI evaluation:** [finding](../findings/masster/masster_ei_evaluation.md#frozen-masster-ei-checkpoint-on-measured-hr-ei-spectra): all mapping-consistent records included regardless of training overlap; exact monoisotopic outputs evaluated. Exact isotope output is now implemented; see the preparation plan above.
- **HR-EI raw audit complete:** [peak/structure/overlap findings](../findings/masster/masster_ei_evaluation.md#hr-ei-raw-peak-and-structure-audit). All four peak tables parse; HREI-MSDB/Exposome are first mapping candidates. Metadata defects, derivative handling, and local training overlap require exclusions; no final benchmark yet.
- **Best result:** e1 (natural isotopes), [W&B `jeljdikj`](https://wandb.ai/frag-gnn/frag-gnn/runs/jeljdikj) epoch 37, `config/masster/ei/e1_nominal_natural_isotopes.yml`: full `nist23_ei_inchikey` test (51,685 spectra) mean 1 Da cosine 0.869984, Stein weighted similarity 0.886845; 0.899167 on the 26,372 RASSP-eligible spectra (RASSP FormulaNet reports 0.929 on its own NIST17 set, descriptive only).
- **Isotope effect (one seed):** mean `val_cos_sim_1.0` over epochs 54-63 is 0.867738 (e1) vs 0.859883 (e0 `he3j5y3o`, no isotopes), +0.0079.
- **Where the gap is:** a 512-spectrum support oracle scores 0.99615 vs the model's 0.88703, so the gap is intensity allocation, not candidate support. Output power/mass tilts do not help. Stein falls from 0.9202 (2-16 atoms) to 0.8556 (>64).
- **Prepared, not launched:** matched `e2` (fresh control), `e3` (full attention), `e4` (EMA 0.999), W&B group `nist23_ei_intensity_2026_10_01`, same seed 397623, 64 epochs.
- **Uncommitted/unverified:** local isotope-offset and SHR-parsing audit fixes are not verified in the remote e0/e1 runs (no Git revision in remote metadata). No submitter stage for `config/masster/ei/`; the data may not be on Fir.
- **Next action:** record source provenance, check full-attention batch-512 memory, then run e2/e3/e4; add seeds before claiming the isotope effect. Then a matched RASSP comparison (same split, same Stein).

## Open threads

| Thread | Status | Next concrete step |
| --- | --- | --- |
| [HR-EI-excluded fresh training](../plans/masster/hr_ei_isotopes_holdout_plan.md) | RAN, read ([finding](../findings/masster/masster_ei_evaluation.md#held-out-e5-ei-checkpoint-on-measured-hr-ei-spectra)) | Recheck Metabolome overlap against native parents; e5 seed replicate. Castro identities remain unresolved |
| [Matched intensity arms e2/e3/e4](../plans/masster/masster_ei_nominal_plan.md#matched-intensity-configs-2026-10-01-prepared-not-launched) | PREPARED | Launch with provenance; compare cosine and Stein by size and on the RASSP subset |
| [Matched RASSP comparison](../findings/masster/masster_ei_evaluation.md#masster-ei-performance-audit) | OPEN | Check training overlap and preprocessing; run both models on one eligible split |
| Mass-aware intensity objective | PROPOSED | Test only if matched arms leave a Stein gap (audit priority 3) |
| H-shift width +/-6 (trained ablation) | OPEN | None until a trained-width ablation is wanted; keep +/-4 |
| NEIMS baseline, EI library metrics (composite cosine, retrieval) | NOT BUILT | Build NEIMS on `nist23_ei_inchikey` at 1 Da |
| Large-molecule errors | OPEN | Larger val sample with size/chemistry strata before changing candidates, H shift or SMARTS |

## Settled — read before re-proposing

- **Setup decisions (user):** nominal `nist23_ei`, not the SHR set; select on `val_cos_sim_1.0`; model isotopes; `loss_tolerance_abs` 0.5 ([plan](../plans/masster/masster_ei_nominal_plan.md)).
- EI SMARTS screen (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_ei_smarts_screen_2026_10_01.md`) — CLOSED NULL: anisoyl fluoride and para-hydroxyazobenzene add zero coverage; sulfonamide O-transfer +2.414% on seven train molecules, no val examples.
- [H-shift screen](../plans/masster/masster_ei_nominal_plan.md#ei-h-transfer-width-screen-2026-09-30-session-retain-4-for-now) — keep +/-4: L1 coverage 0.9040 at +/-2, 0.9339 at +/-3, 0.9482 at +/-4; +/-6 adds 1.31 points and 0.00314 cosine ceiling.
- [e1 annotation read](../plans/masster/masster_ei_nominal_plan.md#e1-checkpoint-annotation-read-2026-10-01-4-used-6-unresolved) — H+/-4 carries 3.10% of predicted intensity; dropping it costs 0.0010 cosine.
- [Fragment-count penalty](../plans/masster/masster_ei_nominal_plan.md#fragment-count-penalty-decision-2026-10-01-do-not-add-yet) — do not add: the loss cannot see which fragment explains a peak.
- [Test metrics by size](../plans/masster/masster_ei_nominal_plan.md#full-test-performance-by-molecule-size-2026-10-01) — unweighted cosine nearly flat by size; Stein declines. >48 atoms is extrapolation, not a RASSP-matched subset.
- [Output tilts](../findings/masster/masster_ei_evaluation.md#masster-ei-performance-audit) — peak powers 0.6-1.4 and m/z^alpha tilts all score below 0.88703.
- Related MASSter conventions live in [MASSter core](masster_core.md); EI SMARTS Enveda per-class gate is in [MASSter generalization](masster_generalization.md).
- **Retired 2026-10-18:** 1 document of this workstream removed; summaries and open items in [retired documents](../history/retired_docs.md#nist23_ei).

## All documents

| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [EI evaluation](../findings/masster/masster_ei_evaluation.md) | FINDING (2026-10-18, merged) | Four parts: NIST23 `e1` performance audit (Stein vs RASSP, support oracle); HR-EI raw audit; frozen `e1` on HR-EI; held-out `e5` on 962 HR-EI spectra split by `e1` training overlap (headline) |
| [Exact isotopes and HR-EI exclusion](../plans/masster/hr_ei_isotopes_holdout_plan.md) | PREPARED (2026-10-17) | Natural exact-mass output, conservative excluded split, and fresh e5 recipe |
| [MASSter EI-MS method](../reference/masster_ei_method.md) | REFERENCE (2026-10-16) | Probabilistic EI formulation, nominal channels, natural isotope assembly, and interpretation limits |
| [Plan: MASSter on NIST23 EI](../plans/masster/masster_ei_nominal_plan.md) | ACTIVE (2026-10-06) | Stage 0-1, W&B completion, audits, H-shift and SMARTS screens, size tables, e2-e4 configs |

## HR-EI raw downloads (2026-10-16)

Original publisher files are stored separately under `data/raw/hr_ei_ms/`:

| Source | Subdirectory | Records in main spectral file | Files and license |
| --- | --- | ---: | --- |
| [HREI-MSDB](https://doi.org/10.6084/m9.figshare.29713460) | `hrei_msdb/` | 329 | All 15 deposited files: MSP, SDF, ZIP archives, compound/quality spreadsheets, and information PDF; CC BY 4.0 |
| [RECETOX Exposome](https://doi.org/10.5281/zenodo.4471217) | `recetox_exposome/` | 386 | MSP spectra and SDF structures; CC-BY-NC per provider |
| [RECETOX Metabolome](https://doi.org/10.5281/zenodo.5483565) | `recetox_metabolome/` | 265 | MSP spectra and SDF structures; CC-BY-NC per provider |
| [Castro GC-QTOF](https://zenodo.org/records/5647960) | `castro_gc_qtof/` | 356 | JCAMP and PCDL libraries; CC BY 4.0 in Zenodo API metadata |

RECETOX files were downloaded from the [provider's reference-library page](https://www.recetox.muni.cz/en/services/recetox-central-laboratories/biomarker-analysis-laboratories/recetox-mass-spectrum-reference-library).
The inventory is 21 payload files totaling 21,798,880 bytes; record counts are
MSP `Name:` headers or JCAMP `##TITLE=` headers, not deduplicated molecule counts.
HREI-MSDB subset files and archives overlap its combined library and must not be
counted as additional independent spectra.

`data/raw/hr_ei_ms/download_manifest_2026_10_16.json` records file URLs, sizes,
SHA-256 hashes, publisher MD5 hashes where supplied, and licenses. Figshare and
Zenodo API metadata are saved alongside their libraries. All 21 local file hashes
were verified, all 17 publisher-supplied MD5 hashes matched, and all seven ZIP
archives passed CRC checks. Existing top-level RECETOX MSP files were preserved.
No preprocessing, archive extraction, structure validation, or training was performed.

Before using these data, map derivatized species to their measured structures:
RECETOX Metabolome uses methoximation/silylation, and Castro includes TMS derivatives
whose CAS identifiers can describe the original compound. Audit overlap both across
libraries and against NIST23 nominal-EI training before defining held-out evaluation.

## Key artifacts

- W&B: e1 `jeljdikj` (best ep 37), e0 `he3j5y3o` (best 0.861611 at ep 57), e5 `eugy8s9n` (best ep 45, val 0.86896; `saved_ckpts/e5_hr_ei_holdout_nominal_eugy8s9n.ckpt`).
- Configs: `config/masster/ei/{e0_nominal_no_isotopes,e1_nominal_natural_isotopes,e2_nominal_control,e3_nominal_full_attention,e4_nominal_ema999}.yml`.
- Data: `data/proc/nist23_ei/`, `data/split/nist23_ei_inchikey` (154,981 / 51,700 / 51,685), raw `data/df/nist23_ei_df.csv`.
- Outputs (ignored): `output/masster_ei_jeljdikj_analysis_2026_10_01/` (`test_all_stein/size_performance_stein_bins.csv`, `sample512_cosine/`), `output/masster_ei_smarts_2026_10_01/`, `output/masster_ei_nominal_2026_09_30/`.
- Scripts: `scripts/debug/ei/{debug_masster_ei_nominal_coverage,debug_masster_ei_smarts_screen,analyze_masster_ei_checkpoint,plot_masster_ei_size_performance}.py`; test `tests/test_debug_masster_ei_smarts_screen.py`.
