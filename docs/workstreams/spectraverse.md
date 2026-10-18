# Spectraverse (SPV) workstream

> **Status:** HUB · **Updated:** 2026-10-18
> MASSter on Spectraverse v1.0.2 (488,797 spectra, ten MCES folds): pipeline, data audits, filter screens, and the prepared training arms. v1.0.3 (464,228 spectra) is preprocessed through the benchmark folds; nothing trained on it.

## Where it stands

- **v1.0.3 preprocessed 2026-10-18** (commands, counts and md5s in the [preproc README](../../preproc_scripts/README.md#spectraverse-v103)): `data/df_spectraverse_1_0_3/`, `data/proc/spectraverse_1_0_3/`, `data/split/spectraverse_1_0_3_benchmark/cv0-cv9` (built from the MGF, md5 in each `stats.json`). v1.0.3 `TITLE` numbers are sparse, so the split and source scripts now key MGF records on `TITLE` number, and all `preproc_scripts/spectraverse/` defaults point at v1.0.3 (v1.0.2 rebuilds need explicit paths). No config points at it yet, and no dedup or other train-side variant has been built. [Leak audit](../findings/spectraverse_1_0_3_split_leak_audit_2026_10_18.md): no molecule shared between splits under five identity keys; lipid analogs (ECFP6 Tanimoto 1.0) cross folds, 3.9%/3.5% of cv0/cv1 test spectra.
- **No new SPV result since the oracle read.** Historical controls (`config/masster/spectraverse/cv0-cv9`, cv0 `ggifzj9m`, cv1 `u46v0k0g`) reach `cos_sim_0.01` 0.458 on 2,048 sampled val spectra; val peaks at epoch 15 and declines to epoch 39.
- **Why it is stuck** ([oracle ceiling](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling)): the model reaches 62% of its own emitted-bin ceiling (0.734), so intensity allocation is the largest gap (0.276); 14% of spectra have a subformula ceiling <= 0.5 and score 0.04. Train cosine is 0.590, so the gap is mostly generalization.
- **Data causes** ([source audit](../findings/masster/spectraverse_masster_findings.md#spectraverse-sources-replicates-and-merge-shape)): MS-DIAL (20% of spectra) is 46% of the unexplainable block; 33% of spectra have a cosine > 0.99 copy; replicates agree at 0.70-0.76 vs the model's 0.46.
- **Prepared, nothing submitted (all cv0/cv1, Fir `fir:h100-3g.40gb`, 12 h):** `masster_c0rerun` (fresh control) and `masster_dedup`; the split dir `data/split/spectraverse_benchmark_dedup` is local-only and must be copied to Fir. The `short18`/`subceil05`, `merged_nomsdial` and `abs005` plans were retired 2026-10-18 (configs and split dirs remain; see Settled).
- **Pipeline** (2026-10-06, committed in `c5bf610a`): `01_prepare_df.py` reads v1.0.2 and `build_benchmark_split.py` rebuilds all 30 fold files.
- **Next action:** copy the three split dirs to Fir and launch the five arms next to the control; read each against `masster_c0rerun` (best-epoch `val_cos_sim_0.01_epoch/mean`, last-5 tail; seed spread 0.0042; both folds positive).

## Open threads

| Thread | Status | Next concrete step |
| --- | --- | --- |
| [Dedup train](../plans/masster/spectraverse_dedup_plan.md) | PREPARED | Copy `spectraverse_benchmark_dedup/`, check md5s, launch; if it wins, build merged on the dedup split |
| Recall window calibration | OPEN | Calibrate on trusted fragment assignments plus shifted-mass controls before changing any loss window ([recall finding](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance)) |
| v1.0.3 training | OPEN | Point a config at `data/proc/spectraverse_1_0_3` and `data/split/spectraverse_1_0_3_benchmark`; decide which train-side variants to rebuild |
| v1.0.2 `ion_mode` labels | OPEN, harmless so far | Stored `data/proc/spectraverse/spec_df.pkl` labels 30,550 `[M+Cl]-`/`[M+HCOOH-H]-`/`[M+CH3COOH-H]-` spectra `P` (reader fixed 2026-10-18); only the `--ion_modes` split filter reads it. Rebuild or patch before filtering v1.0.2 by ion mode |
| Quaternary-ammonium candidate gap (m/z 60.0808) | OPEN, small (0.26% of spectra) | Decide on a `valence` bound change ([`[M]+` finding](../findings/masster/spectraverse_masster_findings.md#masster-m-handling-on-spectraverse)) |

## Settled — read before re-proposing

- **Folds and data**
  - [v1.0.3 leak audit](../findings/spectraverse_1_0_3_split_leak_audit_2026_10_18.md) — no identity leak in any cv (mol_id, 2D InChIKey, MGF InChIKey, standardized parent); 2.3% of spectra have a cross-fold ECFP6 Tanimoto-1.0 analog, 86% lipids/long chains, concentrated in folds 2-3.
  - **User decision (2026-10-18):** the leak criterion is InChIKey identity; v1.0.3 passes, so the release folds are used unchanged and cross-fold analogs are neither removed nor reported separately.
  - [Folds are the MGF `FOLD` field](../findings/masster/spectraverse_masster_findings.md#data-version-and-folds) — MCES split, v1.0.2 md5-verified; old `data/df` fold CSVs are v1.0.0; `mol_id`/`group_id` columns are stale, join on `spec_id`.
  - [Curation paper note](../notes/spectraverse_curation_paper.md) — the paper's cosine > 0.99 near-duplicate rule does not hold in released v1.0.2.
  - [Spurious evaluations note](../notes/spurious_evaluations_spectraverse.md) — spectrum-blind baseline gets 26.9% top-1 on our split; not a clean target.
- **Filters**
  - Subformula filter screen (removed 2026-10-18; `git show 731fe493:docs/findings/masster/spectraverse_subformula_filter_2026_10_01.md`) — STOPPED: removes 26.2% of intensity, 15.2% of base peaks, empties 15,074 spectra; 10x window still misses 7.5%.
  - [Na/K read](../findings/masster/spectraverse_masster_findings.md#why-spectraverse-mk-and-mna-spectra-are-unexplained) — `[M+K]+` is a potassium/background problem (82% lack the precursor); `[M+Na]+` failures cluster in a few source blocks.
- **Handling and metrics**
  - [`[M]+` handling](../findings/masster/spectraverse_masster_findings.md#masster-m-handling-on-spectraverse) — always a pre-charged cation (3,405 spectra, 457 molecules); `/tmp/spv_recall_ckpts` and old SPV ckpts carry the retired `FT` label.
  - [Recall at instrument tolerance](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) — flat 10 ppm to instrument tolerance lifts peak recall 0.462/0.410 to 0.542/0.540; +/-0.1 Da gives 0.822/0.780 but is not a calibrated window.
- **Ceiling**
  - [Oracle ceiling](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) — 0.458 vs 0.734 emitted-bin, 0.803 instrument-tolerance, 0.855 all-subformula; isotope rows add <= 0.019. MSGYM has the same 63% efficiency but a flat val curve.
- **Retired 2026-10-18:** 3 documents of this workstream removed; summaries and open items in [retired documents](../history/retired_docs.md#spectraverse).
- Short-schedule and clean-train arms — retired 2026-10-18 while PREPARED (`git show 95341fce:docs/plans/masster/spectraverse_short_clean_arms_plan.md`); if revived: copy `spectraverse_benchmark_subceil05/`, launch 3 arms x cv0/cv1.
- Merged train, no MS-DIAL — retired 2026-10-18 while PREPARED (`git show 95341fce:docs/plans/masster/spectraverse_merged_train_plan.md`); if revived: copy `spectraverse_benchmark_nomsdial/` (cluster code needs `merge_train_only`), launch; if it wins, split merge-only vs MS-DIAL-drop.
- Fixed +/-0.05 Da arm — retired 2026-10-18 while PREPARED (`git show 95341fce:docs/plans/masster/spectraverse_abs005_training_plan.md`); if revived: record launch provenance, compare cv0/cv1 before other folds.
- **User decisions (2026-10-03):** merge the TRAIN split only (val/test stay per-spectrum) and drop MS-DIAL from training only; `merge_train_only` lives in `BaseDataset._base_init` (`tests/test_merge_train_only.py`) and must be on the Fir checkout. A new arm uses fixed +/-0.05 Da ([recall finding](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance)).

## All documents

| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [Dedup plan](../plans/masster/spectraverse_dedup_plan.md) | PREPARED (2026-10-06) | Drops 106,019 spectra (21.7%); cv0 train 387,399 to 306,087 |
| [Spectraverse MASSter findings](../findings/masster/spectraverse_masster_findings.md) | FINDING (2026-10-18, merged) | Five parts: recall at instrument tolerance; `[M]+` pre-charged cations only; oracle ceiling (where the cosine gap sits); sources/MS-DIAL/duplicates/folds and reproducible pipeline; why Na/K spectra are unexplained |
| [v1.0.3 leak audit](../findings/spectraverse_1_0_3_split_leak_audit_2026_10_18.md) | FINDING (2026-10-18) | No identity leak across splits; cross-fold lipid analogs per cv |
| [Curation paper note](../notes/spectraverse_curation_paper.md) | NOTE | Release comparison v1.0.0-1.0.2, paper filters |
| [Spurious evaluations note](../notes/spurious_evaluations_spectraverse.md) | NOTE | Benchmark pitfalls, MCES folds |
| [Preproc README](../../preproc_scripts/README.md) | REFERENCE | Commands for the reproducible SPV pipeline |

## Key artifacts

- Data: `data/raw/spectraverse/` (v1.0.2 MGF md5 `50803b37...ef6895`; as of 2026-10-18 only `spectraverse-1.0.3/` is on the workstation, md5 `52ed27db...a004aa`), `data/proc/spectraverse/` (`spec_source.csv`, `subformula_ceiling.csv`), `data/split/spectraverse_benchmark{,_dedup,_nomsdial,_subceil05}/`. v1.0.3: `data/{df_spectraverse_1_0_3,proc/spectraverse_1_0_3,split/spectraverse_1_0_3_benchmark}/`. Do not join `data/df/spectraverse_df.csv` (different build).
- Dedup train md5: cv0 `cc7ec07b...`, cv1 `74e513f1...`; short/clean: cv0 `43749a16...`, cv1 `cb46ac1c...`.
- Configs: `config/spectraverse/{masster,masster_c0rerun,masster_short18,masster_subceil05,masster_merged_nomsdial,masster_dedup,masster_abs005}/`.
- Scripts: `preproc_scripts/spectraverse/{build_benchmark_split,clean_train_split,dedup_train_split,source_filter_split}.py`; `scripts/debug/spectraverse/{spectraverse_leak_check,spectraverse_oracle_ceiling,spectraverse_data_check,analyze_spectraverse_recall}.py`; tests `tests/test_spectraverse_{mgf_reader,benchmark_split,dedup}.py`.
- Neighbours: [MASSter core](masster_core.md), [MASSter generalization (`i1`, ORB instrument findings)](masster_generalization.md), [MSGYM](msgym.md), [FraGNNet](fragnnet.md).
