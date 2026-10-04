# Spectraverse MASSter: merged train without MS-DIAL

> **Status:** PREPARED · **Updated:** 2026-10-03 · Code, two fold configs and the split are
> built; tests and a real-data CUDA smoke pass. Nothing has been submitted.

## Why

Each SPV (molecule, adduct, instrument) group holds 5.16 training spectra on average. Many are
cross-library duplicates, and replicates agree far above the model (0.70–0.76 vs 0.46). MS-DIAL
is the noisiest source by a wide margin ([source audit](../../findings/masster/spectraverse_source_replicate_audit_2026_10_03.md)).
Merging each group into one consensus target removes the duplicates and averages the noise.
Each step then has to learn molecule-level structure instead of fitting per-spectrum residuals.
No controlled merged-vs-unmerged MASSter comparison exists; the NIST20 merged arms are
confounded ([merged control guide](../../guides/benchmark_masster_merged_control.md)). This arm
keeps CMF and dual CE on, so only the training target and the MS-DIAL drop differ.

User decisions (2026-10-03): merge the train split only, so val/test stay per-spectrum and
comparable with the SPV benchmark and `masster_c0rerun`; drop MS-DIAL from training only.

## Arm

| Dir (`config/spectraverse/…`) | Change against `masster_c0rerun/cv{0,1}.yml` |
| --- | --- |
| `masster_merged_nomsdial` | `split_dp` `data/split/spectraverse_benchmark_nomsdial/cv{f}`; `spec_params.merge_train_only: true` with the NIST `ma_mi_merged` consensus recipe (`tolerance_consensus`, common overlap, normalize each, equal CE weight, `legacy` CE aggregation) and `Other`/`NA` merge tolerances of 20 ppm / 500 Da; `train_target_top_k_peaks: 512`; 220 epochs; `lr_total_steps` 30,360 / 30,800. |

- **Schedule.** cv0 has 70,368 merged train groups (138 steps per epoch), cv1 has 71,435 (140).
  220 epochs match the control's ~30.2k optimizer steps, about 206 passes per group. Validation
  still runs every epoch, so the arm makes 220 validation passes against the control's 40.
- **Top-512 cap.** The consensus target is the union of member peaks: p99 1,712, max 14,951 on
  cv0. The cap applies to train targets only (`eval_target_apply_train_rank_filters` is false),
  so validation scores the full per-spectrum targets.
- **CE.** Merged rows condition on their pooled CE ladder (mean-pooled embeddings). 68% of
  merged cv0 train rows include the missing-CE sentinel in their ladder.

## Code

`spec_params.merge_train_only` (template default `False`). In `BaseDataset._base_init`, the
train split is built with `merge: True` and `merge_keep_ces` set to `nce or ace`; other splits
are unchanged. It refuses `merge: True`. Precursor-pair loss and the JEPA term refuse it, since
they need unmerged train rows. Model hparams keep `merge: False`, so the result trackers and
`_check_ce_params` see an unmerged run.

`preproc_scripts/spectraverse/source_filter_split.py --folds 0 1 --drop-sources MS-DIAL` builds
the split and `data/proc/spectraverse/spec_source.csv`.

| Fold | Train before | After | Groups after | Molecules before → after |
| --- | ---: | ---: | ---: | --- |
| cv0 | 387,399 | 307,989 | 70,368 | 33,947 → 32,892 |
| cv1 | 386,134 | 300,463 | 71,435 | 33,770 → 32,621 |

Train `train_ids.csv` md5: cv0 `ea871544f36c67c77f4aac3020e134c1`, cv1
`925dd3cd4ab9b20483f6ff131b705cca`. Val and test ids are byte copies of the source split.

## Validation

- `tests/test_merge_train_only.py` (4 tests): train merges by group with the equal-CE consensus
  and keeps the ladder including the `-1` sentinel; val stays per-spectrum; the caller's
  `spec_params` is not mutated; `merge: True` is refused. With the existing merge, CE-context
  and subsample tests: 272 passed. Ruff clean on changed files.
- Both configs resolve through `load_config`. Against the control, the resolved differences are
  exactly the keys in the table above plus W&B name, group and tags.
- Real-data smoke (cv0, RTX 8 GB): `init_dataset` built the full merged train set (70,368 rows,
  `id_key` `group_id`) and a 2% per-spectrum val set (921 rows) in 308 s. A train batch holding
  the two largest merged targets (14,951 and 12,854 peaks) gave a finite loss of 1.8124 with
  160 finite gradient tensors and a 0.52 GiB peak. A val batch gave a finite loss and
  `cos_sim_0.01` / `cos_hun_inst_tol`. Script:
  `output/analysis/spectraverse_source_audit_2026_10_03/smoke_merged_nomsdial.py`.
- Not run: a full epoch or a full validation pass.

## Launch (Fir, needs MFA)

1. Copy `data/split/spectraverse_benchmark_nomsdial/` to the Fir checkout and check the md5s.
   The cluster code must include `merge_train_only`.
2. Run `setup_job_cc.py --cv_config_dp config/spectraverse/masster_merged_nomsdial --cv_start 0 --cv_end 1 --cv_use_job_id`
   on `fir:h100-3g.40gb` (12 h), next to the three [short/clean arms](spectraverse_short_clean_arms_plan.md).
   Dataset construction adds about 5 min per job.
3. Record job IDs and W&B runs here.

## Read

Primary: best-epoch `val_cos_sim_0.01_epoch/mean` per fold against `masster_c0rerun`, plus a
last-5-epoch tail mean. Promotion needs both folds positive; the NIST20 seed spread is 0.0042.
Compare by wall clock as well as by step. This arm changes two things (merge and the MS-DIAL
drop), so a win does not say which one helped. If it wins, the follow-ups are merge-only with
MS-DIAL kept, and the MS-DIAL drop without merging. Also rescore per source with
`scripts/debug/spectraverse/spectraverse_oracle_ceiling.py` joined to `spec_source.csv`, to check that the
arm does not simply give up on MS-DIAL validation spectra.
