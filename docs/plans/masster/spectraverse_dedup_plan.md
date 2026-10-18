# Spectraverse MASSter: deduplicated train

> **Status:** PREPARED · **Updated:** 2026-10-06
> Split built for cv0/cv1 (grouped by molecule and adduct), configs resolve, 16 tests pass; nothing submitted.

## Why

The curation paper removes near-duplicates: spectra with the same InChIKey first block and
polarity whose cosine exceeds 0.99. The released v1.0.2 data does not satisfy that rule
([curation note](../../notes/spectraverse_curation_paper.md)). Duplicates make the training set
look larger than it is and over-weight redistributed compounds. That is one candidate reason SPV
val peaks at epoch 15 ([oracle finding](../../findings/masster/spectraverse_oracle_ceiling_2026_10_03.md)).
This arm isolates deduplication: no merge, MS-DIAL kept, everything else as `masster_c0rerun`.

## Rule

`preproc_scripts/spectraverse/dedup_train_split.py --folds 0 1`, applying the paper's cosine rule over
all spectra:

- **Pairs: same molecule and adduct; instrument type and CE must match** (user decisions,
  2026-10-04).
  - CE is the ACE readings (primary and up to two extra), rounded to 0.1 eV.
  - A missing value (instrument "NA", or no CE) matches anything; known values must be equal.
  - A pair with identical instrument and CE is a `same` edge.
  - A pair matched only through a missing value is a `wildcard` edge. It drops the
    less-specified spectrum: more missing values, or on a tie fewer peaks. It never merges two
    labelled spectra, so a missing-value spectrum cannot bridge different instruments or CEs.
  - The paper groups by InChIKey first block and polarity. That wider key also joined pairs
    across adducts, instruments and CEs.
- **Decision cosine:** the repo's tolerance-matched `cos_hun_helper` (Hungarian, stored
  base-peak-normalized intensities) at 5 ppm with a 200 m/z floor (at least 1 mDa), > 0.99.
- **Prefilter:** binned cosine (0.01 Da bins on two half-offset grids, max of the two) > 0.95. On
  27,626 sampled pairs, every 5 ppm pair above 0.99 had a binned score of at least 0.965.
- **Clusters:** connected components of `same` edges; each keeps the spectrum with the most peaks,
  then the lowest `spec_id`. Wildcard losers are dropped on top.
- **Scope:** train only. Val/test are byte copies of the source split, and their 24,707 (cv0) /
  28,341 (cv1) near-duplicates stay.
- **Outputs:** `data/proc/spectraverse/near_dup_clusters.csv` (every spectrum) and
  `near_dup_edges.csv` (every pair, with its median |Δm/z| over matched peaks). Runtime is about
  2.5 min on 10 workers.

**Rejected variants** (2026-10-04)

| Variant | Spectra dropped | Why rejected |
| --- | ---: | --- |
| InChIKey14 + polarity, 0.01 Da binned cosine | 138,749 | also merges independent acquisitions 5–50 ppm apart |
| (molecule, adduct), 5 ppm, CE-agnostic | 125,204 | merges different CEs (20,852 pairs with known, different CE) |
| plus an identical-m/z "copy" test | 131,283 | identical m/z does not prove a copy (see below) |
| (molecule, adduct), never across known different CE, CE-stripped copies merged | 116,021 | pairs must share instrument and CE |
| (molecule, adduct, instrument, CE) strictly; missing matches only missing | 64,640 | a missing value should match anything, with the missing-value copy dropped |

The copy test failed on this pair: GNPS 355242 (no CE) and MS-DIAL 136299 (NCE 40) share all 7
m/z at 3 decimals but differ in intensity (base peak 325.142 vs 474.349, cosine 0.961). They are two
CEs of one library entry. Libraries that report one m/z list per compound give every CE the same
m/z, and 3-decimal rounding equalizes independent acquisitions.

## Result of the rule

| | Value |
| --- | --- |
| Near-duplicate pairs | 250,608 (165,541 `same`, 85,067 `wildcard`) |
| Spectra dropped, all splits | 106,019 (21.7%; 49,367 as wildcard losers) |
| cv0 train | 387,399 → 306,087 (all 33,947 molecules kept) |
| cv1 train | 386,134 → 308,456 (all 33,770 molecules kept) |
| cv0 train with known CE / NA instrument | 56.7% → 65.3% / 2.3% → 1.2% |
| Most-affected sources | MiMeDB 64%, HMDB 46%, Zheng 2024 44%, FooDB 43%, GNPS 38%, MoNA 27% |

**Invariants checked on the full run:**

- 0 pairs join two known, different instruments or CEs.
- In 0 of 85,067 wildcard edges does the loser have fewer missing values than the winner.
- 0 of 106,019 dropped spectra map to a representative with a different known instrument or CE.

Train `train_ids.csv` md5: cv0 `cc7ec07b88b8662a680b696f0726e3d3`, cv1
`74e513f16445a4cd515e358ada5514dc`. `data/` is ignored, so the split dir must be copied to Fir.

## Arm

| Dir (`config/spectraverse/…`) | Change against `masster_c0rerun/cv{0,1}.yml` |
| --- | --- |
| `masster_dedup` | `split_dp` `data/split/spectraverse_benchmark_dedup/cv{f}`; `lr_total_steps` 23,920 / 24,120 (40 × ceil(train / 512)). |

The resolved diff against the control is exactly `split_dp`, `lr_total_steps` and the W&B name,
group and tags. 40 epochs as in `masster_subceil05`, so the arm takes 79% of the control's 30,280 steps (cv0).

## Validation

- `tests/test_spectraverse_dedup.py` (5 tests): identical spectra score 1 with zero Δm/z; a
  3-decimal copy matches; a 20 ppm shift does not; the binned prefilter survives a bin edge;
  group edges keep only pairs above the threshold.
- Ruff clean.
- No training smoke: the dataset path is unchanged; only the split file differs.

## Launch (Fir, needs MFA)

1. Copy `data/split/spectraverse_benchmark_dedup/` to the Fir checkout and check the md5s.
2. `setup_job_cc.py --cv_config_dp config/spectraverse/masster_dedup --cv_start 0 --cv_end 1 --cv_use_job_id`
   on `fir:h100-3g.40gb` (12 h), next to the
   short/clean arms (removed 2026-10-18; `git show 95341fce:docs/plans/masster/spectraverse_short_clean_arms_plan.md`) and the
   merged-train arm (removed 2026-10-18; `git show 95341fce:docs/plans/masster/spectraverse_merged_train_plan.md`).
3. Record job IDs and W&B runs here.

## Read

- **Primary metric:** best-epoch `val_cos_sim_0.01_epoch/mean` per fold against
  `masster_c0rerun`, plus a last-5-epoch tail mean. Promotion needs both folds positive; the
  NIST20 seed spread is 0.0042. Compare by wall clock too, since the arm runs 79% of the control's steps.
- **Peak epoch:** does it move later than epoch 15?
- **Interactions:** the merged-train arm also collapses duplicates, by averaging within groups
  and alongside the MS-DIAL drop. If dedup alone wins, a merged arm built on the dedup split is
  the next step.
