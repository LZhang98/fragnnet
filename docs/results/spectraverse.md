# Spectraverse results

> **Status:** LEDGER · **Updated:** 2026-10-18
> Every measured model result on Spectraverse, one row per run, checkpoint, fold, split and evaluation. Numbers are copied from the linked source, which stays canonical.

Add a row when a finding or plan records a new measurement; fix a row when its source is corrected. Hub: [spectraverse](../workstreams/spectraverse.md).

## Current best

- Highest logged validation `cos_sim_0.01` on full val: cv1 `u46v0k0g`@15-18 0.465 (cv0 `ggifzj9m`@15 0.439); last epoch 0.456 / 0.430. Source: [train vs val](../findings/masster/spectraverse_masster_findings.md#train-versus-validation-cv0-added-2026-10-03).
- Pooled sampled-val (n = 2,048, last checkpoints) `cos_sim_0.01` 0.458 [0.443, 0.473] against an emitted-bin oracle of 0.734 [0.718, 0.750]. Source: [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling).
- Only runs: the historical controls `ggifzj9m` (cv0) and `u46v0k0g` (cv1), v1.0.2. No v1.0.3 model and no c0rerun/dedup/merged/abs005 arm has been trained ([hub](../workstreams/spectraverse.md#where-it-stands)).

## Comparability

- All model rows are v1.0.2 (488,797 spectra), folds = MGF `FOLD` (cv0: test 1, val 10), from two checkpoints trained with the retired `FT` instrument label ([`[M]+` finding](../findings/masster/spectraverse_masster_findings.md#old-spv-checkpoints-carry-the-retired-ft-label)). Loading them with the current config swaps the OB and NA instrument rows: recall rows were scored with that swap, so OB rows used the NA embedding and QTOF rows are unaffected ([recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance)); oracle rows use row-swapped copies with EMA weights and the current NA/ORB/Other/QTOF order ([method](../findings/masster/spectraverse_masster_findings.md#method-and-limits)).
- The controls predate `fill_missing_ace`, the ORB rename and the `cos_hun_inst_tol` checkpoint metric ([retired short/clean-arms plan](../history/retired_docs.md#spectraverse)).
- Sampled rows are last-checkpoint, not best-checkpoint, not full validation; sample seed 20261001 (recall 4,096 spectra per fold, oracle 1,024 per fold, tolerance sweeps 512 per fold). W&B `val_recall`/`val_wrecall` use flat 10 ppm / 200 Da, stricter than the instrument-tolerance loss for non-OB spectra; logging was corrected 2026-10-01 and historical W&B values were not rewritten.
- Instrument label is `ORB` in oracle rows and `OB` in recall rows.
- v1.0.3 folds ([leak audit](../findings/datasets/spectraverse_1_0_3_split_leak_audit_2026_10_18.md#result)): no molecule shared between splits under five identity keys; 3,716 cross-fold ECFP6 Tanimoto-1.0 molecule pairs (10,909 spectra, 2.3%); 3.9% / 3.5% of cv0 / cv1 test spectra have such a train partner. How much this raises scores is not measured; no row below was read against it. User decision 2026-10-18: release folds used unchanged ([hub](../workstreams/spectraverse.md#settled--read-before-re-proposing)).
- v1.0.2 folds are disjoint by 2D InChIKey but not by formula (4,481 formulas span folds); 10.6% of Morgan-2048 Tanimoto >= 0.9 analogs sit in another fold ([data version and folds](../findings/masster/spectraverse_masster_findings.md#data-version-and-folds)).

## W&B validation curve (full validation, v1.0.2)

| Date | Model / arm | Run@epoch | Fold | val `cos_sim_0.01` | val `val_recall` (flat 10 ppm) | vs comparator | Source |
|---|---|---|---|---|---|---|---|
| 2026-10-03 | masster control | `ggifzj9m`@15 | cv0 val | 0.439 | — | — | [train vs val](../findings/masster/spectraverse_masster_findings.md#train-versus-validation-cv0-added-2026-10-03) |
| 2026-10-03 | masster control | `u46v0k0g`@15-18 | cv1 val | 0.465 | — | — | [train vs val](../findings/masster/spectraverse_masster_findings.md#train-versus-validation-cv0-added-2026-10-03) |
| 2026-10-03 | masster control | `ggifzj9m`@39 | cv0 val | 0.430 | 0.4499 | — | [train vs val](../findings/masster/spectraverse_masster_findings.md#train-versus-validation-cv0-added-2026-10-03); [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-03 | masster control | `u46v0k0g`@39 | cv1 val | 0.456 | 0.4125 | — | [train vs val](../findings/masster/spectraverse_masster_findings.md#train-versus-validation-cv0-added-2026-10-03); [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |

## Oracle ceilings on sampled validation (v1.0.2, 1,024 val spectra per fold)

| Date | Model / arm | Run@epoch | Fold | n | `model` (`cos_sim_0.01`) | `opt_emit` | `cand_inst` | `cand_inst_iso` | `cand_005` | `sub_inst` | `sub_005` | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | masster control, oracle read | `ggifzj9m`@last | cv0 val | 1,024 | 0.457 | 0.737 | 0.806 | 0.827 | 0.898 | 0.881 | 0.940 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control, oracle read | `u46v0k0g`@last | cv1 val | 1,024 | 0.459 | 0.732 | 0.799 | 0.818 | 0.895 | 0.830 | 0.927 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control, oracle read | both@last | cv0+cv1 pooled val | 2,048 | 0.458 [0.443, 0.473] | 0.734 [0.718, 0.750] | 0.803 [0.788, 0.817] | 0.822 | 0.896 | 0.855 [0.843, 0.869] | 0.934 | `model` reaches 62% of `opt_emit` | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |

## Oracle strata (same read, pooled by subset)

| Date | Model / arm | Run@epoch | Fold | Stratum | n | `model` | `opt_emit` | `cand_005` | `sub_inst` | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | `sub_inst` <= 0.5 | 287 | 0.037 | 0.127 | 0.467 | 0.141 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | `sub_inst` 0.5-0.9 | 142 | 0.204 | 0.497 | 0.779 | 0.721 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | `sub_inst` 0.9-0.99 | 227 | 0.443 | 0.766 | 0.954 | 0.961 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | `sub_inst` > 0.99 | 1,392 | 0.574 | 0.879 | 0.987 | 0.999 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | `sub_inst` > 0.5 (86% of spectra) | — | 0.527 | 0.833 | — | — | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | instrument ORB | 1,154 | 0.502 | 0.771 | — | 0.839 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | instrument QTOF | 750 | 0.419 | 0.699 | — | 0.888 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | instrument NA | 85 | 0.475 | 0.893 | — | 0.976 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | instrument Other | 59 | 0.084 | 0.242 | — | 0.584 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | adduct `[M+H]+` | 1,041 | 0.488 | 0.782 | — | 0.883 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | adduct `[M-H]-` | 471 | 0.538 | 0.809 | — | 0.924 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | adduct `[M+Na]+` | 238 | 0.208 | 0.404 | — | 0.590 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | adduct `[M+NH4]+` | 137 | 0.461 | 0.771 | — | 0.895 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | adduct `[M+K]+` | 46 | 0.117 | 0.340 | — | 0.489 | — | [oracle](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling) |

## Train versus validation (cv0, v1.0.2, 1,024 sampled spectra)

| Date | Model / arm | Run@epoch | Fold | Split | `model` | `opt_emit` | `model` / `opt_emit` | `model`, `sub_inst` > 0.99 | `model`, `sub_inst` <= 0.5 | `sub_inst` | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | masster control | `ggifzj9m`@last | cv0 | train | 0.590 | 0.706 | 0.84 | 0.719 | 0.050 | 0.871 | [train vs val](../findings/masster/spectraverse_masster_findings.md#train-versus-validation-cv0-added-2026-10-03); [MSGYM comparison](../findings/masster/spectraverse_masster_findings.md#massspecgym-comparison-added-2026-10-03) |
| 2026-10-03 | masster control | `ggifzj9m`@last | cv0 | val | 0.457 | 0.737 | 0.62 | 0.555 | 0.029 | 0.881 | [train vs val](../findings/masster/spectraverse_masster_findings.md#train-versus-validation-cv0-added-2026-10-03); [MSGYM comparison](../findings/masster/spectraverse_masster_findings.md#massspecgym-comparison-added-2026-10-03) |

## MassSpecGym comparator in the oracle script (1,024 sampled spectra per split)

| Date | Model / arm | Run@epoch | Fold | Split | `model` | `opt_emit` | `model` / `opt_emit` | `sub_inst` | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | MSGYM MASSter | `ll04i00m`@60 | MSGYM | train | 0.884 | 0.952 | 0.93 | 0.997 | SPV cv0 train 0.590 | [MSGYM comparison](../findings/masster/spectraverse_masster_findings.md#massspecgym-comparison-added-2026-10-03) |
| 2026-10-03 | MSGYM MASSter | `ll04i00m`@60 | MSGYM | val | 0.584 | 0.933 | 0.63 | 0.997 | SPV cv0 val 0.457 | [MSGYM comparison](../findings/masster/spectraverse_masster_findings.md#massspecgym-comparison-added-2026-10-03) |
| 2026-10-03 | MSGYM MASSter, ORB | `ll04i00m`@60 | MSGYM | train | 0.920 | — | — | — | — | [MSGYM comparison](../findings/masster/spectraverse_masster_findings.md#massspecgym-comparison-added-2026-10-03) |
| 2026-10-03 | MSGYM MASSter, ORB | `ll04i00m`@60 | MSGYM | val | 0.605 | — | — | — | — | [MSGYM comparison](../findings/masster/spectraverse_masster_findings.md#massspecgym-comparison-added-2026-10-03) |
| 2026-10-03 | MSGYM MASSter, QTOF | `ll04i00m`@60 | MSGYM | train | 0.816 | — | — | — | — | [MSGYM comparison](../findings/masster/spectraverse_masster_findings.md#massspecgym-comparison-added-2026-10-03) |
| 2026-10-03 | MSGYM MASSter, QTOF | `ll04i00m`@60 | MSGYM | val | 0.484 | — | — | — | — | [MSGYM comparison](../findings/masster/spectraverse_masster_findings.md#massspecgym-comparison-added-2026-10-03) |

## Recall rescored at instrument tolerance (4,096 sampled val spectra per fold)

| Date | Model / arm | Run@epoch | Fold | Subset (n) | Peak recall, flat 10 ppm | Peak recall, instrument tolerance | Intensity-weighted recall, flat 10 ppm | Intensity-weighted recall, instrument tolerance | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | masster control, recall rescore | `ggifzj9m`@last | cv0 val | all (4,096) | 0.4619 | 0.5419 | 0.6415 | 0.7068 | W&B full-val last-epoch `val_recall` 0.4499 | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `u46v0k0g`@last | cv1 val | all (4,096) | 0.4098 | 0.5399 | 0.6184 | 0.7302 | W&B full-val last-epoch `val_recall` 0.4125 | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `ggifzj9m`@last | cv0 val | QTOF (1,339) | 0.2817 | 0.5139 | — | — | — | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `u46v0k0g`@last | cv1 val | QTOF (1,663) | 0.3985 | 0.6935 | — | — | — | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `ggifzj9m`@last | cv0 val | OB (n not stated); wrong instrument row | 0.5597 | 0.5597 | — | — | same tolerance in both reads | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `u46v0k0g`@last | cv1 val | OB (n not stated); wrong instrument row | 0.4202 | 0.4202 | — | — | same tolerance in both reads | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `ggifzj9m`@last | cv0 val | Other (73) | — | 0.1129 | — | — | — | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `u46v0k0g`@last | cv1 val | Other (156) | — | 0.1752 | — | — | — | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `ggifzj9m`@last | cv0 val | NA (116) | — | 0.7417 | — | — | tolerance not stated for this value | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |
| 2026-10-01 | masster control, recall rescore | `u46v0k0g`@last | cv1 val | NA (210) | — | 0.7726 | — | — | tolerance not stated for this value | [recall](../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-recall-at-instrument-tolerance) |

## Wider-window diagnostic with shifted-target control (512 val spectra per fold)

| Date | Model / arm | Run@epoch | Fold | Instrument (n), window | Candidate recall | Candidate wRecall | Shifted-target recall | Shifted-target wRecall | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | masster control, window sweep | both@last | cv0+cv1 pooled val (512 per fold) | OB (567), current 10 ppm / 200 Da | 0.521 | 0.733 | 0.011 | 0.009 | shifted target = target m/z + 0.1 Da | [wider window](../findings/masster/spectraverse_masster_findings.md#wider-window-diagnostic-2026-10-01) |
| 2026-10-01 | masster control, window sweep | both@last | cv0+cv1 pooled val (512 per fold) | OB (567), 20 ppm / 500 Da | 0.610 | 0.773 | 0.053 | 0.043 | shifted target = target m/z + 0.1 Da | [wider window](../findings/masster/spectraverse_masster_findings.md#wider-window-diagnostic-2026-10-01) |
| 2026-10-01 | masster control, window sweep | both@last | cv0+cv1 pooled val (512 per fold) | OB (567), 50 ppm / 750 Da | 0.745 | 0.865 | 0.142 | 0.119 | shifted target = target m/z + 0.1 Da | [wider window](../findings/masster/spectraverse_masster_findings.md#wider-window-diagnostic-2026-10-01) |
| 2026-10-01 | masster control, window sweep | both@last | cv0+cv1 pooled val (512 per fold) | QTOF (375), current 20 ppm / 750 Da | 0.603 | 0.735 | 0.061 | 0.053 | shifted target = target m/z + 0.1 Da | [wider window](../findings/masster/spectraverse_masster_findings.md#wider-window-diagnostic-2026-10-01) |
| 2026-10-01 | masster control, window sweep | both@last | cv0+cv1 pooled val (512 per fold) | QTOF (375), 30 ppm / 750 Da | 0.645 | 0.766 | 0.080 | 0.068 | shifted target = target m/z + 0.1 Da | [wider window](../findings/masster/spectraverse_masster_findings.md#wider-window-diagnostic-2026-10-01) |
| 2026-10-01 | masster control, window sweep | both@last | cv0+cv1 pooled val (512 per fold) | QTOF (375), 50 ppm / 750 Da | 0.692 | 0.803 | 0.126 | 0.110 | shifted target = target m/z + 0.1 Da | [wider window](../findings/masster/spectraverse_masster_findings.md#wider-window-diagnostic-2026-10-01) |

## Fixed absolute-window candidate recall (512 val spectra per fold)

| Date | Model / arm | Run@epoch | Fold | Subset | Instrument-tol. cand. recall | Instrument-tol. cand. wRecall | ±0.1 Da cand. recall | ±0.1 Da cand. wRecall | ±0.05 Da cand. recall | ±0.05 Da cand. wRecall | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | masster control, fixed-window read | `ggifzj9m`@last | cv0 val (512) | all | 0.551 | 0.716 | 0.822 | 0.889 | 0.7589 | 0.8443 | ±0.1 Da on the same spectra 0.8218 / 0.8890 (±0.05 Da source) | [±0.1 Da](../findings/masster/spectraverse_masster_findings.md#fixed-01-da-diagnostic-2026-10-01); [±0.05 Da](../findings/masster/spectraverse_masster_findings.md#fixed-005-da-read-2026-10-01) |
| 2026-10-01 | masster control, fixed-window read | `u46v0k0g`@last | cv1 val (512) | all | 0.546 | 0.733 | 0.780 | 0.894 | 0.7145 | 0.8463 | ±0.1 Da on the same spectra 0.7803 / 0.8944 (±0.05 Da source) | [±0.1 Da](../findings/masster/spectraverse_masster_findings.md#fixed-01-da-diagnostic-2026-10-01); [±0.05 Da](../findings/masster/spectraverse_masster_findings.md#fixed-005-da-read-2026-10-01) |
| 2026-10-01 | masster control, fixed-window read | `ggifzj9m`@last | cv0 val (512) | OB (wrong instrument row) | 0.586 | — | 0.884 | — | 0.8337 | 0.9117 | — | [±0.1 Da](../findings/masster/spectraverse_masster_findings.md#fixed-01-da-diagnostic-2026-10-01); [±0.05 Da](../findings/masster/spectraverse_masster_findings.md#fixed-005-da-read-2026-10-01) |
| 2026-10-01 | masster control, fixed-window read | `u46v0k0g`@last | cv1 val (512) | OB (wrong instrument row) | 0.433 | — | 0.742 | — | 0.6721 | 0.8323 | — | [±0.1 Da](../findings/masster/spectraverse_masster_findings.md#fixed-01-da-diagnostic-2026-10-01); [±0.05 Da](../findings/masster/spectraverse_masster_findings.md#fixed-005-da-read-2026-10-01) |
| 2026-10-01 | masster control, fixed-window read | `ggifzj9m`@last | cv0 val (512) | QTOF | 0.494 | — | 0.712 | — | 0.6289 | 0.7244 | — | [±0.1 Da](../findings/masster/spectraverse_masster_findings.md#fixed-01-da-diagnostic-2026-10-01); [±0.05 Da](../findings/masster/spectraverse_masster_findings.md#fixed-005-da-read-2026-10-01) |
| 2026-10-01 | masster control, fixed-window read | `u46v0k0g`@last | cv1 val (512) | QTOF | 0.691 | — | 0.825 | — | 0.7756 | 0.8816 | — | [±0.1 Da](../findings/masster/spectraverse_masster_findings.md#fixed-01-da-diagnostic-2026-10-01); [±0.05 Da](../findings/masster/spectraverse_masster_findings.md#fixed-005-da-read-2026-10-01) |

Control: +0.5 Da shifted-target peak recall is 0.026 / 0.032 (fold order not stated), a weak null because it changes mass defect ([source](../findings/masster/spectraverse_masster_findings.md#fixed-01-da-diagnostic-2026-10-01)).

## Candidate recall by adduct (4,096 val spectra per fold, cv0 + cv1, instrument tolerance)

| Date | Model / arm | Run@epoch | Fold | Adduct | n | Mean intensity-weighted candidate recall | Median | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | masster control, recall rescore | both@last | cv0+cv1 pooled val | `[M+H]+` | 4,080 | 0.761 | 0.925 | OB rows scored with NA embedding | [`[M]+` census](../findings/masster/spectraverse_masster_findings.md#spv-m-census) |
| 2026-10-06 | masster control, recall rescore | both@last | cv0+cv1 pooled val | `[M-H]-` | 1,947 | 0.809 | 0.923 | OB rows scored with NA embedding | [`[M]+` census](../findings/masster/spectraverse_masster_findings.md#spv-m-census) |
| 2026-10-06 | masster control, recall rescore | both@last | cv0+cv1 pooled val | `[M+Na]+` | 988 | 0.388 | 0.283 | OB rows scored with NA embedding | [`[M]+` census](../findings/masster/spectraverse_masster_findings.md#spv-m-census) |
| 2026-10-06 | masster control, recall rescore | both@last | cv0+cv1 pooled val | `[M+NH4]+` | 559 | 0.725 | 0.856 | OB rows scored with NA embedding | [`[M]+` census](../findings/masster/spectraverse_masster_findings.md#spv-m-census) |
| 2026-10-06 | masster control, recall rescore | both@last | cv0+cv1 pooled val | `[M]+` | 101 | 0.874 | 0.939 | OB rows scored with NA embedding | [`[M]+` census](../findings/masster/spectraverse_masster_findings.md#spv-m-census) |

## Model score re-grouped by source (oracle sample, 1,024 val spectra per fold)

| Date | Model / arm | Run@epoch | Fold | Subset | `cos_sim_0.01` | vs comparator | Source |
|---|---|---|---|---|---|---|---|
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | all sampled validation | 0.458 | — | [source audit](../findings/masster/spectraverse_masster_findings.md#spectraverse-sources-replicates-and-merge-shape) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | without MS-DIAL | 0.508 | all 0.458 | [source audit](../findings/masster/spectraverse_masster_findings.md#spectraverse-sources-replicates-and-merge-shape) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | `[M+Na]+` inside MS-DIAL | 0.120 | outside MS-DIAL 0.318 | [source audit](../findings/masster/spectraverse_masster_findings.md#spectraverse-sources-replicates-and-merge-shape) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | `[M+Na]+` outside MS-DIAL | 0.318 | inside MS-DIAL 0.120 | [source audit](../findings/masster/spectraverse_masster_findings.md#spectraverse-sources-replicates-and-merge-shape) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | CE-missing spectra | 0.461 | CE-bearing 0.456 | [source audit](../findings/masster/spectraverse_masster_findings.md#spectraverse-sources-replicates-and-merge-shape) |
| 2026-10-03 | masster control | both@last | cv0+cv1 pooled val | CE-bearing spectra | 0.456 | CE-missing 0.461 | [source audit](../findings/masster/spectraverse_masster_findings.md#spectraverse-sources-replicates-and-merge-shape) |

## Replicate agreement (same-molecule comparator, not a model; cv0 train groups, 4,000 sampled groups with >= 3 spectra)

| Date | Model / arm | Run@epoch | Fold | Comparator | n | Mean cosine (0.01 Da, raw intensity) | Median | vs comparator | Source |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | replicate agreement | — | cv0 train groups | mean of other replicates, any CE | 4,000 | 0.700 | 0.795 | model 0.458 | [replicates](../findings/masster/spectraverse_masster_findings.md#replicates) |
| 2026-10-03 | replicate agreement | — | cv0 train groups | nearest replicate, any CE | 4,000 | 0.842 | 0.954 | model 0.458 | [replicates](../findings/masster/spectraverse_masster_findings.md#replicates) |
| 2026-10-03 | replicate agreement | — | cv0 train groups | mean of replicates within ±3 eV | 1,476 | 0.761 | 0.872 | model 0.458 | [replicates](../findings/masster/spectraverse_masster_findings.md#replicates) |
| 2026-10-03 | replicate agreement | — | cv0 train groups | same-source pair within ±3 eV | 1,275 | 0.689 | 0.775 | model 0.458 | [replicates](../findings/masster/spectraverse_masster_findings.md#replicates) |
| 2026-10-03 | replicate agreement | — | cv0 train groups | cross-source pair within ±3 eV | 276 | 0.849 | 0.994 | model 0.458 | [replicates](../findings/masster/spectraverse_masster_findings.md#replicates) |

## Isotope-envelope ceiling (model-free: perfect monoisotopic model plus best scalar envelope)

| Date | Model / arm | Run@epoch | Fold | Subset | Cosine recovered | vs comparator | Source |
|---|---|---|---|---|---|---|---|
| 2026-10-03 | isotope-envelope gate (oracle) | — | v1.0.2, all, positive | Spectraverse (488,797 spectra) | 0.0042 | gate 0.005 | [retired isotope plan](../history/retired_docs.md#masster_generalization) |
| 2026-10-03 | isotope-envelope gate (oracle) | — | v1.0.2, QTOF positive | Spectraverse | 0.0074 | gate 0.005 | [retired isotope plan](../history/retired_docs.md#masster_generalization) |
