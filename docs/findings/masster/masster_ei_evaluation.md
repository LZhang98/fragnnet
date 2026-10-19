# MASSter EI checkpoints: NIST23 audit and measured HR-EI evaluation

> **Status:** FINDING · **Updated:** 2026-10-19
> Four EI evaluation findings: the `e1` NIST23 performance audit, the HR-EI raw-data audit, frozen `e1` on HR-EI (superseded: not a valid HR-EI comparison), and held-out `e5` on HR-EI (the headline).

Merged 2026-10-18 from 4 findings on one topic. Each part below is the original
document unchanged except for heading levels; its own status line keeps its original date.
Exception, 2026-10-19 (user decision): `e1` is not a valid comparison for HR-EI, so the frozen
`e1` part is marked `SUPERSEDED` and every `e1` comparison was removed from the `e5` part.
Former file names are listed so old links and `git log -- <former path>` can be traced.

| Part | Updated | Verdict | Former file |
| --- | --- | --- | --- |
| [MASSter EI performance audit](#masster-ei-performance-audit) | 2026-10-06 | `e1` `jeljdikj`@37: Stein 0.8868 on NIST23 test; RASSP comparison unrun | `masster_ei_performance_audit_2026_10_01.md` |
| [HR-EI raw peak and structure audit](#hr-ei-raw-peak-and-structure-audit) | 2026-10-17 | All four HR-EI libraries have fractional-mass peaks; structure mappings need source-specific exclusions | `hr_ei_raw_audit_2026_10_17.md` |
| [Frozen MASSter EI checkpoint on measured HR-EI spectra](#frozen-masster-ei-checkpoint-on-measured-hr-ei-spectra) | 2026-10-19 | SUPERSEDED: not a valid HR-EI comparison (`e1` trained on most of these molecules); kept as an implementation record | `hr_ei_checkpoint_eval_2026_10_17.md` |
| [Held-out e5 EI checkpoint on measured HR-EI spectra](#held-out-e5-ei-checkpoint-on-measured-hr-ei-spectra) | 2026-10-19 | Held-out `e5`, 10 ppm, natural isotopes: HREI-MSDB 0.850, Exposome 0.706, Metabolome 0.573 | `hr_ei_e5_holdout_eval_2026_10_18.md` |

## MASSter EI performance audit

> **Status:** FINDING · **Updated:** 2026-10-06
> Checkpoint and inference audit complete (e1 `jeljdikj`@37: Stein 0.886845, RASSP-eligible subset 0.899167); proposed training comparisons remain unrun.

### Scope and comparison

The natural-isotope `e1` checkpoint, [W&B `jeljdikj` epoch 37](https://wandb.ai/frag-gnn/frag-gnn/runs/jeljdikj), scores **0.886845 mean Stein weighted similarity** and **0.869983 mean 1 Da cosine** on all 51,685 `nist23_ei_inchikey` test spectra. The 26,372 test spectra passing RASSP's published NIST17 filters (H/C/N/O/F/P/S/Cl, at most 48 **total** atoms including H, at most 4096 formulae) score **0.899167 mean Stein**. The [RASSP paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9909676/) reports **0.929 average Stein** for FormulaNet on its NIST17 dataset. The 0.030 difference is descriptive: training overlap, dataset, preprocessing and split are unmatched, and RASSP has not been run on our test set. See the [full test score and filter record](../../plans/masster/masster_ei_nominal_plan.md#stein-weighted-similarity-on-the-full-test-split-2026-10-01).

Stein similarity uses the [RASSP metric](https://github.com/thejonaslab/rassp-public/blob/main/rassp/metrics.py): cosine after transforming each nominal peak to `m/z**3 * intensity**0.6`. On the full test set, mean Stein falls from **0.9202** at 2–16 total atoms to **0.8556** above 64 atoms; see the [size bin table](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/size_performance_stein_bins.csv). The published FormulaNet inference limit is 48 total atoms, so the >48 groups describe our model's extrapolation and do not form a RASSP matched subset.

### What limits the current score

On a 512-spectrum validation sample, an oracle that keeps the model's predicted nominal m/z support and assigns the measured intensities scores **0.99615 mean Stein**, versus the model's **0.88703**. Mean target Stein weighted squared norm covered by predicted m/z is **0.99321**. In the >64-total-atom sample group (45 spectra), model/oracle Stein is **0.8234/0.9902**. This oracle is an upper bound using the answer at inference; it establishes available m/z support, not attainable model performance. The leading observed gap is intensity allocation among mostly available masses.

The current EI recipe trains with normalized-target cross entropy and selects its best checkpoint by unweighted 1 Da cosine. Neither directly optimizes Stein's mass-weighted peak vector. On the same validation sample, post-hoc peak powers 0.6, 0.8, 1.2 and 1.4 all score below the unmodified prediction's **0.88703** Stein; multiplying intensities by `m/z**alpha` for alpha −1, −0.5, +0.5 or +1 also scores lower. A global output rescaling is therefore unsupported by this check. Sample target Stein weighted norm is concentrated at m/z at least half the precursor (**0.93334**), so a learned, mass-aware ranking objective is worth testing, but no improvement has been measured.

Test molecules with replicate spectra yield **0.91889 mean pairwise Stein** across 2,245 pairs; the model's mean on those replicate-bearing spectra is **0.85360**. Replicates can differ in acquisition conditions, and this is a selected subset; the pairwise number is context for attainable agreement, not a universal noise ceiling.

### Priority actions

1. **Establish a matched RASSP comparison.** Check training overlap and preprocessing, then run both models on one held-out, RASSP-eligible split with the same Stein implementation. Until then, do not interpret the published 0.929 as a head-to-head loss.
2. **Use the prepared intensity experiments.** Compare the fresh `e2` control with `e3` full candidate attention and `e4` EMA 0.999 on the same validation split, recording both 1 Da cosine and Stein, by size and on the RASSP-eligible subset. The [configs and controls](../../plans/masster/masster_ei_nominal_plan.md#matched-intensity-configs-2026-10-01-prepared-not-launched) are prepared but untrained. Record source provenance first because the original checkpoint's remote source revision is unavailable.
3. **If the matched arms leave a Stein gap, test a mass-aware intensity objective** against the fresh control, selecting and reporting both metrics on validation. The sample's failed output tilts do not test this training hypothesis. Preserve a separate final test read.
4. **Investigate large-molecule errors with a larger validation sample** and matched size/chemistry strata before altering candidate count, H-shift width or SMARTS rules. The current support oracle and prior EI SMARTS screen do not justify those changes for the mean score.

### Artifacts and limits

Full test scores and replicate pairs: `output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/spectrum_rows.csv` and `audit_replicate_pairs.csv`. Validation support-oracle rows and predicted peaks: `output/masster_ei_jeljdikj_analysis_2026_10_01/sample512_cosine/audit_stein_ceiling_rows.csv` and `peak_rows.csv.gz`. These outputs are local ignored artifacts; the 512-spectrum sample is diagnostic, and no uncertainty interval or replicated seed has been established. Five individual test predictions differ by over 0.01 cosine between two full inference runs, though their mean differs by only 4.2e-7; the individual discrepancy remains unresolved.

## HR-EI raw peak and structure audit

> **Status:** FINDING · **Updated:** 2026-10-17
> All four libraries contain fractional-mass peak tables; supplied structure mappings require source-specific exclusions before a prediction benchmark.

### Scope and reproduction

Audit the four downloaded primary libraries without modifying raw files or producing
training datasets. Check peak parsing, declared counts, positive finite values,
SMILES validity, elemental formula agreement, InChIKey agreement, SDF parsing,
cross-library connectivity overlap, and overlap with the local nominal-EI training split.

Run from the repository root:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/ei/audit_hr_ei_raw.py --output-dir output/hr_ei_raw_audit_2026_10_17/verified
```

The output directory must not exist. A rerun requires a fresh subdirectory.
The script is [audit_hr_ei_raw.py](../../../scripts/debug/ei/audit_hr_ei_raw.py).
Source checkout: `7b16308c67892f8298f0c37865d0c9eb6a0fd11b` plus the new audit script;
the repository also had unrelated UMAP edits during the audit. RDKit and input
identities are recorded in `output/hr_ei_raw_audit_2026_10_17/verified/provenance.json`.
The raw download inventory and licenses remain in the
[EI hub](../../workstreams/nist23_ei.md#hr-ei-raw-downloads-2026-10-16).

### Observed peak and mapping checks

| Library | Spectra | Parseable supplied SMILES | Consistent mapping candidates | Candidates outside nominal train |
| --- | ---: | ---: | ---: | ---: |
| HREI-MSDB | 329 | 329 | 329 | 132 spectra / 113 connectivity keys |
| RECETOX Exposome | 386 | 386 | 383 | 163 spectra / 150 connectivity keys |
| RECETOX Metabolome | 265 | 257 | 250 | 136 spectra / 116 connectivity keys |
| Castro GC-QTOF | 356 | 0 in JCAMP | unresolved | unknown |

All 1,336 spectra have at least one fractional m/z peak, positive finite peak
coordinates/intensities, and agreement with their declared peak count. Of 77,283
peak rows, 77,281 differ from the nearest integer by more than 1e-6 Da.
This confirms fractional mass data in the downloaded files; it does not independently
measure instrument resolving power, calibration accuracy, or impurity levels.

“Consistent mapping candidates” requires a parseable SMILES, matching elemental
formula, matching first InChIKey block, and no isotope-labelled atoms. It is a
metadata-consistency screen, not experimental verification of molecular identity.
Stereo-only key differences are retained but must be considered before a stereo-specific
benchmark. The combined candidate pool has 825 distinct connectivity keys before
any benchmark selection or chemical-domain filtering.

Record-level evidence is in `records.csv`; aggregate counts are in `summary.csv`.
Both are under `output/hr_ei_raw_audit_2026_10_17/verified/`.

### Source-specific issues

- **HREI-MSDB:** all 329 MSP SMILES agree with the supplied formulas and full InChIKeys.
  Its SDF has 329 record delimiters but lacks `M  END` terminators and RDKit parses
  none of its molecular records. Use the MSP SMILES for mapping; the raw SDF is preserved.
- **RECETOX Exposome:** the SDF contains 369 parseable structures, whereas the MSP has
  386 spectra. Map using spectrum-level SMILES rather than positional SDF alignment.
  Two spectra are isotope-labelled and excluded from the simple formula screen.
  `Phosmet` (zero-based MSP record 218) has SMILES formula `C11H12NO4PS2`, but its
  supplied formula is `C10H7NO4` and its supplied connectivity identifier disagrees.
  Of 42 full-key mismatches, 41 retain the same connectivity block.
- **RECETOX Metabolome:** the SDF contains 163 native structures; it must not replace
  derivative SMILES from the 265-spectrum MSP. Eight MSP records lack SMILES.
  Among the 257 parseable SMILES, 250 contain silicon. Three supplied formulas
  disagree with their SMILES: `Tryptophan_2TMS` (71), `Arginine_3TMS` (121), and
  `Cystine_4TMS` (141). Five computed connectivity blocks disagree with supplied
  identifiers: records 27, 47, 71, 127, and 163. Their union removes seven mapped
  records from the consistency pool. All indices are zero-based and names/values
  are retained in `records.csv`; no chemical corrections were inferred.
- **Castro:** all 356 JCAMP tables parse, but they provide names, CAS identifiers, and
  formulas rather than SMILES/InChIKeys. There are 117 records whose names match the
  derivative-name heuristic (`TMS`, `trimethylsilyl`, or `methoxim`); this is not an
  independently verified derivative count. Structure mapping remains unresolved.
  The PCDL file was preserved but not decoded in this audit.

SDF record-level evidence is in `sdf_records.csv` under the same output directory.

### Overlap and benchmark implications

Using computed SMILES connectivity keys, 197 HREI-MSDB spectra, 222 Exposome spectra,
and 115 mapped Metabolome spectra overlap the local nominal-EI training split.
Castro overlap is unknown, not zero. Twenty connectivity keys occur in more than
one mapped library; `cross_library_overlap.csv` lists the implicated records.
Within-source repeated connectivity keys can include stereoisomers or multiple
derivative forms and are not automatically redundant experimental measurements.

The audit joins `data/split/nist23_ei_inchikey/train_ids.csv` to
`data/proc/nist23_ei/mol_df.pkl` by `mol_id` and uses stored `inchikey_s` identifiers.
The current molecule table has 253,755 rows, differing from the older 253,649-molecule
count in the EI design record. Input hashes identify the local files used; the audit
does not establish that they are identical to the inputs of the remote e0/e1 runs.
“Outside train” therefore means absent from this local split, not proven absent from
every training corpus or free of close structural analogues.

Use HREI-MSDB and Exposome as the first mapping candidates. Resolve flagged records,
deduplicate evaluation by a declared molecular identity policy, and separate
unseen-molecule prediction from accurate-mass prediction for training-seen molecules.
Metabolome requires derivative-aware handling; Castro requires structure resolution.
No spectra have yet been declared a final benchmark, preprocessed, or used for training.

### Validation

Four focused tests pass for MSP annotations/record boundaries, JCAMP multiple
pairs/records, malformed peak rejection, and formula-order comparison. Ruff checks
pass for the audit script and tests. The complete audit exits successfully and
accounts for every primary record. These checks validate the audit mechanics;
they do not certify experimental mass accuracy or chemical identity.

## Frozen MASSter EI checkpoint on measured HR-EI spectra

> **Status:** SUPERSEDED · **Updated:** 2026-10-19
> Not a valid comparison for HR-EI (user decision 2026-10-19). `e1` trained on the NIST23 spectra of 531 of these 962 records' molecules (raw-audit flag), and possibly on native parents of others, so its scores are not a held-out HR-EI measurement. Replaced by [held-out e5](#held-out-e5-ei-checkpoint-on-measured-hr-ei-spectra). Kept as the record of the scorer and exact-isotope implementation checks; do not cite these numbers as model performance or compare them with `e5`.

### Question and setup

Can the current nominal-trained EI model predict accurate-mass spectra reasonably
well without retraining? The user explicitly requested inclusion regardless of
nominal-training overlap. We evaluated the mapping-consistent pool from the
[raw audit](#hr-ei-raw-peak-and-structure-audit): 329 HREI-MSDB, 383 RECETOX Exposome,
and 250 RECETOX Metabolome spectra. Castro remains excluded because its JCAMP
records have no resolved structures. Isotope-labelled and inconsistent mappings
remain excluded. Counts are spectrum records, not independent unique molecules;
cross-library overlap and stereoisomer/repeat records are retained.

The model is e1, W&B `jeljdikj`, epoch 37:
`saved_ckpts/e1_nominal_natural_isotopes_jeljdikj/ckpt/model-epoch=037.ckpt`, SHA-256
`b281a13a892fa5e3e4ed6a268c06284b1166a79dc60893e3546713d205b53831`.
All trained weights loaded strictly. Saved hyperparameters override current template
defaults; missing newer settings are supplied by the template. Retired `OB`/`OB_F`
keys in two metadata dictionaries are renamed in memory before migration.
Inference uses CPU, four torch threads, one joblib worker, seed 20261017,
batch size four, and the saved 256-fragment target/4,096-mask cap. The checkpoint
file and raw data are unchanged. Unrelated UMAP edits in the working tree were left alone.

Two inference settings change: mass mode becomes `exact` and isotope envelope becomes
`none`. The existing natural-envelope implementation requires nominal mass mode.
Consequently this is a test of exact **monoisotopic channel predictions**, not a
complete high-resolution isotope-envelope simulation. Reference isotope peaks are
retained and can lower similarity. Linear intensity transform (`none`) is retained.
No targets are used for model inference; raw measured intensities are scored externally.

### Results

Mean linear-intensity cosine with maximum-product one-to-one peak assignment:

| Reference | Spectra | 5 ppm | 10 ppm | 20 ppm | Rounded-mass control |
| --- | ---: | ---: | ---: | ---: | ---: |
| HREI-MSDB | 329 | 0.8390 | 0.8391 | 0.8392 | 0.8491 |
| RECETOX Exposome | 383 | 0.6114 | 0.6549 | 0.6592 | 0.6902 |
| RECETOX Metabolome | 250 | 0.6009 | 0.6009 | 0.6025 | 0.6517 |

The 5/10/20 ppm settings have absolute floors of 0.0005/0.001/0.002 Da respectively.
Matched contributions at the same mass are first aggregated using a four-decimal
mass key; the intensity-weighted mass is retained for tolerance matching. Cosine
norms include unmatched predicted and reference peaks. All positive predictions are
retained, without a relative-intensity cutoff. HREI-MSDB predictions are restricted
to its declared acquisition window, 40–500 m/z. Other predictions use 0–1500 m/z;
their acquisition cutoffs were not inferred from individual target peak lists.

The rounded-mass control rounds these same monoisotopic predictions and measured
masses to integers and sums contributions. It is not a separate forward pass in
the original nominal/natural-isotope configuration. Numerical comparisons admit
1e-10 Da roundoff so aggregated integer masses match reliably.

At 5 ppm, median cosine is 0.8662, 0.6997, and 0.6263 respectively. About 69.3%,
36.6%, and 28.4% of spectra reach cosine >=0.8. Mean measured intensity with any
predicted supporting mass is 87.65%, 61.12%, and 57.94%; this support metric does
not assess whether predicted intensities are correct.

HREI-MSDB shows little degradation from the rounded control at 5 ppm. Exposome
improves by 0.0435 from the first to second tolerance setting, indicating sensitivity
to mass matching; both the ppm threshold and absolute floor change. This does not
isolate calibration, chemistry, or instrument effects. The remaining source gaps
can involve intensity allocation, missing isotope envelopes, different acquisition
windows, derivative mapping, and domain differences. This experiment does not
separate those explanations or establish unseen-molecule generalization.

### Outputs and reproduction

Canonical scored results are in `output/hr_ei_checkpoint_eval_2026_10_17/scored/`:

- `scores.csv`: all 962 per-spectrum scores, names, and local training-overlap flags.
- `metric_summary.csv`: source counts, means, medians, and intensity recall.
- `hr_vs_nominal.png`: paired scores and tolerance sensitivity.
- `median_examples.png` and `plotted_examples.csv`: one median-scoring example per source.
- `cache/exact_predictions.npz`: exact-mass prediction arrays.
- `selected_records.csv`, `resolved_config.json`, `provenance.json`: identities and settings.

The frozen-weight forward pass took approximately 149 seconds on CPU. Original
inference inputs and its uncorrected nominal-control scores remain under
`output/hr_ei_checkpoint_eval_2026_10_17/monoisotopic/`; use `scored/` for conclusions.
The original control used zero tolerance on intensity-weighted integer masses and
missed some HREI-MSDB peaks because of floating-point roundoff. Rescoring corrected
that issue; HR scores were unchanged. A regression test covers this failure.

`output/hr_ei_checkpoint_eval_2026_10_17/run_record.json` records the forward command,
Git revision, dirty-diff identity, and script/template hashes. Primary MSP hashes
were checked against the audit provenance. Regenerate into fresh output directories:

```bash
FRAGNNET_JOBLIB_N_JOBS=1 MPLCONFIGDIR=/tmp/fragnnet_mpl \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m scripts.debug.ei.evaluate_hr_ei_checkpoint \
  --output output/hr_ei_checkpoint_eval_2026_10_17/rerun
MPLCONFIGDIR=/tmp/fragnnet_mpl FRAGNNET_JOBLIB_N_JOBS=1 \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m scripts.debug.ei.plot_hr_ei_checkpoint_eval \
  output/hr_ei_checkpoint_eval_2026_10_17/rerun
```

The scorer also accepts `--rescore-from` to evaluate saved predictions into a fresh
directory. Source scripts are
[evaluate_hr_ei_checkpoint.py](../../../scripts/debug/ei/evaluate_hr_ei_checkpoint.py)
and [plot_hr_ei_checkpoint_eval.py](../../../scripts/debug/ei/plot_hr_ei_checkpoint_eval.py).

### Validation and next check

Five focused scoring tests pass: aggregation, nominal-vs-exact discrimination,
one-to-one assignment with unmatched normalization, ppm/floor thresholds, and
integer-mass roundoff. Ruff checks pass. All 962 selected spectra survive dataset
preparation and receive scores; plots were visually inspected.

### Follow-up: natural exact-mass isotope output

The [exact isotope/exclusion implementation](../../plans/masster/hr_ei_isotopes_holdout_plan.md)
now supports natural exact-mass fine structure. The same frozen e1 weights were
evaluated on all 962 reference records with `--isotope-envelope natural`; the
other inference/scoring settings remain as above. This is still a contaminated
implementation baseline, not a held-out result for the new excluded split.

| Reference | 5 ppm, monoisotopic | 5 ppm, natural exact | 10 ppm, natural exact | 20 ppm, natural exact |
| --- | ---: | ---: | ---: | ---: |
| HREI-MSDB | 0.8390 | 0.8617 | 0.8618 | 0.8619 |
| RECETOX Exposome | 0.6114 | 0.6804 | 0.7549 | 0.7615 |
| RECETOX Metabolome | 0.6009 | 0.5971 | 0.5973 | 0.5986 |

Natural isotope output improves mean 5 ppm cosine by 0.0227 on HREI-MSDB and
0.0690 on Exposome, but decreases it by 0.0037 on Metabolome. It therefore does
not establish a uniform improvement across sources. Isotope fine structure is
emitted before any instrument-specific peak broadening or unresolved-peak centroiding.

Outputs, score summaries, prediction arrays, and plots are under
`output/hr_ei_checkpoint_eval_2026_10_17/natural_exact/`, using the same filenames
as `scored/`. Its `provenance.json` records the natural-envelope setting and
approximately 392-second CPU evaluation runtime. `implementation_record.json` adds the command,
Git/dirty-diff identity, and hashes of the implementation files and template.
All 962 spectra completed successfully with strict checkpoint loading. The
original monoisotopic results remain unchanged.

Next: train e5 from scratch on the HR-EI-excluded split, then repeat this comparison
with that fresh model. Nominal isotope offsets must not simply be added as integer
shifts to exact masses.

## Held-out e5 EI checkpoint on measured HR-EI spectra

> **Status:** FINDING · **Updated:** 2026-10-19
> e5 (trained without the HR-EI identities) is the HR-EI result: at 10 ppm with natural isotopes, mean cosine HREI-MSDB 0.850, Exposome 0.706, Metabolome 0.573. `e1` is not a valid comparison and was removed 2026-10-19.

### Setup

Model: e5 `e5_hr_ei_holdout_nominal`, [W&B `eugy8s9n`](https://wandb.ai/frag-gnn/frag-gnn/runs/eugy8s9n),
best epoch 45 (val `cos_sim_1.0` 0.86896; the only `model-epoch` file on the run), trained
from scratch on `data/split/nist23_ei_inchikey_hr_ei_excluded_2026_10_17`
([plan](../../plans/masster/hr_ei_isotopes_holdout_plan.md)). Local checkpoint
`saved_ckpts/e5_hr_ei_holdout_nominal_eugy8s9n.ckpt`, SHA-256
`307f89467f60e708f344b38804604e395987665f3a31954dd3ceba79a66eb4cd`.

Evaluation uses the record set, scorer, seed, and mass windows described in the
[frozen e1 evaluation](#frozen-masster-ei-checkpoint-on-measured-hr-ei-spectra): 962 mapping-consistent records,
at Git `731fe493`, on the local RTX 2070 SUPER (CUDA candidates).
Exact mass mode, with natural fine-structure isotopes and also monoisotopic output.
All 962 records are absent from e5's training, validation, and internal test partitions.

### Results

Headline tolerance is 10 ppm with a 1 mDa floor (user decision 2026-10-18). Isotopes are natural exact-mass
fine structure (`masster_isotope_envelope: natural`, order 4, minimum ratio 0.001) unless stated.

| Reference | Spectra | e5 mean | e5 median | e5 monoisotopic | e5 5 ppm |
| --- | ---: | ---: | ---: | ---: | ---: |
| HREI-MSDB | 329 | 0.8497 | 0.8712 | 0.8272 | 0.8495 |
| RECETOX Exposome | 383 | 0.7062 | 0.7772 | 0.6127 | 0.6315 |
| RECETOX Metabolome | 250 | 0.5726 | 0.5743 | 0.5764 | 0.5724 |

Isotopes on minus off for e5 at 10 ppm, paired: HREI-MSDB +0.0225 [+0.0151, +0.0310], Exposome
+0.0935 [+0.0759, +0.1104], Metabolome −0.0038 [−0.0057, −0.0021]. Mean measured intensity with a
predicted match (10 ppm, isotopes on) is 0.973, 0.844 and 0.768.

A paired e5 − e1 comparison split by `e1` training overlap was removed 2026-10-19 (user
decision: `e1` is not a valid HR-EI comparison); it is in git history at `d57d4224`.

#### By molecular-ion m/z (e5, 10 ppm, isotopes on)

`output/hr_ei_e5_eugy8s9n_eval_2026_10_18/natural_exact/mz_bin_sina_10ppm.{png,csv}`, from
[plot_hr_ei_mz_sina.py](../../../scripts/debug/ei/plot_hr_ei_mz_sina.py). Median cosine per bin:

| m/z bin | HREI-MSDB | Exposome | Metabolome |
| --- | --- | --- | --- |
| <150 | 0.93 (31), 0.91 (119) | 0.82 (10) | 0.82 (1) |
| 150–200 | 0.80 (102) | 0.85 (34) | 0.71 (3) |
| 200–250 | 0.82 (47) | 0.87 (69) | 0.67 (12) |
| 250–300 | 0.96 (17) | 0.77 (69) | 0.63 (41) |
| 300–400 | 0.94 (5) | 0.79 (142) | 0.64 (90) |
| 400–500 | 0.78 (8) | 0.54 (50) | 0.63 (57) |
| ≥500 | — | 0.35 (9) | 0.41 (46) |

Above m/z 400, every source drops. Mass explains part of the source gap: the median Metabolome molecular
ion is about 400, versus about 167 for HREI-MSDB. Mass does not explain all of it, because Metabolome is
0.13–0.20 below Exposome in each bin from 200 to 400. Exposome has a low tail (cosine <0.3) in every
bin from 200 to 500; those records have not been examined.

#### Why RECETOX Metabolome is lowest

243 of the 250 Metabolome records are TMS derivatives (GC-Orbitrap, methoximation/silylation).
At 10 ppm, the model leaves 23% of measured intensity unmatched, versus 3% on HREI-MSDB, and 45%
of predicted intensity has no measured match. Three causes are measured; the counterfactual scores
change only the stated ions:

1. **m/z 149.045 (C4H13O2Si2+) is much stronger than in NIST23.** It is ≥10% of base in 145 of 250 spectra
   and the base peak in 12%. The 176 records whose connectivity is in NIST23 have it at mean 30.5%
   of base (median 17%), while the NIST23 spectra of the same molecules have mean 3.5%. It is the base peak in
   the glucose, mannose, galactose, succinic and maleic acid TMS spectra. Removing that one measured peak raises
   the mean from 0.5726 to 0.5927. RECETOX annotates it as [C4H13O2Si2]+ (Me2Si=O+–SiMe2OH), so it
   is measured signal, not contamination. It equals 147 + H2O − CH4, and 147 is weaker in RECETOX than in
   NIST23 (mean 0.125 vs 0.225 of base). An ion–molecule reaction with residual water during C-trap
   storage would explain both (hypothesis, not tested).
2. **m/z 147 gets the wrong composition.** The measured ion is 147.0656 (C5H15OSi2+, the TMS-migration ion
   Me3Si–O=SiMe2+). It contains two Si and no backbone carbon, so it is not a connected substructure and
   the bond-cut candidates cannot generate it. The nominal-trained model instead puts that intensity at
   147.0836 (C6H15O2Si+), a reachable ion with the same nominal mass. That scores at 1 Da but misses at 10 ppm,
   and it explains part of the nominal-to-HR drop (0.638 to 0.573). Moving the predicted nominal-147 intensity
   to 147.0656 gives 0.5896; doing both 1 and 2 gives 0.6172.
3. **m/z 73 (Me3Si+) is over-allocated.** Pooled predicted share is 13.6% versus 7.0% measured; the
   mass is right, so this is intensity allocation.

The remaining gap to Exposome (0.706) also includes larger molecules (median molecular ion about
400, see the bins above). The 147 composition problem would need Si–O–Si migration candidates; the 149
ion looks instrument-specific, so a NIST-trained model will not learn it. Diagnostic script:
`/tmp/claude-1002/-home-feiw-gitRepos-frag-gnn/f9050d06-e4a4-4b6e-92ba-b61bdf7696c3/scratchpad/metabolome_diag.py`
(session scratch; numbers above are reproduced inline).

#### Isotope output against the HR references

e5 emits natural exact-mass fine structure at inference, computed analytically from each predicted ion
composition (order ≤4, share ≥0.001; no learned isotope parameters). At 10 ppm it raises intensity recall by
0.095/0.196/0.177 and cosine by +0.022/+0.094/−0.004 (HREI-MSDB/Exposome/Metabolome). On Metabolome,
cosine falls slightly in the >0.5 nominal-cosine strata and only 42% of spectra improve.

- **Instrument resolution is not the cause.** Merging predicted peaks closer than an Orbitrap FWHM
  (R = 30k–120k at m/z 200, scaled by √(200/m)) changes 10 ppm scores by ≤0.001 (5 ppm by ≤0.007).
  Merging within 3–8 mDa onto the strongest peak changes scores by ≤0.002.
- **The Metabolome reference isotope peaks look incomplete or merged.** In the raw Alanine_2TMS record, one
  peak at 74.04636 (the 29Si mass of Me3Si+) holds 8.7% of m/z 73. That is about the summed expected 29Si
  (≈5.1%) and 13C (≈3.3%) shares, and there is no peak at the 13C mass (74.0502), although a 60k Orbitrap
  resolves 3.8 mDa at m/z 74. Pooled over Metabolome, the 30Si M+2 of Me3Si+ (75.0437) carries 0.018% of
  measured intensity versus 0.56% predicted. Isotope output therefore moves intensity onto peaks
  the reference does not report.
- **Exposome masses are offset.** Its peak annotations list theoretical masses about 7 ppm above the
  reported ones (Anthracene: 74.0151 vs 74.01565, 75.02296 vs 75.023475). That explains its 5→10 ppm
  sensitivity.

An isotope pattern is only as right as the composition it comes from. Where the composition is wrong (m/z 147 above),
the pattern is wrong too. Scripts: session scratch `resolution_merge.py`; ratio checks are inline.

#### TMS vs non-TMS

A TMS group is matched by SMARTS `[O,N,S][Si]([CH3])([CH3])[CH3]`, counted per molecule. TMS molecules
make up 2.7% of e5 training (4,224 of 154,206 spectra; 2,031 with ≥2 TMS groups).

**NIST23 internal test (nominal, 1 Da).** Scored with
[analyze_masster_ei_checkpoint.py](../../../scripts/debug/ei/analyze_masster_ei_checkpoint.py), which now
has `--config` (e5 config, so e5's own test partition) and `--spec-ids`. The set is all 1,436 TMS test
spectra plus a seeded sample of 6,000 non-TMS test spectra; output is in `output/ei_tms_vs_non_tms_e5_2026_10_18/`.

| TMS groups | Spectra | Mean cosine | Mean Stein | Median exact mass |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6,000 | 0.868 | 0.889 | 281 |
| 1 | 696 | 0.846 | 0.865 | 280 |
| 2 | 429 | 0.817 | 0.815 | 340 |
| 3 | 165 | 0.794 | 0.772 | 429 |
| ≥4 | 146 | 0.806 | 0.708 | 627 |

TMS spectra are heavier. Reweighting non-TMS spectra to the TMS mass distribution (six exact-mass bins)
gives a TMS penalty of −0.035 cosine [−0.046, −0.024] and −0.044 Stein [−0.055, −0.034] (500 bootstrap
resamples). The penalty grows with mass: within a bin, TMS minus non-TMS is −0.005 at 150–250, −0.028 at 250–350,
−0.051 at 350–450 and −0.093 at 450–600.

**HR-EI references (10 ppm, natural isotopes).** Every TMS record is in Metabolome, so source is confounded.
Within Metabolome, scores by TMS count 0/1/2/3/≥4 are 0.679/0.683/0.601/0.557/0.473 (n 12/43/71/52/72).
The nominal-control score minus the 10 ppm score is 0.009/0.035/0.073/0.056/0.091, so the exact-mass loss
grows once two TMS groups can form the Si2 ions. On the same model, ≥4-TMS spectra score 0.806 nominal on
NIST23 versus 0.564 nominal on RECETOX. Most of the Metabolome deficit is therefore specific to these
references and their exact masses, not to TMS chemistry alone.

#### Measured HR-EI vs NIST23 nominal EI for the same molecules

[compare_nist_hr_ei_libraries.py](../../../scripts/debug/ei/compare_nist_hr_ei_libraries.py) matches each
HR record to NIST23 EI spectra with the same 2D connectivity: 860 of 962 records (321/363/176 by source).
e5 never trained on these NIST spectra. All cosines are nominal and linear, with HR spectra and exact
predictions rounded and summed. Everything is cut to the source window: 40–500 for HREI-MSDB, and ≥67 for
Exposome and ≥70 for Metabolome, which are the lowest peaks seen in any record of those sources. Means are
over the NIST spectra of each molecule. Output is in `output/hr_ei_vs_nist23_2026_10_18/` (`pairs.csv`, `summary.csv`).

| Mean nominal cosine (window) | HREI-MSDB | Exposome | Metabolome |
| --- | ---: | ---: | ---: |
| NIST23 replicate vs replicate | 0.955 (84) | 0.947 (59) | 0.947 (53) |
| HR library vs NIST23 | 0.904 | 0.908 | 0.759 |
| e5 vs NIST23 | **0.934** | 0.846 | **0.882** |
| e5 vs HR library | 0.862 | 0.778 | 0.676 |

Without the window, the HR-vs-NIST values are 0.884, 0.874 and 0.750. With nominal m/z 149 removed from every
Metabolome spectrum, HR vs NIST is 0.794, e5 vs NIST 0.882 and e5 vs HR 0.704.

- On HREI-MSDB and Metabolome, e5 predicts the NIST23 spectrum better than the measured HR library
  matches NIST23. The low HR scores on those sources mostly measure the difference between libraries.
- On Exposome, e5 vs NIST23 (0.846) is below library agreement (0.908), so that gap is model error.
- The difference between libraries is systematic. For matched pairs, the mean intensity share by m/z as a
  fraction of the molecular ion (<20%, 20–40%, 40–60%, 60–80%, 80–95%, ±5% of M) shows the HR spectra put
  less intensity below 40% of M and more at 60–95% than NIST23 does. For example, Metabolome HR is 0.062/0.276/0.316/0.185/0.079/0.082
  vs NIST23 0.120/0.356/0.271/0.133/0.053/0.066. e5 tracks NIST23 within 0.01 in every bin and every source.
  The direction is the same in all three HR sources, consistent with an instrument or tuning mass
  bias relative to NIST23. This was not tested further.

#### Are the Orbitrap HR spectra softer than NIST23?

All three HR libraries are Orbitrap GC: Q Exactive GC for RECETOX, and Exploris GC for HREI-MSDB according to its
Figshare description. The test pairs each HR record with its first NIST23 spectrum, cut to the source window,
and uses Wilcoxon paired tests (n 321/363/176):

| Median, HR vs NIST23 | HREI-MSDB | Exposome | Metabolome |
| --- | --- | --- | --- |
| M+• share of total intensity | 0.020 vs 0.040 (p 0.66) | 0.014 vs 0.029 (p 9e−7) | 0.000 vs 0.002 (p 5e−3) |
| M+• relative to base peak | 0.103 vs 0.243 (p 3e−16) | 0.064 vs 0.148 (p 1e−22) | 0.000 vs 0.006 (p 8e−8) |
| Intensity-weighted m/z ÷ M | 0.630 vs 0.588 (p 3e−42) | 0.642 vs 0.601 (p 3e−24) | 0.521 vs 0.454 (p 1e−24) |
| Spectral entropy (nats) | 2.67 vs 2.85 | 2.93 vs 3.24 | 2.91 vs 3.13 |
| Peaks ≥1% of base | 30 vs 33 | 46 vs 51 | 45 vs 46 (p 0.60) |

The HR spectra are not softer in the usual sense, because the molecular ion is equal or weaker. Instead,
intensity moves from deep low-mass fragments to higher-mass fragments, and the spectra are more concentrated (lower
entropy, fewer small peaks). For peaks ≥2% of base in both spectra, a joint regression of log2(HR/NIST) intensity
on absolute m/z and on m/z ÷ M gives positive slopes for both. Relative position is stronger for HREI-MSDB
(+0.24 vs +0.10 log2 per SD) and Exposome (+0.14 vs +0.03); the two are equal for Metabolome (+0.11 each). So a pure
absolute-mass transmission bias does not explain the shift. One untested explanation: slow (metastable) M+• decay while ions
are stored in the C-trap would lower M+• and raise primary fragments. Low-mass transmission or thresholding during library
curation may add to it. Script: session scratch `softness.py` (numbers reproduced here).

#### Chemical space: training vs references

[hr_ei_umap.py](../../../scripts/debug/ei/hr_ei_umap.py) fits one joint UMAP (Morgan r=2, 2048 bits,
Jaccard) on a seeded 40,000-molecule sample of the 149,848 unique e5 training molecules plus all
845 unique reference molecules (by source and 2D InChIKey). Outputs are in `output/hr_ei_umap_2026_10_18/`
(`umap_train_vs_hr_ei.png`, coordinates, `hr_mols.parquet` with nearest-training Tanimoto).

| Reference | Molecules | Median nearest-train Tanimoto | ≥0.7 | <0.4 | Spearman (Tanimoto, e5 cosine) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HREI-MSDB | 278 | 0.667 | 42% | 6% | −0.02 (p 0.68) |
| Exposome | 350 | 0.640 | 30% | 9% | 0.31 (p 5e−9) |
| Metabolome | 217 | 0.706 | 52% | 3% | 0.03 (p 0.69) |

All three sets lie inside the training cloud. Metabolome is the *closest* to training, yet it scores lowest,
and its scores do not track similarity. That is consistent with the reference and composition causes above,
not with chemical novelty. Its worst molecules form one tight cluster: small poly-TMS acids, amino acids and
urea (succinic, fumaric, glutaric, itaconic, aspartic, urea; cosine 0.0–0.3 at Tanimoto 0.5–0.85). In those
spectra, TMS ions such as 73, 147 and 149 dominate the measured intensity. Glutamic acid_2TMS scores 0.00; glutamic acid is known to
cyclize to pyroglutamate during silylation, so its mapped structure may be wrong (not checked). Exposome is the only
source where score rises with similarity (0.50 below 0.4, 0.78 at 0.7–0.999).

Some molecules share an identical fingerprint with a training molecule (9% HREI-MSDB, 6% Metabolome)
despite different 2D InChIKeys. The split excludes by connectivity key, so these are fingerprint
collisions between different connectivities (likely positional isomers; not inspected).

### Interpretation

- The e5 numbers are the held-out HR-EI result: at 10 ppm with natural isotopes, HREI-MSDB 0.850,
  Exposome 0.706, and Metabolome 0.573. One seed; run-to-run variance at this n is unmeasured.
- Exposome loses 0.075 from 10 to 5 ppm, which is why 10 ppm is the headline. It is a property of
  the reference masses or matching floor, not of the checkpoint.

### Outputs and reproduction

`output/hr_ei_e5_eugy8s9n_eval_2026_10_18/{natural_exact,monoisotopic}/`, with the same file
layout as the frozen e1 evaluation (`scores.csv`, `metric_summary.csv`, `provenance.json`, plots,
`cache/exact_predictions.npz`). Runtime on GPU: 57 s natural, 22 s monoisotopic.

```bash
FRAGNNET_JOBLIB_N_JOBS=1 MPLCONFIGDIR=/tmp/fragnnet_mpl \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m scripts.debug.ei.evaluate_hr_ei_checkpoint \
  --checkpoint saved_ckpts/e5_hr_ei_holdout_nominal_eugy8s9n.ckpt --isotope-envelope natural \
  --output <fresh dir>
```

The plot script's figure title was hardcoded to "Frozen e1 checkpoint". It now uses the checkpoint
file name from `provenance.json`; e1 figures generated earlier keep the old title.

### Next

Add an e5 seed to measure run-to-run variance. Castro still needs structure mapping before it can be scored.
