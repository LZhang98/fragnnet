# MASSter EI performance audit

> **Status:** FINDING · **Updated:** 2026-10-06
> Checkpoint and inference audit complete (e1 `jeljdikj`@37: Stein 0.886845, RASSP-eligible subset 0.899167); proposed training comparisons remain unrun.

## Scope and comparison

The natural-isotope `e1` checkpoint, [W&B `jeljdikj` epoch 37](https://wandb.ai/frag-gnn/frag-gnn/runs/jeljdikj), scores **0.886845 mean Stein weighted similarity** and **0.869983 mean 1 Da cosine** on all 51,685 `nist23_ei_inchikey` test spectra. The 26,372 test spectra passing RASSP's published NIST17 filters (H/C/N/O/F/P/S/Cl, at most 48 **total** atoms including H, at most 4096 formulae) score **0.899167 mean Stein**. The [RASSP paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9909676/) reports **0.929 average Stein** for FormulaNet on its NIST17 dataset. The 0.030 difference is descriptive: training overlap, dataset, preprocessing and split are unmatched, and RASSP has not been run on our test set. See the [full test score and filter record](../../plans/masster/masster_ei_nominal_plan.md#stein-weighted-similarity-on-the-full-test-split-2026-10-01).

Stein similarity uses the [RASSP metric](https://github.com/thejonaslab/rassp-public/blob/main/rassp/metrics.py): cosine after transforming each nominal peak to `m/z**3 * intensity**0.6`. On the full test set, mean Stein falls from **0.9202** at 2–16 total atoms to **0.8556** above 64 atoms; see the [size bin table](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/size_performance_stein_bins.csv). The published FormulaNet inference limit is 48 total atoms, so the >48 groups describe our model's extrapolation and do not form a RASSP matched subset.

## What limits the current score

On a 512-spectrum validation sample, an oracle that keeps the model's predicted nominal m/z support and assigns the measured intensities scores **0.99615 mean Stein**, versus the model's **0.88703**. Mean target Stein weighted squared norm covered by predicted m/z is **0.99321**. In the >64-total-atom sample group (45 spectra), model/oracle Stein is **0.8234/0.9902**. This oracle is an upper bound using the answer at inference; it establishes available m/z support, not attainable model performance. The leading observed gap is intensity allocation among mostly available masses.

The current EI recipe trains with normalized-target cross entropy and selects its best checkpoint by unweighted 1 Da cosine. Neither directly optimizes Stein's mass-weighted peak vector. On the same validation sample, post-hoc peak powers 0.6, 0.8, 1.2 and 1.4 all score below the unmodified prediction's **0.88703** Stein; multiplying intensities by `m/z**alpha` for alpha −1, −0.5, +0.5 or +1 also scores lower. A global output rescaling is therefore unsupported by this check. Sample target Stein weighted norm is concentrated at m/z at least half the precursor (**0.93334**), so a learned, mass-aware ranking objective is worth testing, but no improvement has been measured.

Test molecules with replicate spectra yield **0.91889 mean pairwise Stein** across 2,245 pairs; the model's mean on those replicate-bearing spectra is **0.85360**. Replicates can differ in acquisition conditions, and this is a selected subset; the pairwise number is context for attainable agreement, not a universal noise ceiling.

## Priority actions

1. **Establish a matched RASSP comparison.** Check training overlap and preprocessing, then run both models on one held-out, RASSP-eligible split with the same Stein implementation. Until then, do not interpret the published 0.929 as a head-to-head loss.
2. **Use the prepared intensity experiments.** Compare the fresh `e2` control with `e3` full candidate attention and `e4` EMA 0.999 on the same validation split, recording both 1 Da cosine and Stein, by size and on the RASSP-eligible subset. The [configs and controls](../../plans/masster/masster_ei_nominal_plan.md#matched-intensity-configs-2026-10-01-prepared-not-launched) are prepared but untrained. Record source provenance first because the original checkpoint's remote source revision is unavailable.
3. **If the matched arms leave a Stein gap, test a mass-aware intensity objective** against the fresh control, selecting and reporting both metrics on validation. The sample's failed output tilts do not test this training hypothesis. Preserve a separate final test read.
4. **Investigate large-molecule errors with a larger validation sample** and matched size/chemistry strata before altering candidate count, H-shift width or SMARTS rules. The current support oracle and prior EI SMARTS screen do not justify those changes for the mean score.

## Artifacts and limits

Full test scores and replicate pairs: `output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/spectrum_rows.csv` and `audit_replicate_pairs.csv`. Validation support-oracle rows and predicted peaks: `output/masster_ei_jeljdikj_analysis_2026_10_01/sample512_cosine/audit_stein_ceiling_rows.csv` and `peak_rows.csv.gz`. These outputs are local ignored artifacts; the 512-spectrum sample is diagnostic, and no uncertainty interval or replicated seed has been established. Five individual test predictions differ by over 0.01 cosine between two full inference runs, though their mean differs by only 4.2e-7; the individual discrepancy remains unresolved.
