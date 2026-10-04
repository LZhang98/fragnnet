# Ambiguity-preserving base-peak comparisons for i1

> **Status:** FINDING · **Updated:** 2026-09-30
> Step 1 complete: 52,020 validation spectra / 4,222 molecules, fresh annotated inference on
> `i1` epoch 59. Internal artifact audit passed. Historical predictions differ on a small
> subset, documented below. No correction fitted and no new chemical rule claimed.

## Purpose and prior evidence

The user requested a concrete comparison table to investigate why an available fragment receives
little predicted intensity on an unseen molecule. This extends the
[existing base-peak analysis](masster_i1_base_peak_errors_2026_09_28.md): that analysis selected
one nearest candidate mass per measured peak, then retained structures at that mass. This export
retains **all output channels within instrument tolerance**, including neighboring candidate
masses and zero-share channels. It never chooses a true structure from a mass match.

The earlier chemistry comparison and attribute-reweighting null remain evidence; this extraction
does not supersede them or establish a new training signal. The
[worst-prediction analysis](masster_i1_worst_predictions_2026_09_30.md) supplies the existing
instrument-tolerance floor and ranking-versus-coverage context.

## Export result

| Base-peak outcome | Spectra |
| --- | ---: |
| Covered, winner differs (`ranking_pairs.csv`) | 19,325 |
| Winner matches (`controls.csv`) | 31,380 |
| No matching candidate (`coverage_failures.csv`) | 1,315 |
| Total | 52,020 |

The ranking pairs span 3,889 molecules. **14,108 of 19,325 have more than one target-compatible
atom set** (73.0%); 12,922 have more than one winner atom set. These are structural alternatives,
not additional measured labels. In 276 ranking pairs the target group spans multiple rounded
predicted masses. The export holds 295,715 channel rows across all outcomes.

There are 63 ranking rows with tied measured base peaks; 33 do not have a strict measured
preference over the predicted winner. No ranking row has overlapping target/winner groups.
The prior instrument-tolerance floor (330 spectra) contains 276 ranking rows and 54 base-peak
coverage failures here. The latter is a **base-peak** criterion, distinct from the earlier
whole-spectrum coverage ceiling that classified 40 floor spectra as uncovered.

## Validation and historical-reference differences

- 34 focused tests passed: eight for this exporter/audit and 26 for reused peak/chemistry
  helpers. They cover ambiguous mass matches, zero-probability alternatives, summed-peak
  winners, instrument tolerance, uncovered targets, measured ties and original bond indices.
  The audit test deliberately deletes a zero-share alternative and verifies rejection.
- The 64-spectrum smoke matches the saved predictions with a maximum share difference of
  `3.92e-7`. Full local GPU inference completed in 347.8 seconds, with strict checkpoint load.
- The full export audit checks all 52,020 identities and measured targets, all 295,715 channel
  rows, group membership, unique channel IDs, structural counts and summed group shares.
  These checks passed. No full model test suite was run; no model code was changed by this task.
- **The fresh predictions are not an exact reproduction of the older spectral cache.**
  Every base-peak outcome agrees. One predicted winner differs (spectrum 106430, still an
  `under_ranked` case); 12 target-group channel counts and nine winner-group counts differ.
  Predicted group-share differences exceed `atol=3e-6, rtol=1e-4` in 400 spectra. The median
  absolute difference is `1.18e-7`, the 99th percentile `1.13e-6`, and maximum `0.03493`.
- The first strict historical comparison stopped on spectrum 181287. The audit now explicitly
  separates internal/target validation from historical prediction differences; it does not
  relax the original numerical threshold or report historical matching as passed.
  [`reference_differences.csv`](../../../output/i1_fragment_pairs_2026_09_30/val/reference_differences.csv)
  lists the comparison for every spectrum, and `audit.json` reports `reference_status:
  differences_detected`.
- A complete reference-batch-size rerun (`val_batch128/`) still shows the first discrepancy,
  so batch size alone does not explain the historical difference. Comparing the two fresh
  runs gives identical outcomes on all 52,020 spectra, but 14 target-group counts, 12
  winner-group counts and two winner masses change. Maximum target/winner share changes are
  0.01123/0.01211. The underlying candidate-selection sensitivity is unresolved; these are
  observations, not proof of a particular numerical cause.
  [`batch_comparison.json`](../../../output/i1_fragment_pairs_2026_09_30/batch_comparison.json)
  and [`batch_differences.csv`](../../../output/i1_fragment_pairs_2026_09_30/batch_differences.csv)
  preserve the aggregate and per-spectrum check. Use the canonical `val/` table's scores and
  channel memberships together; flag sensitive rows in later analyses and do not substitute
  archived scores for its candidate groups.

## Definitions

- Checkpoint: `i1_inst_ob_obf`, run `i50ig7rf`, epoch 59. Unmerged NIST20 `[M+H]+` validation
  spectra, instrument inputs `OB` and `OB_F`. Normal eval preprocessing and `mz_max` filtering.
- Measured base peak: maximum intensity in the eval-preprocessed target. Ties select the first
  stored peak and are explicitly counted. This is not the unfiltered raw spectrum.
- Target-compatible group: every emitted channel within `1e-5 * max(measured_mz, 200)` Da of
  that peak. This includes all valid emitted H-shift channels, without a probability threshold.
- Predicted winner: sum channel intensities at m/z rounded to four decimals, then take the
  maximum, consistent with the instrument-cosine peak aggregation. Ties select the lowest m/z
  and are counted. The winner group contains its actual contributing channels.
- All reported shares are L1-normalized within their own measured or predicted spectrum.
  At the winner location, both maximum and summed matched measured intensity are recorded.
  Matching uses each measured peak's instrument tolerance.
- `hit`: a covered measured base peak matches the predicted winner. `under_ranked`: the base
  peak is covered but the winner differs. `uncovered`: no emitted channel matches the base peak.
  These are base-peak outcomes, not the whole-spectrum coverage classes in the earlier finding.
- `strict_measured_preference` distinguishes a strict measured preference from a tied base peak.
  Do not train a strict pairwise target on a measured tie. `groups_overlap` flags overlap caused
  by rounded winner bins at a tolerance boundary.
- The target group can span multiple distinct predicted masses. Its summed probability is an
  ambiguity-group statistic, not a peak intensity directly comparable to the winner's share.
- `target_exact_peak_count` counts four-decimal predicted m/z groups, not distinct formulas.
- Structural ambiguity counts distinct retained atom sets; distinct channel counts additionally
  distinguish H shift and ion mode. No target intensity is divided among structures, and
  predicted shares are not interpreted as posterior confidence in the experimental assignment.

## Artifacts and schema

All outputs are under [`output/i1_fragment_pairs_2026_09_30/val/`](../../../output/i1_fragment_pairs_2026_09_30/val/).

| Artifact | Contents |
| --- | --- |
| `pairs.csv`, `pairs.pkl` | One row per spectrum: IDs, NCE, instrument, adduct, precursor m/z, outcome, measured/predicted shares, group sizes, ambiguity and tie flags |
| `ranking_pairs.csv` | Covered base-peak misses (`under_ranked`) |
| `controls.csv` | Base-peak hits; available controls, not a matched-control design |
| `coverage_failures.csv` | Base peak absent from emitted candidates |
| `channels.jsonl` | One row per channel in either group; join on `spec_id`. Group booleans, predicted share, m/z, H shift, ion mode, retained atoms/subgraph, boundary bonds and chemistry descriptors |
| `molecules.csv` | Parent SMILES and atom-mapped parent SMILES; map number = original zero-based atom index + 1 |
| `provenance.json`, `resolved_config.json` | Input hashes, checkpoint/config, Git revision and dirty diff hash, diagnostic source hash, environment/device, timing and counts |
| `audit.json`, `reference_differences.csv` | Internal/target checks and separately reported differences from independently saved spectral predictions |

`boundary_bond_indices` are RDKit bond indices in the original stored molecule. Atom indices use
the original stored molecule, not a reparsed canonical SMILES. `retained_subgraph_smiles` describes
induced connectivity; it is not a fully assigned product ion. Boundary bonds describe isolation of
a candidate subgraph, not an experimentally identified sequence of bond-breaking events.

## Reproduction

Use the configured `FRAGNNET-GPU` interpreter. The output directory must be fresh.

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/masster_memorization/debug_masster_fragment_pairs.py \
  --checkpoint saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt \
  --custom_fp config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml \
  --spec_fp output/i1_worst_predictions_2026_09_30/subset_spec_df_val_all.pkl \
  --output_dp output/i1_fragment_pairs_2026_09_30/val

/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/masster_memorization/validate_masster_fragment_pairs.py \
  --output_dp output/i1_fragment_pairs_2026_09_30/val \
  --reference output/ev_ladder_census_2026_09_28/i1_val.pkl
```

The extractor supports `--limit` for a smoke check and `--device cpu` if CUDA is unavailable.
The first smoke attempt exposed singleton-list NCE values in the dataset and stopped before
writing a summary; `smoke_v2` passed after handling that representation. The complete `val`
export uses the corrected source. Existing model edits were left untouched.

## Limits and next use

This is exploratory validation data from a checkpoint selected on validation. It is **not** an
out-of-fold training set or an untouched confirmation set. All successful rows are retained,
and the [conditional comparison](masster_i1_pair_contrasts_2026_09_30.md) now supplies
strongest-rival controls and matched effects (step 2 complete).
A correction would require molecule-separated discovery/tuning and untouched evaluation.
No training, correction fitting, cluster submission, or dataset mutation is part of this step.
