# Spectraverse curation pipeline (Gupta et al., Anal. Chem. 2026)

> **Status:** NOTE · **Updated:** 2026-10-03 · Read from the PMC full text
> (PMC12903054). Checked against our local v1.0.2 data; the supplementary filter lists and the
> GitHub code were not read.

Gupta V., Qiang H., Chung H.-H., Herbst E., Skinnider M. A. "Comprehensive Curation and
Harmonization of Small-Molecule MS/MS Libraries in Spectraverse." *Anal. Chem.* 98(5),
3934–3943 (2026). [doi:10.1021/acs.analchem.5c06256](https://doi.org/10.1021/acs.analchem.5c06256).
This is the dataset paper. The benchmark (folds, candidate sets) is in a second paper; see
[spurious evaluations](spurious_evaluations_spectraverse.md). The paper names v1.0.1, but its
counts (488,630 spectra, 44,237 molecules) are v1.0.0's; our data is v1.0.2 (488,797 spectra).

## Pipeline as described

1. **Sources.** GNPS, MoNA, MS-DIAL, HMDB and MSnLib, plus libraries from individual papers
   (some reprocessed from MetaboLights raw data). 1,672,217 spectra in. The paper says it
   favoured permissive choices so that users can apply stricter filters themselves.
2. **Metadata repair.** Field names and values were harmonized by hand, then with matchms 0.27.0.
   Missing structures came from the GNPS API, and SMILES were curated by hand for 19,574 spectra.
3. **Early peak filters.** Zero-intensity peaks and peaks outside m/z 10–1000 were removed, and
   intensities were normalized to the base peak. Spectra with uniform intensities or precursor
   m/z > 1000 were removed.
4. **Structure standardization.** Stereochemistry, metals and fragments were removed; tautomers
   and charges were standardized. A zwitterion whose neutralization changes the charge is kept
   as two copies (charged and uncharged) through the second matchms pass. Rare adducts and
   neutral-loss adducts were dropped, leaving nine adducts.
5. **Quality filters.** A spectrum was dropped if all fragment masses have ≤ 2 decimals (low
   resolution), if all fragments lie above the precursor, or if it has a single fragment
   within ±1.6 of the precursor. A second, stricter matchms pass followed.
6. **Near-duplicates.** Pairs with the same InChIKey first block and polarity and cosine > 0.99
   were collapsed to one spectrum, keeping the one with the most peaks.
7. **Final filters.** Precursor m/z error > 10 ppm, triple-quad spectra and radical SMILES were
   removed. Peaks below 0.1% relative intensity were dropped and the top 4,096 peaks kept.
   Instruments were mapped to QTOF / Orbitrap / ion trap / not specified. CE is stored as up to
   three values (ramped: 2, stepped: 3), each in both eV and NCE.
8. **Not done, on purpose.** No centroiding and no electronic denoising: both slightly lowered
   the same-versus-different-compound cosine AUROC. Nothing is source-specific.

## Checked against our v1.0.2 copy (2026-10-03)

| Paper statement | Local data |
| --- | --- |
| Top 4,096 peaks per spectrum | Our unmerged maximum is exactly 4,096. |
| Instruments: QTOF / Orbitrap / ion trap / unspecified | MGF `INSTRUMENT_TYPE` qtof 169,380, orbitrap 290,183, iontrap 16,181, missing 13,053. **Our `Other` is ion trap**; our `NA` is unspecified. |
| CE in eV and NCE | Both fields are present for every spectrum with CE (`ACE = NCE × prec_mz / 500`); our `spec_df` keeps only ACE. GNPS and RIKEN carry none. |
| Near-duplicates (cosine > 0.99) removed | **Not true in v1.0.2.** Sampled 163,230 pairs from 1,500 (skeleton, polarity) groups: 5.9% exceed 0.99 with a greedy 0.1 Da cosine (matchms default), 3.7% with a 0.01 Da binned cosine, and 0.4% are byte-identical. 65% of the > 0.99 pairs share a source (GNPS–GNPS most). Over the whole set, 19,605 spectra (4.0%) have a byte-identical twin for the same molecule, and 23% of those twin sets disagree on ACE. |
| Spectra are not centroided | Consistent with the long, flat peak tails seen in merged targets. |

### Dedup code, and why duplicates survive (read 2026-10-03, corrected 2026-10-04)

Repo: `github.com/skinniderlab/spectraverse-analysis`, `spectraverse/step3-3`…`step3-5`.
The near-duplicate pass is the **only** step that compares spectra with each other. The matchms
YAMLs (`matchms/library_cleaning_{1,2}.yaml`) check metadata and annotation coherence only.
There is no outlier or consensus filter between replicates.

- **Grouping and order.** Groups are (InChIKey first block, ion mode) across all sources.
  `step3-3` writes each group's spectra in original row order
  ([L63-L71](https://github.com/skinniderlab/spectraverse-analysis/blob/cdd342be17a25ae029a01f32f1aa67774b6cdee5/spectraverse/step3-3_uniq-comb.py#L63-L71)).
- **Cosine.** `step3-4` scores pairs in that order with
  `spectral_entropy.similarity(method='dot_product', need_clean_spectra=False, ms2_ppm=5)`
  ([L58-L64](https://github.com/skinniderlab/spectraverse-analysis/blob/cdd342be17a25ae029a01f32f1aa67774b6cdee5/spectraverse/step3-4_uniq-cos-calc.py#L58-L64)).
  It runs on untrimmed spectra, before the 0.1% / top-4,096 step.
- **Index mismatch, current code only.** Commit `1b94e9f` (2026-06-29, "add reordering logic
  based on source and instrument type") re-sorts `dup_df` before selection
  ([L181-L184](https://github.com/skinniderlab/spectraverse-analysis/blob/1b94e9f2142a547f8d0d5a5b7618eb4570e0cf5f/spectraverse/step3-5_uniq-select.py#L181-L184)).
  It then reads matrix positions on the re-sorted rows
  ([L194](https://github.com/skinniderlab/spectraverse-analysis/blob/1b94e9f2142a547f8d0d5a5b7618eb4570e0cf5f/spectraverse/step3-5_uniq-select.py#L194),
  [L203](https://github.com/skinniderlab/spectraverse-analysis/blob/1b94e9f2142a547f8d0d5a5b7618eb4570e0cf5f/spectraverse/step3-5_uniq-select.py#L203)).
  In any group the sort reorders, a > 0.99 pair maps to two other spectra. The earlier version
  `62278d8` (2025-12-10) has no sort, so its positions match
  ([L50-L64](https://github.com/skinniderlab/spectraverse-analysis/blob/62278d84e2d0a4038cbcb5a0ce42b3b0cfb6a0e1/spectraverse/step3-5_uniq-select.py#L50-L64)).
- **Dead commercial branch, both versions.** The commercial-source choice is overwritten on the
  next line (`1b94e9f` L204-L208; `62278d8` L60-L64), so the spectrum with fewer peaks is
  always dropped.
- **Which code built v1.0.2 is unknown.** The repo has no commits between 2025-12-10 (the
  v1.0.1 release) and 2026-06-29. v1.0.2 came out on 2026-05-03, and none of its build code is
  public.

**Why twins survive: the dedup broke in v1.0.1, most likely through the index mismatch
(re-tested 2026-10-04 on all three releases).**

All three Zenodo MGFs are downloaded and md5-verified, under
`data/raw/spectraverse/spectraverse-<v>/`. They are compared by
`scripts/debug/spectraverse/spectraverse_version_compare.py`, log
`output/analysis/spectraverse_source_audit_2026_10_03/version_compare.log`.

| Release | Published | Spectra | `SOURCE` / `FOLD` | Exact twins | Groups ≥ 3 with a twin: reorderable / order fixed |
| --- | --- | ---: | --- | ---: | ---: |
| v1.0.0 | 2025-10-02 | 488,630 | no / no | 4 | 0.02% / 0% |
| v1.0.1 | 2025-12-10 | 488,797 | yes / no | 19,625 (4.0%) | 19.7% / 0.13% |
| v1.0.2 | 2026-05-03 | 488,797 | yes / yes | 19,625 (4.0%) | 19.7% / 0.13% |

- **The paper's count is v1.0.0.** "488,630 spectra from 44,237 molecules" matches v1.0.0, though the
  paper names the release v1.0.1. Our old processed table (`data/df/spectraverse_df.csv`) is
  v1.0.0: 99.94% of its content is found in v1.0.0 and 54% in v1.0.1.
- **v1.0.1 was a rebuild.** Against v1.0.0, 221k spectra (by content) were removed and 191k added:
  MSnLib 90k, GNPS 59k, MS-DIAL 15k, MassBank 14k, RIKEN 8k.
- **v1.0.2 is v1.0.1 plus folds and candidates.** Spectrum content is identical except 694
  spectra whose metadata changed (InChIKey skeletons 44,237 → 44,234).
- **The failure has the index-mismatch signature.** Exact twins survive in 19.7% of
  reorderable groups (size ≥ 3) and 0.13% of order-fixed ones. So dedup ran and worked wherever
  the order could not change. The size bands (v1.0.2, MS-DIAL and RIKEN excluded from "order
  fixed") rise with group size:

  | Group size | Can reorder | Order fixed |
  | --- | ---: | ---: |
  | 2 | 0.22% | 0.98% |
  | 3–4 | 5.1% | 0.16% |
  | 5–8 | 9.5% | 0.03% |
  | 9–16 | 16.5% | 0% |
  | 17+ | 32.8% | 0% |

  Size 2 is the control: a reorder of two rows still maps the pair onto the same two spectra.
- **Timeline.**
  - The public `step3-5_uniq-select.py` at the v1.0.1 release is commit `62278d8` (2025-12-10). Its
    only change from the October original casts `NUM_PEAKS` to int, which changes which copy is
    dropped, not whether one is.
  - The re-sort ("add reordering logic based on source and instrument type", +144 lines) was
    authored and pushed on 2026-06-29 (`1b94e9f`).
  - So v1.0.1 and v1.0.2 were most likely built with that code before it was published.
- **Implication.** The same mismatch also dropped unrelated, non-duplicate spectra from reordered
  groups, which cannot be measured from the releases. The 221k content removed between v1.0.0 and
  v1.0.1 mixes intended changes (rebuild, new filters) with this loss.
- **The scoring call is not the cause.** Replayed (SpectralEntropy master, pure Python, matchms-style
  base-peak normalization, their single-peak padding), it scores 99.8–100% of sampled
  byte-identical twin pairs above 0.99.

## What it means for us

- The noise we measured (MS-DIAL, duplicates, ion-trap `Other`) is in the data as released.
  Spectraverse asks users to train only on its data, but it does not forbid filtering it. A
  train-side filter or merge is a modelling choice, and held-out folds must stay untouched.
- Merged spectra: the paper says NPLIB1's merged spectra "may not resemble individual spectra".
  Our merged-train arm (removed 2026-10-18; `git show 95341fce:docs/plans/masster/spectraverse_merged_train_plan.md`) merges train only
  and scores per spectrum, so its evaluation is unaffected.

Check script (in repo): `scripts/debug/spectraverse/spectraverse_data_check.py` covers the md5, the spec_df
record match, the folds, byte-identical twins and near-duplicate pairs. Pass
`--spectral-entropy-dir <SpectralEntropy checkout>` to also replay the upstream cosine.
Other artifacts (ignored) are in `output/analysis/spectraverse_source_audit_2026_10_03/`:
`fold_mces_proxy.py`, plus `upstream_code/` with the upstream step3 scripts at both commits and
the SpectralEntropy source.
