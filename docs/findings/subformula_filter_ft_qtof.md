# 5 ppm Subformula Filtering on NIST20-HR

> **Status:** FINDING &nbsp;·&nbsp; **Updated:** 2026-09-16
>
> Historical NIST20-HR census for `[M+H]+` and `[M-H]-`, stratified by FT and QTOF.
> The current evaluation scope also includes `[M+Na]+` and `[M+Cl]-`; the existing numerical
> tables remain the two-adduct baseline until the four-adduct run is complete.
> The 2026-09-16 section at the end measures the Cl/Na budget fix on all four adducts and
> records the MASSter scope; it does not restate the 5 ppm tables.

## Question

Is it safe to merge experimental peaks within 20 ppm and then retain only peaks matching a
chemically feasible, precursor-bounded molecular subformula within 5 ppm?

The two operations were evaluated separately because they test different assumptions:

- **20 ppm merging:** cluster peaks against an intensity-weighted centroid and sum their
  intensities.
- **5 ppm subformula filtering:** enumerate nonempty subformulas bounded by the neutral precursor
  formula, require nonnegative RDBE, apply the precursor adduct (`[F+H]+` or `[F-H]-`), and retain
  an observed peak only if its m/z is within 5 ppm of a candidate ion mass. A deprotonated
  candidate must contain a hydrogen that can be removed.

The subformula result therefore measures the **combined effect** of the 5 ppm window, precursor
element budget, adduct model, and RDBE rule. It does not isolate RDBE as the cause of rejection.

### Expanded adduct scope

The current cleaning and DAG evaluation scope is `[M+H]+`, `[M-H]-`, `[M+Na]+`, and `[M+Cl]-`.
Subformula filtering applies the corresponding precursor-ion mass shift for all four types. In DAG
evaluation, leave CMF enabled so sodium-adducted fragments can use the protonated CMF carrier and
chloride-adducted fragments can use the deprotonated CMF carrier. The H-transfer range remains
controlled independently by `--max_h_transfer 4`.

The result tables below predate this expansion and contain only `[M+H]+` and `[M-H]-`. They must
not be interpreted as four-adduct results.

### Instrument-tolerance centroid definition

With `--merge-inst-tol --merge-method common_overlap` (the default for instrument tolerance), each
observed peak defines an instrument-tolerance interval

\[
[m_i - T_i,\; m_i + T_i],
\qquad
T_i = r_{\mathrm{inst}}\max(m_i, m_{\min,\mathrm{inst}}).
\]

A peak joins the current cluster only while every interval in that cluster has a common
intersection:

\[
\max_i(m_i-T_i) \le \min_i(m_i+T_i).
\]

For two peaks with approximately equal tolerances, this allows a separation up to `2 * T`; it
does not compare later peaks with `2 * T` around a moving centroid. The cluster-wide intersection
prevents transitive chaining and centroid drift from joining peaks that cannot share one true m/z.

The merged peak uses the standard intensity-weighted centroid and summed intensity:

\[
m_{\mathrm{centroid}} = \frac{\sum_i I_i m_i}{\sum_i I_i},
\qquad
I_{\mathrm{centroid}} = \sum_i I_i.
\]

The same grouping rule is available with a fixed ppm radius, for example
`--merge-ppm 20 --merge-method common_overlap`. In that case every interval has a ±20 ppm radius,
so two equal-mass-scale peaks can be separated by up to approximately 40 ppm. The fixed
`--merge-ppm` path defaults to the `naive_centroid` method when `--merge-method` is
omitted.

To select the `naive_centroid` method with instrument tolerance, use
`--merge-inst-tol --merge-method naive_centroid`. It compares each new peak with the moving
intensity-weighted centroid using `2 * T`. Both methods use the same intensity-weighted output
centroid and summed intensity; the flag changes only cluster membership.

## Population

The source was `data/proc/nist20/spec_df.pkl`, restricted to `dset == nist20_hr`:

- Source table: 1,148,072 spectra.
- Selected `[M+H]+`/`[M-H]-` and FT/QTOF: 434,926 spectra.
- Processed: 434,917 spectra containing 14,195,520 peaks.
- Excluded explicitly: 9 `[M+H]+` FT spectra from one molecule containing `Sn`, because the
  repository RDBE element table does not support tin.

No sampling was used. Empty spectra were counted after filtering and then dropped from the output
count. The run was stats-only, so it did not write a duplicate cleaned spectrum table.

## 5 ppm results

| Precursor | Instrument | Spectra | Peaks retained by 20 ppm merge | Peaks retained by 5 ppm subformula filter | Intensity retained by subformula filter | Empty after subformula filter | Base peak retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[M+H]+` | FT | 286,490 | 99.93% | 83.90% | 90.27% | 0.68% | 93.07% |
| `[M+H]+` | QTOF | 24,573 | 99.99% | 25.23% | 23.37% | 24.25% | 22.15% |
| `[M-H]-` | FT | 121,450 | 99.95% | 83.79% | 91.22% | 1.56% | 92.83% |
| `[M-H]-` | QTOF | 2,404 | 99.71% | 24.19% | 23.08% | 27.25% | 21.80% |

Across all four groups, 20 ppm merging retained 99.94% of peak records and 100% of total
intensity. It produced no empty spectra and retained the base peak in every spectrum. The merge is
therefore a low-impact cleanup operation on this population.

The 5 ppm subformula filter is strongly instrument-dependent even within NIST20-HR:

- FT retains about 84% of peak records and 90–91% of intensity. Although substantially more
  aggressive than merging, most dominant signal survives.
- QTOF retains only 24–25% of peak records and about 23% of intensity. About one-quarter of QTOF
  spectra become empty, and the original base peak survives in only 22%.

## Supplemental FraGNNet instrument-tolerance comparison

The census was repeated with the exact evaluation-time matching contract:

- FT: `10 ppm × max(observed m/z, 200)`, giving a minimum ±0.002 Da window.
- QTOF: `20 ppm × max(observed m/z, 750)`, giving a minimum ±0.015 Da window.

This is a separate comparison and is not the 5 ppm result above. Only the subformula matching
tolerance changed. Peak merging remained fixed at 20 ppm, and the
precursor-bounded formula, adduct, removable-hydrogen, and RDBE rules were unchanged.

| Precursor | Instrument | Peaks: 5 ppm | Peaks: instrument tolerance | Intensity: 5 ppm | Intensity: instrument tolerance | Empty: 5 ppm | Empty: instrument tolerance | Base peak: instrument tolerance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `[M+H]+` | FT | 83.90% | **91.07%** | 90.27% | **97.11%** | 0.68% | **0.00%** | 98.96% |
| `[M+H]+` | QTOF | 25.23% | **96.63%** | 23.37% | **98.53%** | 24.25% | **0.03%** | 98.93% |
| `[M-H]-` | FT | 83.79% | **91.74%** | 91.22% | **98.20%** | 1.56% | **0.00%** | 99.81% |
| `[M-H]-` | QTOF | 24.19% | **94.81%** | 23.08% | **97.77%** | 27.25% | **0.00%** | 99.04% |

Across all four groups, instrument-tolerance filtering retained 91.45% of peak records, 97.46% of
total intensity, and the base peak in 99.20% of spectra. Only 18 of 434,917 spectra (0.004%) became
empty, compared with 10,452 (2.40%) under 5 ppm.

The QTOF recovery is driven in large part by its low-mass floor. Below m/z 750, the matching window
is ±0.015 Da rather than literal 20 ppm; at m/z 100 this is equivalent to 150 ppm. The improvement
therefore validates use of FraGNNet's established evaluation contract, but should not be described
as a simple 20 ppm result.

### Human review after instrument-tolerance filtering

The repository's existing suspicious-spectrum detector was run on all 434,917 spectra after
FraGNNet instrument-tolerance subformula filtering. Empty filtered spectra were retained for this
step so the detector could flag them directly.

The detector wrote **4,007 candidates (0.921%)** to its audit CSV. Its D3/D4 weighted-recall gate
marked 599 soft-only candidates as sufficiently DAG-explainable and omitted them from visual
review, leaving **3,408 spectra (0.784%)** for human review.

| Precursor | Instrument | Detector candidates | Human-review queue after DAG gate |
|---|---:|---:|---:|
| `[M+H]+` | FT | 2,278 | 1,942 (0.678%) |
| `[M+H]+` | QTOF | 65 | 12 (0.049%) |
| `[M-H]-` | FT | 1,638 | 1,443 (1.188%) |
| `[M-H]-` | QTOF | 26 | 11 (0.458%) |
| **All** | **FT + QTOF** | **4,007** | **3,408 (0.784%)** |

Among the 3,408 ungated candidates, the most frequent reasons were
`single_peak_dominates` (3,059), `substantial_uwpr_contaminant_only_intensity` (198),
`duplicate_spectrum_structure_ambiguity` (113), `substantial_cross_ce_orphan_intensity` (23),
and `empty_spectrum` (18). Reasons are not mutually exclusive. The review score is triage, not an
automatic exclusion decision.

## Conclusion

Use of 20 ppm peak merging is supported for both FT and QTOF in NIST20-HR. The 5 ppm
subformula filter is **not supported**, particularly for QTOF. FraGNNet's instrument tolerance
largely resolves the loss for all four NIST20-HR groups: they retain 97–98.5% of intensity and
98.9–99.8% of base peaks. The 5 ppm QTOF result remains poor, but the instrument-tolerance
floor recovers it for this high-resolution subset.

## DAG coverage at FraGNNet instrument tolerance

DAG coverage is evaluated separately from spectrum cleaning. For the D3/H4 and D4/H4 comparison,
set `--max_h_transfer 4` explicitly and leave CMF enabled (the default). The resulting candidate
support is the union of charge-migration-fragment (CMF) and standard direct-charge-retention ions.
Pass `--no_cmf` only for a standard-ion-only ablation:

```bash
python scripts/eval/dag_oracle_scores.py \
  --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
  --frag_dp data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse/dags \
  --frag_dp_labels d3_h4 d4_h4 \
  --spec_fp <cleaned-spec-df.pkl> \
  --split all \
  --max_h_transfer 4
```

The existing NIST20 DAG oracle was summarized for the NIST20-HR `[M+H]+`/`[M-H]-` × FT/QTOF scope. The
coverage tier is `1_above_prec`, and the matching tolerance is FraGNNet's instrument tolerance
(FT 10 ppm / 200 Da floor; QTOF 20 ppm / 750 Da floor). `peak recall` is the number of observed
peaks matched by at least one DAG ion divided by the number of peaks in that tier. `cosine ceiling`
is the ideal intensity-weighted cosine score if the DAG-achievable peaks receive their true
intensities. It is an upper bound on a model using that DAG.

Unlike the historical oracle CSV, which contains only spectra for molecules with a readable DAG,
the table below restores the full selected population and counts missing-DAG spectra as zero
coverage. In NIST20-HR, missing-DAG coverage is negligible (25 of 434,917 spectra).

| Precursor | Instrument | Spectra | Missing DAG | Peak recall at inst tol | Mean cosine ceiling | Median cosine ceiling | Ceiling < 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[M+H]+` | FT | 286,490 | 0.006% | 66.81% | 0.948 | 1.000 | 3.16% |
| `[M+H]+` | QTOF | 24,573 | 0.00% | 91.03% | 0.982 | 1.000 | 1.17% |
| `[M-H]-` | FT | 121,450 | 0.007% | 72.01% | 0.973 | 1.000 | 1.72% |
| `[M-H]-` | QTOF | 2,404 | 0.00% | 92.18% | 0.991 | 1.000 | 0.54% |
| **All** | **FT + QTOF** | **434,917** | **0.006%** | **69.07%** | **0.957** | **1.000** | **2.63%** |

For comparison, the fixed 10 ppm peak recall in the two QTOF groups was 41.93% for `[M+H]+`
and 45.35% for `[M-H]-`; the instrument-tolerance floor raises these to 91.03% and 92.18%,
respectively. FT is unchanged because FraGNNet's FT rule is the same 10 ppm/200 Da contract as
the fixed baseline.

The result separates two effects:

- The instrument tolerance fixes much of the apparent QTOF mass-matching loss, especially for
  `[M+H]+`.
- DAG structural coverage is substantial in NIST20-HR: the instrument-tolerance oracle covers
  67–72% of FT peaks and 91–92% of QTOF peaks. The
  remaining unmatched peaks still warrant DAG/fragmentation and data-quality checks, but the
  earlier severe QTOF coverage failure does not persist in this high-resolution subset.

Before adopting formula-based filtering, run separate ablations for:

1. mass matching with versus without the RDBE constraint;
2. `[M-H]-` QTOF failures, including calibration, adduct/charge-carrier assumptions, and isotope
   handling; and
3. annotation-only use versus deletion of unmatched peaks or whole spectra.

Until those ablations are complete, unmatched peaks should be marked rather than deleted, and
spectra should not be discarded solely because they contain unmatched peaks.

## Saved subformula-filtered D3/D4 comparison verified 2026-09-13

This is a separate saved comparison from the historical `1_above_prec` table above.
The per-spectrum artifact was reaggregated; no DAG enumeration, preprocessing, or model inference
was rerun. Do not use the earlier fixed-10-ppm D4 cosine as intensity-weighted recall.

Artifacts:

- [Per-spectrum D3/H4 and D4/H4 metrics](../../data/eval/nist20hr_subformula_2x_inst_tol_d3_d4_metrics.csv)
- [Saved summary](../../data/eval/nist20hr_subformula_2x_inst_tol_d3_d4_summary.json)
- Cleaned input: `/tmp/nist20hr_mh_ft_qtof_2x_inst_tol_centroid_subformula_inst_tol.pkl`
- Cleaning metadata: `/tmp/nist20hr_mh_ft_qtof_2x_inst_tol_centroid_subformula_inst_tol.stats.json`

The CSV contains 434,917 unique spectrum IDs from 25,264 molecules. Its IDs exactly match the
cleaned input. The cleaning metadata enables peak merging, subformula filtering, and instrument
tolerance for both operations; this is not the historical 5 ppm-only filter. File names identify
this as the `2x_inst_tol`/centroid export, but the saved metadata does not fully establish the
historical merge implementation or multiplier. It also does not capture the complete oracle
command, source revision, or DAG fingerprints. Do not infer complete reproducibility from names.

The population is NIST20-HR `[M+H]+`/`[M-H]-` across FT and QTOF, including all available splits.
The displayed values are arithmetic means of per-spectrum metrics, not a pooled intensity ratio.
Nineteen empty spectra have NaN metrics and are excluded from these means. Missing-DAG spectra
have zero coverage: 25 spectra/3 molecules for D3, 398 spectra/30 molecules for D4. On the common
DAG-available population, mean weighted recall is 92.54% for D3 and 96.00% for D4.

| Population | Spectra | D3 weighted recall | D4 weighted recall | D4 oracle cosine |
| --- | ---: | ---: | ---: | ---: |
| All | 434,917 | 92.54% | **95.92%** | **0.9774** |
| `[M+H]+`, FT | 286,490 | 91.24% | **96.08%** | 0.9813 |
| `[M+H]+`, QTOF | 24,573 | 97.38% | 98.55% | 0.9932 |
| `[M-H]-`, FT | 121,450 | 94.49% | 94.96% | 0.9650 |
| `[M-H]-`, QTOF | 2,404 | 98.40% | 98.09% | 0.9850 |
| `[M+H]+`, FT, validation IDs only | 57,816 | 91.18% | **96.10%** | — |

Validation membership uses `data/split/nist20_ma_mi_inchikey/val_ids.csv`; it does not imply that
these individually cleaned spectra reproduce merged MASSter validation targets. Overall
unweighted peak recall is 86.59% for D3 and 92.75% for D4. Weighted recall measures covered target
intensity; oracle cosine measures ideal intensity assignment on reachable support. They are
separate quantities and must not be substituted for each other.

### Implication for MASSter and EBM priorities

The [proposal review](masster/masster_proposal_review_2026_09_13.md) records roughly 94% candidate
weighted recall for filtered merged MASSter runs. Against this saved D4 reference, that is
numerically about two percentage points lower, so the observed MASSter coverage is strong.
The two values are **not a matched gap estimate**: target merging/cleaning, evaluation IDs,
acquisition mix, channel chemistry, matching, and metric aggregation must be aligned first.
D4 is not a universal ceiling on MASSter chemistry or search.

Do not treat the entire remaining 6% of MASSter intensity as recoverable through better
bond-cut scores, or claim that an EBM would recover the numerical two-point difference. Before
an architecture change, measure the current MASSter support oracle and learned cosine on
identical targets. A large support-oracle versus predicted-cosine gap prioritizes intensity
assignment. Bond-cut modeling needs demonstrated additional support, lower cost at preserved
coverage, or better downstream spectra at matched coverage; nominal U256 underfilling is not
sufficient motivation. The existing user-requested bond-cut-first sequence remains a diagnostic
order, not evidence that it offers the largest improvement.

Validation: reaggregated all six D3/D4 metric means and matched the summary to absolute tolerance
`1e-12`; verified spectrum-ID uniqueness, cleaned-input ID equality, missing-DAG zero coverage,
and metadata joins. Slice means were recomputed from the saved CSV and cleaned-input metadata.
No scientific comparison with a MASSter checkpoint was run. Next: a matched support/intensity
comparison, preserving this historical artifact rather than silently replacing its protocol.

## Reproduction

Current four-adduct instrument-tolerance cleaning command:

```bash
python -m fragnnet.utils.clean_spectra_utils \
  --spec-fp data/proc/nist20/spec_df.pkl \
  --mol-fp data/proc/nist20/mol_df.pkl \
  --output-fp /tmp/nist20hr_four_adduct_inst_tol_filtered.pkl \
  --stats-fp /tmp/nist20hr_four_adduct_inst_tol_filtered_stats.json \
  --skip-unsupported-formulas \
  --dsets nist20_hr \
  --precursor-types "[M+H]+" "[M-H]-" "[M+Na]+" "[M+Cl]-" \
  --instrument-types FT QTOF \
  --merge-peaks --merge-inst-tol --merge-method common_overlap \
  --filter-subformula-peaks --subformula-inst-tol
```

Use this cleaned table with the D3/H4 and D4/H4 oracle command above. Keep CMF enabled (omit
`--no_cmf`) and set `--max_h_transfer 4`. The commands below reproduce the historical two-adduct
tables.

Full subformula census:

```bash
python -m fragnnet.utils.clean_spectra_utils \
  --spec-fp data/proc/nist20/spec_df.pkl \
  --mol-fp data/proc/nist20/mol_df.pkl \
  --stats-fp /tmp/nist20hr_mh_ft_qtof_cleanup_stats.json \
  --stats-only \
  --skip-unsupported-formulas \
  --dsets nist20_hr \
  --precursor-types "[M+H]+" "[M-H]-" \
  --instrument-types FT QTOF \
  --merge-peaks --merge-ppm 20 \
  --filter-subformula-peaks --subformula-ppm 5 \
  --drop-empty-spectra
```

Grouped merge-only control:

```bash
python -m fragnnet.utils.clean_spectra_utils \
  --spec-fp data/proc/nist20/spec_df.pkl \
  --stats-fp /tmp/nist20hr_mh_ft_qtof_merge_stats.json \
  --stats-only \
  --dsets nist20_hr \
  --precursor-types "[M+H]+" "[M-H]-" \
  --instrument-types FT QTOF \
  --merge-peaks --merge-ppm 20
```

FraGNNet instrument-tolerance census:

```bash
python -m fragnnet.utils.clean_spectra_utils \
  --spec-fp data/proc/nist20/spec_df.pkl \
  --mol-fp data/proc/nist20/mol_df.pkl \
  --stats-fp /tmp/nist20hr_mh_ft_qtof_inst_tol_stats.json \
  --stats-only \
  --skip-unsupported-formulas \
  --dsets nist20_hr \
  --precursor-types "[M+H]+" "[M-H]-" \
  --instrument-types FT QTOF \
  --merge-peaks --merge-ppm 20 \
  --filter-subformula-peaks --subformula-inst-tol \
  --drop-empty-spectra
```

Human-review queue after instrument-tolerance filtering. This filtering run deliberately omits
`--drop-empty-spectra` so the review detector sees empty results:

```bash
python -m fragnnet.utils.clean_spectra_utils \
  --spec-fp data/proc/nist20/spec_df.pkl \
  --mol-fp data/proc/nist20/mol_df.pkl \
  --output-fp /tmp/nist20hr_mh_ft_qtof_inst_tol_filtered.pkl \
  --stats-fp /tmp/nist20hr_mh_ft_qtof_inst_tol_filtered_stats.json \
  --skip-unsupported-formulas \
  --dsets nist20_hr \
  --precursor-types "[M+H]+" "[M-H]-" \
  --instrument-types FT QTOF \
  --merge-peaks --merge-ppm 20 \
  --filter-subformula-peaks --subformula-inst-tol

python scripts/eval/nist_bad_data_review.py \
  --spec_fp /tmp/nist20hr_mh_ft_qtof_inst_tol_filtered.pkl \
  --mol_fp data/proc/nist20/mol_df.pkl \
  --dsets nist20_hr \
  --out_csv /tmp/nist20hr_mh_ft_qtof_inst_tol_human_review.csv
```

DAG coverage summary (uses the existing oracle output and counts missing DAG files as zero):

- Oracle input: `data/eval/oracle_cos_nist20_default/oracle_cos_default.csv`
- NIST20-HR summary: `/tmp/nist20hr_mh_ft_qtof_dag_coverage_inst_tol.json`
- Oracle scorer source: `scripts/eval/dag_oracle_scores.py`

NIST20-HR-only reports:

- `/tmp/nist20hr_mh_ft_qtof_cleanup_stats.json`
- `/tmp/nist20hr_mh_ft_qtof_merge_stats.json`
- `/tmp/nist20hr_mh_ft_qtof_inst_tol_stats.json`
- `/tmp/nist20hr_mh_ft_qtof_inst_tol_filtered_stats.json`
- `/tmp/nist20hr_mh_ft_qtof_inst_tol_human_review.csv`

## Cl/Na budget fix, measured on the multi-adduct split (2026-09-16)

Commits `2922ef7e` and `04133d45` (merged here at `ba9192b7`) rewrote the candidate enumeration.
The filter now bounds each candidate by a **precursor ion** rather than by the neutral `M`, so a
fragment may retain part of the charge carrier, give it up entirely, or be the bare carrier; RDBE
is scored on the molecular part with the carrier stripped, which is what `[M+Cl]-` needs.

Measured pre-fix (`04133d45^`) against post-fix on 2,620 stratified `nist20_ma_mi_inchikey`
validation spectra, at instrument tolerance (FT 10 ppm/200 Da floor, QTOF 20 ppm/750 Da floor):

| Adduct | Instrument | Raw peaks | Pre-fix kept | Post-fix kept | Delta |
|---|---|---:|---:|---:|---:|
| `[M+H]+` | FT | 13,710 | 12,439 | 12,497 | +58 |
| `[M+H]+` | QTOF | 7,839 | 7,577 | 7,582 | +5 |
| `[M-H]-` | FT | 13,438 | 12,357 | 12,357 | 0 |
| `[M-H]-` | QTOF | 7,120 | 6,855 | 6,855 | 0 |
| `[M+Na]+` | FT | 9,799 | 8,477 | 8,481 | +4 |
| `[M+Na]+` | QTOF | 2,214 | 1,742 | 1,742 | 0 |
| `[M+Cl]-` | FT | 8,582 | 7,779 | 7,802 | +23 |

The fix is **directionally safe but small on this population**: +90 peaks of 62,702 (+0.14%), and
every cell is nonnegative — the new enumeration only ever admits peaks, never deletes more than
the old one did. The old code raised no errors on any of these spectra, so this is a retention
change, not a crash fix. Do not read the per-adduct deltas as a scientific gain; they are a
correctness check on the enumeration.

Retention at instrument tolerance on the same sample, for planning target density (this is the
instrument-tolerance contract, not the 5 ppm tables above):

| Adduct | Instrument | Peaks kept | Intensity kept | Spectra emptied |
|---|---|---:|---:|---:|
| `[M+H]+` | FT | 91.2% | 97.1% | 0 |
| `[M+H]+` | QTOF | 96.7% | 98.5% | 0 |
| `[M-H]-` | FT | 92.0% | 98.4% | 0 |
| `[M-H]-` | QTOF | 96.3% | 98.2% | 0 |
| `[M+Na]+` | FT | 86.6% | 97.0% | 0 |
| `[M+Na]+` | QTOF | 78.7% | 93.6% | 0 |
| `[M+Cl]-` | FT | 90.9% | 98.7% | 0 |

### MASSter scope

The fix reaches MASSter through the shared `BaseDataset` cleanup with no code change, but it
was inert for every MASSter config: all of them point at `data/split/nist20_inchikey`, which
holds 209,698 `[M+H]+`/FT spectra and no Na, Cl, QTOF, or IT. `data/split/nist20_ma_mi_inchikey`
carries 480,002 spectra over all four adducts and FT+QTOF (45,976 Na/Cl; **no IT rows**, so the
FT/QTOF-only `INSTRUMENT_TOLERANCES` table covers it and does not raise).

MASSter's own channel enumeration agrees with the filter on which adducts have a real
charge-migration channel: at `max_h_transfer=4`, `cmf_precursor_valid` is True for `[M+Na]+` and
`[M+Cl]-` and False for `[M+H]+`/`[M-H]-`, whose CMF state duplicates CRF h_shift -1/+1 — the same
reason `_precursor_ion_budgets` gives those two adducts a single budget. Over all sub-multisets of
a C6H8N2O3 stand-in, the filter rejects 21.6% of the channels MASSter can emit for `[M+H]+`
against 24.5%/21.4% for `[M+Na]+`/`[M+Cl]-`; the rejects are over-hydrogenated negative-RDBE
channels, so the filter is stricter in the same way for every adduct and creates no MASSter-
specific asymmetry.
