# Instrument Dependence of 5 ppm Subformula Filtering

> **Status:** FINDING &nbsp;·&nbsp; **Updated:** 2026-08-13
>
> Full NIST20 census for `[M+H]+` and `[M-H]-`, stratified by FT and QTOF.

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

## Population

The source was `data/proc/nist20/spec_df.pkl`:

- Source table: 1,148,072 spectra.
- Selected `[M+H]+`/`[M-H]-` and FT/QTOF: 523,911 spectra.
- Processed: 523,902 spectra containing 27,236,569 peaks.
- Excluded explicitly: 9 `[M+H]+` FT spectra from one molecule containing `Sn`, because the
  repository RDBE element table does not support tin.

No sampling was used. Empty spectra were counted after filtering and then dropped from the output
count. The run was stats-only, so it did not write a duplicate cleaned spectrum table.

## Fixed 5 ppm results

| Precursor | Instrument | Spectra | Peaks retained by 20 ppm merge | Peaks retained by 5 ppm subformula filter | Intensity retained by subformula filter | Empty after subformula filter | Base peak retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[M+H]+` | FT | 316,922 | 99.50% | 69.84% | 90.08% | 0.69% | 92.71% |
| `[M+H]+` | QTOF | 50,448 | 99.51% | 17.32% | 24.63% | 15.95% | 41.63% |
| `[M-H]-` | FT | 138,801 | 99.64% | 59.06% | 90.59% | 2.01% | 89.45% |
| `[M-H]-` | QTOF | 17,731 | 99.51% | 10.41% | 20.43% | 15.42% | 51.06% |

Across all four groups, 20 ppm merging retained 99.54% of peak records and 100% of total
intensity. It produced no empty spectra and retained the base peak in every spectrum. The merge is
therefore a low-impact cleanup operation on this population.

The universal 5 ppm subformula filter is strongly instrument-dependent:

- FT retains 59–70% of peak records and about 90% of intensity. Although substantially more
  aggressive than merging, most dominant signal survives.
- QTOF retains only 10–17% of peak records and 20–25% of intensity. Approximately 15–16% of QTOF
  spectra become empty, and the original base peak survives in only 42–51%.

## FraGNNet instrument-tolerance results

The census was repeated with the exact evaluation-time matching contract:

- FT: `10 ppm × max(observed m/z, 200)`, giving a minimum ±0.002 Da window.
- QTOF: `20 ppm × max(observed m/z, 750)`, giving a minimum ±0.015 Da window.

Only the subformula matching tolerance changed. Peak merging remained fixed at 20 ppm, and the
precursor-bounded formula, adduct, removable-hydrogen, and RDBE rules were unchanged.

| Precursor | Instrument | Peaks: fixed 5 ppm | Peaks: instrument tolerance | Intensity: fixed 5 ppm | Intensity: instrument tolerance | Empty: fixed 5 ppm | Empty: instrument tolerance | Base peak: instrument tolerance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `[M+H]+` | FT | 69.84% | **76.78%** | 90.08% | **96.92%** | 0.69% | **0.04%** | 98.36% |
| `[M+H]+` | QTOF | 17.32% | **52.26%** | 24.63% | **91.86%** | 15.95% | **0.07%** | 98.87% |
| `[M-H]-` | FT | 59.06% | **69.04%** | 90.59% | **97.88%** | 2.01% | **0.07%** | 98.47% |
| `[M-H]-` | QTOF | 10.41% | **34.25%** | 20.43% | **63.68%** | 15.42% | **0.69%** | 96.85% |

Across all four groups, instrument-tolerance filtering retained 67.27% of peak records, 96.37% of
total intensity, and the base peak in 98.39% of spectra. Only 361 of 523,902 spectra (0.069%) became
empty, compared with 15,749 (3.01%) under fixed 5 ppm.

The QTOF recovery is driven in large part by its low-mass floor. Below m/z 750, the matching window
is ±0.015 Da rather than literal 20 ppm; at m/z 100 this is equivalent to 150 ppm. The improvement
therefore validates use of FraGNNet's established evaluation contract, but should not be described
as a simple 20 ppm result.

## Conclusion

Use of 20 ppm peak merging is supported for both FT and QTOF in this dataset. A universal 5 ppm
subformula filter is **not supported**, particularly for QTOF. FraGNNet's instrument tolerance
largely resolves the loss for FT and `[M+H]+` QTOF: these groups retain 92–98% of intensity and
98–99% of base peaks. `[M-H]-` QTOF remains a concern, retaining only 34% of peak records and 64%
of intensity even though its base peak usually survives.

## DAG coverage at FraGNNet instrument tolerance

The existing NIST20 DAG oracle was summarized for the same `[M+H]+`/`[M-H]-` × FT/QTOF scope. The
coverage tier is `1_above_prec`, and the matching tolerance is FraGNNet's instrument tolerance
(FT 10 ppm / 200 Da floor; QTOF 20 ppm / 750 Da floor). `peak recall` is the number of observed
peaks matched by at least one DAG ion divided by the number of peaks in that tier. `cosine ceiling`
is the ideal intensity-weighted cosine score if the DAG-achievable peaks receive their true
intensities. It is an upper bound on a model using that DAG.

Unlike the historical oracle CSV, which contains only spectra for molecules with a readable DAG,
the table below restores the full selected population and counts missing-DAG spectra as zero
coverage. This is important for QTOF, where missing-DAG rates are higher.

| Precursor | Instrument | Spectra | Missing DAG | Peak recall at inst tol | Mean cosine ceiling | Median cosine ceiling | Ceiling < 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[M+H]+` | FT | 316,931 | 2.86% | 52.55% | 0.917 | 1.000 | 6.34% |
| `[M+H]+` | QTOF | 50,448 | 12.47% | 27.60% | 0.850 | 1.000 | 14.12% |
| `[M-H]-` | FT | 138,801 | 4.14% | 50.77% | 0.924 | 1.000 | 6.69% |
| `[M-H]-` | QTOF | 17,731 | 20.75% | 17.22% | 0.757 | 0.999 | 23.63% |
| **All** | **FT + QTOF** | **523,911** | **4.73%** | **44.82%** | **0.907** | **1.000** | **7.76%** |

For comparison, the fixed 10 ppm peak recall in the two QTOF groups was only 11.99% for
`[M+H]+` and 6.81% for `[M-H]-`; the instrument-tolerance floor raises these to 27.60% and
17.22%, respectively. FT is unchanged because FraGNNet's FT rule is the same 10 ppm/200 Da
contract as the fixed baseline.

The result separates two effects:

- The instrument tolerance fixes much of the apparent QTOF mass-matching loss, especially for
  `[M+H]+`.
- DAG structural coverage remains limited: even after the correct tolerance, roughly half of FT
  peaks and more than 80% of QTOF `[M-H]-` peaks are not DAG-achievable. This is not a formula
  filtering artifact alone and should be investigated as a DAG/fragmentation or data-quality
  issue.

Before adopting formula-based filtering, run separate ablations for:

1. mass matching with versus without the RDBE constraint;
2. `[M-H]-` QTOF failures, including calibration, adduct/charge-carrier assumptions, and isotope
   handling; and
3. annotation-only use versus deletion of unmatched peaks or whole spectra.

Until those ablations are complete, unmatched peaks should be marked rather than deleted, and
spectra should not be discarded solely because they contain unmatched peaks.

## Reproduction

Full subformula census:

```bash
python preproc_scripts/clean_spectra.py \
  --spec-fp data/proc/nist20/spec_df.pkl \
  --mol-fp data/proc/nist20/mol_df.pkl \
  --stats-fp /tmp/nist20_mh_ft_qtof_cleanup_stats.json \
  --stats-only \
  --skip-unsupported-formulas \
  --precursor-types "[M+H]+" "[M-H]-" \
  --instrument-types FT QTOF \
  --merge-peaks --merge-ppm 20 \
  --filter-subformula-peaks --subformula-ppm 5 \
  --drop-empty-spectra
```

Grouped merge-only control:

```bash
python preproc_scripts/clean_spectra.py \
  --spec-fp data/proc/nist20/spec_df.pkl \
  --stats-fp /tmp/nist20_mh_ft_qtof_merge_stats.json \
  --stats-only \
  --precursor-types "[M+H]+" "[M-H]-" \
  --instrument-types FT QTOF \
  --merge-peaks --merge-ppm 20
```

FraGNNet instrument-tolerance census:

```bash
python preproc_scripts/clean_spectra.py \
  --spec-fp data/proc/nist20/spec_df.pkl \
  --mol-fp data/proc/nist20/mol_df.pkl \
  --stats-fp /tmp/nist20_mh_ft_qtof_inst_tol_stats.json \
  --stats-only \
  --skip-unsupported-formulas \
  --precursor-types "[M+H]+" "[M-H]-" \
  --instrument-types FT QTOF \
  --merge-peaks --merge-ppm 20 \
  --filter-subformula-peaks --subformula-inst-tol \
  --drop-empty-spectra
```

DAG coverage summary (uses the existing oracle output and counts missing DAG files as zero):

- Oracle input: `data/eval/oracle_cos_nist20_default/oracle_cos_default.csv`
- Full-scope summary: `/tmp/nist20_mh_ft_qtof_dag_coverage_inst_tol.json`
- Oracle scorer source: `scripts/eval/dag_oracle_scores.py`

Generated reports:

- `/tmp/nist20_mh_ft_qtof_cleanup_stats.json`
- `/tmp/nist20_mh_ft_qtof_merge_stats.json`
- `/tmp/nist20_mh_ft_qtof_inst_tol_stats.json`
