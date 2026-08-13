# 5 ppm Subformula Filtering on NIST20-HR

> **Status:** FINDING &nbsp;·&nbsp; **Updated:** 2026-08-13
>
> NIST20-HR-only census for `[M+H]+` and `[M-H]-`, stratified by FT and QTOF.

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

## Conclusion

Use of 20 ppm peak merging is supported for both FT and QTOF in NIST20-HR. The 5 ppm
subformula filter is **not supported**, particularly for QTOF. FraGNNet's instrument tolerance
largely resolves the loss for all four NIST20-HR groups: they retain 97–98.5% of intensity and
98.9–99.8% of base peaks. The 5 ppm QTOF result remains poor, but the instrument-tolerance
floor recovers it for this high-resolution subset.

## DAG coverage at FraGNNet instrument tolerance

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

## Reproduction

Full subformula census:

```bash
python preproc_scripts/clean_spectra.py \
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
python preproc_scripts/clean_spectra.py \
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
python preproc_scripts/clean_spectra.py \
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

DAG coverage summary (uses the existing oracle output and counts missing DAG files as zero):

- Oracle input: `data/eval/oracle_cos_nist20_default/oracle_cos_default.csv`
- NIST20-HR summary: `/tmp/nist20hr_mh_ft_qtof_dag_coverage_inst_tol.json`
- Oracle scorer source: `scripts/eval/dag_oracle_scores.py`

NIST20-HR-only reports:

- `/tmp/nist20hr_mh_ft_qtof_cleanup_stats.json`
- `/tmp/nist20hr_mh_ft_qtof_merge_stats.json`
- `/tmp/nist20hr_mh_ft_qtof_inst_tol_stats.json`
