# Instrument Dependence of 5 ppm Subformula Filtering

> **Status:** FINDING &nbsp;·&nbsp; **Updated:** 2026-08-12
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

## Results

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

## Conclusion

Use of 20 ppm peak merging is supported for both FT and QTOF in this dataset. A universal 5 ppm
subformula filter is **not supported**, particularly for QTOF: it removes most measured intensity
and creates a large instrument-specific selection bias.

Before adopting formula-based filtering, run separate ablations for:

1. instrument-appropriate tolerances, such as FT at 5 or 10 ppm and QTOF at 20 ppm;
2. mass matching with versus without the RDBE constraint; and
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

Generated reports:

- `/tmp/nist20_mh_ft_qtof_cleanup_stats.json`
- `/tmp/nist20_mh_ft_qtof_merge_stats.json`
