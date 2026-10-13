# Spectraverse subformula filter tolerance screen

> **Status:** STOPPED · **Updated:** 2026-10-06
> The precursor-bounded subformula filter is not enabled for SPV training (removes 26.2% of intensity); fixed 0.05 Da artifact generated, no training launched, temporary configs withdrawn.

## Decision

Do not enable the current precursor-bounded subformula filter for Spectraverse training. With the configured instrument tolerances, it removes **26.2% of total target intensity** and the original base peak from **15.2% of 488,797 spectra**. It leaves 15,074 spectra (3.1%) empty. This is a target change too large to treat as routine cleaning. The ten temporary filtered fold configs were withdrawn; the original SPV configs and production filter behavior remain unchanged.

Widening the subformula window does not resolve the problem at a reasonable scale. On 8,192 fixed held-out spectra (4,096 from each of cv0 and cv1, seed 20261001), the original base peak is retained in 84.1% at the current window, 86.3% at 2×, 89.4% at 5×, and 92.5% at 10×. Tenfold means a 0.02 Da low-mass floor for OB and a 0.15 Da floor for QTOF. A wide window also admits accidental subformula matches, so these are retention ceilings, not verified chemical assignments.

| Adduct | Full-table spectra | Intensity retained at 1× | Base peak retained at 1× | Sample base peak retained at 2× / 10× |
| --- | ---: | ---: | ---: | ---: |
| `[M+H]+` | 233,103 | 77.3% | 86.9% | 89.3% / 94.2% |
| `[M+Na]+` | 69,475 | 53.7% | 60.7% | 61.4% / 76.8% |
| `[M+K]+` | 9,508 | 49.7% | 40.5% | 50.3% / 69.0% |
| `[M-H]-` | 107,867 | 70.9% | 94.9% | 92.8% / 97.0% |

The sodium and potassium losses dominate the concern. Among 10×-window misses on the 8,192-spectrum sample, 43.2% of sodium and 20.8% of potassium base peaks sit more than 0.5 Da above the precursor; only 2.6% and 6.2%, respectively, are within 0.1 Da of the +1.003 Da isotope offset. This is not mainly a simple +1 isotope case. Precursors themselves are much closer to theoretical m/z: absolute precursor-error p95 is 6.08 ppm for OB and 7.55 ppm for QTOF (excluding three tin spectra). That does not establish fragment calibration, but it weakens a simple whole-spectrum mass-offset explanation. Diagnose the Na/K peak formulas or adduct metadata before attempting another filter arm.

## Method and limits

The full census used `data/proc/spectraverse/spec_df.pkl` and `mol_df.pkl`, with the existing `filter_subformula_peaks` chemistry at the configured instrument windows: OB 10 ppm / 200 Da, QTOF 20 ppm / 750 Da, and the SPV loss settings of 20 ppm / 500 Da for Other and NA. Three spectra with an unsupported Sn formula were left unfiltered and counted. `drop_empty_spectra=True` removed empty spectra after filtering. Exact stage and per-adduct statistics, plus retained train/validation/test counts for all ten fixed CV splits, are in `output/analysis/spectraverse_subformula_2026_10_01/census.json` (ignored artifact store).

The tolerance sweep reused the 8,192 validation spectrum IDs in `output/analysis/spectraverse_recall_2026_10_01/`. For each measured peak, it found the closest mass from `_subformula_masses` and compared that distance with 1×, 1.5×, 2×, 3×, 5× and 10× the instrument window. Per-spectrum outcomes are in `output/analysis/spectraverse_subformula_2026_10_01/tolerance_sweep_8192.csv`. The sweep used the same candidate chemistry as the production filter and reproduces the 1× retention on that sample. No target spectra, split files, or model checkpoints were modified. The study does not determine whether peaks rejected at wide windows are contamination, adduct mislabels, isotope peaks, or real fragments absent from the candidate chemistry.

## Follow-up: 20 ppm centroiding and ±0.05 Da subformula filter

A user-requested full SPV data pass used 20 ppm naive-centroid merging followed by a fixed
±0.05 Da precursor-bounded subformula match. The output is
`data/proc/spectraverse/spec_df_centroid20ppm_subformula005da.pkl`; stage statistics are in
`output/analysis/spectraverse_centroid20ppm_subformula005da_stats.json`. Both are ignored
local artifacts. The source pickle and existing training configs were unchanged. The three
spectra with unsupported Sn formulas were excluded, leaving 488,794 processed spectra.
Empty spectra were retained.

Centroiding reduced 40,242,565 peaks to 38,854,682 without changing total intensity.
The subformula filter retained 28,204,599 peaks (72.59% of centroided peaks) and 83.73%
of centroided intensity. It left 5,789 spectra empty and retained the centroided base
peak in 450,309 spectra. These counts describe the processed target data, not model
performance or verified chemical assignments. The earlier concern about Na/K adduct
chemistry remains unresolved; no training run was launched.
