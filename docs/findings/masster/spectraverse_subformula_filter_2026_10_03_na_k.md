# Why Spectraverse `[M+K]+` and `[M+Na]+` spectra are unexplained

> **Status:** FINDING · **Updated:** 2026-10-03 · Full SPV table, no model. Answers the "inspect
> Na/K peak chemistry" next step of the
> [subformula filter screen](spectraverse_subformula_filter_2026_10_01.md).

"Bad" means a peak-level subformula ceiling ≤ 0.5 (`data/proc/spectraverse/subformula_ceiling.csv`,
built by `preproc_scripts/spectraverse/clean_train_split.py`). Scripts:
`output/analysis/spectraverse_mk_2026_10_03/mk_probe{,2}.py` (ignored).

## `[M+K]+`: 42% bad, a potassium problem, not a source problem

- **Labels are right.** All 9,508 precursor m/z values sit within 0.01 Da of the theoretical
  `[M+K]+` mass (median error 0.0001 Da). Relabelling bad spectra as `[M+H]+`, `[M+Na]+` or
  `[M+NH4]+` lowers or barely moves the ceiling (0.22 / 0.25 / 0.33 against 0.24).
- **Not the dirty sources.** Grouping spectra by `dset_spec_id` blocks of 20,000, `[M+K]+` is
  38–63% bad on ORB even in blocks where `[M+H]+` is only 2–4% bad. None of the bad `[M+K]+`
  spectra fall in blocks where `[M+H]+` is more than 20% bad. ORB is 46% bad, QTOF 13%.
- **The analyte signal is mostly missing.** The precursor is absent from 82% of `[M+K]+`
  spectra, good and bad alike, against 42% for ORB `[M+H]+`. Bare K+ (38.963) never appears;
  the median lowest recorded m/z is 53–55. The textbook main channel of alkali adducts under CID
  is loss of the metal cation, which here falls below the scan range. That is an inference: the
  K+ ion cannot be seen in this data.
- **What fills the spectrum is background.** 99.6% of bad spectra have an unexplained base
  peak, 66% of their intensity sits below the precursor, and the same peaks recur across
  hundreds of molecules (exact median m/z, brute-force formula within 1 mDa):

| m/z | Formula | Spectra / molecules | Likely source |
| --- | --- | ---: | --- |
| 56.9651 | CaOH+ | 2,087 / 654 | calcium salt |
| 98.9755 | Ca(C2H3O2)+ | 1,993 / 578 | calcium acetate |
| 73.0469 | Si(CH3)3+ | 1,354 / 410 | siloxane / silylation |
| 61.0109 | C2H5S+ | 3,001 / 832 | likely a DMSO-type solvent ion |
| 91.0548 | C7H7+ (0.6 mDa) | 4,391 / 905 | tropylium; not a subformula of these molecules at tolerance |
| 222.8987, 240.9092, 281.9119, 360.8389, 419.8520 | not unique | 128–547 molecules each | strongly negative mass defect: inorganic salt clusters |

Bad spectra also have lower median ACE (32 against 43 eV) and fewer peaks (252 against 339).

## `[M+Na]+`: mostly a source problem

`[M+Na]+` is 39% bad inside the source blocks where `[M+H]+` is also >20% bad (50,250 spectra),
and 17% bad elsewhere (19,225). In those blocks `[M+H]+` itself is 31% bad, against 3% in clean
blocks. Bad ORB `[M+Na]+` and `[M+H]+` spectra share a precursor-absent rate near 76–78%.

## Consequence

Dropping the bad spectra from training (`masster_subceil05`) removes mostly background-only K
spectra and the dirty-source block, not hard chemistry. A filter by source block would catch the
`[M+H]+`/`[M+Na]+` part but not the K part.
