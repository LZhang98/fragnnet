# HR-EI raw peak and structure audit

> **Status:** FINDING · **Updated:** 2026-10-17
> All four libraries contain fractional-mass peak tables; supplied structure mappings require source-specific exclusions before a prediction benchmark.

## Scope and reproduction

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

## Observed peak and mapping checks

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

## Source-specific issues

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

## Overlap and benchmark implications

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

## Validation

Four focused tests pass for MSP annotations/record boundaries, JCAMP multiple
pairs/records, malformed peak rejection, and formula-order comparison. Ruff checks
pass for the audit script and tests. The complete audit exits successfully and
accounts for every primary record. These checks validate the audit mechanics;
they do not certify experimental mass accuracy or chemical identity.
