# Enveda-180 dataset: contents and preprocessing

> **Status:** FINDING · **Updated:** 2026-09-25 · Steps 1-2 (df, proc) have run locally on the
> filtered export. There is no split and no DAG yet, and the split choice is open (§7).

## 1. Source

- The local copy is `data/raw/enveda-180/`, downloaded 2026-07-08. The release is
  [enveda/enveda-180](https://github.com/enveda/enveda-180). Its README does not describe the
  filtered export, the CE field or a split, so §2-§4 are measured from the files.
- 1,158,293 spectra of 184,330 unique source SMILES (183,476 InChIKey first blocks), all Enamine
  synthetic compounds. One instrument: Bruker timsTOF Pro 2 (`Q-TOF`), ESI, HPLC on C18.
  There are 891,903 positive and 266,390 negative spectra.
- Each record also carries CCS, ion mobility, retention time, an MS1 isotope pattern,
  per-spectrum mass error and explained-intensity fraction, and Enamine and PubChem ids. The df
  CSV keeps the scalar fields and drops the isotope pattern.
- Molecules barely overlap our other sets. Of the 183,476 standardized molecules, NIST23 shares
  66, NIST20 46, MSGYM 1.5 (`ms_gym_all_1_5`) 26 and Spectraverse 26, matched by InChIKey
  first block.

## 2. The `-filtered` export is a 1% base-peak floor

- `enveda-180-filtered.jsonl.gz` holds the same spectra in the same order as
  `enveda-180.jsonl.gz`.
- Its peaks are exactly the raw peaks at or above 1% of the base peak. That holds for all of the
  first 20,000 spectra. The median spectrum drops from 138 peaks to 14, and p99 from 1,592 to
  104.
- It is not a subformula filter: the first spectrum keeps 27 peaks, and 21 of them are explained.

We preprocess the filtered export. The raw export differs only in its sub-1% peaks.

## 3. Collision energy

- Energies are 20, 40 and 60 eV. Negative-mode records write them with the polarity sign
  (`-40`), which the reader drops.
- 850,373 spectra have one energy. 305,974 list three (`20,40,60` or `-60,-40,-20`), and 1,946
  list two.
- A multi-energy record is a separate stepped-CE acquisition, not a merge of the single-energy
  spectra. It has its own retention time and peaks that no single-energy spectrum has. For
  `enveda_000003` `[2M+Na]+`, none of its 3 peaks appear among the 8 of the single-energy spectra.
- The reader writes the energies in ascending order to `col_energy`, `col_energy_extra_1` and
  `col_energy_extra_2`. Three energies give `ce_type: stepped`.
- **Two energies give `ce_type: ramped`**, the pipeline's label for any two-value CE.
  [`base_dataset`](../../src/fragnnet/dataset/base_dataset.py) expands a ramped pair in 1 eV
  steps, so 20 and 40 become 21 energies. Enveda has no ramped spectra, so pass
  `--ce_types stepped single` to `04_prepare_split.py` to drop exactly these 1,946.
- No NCE is recorded. `nce` is empty in `spec_df` and must be filled from `ace` if a model reads
  NCE.

## 4. Adducts

| Adduct | Spectra | Share |
| --- | ---: | ---: |
| `[M+H]+` | 590,683 | 51.0% |
| `[2M+Na]+` | 188,432 | 16.3% |
| `[M-H]-` | 185,030 | 16.0% |
| `[2M+H]+` | 89,625 | 7.7% |
| `[2M-H]-` | 53,005 | 4.6% |
| `[M+FA-H]-` (stored as `[M+CH2O2-H]-`) | 24,648 | 2.1% |
| `[M+Na]+` | 16,153 | 1.4% |
| `[M+NH4]+` | 5,913 | 0.5% |
| `[M+Cl]-` | 2,410 | 0.2% |
| `[M+K]+` | 1,097 | 0.1% |
| `[2M+FA-H]-`, `[M+Hac-H]-`, `[2M+Hac-H]-`, `[M+Br]-` | 1,297 | 0.1% |

- Dimer precursors (`[2M+...]`) are 331,923 spectra, 28.7%.
- The proc tables keep every adduct; the split step selects them with `--prec_types`.

## 5. Preprocessing run (2026-09-25)

Run on the workstation from `27c7e542` plus the uncommitted reader diff. Provenance and logs
are in `output/enveda_180_preproc_2026_09_25/logs/`: `git_rev.txt`, `01_prepare_df.diff`,
`01_prepare_df.log` and `02_prepare_proc.log`.

```bash
python preproc_scripts/01_prepare_df.py --input_format enveda \
    --jsonl_file enveda-180/enveda-180-filtered.jsonl.gz --output_format csv \
    --output_name enveda_180
python preproc_scripts/02_prepare_proc.py --df_dp data/df --dsets enveda_180 \
    --proc_dp data/proc/enveda_180
```

| Output | Size | Contents |
| --- | ---: | --- |
| `data/df/enveda_180_df.csv` | 909 MB | 1,158,293 rows; step 1 took 80 s |
| `data/proc/enveda_180/spec_df.pkl` | 655 MB | 1,158,293 spectra, 351,024 groups |
| `data/proc/enveda_180/mol_df.pkl` | 119 MB | 183,476 molecules |

- Step 2 took 29.5 min on 12 cores, with 23.3 GB peak RSS; it swapped on this 32 GB machine.
- No spectrum was dropped: there were no unparsable molecules, no formula mismatches and no mass
  mismatches.
- Standardization (stereo removed, tautomers canonicalized) merged 184,330 source SMILES into
  183,476 molecules.
- For 910 molecules (6,368 spectra), the source InChIKey first block differs from the
  standardized molecule's. This is diagnostic only; those spectra are in
  `diff_inchikey_df.pkl`.
- Checks on the proc tables:
  - Precursor m/z against the standardized molecule is within 4.1 ppm at p99 for `[M+H]+` and
    2.7 ppm for `[M-H]-`, with none past 20 ppm.
  - Every precursor is below m/z 1,000.
  - `inst_type` is `QTOF` and `frag_mode` is `CID` throughout.
  - `ann_df` is empty, since the source has no peak annotations.

## 6. Code

- `process_enveda` and `parse_enveda_ce_str` in
  [01_prepare_df.py](../../preproc_scripts/01_prepare_df.py) implement `--input_format enveda`.
- Tests: `tests/test_01_prepare_df.py`, where 11 pass.
- The command recipe is in [preproc_scripts/README.md](../../preproc_scripts/README.md).

## 7. Open: the split

The release ships no split or fold column. These choices for `04_prepare_split.py` are not made:

- Split key: a random 2D-InChIKey split as for NIST (`--split_type random --split_key
  inchikey_s`), or a scaffold split.
- Use: train alongside NIST/MSGYM, or hold Enveda out as a timsTOF QTOF test set. The overlap in
  §1 is small enough for either.
- Which adducts and CE types to keep (§3, §4). FraGNNet DAGs (step 3) take the same adduct
  filters and are not generated; MASSter needs no DAGs.
