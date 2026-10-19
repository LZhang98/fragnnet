# ZenoTOF CID + EAD libraries: MultiMS2, MS-DIAL 5 standards, CIeaD

> **Status:** FINDING · **Updated:** 2026-10-15
> MultiMS2 and MS-DIAL 5 are preprocessed (steps 1-2) with a new `EAD` frag_mode; CIeaD is deferred because it ships no structures. No split, no training run.

Three public SCIEX ZenoTOF 7600 libraries with both collision-induced (CID) and electron-activated
(EAD) dissociation spectra. None of our other datasets has EAD; Spectraverse 1.0.2, MassBank 2026.03
and MoNA have no ZenoTOF or EAD records (checked 2026-10-15).

## 1. Sources and raw files (`data/raw/`)

| Dataset | Source | Licence | Raw path | md5 |
| --- | --- | --- | --- | --- |
| MultiMS2 (Rutz, Correia, Zamboni, GigaScience 2026, [10.1093/gigascience/giag069](https://doi.org/10.1093/gigascience/giag069)) | Zenodo [10.5281/zenodo.17417089](https://doi.org/10.5281/zenodo.17417089) (MGF + code, v0.0.1) | data CC0, code MIT | `multims2/MultiMS2-0.0.1.zip`, extracted `multims2/MultiMS2/data/multims2_spectra.mgf` | zip `fd68056c580e672438e5cb83d6dc3bed`, MGF `22b9ab58b552c2bc2396350ab15ce636` |
| MS-DIAL 5 standards (Takeda et al., Nat. Commun. 2024, [10.1038/s41467-024-54137-w](https://doi.org/10.1038/s41467-024-54137-w)) | RIKEN DropMet DM0054, `https://prime.psc.riken.jp/archives/data/DropMet/077/` | not stated | `msdial5_ead/mspformat_716authenticstandards.zip` (+ `mspformat_65lipidstandards.zip`, `metadata.xlsx`) | `e8e2476a21aabbd6298e498c820a834f` |
| CIeaD (Singh et al., JASMS 2026, 37(1):8-12, [10.1021/jasms.5c00329](https://doi.org/10.1021/jasms.5c00329)) | `http://www.moleculardetective.org/CIeaD_library.zip`; MSP copy at Zenodo [10.5281/zenodo.17852375](https://doi.org/10.5281/zenodo.17852375) | Zenodo copy CC-BY-4.0 | `cieAD/` | full zip `ab09c3e7f4b0fe6f3bbedfa5b7524166` |

Not downloaded: the MultiMS2 raw mzML (Zenodo 17250693, ~7.5 GB) and the MS-DIAL 5 figure raw data
(~14 GB). RIKEN refuses plain `curl`; `--http1.1 --tlsv1.2` works. GNPS2 and MassIVE FTP were
unreachable from the workstation.

## 2. What each library contains

- **MultiMS2:** 43,728 single-scan direct-infusion ESI spectra, 2,926 SMILES. `FRAGMENTATION_METHOD`
  CID at 20/40/60 (35,673) or EAD at electron kinetic energy 12/16/24 eV (8,055); 75% positive mode.
  Structures, adducts and InChIKeys are present. 1,789 scans are annotated as two or three
  *different* compounds (co-infused isobars), 3,651 records in all. 3,001 records carry
  `QUALITY_CHIMERIC=MISSING_MAIN_SIGNAL`.
- **MS-DIAL 5:** 716 compounds (IROA MSMLS + an in-house natural-product set), LC-MS/MS, positive
  mode only, nine conditions each: CID 10/20/40 V, the same with a 15 V CE spread (CES), and EAD
  10/15/20 eV with CID 10 V. `subtracted/` holds the background-subtracted spectra (6,444; median
  749 peaks); `peak_region/` and `noise_region/` are its inputs. The 64 lipid standards (640
  spectra, EAD 8-20 eV, ~10k peaks each) are near-raw and not used.
- **CIeaD:** 13,895 unique spectra (paper: 13,887) for ~2,300 plant metabolites. The four
  "Pos/Neg CID/EAD" exports (MSP, SDF, txt, `.lbp`) are overlapping views of one pool: each record
  lists every library its compound belongs to, and every file mixes polarities. **No export
  contains a structure** (the SDFs have no mol block, the `.lbp` no SMILES/InChI/molfile), and no
  record says CID or EAD. CE ±12/±10 V with spread 0 is probably EAD and CE ±30/±40 V with spread
  20 probably CID; that is inferred, and the electron energy is not recorded. Adducts are not
  given; precursor minus monoisotopic mass is ±1.007 for >99% of records.

## 3. Pipeline decisions (user, 2026-10-15)

- **New `frag_mode` `EAD`.** `parse_inst_info` keeps an explicit `EAD`; before, any value other
  than HCD/CID became `NA`.
- **For EAD rows `ace` is the electron kinetic energy**; `frag_mode` says which scale `ace` is on,
  as ORB vs ORB_F do for NCE. The CID voltage that accompanies EAD (10 V in MS-DIAL 5, unrecorded
  in MultiMS2) is dropped. No new `spec_df` column.
- **`group_id` key is now (`mol_id`, `prec_type`, `inst_type`, `frag_mode`)** in step 2 and
  `merge_datasets.py`, so spectrum merging never sums CID and EAD. Checked on every
  `data/proc/*/spec_df.pkl` (NIST20, NIST23, NIST23 EI, MSGYM, Spectraverse, PCDL, Enveda-180):
  no existing group has two frag modes, so a rebuild gives identical groups and ids.
  `preproc_scripts/inference/02_inf_prepare_spec_df.py` still groups on the old trio; it writes
  only `CID`, so the result is the same.
- **Standalone datasets**, one proc directory each; **CIeaD deferred** until structures are
  obtained from the authors (Sixue Chen lab).
- Instrument label is `QTOF` (`inst` = "SCIEX ZenoTOF 7600"). Models read `frag_mode` through
  `spec_params.frag_mode: true` and `frag_modes`; a config must list `"EAD"` there, or EAD falls
  into the unknown index.

## 4. Reader behaviour and processed outputs

| | MultiMS2 (`--input_format multims2`) | MS-DIAL 5 (`--input_format msdial5_ead`) |
| --- | --- | --- |
| Reader drops | 3,651 records of multi-compound scans | 2,148 CE-spread spectra (actual energies not recorded); zero-intensity peaks |
| Step-1 rows (`data/df/`) | 40,077 | 4,296 |
| Step-2 drops | none | 228 (38 compounds x 6): file formula differs from SMILES formula (salts, charged species) |
| `data/proc/<name>/spec_df.pkl` | 40,077 spectra, 2,748 molecules, 6,621 groups; md5 `355d3da9baedd6f2489d1f575a0aae06` | 4,068 spectra, 657 molecules, 1,250 groups; md5 `296952868a7bdb70dabe229d25b30802` |
| CID | P 20/40/60: 5,943 / 8,655 / 9,030; N: 2,875 / 3,313 / 3,016 | P 10/20/40: 678 each |
| EAD (`ace` = electron KE) | P 12/16/24: 2,013 / 2,305 / 2,067; N: 231 / 170 / 459 | P 10/15/20: 678 each |
| Adducts | `[M+H]+` 22,336, `[M-H]-` 8,269, plus in-source/oxidised forms (`[M+O+H]+`, `[M-H2O+H]+`, ...) | `[M+H]+` 3,828, `[M+Na]+` 240 |
| Peaks per spectrum (median) | 17 | 779 (needs noise filtering) |

MultiMS2 also flags 3,456 spectra in `diff_inchikey_df.pkl` (InChIKey changes under
standardization; kept, as for every dataset). Both are built with default step-2 options
(standardization on). Tests: `tests/test_01_prepare_df.py` (both readers, CE parsing),
`tests/test_02_prepare_proc.py::test_group_id_separates_frag_modes`.

## 5. Open

- Split (step 4, `--dag_filtering False` for MASSter), a training config with `frag_modes`
  including `"EAD"`, and a peak-cleaning choice for the MS-DIAL 5 spectra.
- CIeaD structures from the authors; then its CID/EAD labels must be confirmed, not inferred.
