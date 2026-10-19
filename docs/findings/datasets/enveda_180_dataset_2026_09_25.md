# Enveda-180 dataset: contents and preprocessing

> **Status:** FINDING · **Updated:** 2026-10-06
> Steps 1-2 (df, proc) ran locally and the 8:1:1 InChIKey split exists (section 7); no DAGs. Dimer and misnamed formate/acetate spectra are outside the split; `[M+Br]-` is now supported by the adduct tables (section 8). Isotopes (section 9): 13C satellites are 7.7% of positive-mode target intensity; the release's MS1 isotope fields are unreliable. Baseline config section added at the end.

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

## 7. Split `data/split/enveda_180_inchikey_811` (2026-09-25)

The release ships no split. The first one is a random split by InChIKey first block, 8:1:1 by
molecule, with seed 42. The log is `output/enveda_180_preproc_2026_09_25/logs/04_prepare_split.log`.

```bash
python preproc_scripts/04_prepare_split.py --split_type random --split_key inchikey_s \
    --test_frac 0.1 --val_frac 0.1 --meta_rseed 42 --primary_dsets enveda_180 \
    --proc_dp data/proc/enveda_180 --dag_filtering False \
    --prec_types "[M+H]+" "[M-H]-" "[M+Na]+" "[M+NH4]+" "[M+Cl]-" "[M+K]+" \
    --ce_types stepped single --split_dp data/split/enveda_180_inchikey_811
```

| Part | Molecules | Spectra | `[M+H]+` | `[M-H]-` | Other adducts |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 143,568 | 640,475 | 472,208 | 147,912 | 20,355 |
| val | 17,946 | 79,881 | 59,111 | 18,209 | 2,561 |
| test | 17,946 | 80,152 | 59,266 | 18,379 | 2,507 |

- The split holds 800,508 of the 1,158,293 spectra (69.1%). Every other default filter passes
  everything, so the whole reduction is from adducts and the two-energy spectra.
- Across the three parts, no InChIKey first block and no group is shared. 52.8% of test scaffolds
  also occur in train.
- **Adducts are limited to the six the models' mass table knows** (`PREC_TYPE_TO_MASS_DIFF` in
  `formula_utils.py`). The dataset raises `KeyError` on any adduct missing from a config's
  `prec_types`, and no config enables `filter_prec_types`. So a config that trains on fewer than
  six adducts must set `split_filters.<split>.filter_prec_types: true`.
- **Left out:**
  - The 331,923 dimer spectra: the models have no `[2M+...]` entries.
  - 25,059 formate and acetate spectra, because of a naming mismatch. `parse_prec_type_str`
    stores `[M+FA-H]-` as `[M+CH2O2-H]-`, and `[M+Hac-H]-` stays as it is, but the mass table
    spells them `[M+HCOOH-H]-` and `[M+CH3COOH-H]-`. Adding them needs that mapping fixed, a
    step 2 rerun and a new split.
  - The 25 `[M+Br]-` spectra (13 molecules). The split predates `[M+Br]-` support (§8), and
    adding them needs a new split. Because `04_prepare_split.py` draws the random assignment
    from the key set, a new split reassigns every molecule, not only these 13.
- **Open:**
  - Whether Enveda trains alongside NIST/MSGYM. The molecule overlap in §1 is small enough
    for either.
  - FraGNNet DAGs (step 3) are not generated; MASSter needs no DAGs.

## 8. `[M+Br]-` support (2026-09-25, uncommitted)

`[M+Br]-` is now in the four adduct tables in `formula_utils.py`, modelled on `[M+Cl]-`:

| Table | Entry |
| --- | --- |
| `PREC_TYPE_TO_MASS_DIFF` | `BR_MASS + ELECTRON_MASS`, the 79Br mass (78.91834 Da) |
| `PREC_TYPE_TO_FORMULA_DIFF` | `Br1` (charge retention keeps the bromide) |
| `PREC_TYPE_TO_CMF_MASS_DIFF`, `PREC_TYPE_TO_CMF_FORMULA_DIFF` | `[F-H]-`, `H-1` (charge migration: HBr leaves) |

- Everything else reads these tables, so nothing else needed code: the dataset, `MassterModel`,
  CMF enumeration, subformula cleaning and isotope detection all do. Isotope detection already
  counts 81Br in its M+2 table.
- Tests:
  - `test_formula_utils.py::TestPrecTypeDiffs::test_bromide_adduct_values` checks the values.
  - `[M+Br]-` joins the adduct parametrizations in `test_cmf_logic.py` and
    `test_precursor_formula_validity.py`.
  - 300 pass across those files, `test_clean_spectra_utils.py` and
    `test_isotope_detection.py`. The full suite passes: 4,448, with 1 skipped and 2 xfailed.
- Other datasets: NIST23 holds 224 `[M+Br]-` spectra, and 9 `[M+Br+2i]-`, which stay
  unsupported like every `+Ni` label. MSGYM 1.5 and Spectraverse hold none. No existing split
  includes `[M+Br]-`, so no split or checkpoint changes.

Measured on the 25 Enveda `[M+Br]-` spectra:

- **Precursor m/z** matches M + 79Br + e within 3.2 ppm on every spectrum.
- **Fragment attribution** by brute-force subformula matching (±2 H, 10 ppm):
  - 56% of fragment intensity fits only Br-retaining `[F+Br]-`: 51% as 79Br and 5% as 81Br.
  - 4% fits only `[F-H]-`, 27% fits either, and 13% neither.
  - So charge retention dominates, and CMF is a minor channel, as the tables assume.
- **`filter_subformula_peaks`** at 10 ppm keeps 68.8% of `[M+Br]-` intensity, against 93.5% for
  200 sampled `[M+Cl]-` spectra. Of the dropped intensity:
  - 2.4% is 81Br isotopologues of kept fragments, from co-isolating the 81Br precursor, which no
    channel models.
  - The bare Br- ion (m/z 78.9174, 12 ppm from theory) is 18.6% of one spectrum. It is not a
    fragment of M, so neither model can emit it.
  - The rest is a few large peaks that need more than a ±2 H shift or are unassignable.

## 9. Isotopes (2026-09-25)

The probe is [debug_enveda_isotope_probe.py](../../scripts/debug/isotopes/debug_enveda_isotope_probe.py). It
reads the first 60,000 records of each export and runs in 24 s. The log is
`output/enveda_180_isotope_probe_2026_09_25/probe.log`. Tests 1-3 read the raw export, because the
filtered file's 1% floor removes most isotope peaks. Test 4 reads the filtered export, which is
the training target.

In MS2, every fragment isotope peak comes from co-isolated precursor isotopologues, because the
monoisotopic precursor holds no heavy isotope. So the co-isolation fractions in §9.3 set how
large those peaks are.

### 9.1 This timsTOF separates 13C from a hydrogen shift below m/z 300

Peak pairs spaced 1.000-1.015 Da apart, both peaks at or above 0.5% of the base peak, in
0.25 mDa bins:

| Lower peak m/z | Pairs | 13C mode (theory 1003.355 mDa) | H-shift mode (theory 1007.825 mDa) | Trough / 13C peak |
| --- | ---: | ---: | ---: | ---: |
| < 300 | 523,575 | 1003.38 | 1007.88 | 0.21 |
| 300-600 | 82,568 | 1002.88 | no second mode | 0.18 |
| ≥ 600 | 1,117 | too few pairs | too few pairs | — |

- Below m/z 300 the two spacings form separate modes 4.5 mDa apart, each about 1.3 mDa wide at
  half height.
- Between 300 and 600 the 13C mode is still sharp but sits 0.5 mDa low, a calibration offset. Pairs
  spaced by one hydrogen are too rare there to form a mode, so separation is untested above 300.
- The detector gates QTOF out (`_INST_MZ_TOL` has only FT) on the premise that QTOF cannot
  separate the two. That premise does not hold for this instrument below m/z 300. This measures
  resolution only; it does not change the position that hard isotope removal is unsafe.

### 9.2 The release's MS1 isotope fields are unreliable

- `isotope_mzs` are extracted at 13C spacing. The listed M+2 sits at +2.0065 Da even for Cl and Br
  molecules, whose real M+2 is at +1.9971 or +1.9980.
- So the halogen M+2 is missed: the listed Br M+2 is a median 0.011 of theory, and the Cl M+2 is 0.
- Even without halogens, M+1 is 0.72-0.77 of theory. The intensities are also rounded to two
  decimals.
- Do not use these fields as MS1 isotope ground truth or as a co-isolation reference.

### 9.3 Co-isolation: M+1 mostly passes, M+2 is a per-molecule switch

`f_k` is (MS2 M+k / M+0 of the precursor) divided by the theoretical ratio of the ion formula.
It is measured on 15,378 `[M+H]+`/`[M-H]-` spectra whose precursor keeps at least 5% of the base
peak.

- **M+1:** median f1 is 0.934 in positive mode and 0.480 in negative mode. So in positive mode
  fragment 13C satellites are close to the natural envelope, and in negative mode they are about
  half of it.
- **M+2, Cl/Br molecules:**
  - Negative mode almost never transmits it (0.9% of spectra have f2 > 0.5).
  - In positive mode, 38% of spectra transmit it, at about 1.07× natural; the rest sit at about
    0.03×.
  - Transmission is fixed per molecule. Of 613 positive-mode molecules with two or more such
    spectra, 245 transmit M+2 in all of them, 365 in none and 3 in some. A 200,000-record pass
    gives 730 / 1,334 / 7 of 2,071.
  - It does not follow CE, and there is no trend across the compound-id deciles. So it reflects
    how the compound was acquired, and the structure cannot predict it.

### 9.4 What a monoisotopic prediction cannot reach

Share of training-target intensity (filtered export) on a peak whose partner is less massive,
more intense, and at the given spacing:

| Spectra | 13C satellites (1.0020-1.0053 Da) | H-spaced (1.0060-1.0095 Da) | Halogen M+2 partner |
| --- | ---: | ---: | ---: |
| positive (46,197) | 7.7% (median 6.9%, p90 15.2%) | 1.4% | — |
| negative (13,803) | 3.9% (median 2.8%, p90 9.5%) | 1.1% | — |
| Cl/Br molecules (8,163) | — | — | 6.2% (p90 22.0%) |

- 59.2% of spectra have at least 5% of their intensity on 13C satellites.
- Every current config predicts the monoisotopic peak only (`num_isotopes: 1`). That intensity is
  therefore unreachable, so it caps cosine on Enveda before any fitting.
- Predicting the envelope by splitting a formula's probability across its isotopologues with the
  natural-abundance prior (`num_isotopes > 1`, no new parameters) matches positive mode, where
  f1 ≈ 0.93. It would overstate negative mode, where f1 ≈ 0.48. It cannot know whether a
  molecule's halogen M+2 was transmitted. Nothing here has been trained.
- **Measured 2026-09-27: not worth modelling for cosine.**
  - A perfect monoisotopic model plus the best scalar envelope recovers 0.0047 of val cosine in
    positive mode and 0.0010 in negative.
  - The natural prior recovers nothing in positive mode and costs 0.013 in negative, from halogen
    M+2 peaks the spectra do not have.
  - The isotope envelope plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_isotope_envelope_plan.md`)
    closed null on it.

## 10. Baseline config and run state (2026-09-27/28, from project-state handoff)

- **Config:** `config/masster/enveda/enveda_180_inchikey_811_unmerged_mh_mmh_ace_fourier.yml`. Enveda-180 trains standalone, not with NIST/MSGYM (user decision).
- **CE:** `ce_insert_type: embed` with `int_embedder: fourier`, as in the Spectraverse folds, replacing `qet` after `n1`-`n3` trailed `c0` on NIST20.
- **ACE only, no EMA (user decision, 2026-09-28).** `spec_params.nce: false` and no `ema_decay`, because Enveda records only ACE and the dataset's NCE is derived from it. Renamed from `...dual_ce_fourier_ema999` (never submitted); the fork origin is the MSGYM non-EMA arm. A one-epoch smoke exits 0, and its checkpoint holds only an ACE table and no EMA state.
- **Other deltas from the MSGYM arm:** `[M+H]+`/`[M-H]-` through `filter_prec_types`, a 10 ppm QTOF loss tolerance, 20 epochs (24,240 steps).
- **Status:** not submitted; it needs a second seed before any arm is read against it. Local runs need a short `TMPDIR` on the root disk (`/tmp` is 3.7 GB, and a long `TMPDIR` path breaks the dataloader sockets).
- The original role of this file was control `i0` of the closed-null isotope envelope plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_isotope_envelope_plan.md`): on the 77,320 `[M+H]+`/`[M-H]-` val spectra, a perfect monoisotopic model plus the best scalar envelope recovers only 0.0047 cosine (positive) and 0.0010 (negative) against a 0.005 gate, so FraGNNet on Enveda keeps `num_isotopes: 1`.
