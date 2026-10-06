# Data Preprocessing

## Step 0: Exporting the NIST 2020 MS/MS library

All of the data used in the paper is based on the NIST 2020 MS/MS library. Since this library is not publicly available, we are unfortunately unable to provide access to it.

The raw NIST data can be exported using the instructions [here](https://github.com/Roestlab/massformer).

## Step 1: Processing raw data into CSV

This step Writes `<output_name>_df.csv` (or `.json`) in `data/df/` by default. Each row is one spectrum. Columns are defined here:
- `spec_id`: integer id within this exported file.
- `dset`: dataset/source name.
- `dset_spec_id`: source-specific spectrum identifier.
- `peaks`: multiline string, one line per peak as `mz intensity`.
- `prec_mz`: precursor m/z (float).
- `prec_type`: precursor/adduct string (e.g., `[M+H]+`).
- `spec_type`: spectrum type/level (e.g., `MS2`).
- `ion_mode`: `P`/`N` when known.
- `ion_type`: ionization type (e.g., `ESI`, `EI`) when known.
- `inst_type` / `inst`: instrument metadata when present.
- `col_energy`, `col_energy_extra_1`, `col_energy_extra_2`: collision energies (strings with units).
- `frag_mode`: fragmentation mode (e.g., `CID`, `HCD`, `EI`) when present.
- `num_peaks`: peak count if provided by source.
- `exact_mass` / `mw`: exact or molecular weight when available.
- `formula` / `smiles` / `inchi` / `inchikey`: structure metadata if present.
- `name` / `title`: compound name or title when present.
- `ramped` / `stepped` / `normalized`: flags parsed from collision energy fields when applicable.
- Misc placeholders: `notes`, `rating`, `cas_num`, `pressure`, `ri`, `ionization` extras, etc., kept for upstream completeness.

Some input formats also emit helper fold files next to the df. MassSpecGym and Spectraverse v1.0.2 write `<output_name>_fold.csv` (one file for the whole set). For Spectraverse its columns are `spec_id,dset_spec_id,fold`, and `fold` is the MGF's own `FOLD=` value (1..10). Older Spectraverse MGFs with `FOLD_INCHI`/`FOLD_FORM`/`FOLD_MCES`/`FOLD_MCESFORM` keys write one `<output_name>_<fold_col>_<k>.csv` per fold column and value instead. A helper holds only ids and fold labels, for downstream splitting.

### NIST 20
This assumes that the NIST20 data is in a folder `data/raw/nist_20`, containing two files: `hr_nist_msms.MSP` (the MSP file with the spectrum information) and `hr_nist_msms.MOL` (the MOL file with the molecule structures).

This will create a new directory `data/df`

```bash
python preproc_scripts/01_prepare_df.py --msp_file nist_20/hr_nist_msms.MSP --mol_dir nist_20/hr_nist_msms.MOL --input_format msp+mol --output_format csv --output_dp data/df --output_name nist20_hr
```

`--msp_file` may also be gzipped (`.msp.gz`) or a `.zip` holding one `.msp`; the file is streamed,
not unpacked. `data/raw/nist_20/nist_20_hr_msms.zip` (re-exported 2026-09-28) is such an MSP. It
names each spectrum's instrument model (`Instrument:`, e.g. `Orbitrap Fusion Lumos`), which
`02_prepare_proc.py` turns into `ORB` / `ORB_F`, but it carries no structures, so a full build still
needs the MOL files.

### MassSpecGym
This assumes that the MassSpecGym data is in `data/raw/msgym/MassSpecGym_with_test.tsv`, this will use `ms_gym`,`ms_gym_extra` as dataset name for each part of MassSpecGym

```bash
# for sim challenege
python preproc_scripts/01_prepare_df.py --input_format=ms_gym --output_format=csv --output_name=ms_gym  --ms_gym_tsv=msgym/MassSpecGym_with_test.tsv

# for extra data in msgym
python preproc_scripts/01_prepare_df.py --input_format=ms_gym_extra --output_format=csv --output_name=ms_gym_extra --ms_gym_tsv=msgym/MassSpecGym_with_test.tsv
```

### Spectraverse
The current data is Spectraverse v1.0.2, `data/raw/spectraverse/spectraverse-1.0.2/spectraverse-1.0.2.mgf`
(md5 `50803b370a118299bfef3e9485ef6895`, 488,797 records). Each record has a `SOURCE=` field and one
`FOLD=` field (1..10, the benchmark's MCES fold, disjoint by 2D InChIKey); `TITLE=SPECTRAVERSE%010d`.
`--mgf_file` is joined to `--raw_data_dp` (default `data/raw`).

```bash
python preproc_scripts/01_prepare_df.py \
    --input_format=spectraverse \
    --output_format=csv \
    --raw_data_dp=data/raw \
    --output_dp=data/df_spectraverse_1_0_2 \
    --output_name=spectraverse \
    --mgf_file=spectraverse/spectraverse-1.0.2/spectraverse-1.0.2.mgf
```

This writes `spectraverse_df.csv` and `spectraverse_fold.csv` (`spec_id,dset_spec_id,fold`) to
`data/df_spectraverse_1_0_2/`. `spec_id` is the 0-based MGF record order and `dset_spec_id` is the
integer in `TITLE` (the 1-based record index, `SPECTRAVERSE0000000001` gives 1). The reader raises on
a truncated record, a peak count that differs from `NUM_PEAKS`, a bad `TITLE`, or a `FOLD` outside 1..10,
and only warns when an MGF has no fold field at all (v1.0.0/v1.0.1).

`--output_dp` is a new directory because `data/df/spectraverse_df.csv` already exists and must not be
overwritten. A new `--output_name` would not do instead: step 2 labels the `dset` column with the
`--dsets` name, and the Spectraverse configs and splits select the dataset `spectraverse`.

**Never join `data/df/spectraverse_df.csv` or `data/df/spectraverse_fold_*.csv` to the current
`data/proc/spectraverse`.** They are an older build (v1.0.0 era; `INDEX` and `FOLD_*` keys, 8-digit
titles, 478,530 rows) whose `spec_id`/`dset_spec_id` do not match the current ones. See
[the source and replicate audit](../docs/findings/masster/spectraverse_source_replicate_audit_2026_10_03.md)
and [the curation note](../docs/notes/spectraverse_curation_paper.md).

### Enveda-180
This assumes the Enveda-180 release is in `data/raw/enveda-180/`. Use the `-filtered` export:
it holds the same spectra as `enveda-180.jsonl.gz` with peaks below 1% of the base peak removed.
Stepped-CE entries (`20,40,60`, `-60,-40,-20`) become ascending 20/40/60 eV in
`col_energy`/`col_energy_extra_1`/`col_energy_extra_2`; the negative sign is polarity only.
See [the dataset finding](../docs/findings/enveda_180_dataset_2026_09_25.md).

```bash
python preproc_scripts/01_prepare_df.py \
    --input_format=enveda \
    --output_format=csv \
    --output_name=enveda_180 \
    --jsonl_file=enveda-180/enveda-180-filtered.jsonl.gz
```

## Step 2: Processing CSV into Pickle Files

This step  reads the stage-1 CSVs, converts the data into binary format (pickle), writes pickles to `proc_dp` for training and evaluation.

- `spec_df.pkl` — contains spectrum information, one row per spectrum:
	- `spec_id`: integer id unique within the processed set.
	- `mol_id`: integer id linking to `mol_df`.
	- `prec_type`: standardized precursor/adduct string.
	- `inst_type`: standardized instrument type: `ORB_F` (Orbitrap Fusion family, e.g. NIST's Lumos), `ORB` (every other Orbitrap: Elite, Velos, unnamed), `QTOF`, `IT`, `QQQ`, `EI`, `Other`. `FT` is retired and refused on load.
	- `frag_mode`: standardized fragmentation mode.
	- `spec_type`: spectrum type (e.g., `MS2`).
	- `ion_mode`: `P`/`N`.
	- `dset`: dataset name.
	- `dset_spec_id`: source-specific spectrum id.
	- `col_gas`: collision gas (may be NaN).
	- `res`: inferred m/z resolution (currently informational).
	- `ace`, `ace_extra_1`, `ace_extra_2`: parsed absolute collision energies (first/second/third for ramped or stepped).
	- `nce`, `nce_extra_1`, `nce_extra_2`: parsed normalized collision energies (first/second/third for ramped or stepped).
	- `has_ace`, `has_nce`: booleans for CE presence.
	- `prec_mz`: precursor m/z (float; inferred when missing using molecular mass).
	- `peaks`: list of `(mz, intensity)` float pairs.
	- `ri`: retention index if present.
	- `formula`: molecular formula from source (dropped before saving when inconsistent with molecule formula).
	- `inchikey`: full InChIKey from source (dropped before saving when inconsistent).
	- `exact_mass`: exact mass from source (dropped before saving when inconsistent).
	- `group_id`: grouping id for spectra sharing compound/precursor context.

- `mol_df.pkl` — contains molecule information, one row per unique canonical SMILES:
	- `smiles`, `mol_id`, `mol` (RDKit Mol object).
	- `inchikey_s`: first 14 chars of InChIKey.
	- `scaffold`: Murcko scaffold SMILES.
	- `formula`, `inchi`.
	- `mw`, `exact_mw`: average and exact molecular weights.
	- `num_atoms`, `num_bonds`, `num_radicals`.
	- `charge`, `single_mol` (connected component flag).

- `ann_df.pkl` — contains annotation information, one row per annotated spectrum (only when annotations parsed):
	- `dset_spec_id`, `spec_id`, `mol_id`, `prec_type`, `formula`.
	- `ann_peak_mzs`: list of annotated peak m/z values.
	- `ann_products`: product formula strings.
	- `ann_losses`: neutral loss formula strings.
	- `ann_isotopes`: isotope annotations.
	- `ann_exact_mzs`: exact m/z values for annotations.

- Debug pickles: `no_mol_df.pkl`, `diff_formula_df.pkl`, `diff_inchikey_df.pkl`, `diff_mass_df.pkl` record rows dropped or with inconsistencies.


### NIST 20
```bash
python preproc_scripts/02_prepare_proc.py \
    --df_dp data/df \
    --dsets nist20_hr \
    --proc_dp data/proc/nist20
```

### MSGYM
```bash
# for all data in msgym
python preproc_scripts/02_prepare_proc.py \
    --proc_dp=data/proc/ms_gym_all \
    --dsets ms_gym ms_gym_extra
```
### Spectraverse
```bash
python preproc_scripts/02_prepare_proc.py \
    --df_dp=data/df_spectraverse_1_0_2 \
    --proc_dp=data/proc/spectraverse_rebuild \
    --dsets spectraverse \
    --skip_standardize
```
`--skip_standardize` is required to reproduce the stored pickles: without it the standardizer rewrites
sulfoxide SMILES to `[S+][O-]`, and the stored `mol_df.pkl` has none. `--df_dp` must point at the
step 1 directory above (its default `data/df` holds the old CSV). `--proc_dp` is a new directory
because `--ow_spec_mol` defaults to true and would overwrite `spec_df.pkl`, `mol_df.pkl` and `ann_df.pkl`;
the live `data/proc/spectraverse` also holds files the Step 4 variants read (`near_dup_*.csv`,
`spec_source.csv`, `subformula_ceiling.csv`). No spectrum is dropped: `spec_df.pkl` has one row per MGF
record (488,797). `inst_type` comes from the MGF `INSTRUMENT_TYPE` (`orbitrap` gives `ORB` (290,183), `qtof` gives
`QTOF` (169,380), `iontrap` gives `Other` (16,181) and `nan` gives `NA` (13,053); there is no
`ORB_F`). The `NORMALIZED_COLLISION_ENERGY_*` keys are not read, so all `nce*` columns are NaN.

The stored `spec_df.pkl` was built on 2026-05-04 with the retired `FT` label; it was relabelled in place
to `OB`/`ORB` on 2026-09-28 and renamed to `ORB` on 2026-10-01 (the two sections below). A fresh run
emits `ORB` directly, so no migration is needed on a rebuild.
### Enveda-180
```bash
python preproc_scripts/02_prepare_proc.py \
    --proc_dp=data/proc/enveda_180 \
    --dsets enveda_180
```

### Relabelling existing Orbitrap spectra (FT → ORB / ORB_F, 2026-09-28)
`spec_df.pkl` files built before 2026-09-28 label every Orbitrap `FT`, which training now
refuses. Relabel them in place; each run copies the original to `spec_df_orig_<timestamp>.pkl`.
The label comes from the raw instrument name, joined on `dset_spec_id` (the NIST#): "Fusion"
means `ORB_F`, any other name `ORB`, and FT spectra without a name (contributed libraries) are
unnamed Orbitraps, `ORB`. `--inst_fp` takes the name sources in priority order, as `data/df` tables
(`dset_spec_id`, `inst`) or raw MSPs (`.msp`, `.msp.gz`, `.zip`).
```bash
# NIST23
python preproc_scripts/dataset_migration/relabel_orbitrap_inst_type.py \
    --spec_df_fp data/proc/nist23/spec_df.pkl --inst_fp data/df/nist23_hr_df.csv
# NIST20: its raw export, then NIST23 for the 5,537 NIST# the export lacks (NIST# is stable
# across releases; the two never disagree on an instrument)
python preproc_scripts/dataset_migration/relabel_orbitrap_inst_type.py \
    --spec_df_fp data/proc/nist20/spec_df.pkl \
    --inst_fp data/raw/nist_20/nist_20_hr_msms.zip data/df/nist23_hr_df.csv
# Any other dataset: its data/df table if it names instruments, else every FT becomes ORB
python preproc_scripts/dataset_migration/relabel_orbitrap_inst_type.py \
    --spec_df_fp <proc_dp>/spec_df.pkl [--inst_fp data/df/<name>_df.csv]
```
Group ids are split by instrument, never regrouped. Details and the NIST counts are in
[the Orbitrap instrument finding](../docs/findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md).

### Renaming Orbitrap labels (OB / OB_F → ORB / ORB_F, 2026-10-01)
The Orbitrap labels were `OB` / `OB_F` from 2026-09-28 to 2026-10-01. Configs, `spec_df.pkl`
files and checkpoint hparams that still name them are refused. The rename is one-to-one, so a
`spec_df.pkl` changes only its `inst_type` values, and a checkpoint changes only its
`hyper_parameters`: instrument-embedding rows follow the sorted label order, which the rename
keeps. Each file is replaced in place with no backup; rerunning on a migrated file is a no-op.
```bash
python preproc_scripts/dataset_migration/rename_orbitrap_inst_type.py \
    --spec_df_fp data/proc/nist20/spec_df.pkl data/proc/nist23/spec_df.pkl
python preproc_scripts/dataset_migration/rename_orbitrap_inst_type.py \
    --ckpt_fp saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt
```
A `spec_df.pkl` that still has `FT` rows needs the relabel above first; it now writes
`ORB` / `ORB_F` directly.

## Step 3: Generating Fragmentation DAGs

Multi-cut BFS now defaults to `--restrict_ring_open False`, allowing isolated rings
to open even when the molecule also contains fused rings. This can change generated
DAGs and increase generation time. To reproduce runs that used the previous default,
pass `--restrict_ring_open True` explicitly. Existing saved DAGs are unaffected.

With `--smarts_prepass True`, both BFS modes now include neutral-loss rearrangements
such as SO₂ extrusion. Previously these products were injected only in standard
BFS. Regenerate multi-cut DAGs to obtain the added candidates; keep existing DAGs
when reproducing experiments run before this correction.

On 2026-09-27 two prepass rules were corrected, which changes `--smarts_prepass True`
DAGs built afterwards. `sulfoxide_so_loss` now matches the `[S+][O-]` form the
standardizer writes, and it matches all 120 NIST20 sulfoxides, where it matched none
before. The two anthraquinone rules now keep every mirror embedding: 2 and 4 partitions
per anthraquinone instead of 1. Every other rule's output is unchanged and in the same
order. Existing DAGs are unaffected; keep them when reproducing earlier `smpTrue` runs.

### NIST 20
This step generates fragmentation DAGs that are used for the FraGNNet models. Since fragmentation can be slow, we recommend using a compute with many cores (by default, the script will use all available cores). 

Frag configuration: depth 3 (used for FraGNNet-D3), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 3 --frag_dp data/frag/nist20_d3 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types ORB ORB_F --frag_modes HCD --ion_modes P --wandb_mode disabled
```

Frag configuration: depth 4 (used for FraGNNet-D4), isomorphism check with no bond information (nb, 3 WL iterations)

```bash
python preproc_scripts/03_prepare_dag_feats.py --max_depth 4 --frag_dp data/frag/nist20_d4 --proc_dp data/proc/nist20 --prec_types "[M+H]+" --inst_types ORB ORB_F --frag_modes HCD --ion_modes P --dsets nist20_hr--wandb_mode disabled
```

### MassSpecGym

```bash
# Benchmark split (sim chanllenge)
python preproc_scripts/04_prepare_split.py \
  --proc_dp=data/proc/ms_gym_all \
  --ces ace \
  --split_dp=./data/split/ms_gym_d3_h4_benchmarkv2 \
  --frag_dp=./data/frag/ms_gym_d3_h4v2 \
  --primary_dsets ms_gym \
  --max_prec_mz=1000.0 \
  --inst_types ORB ORB_F QTOF \
  --split_type=predefined_dsetid_csv \
  --predefined_dsetid_csv=data/df/ms_gym_fold.csv
```

## Step 4: Preparing Splits

This step splits the data into training, validation, and test sets. To exactly reproduce the splits used in the paper, download the split id files ([inchikey](), [scaffold]()) and place them in `data/split/nist20_inchikey/` and `data/split/nist20_scaffold/` respectively. Then, run the following scripts to remap the ids to work with the processed dataset from the previous step:

Inchikey split:

```bash
python preproc_scripts/04_prepare_split.py --split_type predefined --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types ORB ORB_F --frag_modes HCD --ion_modes P --dag_filtering False --proc_dp data/proc/nist20 --predefined_id_type dset_spec_id --predefined_train_id_fp data/split/nist20_inchikey/train_dset_ids.csv --predefined_val_id_fp data/split/nist20_inchikey/val_dset_ids.csv --predefined_test_id_fp data/split/nist20_inchikey/test_dset_ids.csv --split_dp data/split/nist20_inchikey
```

Scaffold split:

```bash
python preproc_scripts/04_prepare_split.py --split_type predefined --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types ORB ORB_F --frag_modes HCD --ion_modes P --dag_filtering False --proc_dp data/proc/nist20 --predefined_id_type dset_spec_id --predefined_train_id_fp data/split/nist20_scaffold/train_dset_ids.csv --predefined_val_id_fp data/split/nist20_scaffold/val_dset_ids.csv --predefined_test_id_fp data/split/nist20_scaffold/test_dset_ids.csv --split_dp data/split/nist20_scaffold
```

If you don't want to use the exact splits that we used (just ones with similar statistics), you can use the same script to generate random splits:

Inchikey split:

```bash
python preproc_scripts/04_prepare_split.py --split_type random --split_key inchikey_s --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types ORB ORB_F --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_inchikey
```

Scaffold split:

```bash
python preproc_scripts/04_prepare_split.py --split_type random --split_key scaffold --primary_dsets nist20_hr --prec_types "[M+H]+" --inst_types ORB ORB_F --frag_modes HCD --ion_modes P --proc_dp data/proc/nist20 --frag_dp data/frag/nist20_d4 --split_dp data/split/nist20_scaffold
```

### Spectraverse
The benchmark folds `data/split/spectraverse_benchmark/cv0..cv9` come from the MGF's own `FOLD=` field,
not from `04_prepare_split.py`. For `cv<k>`, test is `FOLD k+1`, val is `FOLD k` (`cv0` uses `FOLD 10`)
and train is the other eight folds. Every spectrum is in exactly one of the three, and folds are disjoint
by 2D InChIKey (not by formula). Each `cv` holds `train_ids.csv`, `val_ids.csv`, `test_ids.csv`
(`mol_id,spec_id,group_id`); the existing directory has no `stats.json`, but the rebuild writes one
per `cv<k>`. See
[the source and replicate audit](../docs/findings/masster/spectraverse_source_replicate_audit_2026_10_03.md).

`build_benchmark_split.py` rebuilds them from the MGF, with `mol_id`/`spec_id`/`group_id` taken from the
current `spec_df.pkl`. `--dst-split-dp` is required, has no default, and must not be the existing
benchmark directory. A bare `--mgf-fp` selects the v1.0.2 MGF path above.

```bash
python preproc_scripts/spectraverse/build_benchmark_split.py \
    --spec-fp data/proc/spectraverse/spec_df.pkl \
    --mgf-fp \
    --expect-md5 50803b370a118299bfef3e9485ef6895 \
    --dst-split-dp data/split/spectraverse_benchmark_rebuild \
    --check-against data/split/spectraverse_benchmark
```

Instead of `--mgf-fp`, `--fold-fp data/df_spectraverse_1_0_2/spectraverse_fold.csv` gives byte-identical
id files (joined on `dset_spec_id`; `stats.json` has a null `mgf_md5`). `--check-against` runs after the
files are written and exits 1 only if a `spec_id` set differs; `mol_id`/`group_id` differences are logged.

The existing `data/split/spectraverse_benchmark` has the same `spec_id` sets as this script's output in
all 30 files, but its `mol_id` and `group_id` columns are a stale permutation of the current
`spec_df.pkl`/`mol_df.pkl` numbering (about half the rows disagree). That is harmless, because the
loaders select spectra by `spec_id` only; do not join on those two columns. The splits derived from it
(variants below) inherit the same columns.

#### Spectraverse train-side variants
Each reads `data/split/spectraverse_benchmark` and the current `data/proc/spectraverse/spec_df.pkl` and
writes `cv<k>` directories. The first three also write one `stats.json` and change only the training
spectra: `val_ids.csv` and `test_ids.csv` are copied unchanged, so held-out scores stay comparable with
the benchmark. They require `--folds` and were run for `--folds 0 1` only. `split_by_adduct.py` has no
`--folds` and writes no `stats.json`.

```bash
# Near-duplicate removal -> data/split/spectraverse_benchmark_dedup. One spectrum per cluster of
# same molecule + adduct spectra with cosine > 0.99 (instrument and CE must match; missing values match
# anything). Also writes data/proc/spectraverse/near_dup_{clusters,edges}.csv.
python preproc_scripts/spectraverse/dedup_train_split.py --folds 0 1

# Drop train spectra with subformula ceiling <= 0.5 -> data/split/spectraverse_benchmark_subceil05.
# Reuses data/proc/spectraverse/subformula_ceiling.csv if it exists (delete it after any spec_df
# rebuild), else computes it.
python preproc_scripts/spectraverse/clean_train_split.py --folds 0 1

# Drop train spectra from a raw SOURCE -> data/split/spectraverse_benchmark_nomsdial. Reads SOURCE= from
# the v1.0.2 MGF and writes data/proc/spectraverse/spec_source.csv. The default --dst-split-dp is the
# nomsdial directory whatever --drop-sources is; set --dst-split-dp when dropping another source.
python preproc_scripts/spectraverse/source_filter_split.py --folds 0 1 --drop-sources MS-DIAL

# Per-adduct splits for the NEIMS models -> data/split/spectraverse_benchmark_neims_adduct/<adduct>/cv<k>.
# All ten folds, and val and test are filtered to the adduct as well. The directory tag is the lowercase
# adduct, e.g. m+h+, m-h-, m+.
python preproc_scripts/spectraverse/split_by_adduct.py --adducts "[M+H]+" "[M-H]-" "[M+Na]+"
```

Plans: [dedup](../docs/plans/masster/spectraverse_dedup_plan.md),
[subformula-ceiling arms](../docs/plans/masster/spectraverse_short_clean_arms_plan.md),
[merged train with MS-DIAL dropped](../docs/plans/masster/spectraverse_merged_train_plan.md).
The resulting train sizes are in each variant's `stats.json`.

### Enveda-180
8:1:1 InChIKey split over the six adducts in `PREC_TYPE_TO_MASS_DIFF`. It excludes the two-energy
stepped spectra, which read as `ramped`
([dataset finding §7](../docs/findings/enveda_180_dataset_2026_09_25.md)).

```bash
python preproc_scripts/04_prepare_split.py --split_type random --split_key inchikey_s \
    --test_frac 0.1 --val_frac 0.1 --meta_rseed 42 --primary_dsets enveda_180 \
    --proc_dp data/proc/enveda_180 --dag_filtering False \
    --prec_types "[M+H]+" "[M-H]-" "[M+Na]+" "[M+NH4]+" "[M+Cl]-" "[M+K]+" \
    --ce_types stepped single --split_dp data/split/enveda_180_inchikey_811
```

## Step 5: Preparing Optimal MAGMa DAGs (ICEBERG)

This step generates optimal MAGMa DAGs for training/evaluating the ICEBERG generator model optimal ICEBERG intensity model.

```bash
python preproc_scripts/05_prepare_magma_feats.py --magma_dp data/magma/gen/nist20 --proc_dp data/proc/nist20 --dsets nist20_hr
```

## Step 6: Preparing Approximate MAGMa DAGs (ICEBERG)

This step generates approximate MAGMa DAGs using a trained ICEBERG generator model, for training and evaluating the ICEBERG intensity model.

For each seed `${SEED}` (see [config README](../config/README.md)) repeat the following:

```bash
python preproc_scripts/06_predict_magma_dags.py --magma_dp data/magma/inten/nist20_inchikey_s$SEED --proc_dp data/proc/nist20 --gen_ckpt_fp data/magma/ckpt/nist20_inchikey_s$SEED.ckpt --dsets nist20_hr
```

Don't forget to symlink the formula directory:

```bash
ln -s $HOME/frag-gnn/data/magma/gen/nist20/magma_formula $HOME/frag-gnn/data/magma/inten/nist20_inchikey_s$SEED/magma_formula
```

## Step 8: Preparing ClassyFire Labels

This step merges ClassyFire chemical taxonomy annotations onto the molecule dataframe via SMILES canonicalization. The output is a pickle keyed by `mol_id` with columns: `kingdom`, `superklass`, `klass`, `subklass`, `direct_parent`, `alternative_parents`, `substituents`. Run once per dataset.

### NIST 20
```bash
python preproc_scripts/08_prepare_classyfire.py \
  --mol_df data/proc/nist20/mol_df.pkl \
  --cf_zip data/classyfire/20231201_nist20mona23_1.zip \
  --output data/classyfire/nist20_mol_classyfire.pkl
```

### NIST 23
```bash
python preproc_scripts/08_prepare_classyfire.py \
  --mol_df data/proc/nist23/mol_df.pkl \
  --cf_zip data/classyfire/20231201_nist23mona23_1.zip \
  --output data/classyfire/nist23_mol_classyfire.pkl
```
