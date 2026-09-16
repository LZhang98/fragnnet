# FraGNNet Inference

> **Status:** GUIDE &nbsp;·&nbsp; **Updated:** 2026-01-23
>
> How to run inference end to end.

## Data Preparation (Two-Step Process)

The inference input preparation is split into two scripts:

1. **Step 1**: `01_inf_prepare_mol_df.py` - Create molecular dataframe from SMILES
2. **Step 2**: `02_inf_prepare_spec_df.py` - Create spectrum dataframe from mol_df and experimental conditions

This separation allows independent execution and easier customization.

### 1. `01_inf_prepare_mol_df.py`

Prepares a molecular dataframe for FraGNNet inference from a list of SMILES strings.

#### Features
- Converts SMILES to RDKit molecules with computed properties
- Supports multiple input formats: command-line lists, text files, CSV/TSV files
- Flexible molecule ID handling with custom column mapping
- Quality filtering: removes non-single molecules, charged molecules, and radicals
- Comprehensive molecular property computation and validation
- Generates `mol_df.pkl.gz` and `mol_df_config.json` files
- Detailed logging and error handling

#### Usage

From command line SMILES list:
```bash
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_list <SMILES> [<SMILES> ...] \
  --output_dir <OUTPUT_DIR> \
  [--max_heavy_atoms <MAX_ATOMS>]
```

From a text file:
```bash
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_file <SMILES_FILE> \
  --output_dir <OUTPUT_DIR>
```

From a CSV/TSV file with custom columns:
```bash
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_file <CSV_FILE> \
  --smiles_colname <SMILES_COLUMN> \
  --id_colname <ID_COLUMN> \
  --output_dir <OUTPUT_DIR>
```

#### Arguments

**Input Options** (mutually exclusive):
- `--smiles_list`: List of SMILES strings
- `--smiles_file`: File containing SMILES (supports .txt, .smi, .smiles, .csv, .tsv)

**File Parsing Options** (for CSV/TSV files):
- `--smiles_colname`: Column name containing SMILES strings
- `--id_colname`: Column name containing molecule IDs (optional)

**Output Options**:
- `--output_dir`: Output directory for `mol_df.pkl.gz` and `mol_df_config.json`
- `--max_heavy_atoms`: Maximum number of heavy atoms in a molecule (default: 128)

#### Sample Commands

##### Basic Single Molecule Example
```bash
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_list "CCO" \
  --output_dir output/single_molecule
```

##### From CSV File with Custom Columns
```bash
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_file data/metabolites.csv \
  --smiles_colname smiles \
  --id_colname compound_id \
  --output_dir output/metabolites
```

##### Multiple Molecules
```bash
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_list "CCO" "CC(=O)O" "CCN" "CC(C)O" \
  --output_dir output/multi_molecule
```

##### From SMILES File
```bash
# First create a SMILES file
echo -e "CCO\nCC(=O)O\nCCN\nCC(C)O\nCCC" > my_molecules.txt

# Then run the script
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_file my_molecules.txt \
  --output_dir output/from_file
```

##### Drug Molecules Example
```bash
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_list \
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F" \
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \
    "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --output_dir output/drugs
```

#### Output Files

The script generates two files in the specified output directory:

1. **`mol_df.pkl.gz`**: Molecule dataframe with computed properties
   - Columns: `smiles`, `mol_id`, `input_mol_id`, `mol`, `inchikey_s`, `formula`, `mw`, `exact_mw`, `num_atoms`, `num_bonds`, `charge`, `single_mol`, `num_radicals`, `num_heavy_atoms`, `num_out_set_elements`
   - Only includes molecules that pass quality filtering (single molecules, no charge, no radicals)

2. **`mol_df_config.json`**: Configuration file with input parameters

#### Quality Filtering

The script automatically filters out molecules that don't meet quality standards:
- **Non-single molecules**: Removes molecules that contain multiple disconnected components
- **Charged molecules**: Removes molecules with formal charges (charge ≠ 0)
- **Radical molecules**: Removes molecules with unpaired electrons (num_radicals ≠ 0)
- **Heavy atoms**: Removes molecules exceeding `--max_heavy_atoms` limit
- **Out-of-set elements**: Removes molecules containing elements not in the fragmentation model

Filtered molecules are logged with detailed counts and reasons for removal.

---

### 2. `02_inf_prepare_spec_df.py`

Prepares spectrum metadata dataframe from the mol_df created in step 1 and experimental conditions.

#### Features
- Creates all combinations of molecules × experimental conditions
- Handles instrument-specific collision energy units (NCE for Orbitrap, eV for others)
- Supports multiple precursor types and FT/QTOF instruments in a single run
- Supports alternate grouping with merged spectra mode
- Automatically computes precursor m/z values
- Generates `spec_df.pkl.gz` and `spec_df_config.json` files
- Detailed logging of spectrum generation

#### Usage

Basic usage with experimental conditions:
```bash
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path <MOL_DF_PATH> \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --ft_ces <NCE_VALS> [<NCE_VALS> ...] \
  --inst_types <INST_TYPE> [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR> \
  [--merged_spectra true|false]
```

For QToF instruments with eV values:
```bash
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path <MOL_DF_PATH> \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --qtof_ces <CE_EV> [<CE_EV> ...] \
  --inst_types "QTOF" [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR>
```

#### Arguments

**Input**:
- `--mol_df_path`: Path to `mol_df.pkl.gz` file (required, created by `01_inf_prepare_mol_df.py`)

**Experimental Conditions** (at least one required):
- `--prec_types`: Precursor types. Choices: `[M+H]+`, `[M-H]-` (required)
- `--qtof_ces`: Collision energies in eV for QToF instruments (e.g., `10.0eV 20.0eV`)
- `--ft_ces`: Normalized collision energies for Orbitrap (e.g., `15.0NCE 25.0% 30.0eV`)
- `--inst_types`: Instrument types (required). The current preparation logic supports `FT` and `QTOF`; other parser choices are not implemented.

**Output Options**:
- `--output_dir`: Output directory for `spec_df.pkl.gz` and `spec_df_config.json`
- `--merged_spectra`: Generate merged spectra (one per prec_type + inst_type combo). Choices: `true`, `false` (default: `false`)

#### Sample Commands

##### Single Condition (FT Orbitrap with NCE)
```bash
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --ft_ces 25.0NCE \
  --inst_types "FT" \
  --output_dir output/
```

##### Multiple Conditions
```bash
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" "[M-H]-" \
  --ft_ces 15.0NCE 25.0NCE 35.0NCE \
  --inst_types "FT" "QTOF" \
  --output_dir output/
```

##### Merged Spectra Mode
```bash
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --qtof_ces 20.0eV \
  --inst_types "QTOF" \
  --merged_spectra true \
  --output_dir output/
```

##### QToF with eV Collision Energies
```bash
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --qtof_ces 10.0eV 20.0eV 30.0eV \
  --inst_types "QTOF" \
  --output_dir output/
```

##### Orbitrap Only
```bash
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --ft_ces 15.0% 25.0% 35.0% \
  --inst_types "FT" \
  --output_dir output/
```

#### Collision Energy Units

Different instrument types use different collision energy units:

| Instrument Type | Unit | Field Used | Description |
|-----------------|------|------------|-------------|
| FT (Orbitrap)   | NCE (%) | `nce` | Normalized Collision Energy |
| QTOF            | eV      | `ace` | Absolute Collision Energy |

Although the CLI parser also lists `IT`, `Q`, `TOF`, and `QQQ`, collision-energy combination
generation for those instruments is not implemented.

#### Output Files

The script generates two files in the specified output directory:

1. **`spec_df.pkl.gz`**: Spectrum dataframe with experimental conditions
   - Columns: `spec_id`, `mol_id`, `prec_type`, `nce`, `ace`, `inst_type`, `group_id`, `dset`, `dset_spec_id`, `prec_mz`, `peaks`, `frag_mode`, `spec_type`, `ion_mode`, `formula`, `inchikey`, `exact_mass`
   - In merged spectra mode: one entry per (molecule, prec_type, inst_type) combination
   - In standard mode: one entry per (molecule, prec_type, collision_energy, inst_type) combination

2. **`spec_df_config.json`**: Configuration file with:
   - Input parameters (precursor types, collision energies, instrument types)
   - Detailed combination list for reproducibility
   - Output file paths

#### Merged Spectra Mode

When `--merged_spectra true` is specified, the script retains one row per molecule and condition
but assigns a shared `group_id` to rows with the same precursor and instrument types. It does not
reduce the number of spectrum rows.

---

## Complete Workflow Example

### Step-by-step example:

```bash
# Step 1: Create molecular dataframe
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_list "CCO" "CC(=O)O" "CCN" \
  --output_dir data/inference

# Step 2: Create spectrum dataframe with multiple conditions
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path data/inference/mol_df.pkl.gz \
  --prec_types "[M+H]+" "[M-H]-" \
  --ft_ces 15.0NCE 25.0NCE 35.0NCE \
  --qtof_ces 20.0eV 30.0eV \
  --inst_types "FT" "QTOF" \
  --output_dir data/inference
```

### Step-by-step with custom CSV:

```bash
# Step 1: Create molecular dataframe from CSV
python preproc_scripts/inference/01_inf_prepare_mol_df.py \
  --smiles_file compounds.csv \
  --smiles_colname smiles \
  --id_colname compound_id \
  --output_dir data/inference

# Step 2: Create spectrum dataframe for Orbitrap
python preproc_scripts/inference/02_inf_prepare_spec_df.py \
  --mol_df_path data/inference/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --ft_ces 10eV 20eV 30eV \
  --inst_types FT \
  --output_dir data/inference
```

---

### 3. `scripts/run_inference.py`

Runs FraGNNet inference on prepared data and saves predictions. Supports resuming from interrupted
runs. If `--frag_dp` is omitted, the script generates fragment DAGs in a temporary directory.

#### Usage

```bash
mkdir -p <OUTPUT_DIRECTORY>

python scripts/run_inference.py \
  --mol_fp <MOL_DF_PATH> \
  --spec_fp <SPEC_DF_PATH> \
  --model_ckpt <MODEL_CHECKPOINT> \
  --save_dp <OUTPUT_DIRECTORY> \
  [--frag_dp <FRAG_DIR>] \
  [--save_format pkl|pkl.gz]
```

#### Arguments

**Input Data**:
- `--mol_fp`: Path to `mol_df.pkl.gz` (from step 1)
- `--spec_fp`: Path to `spec_df.pkl.gz` (from step 2)
- `--frag_dp`: Optional directory containing precomputed fragment graphs. If omitted, DAGs are generated automatically.

**Model Loading**:
- `--model_ckpt`: Path to local model checkpoint (.ckpt)
- `--wandb_run_id`: W&B run ID to download checkpoint from (if model_ckpt not provided)
- `--custom_fp`: Path to custom config file (optional)

**Output Options**:
- `--save_dp`: Directory to save inference results
- `--save_format`: Output format (`pkl` or `pkl.gz`; default: `pkl`). Parquet is not
  supported — prediction cells are `torch.Tensor`, which pyarrow cannot serialize.
- `--save_chunk_size`: Number of predictions per chunk file (default: 10,000)
- `--output_subset`: Which prediction arrays to compute (optional; `--list_output_subset`
  prints the tokens). Everything outside the subset comes back as NaN and is removed by
  `--drop_none_cols`.
- `--return_formula_str` / `--return_cc_hash`: annotation columns (default: True). See
  [Annotation without probabilities](#annotation-without-probabilities).

**Inference Options**:
- `--batch_size`: Fixed batch size for inference (default: 16, matching `eval_batch_size` in
  `config/template.yml`). Ignored when `--dynamic_batch_max` is set. Do not set it to 1
  casually: besides the obvious per-call overhead, it is also what `init_dataloader` passes as
  `return_batch_at`, so a config with `dynamic_batch_sampler: True` would flush a batch after
  every single sample and never reach `dynamic_batch_sampler_max`. Batch size 1 has also
  historically masked index-offset bugs, since spectrum 0 has offset 0 under any scheme.
- `--dynamic_batch_max` / `--dynamic_batch_limited_by`: pack batches greedily up to this many
  `frag_node` (default) or `frag_edge` per batch instead of using a fixed `--batch_size`. Backed
  by `IterableSpecMolFragDataset`, which forms batches lazily while streaming, so each DAG is
  loaded exactly once and no size cache (`meta_info.json` / `precom_meta_info`) is needed. The
  tradeoff is that the batch count is unknown until the run finishes, so the progress bar has no
  total. Dynamic batching currently runs with `num_workers=0`; the fixed-batch path still uses
  the configured worker count. For scale, training uses `dynamic_batch_sampler_max: 100000` in
  `frag_node` units.
  Only pass `frag_edge` for a config that materializes fragment edges — otherwise `num_edges` is
  0 for every sample, which would put the entire run in a single batch (this now raises rather
  than OOMing).
- `--min_intensity`: Minimum intensity threshold for peaks (default: 1.0). `pred_ints` is
  rescaled per spectrum to a 0-1000 base peak and filtered with a strict `>`, so the value is
  percent-of-base-peak x 10 (1.0 = 0.1%, 10.0 = 1%). Measured on NIST20 val (2,000 spectra,
  `9t63zils`): cosine is flat to four decimals from 0.0 through 10.0 (cos_hun 0.5638 -> 0.5637)
  while the peak count falls ~3x (p50 39 -> 13, p90 202 -> 60), and 13/60 matches what real
  spectra carry (p50 11, p90 45). **Prefer 10.0 for predicted libraries** - same score, ~3x
  smaller output. Keep it low only if you need to recover DAG support from the export, which
  even 1.0 largely prevents (see
  [dominant_peak_outcome_decomposition.md](../findings/dominant_peak_outcome_decomposition.md)).
- `--disable_preproc`: Disable dataset preprocessing to save memory (default: True). This
  forces both `frag_params.preprocess` and `frag_params.preload` off, so DAGs are read and
  processed on access. Since `__getitem__` is per-spectrum while a DAG is per-molecule, the
  repeated work is absorbed by a per-molecule LRU sized by `frag_params.cache_size` (default
  16, 0 disables). Hit rate depends on how clustered a molecule's rows are in the input
  `spec_df` — sorting it by `mol_id` maximizes reuse. Dynamic batching runs in one process, so it
  also shares one LRU across the full index stream.

#### Fragment annotation

`pred_formula_strs`, `pred_formula_adducts` and `pred_node_ccs` carry the fragment
annotation. They are addressed by `pred_formula_formula_idxs` / `pred_node_node_idxs`,
which used to be requested only by the `pred_formula_probs` / `pred_node_probs` tokens — so
annotation came out NaN unless a probability was also asked for. `--return_formula_str` /
`--return_cc_hash` now request those index arrays themselves (and, for cc hashes, the joint
grid that prunes them).

```bash
--output_subset pred_mzs pred_ints pred_formula_probs pred_formula_node_probs \
--return_formula_str True --return_cc_hash True --min_intensity 10.0
```

All indices are rebased per spectrum, so every lookup below is local:

| Column | One entry per | Meaning |
| --- | --- | --- |
| `pred_mzs`, `pred_ints` | peak | the spectrum |
| `frag_formula_peak_idxs` | peak | peak -> formula |
| `pred_formula_strs` | formula | **neutral** fragment formula |
| `pred_formula_adducts` | formula | precursor delta for that row, signed (`+H`, `-H`, `+Na`) |
| `pred_formula_probs` | formula | P(f) |
| `pred_joint_formula_idxs`, `pred_joint_node_idxs` | joint row | formula -> candidate node |
| `pred_formula_node_probs` | joint row | P(n\|f) |
| `pred_node_ccs` | node | atom bitmask (2 x int64, `MASK_SIZE=128`) |

Peak `k`'s formula is `pred_formula_strs[frag_formula_peak_idxs[k]]`. Its fragment is the
`argmax` of `pred_formula_node_probs` over the joint rows carrying that formula — which is
why the annotation needs P(n|f): one formula is generally explained by several DAG nodes
(4 on average, see below). Resolve the cc bitmask to a SMILES against the parent molecule
downstream.

**The formula string is the neutral fragment, not the ion.** `pred_formula_adducts` carries
the precursor delta that turns it into the observed ion, so a peak reads as
`"C6H4O2 -H"` and the annotated fragment is that cc **plus** that adduct:

```python
f = row["frag_formula_peak_idxs"][k]                       # peak -> formula row
rows = (row["pred_joint_formula_idxs"] == f).nonzero()     # its candidate nodes
n = row["pred_joint_node_idxs"][rows[row["pred_formula_node_probs"][rows].argmax()]]

ion_formula = f"{row['pred_formula_strs'][f]} {row['pred_formula_adducts'][f]}"
fragment    = row["pred_node_ccs"][n]                      # atoms, ionized by that adduct
```

Verified against the measured m/z: neutral formula + exported delta reproduces every
`pred_mzs` entry to 2.6e-5 Da.

**Charge migration (CMF).** With `frag_params.include_cmf: True` the formula axis is laid
out as `NULL, CRF[1:F], CMF[1:F]`, and the two halves carry *different* deltas — for
`[M+Na]+` the CRF fragment keeps the sodium (`+Na`) while the CMF fragment is protonated
(`+H`). The same neutral formula therefore appears twice with two adducts, and
`pred_formula_adducts` is what tells them apart. `[M+H]+` and `[M-H]-` emit no CMF half at
all: their CMF ion is already the CRF state at `delta_h = -/+1`
(`cmf_duplicate_crf_h_shift`), so every row keeps the CRF delta.

`pred_formula_adducts` is emitted whenever **either** `--return_formula_str` or
`--return_cc_hash` is set: it is what makes both of them resolvable.

**`--min_intensity` prunes the node arrays too.** It used to prune peaks, formulas and the
joint grid but leave `pred_node_probs` / `pred_node_node_idxs` — and therefore
`pred_node_ccs` — covering every node of the DAG at any threshold. Measured on NIST20 val
with `9t63zils`, per spectrum:

| Array | `--min_intensity 0` (400 spectra) | `--min_intensity 10.0` (1,000 spectra) |
| --- | --- | --- |
| peaks | p50 2,075 / mean 2,689 | p50 12 / mean 20.8 |
| formulas | p50 2,076 / mean 2,690 | p50 12 / mean 20.8 |
| joint rows | p50 8,545 / mean 29,884 | p50 49 / mean 102.9 |
| nodes | p50 1,042 / mean 2,350 | p50 38 / mean 64.4 |

Over the SYNCED set (4.05M spectra) that is the difference between ~152 GB and ~4.2 GB of
cc bitmasks, and ~2.4 TB and ~8.3 GB of joint rows. The node count was threshold-independent
before the fix, so the left column is also what `--min_intensity 10.0` used to produce.

#### Resume Capability
The script automatically checks the output directory (`--save_dp`) for existing prediction chunks. If found, it skips the spectra that have already been processed, allowing you to resume interrupted jobs without restarting from scratch.

#### Sharding (SLURM array jobs)

`--start_mol_idx` / `--num_mols` slice **mol_df by row position** and keep only the spectra
of the selected molecules. The DAG preprocessing arrays
(`preproc_infer_synced_dag_arr*.sh`) shard mol_df the same way, so array task *i* of the
inference job lines up with frag shard *i* and the shards do **not** have to be merged
first — `merge_dag_h5_shards.py` is only needed to run the whole set as a single job.

`--frag_dp` takes either backend:

| Preprocessing output | `--frag_dp` argument |
| --- | --- |
| tar of `<save_name>/dags/<mol_id>.pkl.bz2` | the extracted `<save_name>/dags` directory |
| `shard_<i>/dags.h5` (packed HDF5) | the `dags.h5` file itself |

For HDF5 the `meta_info.json` written next to the `.h5` must be staged with it: it is both
the dataset's node/edge count source and the list of molecules that actually fragmented
(the HDF5 backend does no missing-entry check, so a molecule that failed fragmentation
would otherwise raise a `KeyError` at dataset build time).

Each task must get its **own** `--save_dp`: chunk files are named
`predictions_chunk_<n>.<ext>` and are only unique within a directory. Resume then works
per task.

#### Choosing the checkpoint

`run_inference.py` takes either `--model_ckpt <path>` or `--wandb_run_id <id>`
(`--model_save_dp` is the download cache). `get_ckpt_path` falls back to the W&B
download whenever `--model_ckpt` is absent **or points at a file that does not exist** —
a typo'd path silently becomes a W&B lookup, so check the `>>> Using local checkpoint:`
line in the log.

`--wandb_run_id` works from the compute nodes, but in an **array** job every task
resolves the checkpoint independently. With a cold cache they all download and all
`shutil.copy` — not atomic — onto the same `<model_save_dp>/<run_name>_<run_id>.ckpt`,
so a task can read a half-written file. Warm the cache once before submitting:

```bash
python -c "from fragnnet.utils.misc_utils import get_best_ckpt_from_wandb; \
           print(get_best_ckpt_from_wandb('saved_ckpts', '<run_id>'))"
```

After that the tasks just hit the cache (`use_cached=True`); passing the printed path as
`--model_ckpt` is equivalent.

In the array scripts these are the `MODEL_CKPT` / `WANDB_RUN_ID` variables at the top of
`infer_synced_arr*.sh` (`MODEL_CKPT` wins when both are set; relative paths resolve
against `FRAGNNET_DP`), with `CUSTOM_FP` for the checkpoint's training config.

Ready-made array jobs for the SYNCED set:

```bash
# tar backend
sbatch scripts/manual_slurm/fir/infer_synced_arr.sh
# packed HDF5 backend
sbatch scripts/manual_slurm/fir/infer_synced_arr_hdf5.sh
```

Both run one task per frag shard (`--array=0-14`, `CHUNK_SIZE=20000`): task i consumes
shard i and predicts exactly the mol_df slice that built it, so no merge step is needed.
A full sweep needs `--array` to cover exactly the shards preprocessing wrote. A narrower
array is allowed — `sbatch --array=0-0` to run one shard, or a subset to rerun failed
tasks — and the scripts only print a note comparing task count to shard count. Nothing
fails when shards are left out: those molecules are simply not predicted and every task
still exits 0, so check the note before calling a sweep complete.

Every chunk save and the end of the run print a `>> resources:` line to the task's
`#SBATCH --output` file — throughput plus peak GPU and host memory, as high-water marks
rather than samples. This is unconditional; `WANDB_TRACK_MODE=online sbatch ...` (which the
scripts pass through as `--wandb_mode`, default `disabled`) only adds the same numbers to
a W&B run. Size the next `--gpu_profile` and `--time` from that line,
not from a guess. Note that NVML reports no SM utilization for a MIG slice, so W&B's
system panels cannot tell you whether a `h100_3g.40gb` instance is saturated — only the
memory figures and throughput are available.

A task that runs out of wall time is resumed by resubmitting the array — each task skips
the spec_ids already present in its own output dir. Output goes to
`${OUT_DP}/shard_<task>/`, where `OUT_DP` carries the model tag (the W&B run id, or the
checkpoint basename) so two models never share a directory and resume across them.

## Workflow Overview

1. **Prepare Molecules**: Use `01_inf_prepare_mol_df.py` to create the molecule dataframe from SMILES.
2. **Prepare Spectra**: Use `02_inf_prepare_spec_df.py` to create the spectrum dataframe with experimental conditions.
3. **Run Inference**: Use `scripts/run_inference.py` to generate fragment DAGs if needed and save predictions.
4. **Analyze Results**: Process the prediction outputs for your analysis.

## Important Notes

### File Format Requirements

- Mol_df and spec_df files use pickle compression (`.pkl.gz` format)
- Spec_df must be created from mol_df in the same session or from the same output directory
- Configuration files are JSON-formatted for human readability

### Input/Output Formats

- Input files support multiple formats: .txt/.smi/.smiles (one SMILES per line) and .csv/.tsv (with column headers)
- For CSV/TSV files, `--smiles_colname` is required; `--id_colname` is optional for custom molecule IDs
- Invalid SMILES strings are automatically filtered out with warnings

### Quality Filtering

- Removes non-single molecules, charged molecules, and molecules with radicals
- Filters by heavy atom limit (default: 128 atoms)
- Removes molecules with out-of-set elements not in fragmentation model
- All filtering criteria are logged with detailed counts

### Molecular Properties

- All molecular properties are computed using RDKit
- Includes: MW, exact mass, formula, InChIKey, atom/bond counts, charge, radicals
- Precursor m/z values are automatically calculated from molecular mass and precursor type

### Collision Energies

- FT/Orbitrap: Use NCE (%) values
- QTOF/Q/TOF/QQQ: Use eV values
- Each instrument type uses appropriate energy values automatically

### Merged Spectra Mode

- The `--merged_spectra` option accepts various boolean formats: `true/false`, `yes/no`, `1/0`
- Merged mode reduces the number of predictions while maintaining experimental diversity
- One entry per (molecule, prec_type, inst_type) combination instead of per collision energy

### Reproducibility

- Configuration JSON files provide complete traceability of all parameters
- Timestamps recorded for all runs
- All input parameters logged to config files

## Inference peak threshold (`--min_intensity`)

`scripts/run_inference.py` rescales `pred_ints` per spectrum to a 0-1000 base peak, then keeps
peaks with intensity **strictly greater than** `--min_intensity`. The value is therefore
percent-of-base-peak x 10.

Measured on NIST20 val (2,000 spectra, `9t63zils`):

| `--min_intensity` | % base peak | pred peaks p50 | pred peaks p90 | cos_hun | delta |
| ----------------- | ----------- | -------------- | -------------- | ------- | ------ |
| 0.0               | no filter   | 418            | 3138           | 0.5638  | —      |
| 1.0 (script default) | 0.1%     | 39             | 202            | 0.5638  | -0.0000 |
| 5.0               | 0.5%        | 18             | 87             | 0.5638  | -0.0000 |
| **10.0**          | **1%**      | **13**         | **60**         | 0.5637  | **-0.0001** |
| 20.0              | 2%          | 9              | 41             | 0.5635  | -0.0003 |
| 50.0              | 5%          | 6              | 23             | 0.5628  | -0.0010 |

Cosine is flat to four decimals through 1% while the peak count drops ~3x. Real spectra in the
same subset carry p50 11 / p90 45 peaks, so **10.0 matches the reference peak count and 1.0
over-emits by 3-4x** — prefer `10.0` for predicted libraries and spectral matching. It is the
default in `scripts/manual_slurm/fir/infer_synced_arr*.sh`.

Keep the threshold low only if you need to recover DAG support from the export: even 1.0 leaves
the median spectrum with just 5.3% of its candidate m/z, so peak-coverage analysis needs a
low-threshold re-run, not a re-filter. See
[docs/findings/dominant_peak_outcome_decomposition.md](docs/findings/dominant_peak_outcome_decomposition.md).

Sweep it yourself with [scripts/debug/debug_peak_count_diag.py](scripts/debug/debug_peak_count_diag.py),
which reports peak counts and `cos_hun` at `[0, 1, 5, 10, 20, 50]`. Ground-truth peak counts per
adduct x instrument x CE are in `notebooks/debug_notebooks/debug_nits20_peaks.ipynb` (cells 8-9).
