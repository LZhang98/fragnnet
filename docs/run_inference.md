# FraGNNet Inference

## Data Preparation (Two-Step Process)

The inference input preparation is split into two scripts:
1. **Step 1**: `02_prepare_mol_df.py` - Create molecular dataframe from SMILES
2. **Step 2**: `03_prepare_spec_df.py` - Create spectrum dataframe from mol_df and experimental conditions

This separation allows independent execution and easier customization.

### 1. `02_prepare_mol_df.py`

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
python preproc_scripts/inference/02_prepare_mol_df.py \
  --smiles_list <SMILES> [<SMILES> ...] \
  --output_dir <OUTPUT_DIR> \
  [--max_heavy_atoms <MAX_ATOMS>]
```

From a text file:
```bash
python preproc_scripts/inference/02_prepare_mol_df.py \
  --smiles_file <SMILES_FILE> \
  --output_dir <OUTPUT_DIR>
```

From a CSV/TSV file with custom columns:
```bash
python preproc_scripts/inference/02_prepare_mol_df.py \
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
- `--max_heavy_atoms`: Maximum number of heavy atoms in a molecule (default: 60)

#### Sample Commands

##### Basic Single Molecule Example
```bash
python preproc_scripts/inference/02_prepare_mol_df.py \
  --smiles_list "CCO" \
  --output_dir output/single_molecule
```

##### From CSV File with Custom Columns
```bash
python preproc_scripts/inference/02_prepare_mol_df.py \
  --smiles_file data/metabolites.csv \
  --smiles_colname smiles \
  --id_colname compound_id \
  --output_dir output/metabolites
```

##### Multiple Molecules
```bash
python preproc_scripts/inference/02_prepare_mol_df.py \
  --smiles_list "CCO" "CC(=O)O" "CCN" "CC(C)O" \
  --output_dir output/multi_molecule
```

##### From SMILES File
```bash
# First create a SMILES file
echo -e "CCO\nCC(=O)O\nCCN\nCC(C)O\nCCC" > my_molecules.txt

# Then run the script
python preproc_scripts/inference/02_prepare_mol_df.py \
  --smiles_file my_molecules.txt \
  --output_dir output/from_file
```

##### Drug Molecules Example
```bash
python preproc_scripts/inference/02_prepare_mol_df.py \
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

### 2. `03_prepare_spec_df.py`

Prepares spectrum metadata dataframe from the mol_df created in step 1 and experimental conditions.

#### Features
- Creates all combinations of molecules × experimental conditions
- Handles instrument-specific collision energy units (NCE for Orbitrap, eV for others)
- Supports multiple precursor types and instruments in a single run
- Flexible merged spectra mode for simplified predictions
- Automatically computes precursor m/z values
- Generates `spec_df.pkl.gz` and `spec_df_config.json` files
- Detailed logging of spectrum generation

#### Usage

Basic usage with experimental conditions:
```bash
python preproc_scripts/inference/03_prepare_spec_df.py \
  --mol_df_path <MOL_DF_PATH> \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --ft_ces <NCE_VALS> [<NCE_VALS> ...] \
  --inst_types <INST_TYPE> [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR> \
  [--merged_spectra true|false]
```

For QToF instruments with eV values:
```bash
python preproc_scripts/inference/03_prepare_spec_df.py \
  --mol_df_path <MOL_DF_PATH> \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --qtof_ces <CE_EV> [<CE_EV> ...] \
  --inst_types "QTOF" [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR>
```

#### Arguments

**Input**:
- `--mol_df_path`: Path to `mol_df.pkl.gz` file (required, created by `02_prepare_mol_df.py`)

**Experimental Conditions** (at least one required):
- `--prec_types`: Precursor types. Choices: `[M+H]+`, `[M-H]-` (required)
- `--qtof_ces`: Collision energies in eV for QToF instruments (e.g., `10.0eV 20.0eV`)
- `--ft_ces`: Normalized collision energies for Orbitrap (e.g., `15.0NCE 25.0% 30.0eV`)
- `--inst_types`: Instrument types. Choices: `FT`, `QTOF`, `IT`, `Q`, `TOF`, `QQQ` (required)

**Output Options**:
- `--output_dir`: Output directory for `spec_df.pkl.gz` and `spec_df_config.json`
- `--merged_spectra`: Generate merged spectra (one per prec_type + inst_type combo). Choices: `true`, `false` (default: `false`)

#### Sample Commands

##### Single Condition (FT Orbitrap with NCE)
```bash
python preproc_scripts/inference/03_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --ft_ces 25.0NCE \
  --inst_types "FT" \
  --output_dir output/
```

##### Multiple Conditions
```bash
python preproc_scripts/inference/03_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" "[M-H]-" \
  --ft_ces 15.0NCE 25.0NCE 35.0NCE \
  --inst_types "FT" "QTOF" \
  --output_dir output/
```

##### Merged Spectra Mode
```bash
python preproc_scripts/inference/03_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --qtof_ces 20.0eV \
  --inst_types "QTOF" \
  --merged_spectra true \
  --output_dir output/
```

##### QToF with eV Collision Energies
```bash
python preproc_scripts/inference/03_prepare_spec_df.py \
  --mol_df_path output/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --qtof_ces 10.0eV 20.0eV 30.0eV \
  --inst_types "QTOF" \
  --output_dir output/
```

##### Orbitrap Only
```bash
python preproc_scripts/inference/03_prepare_spec_df.py \
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
| IT (Ion Trap)   | NCE (%) | `nce` | Normalized Collision Energy |
| QTOF            | eV      | `ace` | Absolute Collision Energy |
| Q (Single Quad) | eV      | `ace` | Absolute Collision Energy |
| TOF             | eV      | `ace` | Absolute Collision Energy |
| QQQ (Triple Quad) | eV    | `ace` | Absolute Collision Energy |

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

When `--merged_spectra true` is specified:
- Generates one spectrum entry per unique (prec_type, inst_type) combination across all molecules
- Groups spectra by precursor type and instrument type
- Simplifies downstream analysis for molecule-level predictions
- Reduces total number of predictions

---

## Complete Workflow Example

### Step-by-step example:

```bash
# Step 1: Create molecular dataframe
python preproc_scripts/inference/02_prepare_mol_df.py \
  --smiles_list "CCO" "CC(=O)O" "CCN" \
  --output_dir data/inference

# Step 2: Create spectrum dataframe with multiple conditions
python preproc_scripts/inference/03_prepare_spec_df.py \
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
python preproc_scripts/inference/02_prepare_mol_df.py \
  --smiles_file compounds.csv \
  --smiles_colname smiles \
  --id_colname compound_id \
  --output_dir data/inference

# Step 2: Create spectrum dataframe for Orbitrap
python preproc_scripts/inference/03_prepare_spec_df.py \
  --mol_df_path data/inference/mol_df.pkl.gz \
  --prec_types "[M+H]+" \
  --ft_ces 10eV 20eV 30eV \
  --inst_types FT \
  --output_dir data/inference
```

---

### 3. `02_run_inference_frag_gen.py`

Runs FraGNNet inference on prepared data and saves predictions. Supports resuming from interrupted runs.

#### Usage

```bash
python scripts/inference/run_inference.py \
  --mol_fp <MOL_DF_PATH> \
  --spec_fp <SPEC_DF_PATH> \
  --frag_dp <FRAG_DIR> \
  --model_ckpt <MODEL_CHECKPOINT> \
  --save_dp <OUTPUT_DIRECTORY> \
  [--save_format parquet|pkl]
```

#### Arguments

**Input Data**:
- `--mol_fp`: Path to `mol_df.pkl.gz` (from step 1)
- `--spec_fp`: Path to `spec_df.pkl.gz` (from step 1)
- `--frag_dp`: Directory containing precomputed fragment graphs (from step 2)

**Model Loading**:
- `--model_ckpt`: Path to local model checkpoint (.ckpt)
- `--wandb_run_id`: W&B run ID to download checkpoint from (if model_ckpt not provided)
- `--custom_fp`: Path to custom config file (optional)

**Output Options**:
- `--save_dp`: Directory to save inference results
- `--save_format`: Output format (`parquet` or `pkl`, default: `parquet`)
- `--save_chunk_size`: Number of predictions per chunk file (default: 10,000)
- `--output_subset`: Subset of columns to output (optional)

**Inference Options**:
- `--eval_batch_size`: Batch size for inference (default: 1)
- `--min_intensity`: Minimum intensity threshold for peaks (default: 1.0)
- `--disable_preproc`: Disable dataset preprocessing to save memory (default: True)

#### Resume Capability
The script automatically checks the output directory (`--save_dp`) for existing prediction chunks. If found, it skips the spectra that have already been processed, allowing you to resume interrupted jobs without restarting from scratch.

## Workflow Overview

1. **Prepare Molecules**: Use `02_prepare_mol_df.py` to create molecule dataframe from SMILES
2. **Prepare Spectra**: Use `03_prepare_spec_df.py` to create spectrum dataframe with experimental conditions
3. **Generate Fragments**: Use `02_run_inference_frag_gen.py` to generate fragmentation trees
4. **Run Inference**: Use `run_inference.py` to generate predictions
5. **Analyze Results**: Process the prediction outputs for your analysis

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
- Filters by heavy atom limit (default: 60 atoms)
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
