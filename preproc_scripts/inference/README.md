# FraGNNet Inference Scripts

This directory contains scripts for preparing data and running inference with FraGNNet models.

## Scripts

### 1. `prepare_inference_input.py`

Prepares molecular and spectral dataframes for FraGNNet inference from lists of SMILES, precursor types, collision energies, and instrument types.

#### Features
- Converts SMILES to RDKit molecules with computed properties
- Creates all combinations of molecules × experimental conditions
- Handles instrument-specific collision energy units (NCE for Orbitrap, eV for others)
- Supports multiple input formats: command-line lists, text files, CSV/TSV files
- Flexible molecule ID handling with custom column mapping
- Quality filtering: removes non-single molecules, charged molecules, and radicals
- Merged spectra mode for simplified predictions (one entry per molecule)
- Comprehensive molecular property computation and validation
- Generates `mol_df.pkl.gz`, `spec_df.pkl.gz`, and `config.json` files
- Detailed logging and error handling

#### Usage

From command line SMILES list:
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_list <SMILES> [<SMILES> ...] \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --ces <CE_EV> [<CE_EV> ...] \
  --nces <NCE_PCT> [<NCE_PCT> ...] \
  --inst_types <INST_TYPE> [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR> \
  [--merged_spectra true|false]
```

From a text file:
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_file <SMILES_FILE> \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --ces <CE_EV> [<CE_EV> ...] \
  --nces <NCE_PCT> [<NCE_PCT> ...] \
  --inst_types <INST_TYPE> [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR>
```

From a CSV/TSV file with custom columns:
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_file <CSV_FILE> \
  --smiles_colname <SMILES_COLUMN> \
  --id_colname <ID_COLUMN> \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --ces <CE_EV> [<CE_EV> ...] \
  --nces <NCE_PCT> [<NCE_PCT> ...] \
  --inst_types <INST_TYPE> [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR>
```

#### Arguments

**Input Options** (mutually exclusive):
- `--smiles_list`: List of SMILES strings
- `--smiles_file`: File containing SMILES (supports .txt, .smi, .smiles, .csv, .tsv)

**File Parsing Options** (for CSV/TSV files):
- `--smiles_colname`: Column name containing SMILES strings
- `--id_colname`: Column name containing molecule IDs (optional)

**Experimental Conditions** (all required):
- `--prec_types`: Precursor types. Choices: `[M+H]+`, `[M-H]-`
- `--ces`: Collision energies in eV (used by QTOF, Q, TOF, QQQ instruments)
- `--nces`: Normalized collision energies in % (used by FT, IT instruments)
- `--inst_types`: Instrument types. Choices: `FT`, `QTOF`, `IT`, `Q`, `TOF`, `QQQ`

**Output Options**:
- `--output_dir`: Output directory for generated files
- `--merged_spectra`: Generate merged spectra (one per adduct+mol+inst_type). Choices: `true`, `false`

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

#### Sample Commands

##### Basic Single Molecule Example
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_list "CCO" \
  --prec_types "[M+H]+" \
  --ces 15.0 25.0 \
  --nces 20.0 30.0 \
  --inst_types "QTOF" "FT" \
  --output_dir output/single_molecule
```

##### Merged Spectra Example (One Prediction Per Molecule)
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_list "CCO" "CC(=O)O" "CCN" \
  --prec_types "[M+H]+" "[M-H]-" \
  --ces 20.0 30.0 \
  --nces 25.0 35.0 \
  --inst_types "QTOF" "FT" \
  --merged_spectra true \
  --output_dir output/merged_spectra
```

##### From CSV File with Custom Columns
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_file data/metabolites.csv \
  --smiles_colname smiles \
  --id_colname compound_id \
  --prec_types "[M+H]+" "[M-H]-" \
  --ces 10.0 20.0 40.0 \
  --nces 20.0 40.0 60.0 \
  --inst_types "QTOF" "FT" \
  --output_dir output/metabolites
```

##### Multiple Molecules with Multiple Conditions
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_list "CCO" "CC(=O)O" "CCN" "CC(C)O" \
  --prec_types "[M+H]+" "[M-H]-" \
  --ces 10.0 20.0 30.0 \
  --nces 15.0 25.0 35.0 \
  --inst_types "QTOF" "FT" "IT" \
  --output_dir output/multi_molecule
```

##### From SMILES File
```bash
# First create a SMILES file
echo -e "CCO\nCC(=O)O\nCCN\nCC(C)O\nCCC" > my_molecules.txt

# Then run the script
python scripts/inference/prepare_inference_input.py \
  --smiles_file my_molecules.txt \
  --prec_types "[M+H]+" \
  --ces 15.0 25.0 \
  --nces 20.0 30.0 \
  --inst_types "QTOF" "FT" \
  --output_dir output/from_file
```

##### Drug Molecules Example
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_list \
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F" \
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \
    "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --prec_types "[M+H]+" "[M-H]-" \
  --ces 20.0 30.0 \
  --nces 25.0 35.0 \
  --inst_types "QTOF" "FT" \
  --output_dir output/drugs
```

##### Orbitrap Only (NCE values)
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_list "CCO" "CC(=O)O" \
  --prec_types "[M+H]+" \
  --ces 20.0 \
  --nces 15.0 25.0 35.0 \
  --inst_types "FT" \
  --output_dir output/orbitrap_only
```

##### QTOF Only (eV values)
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_list "CCO" "CC(=O)O" \
  --prec_types "[M+H]+" "[M-H]-" \
  --ces 10.0 20.0 30.0 \
  --nces 25.0 \
  --inst_types "QTOF" \
  --output_dir output/qtof_only
```

#### Output Files

Each run generates three files in the specified output directory:

1. **`mol_df.pkl.gz`**: Molecule dataframe with computed properties
   - Columns: `smiles`, `mol_id`, `input_mol_id`, `mol`, `inchikey_s`, `formula`, `mw`, `exact_mw`, `num_atoms`, `num_bonds`, `charge`, `single_mol`, `num_radicals`
   - Only includes molecules that pass quality filtering (single molecules, no charge, no radicals)

2. **`spec_df.pkl.gz`**: Spectrum dataframe with experimental conditions
   - Columns: `spec_id`, `mol_id`, `prec_type`, `nce`, `ace`, `inst_type`, `group_id`, `dset`, `dset_spec_id`, `prec_mz`, `peaks`, `frag_mode`, `spec_type`, `ion_mode`, `formula`, `inchikey`, `exact_mass`
   - In merged spectra mode: one entry per molecule with averaged collision energies
   - In standard mode: one entry per molecule × condition combination

3. **`config.json`**: Configuration file with all parameters and combinations
   - Input parameters, combination summary, detailed combination list, and file paths

#### Quality Filtering

The script automatically filters out molecules that don't meet quality standards:
- **Non-single molecules**: Removes molecules that contain multiple disconnected components
- **Charged molecules**: Removes molecules with formal charges (charge ≠ 0)
- **Radical molecules**: Removes molecules with unpaired electrons (num_radicals ≠ 0)

Filtered molecules are logged with detailed counts and reasons for removal.

#### Merged Spectra Mode

When `--merged_spectra true` is specified:
- Generates one spectrum entry per molecule instead of per condition
- Groups spectra by precursor type and instrument type combinations
- Averages collision energies across similar conditions
- Simplifies downstream analysis for molecule-level predictions
- Reduces total number of predictions while preserving experimental diversity

#### Configuration Example

```json
{
  "timestamp": "2025-09-11T10:30:45.123456",
  "input_parameters": {
    "num_smiles": 2,
    "smiles_list": ["CCO", "CC(=O)O"],
    "smiles_file": null,
    "precursor_types": ["[M+H]+", "[M-H]-"],
    "collision_energies_eV": [15.0, 25.0],
    "normalized_collision_energies_pct": [20.0, 30.0],
    "instrument_types": ["QTOF", "FT"],
    "output_directory": "output/example",
    "merged_spectra": false
  },
  "combinations": [
    {
      "id": 0,
      "prec_type": "[M+H]+",
      "ce": 15.0,
      "ce_unit": "eV",
      "inst_type": "QTOF"
    },
    {
      "id": 1,
      "prec_type": "[M+H]+",
      "ce": 20.0,
      "ce_unit": "NCE",
      "inst_type": "FT"
    }
  ],
  "output_files": {
    "mol_df": "output/example/mol_df.pkl.gz",
    "spec_df": "output/example/spec_df.pkl.gz",
    "config": "output/example/config.json"
  }
}
```

### 2. `run_save_inference.py`

Runs FraGNNet inference on prepared data and saves predictions.

#### Usage

```bash
python scripts/inference/run_save_inference.py \
  --task_dp <DATA_DIRECTORY> \
  --model_ckpt <MODEL_CHECKPOINT> \
  --save_dp <OUTPUT_DIRECTORY>
```

Where `<DATA_DIRECTORY>` contains the `mol_df.pkl.gz` and `spec_df.pkl.gz` files generated by `prepare_inference_input.py`.

## Workflow

1. **Prepare Data**: Use `prepare_inference_input.py` to create molecule and spectrum dataframes
2. **Run Inference**: Use `run_save_inference.py` to generate predictions
3. **Analyze Results**: Process the prediction outputs for your analysis

## Notes

- Both `--ces` and `--nces` arguments are required, even if you're only using one instrument type
- The script automatically selects the appropriate collision energy values based on instrument type
- Input files support multiple formats: .txt/.smi/.smiles (one SMILES per line) and .csv/.tsv (with column headers)
- For CSV/TSV files, `--smiles_colname` is required; `--id_colname` is optional for custom molecule IDs
- Invalid SMILES strings are automatically filtered out with warnings
- Quality filtering removes non-single molecules, charged molecules, and molecules with radicals
- All molecular properties are computed using RDKit
- Precursor m/z values are automatically calculated from molecular mass and precursor type
- The `--merged_spectra` option accepts various boolean formats: true/false, yes/no, 1/0
- Merged spectra mode reduces the number of predictions while maintaining experimental diversity
- Configuration JSON files provide complete reproducibility and traceability of all parameters