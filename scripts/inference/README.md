# FraGNNet Inference Scripts

This directory contains scripts for preparing data and running inference with FraGNNet models.

## Scripts

### 1. `prepare_inference_input.py`

Prepares molecular and spectral dataframes for FraGNNet inference from lists of SMILES, precursor types, collision energies, and instrument types.

#### Features
- Converts SMILES to RDKit molecules with computed properties
- Creates all combinations of molecules × experimental conditions
- Handles instrument-specific collision energy units (NCE for Orbitrap, eV for others)
- Generates `mol_df.pkl.gz`, `spec_df.pkl.gz`, and `config.json` files
- Validates input parameters and provides detailed logging

#### Usage

```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_list <SMILES> [<SMILES> ...] \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --ces <CE_EV> [<CE_EV> ...] \
  --nces <NCE_PCT> [<NCE_PCT> ...] \
  --inst_types <INST_TYPE> [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR>
```

Or from a file:
```bash
python scripts/inference/prepare_inference_input.py \
  --smiles_file <SMILES_FILE> \
  --prec_types <PREC_TYPE> [<PREC_TYPE> ...] \
  --ces <CE_EV> [<CE_EV> ...] \
  --nces <NCE_PCT> [<NCE_PCT> ...] \
  --inst_types <INST_TYPE> [<INST_TYPE> ...] \
  --output_dir <OUTPUT_DIR>
```

#### Arguments

- `--smiles_list`: List of SMILES strings (mutually exclusive with `--smiles_file`)
- `--smiles_file`: File containing SMILES strings (one per line)
- `--prec_types`: Precursor types. Choices: `[M+H]+`, `[M-H]-`
- `--ces`: Collision energies in eV (used by QTOF, Q, TOF, QQQ instruments)
- `--nces`: Normalized collision energies in % (used by FT, IT instruments)
- `--inst_types`: Instrument types. Choices: `FT`, `QTOF`, `IT`, `Q`, `TOF`, `QQQ`
- `--output_dir`: Output directory for generated files

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
   - Columns: `smiles`, `mol`, `inchikey_s`, `scaffold`, `formula`, `inchi`, `mw`, `exact_mw`, `num_atoms`, `num_bonds`, `charge`, `single_mol`, `num_radicals`

2. **`spec_df.pkl.gz`**: Spectrum dataframe with experimental conditions
   - Columns: `spec_id`, `mol_id`, `prec_type`, `nce`, `ace`, `inst_type`, `group_id`, `dset`, `dset_spec_id`, `prec_mz`, `peaks`, `frag_mode`, `spec_type`, `ion_mode`, `formula`, `inchikey`, `exact_mass`

3. **`config.json`**: Configuration file with all parameters and combinations
   - Input parameters, combination summary, and detailed combination list

#### Configuration Example

```json
{
  "timestamp": "2025-09-11T10:30:45.123456",
  "input_parameters": {
    "num_smiles": 2,
    "smiles_list": ["CCO", "CC(=O)O"],
    "precursor_types": ["[M+H]+", "[M-H]-"],
    "collision_energies_eV": [15.0, 25.0],
    "normalized_collision_energies_pct": [20.0, 30.0],
    "instrument_types": ["QTOF", "FT"]
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
    "mol_df": "output/mol_df.pkl.gz",
    "spec_df": "output/spec_df.pkl.gz",
    "config": "output/config.json"
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
- Invalid SMILES strings are automatically filtered out with warnings
- All molecular properties are computed using RDKit
- Precursor m/z values are automatically calculated from molecular mass and precursor type