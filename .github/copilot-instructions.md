## Project Overview

This is a machine learning research project for model training and inference pipelines.
- **Language**: Python 3.11
- **Core libraries**: PyTorch, NumPy, scikit-learn
- **Data processing**: Pandas, Polars
- **Testing**: pytest with fixtures
- **Execution**: SLURM on HPC clusters or local GPU training
- **Key patterns**: Modular src/ package, numbered preprocessing scripts, experiment config 

### Install
1. `python -m venv venv`
2. `source venv/bin/activate`  (Unix) or `venv\Scripts\activate.ps1` (Windows)
3. `pip install -r requirements.txt`
4. `pip install -r requirements-dev.txt` (for testing/development)

### Test
- **Run all tests**: `pytest tests/ -v`
- **Run specific test file**: `pytest tests/test_models.py -v`
- **Run with coverage**: `pytest --cov=src tests/`

### Data Processing
- `python preprocess_scripts/01_extract_data.py --config configs/preprocess.yaml`
- `python preprocess_scripts/02_create_dataset.py --config configs/dataset.yaml`

### Training
- **Local GPU**: `python scripts/fine_tuning.py --config configs/model_config.yaml`
- **SLURM cluster**: `sbatch slurm_scripts/train_job.sh`

### Inference
- `python scripts/inference.py --model weights/model.pt --input data/test.csv`

## Project Structure
src/
  └── fragnnet/               # Main Python package
      ├── __init__.py
      ├── runner.py           # Training/inference runner
      ├── model/              # Model definitions
      │   ├── __init__.py
      │   ├── base_model.py           # Base model classes
      │   ├── fragnnet_model.py       # FraGNNet architecture
      │   ├── gnn_model.py            # GNN components
      │   ├── spectrum_encoder.py     # Spectrum encoding
      │   ├── mol_encoder.py          # Molecule encoding
      │   ├── form_embedder.py        # Formula embedder
      │   ├── precursor_model.py      # Precursor models
      │   ├── siamese_gnn_model.py    # Siamese GNN
      │   ├── neims_model.py          # NEIMS models
      │   ├── nn_blocks.py            # Neural network blocks
      │   └── loss.py                 # Loss functions
      ├── dataset/            # Dataset classes
      │   ├── __init__.py
      │   ├── base_dataset.py         # Base dataset class
      │   ├── spec_mol_dataset.py     # Spectrum-molecule dataset
      │   ├── spec_mol_frag_dataset.py # Spectrum-molecule-fragment dataset
      │   └── mces_dataset.py         # MCES dataset
      ├── frag/               # Fragmentation logic
      │   └── ...
      ├── graff/              # GRAFF module
      │   └── ...
      ├── iceberg/            # Iceberg module
      │   └── ...
      ├── massformer/         # MassFormer module
      │   └── ...
      ├── pl_model/           # PyTorch Lightning models
      │   └── ...
      └── utils/              # Shared utilities
          ├── __init__.py
          ├── data_utils.py           # Data loading utilities
          ├── feat_utils.py           # Feature utilities
          ├── formula_utils.py        # Formula processing
          ├── frag_utils.py           # Fragmentation utilities
          ├── spec_utils.py           # Spectrum utilities
          ├── ms2c_utils.py           # MS2C utilities
          ├── nn_utils.py             # Neural network utilities
          ├── pl_utils.py             # PyTorch Lightning utilities
          ├── proc_utils.py           # Processing utilities
          ├── plot_utils.py           # Plotting utilities
          ├── profile_utils.py        # Profiling utilities
          ├── script_utils.py         # Script utilities
          ├── setup_utils.py          # Setup utilities
          ├── dgl_compat_utils.py     # DGL compatibility
          └── misc_utils.py           # Miscellaneous utilities

tests/
  ├── test_mces_weight_loading.py     # MCES weight loading tests
  └── test_dmpnn_edge_format.py       # DMPNN edge format tests

preproc_scripts/
  ├── 01_prepare_df.py              # Prepare dataframe
  ├── 02_prepare_proc.py            # Data processing
  ├── 03_prepare_dag_feats.py       # DAG feature preparation
  ├── 04_prepare_split.py           # Dataset splitting
  ├── 05_prepare_magma_feats.py     # MAGMA feature preparation
  ├── 06_predict_magma_dags.py      # MAGMA DAG prediction
  └── 07_prepapre_dag_hdf5.py       # HDF5 conversion

scripts/
  ├── run_pl_model_fit.py           # PyTorch Lightning model training
  ├── run_inference.py              # Inference script
  ├── run_compute_frags.py          # Fragment computation
  ├── run_wandb_sweep.py            # W&B hyperparameter sweep
  └── setup_job_cc.sh               # Compute Canada job setup

notebooks/
  ├── ablation_exp.ipynb
  ├── c2ms_sim_exp.ipynb
  ├── formula_ann_exp.ipynb
  ├── frag_ann_exp.ipynb
  ├── frag_stats.ipynb
  ├── ms2c_retrieval_exp.ipynb
  ├── param_counts.ipynb
  ├── condition_notebooks/
  ├── data_notebooks/
  ├── debug_notebooks/
  ├── msg_notebooks/
  ├── smarts_notebooks/
  └── test_notebooks/

config/
  ├── template.yml                 # Config template
  ├── ablations/
  ├── charge_migration_fragmentation/
  ├── compare_interstage/
  ├── compare_mol_pooling/
  ├── debug/
  ├── entropy/
  ├── exp_fraggnn_ma_mi/
  ├── fragnnet_finetuning/
  ├── loss_funcs/
  ├── massspecgym_merged/
  ├── mces_pretraining/
  ├── mix_percision/
  ├── nist20_merged_inchikey/
  ├── nist20_merged_scaffold/
  ├── nist23_eims_inchikey/
  ├── pairwise_loss/
  ├── spectraverse/
  └── sweeps/

slurm_scripts/
  └── ...                          # HPC job submission scripts

.github/
  └── copilot-instructions.md

## Coding Standards
- **Type hints**: All functions must have type hints where applicable
- **Naming**: Snake case for functions/vars, PascalCase for classes
- **Device handling**: Always check for CUDA availability; support CPU fallback
- **Random seeds**: Set seeds (torch, numpy, random) for reproducibility
- **Logging**: Use Python logging module, not print statements
- **Tests**: Write unit tests for models and utilities; use pytest fixtures
- **Configs**: Use YAML for all experiment configs; never hardcode hyperparameters

## Important Constraints

⚠️ **Never modify**:
- Model input/output shapes without updating all downstream code
- The config schema without deprecation warnings
- The preprocessing steps (breaks reproducibility of past experiments)

✓ **Always do**:
- Save model checkpoints with seed and config info
- Add tests for new model architectures
- Update configs/ when adding new hyperparameters
- Include docstrings explaining mathematical operations
- Run full test suite and validate on validation set before submitting PR