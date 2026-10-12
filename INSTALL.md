# Installation Guide

This project supports multiple installation methods depending on your use case.

## Quick Start (Recommended for Most Users)

### Using Conda (GPU)

This is the **recommended approach** for GPU development and training.

```bash
# Run the environment setup script
bash env/env_setup_gpu_py311_cu121_250602.sh
```

This script will:
1. Create a conda environment named `FRAGNNET-GPU`
2. Install Python 3.11 and all required dependencies
3. Install PyTorch with CUDA 12.1 support
4. Install the fragnnet package in editable mode

**Manual activation** (if needed later):
```bash
conda activate FRAGNNET-GPU
```

### Available Local GPU Environment Setup Scripts

Choose based on your CUDA version and Python preference:

- **CUDA 12.1, Python 3.11 (Recommended)**: `env/env_setup_gpu_py311_cu121_250602.sh`
- **CUDA 12.1, Python 3.12**: `env/env_setup_gpu_py312_cu121_251105.sh`
- **CUDA 11.8, Python 3.10 (Legacy)**: `env/env_setup_gpu_py310_cu118.sh`

`env/fragnnet_gpu_py311.lock.txt` pins the exact package versions of the reference
`FRAGNNET-GPU` environment (Python 3.11, CUDA 12.1). To reproduce it, install the lock into a
fresh Python 3.11 environment with `pip install -r env/fragnnet_gpu_py311.lock.txt`, then
install the package with `pip install --no-deps -e .`.

### HPC/SLURM Cluster Setup (Compute Canada)

For training on Compute Canada clusters, use the appropriate environment setup script:

- **Python 3.11**: `env/env_cc_setup_py311.sh`
- **Python 3.10**: `env/env_cc_setup_py310.sh`

**Note**: These scripts create a virtual environment (not conda) and use `--no-index` to install packages from local mirrors without internet access.

```bash
# For Python 3.11 (recommended)
bash env/env_cc_setup_py311.sh

# For Python 3.10
bash env/env_cc_setup_py310.sh
```

## Alternative Installation Methods

### Manual Pip Installation

If you prefer manual installation without conda:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install the package
pip install -e .
```

For development with testing tools:
```bash
pip install -e ".[dev]"
```

This includes pytest, ruff, black, mypy, and other development tools, plus scikit-learn,
which the analysis and evaluation scripts under `scripts/` import (six tests load those
scripts). To run the scripts without the development tools:
```bash
pip install -e ".[scripts]"
```

For the `chemlm` model (a pretrained SMILES language model), also install its pinned
Hugging Face packages:
```bash
pip install -e ".[chemlm]"
```

It loads a local checkpoint directory, by default `data/pretrained/ChemBERTa-100M-MLM`
(Hugging Face `DeepChem/ChemBERTa-100M-MLM`, revision in its `REVISION` file), and never
downloads.

## Verify Installation

After installation, verify everything works:

```bash
# Activate environment (if using conda)
conda activate FRAGNNET-GPU

# Test basic imports
python -c "import fragnnet; print('fragnnet imported successfully')"

# Test GPU dependencies
python -c "import torch; import lightning; print(f'GPU available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/ -v
```

## Troubleshooting

### Lightning Module Not Found
If you see `ModuleNotFoundError: No module named 'lightning'`:
```bash
# Install lightning explicitly
pip install lightning==2.4.0
```

### CUDA Version Mismatch
Check your system CUDA version and use the matching environment script:
```bash
# Check CUDA version
nvcc --version

# For CUDA 12.1, use:
bash env/env_setup_gpu_py311_cu121_250602.sh

# For CUDA 11.8, use:
bash env/env_setup_gpu_py310_cu118.sh
```

### Cython Build Issues
If you see compilation errors during installation:
```bash
# Clean previous builds
rm -rf build/ src/fragnnet/*.so src/fragnnet.egg-info

# Ensure build dependencies are installed
pip install cython==3.0.12 numpy>=1.24,<2

# Rebuild
pip install -e . --no-cache-dir
```

### Import Errors After Installation
If imports fail after successful installation:
```bash
# Verify the package is installed
pip list | grep fragnnet

# Reinstall in editable mode
pip install --no-deps -I -e .
```

## Which Method Should I Use?

| Use Case | Recommended Method |
|----------|-------------------|
| GPU training on local machine (CUDA 12.1) | Run: `bash env/env_setup_gpu_py311_cu121_250602.sh` |
| GPU training on local machine (CUDA 11.8) | Run: `bash env/env_setup_gpu_py310_cu118.sh` |
| Manual pip installation | `pip install -e .` (after installing PyTorch manually) |
| Development + testing (local) | After conda setup, run `pip install -e ".[dev]"` |
| Training on Compute Canada (Python 3.11) | Run: `bash env/env_cc_setup_py311.sh` |
| Training on Compute Canada (Python 3.10) | Run: `bash env/env_cc_setup_py310.sh` |

## Compute Canada / HPC Workflows

For generating and submitting Fir and Vulcan training jobs, see the canonical
[cluster submission guide](docs/guides/submit_cluster_runs.md). It covers resource profiles,
cluster setup requirements, single runs, CV arrays, staging, and persistent outputs.

## Environment Variables

When using conda environments, VS Code will auto-detect them. To manually select:

1. Open command palette: `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Search: "Python: Select Interpreter"
3. Choose `FRAGNNET-GPU` or your active environment

## Development Workflow

For contributors and developers:

1. **Initial setup**:
   ```bash
   bash env/env_setup_gpu_py311_cu121_250602.sh
   conda activate FRAGNNET-GPU
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests before committing**:
   ```bash
   pytest tests/ -v
   ```

4. **Code formatting** (optional, for consistency):
   ```bash
   ruff check src/ tests/
   black src/ tests/
   ```

## Dependencies

### Core Dependencies
- Python 3.10-3.12 (the pinned PyTorch and RDKit, and the compiled extensions, do not
  support 3.13)
- PyTorch 2.4.1
- PyTorch Lightning 2.4.0
- RDKit 2022.09-2023.09 (cheminformatics)
- NumPy <2.0
- Pandas 2.2.3
- PyYAML, NetworkX, Pillow
- Cython 3.0.12 (for building fragmentation modules)

### Development Dependencies (optional)
- pytest (testing)
- ruff, black, isort (code formatting)
- mypy (type checking)
- wandb (experiment tracking)
- Jupyter notebook
- scikit-learn (also the `scripts` extra)

See [pyproject.toml](pyproject.toml) for the complete dependency list.

## Next Steps

- See [docs/guides/finetune_fragnnet.md](docs/guides/finetune_fragnnet.md) for model fine-tuning
- See [docs/guides/run_inference.md](docs/guides/run_inference.md) for inference pipelines
- Run tests: `pytest tests/ -v`
- Check configs: See `config/template.yml` for experiment configuration
