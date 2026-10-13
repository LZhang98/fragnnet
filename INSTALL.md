# Installation Guide

`pyproject.toml` lists the dependencies. Only torch and the PyG extensions (`torch_scatter`,
`pyg_lib`) depend on the CUDA build, so they come from their own indexes; everything else
resolves from `pyproject.toml`.

## Local GPU (uv)

```bash
uv sync --extra dev               # creates .venv: Python 3.11, torch 2.4.1+cu121
source .venv/bin/activate         # or prefix commands with `uv run`
```

`uv.lock` pins every package for Linux x86_64 and Python 3.10-3.12 (`uv sync --python 3.12`).
The `[tool.uv]` section of `pyproject.toml` sends torch to the PyTorch cu121 index and
`torch_scatter`/`pyg_lib` to the PyG index, and pins the reference rdkit (2022.9.5; 2023.9.5
on Python 3.12) and the packages the env and cluster scripts pin (`wandb`, `joblib`,
`pandarallel`, `pyteomics`). The core stack matches the reference environment; other packages
are newer than in `env/fragnnet_gpu_py311.lock.txt`. After changing a `.pyx` file, `uv sync`
rebuilds the extensions.

`uv lock` re-resolves after a dependency change; `uv lock --upgrade-package <name>` upgrades one
package. For a CUDA 11.8 or CPU build, skip the lock and install into a venv with `uv pip`
(here `cpu`):

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install torch_scatter==2.1.2 pyg_lib==0.4.0 -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
uv pip install --no-sources -e ".[dev]"
```

`--no-sources` stops uv from fetching the cu121 builds named in `[tool.uv.sources]`.

## Local GPU (conda)

```bash
bash env/env_setup_conda.sh            # Python 3.11, env FRAGNNET-GPU, CUDA 12.1
conda activate FRAGNNET-GPU
```

Optional positional arguments: Python version (`3.10`, `3.11`, `3.12`), environment name, and
torch build (`cu121`, `cu118`, `cpu`). For example, `bash env/env_setup_conda.sh 3.12
FRAGNNET-PY312 cu121`. The script stops if an environment of that name already exists.

The script installs torch 2.4.1, `torch_scatter` and `pyg_lib` for the chosen build, the
reference rdkit (2022.9.5, which has no Python 3.12 wheel; 3.12 gets 2023.9.5, the newest
release `pyproject.toml` allows), then `pip install -e ".[dev]"`.

`torch_scatter` is not imported by repository code, but keep it: PyG routes CUDA `max`/`min`
scatter with gradients through it when it is installed, which changes the gradient at tied maxima
for `frag_pool_type: max` (see the [reorganization plan, Phase 5](docs/plans/repo_reorg_plan.md#phase-5--packaging)).
Environments without it are not numerically equivalent.

### Exact reference environment

`env/fragnnet_gpu_py311.lock.txt` pins every package of the reference `FRAGNNET-GPU`
environment (Python 3.11, CUDA 12.1) and names the torch and PyG indexes itself. To reproduce
it in a fresh Python 3.11 environment:

```bash
pip install -r env/fragnnet_gpu_py311.lock.txt
pip install --no-deps -e .
```

## Manual pip installation

```bash
python -m venv venv
source venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install -e ".[dev]"
```

Replace `cu121` with `cu118` or `cpu` in both URLs for another build. Extras:

- `dev`: pytest, ruff, black, mypy, wandb, Jupyter, Cython (for in-place rebuilds) and
  scikit-learn, which the scripts under `scripts/` import (six tests load those scripts).
- `scripts`: scikit-learn only.
- `speed`: `torch-linear-assignment` (GPU linear assignment; scipy is the fallback).

## Compute Canada

Training jobs build their own virtualenv; generate them with `scripts/setup_job_cc.py` (see the
[cluster submission guide](docs/guides/submit_cluster_runs.md)). For an interactive venv on a
login node, from the repository root:

```bash
bash env/env_setup_cc.sh          # Python 3.11, venv FRAGNNET
bash env/env_setup_cc.sh 3.10     # Python 3.10
```

It installs from the cluster wheelhouse (`--no-index`) and loads the `rdkit/2023.09.5` module.
pip cannot see a module-provided rdkit, so fragnnet itself installs with `--no-deps`.

## Verify the installation

```bash
python -c "import fragnnet; print('fragnnet imported successfully')"
python -c "import torch, lightning; print(f'GPU available: {torch.cuda.is_available()}')"
pytest tests/ -v
```

## Troubleshooting

### Cython extensions

`pip install` builds the extensions in an isolated environment and fetches Cython and numpy
from `[build-system]`, so neither needs to be preinstalled. After changing a `.pyx` file or
switching branches, rebuild in place (needs the `dev` extra):

```bash
python setup.py build_ext --inplace
```

For a clean rebuild (the compiled modules sit in subpackages such as
`src/fragnnet/models/fragnnet/dag/`, so search the whole tree):

```bash
rm -rf build/ src/fragnnet.egg-info
find src/fragnnet -name '*.so' -delete
pip install --no-deps -e . --no-cache-dir
```

### CUDA build mismatch

`torch_scatter` must come from the PyG index page that matches the installed torch version and
build (`torch-2.4.1+cu121.html` for torch 2.4.1+cu121). Check the build with
`python -c "import torch; print(torch.__version__, torch.version.cuda)"`.

### Import errors after installation

```bash
pip list | grep fragnnet
pip install --no-deps -I -e .
```

## Core dependencies

- Python 3.10-3.12 (the pinned torch and rdkit, and the compiled extensions, do not support 3.13)
- PyTorch 2.4.1, Lightning 2.4.0, PyG 2.6.1 with `torch_scatter` 2.1.2
- RDKit 2022.09-2023.09
- NumPy <2.0, pandas 2.2.3, SciPy 1.15.2
- Cython 3.0.12 (build time; `dev` extra for in-place rebuilds)

See [pyproject.toml](pyproject.toml) for the complete list.

## Next steps

- [docs/guides/finetune_fragnnet.md](docs/guides/finetune_fragnnet.md) for fine-tuning
- [docs/guides/run_inference.md](docs/guides/run_inference.md) for inference pipelines
- `config/template.yml` for experiment configuration
