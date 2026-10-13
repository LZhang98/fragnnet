#!/bin/bash
# Create a conda environment with fragnnet installed in editable mode.
#
# Usage: bash env/env_setup_conda.sh [python_version] [env_name] [torch_build]
#   python_version  3.10, 3.11 (default) or 3.12
#   env_name        FRAGNNET-GPU (default)
#   torch_build     cu121 (default), cu118 or cpu
#
# The reference environment (Python 3.11, cu121) is pinned exactly in
# env/fragnnet_gpu_py311.lock.txt; see INSTALL.md.

set -e

PY_VERSION=${1:-3.11}
ENV_NAME=${2:-FRAGNNET-GPU}
TORCH_BUILD=${3:-cu121}
TORCH_VERSION=2.4.1

cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
# conda create -y would delete an existing environment of the same name
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda environment $ENV_NAME already exists; pass another name or remove it first" >&2
  exit 1
fi
conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
conda activate "$ENV_NAME"

# torch first, from the index of the chosen build. The PyG extensions must match that build:
# PyG routes CUDA max/min scatter through torch_scatter when it is installed, which changes the
# gradient at tied maxima (docs/plans/repo_reorg_plan.md, Phase 5).
pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_BUILD"
pip install torch_scatter==2.1.2 pyg_lib==0.4.0 \
  -f "https://data.pyg.org/whl/torch-$TORCH_VERSION+$TORCH_BUILD.html"

# rdkit of the reference environment; it has no Python 3.12 wheel, so 3.12 takes 2023.9.5
if [ "$PY_VERSION" = "3.12" ]; then
  pip install rdkit==2023.9.5
else
  pip install rdkit==2022.9.5
fi

# everything else resolves from pyproject.toml
pip install -e ".[dev]"
