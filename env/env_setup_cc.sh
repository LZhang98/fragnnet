#!/bin/bash
# Create a Compute Canada virtualenv with fragnnet installed in editable mode.
# Run on a login node; the venv is created in the repository root.
#
# Usage: bash env/env_setup_cc.sh [python_version] [venv_name]
#   python_version  3.10 or 3.11 (default)
#   venv_name       FRAGNNET (default)
#
# Packages come from the cluster wheelhouse (--no-index). rdkit comes from the rdkit module,
# which pip cannot see, so fragnnet itself installs with --no-deps. Job scripts build their own
# venv in scripts/setup_job_cc.py.

set -e

PY_VERSION=${1:-3.11}
VENV_NAME=${2:-FRAGNNET}

cd "$(dirname "$0")/.."

source ~/.bashrc

module load StdEnv/2023
module load "python/$PY_VERSION"
module load rdkit/2023.09.5

virtualenv "$VENV_NAME"
source "$VENV_NAME/bin/activate"
pip install --no-index --upgrade pip

pip install --no-index \
  torch==2.4.1 \
  torch_geometric==2.6.1 \
  torch_scatter==2.1.2 \
  pandas \
  wandb \
  tqdm \
  scipy==1.15.2 \
  joblib==1.3.2 \
  matplotlib \
  lightning==2.4.0 \
  cython==3.0.12 \
  pyopenms==3.3.0 \
  pandarallel==1.6.5 \
  h5py

pip install pyteomics torch-linear-assignment

pip install --no-deps -I -e .
