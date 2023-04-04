#!/bin/bash

set -e

module load StdEnv/2020 gcc/9.3.0 python/3.10 rdkit/2022.09.4 cuda/11.1.1 rust/1.65.0

virtualenv --no-download GFN
source FRAG-GNN/bin/activate

pip install --upgrade pip

# note: rdkit is already loaded

# GPU dependencies
pip install torch==1.13.1+computecanada

# non-GPU dependencies
pip install wandb
pip install tqdm
pip install cython
pip install pandas
pip install joblib
pip install networkx

# frag-gnn itself
pip install .
