#!/bin/bash

set -e

module load StdEnv/2020 gcc/9.3.0 python/3.10 rdkit/2022.09.4 cuda/11.1.1

virtualenv --no-download $HOME/envs/FRAG-GNN
source $HOME/envs/FRAG-GNN/bin/activate

pip install --upgrade pip

# note: rdkit is already loaded

# GPU dependencies
pip install torch==1.13.1+computecanada

# non-GPU dependencies
pip install cython
pip install pandas
pip install networkx
pip install wandb
pip install tqdm
pip install scipy
pip install joblib
pip install matplotlib

pip install pytest
# TODO: install dgl
# TODO: install cysignals
# TODO: install torch-scatter

# frag-gnn itself
pip install -I -e .
